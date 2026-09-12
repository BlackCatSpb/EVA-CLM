"""EVA: stack module."""

import math, os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import EVAConfig
from .block import EVABlock, PrecisionGate, ExactSequenceMemory
from .bridge import SemanticBridge
from .maturation import MaturationController
from .layer_bridge_gate import LayerBridgeGate
from .embedding import PartitionedEmbedding, LmHead, PartitionedHead, SigmoidCodedHead, CognitiveCodedHead
from .reasoning import ReasoningMemory, ReasoningGate
from .vsa_utils import dct_basis, zeckendorf_codes, sparse_block_codes, vsa_prefix_scan
from .memory_bank import StreamingMemoryBank
from .concept_layer import UnifiedConceptLayer
from .tau_config import TauConfig
from .adaptive_controller import AdaptiveController
from .lr_scheduler import MirrorLRScheduler
from .losses import compute_losses as _compute_losses_fn
from .logit_cache import LogitCacheAttention

class EVAStack(nn.Module):
    """Stack of EVABlock layers with embedding and lm_head."""
    
    def __init__(self, cfg: EVAConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PartitionedEmbedding(cfg)
        head_mode = getattr(cfg, 'head_mode', 'sigmoid_coded')
        if head_mode == 'sigmoid_coded':
            self.lm_head = SigmoidCodedHead(cfg, embed_basis=self.embed.basis, rope=self.embed.rope)
        elif head_mode == 'cognitive_coded':
            self.lm_head = CognitiveCodedHead(cfg, embed_basis=self.embed.basis, rope=self.embed.rope,
                                              k_mirror=cfg.mirror_k)
        elif head_mode == 'partitioned':
            self.lm_head = PartitionedHead(cfg, embed_basis=self.embed.basis, rope=self.embed.rope)
        else:
            raise ValueError(f'Unknown head_mode: {head_mode}')
        
        # τ-config must be created before layers (blocks need τ_norm)
        self.tau_config = TauConfig(
            n_layers=cfg.n_layers,
            tau_min=getattr(cfg, 'tau_min', 8.0),
            tau_max=getattr(cfg, 'tau_max', 512.0),
            dev_max=getattr(cfg, 'tau_dev_max', 0.3),
            T0=getattr(cfg, 'matur_T0', 8000.0),
            T_delay=getattr(cfg, 'matur_T_delay', 8000.0),
            delta_t=getattr(cfg, 'matur_delta', 4000.0),
            llrd_gamma=getattr(cfg, 'tau_llrd_gamma', 0.65),
            mem_tau_ref=getattr(cfg, 'tau_mem_ref', 64.0),
            gate_tau_min=getattr(cfg, 'gate_tau_min', 0.3),
            gate_tau_max=getattr(cfg, 'gate_tau_max', 5.0),
        )
        self.tau_config.update()  # initial computation

        self.layers = nn.ModuleList([
            EVABlock(cfg, i, tau_config=self.tau_config) for i in range(cfg.n_layers)
        ])

        # ─── Explicit Reasoning ───
        self.explicit_reasoning = getattr(cfg, 'explicit_reasoning', False)
        if self.explicit_reasoning:
            self.reasoning_memory = ReasoningMemory(cfg.D, max_steps=getattr(cfg, 'reasoning_max_steps', 8))
            self.reasoning_gate = None
            if getattr(cfg, 'reasoning_adaptive', False):
                self.reasoning_gate = ReasoningGate(cfg.D, max_steps=getattr(cfg, 'reasoning_max_steps', 8), know_dim=8)
            self._reasoning_buffer = None
            self._reasoning_count = None
            self.register_buffer('_reasoning_gates', torch.zeros(getattr(cfg, 'reasoning_max_steps', 8)), persistent=False)
            self.reasoning_enabled_step = 0
            self.reasoning_scale_override = None

        self.register_buffer('final_norm_w', torch.ones(cfg.D))
        # ─── Intent Bridge (wrapper): восходящий intent + нисходящая трансляция ───
        # intent_state параллелен global_state; эксперты «ловят» его через
        # zero-init w_intent/b_intent (см. mirror.py). default-off → модель нетронута.
        self.intent_bridge = getattr(cfg, 'intent_bridge', False)
        # B1: mirror geometry needed even with the bridge off (bus_head_proj /
        # _w_alpha_expert below referenced them only inside the if — build crash)
        self._n_experts = int(self.layers[0].mirror.G)
        self._K_max = max(int(l.mirror.k) for l in self.layers)
        if self.intent_bridge:
            # Per-head intent probe: h -> (G, k) per expert. Mirror k VARIES
            # per layer, so we project to G*K_max and slice per layer below.
            # Each expert owns its own intent subspace => no shared D-source,
            # heads don't contend for parameters and complement each other.
            self._n_experts = int(self.layers[0].mirror.G)
            self._K_max = max(int(l.mirror.k) for l in self.layers)
            self.intent_probe = nn.Linear(cfg.D, self._n_experts * self._K_max)
        # Intent stream: per-layer list of (1,1,G,_K_max) per-head contexts.
        # Stored in the PROBE's full _K_max space (not per-layer k) so a single
        # cross-layer bus can be averaged across layers; truncated to k_i at the
        # mirror gate. Flows layer->layer (depth) within a step and recurs in time.
        self._intent_stream = None
        self._last_salience = None
        # Phase-2 stencil: the aggregated cross-layer bus biases the head's group
        # readout -> a "template of connections hidden->projector" (free cache
        # without a cache, complement to VSA memory). Zero-init => head unchanged
        # at start (checkpoint-safe); bus inputs are detached (no cross-step BPTT).
        _head_K = getattr(self.lm_head, 'K', int(self.layers[0].mirror.G))
        self.bus_head_proj = nn.Linear(self._n_experts * self._K_max, _head_K, bias=False)
        nn.init.zeros_(self.bus_head_proj.weight)
        # U8: τ-modulated intent bridge alpha
        self._w_alpha_expert = nn.Parameter(torch.zeros(self._n_experts))
        self._last_bus = None
        # U12: running-RMS of the bus, fed to the ZERO-INIT head stencil. A raw bus
        # whose magnitude grows with content (healthy bus_max 240..1300 in the first
        # ~20 steps of the polygon run) makes `bus_head_proj`'s FIRST Adam step
        # (per-coordinate ~±lr from a zero start) dominate the head gate logits ->
        # CE 6.9 -> 86 in one step. Normalizing the (detached) stencil input by its
        # own fast-EMA RMS decouples the stencil's scale from bus growth; a linear
        # module reaches the same readout bias on an O(1) input with stable gradients.
        self.register_buffer('_bus_rms', torch.zeros(1))
        # ─── Semantic Bridge (in-pipeline per-layer) ───
        # Runs INSIDE the forward (train + inference). At every layer a shared
        # probe emits a semantic vector, a persistent cross-layer stream is
        # injected back into the hidden state, and (training only) the probe is
        # self-supervised to predict the next token's embedding. Off unless
        # cfg.bridge_conn > 0. default-off => model untouched when disabled.
        self.bridge = SemanticBridge(
            cfg.D, cfg.n_layers,
            bridge_dim=getattr(cfg, 'bridge_dim', 256),
            depth=getattr(cfg, 'bridge_depth', True),
            cfg=cfg,
        ) if getattr(cfg, 'bridge_conn', 0.0) > 0.0 else None
        # ─── Layer Bridge Gate (intelligent per-layer gating to bridge) ───
        # Каждый слой получает per-layer health MLP, gate = sigmoid(health) * tau.
        # Управляет вкладом каждого слоя в SemanticBridge на основе diagnostics.
        self.layer_bridge_gate = LayerBridgeGate(
            cfg.n_layers,
            health_features=6,
            tau_min=getattr(cfg, 'gate_tau_min', 0.3),
            tau_max=getattr(cfg, 'gate_tau_max', 5.0),
        ) if getattr(cfg, 'bridge_conn', 0.0) > 0.0 else None
        # ─── Unified τ-field (TauConfig) ───
        # (already created above, before layers, for U1/U3 block initialization)
        self._vsa_log_param = nn.Parameter(torch.tensor([1.7918, 1.2321, 1.1304, 1.1065]))
        self._tau_l_dev = self.tau_config._tau_dev  # alias — shared gradient
        # _tau_intent_dev removed: tau_config.intent_alpha replaces it entirely
        # ─── τ-derived values (computed from tau_config) ───
        self.tau_config.update()  # initial computation (mat_gate=None → defaults)
        tau_l = self.tau_config.tau_l
        # ─── Streaming Memory Bank (hierarchical L1+L2) ───
        # After embedding, before first layer. Read at every token position.
        # Write at sentence boundaries. Writes gated by maturation (like private_mem).
        self.memory_bank = StreamingMemoryBank(
            D=cfg.D,
            bridge_dim=getattr(cfg, 'mem_bridge_dim', getattr(cfg, 'bridge_dim', 256)),
            l1_slots=getattr(cfg, 'mem_l1_slots', 3),
            l2_slots=getattr(cfg, 'mem_l2_slots', 32),
            min_write_maturation=getattr(cfg, 'mem_min_write_mat', 0.3),
            cfg=cfg,
            tau_config=self.tau_config,
        ) if getattr(cfg, 'memory_bank', False) else None
        # ─── Unified Concept Layer (replaces per-block CollectiveConceptLayer + L3Concepts) ───
        # Global concept layer: sits after embedding, reads/writes concepts from expert K-space.
        # τ-driven thresholds, continuous maturity, gradient flow through writes.
        self.concept_layer = UnifiedConceptLayer(
            D=cfg.D,
            k=int(self.layers[0].mirror.k),  # match first layer's expert K-space
            bridge_dim=getattr(cfg, 'bridge_dim', 256),
            S=getattr(cfg, 'unified_concept_S', 8),
            seed=42,
            cfg=cfg,
            softmax_free=getattr(cfg, 'softmax_free', True),
        ) if getattr(cfg, 'unified_concept_layer', True) else None
        # ─── Maturation controller (unified wake-up gate) ───
        # Uses tau_config.mat_delay for per-layer timing.
        if getattr(cfg, 'maturation_enabled', True):
            self.maturation = MaturationController(
                cfg.n_layers, tau_l[0].item(), tau_l[-1].item(), cfg,
                tau_config=self.tau_config)
        else:
            self.maturation = None
        # EMA for exploration (smoothed over ~500 steps)
        self.register_buffer('_expl_ema', torch.zeros(1), persistent=False)
        # Триада: сколько ре-циркуляций сделал Рассудок на последнем проходе (диагностика)
        self._triad_passes = 0
        # ─── Layer Bridge Gate diagnostics cache ───
        self._layer_diagnostics = {}  # filled during forward
        # U2: τ-norm for reasoning budget (mean across layers)
        self._tau_norm_reasoning = self.tau_config.tau_norm.mean().item()
        # ─── Logit Cache with Attention (long-context memory) ───
        # Dual-mode: stores h during training (gradient flows), logits during inference (compressed).
        # VSA-driven compression: model LEARNS vsa_scales to control compression per scale.
        # Decision #3 (включить): re-plumbed in CODE SPACE (per-bit evidence
        # via the head's own sparse block codebook — the retired design used
        # three V×D matrices ≈503M dead params, against README §1.1) and
        # integrated into forward via augment() below (cache_gate zero-init
        # ⇒ identity at resume; entries of past steps are detached on
        # retrieve — same-step gradient only, honoring the no-BPTT contract).
        # max_tokens (102_400, counted in ENTRIES despite the name → each
        # entry is a full (B,L,D) tensor) became a sane max_entries window.
        self.logit_cache = LogitCacheAttention(
            D=cfg.D,
            V=cfg.vocab,
            max_entries=getattr(cfg, 'logit_cache_max_entries', 64),
            n_heads=getattr(cfg, 'logit_cache_n_heads', 8),
            scheduled_sampling_ratio=getattr(cfg, 'logit_cache_scheduled_sampling', 0.05),
            codes=getattr(self.lm_head, 'codes', None),
            sparsity=float(getattr(cfg, 'code_sparsity', 4)),
        ) if getattr(cfg, 'logit_cache_enabled', True) else None
    
    def forward(self, h, state=None, global_state=None, pred_weight=None, adaptive=True,
                context_mem=None, allow_write=None, step=None,
                reasoning_buffer=None, reasoning_count=None, intent_state=None,
                tokens=None, _triad_depth: int = 0):
        """h: (B, L, D) — pre-embedded tokens
           state: per-layer memory states from previous forward (or None)
           global_state: cross-layer EMA self-model (or None, created fresh)
           pred_weight: adaptive alpha auxiliary loss weight (or None to compute)
           adaptive: if True, run AdaptiveController (training); if False, skip for speed (inference)
           reasoning_buffer: (B, max_steps, D) tensor of previous reasoning steps
           (or None → use module attribute, legacy path); reasoning_count: scalar
           long tensor = valid rows (or None)
           tokens: (B, L) — raw token ids for memory bank boundary detection (or None)
           Returns (h, state, global_state, (reasoning_buffer, reasoning_count)).
        """
        if state is None:
            state = [None] * len(self.layers)
        B, L, D = h.shape
        # Батч-несовпадение состояния с входом (e.g. resume при другом batch):
        # сброс всех внутренних состояний (иначе device-side assert / shape miss)
        if state is not None and any(s is not None for s in state):
            s0 = next(s for s in state if s is not None)
            if isinstance(s0, tuple):
                s0 = next((t for t in s0 if isinstance(t, torch.Tensor)), None)
            sB = s0.shape[0] if isinstance(s0, torch.Tensor) else -1
            if sB != B:
                state = [None] * len(self.layers)
        if reasoning_buffer is None:
            if self.training:
                # В обучении допускаем перенос deliberation-состояния между шагами
                # (состояние цепочки мысли). При eval — СБРОС: иначе буфер от
                # последнего шага обучения протаскивается в валидацию/генерацию
                # и даёт ложную расходимость (ppl -> 1e6).
                reasoning_buffer = getattr(self, '_reasoning_buffer', None)
                reasoning_count = getattr(self, '_reasoning_count', None)
                _reasoning_attr = True
            else:
                reasoning_buffer = None
                reasoning_count = None
                _reasoning_attr = False
        else:
            _reasoning_attr = False
        if self.explicit_reasoning and reasoning_buffer is not None:
            sB = reasoning_buffer.shape[0]
            if sB != B:
                reasoning_buffer = None
                reasoning_count = None
        
        # ─── Unified τ-field: update all τ-derived values ───
        mat_gate_for_tau = None
        if self.maturation is not None and step is not None:
            mat_gate_for_tau = self.maturation.gate  # use current gate values
        self.tau_config.update(mat_gate_for_tau)
        # Per-layer tau from tau_config (for maturation, intent, LLRD, diagnostics)
        tau_l = self.tau_config.tau_l  # (n_layers,) — per-layer temporal scale
        tau_min = tau_l[0]
        tau_max = tau_l[-1]
        tau_mid = (tau_min * tau_max).sqrt()
        # Per-scale tau from _vsa_log_param (for block-level VSA memory, S=4)
        # U1: τ-consistent VSA scales: blend learnable base with τ-derived scaling
        _base_vsa = torch.exp(torch.cumsum(F.softplus(self._vsa_log_param), dim=0)) + 1.0  # (4,)
        c_ema = (1.0 / math.sqrt(self.cfg.D)) * tau_mid
        n_layers = len(self.layers)
        
        # ─── Adaptive gate biases from mirror stats (per-layer) ───
        if adaptive:
            with torch.no_grad():
                # Cache per-layer stats to avoid double computation
                _layer_stats_cache = {}
                for i, layer in enumerate(self.layers):
                    _layer_stats_cache[i] = AdaptiveController.layer_stats(layer,
                        expl_thresh=self.cfg.exploration_threshold,
                        diff_thresh=self.cfg.differentiation_threshold)

                expl_raw = sum(e for e, _ in _layer_stats_cache.values()) / n_layers
                diff = sum(d for _, d in _layer_stats_cache.values()) / n_layers
                self._expl_ema.mul_(0.998).add_(expl_raw * (1.0 - 0.998))
                global_expl = self._expl_ema.clamp(0.0, 1.0).item()
                
                self._pred_weight = (pred_weight if pred_weight is not None
                    else AdaptiveController.pred_weight(self.layers))
                    # λ-tied defaults (λ⁻⁶..λ⁻²) — audit M7: a local 0.05/0.3
                    # override silently diverged from the LambdaConfig range
                
                for i, layer in enumerate(self.layers):
                    l_expl, l_diff = _layer_stats_cache[i]
                    lf = i / max(len(self.layers) - 1, 1)
                    tau_l_val = self.tau_config.tau_l[i].item()
                    b_i_val = AdaptiveController.layer_b_i(layer, expl=l_expl, tau_l=tau_l_val)
                    b_d_max = getattr(self.cfg, 'vsa_b_d_max', 12.0)
                    b_d_val = AdaptiveController.layer_b_d(layer, expl=l_expl,
                        b_d_max=b_d_max)
                    smooth = getattr(self.cfg, 'vsa_b_d_smooth', 0.999)
                    if smooth >= 1.0:
                        layer.b_i.fill_(b_i_val)
                        layer.b_d.fill_(b_d_val)
                    else:
                        b_d_t = torch.tensor(b_d_val, device=layer.b_d.device, dtype=layer.b_d.dtype)
                        b_i_t = torch.tensor(b_i_val, device=layer.b_i.device, dtype=layer.b_i.dtype)
                        layer.b_d.data.lerp_(b_d_t, 1.0 - smooth)
                        layer.b_i.data.lerp_(b_i_t, 1.0 - smooth)
        
        # Global self-model: running EMA of layer memory centroids
        # Per-layer EMA rates proportional to 1/τ (Proposal V)
        if global_state is None:
            global_state = torch.zeros(n_layers, 1, D, device=h.device, dtype=h.dtype)
        if global_state.dim() == 2:
            global_state = global_state.unsqueeze(0).expand(n_layers, -1, -1).clone()
        elif global_state.shape[0] != n_layers:
            global_state = global_state[0:1].expand(n_layers, -1, -1).clone()
        # Copy before in-place updates: aot_export forbids mutating graph inputs
        # that require gradients (global_state is updated per layer below).
        global_state = global_state.clone()

        # ─── Intent Bridge: depth-flowing per-head intent stream ───
        # Per-layer list of (1,1,G,k_i): one k-dim intent per expert, flows
        # through layers within a step (depth) and recurs across steps (time).
        # Mirror k varies per layer, so the stream is per-layer. No shared
        # source => no parameter contention between heads.
        if self.intent_bridge:
            def _to_kmax(s):
                if s is None:
                    return None
                s = s.detach().to(device=h.device, dtype=h.dtype)
                if s.shape[-1] != self._K_max:
                    s = s.new_zeros(1, 1, self._n_experts, self._K_max)
                return s
            if isinstance(self._intent_stream, list) and len(self._intent_stream) == n_layers:
                intent_streams = [_to_kmax(s) for s in self._intent_stream]
            elif isinstance(intent_state, list) and len(intent_state) == n_layers:
                intent_streams = [_to_kmax(s) for s in intent_state]
            else:
                intent_streams = [
                    torch.zeros(1, 1, self._n_experts, self._K_max,
                                device=h.device, dtype=h.dtype)
                    for i in range(n_layers)]
            _sal = self._last_salience
        else:
            intent_streams = None
            _sal = None
        # ─── Momentum warmup for global_state oscillation (Idea 3) ───
        momentum_beta = 0.0
        if adaptive and step is not None and step >= 5000:
            momentum_beta = 0.8 * min(1.0, (step - 5000) / 5000)
        if momentum_beta > 0:
            if not hasattr(self, '_gs_velocity') or self._gs_velocity.shape != global_state.shape:
                self._gs_velocity = torch.zeros_like(global_state)
            else:
                self._gs_velocity = self._gs_velocity.to(global_state.device)
        new_state = []
        pred_errs = []  # per-layer pred_error_norm means for the maturation controller
        # ─── Cross-layer bus scratch (intent bridge) ───
        # Carried streams are the previous step's gist (detached). The bus is
        # STREAMING: layer i sees FRESH intent of already-processed layers (j<=i)
        # and CARRIED intent of downstream layers (j>i) — a flow, not a storage.
        # No cross-step BPTT (carried detached); self-term carries probe gradient.
        _bus_carried = None
        _bus_sum = None
        _bus_running = None
        _bus_le_carried = None
        if self.intent_bridge and intent_streams is not None:
            _bus_carried = list(intent_streams)
            _bus_sum = torch.stack([c.detach() for c in _bus_carried], 0).sum(0)  # (1,1,G,Kmax)
            _bus_running = torch.zeros_like(_bus_sum)
            _bus_le_carried = torch.zeros_like(_bus_sum)
        _last_bus = None
        # ─── Maturation gate for THIS step ───
        # Computed from the previous-step readiness EMA (pred_err for this step is
        # not known yet). M_l gates live BridgeGLU, memory write, bridge injection
        # and the intent bus. At M_l~0 only the frozen base MLP (~0.667) is active.
        mat_gate = None
        _global_ready = False
        if self.maturation is not None:
            if step is None:
                # Inference/eval: reuse the LAST training gate (never force-open, which
                # would scramble eval vs train — the bug that produced ppl 485M).
                mat_gate = self.maturation.gate
                _global_ready = self.maturation.global_ready
            else:
                # Maturation gate: pure time ramp (deep-first).
                # bridge_readiness is NOT used — it's a scalar that would
                # destroy the per-layer gradient by setting all layers equal.
                mat_gate = self.maturation.step_gate(step, self._tau_l_dev.detach())
                _global_ready = self.maturation.global_ready

        if self.bridge is not None:
            self.bridge.start_forward()
        for i, (layer, s) in enumerate(zip(self.layers, state)):
            if adaptive:
                l_expl, l_diff = _layer_stats_cache[i]
                mem2v_scale = AdaptiveController.layer_w_mem2v_scale(layer,
                    min_val=self.cfg.w_mem2v_scale_min, max_val=self.cfg.w_mem2v_scale_max,
                    diff=l_diff)
                nscale = AdaptiveController.layer_noise_scale(layer,
                    min_val=self.cfg.noise_scale_min, max_val=self.cfg.noise_scale_max,
                    diff=l_diff)
                tanh_bias_mod = AdaptiveController.tanh_bias_modulation(layer, expl=l_expl)
                spectral_mod = AdaptiveController.spectral_modulation(layer, diff=l_diff)
                pred_scale_mod = AdaptiveController.pred_scale_mod(layer)
            else:
                l_expl = l_diff = 0.5
                mem2v_scale = 1.0
                nscale = 0.0
                tanh_bias_mod = 1.0
                spectral_mod = 1.0
                pred_scale_mod = None
            
            gs_i = global_state[i:i+1].detach().clone()  # (1, 1, D), no grad through global_state (EMA-only)
            # Intent Bridge: derive this layer's intent from its INPUT hidden state
            # BEFORE the block so the bridge gate computed inside the block carries
            # gradient back into intent_probe. (The old .detach() froze the probe:
            # it must stay trainable as the layer params evolve.) The carried
            # intent_streams[i] is detached (saved per-step at the end of forward),
            # so only local_intent (probe) contributes gradient — no cross-step BPTT.
            intent_i = None
            if self.intent_bridge:
                _ki = self.layers[i].mirror.k
                probe_out = self.intent_probe(h).reshape(
                    h.shape[0], h.shape[1], self._n_experts, self._K_max)  # (B,L,G,Kmax)
                if self._last_salience is not None and \
                   self._last_salience.shape[0] == h.shape[0] and \
                   self._last_salience.shape[1] == h.shape[1]:
                    # Per-position salience weighting: highlight the semantically
                    # important parts of each expert's intent (word importance from
                    # the head). (B,L,1) -> (B,L,1,1) broadcasts over (G,Kmax).
                    probe_out = probe_out * self._last_salience.unsqueeze(-1)
                fresh_i = probe_out.mean(dim=(0, 1), keepdim=True)  # (1,1,G,Kmax), grad
                # Intent-stream carry coefficient: classic EMA whose carry
                # fraction is the layer's OWN τ (horizon=τ_l): a = 1 − 1/τ_l.
                # Audit decision #4 exposed the old form: intent_alpha(v2)=
                # 1−exp(−τ_l/τ_min) saturates to 1.0 for τ_l ≥ 64·τ_min, so the
                # deep-layer streams were frozen at their zero-init FOREVER
                # (fresh never entered; 'slow integration' meant 'never').
                # 1−1/τ keeps the ordering (deep carries longer) while every
                # stream always admits 1/τ of new content. intent_alpha keeps
                # its amplitude-authority role in the mirror.
                _tau_l_i = self.tau_config.tau_l[i].detach().clamp(min=2.0)
                _alpha_i = (1.0 - 1.0 / _tau_l_i)
                # U8: τ-scheduled per-expert DEVIATION of the carry fraction:
                # centered (2σ(w)−1) so w=0 means exactly the τ-horizon base.
                _expert_mod = (2.0 * torch.sigmoid(self._w_alpha_expert) - 1.0) * (2.0 * self.tau_config.tau_norm[i].item() - 1.0)
                _alpha_i_per_expert = (_alpha_i * (1.0 + _expert_mod)).clamp(0.0, 0.999)  # (G,)
                _a = _alpha_i_per_expert.view(1, 1, -1, 1)  # (1, 1, G, 1)
                intent_streams[i] = _a * _bus_carried[i] + (1.0 - _a) * fresh_i
                # Streaming cross-layer bus (Bus): network-wide gist = mean over
                # layers; FRESH for j<=i, CARRIED for j>i. Self-term (fresh_i)
                # keeps the probe trainable; others give cross-layer communication.
                # U8 LEARNABLE (audit decision #4): layer i additionally injects
                # its per-expert-gated carried stream (fresh + α·carried, α=0
                # when cold ⇒ step-0 identical). _w_alpha_expert therefore gets a
                # same-step CE gradient through the mirror intent gate, while the
                # probe keeps FULL fresh weight (a (1−α)·fresh own-term would
                # have decoupled the probe exactly at the deep layers where α≈1).
                _bus_running = _bus_running + fresh_i + _a * _bus_carried[i]
                _bus_le_carried = _bus_le_carried + _bus_carried[i]
                bus_i = (_bus_running + (_bus_sum - _bus_le_carried)) / n_layers  # (1,1,G,Kmax)
                with torch.no_grad():
                    _bus_rms = bus_i.detach().pow(2).mean().sqrt()
                    if self._bus_rms.item() == 0.0:  # cold-start: baseline = first level
                        self._bus_rms.copy_(_bus_rms)
                    else:
                        self._bus_rms.mul_(0.99).add_(_bus_rms, alpha=0.01)
                _last_bus = bus_i / self._bus_rms.clamp_min(1e-6)  # stencil reads a normalized bus
                intent_i = bus_i[..., :_ki]            # truncate to layer k
                # NB: mat_gate НЕ масштабирует intent_i — зеркало уже управляет
                # зрелостью через bridge_glu_net(delta)*maturity и expert_gate.
                # Двойное гейтирование убивало gradient(w_intent) без пользы
                # (mat_gate~0.09 → ik≈0 → hp-ik≈hp → gradient вырожден).
            # ─── Semantic Bridge (in-pipeline per-layer) ───
            # Inject the carried cross-layer stream into this layer's hidden state,
            # then emit + record the layer's semantic vector and EMA-update the
            # persistent stream (so lower layers see a FRESH bridge from it within
            # this step, and upper layers a CARRIED one — same streaming pattern as
            # the Intent Bridge). Runs in train and inference alike.
            if self.bridge is not None:
                _mat_i = mat_gate[i] if mat_gate is not None else None
                # U4: pass τ-normalized value for injection coupling
                _tau_norm_i = self.tau_config.tau_norm[i]
                h = self.bridge.inject_layer(i, h, maturity=_mat_i, tau_norm=_tau_norm_i)
                # Probe reads a DETACHED hidden state: the bridge is a semantic
                # read-out head that learns from its own self-supervised loss
                # (1-cos vs next-token embed) WITHOUT back-propagating into the
                # main trunk. That per-layer gradient into h destabilised CE;
                # detaching keeps the bridge's
                # forward signal (stream injection) while removing the diverging
                # gradient path. The gate still gets its gradient from the main CE.
                # ─── Layer Bridge Gate: scale probe input by per-layer health ───
                # Before global_ready: simple maturation gating (no SpectrumGate).
                # After global_ready: full per-layer SpectrumGate with tau-driven
                # diversity. Single source: LayerBridgeGate.layer_gate, fed by the
                # LIVE τ-field gate ladder (audit M2: the stack's inline copy
                # diverged and re-derived tau from its own literals).
                if self.layer_bridge_gate is not None and i in self._layer_diagnostics:
                    _tau_i = mat_gate[i] if mat_gate is not None else torch.ones((), device=h.device)
                    _gate_i = self.layer_bridge_gate.layer_gate(
                        i, self._layer_diagnostics[i], _tau_i, _global_ready,
                        tau_external=self.tau_config.gate_tau[i])
                    # B3 parity (agent E + bridge probe): the health-gate was
                    # fed by _layer_diagnostics of the PREVIOUS forward, so
                    # the probe input changed with cache freshness — train/eval
                    # features diverged (measured s_l maxdiff 1.3 at layer 1,
                    # bridge loss 0.26 train vs 3.69 eval < chance). The probe
                    # reads the raw detached state; the gate stays computed
                    # for the diagnostics dashboard only.
                    _s_l = self.bridge.probe_layer(h.detach())
                else:
                    _s_l = self.bridge.probe_layer(h.detach())
                self.bridge.record(_s_l)
                self.bridge.update_stream(i, _s_l)

            # ─── Streaming Memory Bank: per-layer read/write ───
            # Skip when maturation too low — pure waste of compute
            if (self.memory_bank is not None and tokens is not None
                    and (mat_gate is None or mat_gate[i].item() >= self.memory_bank._min_write_maturation)):
                _mb_mat_i = mat_gate[i].item() if mat_gate is not None else 1.0
                h = self.memory_bank(h, tokens, step=step, mat_gate=_mb_mat_i)

            if self.cfg.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint as _cp
                _saved_pen = layer.mirror._cached_pred_error_norm
                _saved_hp = layer.mirror._cached_hp
                # U1: per-layer τ-consistent VSA scales
                _layer_tau_ratio = tau_l[i] / tau_mid
                _vsa_tau_i = _base_vsa * _layer_tau_ratio
                _out = _cp(
                    EVAStack._checkpointed_block,
                    layer, h, s, gs_i,
                    _saved_pen, _saved_hp,
                    mem2v_scale, l_diff, nscale,
                    tanh_bias_mod, pred_scale_mod, spectral_mod,
                    context_mem, allow_write, _vsa_tau_i, step, intent_i,
                    salience=_sal, maturity=(mat_gate[i] if mat_gate is not None else None),
                    use_reentrant=False,
                )
                h, s_out, layer.mirror._cached_pred_error_norm, layer.mirror._cached_hp = _out
            else:
                # U1: per-layer τ-consistent VSA scales
                _layer_tau_ratio = tau_l[i] / tau_mid
                _vsa_tau_i = _base_vsa * _layer_tau_ratio
                h, s_out = layer(h, s, global_state=gs_i,
                                 mem2v_scale=mem2v_scale, diff=l_diff, noise_scale=nscale,
                                 tanh_bias_mod=tanh_bias_mod, pred_scale_mod=pred_scale_mod,
                                 spectral_mod=spectral_mod,
                                 context_mem=context_mem, allow_write=allow_write,
                                  tau_s=_vsa_tau_i, step=step, intent=intent_i, salience=_sal,
                                   maturity=(mat_gate[i] if mat_gate is not None else None))
            # ─── Unified Concept Layer (global, after first layer provides hp) ───
            # Called once after first layer to read/write concepts from expert K-space.
            # Injects concept-augmented signal into h for all subsequent layers.
            if (self.concept_layer is not None and i == 0
                    and getattr(layer.mirror, '_cached_hp', None) is not None
                    and layer.mirror._cached_hp.shape[:2] == h.shape[:2]):
                _hp = layer.mirror._cached_hp
                _pen = layer.mirror._cached_pred_error_norm
                _resvar = layer.mirror._residual_var_ema.mean() if hasattr(layer.mirror, '_residual_var_ema') else None
                _mat = mat_gate[0].item() if mat_gate is not None else 1.0
                # Audit decision #5: the UCL writes during INFERENCE too —
                # 'инференс = обучение' (README §1.4): consolidation is gated
                # by maturation/confidence/novelty, not by train-mode. Eval
                # contamination is quarantined by the M8 buffer snapshot.
                _col_out = self.concept_layer(
                    h, hp=_hp, pen=_pen, resvar=_resvar,
                    mat_gate=_mat, allow_write=True,
                    gate=layer.mirror._cached_gate,
                    tau_norm=self.tau_config.tau_norm[0].item(),
                )
                h = h + _col_out
            if self.maturation is not None:
                _pe = layer.mirror._cached_pred_error_norm
                if _pe is not None:
                    pred_errs.append(_pe.detach().mean())
            # ─── Layer Bridge Gate: collect per-layer diagnostics ───
            if self.layer_bridge_gate is not None:
                with torch.no_grad():
                    mir = layer.mirror
                    _diag = torch.zeros(6, device=h.device, dtype=h.dtype)
                    _pe = getattr(mir, '_cached_pred_error_norm', None)
                    if _pe is not None:
                        _diag[0] = _pe.detach().mean().clamp(0.0, 1.0)
                    _gl = getattr(mir, '_cached_gate_l1', None)
                    if _gl is not None:
                        _diag[1] = _gl.detach().clamp(0.0, 1.0)
                    _mp = getattr(mir, '_cached_pred_k', None)
                    if _mp is not None:
                        _mn = _mp.detach().norm()
                        _diag[2] = (_mn / 1000.0).clamp(0.0, 1.0)
                    _diag[3] = 0.5
                    _hp = getattr(mir, '_cached_hp', None)
                    if _hp is not None:
                        _hp_det = _hp.detach()
                        _hp_norm = torch.sigmoid(_hp_det)
                        _hp_norm = _hp_norm / _hp_norm.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                        _entropy = -(_hp_norm * _hp_norm.clamp_min(1e-9).log()).sum()
                        _max_entropy = math.log(_hp_det.shape[-1])
                        _diag[4] = (_entropy / _max_entropy).clamp(0.0, 1.0)
                    _gl2 = getattr(mir, '_cached_gate_l1', None)
                    if _gl2 is not None:
                        _diag[5] = (1.0 - _gl2).clamp(0.0, 1.0)
                    self._layer_diagnostics[i] = _diag
            if s_out is not None:
                mem_state_out = s_out[0]  # (B, S*D) — multi-scale memory state
                B = h.shape[0]
                S_expected = layer._n_scales
                # Guard: checkpoint can flatten state to (B, D); infer actual S from numel
                mem_flat = mem_state_out.reshape(B, -1)
                S = mem_flat.shape[-1] // layer.D
                if S == 0:
                    S = 1
                # Per-layer tau from unified τ-field (tau_config)
                alpha_l = self.tau_config.intent_alpha[i]
                # (intent tau now computed before the block; see intent_i setup above)
                # Weighted combination of scales для global state
                w = torch.sigmoid(layer.scale_w)  # (S, D), per-channel independent
                if S < S_expected:
                    w = w[:S]  # truncate weights to match available scales
                mem_combined = (mem_flat.reshape(B, S, layer.D) * w.unsqueeze(0)).sum(dim=1)
                mem_avg = mem_combined.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)
                if momentum_beta > 0:
                    vel_update = momentum_beta * self._gs_velocity[i:i+1].detach() + (1.0 - momentum_beta) * (mem_avg - gs_i)
                    self._gs_velocity[i:i+1] = vel_update.detach()
                    global_state[i:i+1] = gs_i + (1.0 - alpha_l.detach()) * self._gs_velocity[i:i+1]
                else:
                    global_state[i:i+1] = alpha_l * gs_i + (1.0 - alpha_l) * mem_avg
                # (intent_streams now updated BEFORE the block so intent_probe
                #  receives gradient; see the intent_i setup above the forward.)
                s_out = tuple(t.detach() if t is not None else None for t in s_out)
            new_state.append(s_out)
            if self.intent_bridge:
                self._intent_stream = [s.detach() for s in intent_streams]
                self._last_bus = _last_bus.detach() if _last_bus is not None else None
            if adaptive:
                mir = layer.mirror
                # (pred aux now travels as a LIVE per-layer scalar
                #  mir._pred_loss_term consumed by losses.compute_losses —
                #  the old _pred_cache of detached tensors was a dead path.)
                
        # ─── Update maturation controller from this step's per-layer pred-error ───
        if self.maturation is not None and step is not None and len(pred_errs) == n_layers:
            self.maturation.update(step, torch.stack(pred_errs))
        
        h = self.final_norm_w * h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-7)

        # ─── Explicit Reasoning ───
        if self.explicit_reasoning:
            # U2: update τ-norm for reasoning budget
            self._tau_norm_reasoning = self.tau_config.tau_norm.mean().item()
            s = self.reasoning_scale
            if s > 0.0:
                if self.reasoning_gate is not None:
                    h = self._adaptive_reasoning(h, s, new_state, reasoning_buffer, reasoning_count)
                else:
                    reasoning_out, reasoning_buffer, reasoning_count = self.reasoning_memory(
                        h, reasoning_buffer, reasoning_count)
                    h = h + s * reasoning_out.unsqueeze(1)
        if _reasoning_attr:
            self._reasoning_buffer = reasoning_buffer
            self._reasoning_count = reasoning_count

        # ─── Триада: Рассудок как участник (замыкание петли) ───
        # После прохода верификатор оценивает уверенность (_last_conf). Если она
        # ниже порога, ствол ре-циркулирует: повторный осмысленный проход с тем
        # же входом => бóльшая эффективная глубина, пока Рассудок не удовлетворён
        # или не исчерпан бюджет. Это превращает верификатор из пассивного
        # читателя в активного участника петли самокоррекции.
        # Только inference/generation: `not self.training` (eval измерение и
        # обучение не трогаем) И `step is not None` (generate передаёт step,
        # валидация — нет). Нет новых параметров => переобучение не нужно.
        self._triad_passes = _triad_depth
        if (getattr(self.cfg, 'triad_reason', False)
                and (not self.training)
                and step is not None
                and _triad_depth < int(getattr(self.cfg, 'triad_max_passes', 3))):
            with torch.no_grad():
                _conf = float(self._last_conf(h).mean().item())
            if _conf < float(getattr(self.cfg, 'triad_conf_thr', 0.5)):
                # B3 side-effect isolation: the deliberation re-pass must not
                # leave traces in shared streaming state — it re-appends to
                # bridge._preds (the outer loss read the LAST triad's probes:
                # train/eval gap 0.33 vs 3.31, 8 probe calls instead of 2)
                # and pumps the stream EMA with recycled h (positive feedback,
                # h-norm 130→204 per recursion). Snapshot-restore both.
                _br_snap = None
                if self.bridge is not None:
                    _br_snap = (self.bridge._preds, self.bridge.bridge_stream.detach().clone())
                h2, new_state, global_state, rb = self.forward(
                    h, state=new_state, global_state=global_state,
                    pred_weight=pred_weight, adaptive=adaptive,
                    context_mem=context_mem, allow_write=allow_write,
                    step=step, reasoning_buffer=reasoning_buffer,
                    reasoning_count=reasoning_count, intent_state=None,
                    _triad_depth=_triad_depth + 1)
                # Консервативный бленд против дрейфа при ре-циркуляции: половина
                # исходного и половина пересмотренного представления.
                h = 0.5 * h + 0.5 * h2
                reasoning_buffer, reasoning_count = rb
                self._triad_passes = _triad_depth + 1
                if _br_snap is not None:
                    self.bridge._preds = _br_snap[0]
                    self.bridge.bridge_stream.data.copy_(_br_snap[1])

        # ─── Logit cache augmentation (decision #3, integrated) ───
        # Runs in TRAIN and INFERENCE alike; zero-init cache_gate ⇒ h passes
        # through unchanged until CE learns to consult the cache.
        if self.logit_cache is not None:
            h = self.logit_cache.augment(h)

        return h, new_state, global_state, (reasoning_buffer, reasoning_count)

    def _knowledge_signal(self, h, state=None):
        """(B, know_dim) — how confident the model is in its own knowledge:
        top-1/top-2 prob of last position, contradiction margin (top1-top2),
        entropy of last position, position-averaged top-1/entropy, plus
        collective-memory agreement and representation activity (experts:
        specialization/collective memory/concept space).
        Zero-initialized know_proj keeps resume unchanged."""
        with torch.no_grad():
            logits = self.lm_head(h)  # (B, L, V)
            if getattr(self.cfg, 'softmax_free', True):
                # Режим Б: per-class уверенность (сигмоида), без нормировки к
                # симплексу и без конкуренции. Верификатор читает потенциал
                # каждого класса независимо, а не победителя softmax.
                p = logits.sigmoid()
            else:
                p = logits.softmax(-1)
            p1 = p.max(-1).values
            p2 = p.topk(2, dim=-1).values[..., 1]
            ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
            p_last = p[:, -1]
            ent_last = -(p_last * p_last.clamp_min(1e-9).log()).sum(-1)
            p1_last = p_last.max(-1).values
            p2_last = p_last.topk(2, dim=-1).values[..., 1]
            mem_agr = torch.zeros_like(p1_last)
            if state is not None and len(state) > 0:
                s_last = state[-1]
                if s_last is not None and len(s_last) > 0 and s_last[0] is not None:
                    mem = s_last[0]  # (B, S*D)
                    B2 = mem.shape[0]
                    if mem.numel() > 0 and mem.dim() == 2 and mem.shape[1] >= h.shape[-1] and mem.shape[1] % h.shape[-1] == 0:
                        S = mem.shape[1] // h.shape[-1]
                        mem_r = mem.reshape(B2, S, h.shape[-1])
                        mem_std = mem_r.std(1).mean(1)  # (B,)
                        mem_norm = mem_r.norm(dim=-1).mean(1).clamp_min(1e-9)
                        mem_agr = (mem_std / mem_norm).clamp_max(5.0)
            h_norm = h.norm(dim=-1).mean(1).clamp_max(5.0)
            know = torch.stack([
                p1_last, p2_last,
                (p1_last - p2_last).clamp_min(0.0),
                ent_last,
                p1.mean(1),
                ent.mean(1),
                mem_agr,
                h_norm,
            ], dim=-1)  # (B, 8)
        return know

    def _last_conf(self, h):
        """p1 of the last position — confidence of the head on `h`."""
        with torch.no_grad():
            logits = self.lm_head(h[:, -1:, :])
            if getattr(self.cfg, 'softmax_free', True):
                p = logits.sigmoid()
            else:
                p = logits.softmax(-1)
            return p.max(-1).values.squeeze(1)  # (B,)

    def _adaptive_reasoning(self, h, s, state=None, reasoning_buffer=None, reasoning_count=None):
        """Adaptive-depth reasoning loop: up to max_steps iterations, each gated
        by ReasoningGate. Returns updated h and records per-step gates for stats.
        Pure CE training signal — no aux losses, no conflict with the rest.
        Gate input = model knowledge (confidence/contradictions) + accumulated
        reasoning state, so depth adapts to knowledge gaps (uncertainty).
        Sequential gating: step i executes only if the gate of step i-1 stayed
        open, so an OFF gate still receives gradient (via the executed previous
        step) and can open later. On resume the first gate is ~1 (bias +10, tanh-saturated) and
        later gates ~0 (bias -8): the loop executes one full step plus one
        ~zero-contribution step — output matches the old single-step path.
        Static-graph form: the loop always executes K iterations, but the
        data-dependent `break` becomes a tensor run-mask — non-running steps
        contribute exactly zero. Numerically identical to the python loop;
        required for torch.export (no python control flow on tensor values)."""
        K_full = getattr(self.cfg, 'reasoning_max_steps', 8)
        # U2: τ-adaptive reasoning budget: scale by layer τ_norm
        K = K_full
        if hasattr(self, '_tau_norm_reasoning'):
            K = max(1, round(K_full * self._tau_norm_reasoning))
        stop_thr = getattr(self.cfg, 'reasoning_gate_stop_threshold', 0.5)
        know = self._knowledge_signal(h)  # (B, 8)
        conf_base = know[:, 0]  # head confidence on the raw h (B,)
        h_acc = h
        weighted = None   # Σ a_i·r_i — взвешенные знаками вклады (разность pos/neg)
        denom_accum = 0.0 # Σ взятых (положительных) весов, floor 0.5 (см. accum ниже)
        gates = []
        buf = reasoning_buffer
        count = reasoning_count
        if buf is None:
            buf = torch.zeros(h.shape[0], K_full, h.shape[-1], device=h.device, dtype=h.dtype)
        if count is None:
            count = torch.zeros((), dtype=torch.long, device=h.device)
        prev_open = torch.ones((), device=h.device)
        for i in range(K):
            # Кандидат формализации — что шаг рассуждения предлагает.
            # Вычисляется ДО гейта: гейт решает «беру/отбрасываю», зная
            # сам кандидат (связь неизвестного с известным), а не вслепую.
            # Запись в буфер коммитится только если гейт открыт (вклад ≠ 0):
            # закрытый шаг не засоряет память (буфер = старое поведение).
            r_i, buf_tmp, count_tmp = self.reasoning_memory(h, buf, count, record=True)
            l_i = self.reasoning_gate.logits(h_acc, know, r_i.unsqueeze(1))[..., i].unsqueeze(-1)
            a_i = torch.tanh(l_i)
            if i > 0:
                # Straight-through для закрытых гейтов (a≈0): обычный градиент
                # tanh'(0)=1 живой, но при насыщении tanh'≈0 мёртв — через
                # логит гейт может открыться/закрыться, если это выгодно.
                a_i = l_i + (a_i - l_i).detach()
            # run: шаг исполняется, если предыдущий не закрыл цикл (break)
            run = prev_open >= stop_thr  # scalar bool tensor
            commit = run & (a_i.detach().mean() >= 0.5)
            buf = torch.where(commit, buf_tmp, buf)
            count = torch.where(commit, count_tmp, count)
            # Вклады шагов глубины нормируются по L2: гейт (tanh ∈ (−1,1))
            # выбирает знак и силу, но не может через норму r_i взорвать
            # h_acc — средневзвешенное стабильно при любом числе шагов.
            # Шаг 0 без нормировки: сохраняет старое одностороннее поведение.
            r_contrib = r_i.unsqueeze(1)
            if i > 0:
                r_contrib = F.normalize(r_contrib, dim=-1)
            contrib = a_i * r_contrib
            w_soft = torch.ones((), device=h.device)
            if i > 0:
                # Валидация схождения «знание → лакуна → знание»: гейт уже дал
                # право на вклад, зная кандидата; валидация лишь масштабирует
                # СИЛУ вклада по реальному приросту уверенности головы. Не
                # обнуляет градиент гейта: он течёт всегда (по знаку — открыть
                # полезный шаг / закрыть вредный).
                with torch.no_grad():
                    h_n = F.normalize(h.mean(1), dim=-1)  # (B, D)
                    field = h_n
                    if state is not None and len(state) > 0:
                        s_last = state[-1]
                        if s_last is not None and len(s_last) > 0 and s_last[0] is not None:
                            mem = s_last[0]
                            if mem.dim() == 2 and mem.shape[1] % h.shape[-1] == 0:
                                mem_n = F.normalize(
                                    mem.reshape(mem.shape[0], -1, h.shape[-1]).mean(1), dim=-1)
                                field = F.normalize(h_n + mem_n, dim=-1)
                    r_n = F.normalize(r_i, dim=-1)
                    sim = (r_n * field).sum(-1)  # (B,) связь с известным
                    contrib_pre = contrib if weighted is None else weighted + contrib
                    conf_after = self._last_conf(
                        h + s * contrib_pre / (denom_accum + a_i.detach().clamp(min=0) + 1e-6))
                    delta_norm = (conf_after - conf_base) / (conf_base + 1e-6)
                    # Валидация по среднему батчу (не per-sample): per-sample
                    # при conf≈0.016 шумит (delta_norm=±0.3), w_soft скачет —
                    # вклад проходит неровно и дестабилизирует h_acc.
                    delta_avg = delta_norm.mean().clamp(-1.0, 1.0)
                    w_soft = torch.sigmoid(20.0 * delta_avg)  # скаляр: сила валидации
                contrib = contrib * w_soft
            # run-маска: неисполненные шаги дают ровно ноль вклада
            contrib = contrib * run.float()
            # Знаменатель — только ВЗЯТЫЕ для рассуждения шаги (a_i > 0):
            # антизнание (a_i < 0) вычитается в числителе — это концепт в
            # своём потенциале, он не должен разбавлять нормировку и
            # ослаблять полезный вклад шага 0.
            w_i = a_i.detach().clamp(min=0) * w_soft * run.float()
            weighted = contrib if weighted is None else weighted + contrib
            denom_accum = denom_accum + w_i
            # Средневзвешенное с учётом разности положительного и
            # отрицательного: знаменатель — Σ взятых (положительных) весов
            # (антизнание не разбавляет нормировку), НО с floor 0.5 — половиной
            # веса одного полнооткрытого шага (a_i∈(−1,1), w_soft≤1 ⇒ шаг weigh-
            # ит ≤1; floor — вывод из диапазона весов, не подбор). Без floor
            # все-отрицательные гейты давали |accum| ≈ |Σneg·r|/1e-6 — взрыв
            # h_acc на 6 порядков (audit M9; accum остаётся ≤ 2·max|r_i|).
            accum = weighted / denom_accum.clamp(min=0.5)
            h_acc = h + s * accum
            prev_open = a_i.detach().mean()
            gates.append(a_i.detach().mean() * w_soft * run.float())
        if gates:
            g = torch.stack(gates)
            self._reasoning_gates.zero_()
            self._reasoning_gates[:g.shape[0]].copy_(g)
        return h_acc

    @property
    def reasoning_scale(self):
        if self.reasoning_scale_override is not None:
            return self.reasoning_scale_override
        k = max(getattr(self.cfg, 'reasoning_ramp_steps', 1000), 1)
        t = self.reasoning_enabled_step
        return max(1.0 - math.exp(-t / k), 1e-3)

    def reset_reasoning(self):
        """Reset reasoning buffer (call at start of new sequence)."""
        self._reasoning_buffer = None
        self._reasoning_count = None

    def embed_tokens(self, tokens):
        """Token indices -> D-space vectors."""
        return self.embed(tokens)

    @property
    def projector_signals(self):
        """(write_event, concept_id) из слоя концептов — сигналы прожектора.

        write_event: (B, L) bool — границы слов (события записи концептов)
        concept_id:  (B, L) long — слот концепта на позиции
        Возвращает (None, None), если слой концептов отключен.
        """
        for l in self.layers:
            col = getattr(l, 'collective', None)
            if col is not None and hasattr(col, '_write_event'):
                return col._write_event, col._concept_id
        return None, None
    
    def compute_loss(self, h, targets, pred_weight=None, h_emb=None):
        """Returns CE only (aux losses applied via gradient scaling in training step)."""
        ce_loss, _ = self.compute_losses(h, targets, pred_weight=pred_weight, h_emb=h_emb)
        return ce_loss

    def compute_salience(self, logits):
        # Word importance from the head's output field (head_mode='sigmoid_coded'):
        # how strongly / confidently the model responds at each position. Now fed
        # the actual head log-probs (via model.lm_head(out)), so this is true
        # prediction confidence rather than a proxy of the hidden state. Normalized
        # to mean 1 => relative per-position weighting in O(1), robust to the head's
        # output scale (log-prob norms are ~0.01-0.5, far smaller than the old h-based
        # ~15-25). Detached => no gradient feedback loop; the intent path is instead
        # regularized via tau_config.intent_alpha.
        s = logits.sigmoid().norm(dim=-1, keepdim=True)  # (B, L, 1)
        return s / s.mean().clamp_min(1e-6)

    @torch.no_grad()
    def observe_output(self, logits):
        # Store salience of THIS step's output for use as the next step's
        # intent signal (1-step delay). Keeps the loop stable and geometry clean.
        self._last_salience = self.compute_salience(logits).detach()

    def process_with_cache(self, h: torch.Tensor, logits: torch.Tensor,
                           use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process logits through cache and augment hidden state.

        Call this after lm_head to:
        1. Store logits in cache (compressed, per-scale)
        2. Attend to cached logits (long-context memory)
        3. Return augmented hidden state

        Args:
            h: (B, L, D) hidden state from forward pass
            logits: (B, L, V) logits from lm_head
            use_cache: if True, use cache; if False, return h and logits unchanged

        Returns:
            h_augmented: (B, L, D) hidden state augmented with cache info
            logits_out: (B, L, V) logits (unchanged)
        """
        if self.logit_cache is None or not use_cache:
            return h, logits

        # Process through cache (stores + attends)
        # training=True: stores h (gradient flows); training=False: stores logits (compressed)
        h_augmented, logits_out = self.logit_cache(
            h, logits,
            training=self.training,
            use_cache=True
        )

        return h_augmented, logits_out

    def reset_cache(self):
        """Clear the logit cache (for new sequence)."""
        if self.logit_cache is not None:
            self.logit_cache.cache.clear()
        # Restore = a fresh healthy state: the mlp-scale observer must re-warm
        # from the restored weights, not carry a pre-rollback baseline.
        for l in self.layers:
            if getattr(l, '_mlp_cnt', None) is not None and l._mlp_cnt.item() != 0:
                l._mlp_cnt.zero_()
                l._mlp_now_ema.zero_()
                l._mlp_base_ema.zero_()
        # Audit M8: rollback restores weights, but the NON-persistent runtime
        # EMAs (bus RMS, ig norms, delta-var, scheduler temps …) survive the
        # load — a NaN-poisoned EMA after rollback means an instant NaN zombie
        # (the skip-step loop never recovers). Scrub any non-finite buffer.
        with torch.no_grad():
            tc = getattr(self, 'tau_config', None)
            if tc is not None:
                for _bn, _bv in (('_log_tau_min', math.log(tc.tau_min)),
                                 ('_log_tau_range', math.log(tc.tau_max / tc.tau_min))):
                    _b = getattr(tc, _bn, None)
                    if _b is not None and not torch.isfinite(_b).all():
                        _b.fill_(_bv)          # recompute, never zero
            for b in self.buffers():
                if b.is_floating_point() and not torch.isfinite(b).all():
                    b.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
            for l in self.layers:              # non-buffer holders survive load_state_dict
                mir = getattr(l, 'mirror', None)
                if mir is not None:
                    for _an in ('_cached_hp', '_cached_pred_k', '_cached_pred_error_norm',
                                '_pred_loss_term', '_cached_gate', '_traj_state'):
                        if hasattr(mir, _an):
                            setattr(mir, _an, None)

    def snapshot_runtime_buffers(self) -> dict:
        """Detached copies of EVERY buffer (incl. persistent=False).

        Eval-isolation contract (audit M8): forward mutates streaming state
        (memory banks, intent bus, EMAs). Validation passes run a FULL forward
        on val documents — without this snapshot/restore pair around eval,
        validation sentences are consolidated into the TRAINING working
        memory (and vice versa), cross-contaminating the VSA 'document state'.
        Parameters are not touched by eval (no optimizer step) → buffers only.
        """
        snap = {k: v.detach().clone() for k, v in self.named_buffers()}
        _ex = {}
        for _an in ('_last_bus', '_intent_stream'):        # B1: non-buffer streaming state
            _a = getattr(self, _an, None)
            if isinstance(_a, torch.Tensor):
                _ex[_an] = _a.detach().clone()
            elif isinstance(_a, (list, tuple)):
                _ex[_an] = [t.detach().clone() if isinstance(t, torch.Tensor) else t for t in _a]
        snap['__attrs__'] = _ex
        return snap

    def restore_runtime_buffers(self, snap: dict) -> None:
        own = dict(self.named_buffers())
        with torch.no_grad():
            for k, v in snap.items():
                if k == '__attrs__':
                    continue
                buf = own.get(k)
                if buf is not None and buf.shape == v.shape:
                    buf.copy_(v)
            for _an, _v in (snap.get('__attrs__') or {}).items():   # B1 restore path
                _cur = getattr(self, _an, None)
                if isinstance(_v, list) and isinstance(_cur, list):
                    for _i2, _t in enumerate(_v):
                        if _i2 < len(_cur) and isinstance(_t, torch.Tensor) \
                                and isinstance(_cur[_i2], torch.Tensor) \
                                and _cur[_i2].shape == _t.shape:
                            _cur[_i2].copy_(_t)
                        else:
                            _cur[_i2] = _t
                elif isinstance(_v, torch.Tensor):
                    setattr(self, _an, _v.clone())
                else:
                    setattr(self, _an, _v)

    def cache_size_mb(self) -> float:
        """Get current cache size in MB."""
        if self.logit_cache is not None:
            return self.logit_cache.cache.size_mb(training=self.training)
        return 0.0

    def compute_losses(self, h, targets, pred_weight=None, h_emb=None):
        """Compute CE and auxiliary losses separately. Returns raw (unweighted) values.

        h_emb: (optional) эмбеддинг-вход для кодечной головы (двухконечное чтение).

        Returns:
            ce_loss: scalar, cross-entropy loss
            aux_dict: dict of named auxiliary losses (raw, unweighted).
        """
        return _compute_losses_fn(self, h, targets, pred_weight, h_emb)
    
    @staticmethod
    def _checkpointed_block(layer, h, state, global_state,
                             _cached_pred_error_norm, _cached_hp,
                             mem2v_scale, diff, noise_scale,
                             tanh_bias_mod, pred_scale_mod, spectral_mod,
                             context_mem, allow_write, tau_s, step, intent=None,
                             salience=None, maturity=None):
        """Wrapper for gradient checkpointing.
        Mirror cache is passed as explicit args/returns so checkpoint saves/restores it,
        preventing stale-cache mismatch between forward and backward recomputation."""
        layer.mirror._cached_pred_error_norm = _cached_pred_error_norm
        layer.mirror._cached_hp = _cached_hp
        h_out, s_out = layer(h, state, global_state=global_state,
                             mem2v_scale=mem2v_scale, diff=diff, noise_scale=noise_scale,
                             tanh_bias_mod=tanh_bias_mod, pred_scale_mod=pred_scale_mod,
                             spectral_mod=spectral_mod, context_mem=context_mem,
                             allow_write=allow_write, tau_s=tau_s, step=step,
                             intent=intent, salience=salience, maturity=maturity)
        return h_out, s_out, layer.mirror._cached_pred_error_norm, layer.mirror._cached_hp

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def apply_mlp_depth_gradient_boost(self, exp=None):
        """Counter vanishing gradient to deep MLP layers (diagnostic: ~10k-20k x
        collapse of MLP gradient by depth 8). Scales the gradient of every MLP
        param in layer i by exp(exp * i) via a backward hook — optimizer- and
        resume-safe (hooks live on the params, not the optimizer state).
        exp defaults to cfg.mlp_depth_lr_exp; 0 disables."""
        import math
        exp = float(exp if exp is not None else getattr(self.cfg, 'mlp_depth_lr_exp', 0.0))
        if exp <= 0:
            return
        n_applied = 0
        for i, layer in enumerate(self.layers):
            boost = math.exp(exp * i)
            if abs(boost - 1.0) < 1e-6:
                continue
            # GroupedMLP internal params (W_up/down, gate, mlp_gate_b)
            for p in layer.mlp.parameters():
                if p.requires_grad:
                    p.register_hook(lambda grad, b=boost: grad * b)
                    n_applied += 1
            # Mirror cognitive gates — hybrid_gate (sigmoid+softmax) and mod_scale_mem.
            # No artificial boost: hybrid_gate learns at the same rate as other
            # mirror parameters through normal backprop. Specialization is controlled
            # by tau (softmax temperature), not gradient amplification.
        print(f'[mlp-boost] deep-MLP gradient x{math.exp(exp):.2f}/layer '
              f'(L0=1.0 .. L{len(self.layers)-1}={math.exp(exp*(len(self.layers)-1)):.1f}), '
              f'{n_applied} params hooked')


    @torch.no_grad()
    def collective_stats(self):
        """Per-layer summary of the Collective Concept Layer (col: log).
        Returns None when the collective layer is disabled."""
        cols = [l for l in self.layers if l.collective is not None]
        if not cols:
            return None
        out = {
            'step': int(cols[0].collective._step.item()),
            'mature': int(sum(1 for l in cols if l.collective._mature.item())),
            'writes': int(sum(l.collective.N_s.sum().item() for l in cols)),
            'act': float(sum(l.collective.U_s.mean().item() for l in cols) / len(cols)),
            'occ': float(sum((l.collective.U_s > 0.01).float().mean().item() for l in cols) / len(cols)),
            'last_write_step': max(l.collective._last_write_step for l in cols),
            'births_allowed': int(sum(l.collective._births_allowed for l in cols)),
            'births_skipped_novelty': int(sum(l.collective._births_skipped_novelty for l in cols)),
            'novelty_threshold': cols[0].collective._novelty_threshold,
        }
        return out
    
    def param_groups(self, lr=None, weight_decay=None, gate_lr_mult=None):
        """Optimizer parameter groups with λ_d LR hierarchy or legacy flat groups.
        
        When cfg.lambda_lr_hierarchy=True (default), groups follow λ_d^p:
          p=-2: embedding, readout       (0.29×)
          p=-1: MLP cores, bind W_proj   (0.54×)
          p= 0: conv, norm, W_out, head  (1.00×)
          p=+1: mirror projections, α    (1.84×)
          p=+2: gates, w_i, b_i, etc     (3.38×)
          vsa:  b_d, b_i                 (λ^{-2})
          bridge: bridge_*, intent_*     (bridge_lr_mult ×, default 0.1×)

        Role routing uses EXACT dotted-name parts (shared with
        core.adaptation._role_lr_mult) — the old substring tests let
        'b_delta_gate'/'w_delta_gate' fall into the λ⁻² VSA bucket before the
        gate branch could claim them (audit M7).
        """
        from .adaptation import _VSA_PARTS, _GATE_PARTS, _MIRROR_PARTS
        cfg = self.cfg
        lr = lr or cfg.lr
        wd = weight_decay or cfg.weight_decay
        bridge_lr = lr  # bridge uses base LR (LayerBridgeGate handles routing)
        
        if getattr(cfg, 'lambda_lr_hierarchy', False):
            from .lambda_utils import lambda_d
            lam = lambda_d(cfg.lambda_d)
            mlr = {
                'embed': lam ** (-2),
                'mlp': lam ** (-1),
                'vsa': lam ** (-2),
                'mirror': lam ** (1),
                'gate': lam ** (1),
            }
            groups = {
                'embed':    {'params': [], 'lr': lr * mlr['embed'], 'weight_decay': 0},
                'embed_wd': {'params': [], 'lr': lr * mlr['embed'], 'weight_decay': wd},
                'mlp':      {'params': [], 'lr': lr * mlr['mlp'],   'weight_decay': 0},
                'mlp_wd':   {'params': [], 'lr': lr * mlr['mlp'],   'weight_decay': wd},
                'mirror':   {'params': [], 'lr': lr * mlr['mirror'],'weight_decay': 0},
                'mirror_wd':{'params': [], 'lr': lr * mlr['mirror'],'weight_decay': wd},
                'gate':     {'params': [], 'lr': lr * mlr['gate'],  'weight_decay': 0},
                'gate_wd':  {'params': [], 'lr': lr * mlr['gate'],  'weight_decay': wd},
                'vsa':      {'params': [], 'lr': lr * mlr['vsa'],   'weight_decay': 0},
                'tau_dev':  {'params': [], 'lr': lr * getattr(cfg, 'tau_dev_lr_mult', 0.2), 'weight_decay': 0},
                'bridge':   {'params': [], 'lr': bridge_lr,          'weight_decay': wd},
                'bridge_nd':{'params': [], 'lr': bridge_lr,          'weight_decay': 0},
                'default':  {'params': [], 'lr': lr,                'weight_decay': 0},
                'default_wd':{'params': [], 'lr': lr,               'weight_decay': wd},
            }
            for name, p in self.named_parameters():
                # block-level standalone-fallback ladder: unused when the
                # stack passes tau_s, so keep it OUT of the optimizer state
                # (audit M7; standalone blocks/tests build their own optimizers)
                if name.endswith('._vsa_tau_log'):
                    continue
                # τ-config params: dedicated groups (check BEFORE bridge to avoid misrouting)
                if 'tau_config.' in name or '_tau_l_dev' in name:
                    groups['tau_dev']['params'].append(p)
                # Bridge params: bridge.*, bridge_glu_net.*, intent_probe, bus_head_proj
                elif ('bridge.' in name or 'bridge_glu_net' in name
                      or 'intent_probe' in name or 'bus_head_proj' in name
                      or 'layer_bridge_gate.' in name):
                    k = 'bridge' if p.ndim >= 2 else 'bridge_nd'
                    groups[k]['params'].append(p)
                elif frozenset(name.split('.')) & _VSA_PARTS:
                    groups['vsa']['params'].append(p)
                elif name.startswith('embed.') or name.startswith('lm_head.readout') or name.startswith('lm_head.proj'):
                    k = 'embed_wd' if p.ndim >= 2 else 'embed'
                    groups[k]['params'].append(p)
                elif any(g in name for g in ['.mirror.alpha_diag',
                                              '.log_skip_alpha', '.mirror.W_proj', '.mirror.W_out',
                                              '.mirror.w_temp', '.mirror.w_global',
                                              '.mirror.log_scale', '.mirror.tanh_bias',
                                              '.log_dvar_mod_scale', '.dvar_mod_bias',
                                              '.log_grad_mod_scale', '.grad_mod_bias']):
                    # Mirror projections, alpha, gates -> mirror LR (1.84x)
                    # alpha_diag is gate-like (G,K) diagonal -> never weight-decayed
                    k = 'mirror_wd' if (p.ndim >= 2 and '.alpha_diag' not in name and '.log_scale' not in name) else 'mirror'
                    groups[k]['params'].append(p)
                elif '.mlp.' in name or '.bind.W_proj.weight' in name or name.endswith('.W_out') or name.endswith('.W_proj'):
                    # Block-level W_proj/W_out (not mirror, caught above) -> mlp speed (0.54x)
                    k = 'mlp_wd' if p.ndim >= 2 else 'mlp'
                    groups[k]['params'].append(p)
                elif 'reasoning_gate' in name:
                    # Adaptive reasoning gates — gate-like LR (fast adaptation), no decay
                    k = 'gate' if p.ndim < 2 else 'gate_wd'
                    groups[k]['params'].append(p)
                elif (frozenset(name.split('.')) & _GATE_PARTS):
                    k = 'gate_wd' if p.ndim >= 2 else 'gate'
                    groups[k]['params'].append(p)
                else:
                    k = 'default_wd' if p.ndim >= 2 else 'default'
                    groups[k]['params'].append(p)
            return [v for v in groups.values() if v['params']]
        
        # ─── Legacy groups (lambda_lr_hierarchy=False) ───
        gate_lr_mult = cfg.gate_lr_mult if gate_lr_mult is None else gate_lr_mult
        decay = []
        no_decay = []
        gate_decay = []
        gate_no_decay = []
        vsa_bias = []
        bridge_decay = []
        bridge_no_decay = []
        for name, p in self.named_parameters():
            if name.endswith('._vsa_tau_log'):
                continue
            if frozenset(name.split('.')) & _VSA_PARTS:
                vsa_bias.append(p)
                continue
            # Bridge params: bridge.*, bridge_glu_net.*, intent_probe, bus_head_proj
            is_bridge = ('bridge.' in name or 'bridge_glu_net' in name
                         or 'intent_probe' in name or 'bus_head_proj' in name
                         or 'layer_bridge_gate.' in name)
            if is_bridge:
                if p.ndim < 2:
                    bridge_no_decay.append(p)
                else:
                    bridge_decay.append(p)
                continue
            _parts = frozenset(name.split('.'))
            is_gate = bool(_parts & (_GATE_PARTS | _MIRROR_PARTS)) \
                or (('W_proj' in _parts or 'W_out' in _parts) and 'mirror' in _parts)
            if 'reasoning_gate' in name:
                is_gate = True
            if is_gate:
                if p.ndim < 2:
                    gate_no_decay.append(p)
                else:
                    gate_decay.append(p)
            else:
                if p.ndim < 2:
                    no_decay.append(p)
                else:
                    decay.append(p)
        groups = [
            {'params': decay, 'lr': lr, 'weight_decay': wd},
            {'params': no_decay, 'lr': lr, 'weight_decay': 0},
        ]
        if gate_decay:
            groups.append({'params': gate_decay, 'lr': lr * gate_lr_mult, 'weight_decay': wd})
        if gate_no_decay:
            groups.append({'params': gate_no_decay, 'lr': lr * gate_lr_mult, 'weight_decay': 0})
        if vsa_bias:
            vsa_lr_mult = getattr(cfg, 'vsa_b_lr_mult', 0.1)
            groups.append({'params': vsa_bias, 'lr': lr * vsa_lr_mult, 'weight_decay': 0})
        if bridge_decay:
            groups.append({'params': bridge_decay, 'lr': bridge_lr, 'weight_decay': wd})
        if bridge_no_decay:
            groups.append({'params': bridge_no_decay, 'lr': bridge_lr, 'weight_decay': 0})
        return groups


# ─── AdaptiveController imported from core.adaptive_controller ─────




# ─── Verify ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = EVAConfig(n_layers=24, D=896, bind_K=32, mlp_groups=8)
    model = EVAStack(cfg).to(device)
    n = model.param_count()
    print(f'  D=896 G=8: params={n:,} ({n/1e6:.2f}M)')
    
    print()
    cfg = EVAConfig(n_layers=4, D=896, bind_K=32)
    model = EVAStack(cfg).to(device)
    
    x = torch.randint(0, cfg.vocab, (2, 16), device=device)
    h = model.embed_tokens(x)
    out, state, _, _ = model(h)
    loss = model.compute_loss(out[:, :-1], x[:, 1:])
    loss.backward()
    
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    out_std = out.std().item()
    print(f'Output: {out.shape}  std={out_std:.4f}')
    print(f'Loss: {loss.item():.4f}  Grad: {total_grad:.4f}')
    print('OK' if not math.isnan(loss.item()) and total_grad > 0 else 'FAIL')
