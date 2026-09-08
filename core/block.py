"""EVA: block module."""

from __future__ import annotations
from typing import Optional, Tuple, List

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import EVAConfig
from .bind import BottleneckBind, SpiralBind, TrajectorySpiralBind, TrajectoryManifoldBind
from .mirror import GroupedCognitiveMirror
from .concept_layer import UnifiedConceptLayer
from .mlp import GroupedMLP
from .vsa_utils import dct_basis, fib_sigmoid_init

# ─── Module-level prefix scan (hoisted from EVABlock.forward) ───

_EPS_SCAN = 1e-6

def _scan_chunk(b_chunk: torch.Tensor, d_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parallel chunk scan from zero state.
    Returns intra-chunk VSA (B, chunk_len, S*D), final state (B, 1, S*D),
    cumulative decay (B, chunk_len, S*D).
    """
    log_a = torch.log(d_chunk.clamp(min=_EPS_SCAN))
    log_cum = torch.cumsum(log_a, dim=1)
    cum_decay = torch.exp(log_cum)
    inv_cum = (1.0 / cum_decay.clamp(min=_EPS_SCAN)).clamp(max=1e6)
    weighted = b_chunk * inv_cum
    cum_w = torch.cumsum(weighted, dim=1)
    intra = cum_decay * cum_w
    final = intra[:, -1:]
    return intra, final, cum_decay

def _combine_chunks(chunk_data: list, initial_state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2nd-level: cross-chunk prefix scan over K chunk states.
    Returns combined (B, L, S*D), final_state (B, S*D), leaf (B, L, S*D).
    """
    inter_decay = torch.cat([cd[:, -1:] for _, _, cd in chunk_data], dim=1)
    inter_input = torch.cat([f for _, f, _ in chunk_data], dim=1)
    s = initial_state.clone() if initial_state is not None else torch.zeros_like(inter_input[:, 0])
    cross_states = []
    for k in range(len(chunk_data)):
        cross_states.append(s.unsqueeze(1))
        s = inter_decay[:, k] * s + inter_input[:, k]
    cross = torch.cat(cross_states, dim=1)
    combined_pieces = []
    leaf_pieces = []
    for k, (intra_k, _, cum_decay_k) in enumerate(chunk_data):
        cross_k = cross[:, k:k+1]
        combined_pieces.append(cross_k * cum_decay_k + intra_k)
        leaf_pieces.append(intra_k)
    combined = torch.cat(combined_pieces, dim=1)
    leaf = torch.cat(leaf_pieces, dim=1)
    return combined, combined[:, -1], leaf


class PrecisionGate(nn.Module):
    def __init__(self, D: int) -> None:
        super().__init__()
        self.gate: nn.Linear = nn.Linear(D, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(h))


class ExactSequenceMemory(nn.Module):
    def __init__(self, D: int, k: int, softmax_free: bool = True) -> None:
        super().__init__()
        self.query: nn.Linear = nn.Linear(D, k)
        self.key: nn.Linear = nn.Linear(D, k)
        self.value: nn.Linear = nn.Linear(D, k)
        self.proj: nn.Linear = nn.Linear(k, D)
        self.k: int = k
        self.softmax_free: bool = softmax_free

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        if self.softmax_free:
            # LaCUR: сигмоид-нормированное среднее (проводимость, не конкуренция).
            # Нормировка по сумме держит выход в выпуклой оболочке -> без взрыва.
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.k)
            A = torch.sigmoid(scores)
            A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            return self.proj(A @ v)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.k), dim=-1)
        return self.proj(attn @ v)

class EVABlock(nn.Module):
    """
    Hybrid block: D -> K (bottleneck bind) + VSA memory + Conv + Spectral + MLP.
    
    Key design decisions:
    - Pre-LN: RMS norm at block start
    - Bind: D->K projection, bilinear in K, K->D projection
    - Memory: VSA vector superposition (not covariance matrix)
    - Gates: per-dim element-wise
    - Conv: depthwise 48-tap
    - Spectral: DCT basis scaling
    - MLP: D -> bottleneck -> D with residual
    """
    
    def __init__(self, cfg: EVAConfig, layer_idx: int, tau_config: Optional[object] = None) -> None:
        super().__init__()
        self.D: int = cfg.D
        self.K: int = cfg.bind_K
        self.layer_idx: int = layer_idx
        self.tie_bind: bool = cfg.tie_bind
        # Store τ_norm for this layer (U1, U3)
        self._tau_norm: Optional[float] = None
        if tau_config is not None and hasattr(tau_config, 'tau_norm'):
            with torch.no_grad():
                self._tau_norm = tau_config.tau_norm[layer_idx].item()
        # Keep the τ-field so the mirror can bind its gate authorities to τ
        # (intent_alpha etc.); previously the mirror always saw tau_config=None,
        # which silently disabled all τ-ties inside GroupedCognitiveMirror.
        self.tau_config: Optional[object] = tau_config
        
        # Pre-LN weight
        self.register_buffer('pre_ln_w', torch.ones(cfg.D))
        self.total_layers = cfg.n_layers
        
        bind_mode = getattr(cfg, "bind_twist_mode", "shift")
        if bind_mode == "trajectory_spiral":
            if getattr(cfg, "traj_manifold", False):
                # FCF-манифолд: лучи переходов + Zeckendorf (fp32-стабильный)
                self.bind = TrajectoryManifoldBind(cfg.D, cfg.bind_K, cfg)
            else:
                self.bind = TrajectorySpiralBind(cfg.D, cfg.bind_K, cfg)
        elif bind_mode == "spiral":
            self.bind = SpiralBind(cfg.D, cfg.bind_K, cfg)
        else:
            self.bind = BottleneckBind(cfg.D, cfg.bind_K, cfg)
        # U10: set τ_norm on bind module for frequency schedule
        if self._tau_norm is not None and hasattr(self.bind, '_tau_norm'):
            self.bind._tau_norm = self._tau_norm

        # Cognitive Mirror (32 эксперта, grouped K-space)
        if getattr(cfg, 'mirror_k_staircase', False):
            # Иерархия k_l: 8/16/32 по третям глубины
            n = cfg.n_layers
            l = layer_idx
            if l < n // 3:
                k = 8      # L0-L(ṇ/3): широкое K-space
            elif l < (2 * n) // 3:
                k = 16     # среднее K-space
            else:
                k = 32     # глубокие слои: узкое K-space
        else:
            k = cfg.mirror_k
        self.mirror = GroupedCognitiveMirror(cfg.D, G=cfg.mlp_groups, k=k,
            log_scale_init_std=cfg.log_scale_init_std,
            delta_var_ema_min=cfg.delta_var_ema_min, delta_var_ema_max=cfg.delta_var_ema_max,
            tie_mirror_proj=cfg.tie_mirror_proj,
            layer_idx=layer_idx, n_layers=cfg.n_layers,
            has_private_mem=getattr(cfg, 'private_mem', False),
            expert_asymmetry=getattr(cfg, 'expert_asymmetry', False),
            meta_trust=getattr(cfg, 'meta_trust', False),
            gate_bias_scale=0.5 + 1.5 * layer_idx / max(cfg.n_layers - 1, 1) if getattr(cfg, 'gate_bias_scale_per_layer', False) else cfg.gate_bias_scale,
            alpha_novelty_weight=getattr(cfg, 'alpha_novelty_weight', 0.0),
            seq_len=cfg.seq_len,
            intent_bridge=getattr(cfg, 'intent_bridge', False),
            bridge_glu=getattr(cfg, 'bridge_glu', False),
            bridge_glu_beta=getattr(cfg, 'bridge_glu_beta', 0.25),
            pm_write_delay=getattr(cfg, 'pm_write_delay', 5000),
            pm_coh_gate_std=getattr(cfg, 'pm_coh_gate_std', 0.02),
            mirror_tau_min=getattr(cfg, 'mirror_tau_min', 2.0),
            mirror_tau_max=getattr(cfg, 'mirror_tau_max', 200.0),
            tau_config=tau_config)
        
        # ─── VSA Memory (multi-scale VSA: S=4 фиксированных τ) ───
        self._n_scales = 4
        # U1: τ-consistent VSA scales: learnable log-space params (base distribution)
        self._vsa_tau_log = nn.Parameter(torch.tensor([math.log(8), math.log(32), math.log(128), math.log(512)]))
        # Keep old buffer for backward compat (unused in forward when tau_config provided)
        tau_s = torch.tensor([8, 32, 128, 512], dtype=torch.float32)
        self.register_buffer('_tau_s', tau_s)
        self.w_i = nn.Parameter(torch.randn(cfg.D))          # content-dependent write gate (shared across scales)
        self.w_d = nn.Parameter(torch.randn(cfg.D) * cfg.w_d_init_std)    # content-dependent decay modulation
        self.w_q = nn.Parameter(torch.full((cfg.D,), 1.0 / math.sqrt(cfg.D)))  # warm read: mem_read ≈ mem_all at init
        self.w_q_leaf = nn.Parameter(torch.full((cfg.D,), 1.0 / math.sqrt(cfg.D)))  # leaf-level within-chunk read
        self.w_q_ctx = nn.Parameter(torch.full((cfg.D,), 0.5 / math.sqrt(cfg.D)))  # cross-chunk context read
        self.w_mem2v = nn.Parameter(torch.randn(cfg.D))
        # Per-expert dynamic VSA memory parameters
        g = self.mirror.G
        d = self.mirror.d
        k = self.mirror.k
        self.w_q_dyn = nn.Parameter(torch.randn(g, k, d) * (1.0 / math.sqrt(k)))
        self.w_i_dyn = nn.Parameter(torch.randn(g, k, d) * (1.0 / math.sqrt(k)))
        self.w_d_pen = nn.Parameter(torch.zeros(g))
        self.w_bind_gate = nn.Parameter(torch.zeros(g))
        # Per-scale per-channel combination weights (logits for softmax)
        self.scale_w = nn.Parameter(fib_sigmoid_init(self._n_scales).unsqueeze(1).expand(-1, cfg.D).clone())
        # Linear decay across layers: shallow → short memory, deep → long
        # Per-channel (D,) — can differentiate via gradient when vsa_b_d_smooth < 1.0
        layer_frac = layer_idx / max(cfg.n_layers - 1, 1)
        b_d_init = 2.0 + 3.0 * layer_frac  # L0: τ≈7, L23: τ≈63, L31: τ≈150
        self.b_i = nn.Parameter(torch.full((cfg.D,), -2.5))   # i_gate ~0.08 init
        self.b_d = nn.Parameter(torch.full((cfg.D,), b_d_init))
        # Surprisal-gated write coefficient γ_l: растёт с τ
        # γ_l = γ_max · σ((ln τ_l - ln 32) / 1.0)
        tau_l = math.exp(b_d_init)
        gamma_max = 0.5
        gamma_init = gamma_max * (1.0 / (1.0 + math.exp(-(math.log(tau_l) - math.log(32.0)))))
        self.gamma_surprisal = nn.Parameter(torch.full((), gamma_init))
        # Когерентность спиралей → запись в VSA-память (опорные точки скрещивания фаз)
        self.bind_coh_gate = nn.Parameter(torch.tensor(0.5))

        # First moment
        self.w_k_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_q_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_mu_mem = nn.Parameter(torch.randn(cfg.D))
        
        # ─── Conv ───
        self.conv = nn.Conv1d(cfg.D, cfg.D, kernel_size=cfg.conv_kernel,
                              padding=cfg.conv_kernel - 1, groups=cfg.D, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        
        # ─── Spectral (self-organizing frequency filters) ───
        self.register_buffer('V_dct', dct_basis(cfg.D))
        # ─── MLP-output runaway tracker (self-referencing, τ-linked) ───
        # fast (0.99) vs slow (0.999) EMA of ‖h_mlp‖ — same a/a_slow convention as
        # the FailureDetector. Healthy ratio ≈ 1; a ×2–50 runaway lifts it past the
        # detector's relative margin within a few steps, unlike an absolute SPC
        # bound which the runaway's growing variance absorbs.
        self.register_buffer('_mlp_now_ema', torch.ones(1), persistent=False)
        self.register_buffer('_mlp_base_ema', torch.ones(1), persistent=False)
        self._mlp_ratio = 1.0
        base = 0.5 + layer_idx / max(cfg.n_layers - 1, 1)
        # Per-dim variation: low frequencies get slight boost, high get slight cut
        # Creates natural 1/f-like distribution encouraging frequency band separation
        freq_scale = torch.linspace(1.0, 0.5, cfg.D)  # DC amp=1, Nyquist=0.5
        per_dim = freq_scale * 0.2  # 20% variation across freq spectrum
        lam = torch.full((cfg.D,), base) + per_dim
        self.lambda_k = nn.Parameter(lam)
        
        # ─── MLP (grouped: per-group 4× expansion, half params) ───
        self.mlp = GroupedMLP(cfg.D, expand=cfg.mlp_expand, groups=cfg.mlp_groups,
                              swiglu=getattr(cfg, 'mlp_swiglu', True),
                              gate_b_init=getattr(cfg, 'mlp_gate_b_init', 0.25))

        # ─── Variable Precision Memory ───
        self.precision_gate = PrecisionGate(cfg.D)
        exact_k = min(64, cfg.D // 4)
        self.exact_memory = ExactSequenceMemory(cfg.D, exact_k,
                                                 getattr(cfg, 'softmax_free', True))
        self.variable_precision = getattr(cfg, 'variable_precision', False)
        self.precision_threshold = getattr(cfg, 'precision_threshold', 0.3)

        # Collective concept layer moved to stack.py (UnifiedConceptLayer — global)
        self.collective = None
    
    def forward(self, h: torch.Tensor, state: Optional[Tuple] = None, global_state: Optional[torch.Tensor] = None,
                mem2v_scale: float = 1.0, diff: Optional[torch.Tensor] = None, noise_scale: float = 0.0,
                tanh_bias_mod: float = 1.0, pred_scale_mod: Optional[torch.Tensor] = None, spectral_mod: float = 1.0,
                context_mem: Optional[torch.Tensor] = None, allow_write: Optional[bool] = None, tau_s: Optional[torch.Tensor] = None, step: Optional[int] = None, intent: Optional[torch.Tensor] = None,
                salience: Optional[torch.Tensor] = None, maturity: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple]:
        mem_state = mu_state = conv_state = traj_state = pen = None
        if state is not None:
            mem_state, mu_state, conv_state = state[:3]
            if len(state) > 3:
                traj_state = state[3]
            if len(state) > 4:
                pen = state[4]
        B, L, D = h.shape
        NaN = float('nan')
        self._nan_at = None
        def _chk(t, label):
            if not self.training:
                return False
            if t.is_floating_point() and (t.isnan().any() or t.isinf().any()):
                self._nan_at = f'L{self.layer_idx}.{label}[{t.min():.2f},{t.max():.2f}]'
                return True
            return False

        device = h.device
        K = self.K
        S = self._n_scales
        
        # Consistent NaN state shapes (prevent ndim mismatch in next step)
        _nan_conv = torch.zeros(B, D, self.conv.padding[0], device=device) * NaN
        _nan_mem = torch.zeros(B, S * D, device=device) * NaN
        
        # Transfer stale mirror cache (with shape & dtype check)
        if pen is None:
            pen = getattr(self.mirror, '_cached_pred_error_norm', None)
        if pen is not None and (pen.shape[-1] != L or pen.shape[0] != B):
            pen = None
        
        # ─── Pre-LN ───
        h = self.pre_ln_w * h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-7)
        
        # ─── Conv ───
        if conv_state is None:
            conv_state = torch.zeros(B, D, self.conv.padding[0], device=device, dtype=h.dtype)
        h_perm = h.transpose(1, 2)
        h_conv = self.conv(torch.cat([conv_state, h_perm], dim=-1))
        h_conv = h_conv[..., :L].transpose(1, 2)
        conv_state_out = h_perm[:, :, -(self.conv.padding[0]):]
        h = h + h_conv
        if _chk(h, 'conv'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        if self.training:
            self._cache_conv_out = h_conv  # for branch_loss (with grad)
        
        if isinstance(self.bind, TrajectorySpiralBind):
            if traj_state is None and self.training:
                traj_state = getattr(self, '_traj_state', None)
            if traj_state is not None and (traj_state.shape[2] != L
                                           or traj_state.shape[0] != B):
                traj_state = None
            bind_out, new_traj, coherence = self.bind(h, traj_state)
            if traj_state is None:
                self._traj_state = new_traj.detach()
                traj_state_out = None
            else:
                traj_state_out = (0.9 * traj_state + 0.1 * new_traj).detach()
        else:
            bind_out = self.bind(h)
            coherence = torch.zeros(B, L, K, device=device, dtype=h.dtype)
            new_traj = None
            traj_state_out = None
        if _chk(bind_out, 'bind'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        
        # ─── VSA Memory (multi-scale: S=4 фиксированных τ) ───
        S = self._n_scales
        tau_s = torch.exp(self._vsa_tau_log) if tau_s is None else tau_s
        d_s = torch.exp(-1.0 / tau_s.to(device))  # (S,) — τ-scales from learnable param
        # Surprisal-gated write: i_gate = softplus(linear + γ·||ê||₂)
        igate_logit = h * self.w_i + self.b_i
        if pen is not None:
            igate_logit = igate_logit + self.gamma_surprisal * pen.unsqueeze(-1)
        i_gate = F.softplus(igate_logit)                    # (B, L, D)
        # Опорные точки скрещивания спиралей: синхронность фаз усиливает запись
        coh_mean = coherence.mean(dim=-1, keepdim=True)     # (B, L, 1)
        i_gate = i_gate * (1.0 + self.bind_coh_gate * coh_mean.to(i_gate.dtype))
        d_mod = torch.sigmoid(h * self.w_d + self.b_d)      # (B, L, D) — content mod of decay
        if noise_scale > 0 and self.training:
            noise = 1.0 + noise_scale * torch.randn_like(i_gate)
            i_gate = i_gate * noise

        # Prediction-error-aware decay modulation (before decay expansion)
        if pen is not None:
            d_pen_factor = 1.0 - 0.5 * torch.sigmoid(pen.unsqueeze(-1) + self.w_d_pen.unsqueeze(0).unsqueeze(0))
            d_mod = (d_mod.reshape(B, L, self.mirror.G, self.mirror.d) * d_pen_factor.to(d_mod.dtype).unsqueeze(-1)).reshape(B, L, D)

        # Vectorize over S scales: (B, L, S, D) — expand-views, no materialized copies
        d_s_vec = d_s.view(1, 1, S, 1).expand(B, L, S, D)
        d_mod_vec = d_mod.unsqueeze(2).expand(-1, -1, S, -1)
        decay = (d_s_vec * d_mod_vec).clamp(min=0.01, max=1.0)  # per-scale per-channel, floor 0.01 cap 1.0

        # Dynamic write modulation (per-expert K-space conditioning)
        hp_cached = self.mirror._cached_hp
        if (hp_cached is not None and self.training
                and hp_cached.shape[0] == B and hp_cached.shape[1] == L):
            g = self.mirror.G
            d = self.mirror.d
            k = self.mirror.k
            BL = B * L
            hp_g = hp_cached.permute(2, 0, 1, 3).reshape(g, BL, k)  # batched matmul (stable under AMP)
            wm = torch.matmul(hp_g, self.w_i_dyn)  # (g, BL, d)
            write_mod = torch.sigmoid(wm.permute(1, 0, 2).view(B, L, g, d) / math.sqrt(k))
            mem_input = (h.reshape(B, L, g, d) * write_mod).reshape(B, L, D) * i_gate
        else:
            mem_input = h * i_gate  # (B, L, D)

        input_vec = mem_input.unsqueeze(2).expand(-1, -1, S, -1)  # (B, L, S, D) expand-view
        
        eps = 1e-6
        CHUNK = 32
        
        # fp32 guard for log-space scan (critical under AMP for long memory)
        _dtype = decay.dtype
        decay_f32 = decay.float() if decay.dtype != torch.float32 else decay
        input_vec_f32 = input_vec.float() if input_vec.dtype != torch.float32 else input_vec
        if mem_state is not None:
            mem_state_f32 = mem_state.reshape(B, S, D).float()
        else:
            mem_state_f32 = None
        
        # Level 1: parallel chunk scans from zero (module-level _scan_chunk)
        chunks = []
        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            intra, final, cum_decay = _scan_chunk(input_vec_f32[:, start:end], decay_f32[:, start:end])
            chunks.append((intra, final, cum_decay))
        
        mem_all_vec, mem_state_out_vec, mem_leaf_vec = _combine_chunks(chunks, mem_state_f32)
        # Keep VSA in fp32 — prefix scan accumulators underflow/overflow in fp16
        
        # Weighted combination: sigmoid per scale per channel (no sum-to-1)
        w = torch.sigmoid(self.scale_w)  # (S, D)
        mem_all = (mem_all_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, L, D)
        mem_leaf = (mem_leaf_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, L, D) — без кросс-чанк контекста
        # Dual read: leaf (within-chunk, 100% покрытие) + context (cross-chunk)
        mem_read = mem_all * self.w_q + mem_leaf * self.w_q_leaf + mem_all * self.w_q_ctx
        mem_state_out = mem_state_out_vec.reshape(B, S * D)
        
        # First moment (same multi-scale decay, scaled input)
        if mu_state is not None:
            mu_state = mu_state.reshape(B, S, D)
        mu_input_vec = (mem_input * self.w_k_mu).unsqueeze(2).expand(-1, -1, S, -1)
        mu_input_f32 = mu_input_vec.float() if mu_input_vec.dtype != torch.float32 else mu_input_vec
        mu_chunks = []
        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            intra, final, cum_decay = _scan_chunk(mu_input_f32[:, start:end], decay_f32[:, start:end])
            mu_chunks.append((intra, final, cum_decay))
        mu_all_vec, mu_state_out_vec, _ = _combine_chunks(mu_chunks, mu_state)
        mu_all = (mu_all_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        mu_read = mu_all * self.w_q_mu
        mem_read = mem_read + mu_read * self.w_mu_mem
        mu_state_out = mu_state_out_vec.reshape(B, S * D)
        if _chk(mem_read, 'mem_read'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        
        # ─── Mirror (self-consistency: local + global) ───
        # fp32-якорь: exp/log/softmax в mirror переполняются в fp16 под AMP
        with torch.autocast(device_type=h.device.type, enabled=False):
            _gs = global_state.float() if isinstance(global_state, torch.Tensor) else global_state
            _ctx = context_mem.float() if isinstance(context_mem, torch.Tensor) else context_mem
            mirror, mlp_mod, mem_mod, hp, pred_error_norm = self.mirror(
                h.float(), mem_all.float(), global_state=_gs, diff=diff,
                tanh_bias_mod=tanh_bias_mod, pred_scale_mod=pred_scale_mod,
                context_mem=_ctx, allow_write=allow_write, step=step, intent=intent,
                salience=salience, maturity=maturity)
            mirror = mirror.to(h.dtype)
            mlp_mod = mlp_mod.to(h.dtype) if isinstance(mlp_mod, torch.Tensor) else mlp_mod
            self._cache_mlp_mod = mlp_mod  # (B,L,G) per-expert MLP gate (gradalign)
            mem_mod = mem_mod.to(h.dtype) if isinstance(mem_mod, torch.Tensor) else mem_mod
        if _chk(mirror, 'mirror'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        if _chk(mlp_mod, 'mlp_mod'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        if _chk(mem_mod, 'mem_mod'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        
        # ─── Output (adaptive memory scale, per-group modulation) ───
        # mem_mod: per-token, per-expert gating of memory contribution
        mm = mem_mod  # (B, L, G)
        mm = mm.unsqueeze(-1)  # (B, L, G, 1)
        g = self.mirror.G
        d = self.mirror.d
        # Per-expert dynamic read (K-space conditioned memory gating)
        if hp is not None:
            BL = B * L
            hp_g = hp.permute(2, 0, 1, 3).reshape(g, BL, self.mirror.k)  # batched matmul (stable under AMP)
            read_mod = torch.matmul(hp_g, self.w_q_dyn)  # (g, BL, d)
            read_mod = torch.sigmoid(read_mod.permute(1, 0, 2).view(B, L, g, d) / math.sqrt(self.mirror.k))
            mem_read_g = mem_read.reshape(B, L, g, d)
            mem_expert = mem_read_g * read_mod
            mem_modulated = (mem_expert * mm).reshape(B, L, D)
        else:
            mem_modulated = (mem_read.reshape(B, L, g, d) * mm).reshape(B, L, D)
        # Bind gating: per-expert modulation of bind output
        bind_gate = torch.sigmoid(self.w_bind_gate).unsqueeze(0).unsqueeze(0)
        bind_gated = (bind_out.reshape(B, L, g, d) * mm * bind_gate.unsqueeze(-1)).reshape(B, L, D)
        enhanced_base = bind_gated + mem_modulated * self.w_mem2v * mem2v_scale
        enhanced = enhanced_base + mirror
        # Concept layer moved to stack.py (UnifiedConceptLayer — global, after embedding)
        if _chk(enhanced, 'enhanced'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        if self.training:
            self._cache_bind_out = enhanced_base  # for branch_loss (with grad)
            self._cache_mirror_out = mirror  # for branch_loss (with grad)
        h = h + enhanced
        if _chk(h, 'post_enhanced'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)

        # ─── Variable Precision Memory ───
        if self.variable_precision:
            # fp32-якорь: softmax в exact_memory переполняется в fp16 под AMP.
            # Гейт встроен тензорной маской (не python-if): статический граф.
            with torch.autocast(device_type=h.device.type, enabled=False):
                precision = self.precision_gate(h.float())
                gate = (precision.mean() > self.precision_threshold).to(h.dtype)
                exact = self.exact_memory(h.float())
                h = h + (precision * exact * gate).to(h.dtype)
            if self.training:
                self._precision_mean = precision.mean()

        if _chk(h, 'vpm'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        
        # ─── Spectral (adaptive: diff modulates frequency shaping) ───
        # fp32-якорь: DCT-базис даёт -inf в fp16 под AMP
        # U3: τ-spectral Chebyshev damping: damp = cos(π·τ_norm/2)
        with torch.autocast(device_type=h.device.type, enabled=False):
            h_dct = h.float() @ self.V_dct.T
            if self._tau_norm is not None:
                _cheb_damp = math.cos(math.pi * self._tau_norm / 2.0)
            else:
                _cheb_damp = 1.0
            h_dct = h_dct * self.lambda_k.float() * float(spectral_mod) * _cheb_damp
            h = h + (h_dct @ self.V_dct).to(h.dtype)
        if _chk(h, 'spectral'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        
        # ─── MLP (mirror-conditioned SwiGLU, variant A) ───
        # mirror_gate = mlp_mod (из зеркала) управляет воротами SwiGLU.
        # Старый пост-множитель h_mlp *= mlp_mod убран (двойное гейтирование).
        h_mlp = self.mlp(h, mirror_gate=mlp_mod)
        self._cache_mlp_out = h_mlp  # raw MLP output (gradalign target source)
        if _chk(h_mlp, 'mlp_out'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        with torch.no_grad():
            _mrms = torch.norm(h_mlp.detach().reshape(-1)).float()
            self._mlp_now_ema.mul_(0.99).add_(_mrms, alpha=0.01)
            self._mlp_base_ema.mul_(0.999).add_(_mrms, alpha=0.001)
            self._mlp_ratio = float((self._mlp_now_ema / (self._mlp_base_ema + 1e-12)).item())
        h = h + h_mlp
        if _chk(h, 'post_mlp'): return h * NaN, (_nan_mem, _nan_mem, _nan_conv, None, None)
        
        return h, (mem_state_out, mu_state_out, conv_state_out, traj_state_out, pen)
    
    @property
    def base_parameters(self) -> List[nn.Parameter]:
        """All params except mirror: pre_ln, conv, bind, VSA, spectral, MLP."""
        return [p for n, p in self.named_parameters() if not n.startswith('mirror.')]
    
    @property
    def mirror_parameters(self) -> List[nn.Parameter]:
        """All params inside GroupedCognitiveMirror."""
        return [p for n, p in self.named_parameters() if n.startswith('mirror.')]


# ─── WideBind Stack ────────────────────────────────────────────────────
