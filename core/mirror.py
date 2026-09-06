"""EVA: mirror module."""

from __future__ import annotations

import math, os
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import EVAConfig
from .vsa_utils import fib_sigmoid_init, dct_basis
from .adaptive_gate import AdaptiveGate

class BridgeGLU(nn.Module):
    """GLU-style gating tooling that lives INSIDE the mirror/bridge.

    Takes the semantic delta (B,L,G,k) in K-space and produces a per-expert gate
    (B,L,G) in (0,1):  glu = sigmoid(Wg·delta) * sigmoid(Wv·delta).
    Replaces the frozen mod_scale_mlp scale, so the MLP gate becomes a live
    function of the bridge's semantic state instead of a stuck parameter.
    """
    def __init__(self, G: int, k: int) -> None:
        super().__init__()
        self.Wg: nn.Linear = nn.Linear(G * k, G)
        self.Wv: nn.Linear = nn.Linear(G * k, G)
        # Learnable gain: init ~2.7 so initial gate ≈ sigmoid(log2)=0.667
        # (matches the frozen mod_scale_mlp capacity). Prevents BridgeGLU from
        # starving the MLP on tasks that need it; stays learnable.
        self.log_gain: nn.Parameter = nn.Parameter(torch.tensor(math.log(2.7)))

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        B, L, G, k = delta.shape
        flat = delta.reshape(B, L, G * k)                    # (B, L, G*k)
        g = torch.sigmoid(self.Wg(flat))                     # (B, L, G)
        v = torch.sigmoid(self.Wv(flat))                     # (B, L, G)
        return torch.sigmoid(self.log_gain) * g * v          # (B, L, G)


class GroupedCognitiveMirror(nn.Module):
    """
    Ансамбль из 32 экспертов-зеркал, каждый в своём d=128 подпространстве (D=4096).
    Для D=896: d=28, d/k=3.5:1. Для D=4096: d=128, d/k=4:1.
    
    Каждый эксперт:
      - Имеет свой K-space (k=32) внутри своего d (128 для D=4096, 28 для D=896)
      - Вычисляет 4 сигнала коррекции: temp, pred, smooth, sym
      - Все 4 сигнала суммируются по всем k размерностям (без lo/hi split)
      - Даёт полный градиент pred_error всем k размерностям
      - Имеет свой tanh_bias + skip_connection + log_scale
      - Имеет meta-gate: учится доверять/игнорировать эксперта
    
    Predictive mirror:
      - alpha: scalar per expert, pred_k = alpha_g * hp_{t-1}
      - Вместо W_pred (G×k×k = 16K params) для сильного градиента
      - pred_error = hp_t - pred(hp_{t-1}) — ошибка предсказания
      - Обучает зеркало динамике VSA-состояния
    
    K-space Merge:
      - Все 4 сигнала (temp, pred, smooth, sym) суммируются по всем k размерностям
      - Без lo/hi split: полный градиент для всех сигналов
      - Замена W_pred на alpha + полный k-градиент = W_pred наконец учится
    
    Gradient-Adaptive Gate:
      - delta_var: running EMA variance дельты K-space
      - Эксперт с высокой variance активен, с низкой — прижат
      - Дополняет внешний grad_norm сигнал внутренней метрикой
    
    Внешний сигнал подкрепления:
      - prev_grad_norm: норма градиента по подпространству (c предыдущего backward)
      - Устанавливается извне через cache_grad_norms(grad_h) после backward
    
    Skip connection (alpha=0.1):
      - mirror = tanh(linear + bias) + alpha * linear
      - Обеспечивает per-dim градиент для log_scale даже при насыщении tanh
    """
    def __init__(self, D: int, G: int = 32, k: int = 32, log_scale_init_std: float = 0.05,
                 delta_var_ema_min: float = 0.8, delta_var_ema_max: float = 0.99, tie_mirror_proj: bool = False,
                 layer_idx: int = 0, n_layers: int = 32, has_private_mem: bool = False,
                 expert_asymmetry: bool = False, meta_trust: bool = False,
                 gate_bias_scale: float = 0.0, alpha_novelty_weight: float = 0.0, seq_len: int = 256,
                    intent_bridge: bool = False, bridge_glu: bool = False,
                    bridge_glu_beta: float = 0.25,
                    pm_write_delay: int = 5000, pm_coh_gate_std: float = 0.02,
                     matur_write_thr: float = 0.3,
                     mirror_tau_min: float = 2.0, mirror_tau_max: float = 200.0,
                     tau_config: Optional[object] = None) -> None:
        super().__init__()
        assert D % G == 0
        self.D: int = D
        self.G: int = G
        self.k: int = k
        self.d: int = D // G
        self.seq_len: int = seq_len
        self.tie_mirror_proj: bool = tie_mirror_proj
        self.bridge_glu: bool = bridge_glu
        # φ — единая когнитивная координата глубины (логарифмическая)
        phi: float = math.log(1 + layer_idx) / math.log(max(n_layers, 2))
        self.register_buffer('phi', torch.tensor(phi))
        
        proj_std: float = 1.0 / (self.d * k) ** 0.25
        
        if expert_asymmetry:
            W_p = torch.empty(G, self.d, k)
            for g in range(G):
                nn.init.orthogonal_(W_p[g])
            self.W_proj = nn.Parameter(W_p)
        else:
            self.W_proj = nn.Parameter(torch.randn(G, self.d, k) * proj_std)
        if tie_mirror_proj:
            self.register_buffer('W_out', torch.zeros(G, k, self.d))
            with torch.no_grad():
                self.W_out.copy_(self.W_proj.permute(0, 2, 1))
            self._hook = self.register_forward_pre_hook(
                lambda mod, args: mod._sync_W_out())
        else:
            self.W_out = nn.Parameter(torch.randn(G, k, self.d) * proj_std)

        self.w_temp = nn.Parameter(torch.randn(G, k))
        self.w_global = nn.Parameter(torch.randn(G, k))
        
        # Depthwise conv per group in K-space (CAUSAL: only past tokens)
        self.conv_smooth: nn.Conv1d = nn.Conv1d(G * k, G * k, 3, padding=0,
                                      groups=G * k, bias=False)
        with torch.no_grad():
            self.conv_smooth.weight.zero_()
            self.conv_smooth.weight[:, :, 1] = 1.0  # all channels get center dirac (x_{t-1})
        
        self.w_sym_u = nn.Parameter(torch.randn(G, k))
        self.w_sym_v = nn.Parameter(torch.randn(G, k))
        
        # Predictive mirror: per-dim alpha per expert
        # pred_k = alpha_kg * hp_prev, per K-dimension timescale
        # Tau hierarchy: K-dimensions span exponential range [mirror_tau_min, mirror_tau_max]
        #   τ_k = mirror_tau_min * (mirror_tau_max/mirror_tau_min)^(k/(K-1))
        #   α_k = exp(-1/τ_k)
        # Each expert inherits the same tau distribution (learnable divergence)
        tau_min: float = mirror_tau_min
        tau_max: float = mirror_tau_max
        if k > 1:
            frac = torch.arange(k, dtype=torch.float32) / (k - 1)
            tau_k = tau_min * (tau_max / tau_min) ** frac
        else:
            tau_k = torch.tensor([(tau_min + tau_max) / 2])
        alpha_init: torch.Tensor = torch.exp(-1.0 / tau_k).view(1, k).expand(G, -1).clone()
        if expert_asymmetry and G > 1:
            for g in range(G):
                init_alpha = 0.85 + (g / (G - 1)) * 0.14
                alpha_init[g] = init_alpha
        self.alpha_diag = nn.Parameter(alpha_init)
        self.tanh_bias = nn.Parameter(torch.zeros(G, k))
        # EMA norms for signal normalization (Proposal V-1)
        n_signals: int = 5 if has_private_mem else 4
        self.register_buffer('_signal_norm_ema', torch.ones(n_signals, G, k), persistent=False)
        if expert_asymmetry and G > 1:
            ls_vals = [math.log(0.05 * (1.5 ** g)) for g in range(G)]
            ls_base: torch.Tensor = torch.tensor(ls_vals).unsqueeze(1).expand(G, self.d)
        else:
            ls_base: torch.Tensor = torch.linspace(-0.3, 0.3, G).unsqueeze(1).expand(G, self.d)
        self.log_scale = nn.Parameter(ls_base + torch.randn(G, self.d) * log_scale_init_std)
        
        # ─── K-space gate (per-token, per-expert from hp) ───
        # w_gate: (G, k) — maps |pred_error| to gate logit per expert
        gate_std: float = 1.0 / (self.k + 1) ** 0.5
        self.w_gate = nn.Parameter(torch.randn(G, self.k) * gate_std)
        self.b_gate = nn.Parameter(torch.zeros(G))
        # w_delta_gate: (G, k) — maps delta (correction) to gate logit
        self.w_delta_gate = nn.Parameter(torch.randn(G, self.k) / math.sqrt(self.k))
        gate_bias_val: torch.Tensor = torch.linspace(-gate_bias_scale, gate_bias_scale, G)
        self.gate_bias = nn.Parameter(gate_bias_val)
        self._alpha_novelty_weight = alpha_novelty_weight

        # ─── Intent Bridge: эксперты «ловят» восходящий intent-сигнал ───
        # Zero-init → вклад 0 при init → чекпоинт грузится без изменений
        # (доучивание, не с нуля). intent_state проецируется через W_proj и
        # добавляется в gate_logits, модулируя открытость экспертов.
        self._intent_bridge = intent_bridge
        if intent_bridge:
            self.w_intent = nn.Parameter(torch.zeros(G, k))
            self.b_intent = nn.Parameter(torch.zeros(G))
            # Per-expert salience->gate bias (word importance -> expert openness).
            self.w_sal = nn.Parameter(torch.zeros(G))  # zero-init => checkpoint-safe

        # External gradient cache (устанавливается hook'ом после backward)
        self.register_buffer('_prev_grad_norm', torch.zeros(G), persistent=True)
        self._expert_asymmetry = expert_asymmetry
        self._meta_trust = meta_trust
        # Private memory bank: expert confident K-space states (cross-expert recall)
        self._has_private_mem = has_private_mem
        # Minimum training forward steps before private memory writes activate (safety
        # floor). Coherence gate (below) lets writes start earlier once states are
        # no longer random. 5000 ≈ 640K tokens at accum=8 (main) or 640K at accum=1 (Mini).
        # G=32 needs more delay than G=8 due to higher echo chamber risk.
        self._pm_write_delay = pm_write_delay
        self._pm_coh_gate_std = pm_coh_gate_std
        self._matur_write_thr = float(matur_write_thr)
        # 1.0 once the semantic gate is alive (mod_std >= thr): states left the random
        # regime, so private-memory writes are safe (no echo chamber). Updated at end of
        # forward; read by the write gate at the next forward.
        self.register_buffer('_pm_coh', torch.tensor(0.0), persistent=False)
        if has_private_mem:
            self.register_buffer('_private_mem', torch.randn(G, self.k) * 0.01)
            self.register_buffer('_pm_step', torch.zeros(1, dtype=torch.long))
            # w_help init = log(3.0) -> sigmoid ~0.75: strong initial presence, prevents cold-start suppression
            self.w_help = nn.Parameter(torch.full((G, 1), math.log(3.0)))  # per-expert scale for recalled help
            self.w_contra = nn.Parameter(torch.full((G,), 0.01))  # small positive: disagreement opens gate by default
            # Expert knowledge graph: concept similarity, behavior divergence, trust
            self.register_buffer('_concept_sim_ema', torch.eye(G), persistent=True)   # (G, G) — who shares concepts
            self.register_buffer('_behavior_div_ema', torch.zeros(G, G), persistent=True)  # (G, G) — who behaves differently
            self.register_buffer('_trust_matrix', torch.eye(G) * 0.5, persistent=False)     # (G, G) — who helps whom
            if meta_trust:
                self.register_buffer('_prev_trust_matrix', torch.eye(G) * 0.5, persistent=False)
                self.register_buffer('_meta_private_mem', torch.zeros(G), persistent=False)
        self.register_buffer('_hp_grad', torch.zeros(G), persistent=False)
        self.register_buffer('_delta_var', torch.zeros(G), persistent=True)
        self.register_buffer('_fwd_count', torch.zeros(1, dtype=torch.long), persistent=True)
        self.register_buffer('_last_magnitude', torch.zeros(1), persistent=False)
        self.register_buffer('_last_gates', torch.zeros(G), persistent=False)
        self.register_buffer('_last_h_pool', torch.zeros(G, self.d), persistent=False)
        # Eval-only pred caches (fixed shapes; dynamic B/L written via slices).
        # Training keeps None-able plain attributes (see forward).
        seq_max: int = self.seq_len
        self.register_buffer('_cached_hp_buf', torch.zeros(1, seq_max, G, self.k), persistent=False)
        self.register_buffer('_cached_pred_k_buf', torch.zeros(1, seq_max, G, self.k), persistent=False)
        self.register_buffer('_cached_pred_error_norm_buf', torch.zeros(1, seq_max), persistent=False)
        _pos_g: torch.Generator = torch.Generator().manual_seed(12345)
        self.register_buffer('_pos_id_buf', torch.sign(torch.randn(1, 4096, 1, k, generator=_pos_g)), persistent=False)
        # Gate EMA: gradual wakeup for mirror, cold-start at zero (self-adaptive per-expert warmup)
        self.register_buffer('_gate_ema', torch.zeros(G), persistent=True)
        # Alpha override: set to 0.5 during warmup to force large pred_error
        # 0.0 = use learned alpha; >0 = override alpha for all experts
        self.register_buffer('_alpha_override', torch.zeros(1), persistent=False)
        # Cache for alpha auxiliary loss
        self._cached_pred_k = None
        self._cached_hp = None
        self._cached_pred_error_norm = None
        self._cached_contra = None
        self._cached_disagreement = None
        self._cached_contra_graph = None
        self._cached_contra_expert = None
        self._cached_concept_dendrogram = None
        self._cached_dominance = None
        self._cached_isolation = None
        # Residual variance EMA for adaptive tau (self-organizing timescales)
        self.register_buffer('_residual_var_ema', torch.ones(G, k) * 0.1, persistent=True)
        
        # ─── Per-expert learned modulation ───
        # skip_alpha: uniform init 0 → exp(0)=1 (tanh(linear)+linear).
        # Старый градиент L0≈17 / L31≈0.10 УБРАН — это корень коллапса
        # эффективной глубины (L0 доминирует, глубина «спит»). Параметр
        # оставлен (checkpoint-safe), но без межслойного перекоса.
        self.log_skip_alpha = nn.Parameter(torch.zeros(G))
        # ─── SMF: Sequential-Modulated Fusion gate (scoped to mirror) ───
        # Два пути: P = h (последовательный, RMS=1), T = mirror (параллельный).
        # α = σ(einsum(cat([h̄, T̄]), W_α) + b_α) ∈ [0,1]^(B,L,G);
        # hℓ₊₁ = hℓ + α·Tℓ.  b_α=0 → uniform 0.5 (без L0-перекоса).
        self.w_alpha = nn.Parameter(torch.zeros(self.G, 2 * self.d))
        self.b_alpha = nn.Parameter(torch.zeros(self.G))
        # mod_scale: L0≈-0.81, L31≈-2.30 (геометрия init; rho=0.6**layer_idx)
        rho: float = 0.6 ** layer_idx
        log_mod_init: float = -2.30 + (-0.81 - (-2.30)) * rho
        self.log_dvar_mod_scale = nn.Parameter(torch.full((G,), log_mod_init))
        self.log_grad_mod_scale = nn.Parameter(torch.full((G,), log_mod_init))
        self.dvar_mod_bias = nn.Parameter(torch.full((G,), -0.01))
        self.grad_mod_bias = nn.Parameter(torch.full((G,), -0.01))
        self._delta_var_ema_min = delta_var_ema_min
        self._delta_var_ema_max = delta_var_ema_max

        # ─── Learnable signal weights (sigmoid + Fibonacci self-organization) ───
        self._signal_log_weights = nn.Parameter(fib_sigmoid_init(n_signals))
        # U5: τ-scheduled signal temperature
        self._tau_signal_log = nn.Parameter(torch.tensor(0.0))  # τ_signal base (learnable)
        # Store layer-specific τ_norm for signal temperature scheduling
        self._tau_norm_layer = None
        if tau_config is not None and hasattr(tau_config, 'tau_norm'):
            with torch.no_grad():
                tau_norm_val = tau_config.tau_norm[layer_idx]
                self._tau_norm_layer = tau_norm_val.item()

        
        # ─── Self-organizing usefulness predictor (competitive) ───
        # Каждый эксперт предсказывает свою полезность по delta (K-space correction)
        # Softmax по G: эксперты конкурируют за право модулировать слой.
        # Только лучшие эксперты для данного токена получают высокий вес.
        # init: без Sigmoid — raw logits для softmax-конкуренции
        self.usefulness_predictor = nn.Sequential(
            nn.Linear(k, k),
            nn.Tanh(),
            nn.Linear(k, 1),
        )
        # Per-expert масштабы модуляции (learned log-scale)
        self.mod_scale_mlp = nn.Parameter(torch.full((G,), math.log(2.0)))
        self.mod_scale_mem = nn.Parameter(torch.full((G,), math.log(2.0)))
        # Hybrid gate: sigmoid+softmax replaces frozen mod_scale_mlp baseline.
        # base = sigmoid(usefulness_logits) * (1 + softmax(usefulness_logits/tau))
        # per-expert, input-dependent, no hardcoded 0.750.
        self.hybrid_gate = AdaptiveGate(G, tau_init=1.0)
        # BridgeGLU: GLU gating tooling inside the mirror (experimental; replaces
        # the mod_scale_mlp scale with a live gate when enabled).
        self.bridge_glu_net = BridgeGLU(self.G, self.k) if bridge_glu else None
        # BridgeGLU modulation strength around the FROZEN stable baseline. The
        # learnable gate must MODULATE the proven-stable sigmoid(mod_scale_mlp)
        # anchor, not replace it — otherwise the stable ~0.667 MLP floor vanishes.
        # Centre-zero so init mlp_mod ≈ baseline; live semantic gate modulates ±beta around it.
        self.bridge_glu_beta = bridge_glu_beta
        # Softmax temperature: >1 = softer (uniform), <1 = sharper (winner-take-all)
        self.register_buffer('_usefulness_temp', torch.tensor(2.0), persistent=False)
        # Error-gated damping: порог резонансного демпфирования α на инференсе
        self.register_buffer('_damp_tau', torch.tensor(0.1), persistent=False)
    
    def _sync_W_out(self) -> None:
        with torch.no_grad():
            self.W_out.copy_(self.W_proj.permute(0, 2, 1))
    
    def forward(self, h: torch.Tensor, mem_all: torch.Tensor,
                global_state: Optional[torch.Tensor] = None,
                diff: Optional[torch.Tensor] = None,
                tanh_bias_mod: float = 1.0,
                pred_scale_mod: Optional[torch.Tensor] = None,
                context_mem: Optional[torch.Tensor] = None,
                allow_write: Optional[bool] = None,
                step: Optional[int] = None,
                intent: Optional[torch.Tensor] = None,
                salience: Optional[torch.Tensor] = None,
                maturity: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = h.shape
        G, d, k = self.G, self.d, self.k
        
        # Split into subspaces
        h_g = h.reshape(B, L, G, d)           # (B, L, G, d)
        mem_g = mem_all.reshape(B, L, G, d)
        mc_g = mem_g.mean(dim=1, keepdim=True)  # (B, 1, G, d)
        
        # Project each group to its K-space
        hp = torch.einsum('blgd,gdk->blgk', h_g, self.W_proj)    # (B, L, G, k)
        # Hook to capture gradient for grad_mod (only during training)
        if hp.requires_grad:
            hp.register_hook(lambda g: (
                self._prev_grad_norm.copy_(g.detach().norm(dim=-1).mean(dim=(0, 1))),
                None
            )[1])
        mc_k = torch.einsum('b l gd,gdk->b l gk', mc_g, self.W_proj)

        # ─── Bipolar pos_id binding: hp = hp ⊛ pos_id ───
        # Детерминированный буфер позиционных кодов (broadcast на B, G).
        hp = hp * self._pos_id_buf[:, :L]
        
        # hp_prev shared by sym_k and pred_error
        hp_prev = torch.cat([torch.zeros_like(hp[:, 0:1]), hp[:, :-1]], dim=1)
        
        # ─── Slow signals (lo half of K-space) ───
        # Temporal: deviation from memory centroid
        temp_k = (hp - mc_k) * self.w_temp  # (B, L, G, k)
        
        # Global: deviation from cross-layer state
        if global_state is not None:
            gs_k = torch.einsum('b l gd,gdk->b l gk',
                                global_state.reshape(1, 1, G, d), self.W_proj)
            temp_k = temp_k + (hp - gs_k) * self.w_global
        
        # Predictive: error in K-space self-prediction (t-1 -> t)
        # Per-dim alpha: each K-dimension has its own timescale
        # Alpha override smoothly interpolates: override=1 → identity (α=1),
        # override=0 → learned alpha_diag. Provides smooth warmup transition.
        alpha_eff = self.alpha_diag
        override = self._alpha_override.item() if self.training else 0.0
        if override > 0:
            alpha_eff = (1 - override) * alpha_eff + override * 1.0
        pred_k = hp_prev * alpha_eff.view(1, 1, G, k)  # (B, L, G, k)
        _pred_k_aux = pred_k  # undamped — для aux loss, чтобы damping не боролся с ним
        if pred_scale_mod is None:
            dv = self._delta_var
            dv_mean = dv.mean().clamp(min=1e-8)
            pred_scale_mod = (dv / dv_mean).clamp(0.1, 3.0)
        # Нормализованная ошибка предсказания: relative к ||hp||, а не абсолютная
        hp_norm = hp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        raw_pred_error = hp - pred_k
        # Error-gated damping (только инференс): α → 1 когда ||pred_error|| велика
        # На тренировке (teacher forcing) δ_{t-1}=0, резонанс безвреден.
        if not self.training:
            damp = torch.sigmoid(-raw_pred_error.norm(dim=-1).mean() / self._damp_tau)
            alpha_eff = 1.0 + (alpha_eff - 1.0) * damp
            pred_k = hp_prev * alpha_eff.view(1, 1, G, k)
        # Сигнал нормируется EMA вместе с остальными
        pred_error = (hp - pred_k) / hp_norm * pred_scale_mod.view(G, 1)
        # Adaptive tau: K-измерения с высокой ошибкой → короткое τ, с низкой → длинное.
        # alpha_target = sigmoid(2.2 - log(rel_var)):
        #   rel_var=1 (noise) → α=0.9 (init), rel_var=0.5 → α=0.95, rel_var=2 → α=0.82
        if self.training:
            with torch.no_grad():
                override = self._alpha_override.item()
                if override < 0.1:
                    residual_var = pred_error.var(dim=(0, 1), unbiased=False)
                    self._residual_var_ema.lerp_(residual_var, 0.01)
                    rv = self._residual_var_ema
                    rv_mean = rv.mean(dim=-1, keepdim=True)
                    relative_var = rv / (rv_mean + 1e-10)
                    alpha_target = torch.sigmoid(2.2 - torch.log(relative_var))
                    self.alpha_diag.data.lerp_(alpha_target, 0.01)
                    if self._alpha_novelty_weight > 0 and G > 1:
                        alpha_per_expert = self.alpha_diag.mean(dim=-1)
                        alpha_center = alpha_per_expert - alpha_per_expert.mean()
                        alpha_std = alpha_per_expert.std()
                        boost = max(1.0, 0.1 / (alpha_std + 0.01))
                        adapted_w = self._alpha_novelty_weight * boost
                        novelty_push = (adapted_w * 2
                                        * alpha_center.unsqueeze(1).expand(-1, k) / G)
                        self.alpha_diag.data.add_(novelty_push)
                        self.alpha_diag.data.clamp_(0.01, 0.99)
        pred_error_norm = (raw_pred_error / hp_norm).norm(dim=(-2, -1))  # (B, L)
        if self.training:
            self._cached_pred_k = _pred_k_aux.detach() if _pred_k_aux is not None else None
            self._cached_hp = hp.detach()
            self._cached_pred_error_norm = pred_error_norm.detach()
        else:
            if self._cached_hp_buf.shape[0] != B:
                _seq_max = self._cached_hp_buf.shape[1]
                self._cached_hp_buf = torch.zeros(B, _seq_max, G, self.k, device=hp.device)
                self._cached_pred_k_buf = torch.zeros(B, _seq_max, G, self.k, device=hp.device)
                self._cached_pred_error_norm_buf = torch.zeros(B, _seq_max, device=hp.device)
            self._cached_hp_buf[:, :L].copy_(hp.detach())
            if _pred_k_aux is not None:
                self._cached_pred_k_buf[:, :L].copy_(_pred_k_aux.detach())
            else:
                self._cached_pred_k_buf[:, :L].zero_()
            self._cached_pred_error_norm_buf[:, :L].copy_(pred_error_norm.detach())
            self._cached_pred_error_norm = pred_error_norm.detach()
        
        # ─── Private Memory: read via cross-expert attention (when uncertain) ───
        if self._has_private_mem:
            uncert = torch.sigmoid(pred_error.abs())  # (B, L, G, k)
            q = hp * uncert
            keys = self._private_mem.detach().clone()  # (G, k) — frozen snapshot for autograd
            if context_mem is not None:
                keys = context_mem * 0.3 + keys * 0.7
                keys = F.normalize(keys, dim=-1) * self._private_mem.norm(dim=-1, keepdim=True)
            attn = F.sigmoid(q @ keys.T / math.sqrt(self.k))  # (B, L, G, G) — independent gates
            help_k_base = attn @ keys  # (B, L, G, k) — collective confident memory
            # Contradiction gate: disagreement between expert hp and collective help_k
            hp_n = hp.norm(dim=-1).clamp(min=1e-8)  # (B, L, G)
            disagreement = (hp - help_k_base).norm(dim=-1) / hp_n  # relative: 0=agrees, >>1=contradicts
            contra = torch.sigmoid(disagreement - 1.0)  # sigmoid(rel_disagree - 1): <1=agrees, >1=contradicts
            trust = 1.0 - contra  # how much to trust help_k (low when contradictory)
            # Apply contradiction-aware scaling:
            # high disagreement + confident expert → expert is confidently wrong → reduce help_k
            # high disagreement + uncertain expert → collective irrelevant → reduce help_k
            help_k = help_k_base * torch.sigmoid(self.w_help).unsqueeze(0).unsqueeze(0)
            help_k = help_k * trust.unsqueeze(-1)  # trust-weighted collective memory
            self._cached_contra = contra.detach()  # for analysis
            self._cached_disagreement = disagreement.detach()  # for analysis
        else:
            help_k = torch.zeros_like(hp)
            trust = torch.ones_like(hp.norm(dim=-1))  # no contradiction when disabled
        
        # ─── Expert Knowledge Graph update (uses OLD private_mem, runs before write) ───
        _write = self._has_private_mem and (self.training if allow_write is None else allow_write)
        if _write:
            with torch.no_grad():
                pm = self._private_mem
                pm_norm = pm.norm(dim=-1, keepdim=True).clamp(min=1e-10)
                pm_n = pm / pm_norm  # safe normalize (no NaN on zero vectors)
                concept_sim = pm_n @ pm_n.T
                self._concept_sim_ema.mul_(0.99).add_(concept_sim, alpha=0.01)
                hp_avg = hp.mean(dim=(0, 1))
                hp_n = F.normalize(hp_avg, dim=-1)
                behavior_sim = hp_n @ hp_n.T
                behavior_div = 1.0 - behavior_sim
                self._behavior_div_ema.mul_(0.99).add_(behavior_div, alpha=0.01)
                trust_weights = attn.mean(dim=(0, 1))
                if self._meta_trust and self._has_private_mem:
                    self._prev_trust_matrix.copy_(self._trust_matrix)
                self._trust_matrix.mul_(0.99).add_(trust_weights, alpha=0.01)
                if self._meta_trust and self._has_private_mem:
                    delta_tm = self._trust_matrix - self._prev_trust_matrix
                    instability = delta_tm.abs().mean(dim=1)
                    self._meta_private_mem.mul_(0.9).add_(instability, alpha=0.1)
                contra_g = concept_sim * behavior_div
                self._cached_contra_graph = contra_g
                contra_expert = contra_g.mean(dim=-1)
                self._cached_contra_expert = contra_expert
                sim_vals = concept_sim[~torch.eye(self.G, dtype=torch.bool, device=concept_sim.device)]
                q_hi = sim_vals.float().quantile(0.75)
                q_lo = sim_vals.float().quantile(0.25)
                self._cached_concept_dendrogram = (q_hi.item(), q_lo.item())
                dominance = self._trust_matrix.sum(dim=0)
                isolation = 1.0 - (self._trust_matrix.sum(dim=-1) / self.G)
                self._cached_dominance = dominance
                self._cached_isolation = isolation
        
        # ─── Private Memory: write confident K-space states (contradiction-aware) ───
        # ИНТЕЛЛЕКТУАЛЬНЫЙ (когнитивный) контроллер записи:
        #   • При включённой maturation запись открывается ТОЛЬКО когда ЕДИНЫЙ
        #     показатель зрелости M_l(t) = max(time/τ-рампа, bridge_readiness)
        #     пересекает matur_write_thr. M_l уже несёт в себе safety time-floor
        #     (память откроется к ~22k–41k шагу даже если bridge никогда не станет
        #     компетентным) И реальную компетентность bridge. Никакого слепого
        #     переопределения шаговым таймером -> ранне-случайное эхо-камера
        #     невозможно, а компетентный bridge подключает память ровно когда это
        #     полезно. pm_write_delay в этом режиме игнорируется.
        #   • При ВЫКЛЮЧЕННОЙ maturation (legacy) — явный step-delay floor ИЛИ
        #     crutch по std модуляции (исходное поведение). pm_write_delay<=0 тогда
        #     означает «только по когерентности» (без шагового пола).
        _write_ok = _write
        _write_scale: torch.Tensor = torch.tensor(1.0, device=hp.device)
        if _write_ok:
            if maturity is not None:
                # Smooth scaling: sigmoid transition around threshold
                _write_scale = torch.sigmoid((maturity - self._matur_write_thr) * 10.0)
                _write_ok = bool(_write_scale > 0.01)
            else:
                self._pm_step += 1
                coherent = bool(self._pm_coh.item())
                _write_ok = coherent if self._pm_write_delay <= 0 \
                    else (self._pm_step.item() >= self._pm_write_delay) or coherent
        if _write_ok:
            with torch.no_grad():
                conf = torch.sigmoid(-pred_error.abs().mean(dim=-1, keepdim=True))
                contra_u = contra.unsqueeze(-1)
                contra_expert_coll = self._cached_contra_expert.to(conf.device).view(1, 1, G, 1)
                isolation_coll = self._cached_isolation.to(conf.device).view(1, 1, G, 1)
                social_pressure = 1.0 - 0.5 * torch.sigmoid(contra_expert_coll.clamp(min=0) + isolation_coll)
                conf_plastic = conf * (1.0 - contra_u) * social_pressure
                # Soft competition: temperature prevents winner-take-all monoculture
                temp_write = 0.5  # <1 softens competition (true soft), >1 sharpens
                conf_soft = conf_plastic ** temp_write
                conf_bc = conf_soft * self.G / (conf_soft.sum(dim=-2, keepdim=True) + 1e-8)
                weighted_hp = (conf_bc * hp.detach()).mean(dim=(0, 1))
                # Adaptive decay: fast warmup when memory is nascent, slow when stable
                pm_scale = self._private_mem.norm(dim=-1).mean().clamp(min=1e-8)
                warmup_rate = torch.sigmoid(3.0 - pm_scale)  # ~1.0 when pm~0, ~0.0 when pm>3
                pm_decay = 0.999 - 0.009 * warmup_rate  # [0.990, 0.999] — faster decay when memory is empty
                self._private_mem.mul_(pm_decay).add_(weighted_hp * _write_scale, alpha=1.0 - pm_decay)
                self._private_mem.clamp_(-10.0, 10.0)
        
        # ─── Fast signals (hi half of K-space) ───
        # Smoothness: local coherence in K-space (CAUSAL: pad left only)
        hp_perm = hp.permute(0, 2, 3, 1).reshape(B, G * k, L)  # (B, G*k, L)
        hp_pad = F.pad(hp_perm, (2, 0))  # pad 2 zeros on left, 0 on right
        hp_smooth = self.conv_smooth(hp_pad)  # (B, G*k, L) — kernel sees t-2, t-1, t
        hp_smooth = hp_smooth.reshape(B, G, k, L).permute(0, 3, 1, 2)  # (B, L, G, k)
        smooth_k = hp - hp_smooth
        
        # Symmetry: bilinear temporal interaction
        sym_k = (hp * self.w_sym_u) * (hp_prev * self.w_sym_v)
        
        # ─── EMA-нормировка сигналов (соизмеримость перед softmax) ───
        if self._has_private_mem:
            signals = [temp_k, pred_error, smooth_k, sym_k, help_k]
        else:
            signals = [temp_k, pred_error, smooth_k, sym_k]
        signals_normed = []
        decay = 0.01  # ~100-step EMA
        for i, s in enumerate(signals):
            if self.training:
                with torch.no_grad():
                    rms = s.norm(dim=(-2, -1), keepdim=True).mean(dim=(0, 1), keepdim=True)
                    self._signal_norm_ema[i].mul_(1 - decay).add_(rms.squeeze(), alpha=decay)
            s_norm = s / (self._signal_norm_ema[i].unsqueeze(0).unsqueeze(0) + 1e-8)
            signals_normed.append(s_norm)
        
        # ─── Learnable signal weights (sigmoid + Fibonacci self-organization) ───
        # U5: τ-scheduled signal temperature
        if self._tau_norm_layer is not None:
            tau_min_gate = 0.3
            tau_max_gate = 5.0
            tau_norm = self._tau_norm_layer
            tau_signal = tau_min_gate * (tau_max_gate / tau_min_gate) ** (1 - tau_norm)
            tau_signal = max(tau_signal, 0.01)
            w = torch.sigmoid(self._signal_log_weights / tau_signal)
        else:
            w = torch.sigmoid(self._signal_log_weights)  # (n_sig,), no sum-to-1 constraint
        
        # ─── Decorrelation: orthogonalize WEIGHTED signals (gradient flows to _signal_log_weights) ───
        n_sig = len(signals)
        decorr = 0.0
        npairs = 0
        for i in range(n_sig):
            for j in range(i + 1, n_sig):
                si = (signals_normed[i] * w[i]).reshape(-1, signals_normed[i].shape[-2] * signals_normed[i].shape[-1])
                sj = (signals_normed[j] * w[j]).reshape(-1, signals_normed[j].shape[-2] * signals_normed[j].shape[-1])
                si_c = si - si.mean(dim=0, keepdim=True)
                sj_c = sj - sj.mean(dim=0, keepdim=True)
                cos_ij = (si_c * sj_c).sum(dim=-1) / (si_c.norm(dim=-1) * sj_c.norm(dim=-1) + 1e-8)
                decorr = decorr + cos_ij.pow(2).mean()
                npairs = npairs + 1
        if npairs > 0:
            decorr = decorr / npairs
        self._cached_decorr = decorr
        
        # ─── Merge all signals (weighted sum) ───
        delta = sum(w[i] * signals_normed[i] for i in range(n_sig))
        
        delta = delta * torch.rsqrt(delta.pow(2).mean(dim=-1, keepdim=True) + 1e-7)  # rms_norm over k
        delta = delta + self.tanh_bias * tanh_bias_mod
        
        # ─── Gate modulation signals (shared between gate & usefulness) ───
        grad_mod = torch.exp(self.log_grad_mod_scale) * torch.tanh(self._prev_grad_norm + self.grad_mod_bias)
        if self.training:
            with torch.no_grad():
                dvar = delta.var(dim=(0, 1), unbiased=False).mean(dim=-1)  # (G,)
                if diff is not None:
                    ema_alpha = self._delta_var_ema_min + diff * (self._delta_var_ema_max - self._delta_var_ema_min)
                else:
                    ema_alpha = 0.9
                self._delta_var.mul_(ema_alpha).add_(dvar * (1.0 - ema_alpha))
        dvar_mod = torch.exp(self.log_dvar_mod_scale) * torch.tanh(self._delta_var + self.dvar_mod_bias)
        
        usefulness_logits = self.usefulness_predictor(delta).squeeze(-1)
        if self.training:
            self._fwd_count.add_(1)
            n_eff = self._fwd_count
        else:
            # Deterministic in eval: temperature depends only on `step` (graph input),
            # not on the number of calls (would break export determinism).
            n_eff = step if step is not None else 0
        n_eff = torch.as_tensor(n_eff, dtype=torch.float32, device=h.device)
        prog = 1.0 - torch.exp(-n_eff / 200.0)
        temp = torch.clamp(3.0 * torch.exp(-prog * 2.0), min=0.3, max=3.0)
        with torch.no_grad():
            srt_u, _ = torch.sort(usefulness_logits, dim=-1)
            n_u = srt_u.shape[-1]
            if n_u % 2 == 1:
                threshold = srt_u[..., n_u // 2:n_u // 2 + 1]
            else:
                threshold = (srt_u[..., n_u // 2 - 1:n_u // 2]
                             + srt_u[..., n_u // 2:n_u // 2 + 1]) / 2
        usefulness = torch.sigmoid((usefulness_logits - threshold) / temp)


        
        # Per-expert modulation strengths (gated by self-assessment)
        if self.bridge_glu_net is not None:
            glu = self.bridge_glu_net(delta)                       # (B, L, G) — BridgeGLU: live semantic gate, in (0,1)
            # Hybrid gate: sigmoid+softmax replaces frozen mod_scale_mlp baseline.
            # base = sigmoid(usefulness_logits) * (1 + softmax(usefulness_logits/tau))
            # per-expert, input-dependent, no hardcoded 0.750.
            base = self.hybrid_gate(usefulness_logits)             # (B, L, G)
            # Live semantic modulation ANCHORED to the stable baseline (multiplicative,
            # centre-zero): mlp_mod = base * (1 + beta*(2*glu - 1)). At init glu~0.25 ->
            # factor ~0.875 -> mlp_mod ≈ 0.58*usefulness; as glu learns to centre (→0.5)
            # it converges to the proven-stable baseline while staying input/semantic-live.
            live = 2.0 * glu - 1.0
            if maturity is not None:
                # Maturation gate: live modulation is SCALED by layer maturity, so
                # untrained experts cannot perturb the trunk at step 0 (rho(J_l) ~ 1).
                live = live * maturity
            mlp_mod = base * (1.0 + self.bridge_glu_beta * live)
        else:
            mlp_mod = self.hybrid_gate(usefulness_logits)         # (B, L, G)
        mem_mod = usefulness * torch.sigmoid(self.mod_scale_mem).view(1, 1, G)
        self._last_mlp_mod = mlp_mod.detach()                      # for diagnostics (gate spread / aliveness)

        # coherence tracker for the adaptive private-memory write gate (see forward write block)
        if self._has_private_mem:
            with torch.no_grad():
                if maturity is not None:
                    # Unified maturation gate drives the write gate (no hard std crutch).
                    self._pm_coh.fill_(float(maturity))
                else:
                    _ms = mlp_mod.detach().float().std().item()
                    self._pm_coh.fill_(1.0 if _ms >= self._pm_coh_gate_std else 0.0)
        
        # Linear projection + skip connection
        linear = torch.einsum('blgk,gkd->blgd', delta, self.W_out)  # (B, L, G, d)
        skip_alpha = torch.exp(self.log_skip_alpha).view(1, 1, G, 1)
        mirror = torch.tanh(linear) + skip_alpha * linear
        # Adaptive scale: prevents saturation when delta is large
        delta_norm = delta.norm(dim=-1).mean(dim=(0, 1)).detach().clamp(min=1e-8)  # (G,)
        adapt_scale = (1.0 / (1.0 + 0.1 * delta_norm)).view(1, 1, -1, 1)  # (1, 1, G, 1)
        mirror = mirror * torch.exp(self.log_scale) * adapt_scale

        # ─── SMF gate (scoped to mirror only) ───
        # Убирает доминирование L0: α зависит от h (последовательный путь)
        # и от mirror (параллельный путь); deep-слои могут «открыться».
        _hg = h.reshape(B, L, G, d)
        _hn = torch.nn.functional.normalize(_hg, dim=-1)
        _mn = torch.nn.functional.normalize(mirror, dim=-1)
        _gin = torch.cat([_hn, _mn], dim=-1)              # (B, L, G, 2d)
        alpha = torch.sigmoid(torch.einsum('blgk,gk->blg', _gin, self.w_alpha) + self.b_alpha)
        self._cached_alpha = alpha.detach()
        mirror = mirror * alpha.unsqueeze(-1)

        # ─── K-Space Gate (per-token, per-expert) ───
        gate_signal = torch.abs(pred_error)  # (B, L, G, k)
        gate_logits = torch.einsum('blgk,gk->blg', gate_signal, self.w_gate) + self.b_gate
        gate_logits = gate_logits + self.gate_bias.unsqueeze(0).unsqueeze(0)
        # Delta signal: how much correction is mirror applying (complements |pred_err|)
        delta_gate = torch.einsum('blgk,gk->blg', delta, self.w_delta_gate)
        gate_logits = gate_logits + delta_gate
        gate_logits = gate_logits + grad_mod.unsqueeze(0).unsqueeze(0)
        gate_logits = gate_logits + dvar_mod.unsqueeze(0).unsqueeze(0)
        # Intent Bridge: нисходящий intent модулирует открытость экспертов.
        # w_intent/b_intent zero-init → без эффекта при init (checkpoint-safe).
        if intent is not None and self._intent_bridge:
            # intent is the per-head stream already in (…,G,k) expert space.
            ik = intent
            intent_gate = torch.einsum('blgk,gk->blg', hp - ik, self.w_intent) + self.b_intent
            gate_logits = gate_logits + intent_gate
        # Salience (word importance from output) -> per-expert gate bias.
        if salience is not None and self._intent_bridge:
            Lg = gate_logits.shape[1]
            if salience.shape[1] != Lg:
                # In recurrent/autoregressive generation `_last_salience` carries the
                # previous step's sequence length (S), which differs from the current
                # forward length (L). Align it: keep the last L positions, or broadcast
                # the (single/newest) salience across positions. No-op in training (S==L).
                if salience.dim() == 2:
                    salience = salience.unsqueeze(-1)
                if salience.shape[1] >= Lg:
                    salience = salience[:, -Lg:, :]
                else:
                    reps = Lg - salience.shape[1]
                    last = salience[:, -1:, :].expand(-1, reps, -1)
                    salience = torch.cat([salience, last], dim=1)
            gate_logits = gate_logits + salience * self.w_sal.view(1, 1, -1)
        # Contradiction signal: expert vs collective disagreement opens gate (arbiter)
        if self._has_private_mem:
            gate_logits = gate_logits + disagreement * self.w_contra.unsqueeze(0).unsqueeze(0)
            # Concept graph pressure: high contradiction → open gate more
            if self._cached_contra_expert is not None:
                ce = self._cached_contra_expert.to(gate_logits.device).unsqueeze(0).unsqueeze(0)
                gate_logits = gate_logits + ce  # collective contradiction raises gate
            # Soft routing overlay: specialization + consensus modulate gate via w_contra
            spec = self._behavior_div_ema.mean(dim=-1)            # (G,) — avg divergence per expert
            spec = spec / (spec.max() + 1e-10)                    # norm to [0, 1]
            cons = self._concept_sim_ema.mean(dim=-1)             # (G,) — avg similarity per expert
            cons = cons / (cons.max() + 1e-10)                    # norm to [0, 1]
            gate_bonus = (spec * 0.5 + cons * 0.5) * self.w_contra * 0.1
            gate_logits = gate_logits + gate_bonus.unsqueeze(0).unsqueeze(0)
        if self._meta_trust and self._has_private_mem:
            p = self._meta_private_mem.unsqueeze(0).unsqueeze(0)
            gate_logits = gate_logits - 0.5 * p
        if self.training:
            ls = self.log_scale
            ls_dev = ls.mean(dim=-1) - ls.mean()
            ls_var = ls.var().item()
            if ls_var < 0.05:
                boost = 1.0 * torch.sigmoid(3.0 * ls_dev)
                gate_logits = gate_logits + boost.unsqueeze(0).unsqueeze(0)
        
        expert_gate = torch.sigmoid(gate_logits)  # (B, L, G)
        # Cache gate L1 for auxiliary sparsity loss (still in graph for gradients)
        self._cached_gate_l1 = expert_gate.mean()
        # Cache per-expert mean gate for load balancing loss
        self._cached_gate_usage = expert_gate.mean(dim=(0, 1))  # (G,)
        # Gate EMA: self-adaptive per-expert warmup for mirror (cold → full over ~5000 steps)
        if self.training:
            self._gate_ema.mul_(0.99).add_(self._cached_gate_usage.detach(), alpha=0.01)
        # Cache for expert reinforcement loss (gate vs usefulness alignment)
        self._cached_usefulness = usefulness
        self._cached_gate = expert_gate.detach()
        
        mirror = mirror * expert_gate.unsqueeze(-1)
        mirror = mirror.reshape(B, L, D)
        
        with torch.no_grad():
            self._last_magnitude.fill_(mirror.abs().mean())
            self._last_gates.copy_(expert_gate.detach().mean(dim=(0, 1)))
            self._last_h_pool.copy_(h_g.detach().mean(dim=(0, 1)))
        
        return mirror, mlp_mod, mem_mod, hp, pred_error_norm
    
    def cache_grad_norms(self, grad_h: Optional[torch.Tensor] = None) -> None:
        """Call after backward: store per-subspace gradient norm.
        Uses hp hook by default; falls back to explicit grad_h if provided."""
        if grad_h is not None:
            with torch.no_grad():
                g_norms = grad_h.reshape(-1, self.G, self.d).norm(dim=-1).mean(dim=0)
                self._prev_grad_norm.copy_(g_norms)
        else:
            self._prev_grad_norm.copy_(self._hp_grad)

    @torch.no_grad()
    def debug_mind(self) -> Dict[str, object]:
        """Return a dict of meta-cognitive stats for generation interpretability."""
        info: Dict[str, object] = {}
        if not self._has_private_mem:
            return info
        info['private_mem_norm'] = self._private_mem.norm(dim=-1).mean().item()
        info['w_help'] = torch.sigmoid(self.w_help).mean().item()
        info['w_contra'] = self.w_contra.mean().item()
        # U5: τ-scheduled signal temperature (consistent with forward)
        if self._tau_norm_layer is not None:
            tau_min_gate = 0.3
            tau_max_gate = 5.0
            tau_norm = self._tau_norm_layer
            tau_signal = tau_min_gate * (tau_max_gate / tau_min_gate) ** (1 - tau_norm)
            tau_signal = max(tau_signal, 0.01)
            w = torch.sigmoid(self._signal_log_weights / tau_signal)
        else:
            w = torch.sigmoid(self._signal_log_weights)
        w_norm = w / (w.sum() + 1e-10)
        for i, label in enumerate(['temp','pred','smooth','sym','help'][:len(w)]):
            info[f'signal_w_{label}'] = w_norm[i].item()
        if self._cached_contra_expert is not None:
            info['contra_expert'] = self._cached_contra_expert.mean().item()
            info['contra_expert_raw'] = self._cached_contra_expert.tolist()
        if self._cached_contra_graph is not None:
            info['contra_graph_mean'] = self._cached_contra_graph.mean().item()
        if self._cached_dominance is not None:
            info['dominance'] = self._cached_dominance.tolist()
        if self._cached_isolation is not None:
            info['isolation'] = self._cached_isolation.tolist()
        if self._cached_concept_dendrogram is not None:
            info['concept_q_hi'], info['concept_q_lo'] = self._cached_concept_dendrogram
        info['gate_ema'] = self._gate_ema.tolist()
        info['gate_ema_mean'] = self._gate_ema.mean().item()
        info['gate_selectivity'] = self._last_gates.std().item()
        info['ls_var'] = self.log_scale.var(dim=-1).tolist()
        info['ls_var_mean'] = self.log_scale.var(dim=-1).mean().item()
        if self._has_private_mem:
            tm = self._trust_matrix
            info['trust_max'] = tm.max().item()
            info['trust_min'] = tm[tm > 0].min().item() if (tm > 0).any() else 0.0
            info['trust_diag'] = tm.diag().mean().item()
            info['behavior_div'] = self._behavior_div_ema.mean(dim=-1).tolist()
            info['concept_sim'] = self._concept_sim_ema.mean(dim=-1).tolist()
            bd = self._behavior_div_ema.mean(dim=-1)
            cs = self._concept_sim_ema.mean(dim=-1)
            tr = self._trust_matrix.mean(dim=-1)
            info['spec_index'] = (self._last_gates * bd).tolist()
            info['cons_index'] = (cs * tr).tolist()
        return info

    def meta_signals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cheap GPU-friendly meta-cognitive signals (no per-call host sync).

        Returns (trust_max, gate_ema_mean) as 0-dim tensors on the module device.
        Used in the generation hot loop so all 24 layers can be aggregated with a
        single .item() instead of debug_mind() (which syncs ~10x per layer and
        stalls the GPU under autograd-less generation).
        """
        if not self._has_private_mem or self._trust_matrix is None:
            dev = next(self.parameters()).device
            return torch.zeros((), device=dev), torch.full((), 0.5, device=dev)
        return self._trust_matrix.max(), self._gate_ema.mean()

