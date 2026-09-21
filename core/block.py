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
# F2 (math audit): the tail-referenced fp32 scan is safe only while
# CHUNK*|floor_log| < ln(FLT_MAX) = 88.7; the block clamps its floor to this.
_SCAN_LOG_MAX = 88.7


def _stream_cap(x, cap):
    """M50/M51: scale-invariant magnitude cap (the 2970 explosion fuse).
    Values above `cap` are rescaled to it with the direction preserved;
    at the cap the Jacobian is O(1) — unlike the 1/|h| vanishing that made
    a 1e16 stream unrecoverable. Used on the residual stream between blocks
    (stack.py, M50) and on every branch injection (block.py, M51).
    cap <= 0 disables."""
    if cap <= 0.0:
        return x
    m = x.abs().amax(dim=-1, keepdim=True)
    return x * (cap / m.clamp_min(cap))

def pen_decay_factor(pen: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Prediction-error modulation of the decay gate, CENTERED at 1.0 (audit
    M3): the old form 1 − 0.5·σ(pen+w) applied ≈0.75 to EVERY channel already
    at zero prediction error (σ(0)=0.5), silently shrinking the whole τ ladder
    ~3-5x at init. Now: 1 at pen=0, monotonically down to the 0.5 asymptote
    as pen grows; w_d_pen stays the learnable sensitivity/offset.
    """
    return 1.0 - (torch.sigmoid(pen + w) - torch.sigmoid(w))


def _scan_chunk(b_chunk: torch.Tensor, d_chunk: torch.Tensor, floor_log=None,
                use_fp64: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parallel chunk scan from zero state.
    Returns intra-chunk VSA (B, chunk_len, S*D), final state (B, 1, S*D),
    cumulative decay (B, chunk_len, S*D).

    M26 (replaces M20/M25): the floored path is TAIL-REFERENCED — algebraically
    the same scan, rewritten as

        intra_t = (cd_t/cd_last) * cumsum_i( b_i * cd_last/cd_i )

    so EVERY weighted input is bounded by |b| (the exponents of the weighted
    inputs are <=0): no reciprocal of a tiny cum_decay ever materializes,
    forward AND backward are fp32-stable at any ladder floor, and no fp64 graph
    is needed anywhere (the M20 measurement: naive fp64 scans pinned ~8.5GB on
    the A100 — the B19 incident). NOTE (F2, math audit): the PREFACTOR
    e^{A_t−A_last} >= 1 grows with the chunk range (it is not <=0); finiteness
    requires CHUNK*|floor_log| < 88.7, enforced by the floor clamp in
    EVABlock.forward (_SCAN_LOG_MAX).
    The old naive form overflowed in the backward at the production fast-floor
    values (1/cd ~ 1e25-1e27): the step-0 NaN-gradient source that poisoned the
    run (live logs 2026-09-13: 192 non-finite grads at step 0 despite M25).

    floor_log=None keeps the legacy UNBOUNDED fp64 reciprocal path bit-for-bit
    (M3 exactness lock vs vsa_utils.vsa_prefix_scan / test_scan_exactness).
    """
    if floor_log is not None:
        # M26: tail-referenced, caller dtype (fp32). floor_log itself is tiny.
        log_a = torch.log(d_chunk.clamp(min=_EPS_SCAN))
        log_a = log_a.clamp_min(floor_log.to(log_a.dtype))
        log_cum = torch.cumsum(log_a, dim=1)
        anchor = log_cum[:, -1:]                                  # per-column tail
        u = b_chunk * torch.exp(anchor - log_cum)                 # |u| <= |b|
        intra = torch.exp(log_cum - anchor) * torch.cumsum(u, dim=1)
        cum_decay = torch.exp(log_cum)                            # may reach ~1e-26; fp32-safe
        return intra.to(b_chunk.dtype), intra[:, -1:], cum_decay.to(b_chunk.dtype)
    # legacy unbounded path (M3 exactness lock): fp64 reciprocal form
    log_a = torch.log(d_chunk.double().clamp(min=_EPS_SCAN))
    log_cum = torch.cumsum(log_a, dim=1)
    cum_decay = torch.exp(log_cum)
    weighted = b_chunk.double() / cum_decay
    cum_w = torch.cumsum(weighted, dim=1)
    intra_d = cum_decay * cum_w
    intra = intra_d.to(b_chunk.dtype)
    final = intra[:, -1:]
    return intra, final, cum_decay.to(b_chunk.dtype)

def _scan_chunks(b_in: torch.Tensor, d_in: torch.Tensor, floor_log=None,
                 chunk: int = 32):
    """M37: VECTORIZED chunked scan — mathematically identical to calling
    _scan_chunk per chunk in a python loop, but one batched graph instead of
    ~768 python iterations per step (16 chunks x 2 scans x 24 layers). That
    loop was the A100 launch-bound ceiling (121 tok/s at B=2) AND inflated
    the backward memory (every chunk kept its own intermediates; the vector
    keeps one (B,L,S,D) set — peak 35.3GB gets real again).

    Returns (intra (B,L,S,D), final (B,n_chunks,S,D), cum_decay (B,L,S,D)).
    Tail-pads with b=0 / decay=1 — neither moves the real positions' prefix
    sums, and the padded chunk's anchor equals its real tail exactly.
    floor_log=None falls back to the python loop (the M3 fp64 exactness path
    is rarely used — production always floors — and its oracle stays tested).
    """
    if floor_log is None:
        B, L = b_in.shape[0], b_in.shape[1]
        outs = [_scan_chunk(b_in[:, s:min(s + chunk, L)],
                            d_in[:, s:min(s + chunk, L)])
                for s in range(0, L, chunk)]
        intra = torch.cat([o[0] for o in outs], dim=1)
        final = torch.cat([o[1] for o in outs], dim=1)      # (B,nc,S,D)
        cumd = torch.cat([o[2] for o in outs], dim=1)
        return intra, final, cumd
    B, L, S, D = b_in.shape
    nc = (L + chunk - 1) // chunk
    pad = nc * chunk - L
    # M38: fold the chunk axis INTO the batch axis and run the cumsum along
    # dim=1 — the exact contiguous layout the single-chunk kernel was tuned
    # for. (M37 kept (B,nc,32,S,D) and cumsummed along dim=2: a strided axis,
    # measured SLOWER than the loop it replaced — 66 tok/s vs 106 on the A100.
    # Vectorization is only a win when the memory layout matches the kernel.)
    if pad:
        b4 = F.pad(b_in, (0, 0, 0, 0, 0, pad)).reshape(B * nc, chunk, S, D)
        d4 = F.pad(d_in, (0, 0, 0, 0, 0, pad), value=1.0).reshape(B * nc, chunk, S, D)
    else:
        b4 = b_in.reshape(B * nc, chunk, S, D)
        d4 = d_in.reshape(B * nc, chunk, S, D)
    log_a = torch.log(d4.clamp(min=_EPS_SCAN)).clamp_min(floor_log.to(d4.dtype))
    log_cum = torch.cumsum(log_a, dim=1)                      # within-chunk
    anchor = log_cum[:, -1:]                                  # (B*nc,1,S,D)
    u = b4 * torch.exp(anchor - log_cum)                      # |u| <= |b|
    intra = torch.exp(log_cum - anchor) * torch.cumsum(u, dim=1)
    cum_decay = torch.exp(log_cum)
    flat = lambda t: t.reshape(B, nc * chunk, S, D)[:, :L]
    return (flat(intra).to(b_in.dtype),
            intra[:, -1].reshape(B, nc, S, D),
            flat(cum_decay).to(b_in.dtype))


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
        # M51: per-branch injection cap — no single branch (conv/bind/
        # mirror/VPM/spectral/MLP) can dump more than this into the stream
        # per layer; healthy branches measure O(1)–O(1e3), so 1e4 is a
        # no-op until something runs away.
        self.branch_cap: float = float(getattr(cfg, 'branch_cap', 1e4))
        # T9.9 шаг 2 (опции, default off): boundary-aware VSA (мягкий сброс на
        # SEP) и conv-стена (обнуление входа conv на границах).
        self._vsa_bound_reset: float = float(getattr(cfg, 'vsa_boundary_reset', 0.0) or 0.0)
        self._conv_wall: bool = bool(getattr(cfg, 'conv_boundary_wall', False))
        # Store τ_norm for this layer (U1, U3). __init__ value is only the
        # fallback; forward refreshes it from the LIVE τ-field (audit M7:
        # _tau_dev trains during the run, a snapshot froze U3/U10/ψ at their
        # init values while the τ-ladder moved on).
        self._tau_norm: Optional[float] = None
        if tau_config is not None and hasattr(tau_config, 'tau_norm'):
            with torch.no_grad():
                self._tau_norm = tau_config.tau_norm[layer_idx].item()
                self._tau_norm_t = tau_config.tau_norm[layer_idx].detach()
        # Keep the τ-field so the mirror can bind its gate authorities to τ
        # (intent_alpha etc.); previously the mirror always saw tau_config=None,
        # which silently disabled all τ-ties inside GroupedCognitiveMirror.
        # T9-ROOT-FIX: НЕ регистрировать tau_config как подмодуль блока!
        # При обычной присваивании nn.Module регистрировал общий τ-модуль в
        # КАЖДОМ блоке ⇒ его параметры попадали в layer.parameters() и в
        # state_dict (layers.N.tau_config.*), и set_active_depth(k) вызывал
        # requires_grad_(False) на _tau_dev (последний слой k..n−1) — τ-лестница
        # замерла (g_tau_dev=0, _tau_dev=0.0000 в чекпойнтах). object.__setattr__
        # оставляет это обычной ссылкой: владелец модуля — стек.
        object.__setattr__(self, 'tau_config', tau_config)
        
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
        # U10: set τ_norm on bind module for frequency schedule (refreshed
        # LIVE in forward — audit M7)
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
            seq_len=cfg.seq_len,
            intent_bridge=getattr(cfg, 'intent_bridge', False),
            bridge_glu=getattr(cfg, 'bridge_glu', False),
            bridge_glu_beta=getattr(cfg, 'bridge_glu_beta', 0.25),
            pm_write_delay=getattr(cfg, 'pm_write_delay', 0),
            pm_coh_gate_std=getattr(cfg, 'pm_coh_gate_std', 0.02),
            mirror_tau_min=getattr(cfg, 'mirror_tau_min', 2.0),
            mirror_tau_max=getattr(cfg, 'mirror_tau_max', 200.0),
            tau_config=tau_config)
        
        # ─── VSA Memory (multi-scale VSA: S=4 фиксированных τ) ───
        self._n_scales = 4
        self._vsa_floor_k = float(getattr(cfg, 'vsa_decay_floor_k', 2.0))  # B18
        self._scan_floor_bound = False   # P0-1 (F2): sticky "tau_s clamped" flag
        self.register_buffer('_pen_ema', torch.zeros(()), persistent=True)  # B18b
        # U1: τ-consistent VSA scales. This copy is the TRAINABLE ladder for
        # standalone blocks (tau_s=None); inside EVAStack the live source is
        # the stack-level _vsa_log_param (tau_s is always passed), and this
        # copy is EXCLUDED from both optimizer builders so it stops being
        # zero-gradient weight in the production optimizer state (audit M7).
        # T8: значения — единый источник core.tau_api.VSA_LADDER.
        from . import tau_api as _tau_api
        self._vsa_tau_log = nn.Parameter(
            torch.tensor([math.log(x) for x in _tau_api.VSA_LADDER]))
        # Keep old buffer for backward compat (unused in forward when tau_config provided)
        tau_s = torch.tensor(_tau_api.VSA_LADDER, dtype=torch.float32)
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
        # sigmoid bias of the content decay modulation decay=exp(−1/τ_s)·σ(h·w_d+b_d)
        # (L0: σ≈0.88, L23: σ≈0.993 — the τ≈exp(b_d) reading was fictional, audit M3)
        b_d_init = 2.0 + 3.0 * layer_frac
        self.b_i = nn.Parameter(torch.full((cfg.D,), -2.5))   # i_gate ~0.08 init
        self.b_d = nn.Parameter(torch.full((cfg.D,), b_d_init))
        # Surprisal-gated write coefficient γ_l: растёт с τ. Derived from the
        # REAL τ-ladder position (dev=0 → geometric interpolation τ_min..τ_max)
        # around its geometric center τ_mid=√(τ_min·τ_max) — replaces the
        # fictional τ=e^{b_d} and the magic ln 32 (audit M3).
        _tau_min = float(getattr(cfg, 'tau_min', 8.0))
        _tau_max = float(getattr(cfg, 'tau_max', 512.0))
        _tau_l = _tau_min * (_tau_max / _tau_min) ** layer_frac
        _tau_mid = math.sqrt(_tau_min * _tau_max)
        gamma_max = 0.5
        gamma_init = gamma_max / (1.0 + math.exp(-(math.log(_tau_l) - math.log(_tau_mid))))
        self.gamma_surprisal = nn.Parameter(torch.full((), gamma_init))
        # Когерентность спиралей → запись в VSA-память (опорные точки скрещивания фаз)
        self.bind_coh_gate = nn.Parameter(torch.tensor(0.5))

        # First moment
        self.w_k_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_q_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_mu_mem = nn.Parameter(torch.randn(cfg.D))
        
        # ─── Conv ───
        # Audit M11: causality is implemented EXPLICITLY by prepending the
        # streamed conv_state (kernel−1 zeros at stream start) — the module
        # must not add its own padding on top (padding=47 + manual cat(47)
        # double-shifted the window to h[t−94..t−47]: blind to the last 47
        # tokens and STRICTLY zero for sequences shorter than 94 — exactly the
        # zero-grad the dead-parameter sweep reported). Mirror.conv_smooth
        # (padding=0 + F.pad) already uses this correct pattern.
        self.conv = nn.Conv1d(cfg.D, cfg.D, kernel_size=cfg.conv_kernel,
                              padding=0, groups=cfg.D, bias=False)
        self._conv_pad: int = cfg.conv_kernel - 1
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        
        # ─── Spectral (self-organizing frequency filters) ───
        self.register_buffer('V_dct', dct_basis(cfg.D))
        # ─── MLP-output runaway tracker (self-referencing; cadence 0.99/0.999,
        # НЕ τ-linked — T8: горизонт статистики, зарегистрирован в test_tau_lint) ───
        # fast (0.99) vs slow (0.999) EMA of ‖h_mlp‖. Healthy ratio ≈ 1; a
        # ×2–50 runaway lifts it within a few steps, unlike an absolute bound
        # which the runaway's growing variance absorbs.
        self.register_buffer('_mlp_now_ema', torch.ones(1), persistent=False)
        self.register_buffer('_mlp_base_ema', torch.ones(1), persistent=False)
        self.register_buffer('_mlp_cnt', torch.zeros(1), persistent=False)
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

        # ─── T9: Covariance Memory (порт EVA-Ai/FCP; default off) ───
        # Создаётся ПОСЛЕДНЕЙ: RNG-поток всех остальных параметров не сдвигается,
        # A/B cov_memory=False vs True с одним сидом сравнимы (ревью R3).
        # W_out zero-init (rank>0: W_out_b) ⇒ на старте бит-в-бит residual.
        # τ обновляется в forward из живого τ_l слоя (M7-refresh).
        self.cov_memory = None
        if getattr(cfg, 'cov_memory', False):
            from .cov_memory import CovarianceMemory
            # RNG-изоляция (ревью R3): параметры ветви берутся из «теневого»
            # потока, глобальный поток не сдвигается ⇒ модули, созданные ПОСЛЕ
            # блоков (банк/голова/концепты), инициализируются как без ветви —
            # A/B cov_memory=False vs True с одним сидом сравним.
            _rng_cpu = torch.get_rng_state()
            self.cov_memory = CovarianceMemory(
                D=cfg.D,
                n_heads=int(getattr(cfg, 'cov_memory_heads', 4)),
                head_dim=int(getattr(cfg, 'cov_memory_head_dim', 32)),
                tau=64.0,
                chunk=int(getattr(cfg, 'cov_memory_chunk', 64)),
                rank=int(getattr(cfg, 'cov_memory_rank', 128)))
            torch.set_rng_state(_rng_cpu)
        # T9 read-usage телеметрия: ‖cov_y‖/‖h‖ (только training; заполняется
        # в forward, агрегируется training_telemetry) — «градиент ≠ вклад» (T2).
        self._cov_y_norm: Optional[torch.Tensor] = None
        self._cov_h_norm: Optional[torch.Tensor] = None
    
    def forward(self, h: torch.Tensor, state: Optional[Tuple] = None, global_state: Optional[torch.Tensor] = None,
                mem2v_scale: float = 1.0, diff: Optional[torch.Tensor] = None, noise_scale: float = 0.0,
                tanh_bias_mod: float = 1.0, pred_scale_mod: Optional[torch.Tensor] = None, spectral_mod: float = 1.0,
                context_mem: Optional[torch.Tensor] = None, allow_write: Optional[bool] = None, tau_s: Optional[torch.Tensor] = None, step: Optional[int] = None, intent: Optional[torch.Tensor] = None,
                salience: Optional[torch.Tensor] = None, maturity: Optional[torch.Tensor] = None,
                sep_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple]:
        mem_state = mu_state = conv_state = traj_state = pen = cov_state = None
        cov_state_out = None
        if state is not None:
            mem_state, mu_state, conv_state = state[:3]
            if len(state) > 3:
                traj_state = state[3]
            if len(state) > 4:
                pen = state[4]
            if len(state) > 5:
                cov_state = state[5]
        B, L, D = h.shape
        NaN = float('nan')
        self._nan_at = None
        # M7 (live τ-field): the __init__ τ_norm / intent_alpha were one-time
        # snapshots of a ladder that _tau_dev keeps moving. Refresh the block's
        # own U3 damping, the bind U10 frequency schedule, and the mirror's
        # gate-amplitude authority (intent_alpha) + τ-signal ladder every
        # forward so those τ-ties actually track the field (README §7/§8).
        if self.tau_config is not None:
            with torch.no_grad():
                _tn = float(self.tau_config.tau_norm[self.layer_idx].detach())
                _ia = float(self.tau_config.intent_alpha[self.layer_idx].detach())
            self._tau_norm = _tn
            if hasattr(self.bind, '_tau_norm'):
                self.bind._tau_norm = _tn
            if self.cov_memory is not None:
                # T9: τ ковариационной памяти = живой τ_l слоя (единый язык,
                # как VSA/spectral). Флор — канонический TAU_MIN (tau_api),
                # clamp здесь явный (конструкторный clamp живую запись не ловит).
                from .tau_api import TAU_MIN as _TAU_MIN
                self.cov_memory.tau = max(
                    float(self.tau_config.tau_l[self.layer_idx].detach()),
                    float(_TAU_MIN))
            mir = getattr(self, 'mirror', None)
            if mir is not None:
                mir._intent_alpha = _ia
                mir._tau_norm_layer = _tn
        def _chk(t, label):
            if not self.training:
                return False
            if t.is_floating_point() and (t.isnan().any() or t.isinf().any()):
                self._nan_at = f'L{self.layer_idx}.{label}[{t.min():.2f},{t.max():.2f}]'
                return True
            return False

        def _nan_ret(hh):
            # T9 (ревью R1/R2): при включённой ветви NaN-пути несут cov-состояние
            # (не теряют его молча); при выключенной — прежний 5-кортеж.
            _t = (_nan_mem, _nan_mem, _nan_conv, None, None)
            if self.cov_memory is not None:
                _t = _t + (cov_state_out,)
            return hh * NaN, _t

        def _ln(x):
            # M50: overflow-safe (amax-rescale) — x^2 overflows fp32 at 1e19
            _m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            _x = x / _m
            return self.pre_ln_w * _x * torch.rsqrt(_x.pow(2).mean(dim=-1, keepdim=True) + 1e-7)

        device = h.device
        K = self.K
        S = self._n_scales
        
        # Consistent NaN state shapes (prevent ndim mismatch in next step)
        _nan_conv = torch.zeros(B, D, self._conv_pad, device=device) * NaN
        _nan_mem = torch.zeros(B, S * D, device=device) * NaN
        
        # Transfer stale mirror cache (with shape & dtype check)
        if pen is None:
            pen = getattr(self.mirror, '_cached_pred_error_norm', None)
        if pen is not None and (pen.shape[-1] != L or pen.shape[0] != B):
            pen = None
        
        # ─── Pre-LN (M31: per-branch renormalized residual cascade) ───
        # The 1045-step depth audit measured the old shape — ONE LN at the block
        # input, branches accumulating onto the normalized copy, output
        # LN(h)+Σ(branches) — as a 2.3x/layer backward decay (CE-grad L0 3.2e-8
        # vs L23 2.6; the shallow half frozen: Adam SNR ~1e-10). Two partial
        # fixes failed on the real checkpoint: end-compensation (±h_n cancel
        # zeroed the direct conv path) and straight-through LN (restores the
        # identity but exposes the raw cascade gain ~50x/sublayer -> inf).
        # Correct shape (classic pre-LN, adapted to this intra-block cascade):
        # the stream h is the TRUE residual accumulator; every branch reads
        # LN(current stream) via _ln(). Backward: h_out = h_in + Σ f_s(LN(·)) with
        # identity coefficient 1 per sub-layer; each branch is separately fed a
        # unit-scale input, so no cascade can explode and no chain can decay to
        # zero. Forward VALUES differ from the old block (per-branch re-norm) —
        # fresh run required; the norm is scale-invariant so branch DIRECTION
        # semantics are preserved.
        
        # ─── Conv ───
        if conv_state is None:
            conv_state = torch.zeros(B, D, self._conv_pad, device=device, dtype=h.dtype)
        h_perm = _ln(h).transpose(1, 2)
        # T9.9 шаг 2 (опция): conv-стена — обнуление входа conv на границах
        # предложений (SEP). Дешёвый барьер; полная per-tap маска — позже.
        if self._conv_wall and sep_mask is not None:
            _w = (1.0 - sep_mask.to(h_perm.dtype)).view(B, 1, L)
            h_perm = h_perm * _w
        _full = torch.cat([conv_state, h_perm], dim=-1)
        if _full.shape[-1] < self._conv_pad + L:      # B1: defensive left-pad
            _full = F.pad(_full, (self._conv_pad + L - _full.shape[-1], 0))
        h_conv = self.conv(_full)[..., -L:].transpose(1, 2)       # B1: [-L:] == [:L] when aligned
        conv_state_out = h_perm[:, :, -self._conv_pad:]
        if conv_state_out.shape[-1] < self._conv_pad:              # B1: fixed-width carry
            conv_state_out = F.pad(conv_state_out, (self._conv_pad - conv_state_out.shape[-1], 0))
        h = h + _stream_cap(h_conv, self.branch_cap)
        if _chk(h, 'conv'): return _nan_ret(h)
        if self.training:
            self._cache_conv_out = h_conv  # for branch_loss (with grad)
        
        if isinstance(self.bind, TrajectorySpiralBind):
            # B10 (audit 02b F2B-01): TRAINING never reads the detached cache.
            # Reading it shadowed B2's own fix: from window 2 every document ran
            # on window 1's FROZEN content (measured cross-position Jacobian
            # exactly 0.0 in steady training), and any eval forward (traj_state
            # None, no training guard on the write) overwrote the cache with
            # zeros permanently. The cache is now purely a STREAMING carry: read
            # only when not training, written only by training or an explicit
            # stream session — eval neither reads nor writes it.
            if traj_state is None and not self.training:
                traj_state = getattr(self, '_traj_state', None)
            if traj_state is not None and (traj_state.shape[2] != L
                                           or traj_state.shape[0] != B):
                traj_state = None
            bind_out, new_traj, coherence = self.bind(_ln(h), traj_state)
            if traj_state is None:
                if self.training or getattr(self, '_stream_mode', False):
                    self._traj_state = new_traj.detach()
                traj_state_out = None
            else:
                traj_state_out = (0.9 * traj_state + 0.1 * new_traj).detach()
                if getattr(self, '_stream_mode', False):
                    self._traj_state = traj_state_out
        else:
            bind_out = self.bind(_ln(h))
            coherence = torch.zeros(B, L, K, device=device, dtype=h.dtype)
            new_traj = None
            traj_state_out = None
        if _chk(bind_out, 'bind'): return _nan_ret(h)
        
        # ─── VSA Memory (multi-scale: S=4 фиксированных τ) ───
        S = self._n_scales
        tau_s = torch.exp(self._vsa_tau_log) if tau_s is None else tau_s
        # P0-1 (F2, math audit): the tail-referenced fp32 scan is finite only
        # while CHUNK*|floor_log| < ln(FLT_MAX)=88.7, floor_log = -k/tau_s
        # => tau_s > 32k/88.7 ≈ 0.36k. The clamp lives HERE (the single tau_s
        # entry point) so the SEMANTIC time constant is bounded, not just the
        # exponent range; the ladder leaving the safe zone is telemetered.
        if self._vsa_floor_k > 0:
            _tau_safe = 0.5 * self._vsa_floor_k          # k=2 => tau_s >= 1.0
            _bound = bool((tau_s < _tau_safe).any())
            tau_s = tau_s.clamp(min=_tau_safe)
            if self.training and _bound:
                self._scan_floor_bound = True
        d_s = torch.exp(-1.0 / tau_s.to(device))  # (S,) — τ-scales from learnable param
        # Surprisal-gated write: i_gate = softplus(linear + γ·||ê||₂)
        h_v = _ln(h)
        igate_logit = h_v * self.w_i + self.b_i
        if pen is not None:
            igate_logit = igate_logit + self.gamma_surprisal * pen.unsqueeze(-1)
        i_gate = F.softplus(igate_logit)                    # (B, L, D)
        # Опорные точки скрещивания спиралей: синхронность фаз усиливает запись
        coh_mean = coherence.mean(dim=-1, keepdim=True)     # (B, L, 1)
        i_gate = i_gate * (1.0 + self.bind_coh_gate * coh_mean.to(i_gate.dtype))
        # M11: centered (rest = 1.0 = exact ladder; content only shortens).
        # F2B-02 note (audit 02b): the sigmoid(b_d)~0.88 rest floor does compress
        # the tau ladder (measured tau_eff 7.5/31/135 vs nominal). CENTERING d_mod
        # to 1.0 at rest is NOT the fix: it removes the decay floor, memory stops
        # decaying across a 512-window, downstream norm saturates and
        # layer_bridge_gate gradients vanish (caught by the dead-parameter
        # detector in B11; the LBG itself was removed in M64.5). Widening the
        # ladder needs a BOUNDED rest (<1) with
        # (1-a) write-normalization — a B12 design decision, not a silent edit.
        # B18 (audit 02b closure, corrected): the content gate is REST-NORMALIZED,
        # sigma(h.w_d+b_d)/sigma(b_d) — at rest == 1.0, so the tau ladder IS the
        # nominal schedule (the old absolute sigma ~0.88-0.99 multiplied every
        # token and silently divided tau by ~100x across a 512 window). Content
        # can only SHORTEN memory (clamp<=1); the outer floor d_s^k (below)
        # guards the extreme end, never the rest regime. Consequence: the
        # AdaptiveController's b_d->8 lerp becomes benign (modulation -> 1.0).
        d_mod = (torch.sigmoid(h_v * self.w_d + self.b_d)
                 / torch.sigmoid(self.b_d).clamp(min=1e-3)).clamp(max=1.0)  # (B, L, D)
        if noise_scale > 0 and self.training:
            noise = 1.0 + noise_scale * torch.randn_like(i_gate)
            i_gate = i_gate * noise

        # Prediction-error-aware decay modulation (before decay expansion).
        # Centered: pen=0 → factor 1.0 (memory untouched), pen↑ → toward 0.5.
        if pen is not None and getattr(self, '_pen_decay_on', True):
            # P3-1 (regulator ledger): _pen_decay_on=False is the identity clamp
            # of the surprise-gated decay (default True = the old path verbatim).
            # B18b: pen enters as a DEVIATION from its running EMA baseline —
            # at typical surprise the factor is exactly 1.0 (the old absolute
            # form sat at ~0.91 at rest and multiplied away another decade of
            # tau; the dead-parameter detector caught it). w_d_pen keeps a
            # live gradient as the sensitivity around the baseline.
            with torch.no_grad():
                self._pen_ema.mul_(0.999).add_(pen.detach().mean() * 0.001)
            _pc = pen - self._pen_ema
            d_pen_factor = pen_decay_factor(
                _pc.unsqueeze(-1), self.w_d_pen.unsqueeze(0).unsqueeze(0))
            d_mod = (d_mod.reshape(B, L, self.mirror.G, self.mirror.d)
                     * d_pen_factor.to(d_mod.dtype).unsqueeze(-1)).reshape(B, L, D)

        # Vectorize over S scales: (B, L, S, D) — expand-views, no materialized copies
        d_s_vec = d_s.view(1, 1, S, 1).expand(B, L, S, D)
        d_mod_vec = d_mod.unsqueeze(2).expand(-1, -1, S, -1)
        decay = (d_s_vec * d_mod_vec).clamp(min=0.01, max=1.0)  # per-scale per-channel
        # T9.9 шаг 2 (опция): boundary-aware VSA — на границах предложений
        # затухание усиливается (мягкий сброс памяти на SEP; strength<1 —
        # сохраняет форму скана, в отличие от жёсткого обнуления состояния).
        if self._vsa_bound_reset > 0.0 and sep_mask is not None:
            _bf = (1.0 - self._vsa_bound_reset * sep_mask.to(decay.dtype))
            decay = decay * _bf.view(B, L, 1, 1)
        # B18/B19: the ladder floor (content may shorten a scale toward
        # tau_s/k, k=2 default, 0 disables) is enforced inside _scan_chunk
        # in log space — zero extra graph memory.


        # Dynamic write modulation (per-expert K-space conditioning)
        # Audit M10 (A1): the per-expert write modulation ran ONLY in
        # training, so streaming-inference memory writes followed a different
        # rule than training writes. The stale (previous-step) hp cache is
        # causally legal at eval too — run both paths identically.
        hp_cached = self.mirror._cached_hp
        if (hp_cached is not None
                and hp_cached.shape[0] == B and hp_cached.shape[1] == L):
            g = self.mirror.G
            d = self.mirror.d
            k = self.mirror.k
            BL = B * L
            hp_g = hp_cached.permute(2, 0, 1, 3).reshape(g, BL, k)  # batched matmul (stable under AMP)
            wm = torch.matmul(hp_g, self.w_i_dyn)  # (g, BL, d)
            write_mod = torch.sigmoid(wm.permute(1, 0, 2).view(B, L, g, d) / math.sqrt(k))
            mem_input = (h_v.reshape(B, L, g, d) * write_mod).reshape(B, L, D) * i_gate
        else:
            mem_input = h_v * i_gate  # (B, L, D)

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
        _fl18 = None
        if self._vsa_floor_k > 0:
            _fl18 = (self._vsa_floor_k * torch.log(d_s.clamp(min=_EPS_SCAN).double())).view(1, 1, S, 1)
            # F2 (math audit): the tail-referenced fp32 scan is finite only
            # while CHUNK*|floor_log| < ln(FLT_MAX) = 88.7. tau_s comes from a
            # learnable parameter WITHOUT clamps, so a drift toward fast
            # forgetting (tau_s < 32*k/88.7 ≈ 0.72 at k=2) would push
            # e^{A_t-A_last} past fp32 -> inf*0 = NaN (measured: tau_s=0.3 NaN).
            # Bound the floor itself — the binding constraint on |log_a|; a
            # no-op in the healthy regime (tau_s ~ 8..512).
            _fl18 = _fl18.clamp_min(-_SCAN_LOG_MAX / float(CHUNK))
        # M26: floored scans are tail-referenced fp32 (finite fwd+bwd at any
        # floor); floor-off stays the legacy fp64 exactness path. No per-layer
        # precision decision, no GPU sync, no fp64 graph.

        # M37: one vectorized call replaces the 16-iteration python loop; the
        # per-chunk list _combine_chunks consumes is now cheap VIEWS of one
        # batched graph (was 16 separate subgraphs retained for backward).
        _iv, _fv, _cv = _scan_chunks(input_vec_f32, decay_f32,
                                     floor_log=_fl18, chunk=CHUNK)
        chunks = [(_iv[:, s:min(s + CHUNK, L)], _fv[:, k:k + 1],
                   _cv[:, s:min(s + CHUNK, L)])
                  for k, s in enumerate(range(0, L, CHUNK))]
        
        mem_all_vec, mem_state_out_vec, mem_leaf_vec = _combine_chunks(chunks, mem_state_f32)
        # Keep VSA in fp32 — prefix scan accumulators underflow/overflow in fp16
        
        # Weighted combination: sigmoid per scale per channel (no sum-to-1)
        w = torch.sigmoid(self.scale_w)  # (S, D)
        mem_all = (mem_all_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, L, D)
        mem_leaf = (mem_leaf_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, L, D) — без кросс-чанк контекста
        # Dual read: leaf = within-chunk state, ctx = CROSS-chunk state only.
        # Audit M3: the old form multiplied mem_all by both w_q and w_q_ctx —
        # the two paths were parametrically indistinguishable (only their sum
        # mattered) and 'context read' was a fiction. mem_all − mem_leaf is
        # exactly the carried-in cross-chunk component.
        mem_read = (mem_all * self.w_q + mem_leaf * self.w_q_leaf
                    + (mem_all - mem_leaf) * self.w_q_ctx)
        mem_state_out = mem_state_out_vec.reshape(B, S * D)
        
        # First moment (same multi-scale decay, scaled input)
        if mu_state is not None:
            mu_state = mu_state.reshape(B, S, D)
        mu_input_vec = (mem_input * self.w_k_mu).unsqueeze(2).expand(-1, -1, S, -1)
        mu_input_f32 = mu_input_vec.float() if mu_input_vec.dtype != torch.float32 else mu_input_vec
        _mv, _mfv, _mcv = _scan_chunks(mu_input_f32, decay_f32,
                                        floor_log=_fl18, chunk=CHUNK)
        mu_chunks = [(_mv[:, s:min(s + CHUNK, L)], _mfv[:, k:k + 1],
                      _mcv[:, s:min(s + CHUNK, L)])
                     for k, s in enumerate(range(0, L, CHUNK))]
        mu_all_vec, mu_state_out_vec, _ = _combine_chunks(mu_chunks, mu_state)
        mu_all = (mu_all_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        mu_read = mu_all * self.w_q_mu
        mem_read = mem_read + mu_read * self.w_mu_mem
        mu_state_out = mu_state_out_vec.reshape(B, S * D)
        if _chk(mem_read, 'mem_read'): return _nan_ret(h)

        # ─── T9: Covariance Memory (порт EVA-Ai/FCP; default off) ───
        # Второй момент (парные корреляции) поверх той же τ-лестницы; отдельная
        # residual-ветвь с per-branch pre-LN (M31). fp32-якорь: скан в log-space.
        cov_state_out = None
        if self.cov_memory is not None:
            with torch.autocast(device_type=h.device.type, enabled=False):
                _cov_y, cov_state_out = self.cov_memory(_ln(h).float(), cov_state)
            if self.training:
                # T9 read-usage: ‖cov_y‖/‖h‖ — «градиент ≠ вклад» (T2); ratio
                # агрегируется training_telemetry, falsifier для A/B.
                with torch.no_grad():
                    self._cov_y_norm = _cov_y.detach().float().norm()
                    self._cov_h_norm = h.detach().float().norm()
            h = h + _stream_cap(_cov_y.to(h.dtype), self.branch_cap)
            if _chk(h, 'cov_mem'): return _nan_ret(h)
        
        # ─── Mirror (self-consistency: local + global) ───
        # fp32-якорь: exp/log/softmax в mirror переполняются в fp16 под AMP
        with torch.autocast(device_type=h.device.type, enabled=False):
            _gs = global_state.float() if isinstance(global_state, torch.Tensor) else global_state
            _ctx = context_mem.float() if isinstance(context_mem, torch.Tensor) else context_mem
            mirror, mlp_mod, mem_mod, hp, pred_error_norm = self.mirror(
                _ln(h).float(), mem_all.float(), global_state=_gs, diff=diff,
                tanh_bias_mod=tanh_bias_mod, pred_scale_mod=pred_scale_mod,
                context_mem=_ctx, allow_write=allow_write, step=step, intent=intent,
                salience=salience, maturity=maturity)
            mirror = mirror.to(h.dtype)
            mlp_mod = mlp_mod.to(h.dtype) if isinstance(mlp_mod, torch.Tensor) else mlp_mod
            self._cache_mlp_mod = mlp_mod  # (B,L,G) per-expert MLP gate (gradalign)
            mem_mod = mem_mod.to(h.dtype) if isinstance(mem_mod, torch.Tensor) else mem_mod
        if _chk(mirror, 'mirror'): return _nan_ret(h)
        if _chk(mlp_mod, 'mlp_mod'): return _nan_ret(h)
        if _chk(mem_mod, 'mem_mod'): return _nan_ret(h)
        
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
        enhanced = (_stream_cap(enhanced_base, self.branch_cap)
                    + _stream_cap(mirror, self.branch_cap))
        # Concept layer moved to stack.py (UnifiedConceptLayer — global, after embedding)
        if _chk(enhanced, 'enhanced'): return _nan_ret(h)
        if self.training:
            self._cache_bind_out = enhanced_base  # for branch_loss (with grad)
            self._cache_mirror_out = mirror  # for branch_loss (with grad)
        h = h + enhanced
        if _chk(h, 'post_enhanced'): return _nan_ret(h)

        # ─── Variable Precision Memory ───
        if self.variable_precision:
            # fp32-якорь: softmax в exact_memory переполняется в fp16 под AMP.
            # Гейт встроен тензорной маской (не python-if): статический граф.
            with torch.autocast(device_type=h.device.type, enabled=False):
                precision = self.precision_gate(_ln(h).float())
                hard = (precision.mean() > self.precision_threshold).to(h.dtype)
                # Audit M11: a purely boolean gate DEADLOCKS — the moment the
                # mean drops below the threshold, gradients die for the gate
                # AND exact_memory, so nothing can ever reopen it (observed as
                # permanently-zero grads). Straight-through: forward stays the
                # hard switch; d/d(precision) flows through the soft term, so
                # the opener can always learn to re-engage exact memory.
                # value: hard; grad w.r.t. precision.mean(): 1 (even while 0)
                soft_gate = hard + (precision.mean() - precision.mean().detach())
                exact = self.exact_memory(_ln(h).float())
                h = h + _stream_cap((precision * exact * soft_gate).to(h.dtype),
                                    self.branch_cap)
            if self.training:
                self._precision_mean = precision.mean()

        if _chk(h, 'vpm'): return _nan_ret(h)
        
        # ─── Spectral (adaptive: diff modulates frequency shaping) ───
        # fp32-якорь: DCT-базис даёт -inf в fp16 под AMP
        # U3: τ-spectral Chebyshev damping: damp = cos(π·τ_norm/2)
        with torch.autocast(device_type=h.device.type, enabled=False):
            h_dct = _ln(h).float() @ self.V_dct.T
            if self._tau_norm is not None:
                # P3-1 (regulator ledger): _damp_on=False is the identity clamp
                # of the spectral branch (default True = bit-for-bit the old path).
                _cheb_damp = (math.cos(math.pi * self._tau_norm / 2.0)
                              if getattr(self, '_damp_on', True) else 1.0)
            else:
                _cheb_damp = 1.0
            h_dct = h_dct * self.lambda_k.float() * float(spectral_mod) * _cheb_damp
            h = h + _stream_cap((h_dct @ self.V_dct).to(h.dtype),
                                self.branch_cap)
        if _chk(h, 'spectral'): return _nan_ret(h)
        
        # ─── MLP (mirror-conditioned SwiGLU, variant A) ───
        # mirror_gate = mlp_mod (из зеркала) управляет воротами SwiGLU.
        # Старый пост-множитель h_mlp *= mlp_mod убран (двойное гейтирование).
        h_mlp = self.mlp(_ln(h), mirror_gate=mlp_mod)
        self._cache_mlp_out = h_mlp  # raw MLP output (gradalign target source)
        if self.training and h_mlp.requires_grad:
            # gradalign target = ‖∂CE/∂mlp_out‖ per expert, captured by a
            # backward HOOK during the regular CE pass (audit M5: the loop
            # computed it with an EXTRA torch.autograd.grad over the whole
            # network each step ≈ second backward; the hook gets it for free).
            # One-step-stale by design — same streaming convention as
            # _prev_grad_norm in the mirror.
            def _ga_hook(grad, _blk=self):
                if not getattr(_blk, '_ga_record', True):
                    return        # B2: aux/bypass phases must not overwrite the CE target
                _m = _blk.mirror
                gg = grad.detach().float().reshape(grad.shape[0], grad.shape[1], _m.G, -1)
                _blk._gradalign_tgt = gg.pow(2).sum(dim=(0, 1, 3)).sqrt()
            h_mlp.register_hook(_ga_hook)
        if _chk(h_mlp, 'mlp_out'): return _nan_ret(h)
        with torch.no_grad():
            _mrms = torch.norm(h_mlp.detach().reshape(-1)).float()
            if self._mlp_cnt.item() == 0:
                # cold-start: baseline = first observed level, not the init 1.0
                # (else ratio is inflated while the slow EMA climbs for ~700 steps)
                self._mlp_now_ema.copy_(_mrms)
                self._mlp_base_ema.copy_(_mrms)
            else:
                self._mlp_now_ema.mul_(0.99).add_(_mrms, alpha=0.01)
                self._mlp_base_ema.mul_(0.999).add_(_mrms, alpha=0.001)
            self._mlp_cnt.add_(1)
            self._mlp_ratio = float((self._mlp_now_ema / (self._mlp_base_ema + 1e-12)).item())
        h = h + _stream_cap(h_mlp, self.branch_cap)
        if _chk(h, 'post_mlp'): return _nan_ret(h)

        _s_out = (mem_state_out, mu_state_out, conv_state_out, traj_state_out, pen)
        if self.cov_memory is not None:
            # T9: cov-состояние — 6-й элемент, только когда ветвь включена
            # (default off ⇒ контракт 5-кортежа и чекпойнты не меняются).
            _s_out = _s_out + (cov_state_out,)
        return h, _s_out
    
    @property
    def base_parameters(self) -> List[nn.Parameter]:
        """All params except mirror: pre_ln, conv, bind, VSA, spectral, MLP."""
        return [p for n, p in self.named_parameters() if not n.startswith('mirror.')]
    
    @property
    def mirror_parameters(self) -> List[nn.Parameter]:
        """All params inside GroupedCognitiveMirror."""
        return [p for n, p in self.named_parameters() if n.startswith('mirror.')]


# ─── WideBind Stack ────────────────────────────────────────────────────
