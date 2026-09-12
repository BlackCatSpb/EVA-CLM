"""EVA: bind module."""

from __future__ import annotations
from typing import Optional, Tuple, List

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import EVAConfig
from .vsa_utils import dct_basis, fib_sigmoid_init


class _ExpRMSNorm(nn.Module):
    """RMSNorm via explicit formula (ONNX-exportable, equiv to nn.RMSNorm)."""

    def __init__(self, K: int) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-7)

def migrate_bind_state_dict(sd: dict, n_layers: int, mode: str = "off", S: int = 1) -> dict:
    """Convert old (pre-BottleneckBind) state dict keys to new format.
    Old: layers.N.W_proj (D,K)  layers.N.W_out (K,D)  layers.N.w_u (K,)  layers.N.w_v (K,)
    New: layers.N.bind.W_proj.weight (K,D)  layers.N.bind.W_out (K,D|S,K,D)  layers.N.bind.w_u (S,K)  layers.N.bind.w_v (S,K)
    """
    import re
    map_sd: dict = {}
    for key, val in sd.items():
        m = re.match(r'layers\.(\d+)\.(W_proj|W_out|w_u|w_v|w_bind_bias)$', key)
        if not m:
            map_sd[key] = val
            continue
        lidx, param = m.groups()
        new_key = f'layers.{lidx}.bind.{param}'
        if param == 'W_proj':
            new_key = f'layers.{lidx}.bind.{param}.weight'
            map_sd[new_key] = val.t().contiguous()
        elif param == 'w_u' or param == 'w_v':
            map_sd[new_key] = val.unsqueeze(0)
        else:
            map_sd[new_key] = val
    # Remove old keys that were transformed; keep everything else
    return map_sd


# в”Ђв”Ђв”Ђ Grouped MLP в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


def _golden_shifts(K: int, S: int) -> list:
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    shifts, used = [], set()
    s = 1
    while len(shifts) < S:
        sh = int(math.floor(s * K / phi)) % K
        while sh in used or sh == 0:
            sh = (sh + 1) % K
        shifts.append(sh); used.add(sh); s += 1
    return shifts



class BottleneckBind(nn.Module):
    """Bilinear cross-mixing with Fibonacci/golden-angle shifts.

    Modes (cfg.bind_twist_mode):
      "off"     — legacy diagonal bind (u⊙v), no shifts. Exact regression.
      "shift"   — simple sum of S shifted bilinear products.
      "cascade" — Fibonacci-nested cascade, monomials up to order F_S.

    Ocular (cfg.bind_twist_ocular):
      "tied"   — shared W_out = W_proj^T for all shifts. rank(M) ≤ K.
      "multi"  — per-shift W_out′ (S, K, D). rank(M) ≤ min(S·K, D).

    Critical: w_u, w_v init std=1.0. Bilinear gradient scales as std²;
    default 0.02 kills it (8e-6 vs 1.0).
    """

    def __init__(self, D: int, K: int, cfg: object) -> None:
        super().__init__()
        self.D: int = D
        self.K: int = K
        self.mode: str = getattr(cfg, "bind_twist_mode", "off")
        self.S: int = int(getattr(cfg, "bind_twist_S", 4))
        self.softmax_free: bool = getattr(cfg, "softmax_free", True)
        if self.mode == "off":
            self.S = 1
        self.ocular: str = getattr(cfg, "bind_twist_ocular", "tied")
        self.gated: bool = bool(getattr(cfg, "bind_twist_gate", False)) and self.mode != "off"
        scheme: str = getattr(cfg, "bind_twist_scheme", "golden")
        tie_bind: bool = bool(getattr(cfg, "tie_bind", True))

        # Bias for projection
        self.w_bind_bias: nn.Parameter = nn.Parameter(torch.zeros(K))

        # --- objective lens ---
        self.W_proj: nn.Linear = nn.Linear(D, K, bias=False)
        if getattr(cfg, "bind_qk_norm", False):
            self.hp_norm = _ExpRMSNorm(K)
        else:
            self.hp_norm = nn.Identity()

        # --- deterministic shifts ---
        shifts: list = _golden_shifts(K, self.S) if scheme == "golden" else _fibonacci_shifts(K, self.S)
        self.register_buffer("shifts", torch.tensor(shifts, dtype=torch.long), persistent=False)

        # --- bilinear weights: init std=1.0 is CRITICAL ---
        self.w_u: nn.Parameter = nn.Parameter(torch.empty(self.S, K))
        self.w_v: nn.Parameter = nn.Parameter(torch.empty(self.S, K))
        nn.init.normal_(self.w_u, 0.0, 1.0)
        nn.init.normal_(self.w_v, 0.0, 1.0)

        # For shift mode with tie_bind, separate W_out per shift is needed for full rank S*K
        if self.mode == "shift" and tie_bind and self.S > 1:
            self.ocular = "multi"

        # --- ocular(s) ---
        if self.mode != "off" and self.ocular == "multi" and self.S > 1:
            self.W_out: nn.Parameter = nn.Parameter(torch.empty(self.S, K, D))
            nn.init.xavier_uniform_(self.W_out, gain=0.5)
            self._tied: bool = False
        else:
            self.W_out = nn.Parameter(torch.empty(K, D))
            nn.init.xavier_uniform_(self.W_out, gain=0.5)
            self._tied = tie_bind
            if self._tied:
                self.W_proj.register_forward_pre_hook(self._tie_hook)

        # --- gated aperture ---
        if self.gated:
            self.w_gate_proj: nn.Linear = nn.Linear(K, self.S, bias=True)
            nn.init.xavier_uniform_(self.w_gate_proj.weight, gain=0.5)
            nn.init.zeros_(self.w_gate_proj.bias)

        # --- cascade mix ---
        if self.mode == "cascade":
            # B1: fib_sigmoid_init ALREADY returns log-odds — the old double
            # inverse took log(negative) → 100% NaN outputs for mode='cascade'.
            self.mix_logit: nn.Parameter = nn.Parameter(fib_sigmoid_init(self.S).clone())
            self.log_tau: nn.Parameter = nn.Parameter(torch.tensor(1.0))  # shared tau for hybrid mixing

    def _tie_hook(self, module: nn.Linear, inp: tuple) -> None:
        with torch.no_grad():
            self.W_out.data.copy_(self.W_proj.weight.data)

    def _cross(self, left: torch.Tensor, right: torch.Tensor, shift: int) -> torch.Tensor:
        return left * torch.roll(right, shifts=int(shift), dims=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hp = self.hp_norm(self.W_proj(h) + self.w_bind_bias)  # (B, L, K)

        if self.gated:
            g = torch.sigmoid(self.w_gate_proj(hp)).unsqueeze(-1)  # (B,L,S,1)
        else:
            g = None

        # --- OFF: legacy diagonal ---
        if self.mode == "off":
            prod = (hp * self.w_u[0]) * (hp * self.w_v[0])
            return prod @ self.W_out

        # --- SHIFT: simple sum of S shifted products ---
        if self.mode == "shift":
            if not self._tied and self.ocular == "multi":
                out = None
                for s in range(self.S):
                    prod = self._cross(hp * self.w_u[s], hp * self.w_v[s], self.shifts[s])
                    if g is not None:
                        prod = prod * g[:, :, s]
                    term = prod @ self.W_out[s]
                    out = term if out is None else out + term
                return out
            else:
                acc = None
                for s in range(self.S):
                    prod = self._cross(hp * self.w_u[s], hp * self.w_v[s], self.shifts[s])
                    if g is not None:
                        prod = prod * g[:, :, s]
                    acc = prod if acc is None else acc + prod
                return acc @ self.W_out

        # --- CASCADE: Fibonacci-nested ---
        if self.mode == "cascade":
            a = [None] * (self.S + 1)
            a[1] = hp * self.w_u[0]
            a[2] = hp * self.w_v[0] if self.S >= 2 else a[1]
            seed_norm = a[1].norm(dim=-1, keepdim=True).detach()
            for n in range(3, self.S + 1):
                crossed = self._cross(a[n-1] * self.w_u[n-1], a[n-2] * self.w_v[n-1], self.shifts[n-1])
                a[n] = F.normalize(crossed + 1e-10, dim=-1) * seed_norm

            if getattr(self, 'softmax_free', True):
                # HYBRID: sigmoid (independent) * (1 + softmax (competitive))
                # Both paths share the same logits — synchronized gradients
                independent = torch.sigmoid(self.mix_logit)
                relative = F.softmax(self.mix_logit / torch.exp(self.log_tau), dim=0)
                mix = independent * (1.0 + relative)
                mix = mix / mix.sum().clamp(min=1e-6)
            else:
                mix = torch.softmax(self.mix_logit, dim=0)
            if not self._tied and self.ocular == "multi":
                out = None
                for n in range(1, self.S + 1):
                    w = mix[n-1]
                    if g is not None:
                        w = w * g[:, :, n-1]
                    term = a[n] * w.unsqueeze(-1) @ self.W_out[n-1]
                    out = term if out is None else out + term
                return out
            else:
                m = None
                for n in range(1, self.S + 1):
                    w = mix[n-1]
                    if g is not None:
                        w = w * g[:, :, n-1]
                    term = a[n] * w.unsqueeze(-1)
                    m = term if m is None else m + term
                return m @ self.W_out



def _fibonacci_shifts(K: int, S: int) -> list:
    shifts, used, a, b = [], set(), 1, 1
    guard = 0
    while len(shifts) < S and guard < 10 * S:
        sh = b % K
        if sh not in used and sh != 0:
            shifts.append(sh); used.add(sh)
        a, b = b, a + b; guard += 1
    if len(shifts) < S:
        for g in _golden_shifts(K, S):
            if g not in used:
                shifts.append(g); used.add(g)
            if len(shifts) == S:
                break
    return shifts


class SpiralBind(nn.Module):
    def __init__(self, D: int, K: int, cfg: object) -> None:
        super().__init__()
        self.D: int = D
        self.K: int = K
        self.S: int = int(getattr(cfg, "bind_twist_S", 4))
        self.W_proj: nn.Linear = nn.Linear(D, K, bias=True)
        self.w_bind_bias: nn.Parameter = nn.Parameter(torch.zeros(K))
        if getattr(cfg, "bind_qk_norm", False):
            self.hp_norm = _ExpRMSNorm(K)
        else:
            self.hp_norm = nn.Identity()
        self.w_u_re: nn.Parameter = nn.Parameter(torch.randn(self.S, K) * 0.3)
        self.w_u_im: nn.Parameter = nn.Parameter(torch.zeros(self.S, K))
        self.w_v_re: nn.Parameter = nn.Parameter(torch.randn(self.S, K) * 0.3)
        self.w_v_im: nn.Parameter = nn.Parameter(torch.zeros(self.S, K))
        tau_init: torch.Tensor = torch.log(torch.arange(1, K + 1, dtype=torch.float32) / K).unsqueeze(0).expand(self.S, -1).clone()
        self.W_freq: nn.Parameter = nn.Parameter(tau_init)
        self.W_phase: nn.Parameter = nn.Parameter(torch.randn(self.S, K) * 0.1)
        self.W_out: nn.Parameter = nn.Parameter(torch.empty(2 * K, D))
        nn.init.xavier_uniform_(self.W_out, gain=0.5)
        self._tied: bool = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hp: torch.Tensor = self.hp_norm(self.W_proj(h) + self.w_bind_bias)
        K: int = self.K
        out_acc: Optional[torch.Tensor] = None
        for s in range(self.S):
            freq = torch.exp(self.W_freq[s]).unsqueeze(0).unsqueeze(0)
            phase = self.W_phase[s].unsqueeze(0).unsqueeze(0)
            theta = freq * hp + phase
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            u_re = hp * self.w_u_re[s].unsqueeze(0).unsqueeze(0)
            u_im = hp * self.w_u_im[s].unsqueeze(0).unsqueeze(0)
            v_re = hp * self.w_v_re[s].unsqueeze(0).unsqueeze(0)
            v_im = hp * self.w_v_im[s].unsqueeze(0).unsqueeze(0)
            vr_re = v_re * cos_t - v_im * sin_t
            vr_im = v_re * sin_t + v_im * cos_t
            prod_re = u_re * vr_re - u_im * vr_im
            prod_im = u_re * vr_im + u_im * vr_re
            out_s = torch.cat([prod_re, prod_im], dim=-1)
            out_acc = out_s if out_acc is None else out_acc + out_s
        return out_acc @ self.W_out


def _fib_sequence(n_max: int) -> list:
    fibs: list = [1, 2]
    while fibs[-1] + fibs[-2] < n_max:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def _zeckendorf_levels(t: int, max_levels: int = 6) -> torch.Tensor:
    fibs: list = _fib_sequence(100)[:max_levels]
    weights: list = []
    prev: int = -2
    remaining: int = t
    for f in fibs:
        if f <= remaining and f > prev + 1:
            weights.append(1.0 / f)
            prev = f
            remaining -= f
        else:
            weights.append(0.0)
    w = torch.tensor(weights[:max_levels], dtype=torch.float32)
    return w / (w.sum() + 1e-8)


class TrajectorySpiralBind(nn.Module):
    def __init__(self, D: int, K: int, cfg: object) -> None:
        super().__init__()
        self.D: int = D
        self.K: int = K
        self.S: int = int(getattr(cfg, "bind_twist_S", 4))
        self.n_dims: int = int(getattr(cfg, "bind_traj_dims", 3))
        self.W_proj: nn.Linear = nn.Linear(D, K, bias=True)
        self.w_bind_bias: nn.Parameter = nn.Parameter(torch.zeros(K))
        if getattr(cfg, "bind_qk_norm", False):
            self.hp_norm = _ExpRMSNorm(K)
        else:
            self.hp_norm = nn.Identity()
        self.w_u_re: nn.Parameter = nn.Parameter(torch.randn(self.S, self.n_dims, K) * 0.3)
        self.w_u_im: nn.Parameter = nn.Parameter(torch.zeros(self.S, self.n_dims, K))
        self.w_v_re: nn.Parameter = nn.Parameter(torch.randn(self.S, self.n_dims, K) * 0.3)
        self.w_v_im: nn.Parameter = nn.Parameter(torch.zeros(self.S, self.n_dims, K))
        tau_init: torch.Tensor = torch.log(torch.arange(1, K + 1, dtype=torch.float32) / K).unsqueeze(0).unsqueeze(0).expand(self.S, self.n_dims, -1).clone()
        # ONNX-совместимая лог-сетка частот со сдвигом на каждую (s,d): шаг ≈ 1 октава
        tau_base: torch.Tensor = torch.log(torch.arange(1, K + 1, dtype=torch.float32) / K)
        span: float = math.log(K)
        wf: list = []
        for s in range(self.S):
            for d in range(self.n_dims):
                frac = (s * self.n_dims + d + 0.5) / (self.S * self.n_dims)
                wf.append(tau_base + (frac - 0.5) * 2.0 * span)
        self.W_freq: nn.Parameter = nn.Parameter(torch.stack(wf).view(self.S, self.n_dims, K))
        # Масштаб частот: θ = ω·hp·freq_scale. init 2π — фазы пробегают циклы при
        # типичных hp (скрещивания в hp-пространстве); старые чекпоинты через
        # миграцию получают 1.0 (численно эквивалентно прежнему поведению).
        self.freq_scale: nn.Parameter = nn.Parameter(torch.tensor(2 * math.pi))
        # U10: τ-coherent frequency schedule (audit M7: the ladder bounds are
        # read from cfg — the old literals 8.0/512.0 silently froze the
        # schedule to the default ladder even when cfg moved)
        self._eta: nn.Parameter = nn.Parameter(torch.tensor(0.5))  # learnable exponent (init 0.5)
        self._tau_min: float = float(getattr(cfg, 'tau_min', 8.0))
        self._tau_max: float = float(getattr(cfg, 'tau_max', 512.0))
        self._tau_norm: Optional[float] = None  # refreshed LIVE per forward by the block
        self.W_phase: nn.Parameter = nn.Parameter(torch.randn(self.S, self.n_dims, K) * 0.1)
        self.register_buffer('_step_count', torch.zeros(1, dtype=torch.long))
        self.hybrid_alpha_max: float = getattr(cfg, 'hybrid_alpha_max', 0.7)
        self.hybrid_alpha_min: float = getattr(cfg, 'hybrid_alpha_min', 0.3)
        # +K каналов когерентности спиралей (точки скрещивания) — в выход bind
        self.W_out: nn.Parameter = nn.Parameter(torch.empty(self.n_dims * 2 * K + K, D))
        nn.init.xavier_uniform_(self.W_out, gain=0.5)
        self._tied: bool = False
        circ_conv: torch.Tensor = torch.tensor(
            [[(n - t) % K for n in range(K)] for t in range(K)], dtype=torch.long)
        circ_corr: torch.Tensor = torch.tensor(
            [[(t - n) % K for n in range(K)] for t in range(K)], dtype=torch.long)
        self.register_buffer('_circ_conv_idx', circ_conv, persistent=False)
        self.register_buffer('_circ_corr_idx', circ_corr, persistent=False)

    def _hybrid_alpha(self) -> float:
        # B3 (agent A): the old schedule keyed off a CALL-COUNTER buffer
        # (_step_count incremented even under no_grad → eval output differed
        # from train by 0.113 rel at the same weights, and the eval branch
        # carried a second magic constant e^-2). The HRR-vs-outer-product mix
        # is a POSITION-IN-TIME-SCALE question — and τ_l IS the layer's time
        # scale: shallow (fast, τ_norm≈0) ⇒ outer-product-dominant (α_max),
        # deep (slow) ⇒ circulant (α_min). Static per layer, train ≡ eval.
        _tn = 0.5
        if getattr(self, '_tau_norm', None) is not None:
            try:
                _tn = float(self._tau_norm.item() if torch.is_tensor(self._tau_norm) else self._tau_norm)
            except Exception:
                pass
        return self.hybrid_alpha_min + (self.hybrid_alpha_max - self.hybrid_alpha_min) * (1.0 - _tn)

    def _hrr_bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        bg: torch.Tensor = b[..., self._circ_conv_idx]  # (B, L, K, K) circular shifts
        return torch.einsum('blt,bltn->bln', a, bg)

    def _hybrid_bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        alpha: float = self._hybrid_alpha()
        hrr = self._hrr_bind(a, b)
        ewise = a * b
        return alpha * hrr + (1 - alpha) * ewise

    def forward(self, h: torch.Tensor, traj_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h.dim() == 2:
            h = h.unsqueeze(0)
        hp: torch.Tensor = self.hp_norm(self.W_proj(h) + self.w_bind_bias)
        B: int
        L: int
        K: int
        B, L, K = hp.shape
        self._step_count += 1

        # Build trajectory: use traj_state only if sequence length matches
        if traj_state is not None:
            if traj_state.shape[2] != L or traj_state.shape[0] != B:
                traj_state = None
        if traj_state is None and self.training and L >= self.n_dims:
            # B2 (audit A): the trajectory previously fed the spiral either
            # zeros (seq start — measured grad w_v_re[:,d≥1] ≡ 0) or a
            # DETACHED cache (cross-position Jacobian exactly 0 — the
            # 'trajectory' learned nothing through time). In-graph shifts:
            # ∂out_t/∂h_{t−d} becomes the phase-rotation term the README promises.
            _hc = hp.permute(0, 2, 1)
            _chans = [F.pad(_hc, (dd, 0))[:, :, :L].permute(0, 2, 1)
                      for dd in range(1, self.n_dims)]
            traj_state = torch.stack(_chans, dim=1)
        if traj_state is None:
            traj_state = torch.zeros(B, self.n_dims - 1, L, K, device=h.device, dtype=h.dtype)
        traj = torch.cat([hp.unsqueeze(1), traj_state], dim=1)  # (B, n_dims, L, K)

        # U10: τ-coherent frequency schedule
        if self._tau_norm is not None:
            _eta = torch.sigmoid(self._eta)
            # freq_eff = freq_scale * (tau_min / tau_l)^η
            # tau_l = tau_min * (tau_max/tau_min)^tau_norm → tau_min/tau_l = (tau_min/tau_max)^tau_norm
            _tau_ratio = (self._tau_min / self._tau_max) ** self._tau_norm  # = tau_min/tau_l
            _freq_eff = self.freq_scale * (_tau_ratio ** _eta)
        else:
            _freq_eff = self.freq_scale

        out_acc = None
        if self.S > 1 or self.n_dims > 1:
            # Vectorized over (s, d): single batched computation instead of
            # S*n_dims sequential micro-op groups. Mathematically identical.
            hp_v = hp[:, :, None, None, :]                        # (B, L, 1, 1, K)
            wf = (torch.exp(self.W_freq) * _freq_eff).unsqueeze(0).unsqueeze(0)  # (1, 1, S, nd, K)
            wp = self.W_phase.unsqueeze(0).unsqueeze(0)
            theta = wf * hp_v + wp
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            u_re = hp_v * self.w_u_re.unsqueeze(0).unsqueeze(0)
            u_im = hp_v * self.w_u_im.unsqueeze(0).unsqueeze(0)
            traj_stack = traj.permute(0, 2, 1, 3).unsqueeze(2)    # (B, L, 1, nd, K)
            v_re = traj_stack * self.w_v_re.unsqueeze(0).unsqueeze(0)
            v_im = traj_stack * self.w_v_im.unsqueeze(0).unsqueeze(0)
            vr_re = v_re * cos_t - v_im * sin_t
            vr_im = v_re * sin_t + v_im * cos_t
            prod_re = u_re * vr_re - u_im * vr_im
            prod_im = u_re * vr_im + u_im * vr_re
            u_re_v = u_re.reshape(-1, K)
            v_re_v = v_re.reshape(-1, K)
            bg = v_re_v[..., self._circ_conv_idx]                 # (B*L*S*nd, K, K)
            alpha = self._hybrid_alpha()
            hrr = torch.einsum('bt,btn->bn', u_re_v, bg).reshape(
                u_re.shape[0], u_re.shape[1], self.S, self.n_dims, K)
            ewise = u_re * v_re
            hybrid = alpha * hrr + (1 - alpha) * ewise
            prod_re = prod_re + 0.1 * hybrid
            out_s = torch.cat([prod_re, prod_im], dim=-1)         # (B, L, S, nd, 2K)
            out_acc = out_s.sum(dim=2)                            # (B, L, nd*2K)
            out_acc = out_acc.reshape(B, L, self.n_dims * 2 * K)
            # Когерентность спиралей |Σ e^{iθ}|²: точки скрещивания фаз (стандартные ONNX-op: ReduceSum/Pow/Concat)
            if self.S * self.n_dims > 1:
                sum_cos = cos_t.sum(dim=2).sum(dim=2)             # (B, L, K)
                sum_sin = sin_t.sum(dim=2).sum(dim=2)             # (B, L, K)
                coherence = (sum_cos.pow(2) + sum_sin.pow(2)) / ((self.S * self.n_dims) ** 2)
            else:
                coherence = torch.zeros(B, L, K, device=h.device, dtype=h.dtype)
            out_acc = torch.cat([out_acc, coherence], dim=-1)     # (B, L, nd*2K + K)
        else:
            coh_re = None
            coh_im = None
            for s in range(self.S):
                dim_outputs = []
                for d in range(self.n_dims):
                    freq = (torch.exp(self.W_freq[s, d]) * _freq_eff).unsqueeze(0).unsqueeze(0)
                    phase = self.W_phase[s, d].unsqueeze(0).unsqueeze(0)
                    theta = freq * hp + phase
                    cos_t = torch.cos(theta)
                    sin_t = torch.sin(theta)
                    if self.S * self.n_dims > 1:
                        coh_re = cos_t if coh_re is None else coh_re + cos_t
                        coh_im = sin_t if coh_im is None else coh_im + sin_t
                    u_re = hp * self.w_u_re[s, d].unsqueeze(0).unsqueeze(0)
                    u_im = hp * self.w_u_im[s, d].unsqueeze(0).unsqueeze(0)
                    v_re = traj[:, d] * self.w_v_re[s, d].unsqueeze(0).unsqueeze(0)
                    v_im = traj[:, d] * self.w_v_im[s, d].unsqueeze(0).unsqueeze(0)
                    vr_re = v_re * cos_t - v_im * sin_t
                    vr_im = v_re * sin_t + v_im * cos_t
                    prod_re = u_re * vr_re - u_im * vr_im
                    prod_im = u_re * vr_im + u_im * vr_re
                    hybrid = self._hybrid_bind(u_re, v_re)
                    prod_re = prod_re + 0.1 * hybrid
                    dim_outputs.append(torch.cat([prod_re, prod_im], dim=-1))
                out_s = torch.cat(dim_outputs, dim=-1)
                out_acc = out_s if out_acc is None else out_acc + out_s
            if self.S * self.n_dims > 1:
                coherence = (coh_re.pow(2) + coh_im.pow(2)) / ((self.S * self.n_dims) ** 2)
                out_acc = torch.cat([out_acc, coherence], dim=-1)
            else:
                coherence = torch.zeros(B, L, K, device=h.device, dtype=h.dtype)
                out_acc = torch.cat([out_acc, coherence], dim=-1)
        result = out_acc @ self.W_out
        new_traj = traj[:, 1:]  # (B, n_dims - 1, L, K)
        return result, new_traj, coherence


class TrajectoryManifoldBind(TrajectorySpiralBind):
    """РЎРїРёСЂР°Р»СЊРЅР°СЏ С‚СЂР°РµРєС‚РѕСЂРёСЏ + РјР°РЅРёС„РѕР»Рґ РїРµСЂРµС…РѕРґРѕРІ, РјР°СЃС€С‚Р°Р±РёСЂРѕРІР°РЅРЅС‹Рµ РґР»СЏ Р±РѕР»СЊС€РѕР№
    WideBind (D=4096, K=64): РІСЃС‘ С‚Рѕ Р¶Рµ СЃР°РјРѕРµ, РЅРѕ РјР°СЃС€С‚Р°Р± Р±СѓС„РµСЂР° в€љ-Р·Р°РєРѕРЅРѕРј СѓРґРІРѕРµРЅ.

    РўСЂРё РјРµС…Р°РЅРёРєРё РґРѕРїРѕР»РЅСЏСЋС‚ РґСЂСѓРі РґСЂСѓРіР° (FCF-РіРёР±СЂРёРґ):
      1. РЎРїРёСЂР°Р»СЊРЅС‹Р№ bind (max_phase) вЂ” Р»РѕРєР°Р»СЊРЅРѕРµ СЃРєСЂРµС‰РёРІР°РЅРёРµ hp в†” С‚СЂР°РµРєС‚РѕСЂРёСЏ
         (VSA-РїР°РјСЏС‚СЊ + mirror), РєР°Рє РІ TrajectorySpiralBind.
      2. РњР°РЅРёС„РѕР»Рґ РїРµСЂРµС…РѕРґРѕРІ T = unbind(hp_t, hp_{t-1}), РєР»Р°СЃС‚РµСЂРёР·РѕРІР°РЅРЅС‹Р№
         РІ Р»СѓС‡Рё (С†РµРЅС‚СЂРѕРёРґС‹) вЂ” В«С„Р°РєРµР»В»: Р±Р»РёР¶РЅРёР№ РєРѕРЅС‚РµРєСЃС‚ РѕСЃРІРµС‰Р°РµС‚СЃСЏ
         Р»СѓС‡Р°РјРё, Р° РЅРµ РїРѕР»РЅРѕР№ РїР°РјСЏС‚СЊСЋ.
      3. Zeckendorf-Р·Р°С‚СѓС…Р°РЅРёРµ РїРѕ РІРѕР·СЂР°СЃС‚Сѓ Р»СѓС‡Р° (len(zeckendorf(age)))
         РІРјРµСЃС‚Рѕ СЃРІРѕР±РѕРґРЅРѕРіРѕ П„ вЂ” РёРµСЂР°СЂС…РёС‡РµСЃРєРёР№ СЂР°СЃРїР°Рґ.
    Р§С‚РµРЅРёРµ вЂ” VSA-Р±Р°РЅРґР»: sigmoid-РіРµР№С‚С‹ РїРѕ СЃС…РѕР¶РµСЃС‚Рё queryв†”Р»СѓС‡, Р±РµР· РєРІР°РґСЂР°С‚РёС‡РЅРѕР№
    РјР°С‚СЂРёС†С‹ РїРѕРїР°СЂРЅС‹С… РІРЅРёРјР°РЅРёР№ (РЅРµС‚ softmax-РєРѕРЅРєСѓСЂРµРЅС†РёРё).
    """

    def __init__(self, D: int, K: int, cfg: object) -> None:
        super().__init__(D, K, cfg)
        self.buffer_size: int = int(getattr(cfg, "traj_buffer_size", 1024))
        # Р§РёСЃР»Рѕ Р»СѓС‡РµР№ вЂ” РїСЂРѕРёР·РІРѕРґРЅРѕРµ РѕС‚ Р±СѓС„РµСЂР° (ceil(в€љbuffer)); СЃРІРѕР±РѕРґРЅС‹С… П„ РЅРµС‚:
        # РІ П†-С‚СЂР°РµРєС‚РѕСЂРёРё С‡РёСЃР»Рѕ СЂР°Р·Р»РёС‡РёРјС‹С… РєР»Р°СЃС‚РµСЂРѕРІ вЂ” РіРµРѕРјРµС‚СЂРёС‡РµСЃРєР°СЏ СЃРµСЂРµРґРёРЅР°
        # С€РєР°Р»С‹ Р±СѓС„РµСЂР°. Р‘РѕР»СЊС€Р°СЏ: Р±СѓС„РµСЂ 1024 в†’ 32 Р»СѓС‡Р° (Mini: 512 в†’ 23).
        beams_explicit: int = int(getattr(cfg, "traj_beams", 0) or 0)
        self.n_beams: int = beams_explicit if beams_explicit > 0 else int(math.ceil(
            self.buffer_size ** 0.5))
        self.cos_threshold: float = float(getattr(cfg, "traj_cos_threshold", 0.5))
        self.rebuild_interval: int = int(getattr(cfg, "traj_rebuild_interval", 128))
        self.gain: float = float(getattr(cfg, "traj_gain", 0.05))

        # РќРµРїРµСЂСЃРёСЃС‚РµРЅС‚РЅС‹Рµ (СЃР±СЂР°СЃС‹РІР°СЋС‚СЃСЏ РЅР° Р·Р°РіСЂСѓР·РєРµ С‡РµРєРїРѕРёРЅС‚Р°) Р±СѓС„РµСЂС‹ РјР°РЅРёС„РѕР»РґР°
        self.register_buffer("beam_centers", torch.zeros(self.n_beams, self.K),
                             persistent=False)
        self.register_buffer("beam_counts", torch.zeros(self.n_beams, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("beam_age", torch.zeros(self.n_beams, dtype=torch.long),
                             persistent=False)
        self.register_buffer("trans_buf", torch.zeros(self.buffer_size, self.K),
                             persistent=False)
        self.register_buffer("_trans_idx", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("_total", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("_warmz", torch.zeros(1, dtype=torch.float32), persistent=False)

        self.W_man: nn.Parameter = nn.Parameter(torch.empty(self.K, D))
        nn.init.xavier_uniform_(self.W_man, gain=0.25)
        self._fib_cache: list = self._fib_list(self.buffer_size + 1)

        # наклон sigmoid-гейтов чтения (обучаемый, VSA — без softmax/T)
        self.logit_gain: nn.Parameter = nn.Parameter(torch.tensor(3.0))
        self.log_tau: nn.Parameter = nn.Parameter(torch.tensor(1.0))  # shared tau for hybrid attention

    @staticmethod
    def _fib_list(max_n: int) -> list:
        fibs: list = [1, 2]
        while fibs[-1] <= max_n:
            fibs.append(fibs[-1] + fibs[-2])
        return fibs

    @staticmethod
    def _zlen(fibs: list, n: int) -> int:
        if n <= 0:
            return 0
        cnt: int = 0
        i: int = len(fibs) - 1
        while n > 0 and i >= 0:
            if fibs[i] <= n:
                n -= fibs[i]
                cnt += 1
            i -= 1
        return cnt

    def _hrr_unbind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        bg: torch.Tensor = b[..., self._circ_corr_idx]  # (B, L, K, K) circular shifts
        return torch.einsum('blt,bltn->bln', a, bg)

    def _hybrid_unbind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        alpha: float = self._hybrid_alpha()
        return alpha * self._hrr_unbind(a, b) + (1.0 - alpha) * (a * b)

    # в”Ђв”Ђ РїРµСЂСЃРёСЃС‚-РѕР±РЅРѕРІР»РµРЅРёРµ Р»СѓС‡РµР№ (no_grad) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def _push_transitions(self, hp: torch.Tensor) -> None:
        """Записать переходы unbind(hp_t, hp_{t-1}) в кольцевой буфер."""
        if hp.shape[1] < 2:
            return
        hp = hp.float()  # мандолд всегда в fp32 (буферы fp32, FFT стабильна)
        T: torch.Tensor = self._hybrid_unbind(hp[:, 1:], hp[:, :-1])  # (B, L-1, K)
        flat: torch.Tensor = T.reshape(-1, self.K)
        n: int = flat.shape[0]
        if n == 0:
            return
        with torch.no_grad():
            n_t = int(self._trans_idx.item())
            space = self.buffer_size - (n_t % self.buffer_size)
            first = flat[:min(n, space)]
            self.trans_buf[n_t % self.buffer_size:n_t % self.buffer_size + first.shape[0]] = first
            if n > space:
                second = flat[space:]
                self.trans_buf[:second.shape[0]] = second
            self._trans_idx += n
            self._total += n
            if int(self._total.item()) % self.rebuild_interval == 0:
                self._rebuild_beams()

    def _rebuild_beams(self) -> None:
        """Задняя VSA-кластеризация переходов в лучи (как FCF)."""
        n_valid: int = min(int(self._total.item()), self.buffer_size)
        if n_valid < 2:
            return
        samples: torch.Tensor = self.trans_buf[:n_valid].clone().float()
        norms: torch.Tensor = samples.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        samples = samples / norms
        perm: torch.Tensor = torch.randperm(n_valid, device=samples.device)
        centers: list = []
        counts: list = []
        ages: list = []
        for idx in perm.tolist():
            v: torch.Tensor = samples[idx]
            if v.abs().sum() < 1e-10:
                continue
            best: int = -1
            best_sim: float = -1.0
            for i, c in enumerate(centers):
                sim = (v * c).sum().item()
                if sim > best_sim:
                    best, best_sim = i, sim
            if best_sim > self.cos_threshold and best >= 0:
                cnt: float = counts[best]
                nv: torch.Tensor = centers[best] * cnt + v
                centers[best] = nv / nv.norm().clamp(min=1e-10)
                counts[best] = cnt + 1
            elif len(centers) < self.n_beams:
                centers.append(v)
                counts.append(1.0)
                ages.append(int(self._trans_idx.item()))
        if centers:
            n_beams: int = len(centers)
            self.beam_centers = torch.zeros(self.n_beams, self.K, device=samples.device)
            self.beam_centers.data[:n_beams] = torch.stack(centers)
            self.beam_counts.zero_()
            self.beam_counts.data[:n_beams] = torch.tensor(counts, dtype=torch.float32,
                                                           device=samples.device)
            self.beam_age.zero_()
            if ages:
                self.beam_age.data[:len(ages)] = torch.tensor(
                    ages, dtype=torch.long, device=samples.device)
        else:
            self.beam_centers.zero_()
            self.beam_counts.zero_()
            self.beam_age.zero_()

    # в”Ђв”Ђ Zeck-СЂР°СЃРїР°Рґ РІРѕР·СЂР°СЃС‚Р° Р»СѓС‡Р° в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def _zeck_weight(self, age: int) -> float:
        """theta = 1 / (1 + len(zeckendorf(age))) — нет свободного φ."""
        if age <= 0:
            return 1.0
        return 1.0 / (1.0 + self._zlen(self._fib_cache, int(age)))

    def _manifold_read(self, hp: torch.Tensor) -> torch.Tensor:
        """Бандл-чтение: hybrid attention (sigmoid * (1 + softmax/tau)) по сходству q↔beam · Zeck-затухание.

        Hybrid: sigmoid (independent) * (1 + softmax (competitive)) — synchronized gradients.
        """
        B: int
        L: int
        K: int
        B, L, K = hp.shape
        beam: torch.Tensor = self.beam_centers  # (n_beams, K)
        n_eff: int = int(self.beam_counts.clamp(min=0).gt(0).sum().item())
        if n_eff == 0:
            return torch.zeros(B, L, K, device=hp.device, dtype=hp.dtype)
        hp = hp.float()  # fp32-прецизия для стабильных sigmoid-гейтов
        beam = beam[:n_eff] / beam[:n_eff].norm(dim=-1, keepdim=True).clamp(min=1e-10)
        qnorm: torch.Tensor = hp.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        sims: torch.Tensor = (hp / qnorm) @ beam.T  # (B, L, n_beams)
        
        # Hybrid attention: sigmoid (independent) * (1 + softmax (competitive))
        logit_gain: torch.Tensor = self.logit_gain.clamp(min=0.1)
        tau: torch.Tensor = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        independent: torch.Tensor = torch.sigmoid(sims * logit_gain)  # independent per beam
        relative: torch.Tensor = F.softmax(sims * logit_gain / tau, dim=-1)  # competitive relative ranking
        w: torch.Tensor = independent * (1.0 + relative)  # combined effect
        
        now: int = int(self._trans_idx.item())
        ages: list = [max(0, now - int(a)) for a in self.beam_age[:n_eff].tolist()]
        decay: torch.Tensor = torch.tensor([self._zeck_weight(a) for a in ages],
                             device=hp.device, dtype=hp.dtype).view(1, 1, -1)
        w = w * decay
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        return w @ beam  # (B, L, K)

    def forward(self, h: torch.Tensor, traj_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h.dim() == 2:
            h = h.unsqueeze(0)
        result, new_traj, coherence = super().forward(h, traj_state)
        hp: torch.Tensor = self.hp_norm(self.W_proj(h) + self.w_bind_bias)
        self._push_transitions(hp)
        man: torch.Tensor = self._manifold_read(hp).float()
        return result + self.gain * (man @ self.W_man).clamp(-8.0, 8.0), new_traj, coherence


# в”Ђв”Ђв”Ђ WideBind Block в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
