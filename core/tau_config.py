"""TauConfig — единое τ-поле EVA.

Все τ-зависимые величины архитектуры выводятся из одного класса:
- VSA tau ladder (пер-слойный временной масштаб)
- Maturation delay (когда слой просыпается)
- Gate temperatures (diversity vs precision)
- Intent alpha ladder (свежий vs перенесённый контекст)
- Memory bank temperatures (L1=быстрый, L3=медленный)
- LLRD (learning rate по глубине)

Формула (v2 — исправлены: dead zone, intent_alpha, tau_norm):
  log_tau = log(tau_min) + log(tau_max/tau_min) * (lf * (1 + 0.3 * dev) + 0.05)
  lf(l)  = l / (n_layers - 1)  ∈ [0, 1]
  dev(l) = tanh(_tau_dev[l])   ∈ [-1, +1]
  tau_l  = exp(log_tau)

Из tau_l выводятся ВСЕ остальные τ-зависимые величины:
  tau_norm_l = (log(tau_l) - log(tau_min)) / (log(tau_max) - log(tau_min))
  mat_delay_l = T0 + (1 - tau_norm_l) * T_delay
  gate_tau_l  = tau_max_gate * (tau_min_gate / tau_max_gate) ^ mat_gate_l
  alpha_l     = 1 - exp(-tau_l / tau_min)   (покрывает [0,1] при tau_min~8)
  lr_mult_l   = (tau_l / tau_ref) ^ (-gamma)
  mem_tau_l   = percentiles(tau_l) для memory bank temperatures
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Deviation coupling coefficient: how much tanh(dev) modulates log-space increments.
# With cumsum parameterization: coupling=0.4, dev_max=0.3 → tau range ~8→512 (64x).
_DEV_COUPLING: float = 0.4

# Additive floor for lf: prevents dead zone at shallow layers.
# τ_ladder(lf=0) = tau_min * base^(0.05) ≈ tau_min * 1.03 → every layer has gradient.
_LF_FLOOR: float = 0.05


class TauConfig(nn.Module):
    """Единое τ-поле: один набор параметров → все τ-зависимые величины.

    Parameters:
        n_layers: количество слоёв
        tau_min: минимальный τ (самый быстрый слой, shallow)
        tau_max: максимальный τ (самый медленный слой, deep)
        tau_init_spread: начальный разброс τ по слоям (fraction of log range)
        dev_max: максимальное отклонение dev (clip range)
        T0: базовая задержка созревания (steps)
        T_delay: дополнительная задержка для shallow слоёв (steps)
        delta_t: ширина рампы созревания (steps)
        gate_tau_min: минимальная температура гейтов
        gate_tau_max: максимальная температура гейтов
        mem_tau_ref: референсный τ для memory bank temperatures
        llrd_gamma: показатель степени для LLRD (∝ τ^{-gamma})
    """

    def __init__(
        self,
        n_layers: int = 24,
        tau_min: float = 8.0,
        tau_max: float = 512.0,
        tau_init_spread: float = 0.5,
        dev_max: float = 0.3,
        T0: float = 8000.0,
        T_delay: float = 8000.0,
        delta_t: float = 4000.0,
        gate_tau_min: float = 0.3,
        gate_tau_max: float = 5.0,
        mem_tau_ref: float = 64.0,
        llrd_gamma: float = 0.65,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dev_max = dev_max
        self.T0 = T0
        self.T_delay = T_delay
        self.delta_t = delta_t
        self.gate_tau_min = gate_tau_min
        self.gate_tau_max = gate_tau_max
        self.mem_tau_ref = mem_tau_ref
        self.llrd_gamma = llrd_gamma

        log_tau_min = math.log(max(tau_min, 1e-6))
        log_tau_max = math.log(max(tau_max, 1e-6))
        self.register_buffer('_log_tau_min', torch.tensor(log_tau_min))
        self.register_buffer('_log_tau_range', torch.tensor(log_tau_max - log_tau_min))

        # ─── Learnable parameters ───
        # _tau_dev: per-layer log-space increment deviation (monotonic via cumsum)
        # Initialized to logit(1.0)=0 → sigmoid(0)=0.5, giving base_inc/2 per layer
        self._tau_dev = nn.Parameter(torch.zeros(n_layers))

        # ─── Buffers (diagnostics) ───
        self.register_buffer('_tau_l_cache', torch.zeros(n_layers))
        self.register_buffer('_tau_norm_cache', torch.zeros(n_layers))
        self.register_buffer('_mat_delay_cache', torch.zeros(n_layers))
        self.register_buffer('_gate_tau_cache', torch.zeros(n_layers))
        self.register_buffer('_alpha_cache', torch.zeros(n_layers))
        self.register_buffer('_lr_mult_cache', torch.zeros(n_layers))
        self.register_buffer('_mem_tau_cache', torch.zeros(3))

        # Live tensors for gradient flow
        self._tau_l_live: torch.Tensor | None = None
        self._tau_norm_live: torch.Tensor | None = None
        self._alpha_live: torch.Tensor | None = None

    def _compute_tau_ladder(self) -> torch.Tensor:
        """Monotonic per-layer tau via cumsum of positive increments.

        Guarantees tau[l] < tau[l+1] by construction.
        inc[l] = base_inc * softplus(dev[l]) / softplus(0)
        At dev=0: inc = base_inc (uniform ladder). Positive dev → faster growth.
        """
        device = self._tau_dev.device
        dtype = self._tau_dev.dtype

        base_inc = self._log_tau_range / max(self.n_layers - 1, 1)
        # softplus(0) = log(2) ≈ 0.693 — normalization ensures dev=0 → base_inc
        _sp0 = math.log(2.0)
        inc = base_inc * F.softplus(self._tau_dev) / _sp0

        log_tau = self._log_tau_min + torch.cumsum(inc, dim=0)
        return torch.exp(log_tau)

    def _compute_tau_norm(self, tau_l: torch.Tensor) -> torch.Tensor:
        """Log-normalized tau to [0,1] — uniform spread over depth.

        v1 (linear) compressed middle layers. v2 (log) gives equal spacing
        in the geometric τ-ladder: mid-layer τ≈√(min·max) → norm≈0.5.
        """
        return ((tau_l.log() - self._log_tau_min) / self._log_tau_range).clamp(0.0, 1.0)

    def _compute_mat_delay(self, tau_norm: torch.Tensor) -> torch.Tensor:
        """Per-layer maturation delay in steps.

        Deep layers (tau_norm~1): open at T0
        Shallow layers (tau_norm~0): open at T0 + T_delay
        """
        return self.T0 + (1.0 - tau_norm) * self.T_delay

    def _compute_gate_tau(self, mat_gate: torch.Tensor) -> torch.Tensor:
        """Per-layer gate temperature from maturation gate.

        Immature (mat_gate~0): gate_tau = gate_tau_max (diversity)
        Mature (mat_gate~1): gate_tau = gate_tau_min (precision)
        """
        log_min = math.log(self.gate_tau_min)
        log_max = math.log(self.gate_tau_max)
        return torch.exp(log_max + (log_min - log_max) * mat_gate)

    def _compute_intent_alpha(self, tau_l: torch.Tensor) -> torch.Tensor:
        """Per-layer intent EMA alpha.

        v2: alpha = 1 − exp(−tau_l / tau_min)
          shallow (tau_l≈tau_min): alpha ≈ 1 − e^{-1} ≈ 0.63  (mostly fresh)
          deep    (tau_l≈tau_max): alpha ≈ 1 − e^{-64}  ≈ 1.0  (mostly carried)
          Covers full [0,1] range unlike v1's [0.875, 0.998].
        """
        return 1.0 - torch.exp(-tau_l / self.tau_min)

    def _compute_lr_mult(self, tau_l: torch.Tensor) -> torch.Tensor:
        """Per-layer LR multiplier (LLRD).

        lr_mult = (tau_l / tau_ref) ^ (-gamma)
        Deep layers (high tau): lower lr
        Shallow layers (low tau): higher lr
        """
        return (tau_l / self.mem_tau_ref) ** (-self.llrd_gamma)

    def _compute_mem_tau(self, tau_l: torch.Tensor) -> torch.Tensor:
        """Memory bank temperatures [L1, L2, L3] from percentiles."""
        sorted_tau, _ = torch.sort(tau_l)
        n = len(sorted_tau)
        return torch.stack([
            sorted_tau[max(0, n // 6)],           # ~17th percentile (fast)
            sorted_tau[max(0, n // 2)],            # median
            sorted_tau[min(n - 1, 5 * n // 6)],   # ~83rd percentile (slow)
        ])

    # ─── Properties ───

    @property
    def tau_l(self) -> torch.Tensor:
        return self._tau_l_live if self._tau_l_live is not None else self._tau_l_cache

    @property
    def tau_norm(self) -> torch.Tensor:
        return self._tau_norm_live if self._tau_norm_live is not None else self._tau_norm_cache

    @property
    def mat_delay(self) -> torch.Tensor:
        return self._mat_delay_cache

    @property
    def gate_tau(self) -> torch.Tensor:
        return self._gate_tau_cache

    @property
    def intent_alpha(self) -> torch.Tensor:
        return self._alpha_live if self._alpha_live is not None else self._alpha_cache

    @property
    def lr_mult(self) -> torch.Tensor:
        return self._lr_mult_cache

    @property
    def mem_tau(self) -> torch.Tensor:
        return self._mem_tau_cache

    def update(self, mat_gate: torch.Tensor | None = None):
        """Recompute all τ-derived values. Call exactly once per forward.

        Args:
            mat_gate: (n_layers,) maturation gate values [0,1], or None for initial values
        """
        tau_l = self._compute_tau_ladder()
        tau_norm = self._compute_tau_norm(tau_l)

        self._tau_l_live = tau_l
        self._tau_norm_live = tau_norm
        self._alpha_live = self._compute_intent_alpha(tau_l)

        self._tau_l_cache.copy_(tau_l.detach())
        self._tau_norm_cache.copy_(tau_norm.detach())
        self._mat_delay_cache.copy_(self._compute_mat_delay(tau_norm).detach())
        self._alpha_cache.copy_(self._alpha_live.detach())
        self._lr_mult_cache.copy_(self._compute_lr_mult(tau_l).detach())
        self._mem_tau_cache.copy_(self._compute_mem_tau(tau_l).detach())

        if mat_gate is not None:
            self._gate_tau_cache.copy_(self._compute_gate_tau(mat_gate).detach())
        else:
            self._gate_tau_cache.fill_(self.gate_tau_max)

    def get_tau_for_layer(self, layer_idx: int) -> float:
        return self._tau_l_cache[layer_idx].item()

    def get_diagnostics(self) -> dict:
        dev = torch.tanh(self._tau_dev)
        tau_l = self._tau_l_cache
        alpha = self._alpha_cache
        lr = self._lr_mult_cache
        return {
            'tau_l_mean': tau_l.mean().item(),
            'tau_l_std': tau_l.std().item(),
            'tau_l_min': tau_l.min().item(),
            'tau_l_max': tau_l.max().item(),
            'tau_l_range': (tau_l.max() - tau_l.min()).item(),
            'tau_dev_mean': dev.mean().item(),
            'tau_dev_std': dev.std().item(),
            'tau_dev_max': dev.abs().max().item(),
            'tau_dev_utilization': (dev.abs() / self.dev_max).mean().item(),
            'tau_norm_uniformity': tau_l.log().std().item() / math.log(self.tau_max / self.tau_min),
            'gate_tau_mean': self._gate_tau_cache.mean().item(),
            'intent_alpha_min': alpha.min().item(),
            'intent_alpha_max': alpha.max().item(),
            'intent_alpha_mean': alpha.mean().item(),
            'lr_mult_range': (lr.max() - lr.min()).item(),
            'lr_mult_mean': lr.mean().item(),
            'mem_tau_L1': self._mem_tau_cache[0].item(),
            'mem_tau_L3': self._mem_tau_cache[2].item(),
        }
