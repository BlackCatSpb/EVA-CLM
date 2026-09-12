"""EVA: unified per-layer maturation gate.

Replaces the ad-hoc wake-up crutches (hard pm_coh threshold on mlp_mod std,
fixed pm_write_delay, bridge-injection scale=0 hack) with ONE principled
mechanism: each layer l has a maturity M_l(t) in [0,1] that gates EVERY
"wake-up" signal (live BridgeGLU modulation, private-memory write, semantic
bridge injection, intent bus).

  M_l(t) = sigmoid((t - (T0 + alpha*tau_norm_l*T_delay)) / delta_t)

i.e. a SMOOTH TIME/τ RAMP. It starts at ~0 for every layer (so the trunk is
never perturbed by untrained wake-up branches at init -> no divergence), then
opens gradually on a schedule: shallow layers (small tau in the VSA ladder)
open first, deeper layers later, unified with the model's τ geometry
(log-normalised so the ladder spans [0,1] instead of 5 decades of raw tau).

Why a time ramp and not expert-saturation (readiness)? At scale the base model
does NOT learn LM on its own (ce ~ ln(vocab), random baseline), so a
readiness trigger (pred_err must DROP first) deadlocks: gate closed -> no
learning -> pred_err never drops -> gate stays closed forever -> bridge never
engages. The time ramp breaks that by opening the wake-up branches on a fixed
schedule, giving the bridge a chance to supply the learning signal while
staying smooth enough to avoid the original divergence.

The FROZEN base MLP gate (sigmoid(mod_scale_mlp) ~ 0.667) stays OPEN regardless
(no deadlock). `readiness` is still tracked (see update) and exposed for
diagnostics, but it no longer blocks the gate.

PER-LAYER MATURATION: deep layers (large tau) open first, shallow layers (small
tau) open later. This is MONOTONIC (deep-first) and mathematically stable:
skip connections in shallow layers preserve gradient flow while deep layers
learn. Bridge injection per-layer is gated by M_l, so immature layers are
protected from perturbation.

GLOBAL READINESS: When ALL layers have M_l > bridge_control_threshold, the
system is "globally ready" for distributed bridge control (LayerBridgeGate
with SpectrumGate). Before that, bridge uses simple maturation gating only.
This prevents the complex per-layer bridge routing from killing immature layers.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class MaturationController(nn.Module):
    """Per-layer maturation controller using time/τ ramp.

    Each layer has a maturity gate M_l ∈ [0,1] that controls when
    wake-up signals (BridgeGLU, memory write, bridge injection, intent bus)
    become active.

    Args:
        n_layers: Number of layers.
        tau_min: Minimum τ value in the ladder.
        tau_max: Maximum τ value in the ladder.
        cfg: Configuration object with maturation hyperparameters.
        tau_config: Unified τ-field configuration (optional).
    """

    def __init__(
        self,
        n_layers: int,
        tau_min: float,
        tau_max: float,
        cfg: object,
        tau_config: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.n_layers: int = int(n_layers)
        self.tau_min: float = float(tau_min)
        self.tau_max: float = float(tau_max)
        self.tau_config = tau_config
        lf: torch.Tensor = torch.linspace(0.0, 1.0, self.n_layers)
        self.register_buffer("_lf", lf)
        self.register_buffer("tau_norm", torch.zeros(self.n_layers))
        self.register_buffer("readiness", torch.zeros(self.n_layers))
        self.register_buffer("gate", torch.zeros(self.n_layers))
        self.register_buffer("pen_init", torch.full((self.n_layers,), 1.0))
        self.register_buffer("pen_ema", torch.full((self.n_layers,), 1.0))

        self.alpha: float = float(getattr(cfg, "matur_alpha", 1.0))
        self.T0: float = float(getattr(cfg, "matur_T0", 8000.0))
        self.T_delay: float = float(getattr(cfg, "matur_T_delay", 8000.0))
        self.delta_t: float = float(getattr(cfg, "matur_delta", 4000.0))
        self.r0: float = float(getattr(cfg, "matur_r0", 0.3))
        self.rs: float = float(getattr(cfg, "matur_rs", 0.2))
        self.ema: float = float(getattr(cfg, "matur_ema", 0.999))
        self.warm: int = int(getattr(cfg, "matur_warm", 300))

        self.bridge_control_threshold: float = float(
            getattr(cfg, "matur_bridge_control_threshold", 0.1))

        if self.tau_config is None:
            self._update_tau_norm(torch.zeros(self.n_layers))
        else:
            self.tau_config.update()
            self.tau_norm.copy_(self.tau_config.tau_norm)

    def _update_tau_norm(self, dev: torch.Tensor) -> None:
        """Update tau_norm from deviation tensor."""
        log_tau: float = math.log(self.tau_min) + (
            math.log(self.tau_max) - math.log(self.tau_min)
        ) * self._lf * (1.0 + 0.1 * dev)
        denom: float = math.log(self.tau_max) - math.log(self.tau_min)
        if denom <= 0:
            denom = 1.0
        self.tau_norm.copy_(((log_tau - math.log(self.tau_min)) / denom).clamp(0.0, 1.0))

    def step_gate(
        self,
        step: int,
        tau_dev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Maturation gate: per-layer time ramp.

        Deep layers (tau_norm≈1) open first (T_eff = T0).
        Shallow layers (tau_norm≈0) open later (T_eff = T0 + T_delay).

        Args:
            step: Current training step.
            tau_dev: (n_layers,) deviation from base tau ladder (optional).

        Returns:
            gate: (n_layers,) per-layer maturation values in [0, 1].
        """
        _tn_live = None
        if self.tau_config is not None:
            _tn_live = self.tau_config.tau_norm_live()      # B3: grad path to _tau_dev
            self.tau_norm.data.copy_(_tn_live.detach())
        elif tau_dev is not None:
            self._update_tau_norm(tau_dev)
        t: float = max(float(step), 1.0)

        # B3 (agent D): the linear-time sigmoid saturated to fp32-EXACT 1.0 at
        # t≈83k (sigmoid(35.5)) and its τ-derivative was 0 by construction
        # (detach) — wake-up timing was neither learnable nor adjustable late
        # in a 150k+ budget. Log-time reparameterization keeps the FIRST-ORDER
        # behavior at t≈T (log t − log T ≈ (t−T)/T ⇒ arg ≈ (t−T)/Δ, same local
        # slope) while ∂arg/∂log t = T/Δ stays alive forever (grad ~1/t decay).
        _tn = _tn_live if _tn_live is not None else self.tau_norm
        _T_eff = self.T0 + self.alpha * (1.0 - _tn) * self.T_delay
        gate: torch.Tensor = torch.sigmoid(
            (math.log(t) - torch.log(_T_eff.clamp(min=1.0))) * (_T_eff / self.delta_t))

        self.gate.data.copy_(gate.detach())
        return gate

    @property
    def global_ready(self) -> bool:
        """True when ALL layers have maturation above bridge_control_threshold."""
        return bool((self.gate > self.bridge_control_threshold).all().item())

    @property
    def global_readiness_ratio(self) -> float:
        """Fraction of layers that have crossed the bridge_control_threshold."""
        return float((self.gate > self.bridge_control_threshold).float().mean().item())

    def update(self, step: int, pred_err: torch.Tensor) -> None:
        """Update readiness EMA from this step's per-layer pred_err.

        Args:
            step: Current training step.
            pred_err: (n_layers,) per-layer prediction errors (detached).
        """
        with torch.no_grad():
            pe: torch.Tensor = pred_err.detach().float().clamp(min=1e-6)
            if int(step) < self.warm:
                self.pen_init.copy_(torch.maximum(self.pen_init, pe))
                self.pen_ema.copy_(self.pen_init)
            else:
                self.pen_ema.lerp_(pe, 1.0 - self.ema)
            sat: torch.Tensor = (1.0 - self.pen_ema / self.pen_init.clamp(min=1e-6)).clamp(0.0, 1.0)
            self.readiness.copy_(torch.sigmoid((sat - self.r0) / self.rs))
