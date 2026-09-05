"""WideBind: unified per-layer maturation gate.

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

import math

import torch
import torch.nn as nn


class MaturationController(nn.Module):
    def __init__(self, n_layers, tau_min, tau_max, cfg, tau_config=None):
        super().__init__()
        self.n_layers = int(n_layers)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_config = tau_config  # unified τ-field (if provided, replaces manual tau_norm)
        lf = torch.linspace(0.0, 1.0, self.n_layers)
        self.register_buffer("_lf", lf)
        self.register_buffer("tau_norm", torch.zeros(self.n_layers))
        self.register_buffer("readiness", torch.zeros(self.n_layers))
        self.register_buffer("gate", torch.zeros(self.n_layers))
        self.register_buffer("pen_init", torch.full((self.n_layers,), 1.0))
        self.register_buffer("pen_ema", torch.full((self.n_layers,), 1.0))

        self.alpha = float(getattr(cfg, "matur_alpha", 1.0))
        self.T0 = float(getattr(cfg, "matur_T0", 8000.0))
        self.T_delay = float(getattr(cfg, "matur_T_delay", 8000.0))
        self.delta_t = float(getattr(cfg, "matur_delta", 4000.0))
        self.r0 = float(getattr(cfg, "matur_r0", 0.3))
        self.rs = float(getattr(cfg, "matur_rs", 0.2))
        self.ema = float(getattr(cfg, "matur_ema", 0.999))
        self.warm = int(getattr(cfg, "matur_warm", 300))

        # Global readiness threshold: when ALL layers' gates exceed this,
        # the system is ready for distributed bridge control (LayerBridgeGate).
        self.bridge_control_threshold = float(
            getattr(cfg, "matur_bridge_control_threshold", 0.1))

        if self.tau_config is None:
            self._update_tau_norm(torch.zeros(self.n_layers))  # dev = 0 initially
        else:
            self.tau_config.update()
            self.tau_norm.copy_(self.tau_config.tau_norm)

    def _update_tau_norm(self, dev):
        log_tau = math.log(self.tau_min) + (
            math.log(self.tau_max) - math.log(self.tau_min)
        ) * self._lf * (1.0 + 0.1 * dev)
        denom = math.log(self.tau_max) - math.log(self.tau_min)
        if denom <= 0:
            denom = 1.0
        self.tau_norm.copy_(((log_tau - math.log(self.tau_min)) / denom).clamp(0.0, 1.0))

    def step_gate(self, step, tau_dev=None):
        """Maturation gate: per-layer time ramp.

        Deep layers (tau_norm≈1) open first (T_eff = T0).
        Shallow layers (tau_norm≈0) open later (T_eff = T0 + T_delay).
        Pure time ramp — no bridge_readiness override (scalar would destroy
        the per-layer gradient by setting all layers to the same value).

        Args:
            step: current training step
            tau_dev: (n_layers,) deviation from base tau ladder (optional, legacy)

        Returns:
            gate: (n_layers,) per-layer maturation values in [0, 1]
        """
        if self.tau_config is not None:
            # Use tau_config's normalized tau (bilateral, learnable)
            # .data.copy_() avoids creating CopyBackwards grad_fn on the buffer;
            # grad_fn from the previous step's graph would be carried into the
            # next forward pass, causing "backward through graph a second time".
            self.tau_norm.data.copy_(self.tau_config.tau_norm.detach())
        elif tau_dev is not None:
            self._update_tau_norm(tau_dev)
        t = float(step)

        # Per-layer effective T0: deep layers (tau_norm≈1) → T_eff = T0
        # shallow layers (tau_norm≈0) → T_eff = T0 + T_delay
        gate = torch.sigmoid(
            (t - (self.T0 + self.alpha * (1.0 - self.tau_norm.detach()) * self.T_delay)) / self.delta_t)

        self.gate.data.copy_(gate)
        return self.gate

    @property
    def global_ready(self) -> bool:
        """True when ALL layers have maturation above bridge_control_threshold.

        Before global_ready, LayerBridgeGate is bypassed — bridge uses simple
        maturation gating only. This prevents the complex per-layer bridge
        routing from killing immature layers.
        """
        return bool((self.gate > self.bridge_control_threshold).all().item())

    @property
    def global_readiness_ratio(self) -> float:
        """Fraction of layers that have crossed the bridge_control_threshold."""
        return float((self.gate > self.bridge_control_threshold).float().mean().item())

    def update(self, step, pred_err):
        """Update readiness EMA from this step's per-layer pred_err (detached, (n_layers,))."""
        with torch.no_grad():
            pe = pred_err.detach().float().clamp(min=1e-6)
            if int(step) < self.warm:
                self.pen_init.copy_(torch.maximum(self.pen_init, pe))
                self.pen_ema.copy_(self.pen_init)
            else:
                self.pen_ema.lerp_(pe, 1.0 - self.ema)
            sat = (1.0 - self.pen_ema / self.pen_init.clamp(min=1e-6)).clamp(0.0, 1.0)
            self.readiness.copy_(torch.sigmoid((sat - self.r0) / self.rs))
