"""core/training_control.py — единый адаптивный контур EVA.

Single source of truth for the *training-time control plane*. Everything is
either derived from the τ-field (``TauConfig``) or from running data statistics
(EMA/std) — explicit pairwise coupling through τ replaces hand-tuned magic
numbers (project rule: only interconnected parameters let the system find
balance; that interconnection is the τ numbers).

Outer loop
----------
``FailureDetector`` (multi-signal statistical SPC k·σ rule, applied to CE *and*
every protective metric: diversity, gate_mean, mlp_out, effective gate
amplitude). Bootstrap = EMA half-life (1/(1−a) samples) — the guard is live
after ~100 steps, not after a hand-picked warmup. On trigger: rollback to
``best.pt``, fresh Adam, LR rewind, cache invalidation.

``LossBalancer`` (spectral alignment / PCGrad) — the aux-gradient is bounded by
``‖g_CE‖`` *by construction* (cos ∈ [0,1]), so the old ``align_cap`` multiplier
was an unnecessary knob and is gone.

Per-layer gains
---------------
``mirror_hyperparams`` — τ-coordinate + mirror-divergence context for the
per-layer in-forward gains (consumed by ``adaptive_controller.py``); keeps the
dead ``var(log_scale)`` signal out of the loop.

``apply_tau_lr`` — per-layer learning-rate distribution through the τ-field's
``lr_mult`` (τ-LLRD, previously computed but never applied): layer gradients
are scaled by ``scheduler_ls_m · tau_config.lr_mult[l]`` after AGC — the same
mechanism the loop already used for the scheduler's log-scale multiplier.

Stable components (LR schedule, depth plateaus, AGC) keep their homes in
``adaptation.py``; this module owns the failure/loss/per-layer orchestration.
"""

from __future__ import annotations

import gc
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

import torch


# ─────────────────────────────────────────────────────────────────────────────
# τ-context for per-layer gains (mirror_hyperparams)
# ─────────────────────────────────────────────────────────────────────────────

def layer_tau_ctx(layer, tau_config=None, layer_idx: Optional[int] = None) -> Tuple[int, float, float]:
    """(layer_idx, tau_norm, intent_alpha) for a layer from the τ-field.

    Falls back to the mirror's own captured τ primitives when no ``tau_config``
    is in scope (legacy standalone use). tau_norm ∈ [0,1] is the layer's
    logarithmic position on the τ-ladder; intent_alpha = 1 − exp(−τ_l/τ_min) is
    the gate-amplitude authority.
    """
    m = getattr(layer, 'mirror', layer)
    li = getattr(layer, 'layer_idx', 0) or 0
    if tau_config is not None:
        with torch.no_grad():
            tau_norm = float(tau_config.tau_norm[li].detach().item())
            alpha = float(tau_config.intent_alpha[li].detach().item())
    else:
        tn = getattr(m, '_tau_norm_layer', None)
        al = getattr(m, '_intent_alpha', None)
        tau_norm = float(tn) if tn is not None else 1.0
        alpha = float(al) if al is not None else 1.0
    return li, tau_norm, alpha


def mirror_lstats(layer, tau_config=None) -> Tuple[float, float]:
    """(exploration, differentiation) for one layer — τ-aware and live.

    exploration = min(1, |mirror| / λ⁻²): how hard the mirror is correcting.
    differentiation = behavioural divergence of the experts normalized by its
    own running mean (self-referenced ratio, saturating at 1). This replaces
    the old ``var(log_scale)/λ⁻⁴`` signal which froze at 0 whenever `log_scale`
    stopped moving (observed constant in A2-era runs), leaving every
    per-layer gain pinned at its conservative bound forever.
    """
    m = getattr(layer, 'mirror', layer)
    mag = float(getattr(m, '_last_magnitude', torch.tensor(0.0)).detach().item())
    expl = min(1.0, mag / 0.296)  # λ⁻² (λ_d d=3 hierarchy)
    div = float(getattr(m, '_div_run', torch.tensor(0.0)).detach().item())
    rec = float(getattr(m, '_div_run_rec', torch.tensor(1e-8)).detach().item())
    diff = max(0.0, min(1.0, div / (rec + 1e-8)))
    return expl, diff


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer learning-rate distribution (τ-LLRD)
# ─────────────────────────────────────────────────────────────────────────────

def apply_tau_lr(model, tau_config=None, ls_mults: Optional[List[float]] = None) -> None:
    """Scale layer gradients by ``scheduler_ls_m · tau_config.lr_mult``.

    Both factors are per-layer data-derived multiplicatives applied after AGC:
      ls_mults   — scheduler's mirror-log-scale multiplier (existing behavior)
      lr_mult    = (τ_l / τ_ref)^(−gamma), the τ-field's own LR distribution

    This wires the τ-LLRD that was previously computed in ``TauConfig`` but
    never consumed (deep layers learn slower, shallow layers faster).
    """
    lrm = None
    if tau_config is not None:
        lrm = tau_config.lr_mult.detach()
    for i, layer in enumerate(model.layers):
        lm = 1.0
        if ls_mults is not None and i < len(ls_mults):
            lm = float(ls_mults[i])
        tm = 1.0
        if lrm is not None and i < lrm.numel():
            tm = float(lrm[i])
        mult = lm * tm
        for p in layer.base_parameters:
            if p.grad is not None:
                p.grad.mul_(mult)
        for p in layer.mirror_parameters:
            if p.grad is not None:
                p.grad.mul_(mult)


# ─────────────────────────────────────────────────────────────────────────────
# Failure detector — multi-signal statistical divergence (SPC k·σ rule)
# ─────────────────────────────────────────────────────────────────────────────

class FailureDetector:
    """Roll back to ``best.pt`` + fresh Adam + LR rewind on a *statistical*
    divergence of ANY monitored signal.

    The SAME outlier rule is applied to CE and to every protective metric
    (diversity, gate_mean, mlp_out, effective gate amplitude):

        bound = ema + max(k_sigma·σ, rel_margin·|ema|)

    A metric that jumps orders of magnitude (A2 crash: diversity 3.8e22 vs a
    healthy ~0.4) violates the bound immediately; slow exponential drift is
    caught like CE is. Each signal bootstraps its stats for the EMA half-life
    (1/(1−a) samples) — the guard is live from ~step 100, not from a hand
    picked warmup (the A2 explosion unfolded between steps 550–880, i.e.
    inside the old silent warmup window).
    """

    def __init__(self, model: torch.nn.Module, lr_controller,
                 make_optimizer_fn: Callable[[float], torch.optim.Optimizer],
                 best_path: str, base_lr: float, k_sigma: float = 3.0,
                 warmup: int = 2000, recover_max: int = 20,
                 cooldown: int = 50, min_consecutive: int = 3,
                 ema_decay: float = 0.99) -> None:
        self.model = model
        self.lr_controller = lr_controller
        self.make_optimizer_fn = make_optimizer_fn
        self.best_path = best_path
        self.base_lr = float(base_lr)
        self.k_sigma = float(k_sigma)
        self.warmup = int(warmup)  # kept for API compatibility; bootstrap count governs
        self.recover_max = int(recover_max)
        self.cooldown = int(cooldown)
        self.min_consecutive = int(min_consecutive)
        self.a = float(ema_decay)
        self._min_samples = max(3, int(round(1.0 / (1.0 - self.a))))
        self.rel_margin = 0.15  # same relative-outlier floor used for CE
        self._cooldown = 0
        self._viol: Dict[str, int] = {}  # consecutive violations per signal
        self.recover_count = 0
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._stats: Dict[str, List[float]] = {}  # name -> [ema, var, prev, n]

    def _observe(self, name: str, value: float) -> bool:
        """Test the sample against the *prior* distribution, then update.

        Testing against the pre-update EMA/SD (classical SPC semantics) keeps a
        single spike from inflating its own bound: ``bound`` is the historical
        reference, not the chasing one.
        """
        s = self._stats.get(name)
        if s is None:
            self._stats[name] = [float(value), 0.0, float(value), 1]
            return False
        ema, var, prev, n = s
        n += 1
        value = float(value)
        if n < self._min_samples:
            s[2] = value
            s[3] = n
            return False
        std = math.sqrt(var) + 1e-8
        bound = ema + max(self.k_sigma * std, self.rel_margin * abs(ema))
        rising = value >= prev  # plateau at a new level is still a sustained shift
        viol = (value > bound) and rising
        a = self.a
        ema2 = a * ema + (1 - a) * value
        var2 = a * var + (1 - a) * (value - ema2) ** 2
        s[0], s[1], s[2], s[3] = ema2, var2, value, n
        return viol

    def check(self, ce: float, step: int,
              metrics: Optional[Dict[str, float]] = None) -> bool:
        ce = float(ce)
        if self._cooldown > 0:
            self._cooldown -= 1
            return False

        signals = {'ce': ce}
        for k, v in (metrics or {}).items():
            if v is not None and math.isfinite(float(v)):
                signals[k] = float(v)

        trigger = False
        for name, value in signals.items():
            if self._observe(name, value):
                c = self._viol.get(name, 0) + 1
            else:
                c = 0
            self._viol[name] = c
            if c >= self.min_consecutive:
                trigger = True
                break

        if not trigger:
            return False

        # Genuine divergence confirmed on some signal.
        if not os.path.exists(self.best_path):
            print(f'  [FailureDetector] signal spike but no best.pt yet — skipping')
            self._cooldown = self.cooldown
            self._viol = {}
            return False
        print(f'  [FailureDetector] divergence at step {step}: '
              f'{" ".join(f"{k}={v:.2g}" for k, v in signals.items())} '
              f'-> rollback to {self.best_path}')
        ckpt = torch.load(self.best_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'], strict=False)
        if getattr(self.model, '_active_depth', None) is not None:
            from .adaptation import set_active_depth
            set_active_depth(self.model, self.model._active_depth)
        self.recover_count += 1
        new_opt = self.make_optimizer_fn(self.base_lr)  # fresh Adam (no momentum)
        self.optimizer = new_opt
        self.lr_controller.optimizer = new_opt
        self.lr_controller.rewind()
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()
        self._cooldown = self.cooldown
        self._viol = {}
        if self.recover_count > self.recover_max:
            raise RuntimeError(
                f'FailureDetector: {self.recover_count} recoveries exceeded '
                f'max {self.recover_max}; aborting')
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Loss balancing — spectral alignment (PCGrad), no cap
# ─────────────────────────────────────────────────────────────────────────────

class LossBalancer:
    """Combine CE with auxiliary losses WITHOUT per-loss magic weights.

    ``mode='align'`` (default): spectral gradient projection (PCGrad / GradDrop,
    Yu et al. 2020). The aux gradient added to parameters is bounded by
    ``‖g_CE‖`` **by construction**:

        cos  = ⟨g_CE, g_aux⟩ / (‖g_CE‖·‖g_aux‖) ∈ [-1, 1]
        scale = max(cos, 0) · ‖g_CE‖ / (‖g_aux‖ + ε)      ≤ ‖g_CE‖ / (‖g_aux‖ + ε)
        g_final = g_CE + scale·g_aux

    cos ∈ [0,1] already caps the aux projection, so no ``align_cap`` knob is
    needed (the former cap was a leftover magic constant).

    ``mode='balance'``: dimensionless per-aux normalisation by a running EMA of
    |aux_i|, scaled so the aux block tracks |CE| (Kendall & Gal / GradNorm
    style). Returns a scalar loss for normal backward.
    """

    def __init__(self, align: bool = True, align_cap: Optional[float] = None,
                 eval_interval: int = 1000) -> None:
        self.align: bool = bool(align)
        self.align_cap = align_cap  # accepted for config compatibility, NOT used
        self.eval_interval: int = int(eval_interval)
        self.ema_ce: Optional[float] = None
        self.ema_aux: Dict[str, float] = {}
        self.ema_A: Optional[float] = None

    def set_stats(self, eval_interval: int = 1000) -> None:
        self.eval_interval = int(eval_interval)

    def _ema_decay(self) -> float:
        return 1.0 - 1.0 / max(self.eval_interval, 100)

    def _update_balance(self, ce_loss: Any, aux_dict: Dict[str, Any]) -> None:
        d = self._ema_decay()
        ce = float(ce_loss.detach().item()) if isinstance(ce_loss, torch.Tensor) else float(ce_loss)
        if self.ema_ce is None:
            self.ema_ce = abs(ce) + 1e-8
        else:
            self.ema_ce = d * self.ema_ce + (1 - d) * abs(ce)
        A = 0.0
        for k, v in aux_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            val = float(v.detach().item())
            e = self.ema_aux.get(k)
            e = abs(val) + 1e-8 if e is None else d * e + (1 - d) * abs(val)
            self.ema_aux[k] = e
            A += val / e
        A = abs(A) + 1e-8
        self.ema_A = A if self.ema_A is None else d * self.ema_A + (1 - d) * A

    def loss(self, ce_loss: torch.Tensor, aux_dict: Dict[str, Any]) -> torch.Tensor:
        self._update_balance(ce_loss, aux_dict)
        total = ce_loss
        if self.align:
            return total + sum(v for v in aux_dict.values()
                               if isinstance(v, torch.Tensor))
        beta = self.ema_ce / self.ema_A
        for k, v in aux_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            total = total + beta * (v / self.ema_aux.get(k, 1e-8))
        return total

    def backward(self, ce_loss: torch.Tensor, aux_dict: Dict[str, Any],
                 parameters: Iterable[torch.nn.Parameter],
                 retain_graph: bool = False) -> None:
        params = [p for p in parameters if p.requires_grad]
        if not params:
            ce_loss.backward(retain_graph=retain_graph)
            return
        ce_grads = torch.autograd.grad(ce_loss, params, retain_graph=True,
                                       allow_unused=True)
        aux_tensors = [v for v in aux_dict.values() if isinstance(v, torch.Tensor)]
        if not aux_tensors:
            for p, g in zip(params, ce_grads):
                p.grad = g.clone() if g is not None else None
            return

        aux_total = sum(aux_tensors)
        aux_grads = torch.autograd.grad(aux_total, params, retain_graph=retain_graph,
                                        allow_unused=True)

        ce_flat, aux_flat = [], []
        for gce, gau in zip(ce_grads, aux_grads):
            if gce is not None and gau is not None:
                ce_flat.append(gce.flatten())
                aux_flat.append(gau.flatten())
        if ce_flat:
            ce_flat = torch.cat(ce_flat)
            aux_flat = torch.cat(aux_flat)
            cos = torch.nn.functional.cosine_similarity(
                ce_flat.unsqueeze(0), aux_flat.unsqueeze(0)).clamp(min=0.0, max=1.0)
            scale = cos.item() * ce_flat.norm() / (aux_flat.norm() + 1e-8)
        else:
            scale = 0.0

        with torch.no_grad():
            for p, gce, gau in zip(params, ce_grads, aux_grads):
                if gce is not None:
                    p.grad = gce.clone()
                elif gau is not None:
                    p.grad = torch.zeros_like(p)
                else:
                    p.grad = None
                if gau is not None and scale > 0:
                    if p.grad is None:
                        p.grad = gau * scale
                    else:
                        p.grad.add_(gau, alpha=scale)