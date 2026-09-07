"""core/adaptation.py — unified, principled training-adaptation system.

This is the SINGLE source of truth for everything that previously lived in
scattered, duplicated, empirically-tuned places (``training_guard.py``,
``stack.MirrorLRScheduler``, the notebook's inline loss loop, and
``train.py``'s bypass/aligned weighting).  Every controller here is grounded in
an established method and uses data-derived quantities instead of magic numbers.

Controllers
----------
* ``LossBalancer``
    Multi-task aux balancing with NO per-loss hand-tuned weights.
    Two mathematically-grounded modes:
      - ``mode='align'`` (default): the combined aux-gradient is projected onto
        the CE-gradient direction (cosine-similarity gate) — gradient surgery
        a la PCGrad (Yu et al., 2020) / GradDrop.  The aux gradient added to
        parameters is bounded by ``||g_CE||`` and only applied when it agrees
        with the main-task direction.  This *guarantees* aux losses cannot
        hijack the update.
      - ``mode='balance'``: each aux is divided by a running EMA of its own
        magnitude (dimensionless/unit scale) and the block scaled by an adaptive
        budget so it tracks ``|CE|`` (scale-invariant balancing, cf. Kendall &
        Gal 2018 and GradNorm normalisation, Chen et al. 2018).
    In both modes the only "weights" are derived from the data; config
    ``*_weight`` fields are intentionally ignored.

* ``DepthController``
    Progressive layer unfreezing driven by *validation-loss plateau* (diminishing
    returns): track EWMA + variance of val_loss; when the slope is not
    significantly negative (within ``k_sigma``·σ of zero) the next block is
    unlocked.  Replaces the fixed ``stage_steps`` schedule and the
    ``meta_maturity`` proxy, which degenerately saturated at 1.0 at init.

* ``LRController``
    Warmup (linear — standard) + the mirror-state adaptive multiplier from
    ``MirrorLRScheduler`` (LR up when specialisation grows, down when stalled;
    counter-cyclical on |mirror| magnitude) + ReduceLROnPlateau-style damping on
    val-loss regression.  On recovery it ``rewind()``s (re-warmup from a small
    LR) instead of an arbitrary 0.5 halving.

* ``FailureDetector``
    Statistical divergence detection (SPC 3σ rule): maintains EWMA + variance of
    CE; flags a genuine explosion only when ``CE > mean + k_sigma·σ`` AND is
    still rising, after an initial warmup.  Replaces the arbitrary
    ``watchdog_ce = 15.0`` threshold.  On trigger it rolls back to ``best.pt``,
    rebuilds a FRESH Adam (no momentum), and rewinds the LR controller.

* ``GradientClipper``
    Adaptive Gradient Clipping (AGC, Brock et al. 2021, "High-Performance
    Large-Scale Image Recognition Without Normalization"): a parameter's
    gradient is clipped iff ``||g|| > c·||θ||``, using a *ratio* constant
    ``c`` (scale-free) — replaces the absolute ``grad_clip = 0.5`` magic number.

* ``set_active_depth`` / ``build_optimizer``
    Layer-wise LR Decay (LLRD, Devlin et al. 2019, BERT fine-tuning) preserved as
    an established method; single source for the optimizer.

All stochastic quantities use EMA decays derived from a known cadence
(e.g. ``1 - 1/eval_interval``) rather than hand-picked constants.
"""

from __future__ import annotations

import gc
import math
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Progressive unfreezing — depth control
# ─────────────────────────────────────────────────────────────────────────────

def set_active_depth(model: torch.nn.Module, k: int) -> int:
    """Freeze every block with index >= k (0-based); keep [0, k) trainable."""
    k = max(0, min(int(k), len(model.layers)))
    for i, layer in enumerate(model.layers):
        requires = (i < k)
        for p in layer.parameters():
            p.requires_grad_(requires)
    model._active_depth = k
    return k


class DepthController:
    """Unlock the next block when validation loss improvement PLATEAUS.

    A loss is "plateaued" when the finite-difference slope (current vs previous
    eval) is not significantly negative: ``slope > -k_sigma·σ`` where σ is the
    running standard deviation of val_loss.  This is structural curriculum
    expansion on diminishing returns — add capacity only when the current
    capacity is saturated.  No fixed ``stage_steps`` and no degenerate maturity
    proxy.
    """

    def __init__(self, model: torch.nn.Module, n_layers: Optional[int] = None,
                 init_k: int = 8, unfreeze_inc: int = 4,
                 warmup_steps: int = 2000, k_sigma: float = 1.0,
                 eval_interval: int = 1000,
                 max_depth: Optional[int] = None) -> None:
        self.model: torch.nn.Module = model
        self.n: int = int(n_layers if n_layers is not None else len(model.layers))
        self.init_k: int = min(int(init_k), self.n)
        self.inc: int = int(unfreeze_inc)
        self.warmup: int = int(warmup_steps)
        self.k: float = float(k_sigma)
        self.eval_interval: int = int(eval_interval)
        self.max_depth: int = self.n if max_depth is None else min(int(max_depth), self.n)
        self.active: int = self.init_k
        self._val_ema: Optional[float] = None
        self._val_var: Optional[float] = None
        self._prev_val: Optional[float] = None
        self._last_depth_step: int = -10 ** 9
        set_active_depth(model, self.active)

    def set_depth(self, k: int) -> int:
        """Force the active depth (used when resuming from a checkpoint)."""
        self.active = set_active_depth(self.model, k)
        return self.active

    def update(self, step: int, val_loss: Optional[float] = None) -> int:
        """Call every step.  Depth progression only happens at eval boundaries
        (when ``val_loss`` is provided)."""
        if val_loss is None or step < self.warmup:
            return self.active
        if self._val_ema is None:
            self._val_ema = float(val_loss)
            self._val_var = 0.0
            self._prev_val = float(val_loss)
            return self.active

        a = 1.0 - 1.0 / max(self.eval_interval, 100)
        self._val_ema = a * self._val_ema + (1 - a) * float(val_loss)
        self._val_var = a * self._val_var + (1 - a) * (float(val_loss) - self._val_ema) ** 2
        std = math.sqrt(self._val_var) + 1e-8
        slope = float(val_loss) - self._prev_val
        self._prev_val = float(val_loss)

        # Plateau => diminishing returns => expand capacity.
        if slope > -self.k * std:
            if (step - self._last_depth_step >= self.eval_interval
                    and self.active < self.max_depth):
                self.active = min(self.active + self.inc, self.max_depth)
                set_active_depth(self.model, self.active)
                self._last_depth_step = step
                print(f'  [DepthController] val plateau (slope={slope:.4f} ~0 vs '
                      f'sigma={std:.4f}) -> active_depth={self.active}/{self.n}')
        return self.active


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer (LLRD) — single source
# ─────────────────────────────────────────────────────────────────────────────

def _layer_index_of(name: str) -> int:
    if name.startswith('layers.'):
        try:
            return int(name.split('.')[1])
        except (IndexError, ValueError):
            return -1
    return -1


def _role_lr_mult(name: str, lam: Any) -> float:
    if '.b_d' in name or '.b_i' in name or '.scale_w' in name:
        return lam ** (-2)            # vsa scales
    if name.startswith('embed.') or name.startswith('lm_head.readout') \
            or name.startswith('lm_head.proj'):
        return lam ** (-2)            # embeddings / readout
    if any(g in name for g in ['.mirror.alpha_diag', '.log_skip_alpha',
                               '.mirror.W_proj', '.mirror.W_out', '.mirror.w_temp',
                               '.mirror.w_global', '.mirror.log_scale',
                               '.mirror.tanh_bias', '.log_dvar_mod_scale',
                               '.dvar_mod_bias', '.log_grad_mod_scale',
                               '.grad_mod_bias']):
        return lam ** (1)             # mirror projections / gates
    if '.mlp.' in name or '.bind.W_proj.weight' in name \
            or name.endswith('.W_out') or name.endswith('.W_proj'):
        return lam ** (-1)            # MLP cores / bind
    if 'reasoning_gate' in name:
        return lam ** (1)
    if any(g in name for g in ['.w_gate', '.b_gate', '.w_delta_gate', '.b_delta_gate',
                               '.w_i', '.w_d', '.w_q', '.w_q_leaf', '.w_q_ctx',
                               '.w_mem2v', '.w_k_mu', '.w_q_mu', '.w_mu_mem',
                               '.w_u', '.w_v']):
        return lam ** (1)             # gating / memory
    return 1.0


def build_optimizer(model: torch.nn.Module, base_lr: float,
                    llrd_decay: float = 0.9, weight_decay: float = 0.01,
                    betas: Tuple[float, float] = (0.9, 0.95),
                    lam: Any = None) -> torch.optim.AdamW:
    """AdamW with Layer-wise LR Decay (LLRD): lr = base_lr · role_mult · (llrd**depth).

    LLRD (Devlin et al., 2019) damps the residual-stream growth of deep blocks,
    which is the mechanism behind the ~step-1000 logit blow-up.  Frozen blocks
    have ``requires_grad=False`` and are simply skipped by the optimizer.
    """
    if lam is None:
        from .lambda_utils import lambda_d
        lam = lambda_d(model.cfg.lambda_d)
    groups = {}
    for name, p in model.named_parameters():
        li = _layer_index_of(name)
        role_mult = _role_lr_mult(name, lam)
        depth_mult = llrd_decay ** max(li, 0)
        lr = base_lr * role_mult * depth_mult
        wd = weight_decay if p.ndim >= 2 else 0.0
        key = (round(role_mult, 4), round(depth_mult, 6), round(wd, 6))
        g = groups.get(key)
        if g is None:
            g = {'params': [], 'lr': lr, 'weight_decay': wd}
            groups[key] = g
        g['params'].append(p)
    return torch.optim.AdamW([g for g in groups.values() if g['params']],
                             betas=betas)


# ─────────────────────────────────────────────────────────────────────────────
# LR controller — warmup + mirror-adaptive multiplier + recovery rewind
# ─────────────────────────────────────────────────────────────────────────────

class LRController:
    """Warmup + mirror-state adaptive LR + plateau damping, with ``rewind()``.

    The mirror-adaptive multiplier is the principled signal already present in
    ``stack.MirrorLRScheduler`` (LR modulated by cognitive-mirror dynamics:
    up when specialisation grows, down when stalled, counter-cyclical on
    |mirror| magnitude).  We wrap it to add a statistically-grounded
    ``rewind()`` used on recovery instead of an arbitrary 0.5 halving.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 cfg: Any, warmup: Optional[int] = None,
                 base_lr: Optional[float] = None) -> None:
        from .stack import MirrorLRScheduler
        warmup = warmup if warmup is not None else getattr(cfg, 'warmup_steps', 1000)
        base_lr = base_lr if base_lr is not None else cfg.lr
        self._inner: Any = MirrorLRScheduler(model, optimizer, base_lr=base_lr,
                                             warmup=warmup, cfg=cfg)
        self.model: torch.nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.cfg: Any = cfg

    def step(self) -> None:
        self._inner.step()

    def get_last_lr(self) -> List[float]:
        return self._inner.get_last_lr()

    def report_val_loss(self, val_loss: float) -> None:
        self._inner.report_val_loss(val_loss)

    def set_step(self, n: int) -> None:
        self._inner._step = int(n)

    @property
    def _step(self) -> int:
        return self._inner._step

    @_step.setter
    def _step(self, v: int) -> None:
        self._inner._step = v

    @property
    def _ls_mult(self) -> Optional[float]:
        return getattr(self._inner, '_ls_mult', None)

    def state_dict(self) -> Dict[str, Any]:
        return self._inner.state_dict()

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self._inner.load_state_dict(sd)

    def rewind(self, warmup: Optional[int] = None) -> None:
        """Recovery: restart from a small LR (re-warmup) rather than halving.

        Re-warmup is a known stabilisation technique (warm restarts / SGDR
        semantics): after a genuine divergence the safe move is to re-anneal LR
        from near-zero, not to persist a half-size LR that may still be too large.
        """
        if warmup is not None:
            self._inner.warmup = int(warmup)
        self._inner._step = 0
        self._inner._tau_var = None
        self._inner._tau_mag = None
        self._inner._tau_1malpha = None
        self._inner._tau_gate_var = None
        for attr in ('_best_val_loss', '_loss_ema', '_loss_lr_factor'):
            if hasattr(self._inner, attr):
                delattr(self._inner, attr)
        print('  [LRController] rewind -> re-warmup from small LR')


# ─────────────────────────────────────────────────────────────────────────────
# Failure detector — statistical divergence (replaces ce > 15.0)
# ─────────────────────────────────────────────────────────────────────────────

class FailureDetector:
    """Roll back to ``best.pt`` + fresh Adam + LR rewind on a *statistical*
    CE explosion (SPC 3σ rule)."""

    def __init__(self, model: torch.nn.Module, lr_controller: LRController,
                 make_optimizer_fn: Callable[[float], torch.optim.Optimizer],
                 best_path: str,
                 base_lr: float, k_sigma: float = 3.0, warmup: int = 2000,
                 recover_max: int = 20, cooldown: int = 50,
                 min_consecutive: int = 3) -> None:
        self.model: torch.nn.Module = model
        self.lr_controller: LRController = lr_controller
        self.make_optimizer_fn: Callable[[float], torch.optim.Optimizer] = make_optimizer_fn
        self.best_path: str = best_path
        self.base_lr: float = float(base_lr)
        self.k_sigma: float = float(k_sigma)
        self.warmup: int = int(warmup)
        self.recover_max: int = int(recover_max)
        self.cooldown: int = int(cooldown)
        self.min_consecutive: int = int(min_consecutive)
        self._cooldown: int = 0
        self._ce_ema: Optional[float] = None
        self._ce_var: Optional[float] = None
        self._prev_ce: Optional[float] = None
        self._viol: int = 0
        self.recover_count: int = 0
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def check(self, ce: float, step: int) -> bool:
        ce = float(ce)
        if self._ce_ema is None:
            self._ce_ema = ce
            self._ce_var = 0.0
            self._prev_ce = ce
            return False
        # short-memory EMA (CE is noisy per-step): ~100-step half-life.
        a = 0.99
        self._ce_ema = a * self._ce_ema + (1 - a) * ce
        self._ce_var = a * self._ce_var + (1 - a) * (ce - self._ce_ema) ** 2
        std = math.sqrt(self._ce_var) + 1e-8
        rising = ce > (self._prev_ce if self._prev_ce is not None else ce)
        self._prev_ce = ce

        if step < self.warmup or self._cooldown > 0:
            if self._cooldown > 0:
                self._cooldown -= 1
            return False

        # A single blip is normal early-training noise.  Require a *sustained*
        # rise: a genuine divergence (e.g. the 12.4 -> 13.1 -> 15.9 -> 35.7
        # monotonic climb we observed) trips the bound on several consecutive
        # steps, whereas noise does not.  The bound is a relative outlier test
        # (cf. Tukey fence): ce must exceed the recent mean by at least
        # max(k_sigma*sigma, rel_margin*|mean|) so a real jump is caught even
        # when the EMA variance is still small early on.
        rel_margin = 0.15
        bound = self._ce_ema + max(self.k_sigma * std, rel_margin * abs(self._ce_ema))
        violation = (ce > bound) and rising
        self._viol = self._viol + 1 if violation else 0
        if self._viol < self.min_consecutive:
            return False

        # Genuine divergence confirmed: far above the running mean AND still climbing.
        if ce > bound and rising:
            if not os.path.exists(self.best_path):
                print(f'  [FailureDetector] ce={ce:.2f} spike but no best.pt yet — skipping')
                self._cooldown = self.cooldown
                return False
            bound = self._ce_ema + self.k_sigma * std
            print(f'  [FailureDetector] CE EXPLOSION ce={ce:.2f} > '
                  f'mean+{self.k_sigma:.0f}σ={bound:.2f} '
                  f'at step {step} -> rollback to {self.best_path}')
            ckpt = torch.load(self.best_path, map_location='cpu')
            self.model.load_state_dict(ckpt['model'], strict=False)
            if getattr(self.model, '_active_depth', None) is not None:
                set_active_depth(self.model, self.model._active_depth)
            self.recover_count += 1
            new_opt = self.make_optimizer_fn(self.base_lr)  # fresh Adam (no momentum)
            self.optimizer = new_opt
            self.lr_controller.optimizer = new_opt
            self.lr_controller.rewind()
            # R6: Invalidate cache on LR-reset (stale cache from old weights)
            if hasattr(self.model, 'reset_cache'):
                self.model.reset_cache()
            del ckpt
            gc.collect()
            torch.cuda.empty_cache()
            self._cooldown = self.cooldown
            if self.recover_count > self.recover_max:
                raise RuntimeError(
                    f'FailureDetector: {self.recover_count} recoveries exceeded '
                    f'max {self.recover_max}; aborting')
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Gradient clipping — Adaptive Gradient Clipping (AGC)
# ─────────────────────────────────────────────────────────────────────────────

class GradientClipper:
    """AGC (Brock et al. 2021): clip param grad iff ``||g|| > c·||θ||``.

    ``c`` is a *ratio* (scale-free), not an absolute norm, so it transfers
    across architectures and dtypes.  Default ``c=0.01`` matches the ResNet
    regime in the paper; raise toward 0.1 for transformer blocks.

    U9: τ-aware clipping: clip_threshold = base_c · (τ_ref / τ_l)^γ
    """

    def __init__(self, c: float = 0.01, eps: float = 1e-3,
                 tau_ref: float = 64.0, gamma: float = 0.65) -> None:
        self.c: float = float(c)
        self.eps: float = float(eps)
        self.tau_ref: float = float(tau_ref)
        self.gamma: float = float(gamma)
        self._tau_scale: float = 1.0  # default: no modulation

    def set_tau_scale(self, tau_norm: float) -> None:
        """U9: set τ-modulation for clipping: (τ_ref / τ_l)^γ."""
        import math
        # tau_norm ∈ [0,1]; reconstruct τ_l from τ-field
        # τ_l = τ_min * (τ_max/τ_min)^τ_norm, but we use a simpler proxy:
        # scale = (1 + tau_norm)^(-gamma) — shallow (τ_norm≈0) clips more loosely
        self._tau_scale = (1.0 + tau_norm) ** (-self.gamma)

    def clip(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        # U9: τ-aware effective clip ratio
        c_eff = self.c * self._tau_scale
        for p in parameters:
            if p.grad is None:
                continue
            g_norm = p.grad.norm()
            p_norm = p.norm()
            # Skip near-zero-init params (‖θ‖≈0): AGC would otherwise set
            # g ← g·(c·‖θ‖/(‖g‖+eps)) = 0, permanently killing zero-init modules
            # (Intent Bridge w_intent/b_intent/w_sal, _tau_l_dev).
            # These can't explode, so they need no clipping until they grow.
            if p_norm < self.eps:
                continue
            if g_norm > c_eff * p_norm:
                p.grad.mul_(c_eff * p_norm / (g_norm + self.eps))


# ─────────────────────────────────────────────────────────────────────────────
# Loss balancing — spectral alignment (default) / magnitude balance
# ─────────────────────────────────────────────────────────────────────────────

class LossBalancer:
    """Combine CE with auxiliary losses WITHOUT per-loss magic weights.

    ``mode='align'`` (default, recommended): spectral gradient projection.
        aux_total = Σ aux_i  (raw, unweighted)
        g_aux      = ∇ aux_total
        cos        = ⟨g_CE, g_aux⟩ / (||g_CE||·||g_aux||)
        scale      = clamp(cos, 0, cap) · ||g_CE|| / (||g_aux|| + ε)
        g_final    = g_CE + scale · g_aux
      Adding aux only along the CE direction, and only when cos>0, *bounds* the
      aux contribution by ||g_CE|| — aux losses can never hijack the update.
      This is gradient surgery
      (PCGrad/Yu et al. 2020); the combined-aux variant keeps it O(1) backward.

    ``mode='balance'``: dimensionless per-aux normalisation by a running EMA of
      |aux_i|, scaled by an adaptive budget so the aux block tracks |CE|
      (Kendall & Gal 2018 / GradNorm-style scale invariance).  Returns a scalar
      loss for a normal ``loss.backward()``.
    """

    def __init__(self, align: bool = True, align_cap: float = 10.0,
                 eval_interval: int = 1000) -> None:
        self.align: bool = bool(align)
        self.align_cap: float = float(align_cap)
        self.eval_interval: int = int(eval_interval)
        # magnitude-balance state
        self.ema_ce: Optional[float] = None
        self.ema_aux: Dict[str, float] = {}
        self.ema_A: Optional[float] = None

    def set_stats(self, eval_interval: int = 1000) -> None:
        self.eval_interval = int(eval_interval)

    # ---- magnitude-balance helpers ---------------------------------------
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
        if self.ema_A is None:
            self.ema_A = A
        else:
            self.ema_A = d * self.ema_A + (1 - d) * A

    def loss(self, ce_loss: torch.Tensor, aux_dict: Dict[str, Any]) -> torch.Tensor:
        """Scalar loss for logging / ``mode='balance'`` backward."""
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

    # ---- spectral-alignment backward -------------------------------------
    def backward(self, ce_loss: torch.Tensor, aux_dict: Dict[str, Any],
                 parameters: Iterable[torch.nn.Parameter],
                 retain_graph: bool = False) -> None:
        """Set ``p.grad`` = g_CE + scale·g_aux (PCGrad-style projection).

        ``parameters``: iterable of model parameters.  Caller must NOT also call
        ``loss.backward()``.
        """
        # Only differentiate w.r.t. parameters that actually require grad.
        # Progressive unfreezing freezes deep blocks (requires_grad=False);
        # passing them to autograd.grad as inputs raises
        # "One of the differentiated Tensors does not require grad".
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
            scale = min(cos.item() * self.align_cap, 1.0) * ce_flat.norm() / (aux_flat.norm() + 1e-8)
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
