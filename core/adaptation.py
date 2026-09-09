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


# ─── Param-role token sets (audit M7) ───────────────────────────────────────
# Substring routing ('.b_d' in name) silently mis-captured look-alikes:
# 'b_delta_gate'/'w_delta_gate' fell into the VSA-decay λ⁻² bucket before the
# gate branch ever saw them ('silent LR confiscation'). Matching on exact
# dotted NAME PARTS removes the whole class of collisions.
_VSA_PARTS = frozenset({'b_d', 'b_i', 'scale_w'})
_MIRROR_PARTS = frozenset({'alpha_diag', 'log_skip_alpha', 'w_temp', 'w_global',
                           'log_scale', 'tanh_bias', 'log_dvar_mod_scale',
                           'dvar_mod_bias', 'log_grad_mod_scale', 'grad_mod_bias'})
_GATE_PARTS = frozenset({'w_gate', 'b_gate', 'w_delta_gate', 'b_delta_gate',
                         'w_i', 'w_d', 'w_q', 'w_q_leaf', 'w_q_ctx', 'w_mem2v',
                         'w_k_mu', 'w_q_mu', 'w_mu_mem', 'w_u', 'w_v',
                         # intent-bridge gate authorities: previously reached
                         # the gate bucket only through the '.w_i'/'.b_i'
                         # substring accident (b_intent even hit VSA first)
                         'w_intent', 'b_intent', 'w_sal'})


def _role_lr_mult(name: str, lam: Any) -> float:
    parts = frozenset(name.split('.'))
    if parts & _VSA_PARTS:
        return lam ** (-2)            # vsa scales
    if name.startswith('embed.') or name.startswith('lm_head.readout') \
            or name.startswith('lm_head.proj'):
        return lam ** (-2)            # embeddings / readout
    if (parts & _MIRROR_PARTS) or (('W_proj' in parts or 'W_out' in parts)
                                   and 'mirror' in parts):
        return lam ** (1)             # mirror projections / gates
    if '.mlp.' in name or '.bind.W_proj.weight' in name \
            or name.endswith('.W_out') or name.endswith('.W_proj'):
        return lam ** (-1)            # MLP cores / bind
    if 'reasoning_gate' in name:
        return lam ** (1)
    if parts & _GATE_PARTS:
        return lam ** (1)             # gating / memory
    return 1.0


def build_optimizer(model: torch.nn.Module, base_lr: float,
                    llrd_decay: float = 0.9, weight_decay: float = 0.01,
                    betas: Tuple[float, float] = (0.9, 0.95),
                    lam: Any = None, optimizer: str = "adamw",
                    eva_kwargs: Optional[Dict] = None) -> torch.optim.Optimizer:
    """AdamW or EVA-AdamW with Layer-wise LR Decay (LLRD).

    LLRD (Devlin et al., 2019) damps the residual-stream growth of deep blocks,
    which is the mechanism behind the ~step-1000 logit blow-up.  Frozen blocks
    have ``requires_grad=False`` and are simply skipped by the optimizer.

    When ``optimizer`` is ``'eva'``/``'eva_proj'`` the same groups are passed to
    ``EVAAdamW`` (core.eva_optim): ``'eva_proj'`` additionally enables
    AdamP-projected weight decay (Gram–Schmidt + norm-preserving rescale) on
    every dim>=2 group that carries weight decay.
    """
    if lam is None:
        from .lambda_utils import lambda_d
        lam = lambda_d(model.cfg.lambda_d)
    groups = {}
    for name, p in model.named_parameters():
        if name.endswith('._vsa_tau_log'):
            continue  # stack overrides the VSA ladder via tau_s (audit M7)
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

    # Для EVA один проход: роль (по реальным именам архитектуры) определяет
    # wd/trust/cap, lr остаётся LLRD (role_mult · llrd**depth). Это гарантирует,
    # что bridge/intent/mem/zero_init/scale_inv/tau попадают в свои группы
    # независимо от того, попал ли параметр в LLRD-разбиение.
    if optimizer in ("eva", "eva_proj"):
        from .eva_optim import EVAAdamW, _resolve_role
        mk = dict(eva_kwargs or {})
        if optimizer == "eva_proj":
            mk.setdefault("projected_wd", True)
        by_role = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith('._vsa_tau_log'):
                continue  # stack overrides the VSA ladder via tau_s (audit M7)
            r = _resolve_role(name, p.dim())
            li = _layer_index_of(name)
            role_mult = _role_lr_mult(name, lam)
            depth_mult = llrd_decay ** max(li, 0)
            lr = base_lr * role_mult * depth_mult
            wd = weight_decay if r["wd"] else 0.0
            key = (round(role_mult, 4), round(depth_mult, 6), r["role"],
                   r["wd"], r["trust"], r["cap"])
            g = by_role.get(key)
            if g is None:
                g = {"params": [], "lr": lr, "weight_decay": wd,
                     "wd_enabled": r["wd"], "role": r["role"],
                     "trust_key": r["trust"], "update_cap": r["cap"],
                     "layer_idxs": []}
                by_role[key] = g
            g["params"].append(p)
            if li is not None and li >= 0:
                g["layer_idxs"].append(int(li))
        opt = EVAAdamW(list(by_role.values()), lr=base_lr, betas=betas,
                       weight_decay=weight_decay, mode=optimizer,
                       **{k: v for k, v in mk.items()
                          if k in ("cautious", "slow_ema", "beta_slow",
                                   "slow_mix", "trust_enabled",
                                   "trust_floor", "projected_wd", "debug")})
        opt.attach_tau(model)   # константы модификаторов из τ-лестницы
        return opt

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
        self.cfg: Any = cfg

    # ── re-binding after rollback ───────────────────────────────────────────
    # ``FailureDetector.check`` does ``lr_controller.optimizer = new_opt``.
    # That MUST propagate into the wrapped MirrorLRScheduler — it anneals
    # ``_inner.optimizer.param_groups`` from ``_inner._orig_lrs``; assigning
    # only the wrapper attribute left the fresh optimizer unscheduled (no
    # re-warmup, full base LR) while the dead one was annealed. Audit 2026-09.
    @property
    def optimizer(self) -> Any:
        return self._inner.optimizer

    @optimizer.setter
    def optimizer(self, opt: Any) -> None:
        self._inner.optimizer = opt
        # re-snapshot base lrs: a freshly built optimizer carries
        # lr = base·role_mult·depth_mult (pre-scheduler values)
        self._inner._orig_lrs = [pg['lr'] for pg in opt.param_groups]

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
        # per-layer ls-EMA baselines must re-bootstrap too (else the restored
        # model is judged against pre-crash log_scale variance)
        self._inner._ls_fast = None
        self._inner._ls_slow = None
        self._inner._ls_mult = None
        for attr in ('_best_val_loss', '_loss_ema', '_loss_lr_factor'):
            if hasattr(self._inner, attr):
                delattr(self._inner, attr)
        print('  [LRController] rewind -> re-warmup from small LR')


# ─────────────────────────────────────────────────────────────────────────────
# Failure detection & loss balancing live in core/training_control.py:
#   - `FailureDetector` — multi-signal statistical SPC k·sigma rule
#     (CE + diversity + gate_mean + mlp_out + effective gate amplitude),
#     live after the EMA half-life bootstrap (not a hand-picked warmup).
#   - `LossBalancer` — spectral alignment (PCGrad) with NO align_cap:
#     the cos-projection already bounds the aux gradient by ||g_CE||.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Gradient clipping — Adaptive Gradient Clipping (AGC)
# ─────────────────────────────────────────────────────────────────────────────

class GradientClipper:
    """AGC (Brock et al. 2021): clip param grad iff ``||g|| > c·||θ||``.

    ``c`` is a *ratio* (scale-free), not an absolute norm, so it transfers
    across architectures and dtypes.  Default ``c=0.01`` matches the ResNet
    regime in the paper; raise toward 0.1 for transformer blocks.

    τ-aware clipping (docstring == код, исправлено после аудита 2026-09):
    ``attach(model)`` строит поимённый map param→layer, и для параметра слоя l
    эффективный порог ``c_eff = c · (τ_ref / τ_l)^γ``, где ``τ_ref =
    tau_config.mem_tau_ref`` и ``γ = tau_config.llrd_gamma`` — те же
    константы, которыми TauConfig строит τ-LLRD. Вне слоёв (embed/lm_head/
    memory_bank) масштаб = 1. Старый вызов set_tau_scale(mean(tau_norm)) был
    прокси без размерного смысла и не использовал per-layer τ.
    """

    def __init__(self, c: float = 0.01, eps: float = 1e-3) -> None:
        self.c: float = float(c)
        self.eps: float = float(eps)
        self._p_scale: dict = {}   # id(param) -> (τ_ref/τ_l)^γ

    def attach(self, model) -> None:
        import re as _re
        tc = getattr(model, "tau_config", None)
        if tc is None:
            return
        with torch.no_grad():
            tau_l = tc.tau_l.detach().cpu().tolist()
        ref = float(tc.mem_tau_ref)
        gamma = float(tc.llrd_gamma)
        self._p_scale = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            m = _re.match(r"(?:.*\.)?layers\.(\d+)\.", name)
            if m and int(m.group(1)) < len(tau_l):
                self._p_scale[id(p)] = (ref / max(tau_l[int(m.group(1))], 1e-6)) ** gamma

    def set_tau_scale(self, tau_norm: float = 0.0) -> None:
        """Legacy shim: τ-масштаб теперь per-layer из attach(); no-op."""
        return None

    def clip(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        for p in parameters:
            if p.grad is None:
                continue
            # τ-aware effective clip ratio (docstring==код): c·(τ_ref/τ_l)^γ
            c_eff = self.c * self._p_scale.get(id(p), 1.0)
            g_norm = p.grad.norm()
            p_norm = p.norm()
            # Skip near-zero-init params (‖θ‖≈0): AGC would otherwise set
            # g ← g·(c·‖θ‖/(‖g‖+eps)) = 0, permanently killing zero-init modules
            # (Intent Bridge w_intent/b_intent/w_sal, _tau_l_dev).
            # NOTE: zero-init params CAN explode once they grow past eps (run A2:
            # ‖w_intent‖→62k → gate blow-up, loss 3.8e22). They are now bounded
            # in-core: mirror.py divides the gate contribution by a running-RMS
            # EMA (ig/_ig_norm_ema) and scales it by the τ-tied amplitude
            # intent_alpha, so a growing ‖w_intent‖ never reaches gate_logits;
            # skipping AGC here is safe for them.
            if p_norm < self.eps:
                continue
            if g_norm > c_eff * p_norm:
                p.grad.mul_(c_eff * p_norm / (g_norm + self.eps))




# ─────────────────────────────────────────────────────────────────────────────
# Re-export (single adaptive module: core/training_control.py)
# ─────────────────────────────────────────────────────────────────────────────
from .training_control import (   # noqa: F401,E402
    FailureDetector, LossBalancer, apply_tau_lr, layer_tau_ctx, mirror_lstats,
)
