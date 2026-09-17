"""core/training_control.py — единый адаптивный контур EVA.

Single source of truth for the *training-time control plane*. Everything is
either derived from the τ-field (``TauConfig``) or from running data statistics
(EMA/std) — explicit pairwise coupling through τ replaces hand-tuned magic
numbers (project rule: only interconnected parameters let the system find
balance; that interconnection is the τ numbers).

Outer loop
----------
``LossBalancer`` (spectral alignment / PCGrad) — on the align steps the
aux-gradient is bounded by ``‖g_CE‖`` *by construction* (cos ∈ [0,1]), so the
old ``align_cap`` multiplier was an unnecessary knob and is gone. The M64.4
cadence's cheap path (``align_every=k>1``) is only APPROXIMATELY bounded — it
applies the align-measured scale ``s`` to the aux sum (see the class
docstring); the by-construction bound holds on the align steps.

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


def mirror_lstats(layer, tau_config=None, expl_thresh: float = 0.296) -> Tuple[float, float]:
    """(exploration, differentiation) for one layer — τ-aware and live.

    exploration = min(1, |mirror| / expl_thresh): how hard the mirror is
    correcting. expl_thresh is the λ⁻² of the layer's λ-hierarchy — callers
    pass the cfg-derived value (stack passes ``cfg.exploration_threshold``,
    itself synced from LambdaConfig); the 0.296 default is λ₃⁻².
    differentiation = behavioural divergence of the experts normalized by its
    own running mean (self-referenced ratio, saturating at 1). This replaces
    the old ``var(log_scale)/λ⁻⁴`` signal which froze at 0 whenever `log_scale`
    stopped moving (observed constant in A2-era runs), leaving every
    per-layer gain pinned at its conservative bound forever.
    """
    m = getattr(layer, 'mirror', layer)
    mag = float(getattr(m, '_last_magnitude', torch.tensor(0.0)).detach().item())
    expl = min(1.0, mag / max(float(expl_thresh), 1e-6))
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
# Loss balancing — spectral alignment (PCGrad), no cap
# ─────────────────────────────────────────────────────────────────────────────

def codebook_fingerprint(model) -> str:
    """B8 (audit 02a F2A-02): stable identity hash of the token-code geometry.
    Codes are rebuilt from algorithm+seed at every boot; if that rebuild ever
    diverges from what training actually used, resume continues on a model
    whose token identities have silently changed — the worst possible
    'successful' resume. Cheap: one sha256 over the (V,K) code block."""
    import hashlib
    emb = getattr(model, 'embed', None)
    c = getattr(emb, 'codes', None) if emb is not None else None
    if c is None:
        sd = model.state_dict()
        key = 'embed.codes' if 'embed.codes' in sd else next(
            (k for k in sd if k.endswith('.codes')), None)
        if key is None:
            return 'none'
        c = sd[key]
    c = c.detach().cpu().contiguous()
    if c.dtype == torch.bool:
        c = c.to(torch.uint8)
    h = hashlib.sha256(c.numpy().tobytes()).hexdigest()[:16]
    return f"{'x'.join(map(str, c.shape))}-{h}"


def verify_identity_resume(model, ckpt, skipped_keys):
    """B8 (audit 02a F2A-03): a size-mismatch on the identity path must be
    FATAL, not a printed SKIP. The old filter turned a geometry change
    (code_dim/vocab/head edits) into a silent half-resume: optimizer/step/
    cursor restored, token embedding & head re-initialized underneath."""
    IDENT = ('embed.', 'lm_head.', 'final_norm')
    ident = [k for k in (skipped_keys or []) if k.startswith(IDENT)]
    if ident:
        raise RuntimeError(
            f'RESUME BLOCKED (B8): {len(ident)} identity tensors shape-mismatched, '
            f'e.g. {ident[:3]} — this checkpoint was trained under DIFFERENT '
            'code/head geometry. Continuing would silently re-initialize token '
            'identity (roundtrip amnesia). Start fresh (FORCE_FRESH=True) or '
            'write a migration script for it.')
    fp_now = codebook_fingerprint(model)
    fp_ckpt = (ckpt or {}).get('code_fp')
    if fp_ckpt is None:
        print('  [B8] legacy checkpoint without code_fp — relying on shape checks only')
    elif fp_ckpt != fp_now:
        raise RuntimeError(
            f'RESUME BLOCKED (B8): codebook fingerprint {fp_ckpt!r} != rebuilt '
            f'{fp_now!r} — codes were re-derived differently than training used '
            '(seed/algorithm drift): token identities do not match the weights.')
    return fp_now


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

    ``mode='balance'`` (the ``loss()`` API): dimensionless per-aux
    normalisation by a running EMA of |aux_i|, scaled so the aux block tracks
    |CE| (Kendall & Gal / GradNorm style). Returns a scalar loss for a normal
    backward. NOTE (M64.4r3): this value-EMA normalization was REJECTED for
    the cheap backward path — it amplifies the gradient as 1/|v_i| (measured:
    1e8x the CE gradient on a zero-crossing term). The cheap path uses the
    measured gradient-geometry scale instead.

    Bypass (audit M5): terms in BYPASS_AUX (gradalign) are removed from the
    spectral alignment and backwarded DIRECTLY after the align pass — their
    purpose is a local teaching signal for the gate path; the single global
    cos-gate would otherwise zero the whole aux block (including them) on
    steps where the summed aux gradient happens orthogonal to CE. The cos
    value itself is now logged on ``self.last_cos`` (was invisible).

    M64.4 cadence (M63-F): the align path costs THREE graph traversals per
    step (CE / aux / bypass) — the largest structural cost of the run.
    ``align_every=k>1`` aligns on every k-th step and uses a cheap ONE-backward
    total ``ce + s * sum(aux)`` otherwise, where ``s`` is the EMA of the scale
    MEASURED by the last align pass (gradient geometry, not loss values). The
    cheap steps are therefore APPROXIMATELY bounded (up to the gradient drift
    between align steps; ``s`` is capped and is not seeded from noise-level
    measurements), not bounded by construction; the per-parameter sign-mask
    applies on the align steps only. ``align_every=0`` = never align after the
    first seeding call; default 1 = every step (historical, bit-identical).
    ``backward(align=False)`` is a DIFFERENT legacy path: the raw weighted sum
    (s=1.0, no normalization) — documented, not the ``loss()`` balance mode.
    """

    BYPASS_AUX = ('gradalign',)

    def __init__(self, align: bool = True, align_cap: Optional[float] = None,
                 eval_interval: int = 1000, align_every: int = 1,
                 scale_min_ratio: float = 0.05, scale_max: float = 10.0,
                 scale_ema_decay: float = 0.99) -> None:
        self.align: bool = bool(align)
        self.align_cap = align_cap  # accepted for config compatibility, NOT used
        self.eval_interval: int = int(eval_interval)
        # M64.4 (M63-F): the align path costs THREE graph traversals (CE, aux,
        # bypass) — with gradient checkpointing that is ~3 recomputes per step,
        # the single largest structural cost of the run (the step is
        # latency-bound at ~0.3% of the TF32 peak). `align_every=k>1` runs the
        # alignment on every k-th step and the cheap ONE-backward total
        # otherwise; 0 = never align. Default 1 = the historical behaviour.
        # NOTE: "never align" still aligns on the first call — the cheap path
        # needs the measured scale seed.
        self.align_every: int = int(align_every)
        # M64.4 round-3 (R1-verify): the measured scale is only meaningful when
        # the aux gradient is a non-negligible fraction of the CE one. Below
        # `scale_min_ratio` the ratio na/nb is noise (nb -> 0 => s -> inf) and
        # the seed/EMA are NOT updated; `scale_max` hard-caps the measurement
        # (both the seed and the EMA). These are safety bounds, not tuned
        # constants: the reviewer's measured counterexample was s=9.9e5 from a
        # single nb/na~1e-6 align step, giving 6.98e4x the CE gradient on a
        # later drift.
        self.scale_min_ratio: float = float(scale_min_ratio)
        self.scale_max: float = float(scale_max)
        # 0.99 = tau ~100 align steps (~800 train steps at k=8) — fast enough to
        # track regime changes, slow enough to reject a single noisy align pass.
        self.scale_ema_decay: float = float(scale_ema_decay)
        self.ema_ce: Optional[float] = None
        self.ema_aux: Dict[str, float] = {}
        self.ema_A: Optional[float] = None
        self.last_cos: Optional[float] = None   # cos of the LAST call (None on cheap)
        self.last_align_cos: Optional[float] = None  # M64.4r3: survives cheap steps
        self.last_scale: Optional[float] = None      # M64.4r3: the capped measurement
        self.last_path: str = 'align'           # 'align' | 'balance' (M64.4)
        self.scale_ema: Optional[float] = None  # M64.4: the measured align scale
        self.n_align: int = 0                   # M64.4: telemetry counters
        self.n_balance: int = 0

    def set_stats(self, eval_interval: int = 1000) -> None:
        self.eval_interval = int(eval_interval)

    def state_dict(self) -> Dict[str, Any]:
        """Persist the balance EMAs so a resumed run does not re-anneal the
        aux scaling from scratch (audit M12 single-best.pt).

        Convention (R2 review): the checkpoint is PROVENANCE, the config is
        the source of truth — `align`/`align_every` ride here for diagnostics,
        `load_state_dict` does not restore them (the constructor's values win).
        """
        return {
            'ema_ce': self.ema_ce,
            'ema_A': self.ema_A,
            'ema_aux': dict(self.ema_aux),
            'scale_ema': self.scale_ema,        # M64.4: seed for the cheap path
            'align': bool(self.align),
            'align_every': int(self.align_every),   # M64.4
        }

    def load_state_dict(self, sd: Optional[Dict[str, Any]]) -> None:
        if not sd:
            return
        self.ema_ce = sd.get('ema_ce', self.ema_ce)
        self.ema_A = sd.get('ema_A', self.ema_A)
        if sd.get('scale_ema') is not None:     # M64.4: warm-start the cheap path
            # round-4 nit: clamp a legacy/foreign seed to the safety bound
            self.scale_ema = min(float(sd['scale_ema']), self.scale_max)
        if sd.get('ema_aux') is not None:
            self.ema_aux = dict(sd['ema_aux'])

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
        """The legacy scalar-loss API (used by the align=False balance mode).

        M64.4r3 (R1/R3 review): the value-EMA normalization below is NOT the
        cheap backward path — it amplifies the gradient as 1/|v_i| (measured:
        1e8x on a zero-crossing term). Kept for API compatibility only.
        """
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

    def grad_geometry(self, ce_loss, aux_dict, parameters, max_terms: int = 8):
        """B6 (GPT-#34): per-aux gradient geometry against CE.

        Returns {name: (norm_ratio, cos)} where norm_ratio = ||g_aux||/||g_CE||
        and cos = <g_aux,g_CE>/(||g_aux|| ||g_CE||). Answers the question every
        audit this session had to guess at: is this auxiliary actually driving
        the model, fighting CE, or heating air? Pure diagnostic: uses
        autograd.grad (never touches .grad), leaves the graph intact for the
        real backward that follows. Cost: one extra backward pass per term —
        call sites gate it to log_interval frequency.
        """
        params = [p for p in parameters if p.requires_grad]
        if not params:
            return {}
        ce_g = torch.autograd.grad(ce_loss, params, retain_graph=True,
                                   allow_unused=True)
        acc_ce = None
        for g in ce_g:
            if g is None:
                continue
            f = g.reshape(-1)
            d = torch.dot(f, f)
            acc_ce = d if acc_ce is None else acc_ce + d
        if acc_ce is None:
            return {}
        out = {}
        # B12 (agent 3, F3-04): alphabetical cap hid half the ledger —
        # gradalign/pred/intent_tau/signal_ent/w_m2v never appeared in
        # [ggeo]. Priority first, then the rest alphabetically.
        PRIORITY = ('gradalign', 'bridge_conn', 'pred', 'diversity', 'branch',
                    'balance', 'decorr', 'signal_ent', 'intent_tau', 'w_m2v')
        _names = [n for n in PRIORITY if n in aux_dict] + \
                 [n for n in sorted(aux_dict) if n not in PRIORITY]
        for name in _names:
            if len(out) >= max_terms:
                break
            v = aux_dict[name]
            if not isinstance(v, torch.Tensor) or not v.requires_grad:
                continue
            try:
                tg = torch.autograd.grad(v, params, retain_graph=True,
                                         allow_unused=True)
            except RuntimeError:
                continue              # term not connected to this graph
            num = na = nb = None
            for gce, gt in zip(ce_g, tg):
                if gce is None or gt is None:
                    continue
                a, b = gce.reshape(-1), gt.reshape(-1)
                da = torch.dot(a, a); db = torch.dot(b, b)
                dd = torch.dot(a, b)
                num = dd if num is None else num + dd
                na = da if na is None else na + da
                nb = db if nb is None else nb + db
            if num is None or float(na) == 0.0 or float(nb) == 0.0:
                out[name] = (0.0, 0.0)
                continue
            out[name] = (float(nb.sqrt() / (acc_ce.sqrt() + 1e-12)),
                         float(num / (na.sqrt() * nb.sqrt() + 1e-12)))
        return out

    def backward(self, ce_loss: torch.Tensor, aux_dict: Dict[str, Any],
                 parameters: Iterable[torch.nn.Parameter],
                 retain_graph: bool = False,
                 phase_model: Any = None,
                 step: Optional[int] = None) -> None:
        # M64.4 (M63-F): the align cadence. `align_every=k>1` runs the full
        # three-traversal alignment only on the steps where `step % k == 0`
        # and a cheap ONE-backward total on the rest; 0 disables the alignment
        # entirely. Default 1 = every step (historical, bit-identical).
        #
        # The cheap total is `ce + s * sum(aux)`, where `s` is the EMA of the
        # gradient-geometry scale MEASURED by the last align pass
        # (cos+ * ||g_CE||/||g_aux||). R1 review: the value-EMA normalization
        # was rejected — normalizing by the aux VALUE amplifies the gradient
        # as 1/|v_i| (a zero-crossing term gave 1e8x the CE gradient through
        # the 1e-8 floors; sign cancellation in ema_A gave 2e6x). `s` comes
        # from the gradient geometry, so those adversarial cases cannot touch
        # it; the first step always aligns to seed `s`.
        params = [p for p in parameters if p.requires_grad]
        _cheap = (not self.align) or (self.align_every <= 0) or (
            self.align_every > 1 and step is not None
            and int(step) % self.align_every != 0)
        if _cheap and params and (not self.align or self.scale_ema is not None):
            self.last_path = 'balance'
            self.last_cos = None
            self.n_balance += 1
            # R3 review: the gradalign hook must not overwrite its CE-only
            # target with the combined gradient on the cheap steps — freeze it
            # (the target stays from the last align pass, stale <= k-1 steps;
            # the hook is one-step-stale by design, this extends it mildly).
            if phase_model is not None:
                for _l in getattr(phase_model, 'layers', []):
                    _l._ga_record = False
            try:
                # M64.4r3 (R1-verify): s comes from the measured align scale;
                # without a valid measurement the cheap step is CE-ONLY (the
                # aux still trains on the align steps). align=False is the
                # documented raw-sum legacy path (s=1.0, no normalization).
                _s = 0.0
                if not self.align:
                    _s = 1.0
                elif self.scale_ema is not None:
                    _s = float(self.scale_ema)
                _at = [v for v in aux_dict.values() if isinstance(v, torch.Tensor)]
                total = ce_loss + (_s * sum(_at) if _at and _s > 0.0 else 0.0)
                total.backward()
            finally:
                if phase_model is not None:
                    for _l in getattr(phase_model, 'layers', []):
                        _l._ga_record = True
            return
        self.last_path = 'align'
        self.n_align += 1
        self.last_cos = None
        # B2 (audit C): (i) the single GLOBAL cosine gate zeroed every aligned
        # aux term whenever the SUM ⊥ CE (measured: final grad = pure CE, and
        # on the orthogonality toy per-term beats sum-gate 8.000 vs 2.000);
        # replaced by per-coordinate sign agreement with a per-parameter norm
        # bound — the bound "‖aux‖ ≤ ‖CE‖" now holds literally per parameter,
        # and the duty cycle is no longer zero in the cancellation band.
        # (ii) the gradalign hook is now frozen during aux/bypass phases so it
        # records the CE-only magnitude (measured before: n−1 layers stored
        # the AUX gradient, rel-err 1.0).
        # (iii) bypass terms are added under the same sign-mask+bound (was:
        # raw .backward(), measured 100×‖g_CE‖ — the bound claim was false).
        bypass: Dict[str, Any] = {}
        aux_dict = dict(aux_dict)   # never mutate the caller's dict (logging)
        for _k in self.BYPASS_AUX:
            _v = aux_dict.get(_k)
            if isinstance(_v, torch.Tensor) and _v.requires_grad:
                bypass[_k] = aux_dict.pop(_k)
        if not params:
            ce_loss.backward(retain_graph=retain_graph)
            if bypass:
                sum(bypass.values()).backward()
            return
        ce_grads = torch.autograd.grad(ce_loss, params, retain_graph=True,
                                       allow_unused=True)
        aux_tensors = [v for v in aux_dict.values() if isinstance(v, torch.Tensor)]
        if not aux_tensors:
            for p, g in zip(params, ce_grads):
                # clone() ОБЯЗАТЕЛЕН: autograd.grad при retain_graph отдаёт
                # просмотры внутренних буферов движка — их нельзя мутировать
                # в grad-mode (AGC делает p.grad.mul_ in-place).
                p.grad = g.clone() if g is not None else None
            if bypass:
                # B14 (audit 04 F4-10): raw .backward() here landed a
                # bypass-only ledger UNBOUNDED on params (measured 51x CE)
                # and recorded into the gradalign CE target. Sign-mask +
                # per-param clamp + hook-freeze, exactly like the aligned path.
                if phase_model is not None:
                    for _l in getattr(phase_model, 'layers', []):
                        _l._ga_record = False
                bg = torch.autograd.grad(sum(bypass.values()), params,
                                         retain_graph=True, allow_unused=True)
                with torch.no_grad():
                    for p, gce, gb in zip(params, ce_grads, bg):
                        if gce is None and gb is None:
                            p.grad = None
                        elif gce is None:
                            p.grad = torch.zeros_like(p)
                        else:
                            p.grad = gce.clone()
                            if gb is not None:
                                b = gb * ((gce * gb) > 0)
                                _sc = torch.clamp(gce.norm() / (b.norm() + 1e-12), max=1.0)
                                p.grad.add_(b * _sc)
                if phase_model is not None:
                    for _l in getattr(phase_model, 'layers', []):
                        _l._ga_record = True
            return

        if phase_model is not None:
            for _l in getattr(phase_model, 'layers', []):
                _l._ga_record = False
        aux_total = sum(aux_tensors)
        aux_grads = torch.autograd.grad(aux_total, params,
                                        retain_graph=True,
                                        allow_unused=True)

        # Поток без полно-модельных flat-копий (аудит VRAM 2026-09): cos и нормы
        # накапливаются попарными dot на устройстве (0-dim, один .item() в конце).
        # clone() при назначении обязателен: выходы autograd.grad — просмотры
        # no_grad-буферов движка, их нельзя мутировать в grad-mode (AGC mul_).
        num = den_a = den_b = None
        for gce, gau in zip(ce_grads, aux_grads):
            if gce is None or gau is None:
                continue
            a = gce.reshape(-1)
            b = gau.reshape(-1)
            da = torch.dot(a, a)
            db = torch.dot(b, b)
            num = torch.dot(a, b) if num is None else num + torch.dot(a, b)
            den_a = da if den_a is None else den_a + da
            den_b = db if den_b is None else den_b + db
        scale = 0.0
        if num is not None:
            na = float(den_a.sqrt())
            nb = float(den_b.sqrt()) + 1e-8
            cos = float(num) / (na * nb + 1e-8)
            self.last_cos = cos          # raw alignment diagnostic (audit M5)
            self.last_align_cos = cos    # M64.4r3: survives the cheap steps
            # M64.4r3 (R1-verify): seed/update ONLY when the aux gradient is a
            # non-negligible fraction of the CE one — below scale_min_ratio the
            # ratio na/nb is noise (nb -> 0 => s -> inf; measured counterexample:
            # a single nb/na~1e-6 step seeded s=9.9e5 and a later drift gave
            # 6.98e4x the CE gradient). The scale is hard-capped at scale_max.
            if nb >= self.scale_min_ratio * na:
                scale = min(1.0, max(0.0, cos)) * na / nb
                scale = min(scale, self.scale_max)
                self.last_scale = scale
                _d = self.scale_ema_decay
                self.scale_ema = (scale if self.scale_ema is None
                                  else _d * self.scale_ema + (1.0 - _d) * scale)

        with torch.no_grad():
            for p, gce, gau in zip(params, ce_grads, aux_grads):
                if gce is None and gau is None:
                    p.grad = None
                elif gce is None:
                    p.grad = torch.zeros_like(p)
                else:
                    p.grad = gce.clone()
                    if gau is not None:
                        b = gau * ((gce * gau) > 0)          # sign-agreeing coords only
                        _s = torch.clamp(gce.norm() / (b.norm() + 1e-12), max=1.0)
                        p.grad.add_(b * _s)                  # per-param: ‖aux·s‖ ≤ ‖CE‖
        if bypass:
            # (outside no_grad: ops under no_grad fall off the graph)
            bp = torch.autograd.grad(sum(bypass.values()), params,
                                     retain_graph=retain_graph,
                                     allow_unused=True)
            with torch.no_grad():
                for p, gce, gb in zip(params, ce_grads, bp):
                    if gb is None or p.grad is None:
                        continue
                    b = gb * ((p.grad * gb) > 0)
                    _s = torch.clamp(p.grad.norm() / (b.norm() + 1e-12), max=1.0)
                    p.grad.add_(b * _s)
        if phase_model is not None:
            for _l in getattr(phase_model, 'layers', []):
                _l._ga_record = True