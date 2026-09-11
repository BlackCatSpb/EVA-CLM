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

class FailureDetector:
    """ALARM SENSOR (decision D6): flags *relative* divergence of ANY
    monitored signal; ``check()`` then returns True and the CALLER STOPS.
    There is no automatic rollback any more: recovery is a human decision,
    taken from a clean best.pt (last val-improving save) plus full log
    forensics. Continuous self-regulation stays in the adaptive layer
    (MirrorLR damping, AGC, maturation ramps, the M14 soft veto) — a
    discrete emergency actuator was twice the cause of fatal incidents
    (fragmented-allocator torch.load OOMs, 2026-09) while never curing a
    root cause the sensor alone could not already reveal.

    The SAME relative rule is applied to CE and to every protective metric
    (diversity, gate_l1, mlp_ratio, effective gate amplitude). Each signal
    carries a fast EMA (a, half-life ~70) against a slow self-referencing
    baseline (a_slow, half-life ~700):

        viol  = value > slow_ema·(1 + rel_margin)   AND   value ≥ floor

    ``margins``/``floors`` widen the test per signal. Diagnostic magnitudes
    (mlp_ratio, ig_eff, diversity) are NOT steady independent variables: they
    ride a healthy ramp across the whole early training (mlp_ratio 1.0→~1.9,
    ig_eff 0→~1.1) whose trailing slow-EMA lags the current level — a pure
    relative rule would fire on every leg of the ramp. The floor is the healthy
    band's ceiling: mlp_ratio>2.0 (safe range tops ~1.9), ig_eff>1.5 (unity =
    the gate's own running-RMS setpoint; healthy real-model excursions reach
    ~1.1-1.2, a 1.5 kick = sustained +50% over setpoint = runaway), diversity
    >5.0 (healthy dense-expert diversity ≤~0.9; a real blowup is an order-of-
    magnitude explosion, e.g. 3.8e22). A *runaway* is a value OUTSIDE the band
    (the polygon sabotage: ig_eff 1.8-5.2). The absolute floor double-checks
    the relative test by discarding pre-calibration levels; CE itself needs no
    floor (armed post-eval).
    CE additionally sits under a ``ce_armed`` flag: the run's first ~1k steps show a
    *known benign transient* (CE 66→29→11 across 3 healthy restarts); rolling
    back inside it thrashes the LR. CE joins the watch only after the first
    val eval (the notebook arms it), while protective magnitudes are armed from
    bootstrap end — they are the ones that caught A2's diversity explosion.

    A metric that jumps orders of magnitude (A2 crash: diversity 3.8e22 vs a
    healthy ~0.4) violates instantly. Critically, a *slow ramp to a new plateau*
    is caught too: the long baseline still weights the old healthy level, so
    ``value`` stays above ``slow_ema·1.15`` for hundreds of steps even after
    the signal flattens (a flat absolute SPC bound `ema + kσ` goes blind the
    moment the EMA absorbs the new level — the exact failure of the A2-era
    CE-only watchdog at CE≈120).

    Each signal bootstraps for the fast-EMA half-life (1/(1−a) samples); the
    guard is live from ~step 100, not from a hand picked warmup.
    """

    def __init__(self, model: Optional[torch.nn.Module] = None,
                 k_sigma: float = 3.0,
                 warmup: int = 2000,
                 cooldown: int = 50, min_consecutive: int = 3,
                 ema_decay: float = 0.99,
                 margins: Optional[Dict[str, float]] = None,
                 floors: Optional[Dict[str, float]] = None) -> None:
        self.model = model
        self.k_sigma = float(k_sigma)  # kept for API compatibility; rule is relative now
        self.warmup = int(warmup)  # kept for API compatibility; bootstrap count governs
        self.cooldown = int(cooldown)
        self.min_consecutive = int(min_consecutive)
        self.a = float(ema_decay)
        self.a_slow = max(self.a, 1.0 - (1.0 - self.a) / 10.0)  # half-life ~700
        self._min_samples = max(3, int(round(1.0 / (1.0 - self.a))))
        self.rel_margin = 0.15  # relative-outlier floor (CE's own; same for all signals)
        self.margins = dict(margins or {})  # per-signal tighter/looser margins
        self.floors = dict(floors or {})    # per-signal healthy-band floor (see docstring)
        self.ce_armed = False  # CE joins the watch after the first val eval
        self._last_viol_name = None  # debug: which signal fired the last viol
        self._cooldown = 0
        self._viol: Dict[str, int] = {}  # consecutive violations per signal
        self.recover_count = 0
        self._stats: Dict[str, List[float]] = {}  # name -> [ema, var, prev, n]

    def state_dict(self) -> Dict[str, Any]:
        """Full watchdog state for the single-best.pt policy (audit M12): the
        per-signal baselines (fast/slow EMAs + sample counts), violation
        streaks and recovery counters must survive a session restart —
        otherwise the detector re-bootstraps blind every resume and forgets
        the alarm history it already accumulated."""
        return {
            'recover_count': int(self.recover_count),
            'ce_armed': bool(self.ce_armed),
            'cooldown': int(self._cooldown),
            'viol': {str(k): int(v) for k, v in self._viol.items()},
            'stats': {str(k): [float(x) for x in v] for k, v in self._stats.items()},
            'last_viol_name': self._last_viol_name,
        }

    def load_state_dict(self, sd: Optional[Dict[str, Any]]) -> None:
        if not sd:
            return
        self.recover_count = int(sd.get('recover_count', self.recover_count))
        self.ce_armed = bool(sd.get('ce_armed', self.ce_armed))
        self._cooldown = int(sd.get('cooldown', 0))
        self._viol = {str(k): int(v) for k, v in (sd.get('viol') or {}).items()}
        self._stats = {str(k): [float(x) for x in v]
                       for k, v in (sd.get('stats') or {}).items()}
        self._last_viol_name = sd.get('last_viol_name')

    def _observe(self, name: str, value: float) -> bool:
        """Relative-outlier test against a *long* self-referencing baseline.

        States are ``[fast_ema, prev, n, slow_ema]``. The rule is deliberately
        non-SPC: an absolute ``ema + kσ`` bound chases a slow runaway (its mean
        and variance both inflate), going blind exactly when the signal settles
        on a new plateau. ``slow_ema·(1 + rel_margin)`` keeps weighting the
        pre-runaway level for ~700 steps, so a ramp is flagged the whole time
        and a plateau is still flagged long after the fast EMA adapts.
        """
        s = self._stats.get(name)
        if s is None:
            self._stats[name] = [float(value), float(value), 1, float(value)]
            return False
        fast, prev, n, slow = s
        n += 1
        value = float(value)
        if n < self._min_samples:
            fast = self.a * fast + (1 - self.a) * value
            slow = self.a_slow * slow + (1 - self.a_slow) * value
            s[0], s[1], s[2], s[3] = fast, value, n, slow
            return False
        margin = self.margins.get(name, self.rel_margin)
        floor = self.floors.get(name, float('-inf'))
        # No `rising` requirement: an explosion peaks then DECAYS while staying far
        # above the baseline (ig_eff 33 -> 14 -> 9 vs slow*2 ≈ 0.3). Requiring
        # monotonic rise would give "0 consecutive" instead of a sustained watch.
        # Sustained-ness comes from min_consecutive=3; a healthy noise oscillation
        # flips below the threshold within a step, so it never chains 3.
        viol = (value > slow * (1.0 + margin)) and value >= floor
        s[0] = self.a * fast + (1 - self.a) * value
        s[1] = value
        s[2] = n
        s[3] = self.a_slow * slow + (1 - self.a_slow) * value
        if name == 'ce' and not self.ce_armed:
            return False  # stats still warm; CE joins the watch once armed
        if viol:
            self._last_viol_name = name
        return viol

    def arm_ce(self) -> None:
        """Arm CE watching after the first val eval AND re-bootstrap its
        baseline. The stats accumulated while CE was disarmed tracked the
        WARMUP RAMP (a rising/falling LR schedule, not model health); arming
        against them flags the first healthy CE oscillation and rollback-
        thrashes the run (live L4 incident 2026-09: armed at step 1045,
        false divergence at 1052 -> rollback -> retained-graph OOM)."""
        self.ce_armed = True
        self._stats.pop('ce', None)
        self._viol.pop('ce', None)

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
        if not math.isfinite(ce):
            # Non-finite CE is divergence BY DEFINITION (audit M8): the
            # relative chain can never confirm it (NaN comparisons are False),
            # so without this bypass the run NaN-zombies, skipping every step.
            trigger = True
            self._last_viol_name = 'ce:non-finite'
            print(f'  [FailureDetector] non-finite CE at step {step} -> forced ALARM')
        else:
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

        # Confirmed sustained relative divergence — SIREN ONLY (decision D6).
        # The sensor deliberately does not touch weights/optimizer/cache:
        # returning True instructs the loop to STOP, keeping the post-mortem
        # (this frame's weights) exactly as the log shows it, and best.pt on
        # disk remains the last CLEAN val-improving save.
        self.recover_count += 1                     # alarms raised (key kept)
        self._cooldown = self.cooldown
        self._viol = {}
        print(f'  [ALARM] step {step}: '
              f'{" ".join(f"{k}={v:.2g}" for k, v in signals.items())} — sustained '
              f'relative divergence (sensor: {self._last_viol_name or "?"}). '
              f'D6: no auto-rollback; stop, inspect, fix, resume from best.pt.')
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

    Bypass (audit M5): terms in BYPASS_AUX (gradalign) are removed from the
    spectral alignment and backwarded DIRECTLY after the align pass — their
    purpose is a local teaching signal for the gate path; the single global
    cos-gate would otherwise zero the whole aux block (including them) on
    steps where the summed aux gradient happens orthogonal to CE. The cos
    value itself is now logged on ``self.last_cos`` (was invisible).
    """

    BYPASS_AUX = ('gradalign',)

    def __init__(self, align: bool = True, align_cap: Optional[float] = None,
                 eval_interval: int = 1000) -> None:
        self.align: bool = bool(align)
        self.align_cap = align_cap  # accepted for config compatibility, NOT used
        self.eval_interval: int = int(eval_interval)
        self.ema_ce: Optional[float] = None
        self.ema_aux: Dict[str, float] = {}
        self.ema_A: Optional[float] = None
        self.last_cos: Optional[float] = None   # cos(g_CE, g_aux) of last backward

    def set_stats(self, eval_interval: int = 1000) -> None:
        self.eval_interval = int(eval_interval)

    def state_dict(self) -> Dict[str, Any]:
        """Persist the balance EMAs so a resumed run does not re-anneal the
        aux scaling from scratch (audit M12 single-best.pt)."""
        return {
            'ema_ce': self.ema_ce,
            'ema_A': self.ema_A,
            'ema_aux': dict(self.ema_aux),
            'align': bool(self.align),
        }

    def load_state_dict(self, sd: Optional[Dict[str, Any]]) -> None:
        if not sd:
            return
        self.ema_ce = sd.get('ema_ce', self.ema_ce)
        self.ema_A = sd.get('ema_A', self.ema_A)
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
                sum(bypass.values()).backward()
            return

        aux_total = sum(aux_tensors)
        aux_grads = torch.autograd.grad(aux_total, params,
                                        retain_graph=retain_graph or bool(bypass),
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
            scale = min(1.0, max(0.0, cos)) * na / nb

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

        if bypass:
            # direct teaching signal (not cos-gated) — final pass frees the graph
            sum(bypass.values()).backward()