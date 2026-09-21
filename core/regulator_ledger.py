"""P3-1: RegulatorLedger — the kill-switch for forward-path regulators.

A generalization of AuxKillSwitch (M64.12) from aux losses to the control laws.
Every regulator is periodically clamped to IDENTITY on a fixed probe batch and
the paired CE difference is attributed to it:

    delta_i = CE(identity_i) − CE(active)     [nat/token, probe batch, no_grad]

Round-robin over 2-3 regulators per point (the AuxKillSwitch cadence); a Schmitt
trigger with dwell: `delta_ema < eps_off` for `dwell` consecutive rounds =>
DORMANT (a retirement recommendation); `> 2*eps_off` => ACTIVE. Two noise
floors: sigma_re (the rerun determinism noise) and sigma_b (the batch-swap CE
spread); eps_off = max(2*sigma_re, 0.01*sigma_b, 1e-4) is a DECISION threshold
(an effect below 1% of the batch noise is a retirement candidate), not a
measurement threshold.

Auto-off is FORBIDDEN (the `aux_kill_disable` doctrine): the ledger recommends,
the operator / a new A/B decides through a cfg flag.
"""
from __future__ import annotations

import torch
from typing import Callable, Dict, List, Optional


class Reg:
    """A regulator with a reversible identity clamp (attribute level, no graph surgery)."""

    def __init__(self, name, enter: Callable[[], None], leave: Callable[[], None]):
        self.name = name
        self._enter = enter
        self._leave = leave

    def identity(self):
        self._enter()

    def restore(self):
        self._leave()


def _attr_reg(name, obj, attr, ident):
    """Reg for a simple attribute (bool/float/None).

    The attribute may be absent (e.g. _damp_on/_pen_decay_on before the block.py
    patch lands): identity becomes a no-op and restore deletes the temporary
    attribute — the ledger is safe on any code state (those rows then measure
    ~0 and are meaningful only after the patch).
    """
    box = {}

    def enter():
        box['had'] = hasattr(obj, attr)
        box['old'] = getattr(obj, attr, None)
        setattr(obj, attr, ident)

    def leave():
        if box['had']:
            setattr(obj, attr, box['old'])
        else:
            try:
                delattr(obj, attr)
            except AttributeError:
                pass

    return Reg(name, enter, leave)


def _multi(name, regs: List[Reg]) -> Reg:
    return Reg(name, lambda: [r.identity() for r in regs],
               lambda: [r.restore() for r in regs])


def build_registry(model, cfg) -> List[Reg]:
    """v1: the attribute-level regulators + the two flagged block.py paths."""
    R: List[Reg] = []
    hd = getattr(model, 'lm_head', None)
    if hd is not None and getattr(hd, 'temper_on', False):
        R.append(_attr_reg('head_temper', hd, '_temper_active', False))
    if hd is not None and getattr(hd, 'Kp', 0) > 0:
        kp = hd._kp_active
        box = {}

        def _ph_enter(box=box, kp=kp):
            box['v'] = int(kp.item())
            kp.fill_(0)                 # _phantom_mix: (…,0)@(0,K) -> 0

        def _ph_leave(box=box, kp=kp):
            kp.fill_(box['v'])

        R.append(Reg('head_phantom', _ph_enter, _ph_leave))
    ucl = getattr(model, 'concept_layer', None)
    if ucl is not None:
        rs = ucl.read_scale
        box = {}

        def _ucl_enter(box=box, rs=rs):
            box['v'] = rs.detach().clone()
            rs.data.fill_(-30.0)        # sigma(-30) ~ 1e-13

        def _ucl_leave(box=box, rs=rs):
            rs.data.copy_(box['v'])

        R.append(Reg('ucl_read', _ucl_enter, _ucl_leave))
    br = getattr(model, 'bridge', None)
    if br is not None:
        R.append(_attr_reg('bridge_inject', br, 'depth', False))
    if getattr(model, 'intent_bridge', False) and getattr(model, 'bus_head_proj', None) is not None:
        w = model.bus_head_proj.weight
        box = {}

        def _bus_enter(box=box, w=w):
            box['v'] = w.detach().clone()
            w.data.zero_()

        def _bus_leave(box=box, w=w):
            w.data.copy_(box['v'])

        R.append(Reg('bus_stencil', _bus_enter, _bus_leave))
    if getattr(model, 'explicit_reasoning', False):
        R.append(_attr_reg('reasoning', model, 'reasoning_scale_override', 0.0))
    R.append(_attr_reg('triad', cfg, 'triad_reason', False))
    # per-layer regulators are aggregated into ONE entry (otherwise the
    # round-robin over 24x3 stalls)
    R.append(_multi('vpm', [_attr_reg(f'vpm_L{i}', l, 'variable_precision', False)
                            for i, l in enumerate(model.layers)]))
    R.append(_multi('spec_damp', [_attr_reg(f'sd_L{i}', l, '_damp_on', False)
                                  for i, l in enumerate(model.layers)]))
    R.append(_multi('pen_decay', [_attr_reg(f'pd_L{i}', l, '_pen_decay_on', False)
                                  for i, l in enumerate(model.layers)]))
    if getattr(model, 'logit_cache', None) is not None:
        R.append(_attr_reg('logit_cache', model, 'logit_cache', None))
    if getattr(model, 'memory_bank', None) is not None:
        R.append(_attr_reg('memory_bank', model, 'memory_bank', None))
    box2 = {}

    def _m2v_enter(box=box2):
        box['v'] = (cfg.w_mem2v_scale_min, cfg.w_mem2v_scale_max)
        cfg.w_mem2v_scale_min = cfg.w_mem2v_scale_max = 1.0   # scale = 1 at any diff

    def _m2v_leave(box=box2):
        cfg.w_mem2v_scale_min, cfg.w_mem2v_scale_max = box['v']

    R.append(Reg('mem2v_adapt', _m2v_enter, _m2v_leave))
    return R


class RegulatorLedger:
    def __init__(self, model, cfg, probe: Callable, regs: Optional[List[Reg]] = None,
                 per_round: int = 3, dwell: int = 3):
        """probe(x, y) -> ce_raw float (no_grad, model.eval); batches — 2 fixed pairs."""
        self.regs = regs if regs is not None else build_registry(model, cfg)
        self.probe = probe
        self.batches: List = []            # [(x_a, y_a), (x_b, y_b)] — filled by the trainer
        self.per_round = per_round
        self.dwell = dwell
        self.state: Dict[str, dict] = {r.name: dict(delta_ema=0.0, n=0, off=0, on=0,
                                                    status='PROBATION') for r in self.regs}
        self.sigma_re: Optional[float] = None
        self.sigma_b: Optional[float] = None
        self.ptr = 0

    @staticmethod
    def _probe_state(model):
        """Python state NOT covered by snapshot_runtime_buffers (M8): the logit
        cache rings (lists of tensors on the module) and the head's cadence
        counters. Without it the probe is non-deterministic: the active leg
        pushes into the ring and the next leg reads a different cache
        (sigma_re = 0.57 nat on unclosed state — measured)."""
        ex = {}
        lc = getattr(model, 'logit_cache', None)
        if lc is not None:
            c = lc.cache
            ex['cache'] = (list(c._kv_h), list(c._kv_sent), list(c._sent_lens),
                           list(c._h_scores), list(c._h_lens),
                           {k: list(v) for k, v in c._kv_ms.items()},
                           {k: list(v) for k, v in c._ms_lens.items()},
                           list(getattr(c, '_p_cache', [])), c._position)
        hd = getattr(model, 'lm_head', None)
        if hd is not None and hasattr(hd, '_srl_step'):
            ex['srl_step'] = int(hd._srl_step.item())
        return ex

    @staticmethod
    def _probe_restore(model, ex):
        lc = getattr(model, 'logit_cache', None)
        if lc is not None and 'cache' in ex:
            c = lc.cache
            (c._kv_h, c._kv_sent, c._sent_lens, c._h_scores, c._h_lens,
             _kv_ms, _ms_lens, c._p_cache, c._position) = ex['cache']
            c._kv_ms = {k: _kv_ms[k] for k in c._kv_ms}
            c._ms_lens = {k: _ms_lens[k] for k in c._ms_lens}
        hd = getattr(model, 'lm_head', None)
        if hd is not None and 'srl_step' in ex:
            hd._srl_step.fill_(ex['srl_step'])

    @staticmethod
    def _restore_all(model, snap, ex):
        RegulatorLedger._probe_restore(model, ex)
        model.restore_runtime_buffers(snap)

    def _one(self, model, reg: Reg):
        # IMPORTANT: restore between EVERY leg — the probe forward itself mutates
        # buffers (the _bus_rms EMA, _intent_stream, the mirror caches): without
        # a per-leg restore sigma_re = 0.57 nat (measured), with it — fp noise.
        snap = model.snapshot_runtime_buffers()
        ex = self._probe_state(model)
        try:
            ce_a = self.probe(*self.batches[0])
            self._restore_all(model, snap, ex)
            ce_r = self.probe(*self.batches[0])     # rerun: the determinism noise (~0)
            self._restore_all(model, snap, ex)
            ce_b = self.probe(*self.batches[1])     # batch swap: the natural CE spread
            self._restore_all(model, snap, ex)
            reg.identity()
            try:
                ce_off = self.probe(*self.batches[0])
            finally:
                reg.restore()
        finally:
            self._restore_all(model, snap, ex)
        return ce_off - ce_a, abs(ce_r - ce_a), abs(ce_b - ce_a)

    def measure_round(self, model) -> Dict[str, float]:
        out = {}
        for _ in range(min(self.per_round, len(self.regs))):
            reg = self.regs[self.ptr % len(self.regs)]
            self.ptr += 1
            d, s_re, s_b = self._one(model, reg)
            self.sigma_re = s_re if self.sigma_re is None else 0.9 * self.sigma_re + 0.1 * s_re
            self.sigma_b = s_b if self.sigma_b is None else 0.9 * self.sigma_b + 0.1 * s_b
            st = self.state[reg.name]
            st['delta_ema'] = d if st['n'] == 0 else 0.8 * st['delta_ema'] + 0.2 * d
            st['n'] += 1
            # eps_off: significant against the RUN noise (2*sigma_re) AND against
            # 1% of the natural batch CE spread (0.01*sigma_b) — a regulator whose
            # whole effect is below 1% of the batch variability is a retirement
            # candidate (a decision threshold, not a measurement one).
            eps_off = max(2.0 * (self.sigma_re or 0.0),
                          0.01 * (self.sigma_b or 0.0), 1e-4)
            if st['delta_ema'] < eps_off and st['n'] >= self.dwell:
                st['off'] += 1
                st['on'] = 0
            elif st['delta_ema'] > 2.0 * eps_off:
                st['on'] += 1
                st['off'] = 0
            st['status'] = ('DORMANT' if st['off'] >= self.dwell else
                            'ACTIVE' if st['on'] >= 1 else 'PROBATION')
            out[reg.name] = round(st['delta_ema'], 5)
        return out

    def suggestions(self) -> List[str]:
        return sorted(k for k, v in self.state.items() if v['status'] == 'DORMANT')

    def state_dict(self) -> dict:
        return {'state': self.state, 'ptr': self.ptr,
                'sigma_re': self.sigma_re, 'sigma_b': self.sigma_b}

    def load_state_dict(self, sd: Optional[dict]) -> None:
        if not sd:
            return
        for k, v in (sd.get('state') or {}).items():
            if k in self.state:
                self.state[k].update(v)
        self.ptr = int(sd.get('ptr', 0))
        self.sigma_re = sd.get('sigma_re')
        self.sigma_b = sd.get('sigma_b')
