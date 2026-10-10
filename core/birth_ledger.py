"""P3-2: BirthLedger + the MDL price of birth.

Newborn structures (a grown phantom bit, a UCL concept born from a confirmed
phantom) are entered into a ledger with a price `r` (the number of state
magnitudes added) and measured COUNTERFACTUALLY by the same paired probe:

    delta_i = CE(without newborn i) − CE(with all)      [nat/token]
    gain    = delta_i * N_tok(i)                        [nat total since birth]
    cost    = lam * r_i * ln N_tok(i)                   [the BIC price, lam=1]
    verdict = gain − cost

Arithmetic (the project's style — derived numbers): a phantom bit has
r = D + K = 2560 + 64 = 2624; over 2000 steps x 384 tokens = 768k tokens,
cost ~ 2624*ln(768k) ~ 35.6k nat => the break-even ~ 46 millinats/token.
A UCL concept: r = D + bridge_dim = 2816, break-even ~ 49 millinats/token.

Two consecutive negative verdicts => archive + blacklist the direction
(cos > 0.9 forbids re-birth until the cooldown expires). The blacklist is the
main MDL mechanism: it kills the birth/archive churn of the same direction
(the M64 bank fix addressed the same failure at the bank level).
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from typing import List, Optional


class BirthLedger:
    def __init__(self, lam: float = 1.0, horizon_steps: int = 2000,
                 dwell: int = 2, blacklist_steps: int = 20000):
        self.lam = lam
        self.horizon = horizon_steps
        self.dwell = dwell
        self.bl_steps = blacklist_steps
        self.entries: List[dict] = []     # kind, step, d (cpu unit), r, fails, retired
        self.blacklist: List[dict] = []   # d, until_step

    # ─── registration ───
    def record(self, kind: str, direction: torch.Tensor, step: int, r: int,
               slot: Optional[int] = None) -> None:
        """The slot/row MUST be fixed at birth: UCL slots drift between record
        and measure (concept writes happen in eval too — "inference = training"),
        and cos-rematching later loses the newborn (measured in the smoke test).
        The fallback matching is kept for old records only."""
        d = F.normalize(direction.detach().float().cpu().reshape(-1), dim=-1)
        self.entries.append(dict(kind=kind, step=int(step), d=d, r=int(r),
                                 slot=slot, fails=0, retired=False, verdicts=[]))

    def allow_birth(self, direction: torch.Tensor, step: int, thr: float = 0.9) -> bool:
        d = F.normalize(direction.detach().float().cpu().reshape(-1), dim=-1)
        for b in self.blacklist:
            if int(b['until']) > int(step) and float(abs(b['d'] @ d)) > thr:
                return False
        return True

    # ─── the MDL verdict ───
    def mdl(self, delta_ce: float, n_tok: float, r: int) -> float:
        gain = delta_ce * n_tok
        cost = self.lam * r * math.log(max(n_tok, 2.0))
        return gain - cost

    # ─── the counterfactual excision (context manager) ───
    def _locate(self, model, e: dict):
        """(object, index) of the newborn: the recorded slot, else argmax|cos|
        with NO hard threshold (slots drift; S is small, argmax is stable)."""
        head = getattr(model, 'lm_head', None)
        ucl = getattr(model, 'concept_layer', None)
        if e['kind'] == 'phantom_bit' and head is not None and head.Kp > 0:
            kp = int(head._kp_active.item())
            if e.get('slot') is not None and e['slot'] < kp:
                return head, int(e['slot'])
            if kp == 0:
                return None, None
            sim = (head.phantom_basis.data[:kp].float()
                   @ e['d'].to(head.phantom_basis.device)).abs()
            return head, int(sim.argmax())
        if e['kind'] == 'ucl' and ucl is not None:
            if e.get('slot') is not None:
                return ucl, int(e['slot'])
            sim = (F.normalize(ucl.concept_vals.detach().float(), dim=-1)
                   @ e['d'].to(ucl.concept_vals.device)).abs()
            return ucl, int(sim.argmax())
        return None, None

    def _excise(self, model, e: dict):
        obj, idx = self._locate(model, e)
        if obj is None:
            return None
        if e['kind'] == 'phantom_bit':
            col = obj.phantom_mix.data[:, idx].clone()
            obj.phantom_mix.data[:, idx].zero_()
            return lambda: obj.phantom_mix.data[:, idx].copy_(col)
        row = obj.concept_vals.data[idx].clone()
        obj.concept_vals.data[idx].zero_()          # attention to a dead slot stays

        def undo():                                  # — conservative: understates
            obj.concept_vals.data[idx].copy_(row)    # gain, never overstates it
        return undo

    # ─── the measurement (called by the trainer next to the RegulatorLedger) ───
    @torch.no_grad()
    def measure(self, model, probe, batch, step: int, tokens_per_step: int):
        try:
            from .regulator_ledger import RegulatorLedger as _RL
        except ImportError:                      # standalone use
            from regulator_ledger import RegulatorLedger as _RL
        out = {}
        for e in self.entries:
            if e['retired'] or step - e['step'] < self.horizon:
                continue
            # Блок 1: forward-граф-кэши не нужны для rollback пробы (они
            # write-before-read) и их клонирование стоило бы сотни МБ/слой
            snap = model.snapshot_runtime_buffers(include_caches=False)
            ex = _RL._probe_state(model)
            undo = None
            try:
                ce_on = probe(*batch)              # all newborns active
                _RL._probe_restore(model, ex)
                model.restore_runtime_buffers(snap)
                undo = self._excise(model, e)      # excise newborn i
                if undo is None:
                    continue                       # the direction is gone — skip
                ce_off = probe(*batch)
                undo()
                undo = None
            finally:
                if undo is not None:
                    undo()                         # exception safety
                _RL._probe_restore(model, ex)
                model.restore_runtime_buffers(snap)
            n_tok = (step - e['step']) * tokens_per_step
            v = self.mdl(ce_off - ce_on, n_tok, e['r'])
            e['verdicts'].append(round(v, 1))
            if v < 0:
                e['fails'] += 1
            else:
                e['fails'] = 0
            if e['fails'] >= self.dwell:
                e['retired'] = True
                self.blacklist.append(dict(d=e['d'], until=step + self.bl_steps))
                self._retire(model, e)
            out[e['kind']] = out.get(e['kind'], 0) + 1
        return out

    def _retire(self, model, e: dict) -> None:
        obj, idx = self._locate(model, e)
        if obj is None:
            return
        if e['kind'] == 'phantom_bit':
            obj.phantom_mix.data[:, idx].zero_()   # the channel is off forever
            obj.phantom_basis.data[idx].zero_()
            e['slot'] = idx                        # the capacity row is reusable
        else:
            obj.concept_vals.data[idx].zero_()
            obj.concept_keys.data[idx].zero_()
            obj.concept_count[idx] = 0
            obj.concept_confidence[idx] = 0        # the slot is "empty" again

    def free_rows(self, kind: str) -> List[int]:
        return [e.get('slot', -1) for e in self.entries
                if e['retired'] and e['kind'] == kind and e.get('slot', -1) >= 0]

    def stats(self) -> dict:
        return dict(births=len(self.entries),
                    retired=sum(1 for e in self.entries if e['retired']),
                    blacklisted=len(self.blacklist))

    def median_verdict(self) -> Optional[float]:
        """EXT §7.4: медианный вердикт (нат) по всем записанным — контекст для
        «retired=N»: без него число пенсий нечитаемо."""
        vs = [v for e in self.entries for v in e.get('verdicts', [])]
        if not vs:
            return None
        vs = sorted(vs)
        return float(vs[len(vs) // 2])

    def state_dict(self) -> dict:
        # EXT §7.2: 'd' — ОДИН стек-тензор вместо list-of-lists на каждую запись
        # (10.5k записей × 2560 float в Python-списках = +225MB к чекпоинту).
        _meta = [{k: v for k, v in e.items() if k != 'd'} for e in self.entries]
        _bl_meta = [{k: v for k, v in b.items() if k != 'd'} for b in self.blacklist]
        try:
            _d = torch.stack([e['d'] for e in self.entries]) if self.entries else None
        except RuntimeError:                       # разные формы — легаси-путь
            # M65-opt: не молчать — легаси-формат раздувает чекпоинт, и это
            # должно быть видно (иначе тихо теряется весь выигрыш §7.2).
            warnings.warn('BirthLedger.state_dict: d-тензоры разной формы — '
                          'легаси-путь (list-of-lists)', RuntimeWarning, stacklevel=2)
            _meta = [dict(m, d=torch.as_tensor(e['d']).tolist())
                     for m, e in zip(_meta, self.entries)]
            _d = None
        try:
            _bd = (torch.stack([b['d'] for b in self.blacklist])
                   if self.blacklist else None)
        except RuntimeError:
            warnings.warn('BirthLedger.state_dict: blacklist d разной формы — '
                          'легаси-путь', RuntimeWarning, stacklevel=2)
            _bl_meta = [dict(m, d=torch.as_tensor(b['d']).tolist())
                        for m, b in zip(_bl_meta, self.blacklist)]
            _bd = None
        return dict(entries=_meta, d_stack=_d,
                    blacklist=_bl_meta, bl_d_stack=_bd)

    def load_state_dict(self, sd: Optional[dict]) -> None:
        if not sd:
            return
        _entries = sd.get('entries', [])
        _ds = sd.get('d_stack', None)
        if _ds is not None and len(_entries) == len(_ds):
            self.entries = [dict(e, d=_ds[i]) for i, e in enumerate(_entries)]
        else:                                      # легаси: d внутри записи
            self.entries = [dict(e, d=torch.as_tensor(e['d'])) for e in _entries]
        _bl = sd.get('blacklist', [])
        _bds = sd.get('bl_d_stack', None)
        if _bds is not None and len(_bl) == len(_bds):
            self.blacklist = [dict(b, d=_bds[i]) for i, b in enumerate(_bl)]
        else:
            self.blacklist = [dict(b, d=torch.as_tensor(b['d'])) for b in _bl]
