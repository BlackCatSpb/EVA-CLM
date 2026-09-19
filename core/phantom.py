"""M54: the phantom-concept bank — the EVA-Ai lacuna lifecycle, transplanted
from the knowledge graph to hidden states.

The head's lacuna (e_l, ell) is the part of the state that NO known bit can see
(orthogonal to the whole readout). Recurring lacuna directions are candidate NEW
concepts. The bank accumulates them with a cosine-matched EMA, tracks their
recurrence confidence and applies the EVA-Ai lifecycle (confirm 0.75 / archive
0.25 — the ConceptMiner constants). Confirmed phantoms are the candidates for
consolidation into real bits (M55: the head's growth).

All state is in buffers => it rides in the checkpoint. observe() is no_grad and
runs at a cadence; decay() runs every training forward so stale phantoms fade.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhantomBank(nn.Module):
    def __init__(
        self,
        n_slots: int = 16,
        D: int = 2560,
        merge: float = 0.7,        # cosine >= merge -> the same phantom (EVA-Ai dedup 0.7)
        merge_lo: float = 0.2,     # M64 (M63-C): the soft route — cosine >= merge_lo
                                   # takes a similarity-weighted EMA + a partial
                                   # confidence bump. The hard merge=0.7 alone is
                                   # unreachable on D=2560 residuals (measured:
                                   # zero merges in the whole run). R1/R3 review:
                                   # the null best-cos on D=2560 is ~0.06 mean /
                                   # 0.09 max (4e8 pairs) — 0.2 sits between the
                                   # null and the recurring-structure mode (~0.3).
                                   # The real calibration needs the measured
                                   # histogram (stats() now reports the percentiles).
        conf_init: float = 0.5,    # EVA-Ai ConceptMiner hypothesis init
        conf_step: float = 0.05,   # per re-observation
        confirm: float = 0.75,     # EVA-Ai: confirmed
        archive: float = 0.25,     # EVA-Ai: archived
        decay: float = 0.99,       # per-OBSERVE confidence decay (M64: was per-forward).
                                   # Calibrated (R1/R2 review): at the observed
                                   # observed ~0.13 observes/step (checkpoint counters: _obs=822
                                   # over ~6270 active steps) the 0.5 -> 0.25 transition takes
                                   # ~69 observes ~ 526 steps — the confirmation-window
                                   # ballpark for the confirmation window (~200-300
                                   # steps); the old 0.999 gave ~2900 steps.
        ema: float = 0.05,         # direction EMA toward the observed residual
        max_observe: int = 32,     # per-call position budget (the Python loop)
        cycles_before_stable: int = 5,  # T9 (EVA-Ai ConceptMiner): confirmation
                                        # additionally requires >= N observations
                                        # (recurrence gate; 0 = off = old mask)
    ) -> None:
        super().__init__()
        self.n_slots = int(n_slots)
        self.D = int(D)
        self.merge = float(merge)
        self.merge_lo = float(merge_lo)
        self.conf_init = float(conf_init)
        self.conf_step = float(conf_step)
        self.confirm = float(confirm)
        self.archive = float(archive)
        self.decay_rate = float(decay)
        self.ema = float(ema)
        self.max_observe = int(max_observe)
        self.cycles_before_stable = int(cycles_before_stable)
        self.register_buffer('directions', torch.zeros(self.n_slots, self.D))
        self.register_buffer('confidence', torch.zeros(self.n_slots))
        self.register_buffer('count', torch.zeros(self.n_slots, dtype=torch.long))
        self.register_buffer('filled', torch.zeros(self.n_slots, dtype=torch.bool))
        self.register_buffer('coh', torch.zeros(self.n_slots))   # T9: per-slot
                                        # coherence EMA (accepted match cosines)
        self.register_buffer('_births', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_obs', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_merged', torch.zeros(1, dtype=torch.long))    # M64
        self.register_buffer('_dropped', torch.zeros(1, dtype=torch.long))   # M64
        self.register_buffer('_archived', torch.zeros(1, dtype=torch.long))  # M64
        self.register_buffer('_cos_ring', torch.zeros(64))                   # M64: telemetry
        self.register_buffer('_cos_ptr', torch.zeros(1, dtype=torch.long))
        # T9 (EVA-Ai audit-log): ring of recent lifecycle events —
        # (event_code, slot, conf, cos). Codes: 0 birth, 1 merge_hard,
        # 2 merge_soft, 3 drop, 4 archive. Rides in the checkpoint.
        self.register_buffer('_audit_ring', torch.zeros(64, 4))
        self.register_buffer('_audit_ptr', torch.zeros(1, dtype=torch.long))

    def _audit(self, event: int, slot: int, conf: float, cos: float) -> None:
        _k = int(self._audit_ptr.item()) % self._audit_ring.shape[0]
        self._audit_ring[_k, 0] = float(event)
        self._audit_ring[_k, 1] = float(slot)
        self._audit_ring[_k, 2] = float(conf)
        self._audit_ring[_k, 3] = float(cos)
        self._audit_ptr += 1

    @torch.no_grad()
    def decay(self) -> None:
        """Stale phantoms fade toward archival.

        M64 (M63-C): called ONCE PER OBSERVE CALL, not per training forward.
        The old per-forward call made the effective rate depend on the number
        of head forwards per step (measured ~8 with the reasoning/knowledge
        passes), so the slot life (~100 steps) sat BELOW the confirmation time
        (~200 steps) BY ARITHMETIC — no phantom could ever be confirmed
        (measured: max confidence = conf_init for the whole run, i.e. not a
        single merge). Per-observe semantics is deterministic w.r.t. the
        cadence (head_phantom_every) and the arithmetic is now checkable.
        """
        self.confidence.mul_(self.decay_rate)

    @torch.no_grad()
    def observe(self, e_l: torch.Tensor, ell: torch.Tensor, threshold: float) -> int:
        """Accumulate lacuna residuals above `threshold`. Returns the number of
        positions accepted (0 when the lacuna is quiet)."""
        self.decay()   # M64: the fade is part of the observe cadence (see decay())
        E = e_l.reshape(-1, e_l.shape[-1]).float()
        L = ell.reshape(-1).float()
        sel = L > float(threshold)
        n_sel = int(sel.sum())
        if n_sel == 0:
            return 0
        E = E[sel]
        if E.shape[0] > self.max_observe:                      # a deterministic budget
            E = E[:self.max_observe]                           # (no global-RNG draw)
        En = F.normalize(E, dim=-1)
        with torch.no_grad():
            Dn = F.normalize(self.directions, dim=-1)
            sim = En @ Dn.T                                    # (N, slots)
            best, idx = sim.max(dim=-1)
        accepted = 0
        for j in range(E.shape[0]):
            s = float(best[j])
            i = int(idx[j])
            if s >= self.merge and bool(self.filled[i]):
                self.directions[i].mul_(1.0 - self.ema).add_(E[j], alpha=self.ema)
                self.confidence[i] = min(1.0, float(self.confidence[i]) + self.conf_step)
                self.count[i] += 1
                self.coh[i] = 0.9 * float(self.coh[i]) + 0.1 * s   # T9
                self._merged += 1
                self._audit(1, i, float(self.confidence[i]), s)
            elif s >= self.merge_lo and bool(self.filled[i]):
                # M64 (M63-C): the SOFT route. The hard merge=0.7 is
                # unreachable on D=2560 lacuna residuals (measured: max conf =
                # conf_init -> zero merges in the whole run), so before this
                # every unmatched observation EVICTED a slot: the bank was a
                # snapshot of the last <=16 residuals, not a memory. A
                # similarity-weighted EMA lets a recurring direction accumulate
                # confidence across observations (the confirmation path).
                w = (s - self.merge_lo) / max(1e-6, self.merge - self.merge_lo)
                self.directions[i].mul_(1.0 - self.ema * w).add_(E[j], alpha=self.ema * w)
                self.confidence[i] = min(1.0, float(self.confidence[i]) + self.conf_step * w)
                self.count[i] += 1
                self.coh[i] = 0.9 * float(self.coh[i]) + 0.1 * s   # T9
                self._merged += 1
                self._audit(2, i, float(self.confidence[i]), s)
            else:
                free = (~self.filled).nonzero()
                if free.numel():
                    i = int(free[0])
                    self.directions[i].copy_(E[j])
                    self.confidence[i] = self.conf_init
                    self.count[i] = 1
                    self.coh[i] = s                                # T9: стартовая когерентность
                    self.filled[i] = True
                    self._births += 1
                    self._audit(0, i, self.conf_init, s)
                else:
                    # M64: no free slot and no close match -> DROP (the old
                    # `confidence.argmin()` eviction turned a full bank into a
                    # churn: births ~= observations). R1/R2 review: the drop is
                    # only safe because the archival below is REACHABLE — the
                    # old `count > 3` grace combined with the drop froze the
                    # bank on its first <=16 residuals forever (12/16 slots of
                    # the live checkpoint had count=1 and could never archive).
                    self._dropped += 1
                    self._audit(3, i, float(self.confidence[i]), s)
            accepted += 1
        # the observed best-cos telemetry (R3): the merge_lo calibration needs
        # the real histogram, not a guess — a 64-wide ring of per-observe maxes
        _k = int(self._cos_ptr.item()) % self._cos_ring.numel()
        self._cos_ring[_k] = float(best.max())
        self._cos_ptr += 1
        self._obs += 1
        # EVA-Ai lifecycle: archive the faded. M64/R1/R2: no `count > 3` grace —
        # conf_init (0.5) is above `archive` (0.25), so a fresh slot can never
        # trigger this immediately, and the grace was exactly what made the
        # never-recurring slots immortal (the freeze the review caught).
        dead = self.filled & (self.confidence < self.archive)
        if bool(dead.any()):
            for _i in dead.nonzero().reshape(-1).tolist():
                self._audit(4, int(_i), float(self.confidence[_i]), float(self.coh[_i]))
            self._archived += int(dead.sum())
            self.filled[dead] = False
            self.confidence[dead] = 0.0
            self.coh[dead] = 0.0
            self.directions[dead] = 0.0
            self.count[dead] = 0
        return accepted

    @torch.no_grad()
    def types(self) -> torch.Tensor:
        """T9 (EVA-Ai ConceptMiner, адаптировано): per-slot ТИП (статус
        confirmed — отдельно, см. stats()['stable']).
        0=emerging (count>=10 и coh>=0.3 — частый и когерентный),
        1=ambiguous (coh<0.3 — слабые совпадения), 2=nascent, -1=пусто."""
        t = torch.full((self.n_slots,), -1, dtype=torch.long, device=self.filled.device)
        f = self.filled
        emerging = f & (self.count >= 10) & (self.coh >= 0.3)
        ambiguous = f & ~emerging & (self.coh < 0.3)
        nascent = f & ~emerging & ~ambiguous
        t[emerging], t[ambiguous], t[nascent] = 0, 1, 2
        return t

    @torch.no_grad()
    def priority_directions(self) -> torch.Tensor:
        """T9: 'high priority' (EVA-Ai): conf>0.7 и тип emerging/ambiguous —
        кандидаты консолидации (M55) до полного подтверждения."""
        t = self.types()
        m = self.filled & (self.confidence > 0.7) & ((t == 0) | (t == 1))
        if not bool(m.any()):
            return self.directions.new_zeros(0, self.D)
        return F.normalize(self.directions[m], dim=-1)

    @torch.no_grad()
    def audit_tail(self, n: int = 8) -> list:
        """T9: последние n событий лайфцикла (code, slot, conf, cos)."""
        _cnt = int(min(self._audit_ptr.item(), self._audit_ring.shape[0]))
        if _cnt == 0:
            return []
        _k = int(self._audit_ptr.item()) % self._audit_ring.shape[0]
        idx = [(i % self._audit_ring.shape[0]) for i in range(max(0, _k - min(n, _cnt)), _k)]
        names = {0: 'birth', 1: 'merge_hard', 2: 'merge_soft', 3: 'drop', 4: 'archive'}
        return [{'event': names.get(int(self._audit_ring[i, 0]), '?'),
                 'slot': int(self._audit_ring[i, 1]),
                 'conf': float(self._audit_ring[i, 2]),
                 'cos': float(self._audit_ring[i, 3])} for i in idx]

    @torch.no_grad()
    def stats(self) -> dict:
        f = self.filled
        n = int(f.sum())
        conf = float(self.confidence[f].mean()) if n else 0.0
        # the best-cos telemetry (R3): p50/p90/p99 over the last <=64 observes
        _cnt = int(min(self._cos_ptr.item(), self._cos_ring.numel()))
        if _cnt > 0:
            _h = self._cos_ring[:_cnt].float()
            # the q tensor MUST live on _h's device — the first landing built it
            # on the CPU and CRASHED the live run at step 7425 (quantile() q
            # device check). CPU tests could never catch it.
            _q = torch.quantile(_h, torch.tensor([0.5, 0.9, 0.99],
                                                 device=_h.device, dtype=_h.dtype))
            cos_p50, cos_p90, cos_p99 = (float(_q[0]), float(_q[1]), float(_q[2]))
        else:
            cos_p50 = cos_p90 = cos_p99 = 0.0
        _t = self.types()
        _hi = self.filled & (self.confidence > 0.7) & ((_t == 0) | (_t == 1))
        return {
            'phantoms': n,
            'confirmed': int((f & (self.confidence >= self.confirm)
                              & (self.count >= self.cycles_before_stable)).sum()),
            'conf': conf,
            'births': int(self._births),
            'obs': int(self._obs),
            'merged': int(self._merged),
            'dropped': int(self._dropped),
            'archived': int(self._archived),
            'cos_p50': cos_p50,
            'cos_p90': cos_p90,
            'cos_p99': cos_p99,
            # T9 (EVA-Ai classifier): статус stable + типы + high-priority
            'stable': int((f & (self.confidence >= self.confirm)
                           & (self.count >= self.cycles_before_stable)).sum()),
            'emerging': int((_t == 0).sum()),
            'ambiguous': int((_t == 1).sum()),
            'nascent': int((_t == 2).sum()),
            'high_priority': int(_hi.sum()),
        }

    @torch.no_grad()
    def confirmed_directions(self) -> torch.Tensor:
        """(n, D) unit directions of the confirmed phantoms (M55 candidates).
        T9 (EVA-Ai): confirmation требует и conf>=confirm, и count>=cycles."""
        m = (self.filled & (self.confidence >= self.confirm)
             & (self.count >= self.cycles_before_stable))
        if not bool(m.any()):
            return self.directions.new_zeros(0, self.D)
        return F.normalize(self.directions[m], dim=-1)
