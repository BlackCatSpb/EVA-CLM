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
        conf_init: float = 0.5,    # EVA-Ai ConceptMiner hypothesis init
        conf_step: float = 0.05,   # per re-observation
        confirm: float = 0.75,     # EVA-Ai: confirmed
        archive: float = 0.25,     # EVA-Ai: archived
        decay: float = 0.999,      # per-step confidence decay
        ema: float = 0.05,         # direction EMA toward the observed residual
        max_observe: int = 32,     # per-call position budget (the Python loop)
    ) -> None:
        super().__init__()
        self.n_slots = int(n_slots)
        self.D = int(D)
        self.merge = float(merge)
        self.conf_init = float(conf_init)
        self.conf_step = float(conf_step)
        self.confirm = float(confirm)
        self.archive = float(archive)
        self.decay_rate = float(decay)
        self.ema = float(ema)
        self.max_observe = int(max_observe)
        self.register_buffer('directions', torch.zeros(self.n_slots, self.D))
        self.register_buffer('confidence', torch.zeros(self.n_slots))
        self.register_buffer('count', torch.zeros(self.n_slots, dtype=torch.long))
        self.register_buffer('filled', torch.zeros(self.n_slots, dtype=torch.bool))
        self.register_buffer('_births', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_obs', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def decay(self) -> None:
        """Once per training forward: stale phantoms fade toward archival."""
        self.confidence.mul_(self.decay_rate)

    @torch.no_grad()
    def observe(self, e_l: torch.Tensor, ell: torch.Tensor, threshold: float) -> int:
        """Accumulate lacuna residuals above `threshold`. Returns the number of
        positions accepted (0 when the lacuna is quiet)."""
        E = e_l.reshape(-1, e_l.shape[-1]).float()
        L = ell.reshape(-1).float()
        sel = L > float(threshold)
        n_sel = int(sel.sum())
        if n_sel == 0:
            return 0
        E = E[sel]
        if E.shape[0] > self.max_observe:                      # a random budget
            pick = torch.randperm(E.shape[0], device=E.device)[:self.max_observe]
            E = E[pick]
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
            else:
                free = (~self.filled).nonzero()
                i = int(free[0]) if free.numel() else int(self.confidence.argmin())
                self.directions[i].copy_(E[j])
                self.confidence[i] = self.conf_init
                self.count[i] = 1
                self.filled[i] = True
                self._births += 1
            accepted += 1
        self._obs += 1
        # EVA-Ai lifecycle: archive the faded (after a grace period)
        dead = self.filled & (self.confidence < self.archive) & (self.count > 3)
        if bool(dead.any()):
            self.filled[dead] = False
            self.confidence[dead] = 0.0
            self.directions[dead] = 0.0
            self.count[dead] = 0
        return accepted

    @torch.no_grad()
    def stats(self) -> dict:
        f = self.filled
        n = int(f.sum())
        conf = float(self.confidence[f].mean()) if n else 0.0
        return {
            'phantoms': n,
            'confirmed': int((f & (self.confidence >= self.confirm)).sum()),
            'conf': conf,
            'births': int(self._births),
            'obs': int(self._obs),
        }

    @torch.no_grad()
    def confirmed_directions(self) -> torch.Tensor:
        """(n, D) unit directions of the confirmed phantoms (M55 candidates)."""
        m = self.filled & (self.confidence >= self.confirm)
        if not bool(m.any()):
            return self.directions.new_zeros(0, self.D)
        return F.normalize(self.directions[m], dim=-1)
