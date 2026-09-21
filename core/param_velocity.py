"""P3-4: ParamVelocity — the parameter-efficiency KPI.

Continuous telemetry of "which parameters actually move and what it costs":
the relative displacement ‖Δθ‖/‖θ‖ per module over one log interval, without
full-size copies (a deterministic coordinate subsample, ~7MB on the production
model). The M64.7 measurement ("the readout moved −1.9% over 7315 steps") was a
manual post-mortem — this is the same metric on stream.

Derived whiteboard metric: parameter efficiency =
Δ(ce_raw over the interval) / Σ_buckets vel·numel — the CE drop per unit of
relative parametric mass movement. A rising ratio after P3-1/2/3 is the direct
confirmation of "more efficient parameter use".
"""
from __future__ import annotations

import torch
from typing import Dict


class ParamVelocity:
    def __init__(self, model, frac: float = 0.01, min_coords: int = 64, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        self.idx: Dict[str, torch.Tensor] = {}
        self.snap: Dict[str, torch.Tensor] = {}
        self.vel: Dict[str, float] = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            k = max(min_coords, int(p.numel() * frac))
            self.idx[n] = torch.randperm(p.numel(), generator=g)[:min(k, p.numel())]

    @torch.no_grad()
    def sample(self, model, ema: float = 0.8) -> None:
        """Call once per log interval (after optimizer.step())."""
        for n, p in model.named_parameters():
            i = self.idx.get(n)
            if i is None:
                continue
            v = p.detach().reshape(-1)[i.to(p.device)]
            if n in self.snap:
                den = float(self.snap[n].norm())
                # near-zero-init parameters: the relative velocity is meaningless
                # (the same eps class as the AGC skip at ‖θ‖<1e-3).
                if den > 1e-6:
                    rel = float((v - self.snap[n]).norm()) / den
                    self.vel[n] = (rel if n not in self.vel
                                   else ema * self.vel[n] + (1 - ema) * rel)
            self.snap[n] = v.clone()

    @staticmethod
    def _bucket(n: str) -> str:
        # newborns first: phantom_*/pair_V* live under lm_head/layers too
        if 'phantom' in n or 'pair_V' in n:
            return 'newborn'
        if n.startswith('lm_head'):
            return 'head'
        if n.startswith('embed'):
            return 'embed'
        if '.mirror.' in n:
            return 'mirror'
        if '.mlp.' in n:
            return 'mlp'
        if '.bind.' in n:
            return 'bind'
        if 'bridge' in n or 'intent' in n or 'bus_head' in n:
            return 'bridge'
        if 'logit_cache' in n:
            return 'cache'
        if 'reasoning' in n:
            return 'reasoning'
        if 'concept_layer' in n:
            return 'ucl'
        return 'trunk'

    def report(self) -> Dict[str, float]:
        agg: Dict[str, list] = {}
        for n, v in self.vel.items():
            agg.setdefault(self._bucket(n), []).append(v)
        return {k: round(sum(v) / len(v), 6) for k, v in sorted(agg.items())}

    def state_dict(self) -> dict:
        # the indices are seeded => not carried in the checkpoint
        return {'vel': dict(self.vel)}

    def load_state_dict(self, sd) -> None:
        if not sd:
            return
        self.vel = dict(sd.get('vel') or {})
