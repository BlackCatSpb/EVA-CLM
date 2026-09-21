"""core/meta_head.py — P4-2: readout истинных внутренних сигналов из h.

n_meta = 6 (v1; все — уже вычисляемые величины, nan = «недоступно на этом
forward», loss маскуется):
  0 ell_rel   — head._last_lacuna_rel          (скаляр → expand)
  1 conflict  — head._last_conflict            (χ голова↔память, скаляр)
  2 pen       — mean по слоям mirror._cached_pred_error_norm   (B,L)
  3 gate_mean — mean по слоям mirror._cached_gate              (B,L)
  4 chi_time  — mean по слоям block._chi_time (P4-3)           (B,L)
  5 h_norm    — ‖h‖ (B,L) — sanity-цель (декодируема тривиально)

Режимы (флаг cfg.meta_head_grad):
  False (default) — h.detach(): чистый зонд, ствол не меняется;
  True  — градиент aux-лосса давит на ствол: представление УЧИТСЯ быть
          самочитаемым («learning to introspect») — отдельная A/B-рука.
Терм идёт через LossBalancer.BYPASS_AUX (прямой backward): его градиент при
grad=False живёт только в параметрах meta_head — align-путь ствола не тронут.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_META = 6


class MetaHead(nn.Module):
    def __init__(self, D: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, hidden), nn.SiLU(),
                                 nn.Linear(hidden, N_META))
        nn.init.zeros_(self.net[-1].weight)      # zero-init ⇒ прогноз 0 на старте
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:      # (B,L,D)->(B,L,N_META)
        return self.net(h)


def _mean_layers(stack, attr, B, L, nan):
    xs = [getattr(l.mirror, attr, None) for l in stack.layers]
    xs = [x for x in xs if isinstance(x, torch.Tensor)
          and tuple(x.shape[:2]) == (B, L)]
    if not xs:
        return nan
    # _cached_gate несёт ось экспертов (B,L,G) — цель «gate_mean»: среднее и по ней
    xs = [x.mean(dim=tuple(range(2, x.dim()))) if x.dim() > 2 else x for x in xs]
    return torch.stack(xs).mean(0).float()


def _mean_layers_blocks(stack, attr, B, L, nan):
    xs = [getattr(l, attr, None) for l in stack.layers]
    xs = [x for x in xs if isinstance(x, torch.Tensor)
          and tuple(x.shape[:2]) == (B, L)]
    if not xs:
        return nan
    return torch.stack(xs).mean(0).float()


def meta_targets(stack, h) -> torch.Tensor:
    """(B,L,N_META) из живых кэшей; nan — сигнал недоступен (SRL/χ выключены)."""
    B, L, D = h.shape
    nan = torch.full((B, L), float('nan'), device=h.device, dtype=torch.float32)
    def _scal(v):
        if v is None:
            return nan
        return torch.full((B, L), float(v), device=h.device, dtype=torch.float32)
    hd = getattr(stack, 'lm_head', None)
    t = torch.stack([
        _scal(getattr(hd, '_last_lacuna_rel', None)),
        _scal(getattr(hd, '_last_conflict', None)),
        _mean_layers(stack, '_cached_pred_error_norm', B, L, nan),
        _mean_layers(stack, '_cached_gate', B, L, nan),
        _mean_layers_blocks(stack, '_chi_time', B, L, nan),
        h.detach().norm(dim=-1).float(),
    ], dim=-1)
    return t


def meta_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    m = torch.isfinite(tgt)
    if not bool(m.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[m], tgt[m])
