"""Checkpoint migration for backward compatibility.

Handles old checkpoints that carry pre-BottleneckBind keys:
  - bind.W_out: shape mismatch (n_dims*2K -> n_dims*2K+K)
  - bind_coh_gate: missing parameter (coherence gate, legacy)
  - freq_scale: missing parameter (frequency scaling, legacy)

Usage:
    sd, changed = migrate_state_dict(sd, model)

Migration values (audit M10 — honest about what each choice means):
  - W_out: first rows copied, tail zeroed (numeric continuity for the shared
    span; the extra rows start inert)
  - bind_coh_gate = 0.0 — coherence gate closed: the legacy forward had no
    coherence modulation, so 0 disables the new term (exact equivalence)
  - freq_scale = 1.0 — the legacy spiral ran with implicit unit frequency
    scale; NOTE this is NOT the fresh-init value (2*pi, TrajectorySpiralBind
    __init__), so a migrated model keeps the old frequency range instead of
    jumping 6.28x — equivalence with the LEGACY checkpoint, not with a new
    model. Re-anneal if the wider range is desired.
  - freq_scale = 1.0 (legacy frequency scale)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def migrate_state_dict(
    sd: dict[str, torch.Tensor],
    model: nn.Module,
) -> Tuple[dict[str, torch.Tensor], int]:
    """Migrate an old checkpoint's state dict to the current architecture.

    Args:
        sd: State dict from the checkpoint.
        model: Current model (provides target shapes for parameters).

    Returns:
        Tuple of (migrated state dict, number of keys changed).
    """
    changed: int = 0

    # Fix bind.W_out shape mismatch
    for name, p in model.named_parameters():
        if name.endswith('bind.W_out'):
            old: torch.Tensor | None = sd.get(name)
            if old is None:
                continue
            if tuple(old.shape) == tuple(p.shape):
                continue
            new: torch.Tensor = torch.zeros_like(p)
            n: int = min(old.shape[0], p.shape[0])
            new[:n] = old[:n]
            sd[name] = new
            changed += 1

    # Add missing bind_coh_gate and freq_scale
    for name, p in model.named_parameters():
        if name.endswith('bind_coh_gate'):
            if name not in sd:
                sd[name] = torch.zeros_like(p)
                changed += 1
        elif name.endswith('freq_scale'):
            if name not in sd:
                sd[name] = torch.tensor(1.0)
                changed += 1

    return sd, changed
