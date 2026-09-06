"""EVA: mlp module."""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EVAConfig


class GroupedMLP(nn.Module):
    """Grouped bottleneck MLP with per-group expansion (SwiGLU optional).

    Instead of D -> D -> D (rank-bounded by D), splits D into G groups
    and gives each group internal expansion (d -> expand*d -> d).
    Total rank still <= D, but each group learns richer features
    within its d-dim subspace.

    G=32, d=128, expand=4 -> 4x per-group expansion.
    With SwiGLU: gate and up projections both d -> expand*d, down expand*d -> d.

    Args:
        D: Hidden dimension.
        expand: Expansion factor for inner dimension.
        groups: Number of groups.
        swiglu: Whether to use SwiGLU activation.
        gate_b_init: Initial value for the mirror-conditioned gate bias.
    """

    def __init__(
        self,
        D: int,
        expand: int,
        groups: int,
        swiglu: bool = True,
        gate_b_init: float = 0.25,
    ) -> None:
        super().__init__()
        assert D % groups == 0
        self.D: int = D
        self.G: int = groups
        self.d: int = D // groups
        d: int = self.d
        e: int = expand
        self.swiglu: bool = swiglu
        if swiglu:
            hidden: int = e * d
            up_std: float = (2.0 / (d + hidden)) ** 0.5
            down_std: float = (2.0 / (hidden + d)) ** 0.5
            self.W_gate: nn.Parameter = nn.Parameter(torch.randn(groups, d, hidden) * up_std)
            self.W_up: nn.Parameter = nn.Parameter(torch.randn(groups, d, hidden) * up_std)
            self.W_down: nn.Parameter = nn.Parameter(torch.randn(groups, hidden, d) * down_std)
        else:
            up_std = (2.0 / (d + e * d)) ** 0.5
            down_std = (2.0 / (e * d + d)) ** 0.5
            self.W_up = nn.Parameter(torch.randn(groups, d, e * d) * up_std)
            self.W_down = nn.Parameter(torch.randn(groups, e * d, d) * down_std)
        self.norm_w: nn.Parameter = nn.Parameter(torch.ones(D))
        self.mlp_gate_a: nn.Parameter = nn.Parameter(torch.ones(1))
        self.mlp_gate_b: nn.Parameter = nn.Parameter(torch.full((1,), float(gate_b_init)))

    def forward(
        self,
        h: torch.Tensor,
        mirror_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through grouped MLP.

        Args:
            h: (B, L, D) input tensor.
            mirror_gate: (B, L, G) optional mirror conditioning signal.

        Returns:
            output: (B, L, D) transformed tensor.
        """
        B: int
        L: int
        D: int
        B, L, D = h.shape
        h = self.norm_w * h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-7)
        h = h.reshape(B, L, self.G, self.d)
        BL: int = B * L
        hg: torch.Tensor = h.permute(2, 0, 1, 3).reshape(self.G, BL, self.d)
        if self.swiglu:
            gate: torch.Tensor = F.silu(torch.matmul(hg, self.W_gate))
            if mirror_gate is not None:
                mg: torch.Tensor = mirror_gate.float()
                if mg.dim() == 3:
                    mg = mg.permute(2, 0, 1).reshape(self.G, BL, 1)
                else:
                    mg = mg.reshape(self.G, BL, 1)
                gate = gate * (self.mlp_gate_a + self.mlp_gate_b * mg)
            up: torch.Tensor = torch.matmul(hg, self.W_up)
            hf: torch.Tensor = (gate * up).permute(1, 0, 2).reshape(B, L, self.G, -1)
        else:
            h = F.silu(torch.matmul(hg, self.W_up))
            hf = h.permute(1, 0, 2).reshape(B, L, self.G, -1)
        hg2: torch.Tensor = hf.permute(2, 0, 1, 3).reshape(self.G, BL, -1)
        h = torch.matmul(hg2, self.W_down)
        h = h.permute(1, 0, 2).view(B, L, self.G, self.d)
        self._cached_group_out: torch.Tensor = h
        return h.reshape(B, L, D)
