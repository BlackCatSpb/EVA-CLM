"""WideBind: mlp module."""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import WideBindConfig

class GroupedMLP(nn.Module):
    """
    Grouped bottleneck MLP with per-group expansion (SwiGLU optional).

    Instead of D → D → D (rank-bounded by D), splits D into G groups
    and gives each group internal expansion (d → expand*d → d).
    Total rank still ≤ D, but each group learns richer features
    within its d-dim subspace.

    G=32, d=128, expand=4 → 4× per-group expansion.
    With SwiGLU: gate and up projections both d → expand*d, down expand*d → d.
    """
    def __init__(self, D, expand, groups, swiglu=True, gate_b_init=0.25):
        super().__init__()
        assert D % groups == 0
        self.D = D
        self.G = groups
        self.d = D // groups
        d = self.d
        e = expand
        self.swiglu = swiglu
        if swiglu:
            hidden = e * d
            up_std = (2.0 / (d + hidden)) ** 0.5
            down_std = (2.0 / (hidden + d)) ** 0.5
            self.W_gate = nn.Parameter(torch.randn(groups, d, hidden) * up_std)
            self.W_up = nn.Parameter(torch.randn(groups, d, hidden) * up_std)
            self.W_down = nn.Parameter(torch.randn(groups, hidden, d) * down_std)
        else:
            up_std = (2.0 / (d + e * d)) ** 0.5
            down_std = (2.0 / (e * d + d)) ** 0.5
            self.W_up = nn.Parameter(torch.randn(groups, d, e * d) * up_std)
            self.W_down = nn.Parameter(torch.randn(groups, e * d, d) * down_std)
        self.norm_w = nn.Parameter(torch.ones(D))
        # ─── Variant A: mirror-conditioned SwiGLU ───
        # Когнитивное (зеркало) управление воротами ассоциативного MLP.
        # gate = silu(W_gate·h) · (a + b·mirror_gate). init a=1, b=0 → чистый
        # SwiGLU при init; b растёт, обучая MLP откликаться на сигнал зеркала.
        self.mlp_gate_a = nn.Parameter(torch.ones(1))
        self.mlp_gate_b = nn.Parameter(torch.full((1,), float(gate_b_init)))

    def forward(self, h, mirror_gate=None):
        B, L, D = h.shape
        h = self.norm_w * h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-7)
        h = h.reshape(B, L, self.G, self.d)
        BL = B * L
        # Батч-матмул вместо einsum: под autocast einsum(fp16) падает на
        # некоторых GPU (CUBLAS_STATUS_NOT_SUPPORTED), matmul — нет.
        hg = h.permute(2, 0, 1, 3).reshape(self.G, BL, self.d)  # (G, BL, d)
        if self.swiglu:
            gate = F.silu(torch.matmul(hg, self.W_gate))  # (G, BL, f)
            if mirror_gate is not None:
                # mirror_gate: (B, L, G) → (G, BL, 1) — когнитивное управление
                mg = mirror_gate.float()
                if mg.dim() == 3:
                    mg = mg.permute(2, 0, 1).reshape(self.G, BL, 1)
                else:
                    mg = mg.reshape(self.G, BL, 1)
                gate = gate * (self.mlp_gate_a + self.mlp_gate_b * mg)
            up = torch.matmul(hg, self.W_up)              # (G, BL, f)
            hf = (gate * up).permute(1, 0, 2).reshape(B, L, self.G, -1)
        else:
            h = F.silu(torch.matmul(hg, self.W_up))       # (G, BL, f)
            hf = h.permute(1, 0, 2).reshape(B, L, self.G, -1)
        hg2 = hf.permute(2, 0, 1, 3).reshape(self.G, BL, -1)  # (G, BL, f)
        h = torch.matmul(hg2, self.W_down)                # (G, BL, d)
        h = h.permute(1, 0, 2).view(B, L, self.G, self.d)
        self._cached_group_out = h  # (B, L, G, d)
        return h.reshape(B, L, D)


# ─── BottleneckBind (Fibonacci-twisted) ────────────────────────────────
