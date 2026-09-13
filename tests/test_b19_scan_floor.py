"""B19: the log-space scan floor equals the external maximum(decay, d_s^k)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.block import _scan_chunk   # noqa: E402


def test_log_floor_equals_external_max():
    torch.manual_seed(0)
    B, CH, S, Dh = 1, 16, 4, 64
    d_s = torch.tensor([0.70, 0.90, 0.99, 0.9993])          # per-scale nominal
    decay = (d_s.view(1, 1, S, 1) * torch.rand(B, CH, S, Dh) * 1.4).clamp(0.01, 1.0)
    k = 2.0
    floored = torch.maximum(decay, d_s.view(1, 1, S, 1).pow(k))
    # b differs -> recompute with SAME b
    b = torch.randn(B, CH, S, Dh)
    a1 = _scan_chunk(b, decay, floor_log=(k * d_s.clamp(min=1e-6).log()).view(1, 1, S, 1))[1]
    a2 = _scan_chunk(b, floored)[1]
    rel = float((a1 - a2).norm() / a2.norm())
    # external max is fp32, the in-scan floor is fp64 — agree to fp32 round-off
    assert rel < 1e-5, f'log floor != external max: rel {rel:.2e}'
