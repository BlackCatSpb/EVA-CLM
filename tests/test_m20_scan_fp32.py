"""M20: under an active ladder floor the chunk scan runs fp32 (legacy None-floored
path stays fp64 exactly as locked by test_scan_exactness)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.block import _scan_chunk   # noqa: E402


def test_floored_fp32_matches_fp64_reference():
    torch.manual_seed(0)
    B, CH, S, Dh = 1, 32, 4, 128
    d_s = torch.tensor([0.70, 0.90, 0.99, 0.9993])
    decay = (d_s.view(1, 1, S, 1) * torch.rand(B, CH, S, Dh) * 1.4).clamp(0.01, 1.0)
    floored = torch.maximum(decay, d_s.view(1, 1, S, 1).pow(2.0))
    b = torch.randn(B, CH, S, Dh)
    fl32 = (2.0 * d_s.clamp(min=1e-6).log()).view(1, 1, S, 1)
    a_fp32 = _scan_chunk(b, decay, floor_log=fl32)[0]
    a_ref = _scan_chunk(b, floored)[0]              # fp64 path, exact floored
    rel = float((a_fp32 - a_ref).norm() / a_ref.norm())
    assert rel < 1e-4, f'fp32-floored diverges from fp64 reference: {rel:.2e}'


def test_legacy_path_still_fp64():
    # floor_log=None keeps the M3 exactness contract (locked separately);
    # here only assert the signature behaves (no crash, finite).
    torch.manual_seed(1)
    b = torch.randn(1, 8, 2, 16)
    d = torch.rand(1, 8, 2, 16).clamp(0.01, 1.0)
    out = _scan_chunk(b, d)[0]
    assert torch.isfinite(out).all()
