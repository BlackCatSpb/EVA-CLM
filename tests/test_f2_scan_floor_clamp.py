"""F2 (math audit): the tail-referenced fp32 scan is finite only while
CHUNK*|floor_log| < ln(FLT_MAX) = 88.7.

tau_s comes from a learnable parameter WITHOUT clamps, so a drift toward fast
forgetting (tau_s < 32*k/88.7 ≈ 0.72 at k=2) used to push e^{A_t-A_last} past
fp32 -> inf*0 = NaN (measured in the audit: tau_s=0.3 -> NaN). The block now
clamps its floor at -88.7/CHUNK; this test pins both the NaN regime and the fix,
and that the healthy regime is untouched.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.block import _scan_chunk, _SCAN_LOG_MAX   # noqa: E402


def _floors(tau_s, k=2.0, chunk=32):
    d_s = math.exp(-1.0 / tau_s)
    raw = torch.tensor([k * math.log(d_s)], dtype=torch.float64).view(1, 1, 1, 1)
    return d_s, raw, raw.clamp_min(-_SCAN_LOG_MAX / chunk)


def test_f2_deep_floor_is_nan_without_the_clamp():
    torch.manual_seed(0)
    CH, S, Dh = 32, 1, 64
    d_s, raw, _ = _floors(0.3)
    decay = torch.full((1, CH, S, Dh), d_s).clamp(0.01, 1.0)
    b = torch.randn(1, CH, S, Dh)
    out = _scan_chunk(b, decay, floor_log=raw)[0]
    assert not torch.isfinite(out).all(), 'the F2 NaN regime must reproduce'


def test_f2_clamped_floor_keeps_the_scan_finite():
    torch.manual_seed(0)
    CH, S, Dh = 32, 1, 64
    d_s, _, clamped = _floors(0.3)
    decay = torch.full((1, CH, S, Dh), d_s).clamp(0.01, 1.0)
    b = torch.randn(1, CH, S, Dh)
    out = _scan_chunk(b, decay, floor_log=clamped)[0]
    assert torch.isfinite(out).all()


def test_f2_healthy_tau_is_untouched():
    # tau_s = 8 (the live ladder's fastest rung): the clamp is a no-op
    _, raw, clamped = _floors(8.0)
    assert torch.equal(raw, clamped)
    # ... and above the audit's safety threshold 32*k/88.7 = 0.7215
    _, raw75, clamped75 = _floors(0.75)
    assert torch.equal(raw75, clamped75)
