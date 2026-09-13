"""M25: the floored scan must be backward-finite at production L0 values.

M20's fp32 argument covered forward range but not the reciprocal scan's
backward conditioning. A finite step-0 forward must not be allowed to poison
the convolution parameters on optimizer.step().
"""
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adaptation import GradientClipper, nonfinite_gradient_names  # noqa: E402
from core.block import _scan_chunk, _scan_floor_needs_fp64  # noqa: E402


def test_fast_floor_scan_backward_is_finite():
    torch.manual_seed(0)
    d_s = torch.tensor([0.3997, 0.7951, 0.9443, 0.9858])
    decay = (d_s.view(1, 1, 4, 1) * torch.rand(1, 32, 4, 256)).clamp(0.01, 1.0)
    b = torch.randn(1, 32, 4, 256, requires_grad=True)
    decay = decay.requires_grad_()
    floor_log = (2.0 * d_s.log()).view(1, 1, 4, 1)

    out, final, _ = _scan_chunk(b, decay, floor_log=floor_log)
    (out.square().mean() + final.square().mean()).backward()

    assert _scan_floor_needs_fp64(floor_log, 32)
    assert torch.isfinite(out).all()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(decay.grad).all()


def test_deep_floor_keeps_fp32_scan_path():
    d_s = torch.tensor([0.70, 0.90, 0.99, 0.9993])
    floor_log = (2.0 * d_s.log()).view(1, 1, 4, 1)
    assert not _scan_floor_needs_fp64(floor_log, 32)


def test_agc_does_not_turn_inf_gradient_into_nan():
    model = nn.Linear(4, 1)
    model.weight.grad = torch.full_like(model.weight, float('inf'))
    assert nonfinite_gradient_names(model) == ['weight']

    GradientClipper(c=0.1).clip(model.parameters())
    assert torch.isinf(model.weight.grad).all()
    assert not torch.isnan(model.weight.grad).any()
