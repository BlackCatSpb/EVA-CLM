"""M25/M26 locks: the scan is finite in fp32 forward AND backward at the
production fast-floor, without any fp64 graph (the naive reciprocal form was
the step-0 NaN-gradient source that poisoned the live runs)."""
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adaptation import GradientClipper, nonfinite_gradient_names  # noqa: E402
from core.block import _scan_chunk  # noqa: E402


def _floored_case(ds0):
    torch.manual_seed(0)
    d_s = torch.tensor([ds0, 0.7951, 0.9443, 0.9858])
    decay = (d_s.view(1, 1, 4, 1) * torch.rand(1, 32, 4, 256)).clamp(0.01, 1.0)
    decay = decay.requires_grad_()
    b = torch.randn(1, 32, 4, 256, requires_grad=True)
    floor_log = (2.0 * d_s.log()).view(1, 1, 4, 1)
    out, final, _ = _scan_chunk(b, decay, floor_log=floor_log)
    (out.square().mean() + final.square().mean()).backward()
    return out, decay, b


def test_fast_floor_scan_backward_is_finite_and_fp32():
    # the exact production L0 value that broke M20's naive fp32 scan
    out, decay, b = _floored_case(0.3997)
    assert out.dtype == torch.float32, 'floored scan must not promote to fp64'
    assert torch.isfinite(out).all() and torch.isfinite(b.grad).all()
    assert torch.isfinite(decay.grad).all()
    assert decay.grad.dtype == torch.float32


def test_floored_scan_matches_fp64_reference():
    torch.manual_seed(0)
    d_s = torch.tensor([0.3997, 0.7951, 0.9443, 0.9858])
    decay = (d_s.view(1, 1, 4, 1) * torch.rand(1, 32, 4, 256)).clamp(0.01, 1.0)
    floored = torch.maximum(decay, d_s.view(1, 1, 4, 1).pow(2.0))
    b = torch.randn(1, 32, 4, 256)
    floor_log = (2.0 * d_s.log()).view(1, 1, 4, 1)
    new = _scan_chunk(b, decay, floor_log=floor_log)[0]
    ref = _scan_chunk(b, floored)[0]                  # legacy fp64 exact path
    rel = float((new - ref).norm() / ref.norm())
    assert rel < 1e-4, f'tail-referenced fp32 diverges from fp64 ref: {rel:.2e}'


def test_agc_drops_nonfinite_gradients():
    # M28: the loops no longer gate updates; AGC (core.adaptation) is the one
    # place a non-finite gradient may be acted on — it is DROPPED so Adam
    # moments and weights can never be poisoned.
    model = nn.Linear(4, 1)
    good = model.bias.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float('inf'))

    GradientClipper(c=0.1).clip(model.parameters())
    assert model.weight.grad is None
    assert nonfinite_gradient_names(model) == []
