"""M50 lock: the stream fuse caps magnitude with an O(1) Jacobian and the
norms survive 1e20-scale inputs (the 2970 explosion post-mortem)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.stack import _stream_cap  # noqa: E402


def test_fuse_caps_values_keeps_gradient():
    # realistic regime: a block output 10x above the cap — the fuse rescales
    # and the Jacobian to the producer stays O(cap/m) ~ 0.1, i.e. usable
    x = torch.randn(1, 4, 8) * 1e4
    x.requires_grad_()
    y = _stream_cap(x, 1e3)
    assert float(y.abs().amax()) <= 1e3 + 1e-3
    y.sum().backward()
    assert float(x.grad.abs().max()) > 1e-3, 'fuse killed the gradient'


def test_fuse_passthrough_below_cap_and_disable():
    x = torch.randn(1, 4, 8) * 10.0
    assert torch.allclose(_stream_cap(x, 1e3), x)
    assert torch.allclose(_stream_cap(x, 0.0), x)


def test_overflow_safe_norm_survives_1e20():
    x = torch.full((1, 4, 8), 1e20)
    m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    xs = x / m
    y = xs * torch.rsqrt(xs.pow(2).mean(dim=-1, keepdim=True) + 1e-7)
    assert torch.isfinite(y).all() and float(y.abs().max()) > 0.5
