"""M61 lock: the intent stencil's scale-invariant cap (the 165-spike).

At lr 3.3e-4 the live run spiked (ce_raw 428, sat 0.9063, head_wall 7.57) while
intent_eff climbed to 0.48 — `zt += bus_bias` was the unbounded learnable
channel. The cap uses the M50 amax pattern (no sqrt: its derivative is inf at
the zero-init) so the Jacobian stays C/m and never dies.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**{**SMALL, **kw}))


def test_bus_cap_bounds_the_stencil_and_keeps_the_gradient():
    m = _model().eval()
    head = m.lm_head
    head._bus_cap = 3.0
    h = torch.randn(1, 8, m.cfg.D)
    with torch.no_grad():
        zt0 = head._gates(h)
    bb = torch.randn(1, 8, head.K) * 50.0
    bb.requires_grad_()
    zt = head._gates(h, bus_bias=bb)
    shift = float((zt - zt0).abs().max())
    assert shift <= 3.0 + 1e-3, f'the stencil was not capped: {shift}'
    zt.sum().backward()
    assert bb.grad is not None and torch.isfinite(bb.grad).all()
    # the cap is scale-invariant: the magnitude direction has no gradient, but
    # the DIRECTION does (the stencil keeps learning)
    assert float(bb.grad.abs().max()) > 0.0, 'the cap killed the stencil gradient'


def test_bus_cap_is_nan_safe_at_the_zero_init():
    m = _model().eval()
    head = m.lm_head
    head._bus_cap = 3.0
    h = torch.randn(1, 8, m.cfg.D)
    bb = torch.zeros(1, 8, head.K, requires_grad=True)
    zt = head._gates(h, bus_bias=bb)
    zt.sum().backward()
    assert torch.isfinite(bb.grad).all(), 'the zero-init stencil produced NaN grads'


def test_bus_cap_disabled_is_the_old_forward():
    m = _model().eval()
    head = m.lm_head
    head._bus_cap = 0.0
    h = torch.randn(1, 8, m.cfg.D)
    bb = torch.full((1, 8, head.K), 50.0)
    with torch.no_grad():
        zt0 = head._gates(h)
        zt = head._gates(h, bus_bias=bb)
    assert float((zt - zt0).abs().max()) > 3.0, 'cap=0 must be a no-op'
