"""M64 lock: the UCL read-scale floor is gradient-alive (the M63-C freeze).

Found by the M63 agent chain: `scale.clamp(min=floor)` has
d scale/d read_scale = 0 exactly when bound. With read_scale init -4.0
(sigma = 0.017986 < floor 0.1) the parameter was frozen at its init for ALL
5000 floor steps — the 'self-closure' measured after the release was the
frozen init, not a decision, and the write-path gradient (proportional to
scale) was dead the whole floor phase. The fix is the affine
reparameterization scale = floor + (1 - floor) * sigmoid(w).
"""
import math
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


def test_floor_formula_and_minimum():
    # the stack rewrites _scale_floor from the config each forward -> set the knob
    m = _model(ucl_read_scale_floor=0.5, ucl_read_scale_floor_until=10 ** 9).train()
    ucl = m.concept_layer
    with torch.no_grad():
        ucl.read_scale.fill_(-4.0)
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    with torch.no_grad():
        m(h, None, step=0, tokens=x)
    s = float(ucl._last_scale)
    want = 0.5 + 0.5 * (1.0 / (1.0 + math.exp(4.0)))
    assert abs(s - want) < 1e-6, f'floor formula wrong: {s} vs {want}'
    assert s >= 0.5, 'the floor minimum does not hold'


def test_floor_gradient_is_alive():
    """The M63-C regression: with the old clamp this gradient was exactly 0."""
    m = _model(ucl_read_scale_floor=0.5, ucl_read_scale_floor_until=10 ** 9).train()
    ucl = m.concept_layer
    with torch.no_grad():
        ucl.read_scale.fill_(-4.0)
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=0, tokens=x)
    loss = out.float().pow(2).mean()
    loss.backward()
    g = ucl.read_scale.grad
    assert g is not None, 'read_scale got no gradient at all'
    assert torch.isfinite(g).all()
    assert abs(float(g)) > 0.0, 'the floor killed the read_scale gradient (M63-C freeze)'


def test_floor_release_returns_to_plain_sigmoid():
    m = _model(ucl_read_scale_floor=0.5, ucl_read_scale_floor_until=100).train()
    ucl = m.concept_layer
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    with torch.no_grad():
        m(h, None, step=200, tokens=x)   # past the release
        s = float(ucl._last_scale)
        want = float(torch.sigmoid(ucl.read_scale))
    assert abs(s - want) < 1e-6, f'the release is not a plain sigmoid: {s} vs {want}'
