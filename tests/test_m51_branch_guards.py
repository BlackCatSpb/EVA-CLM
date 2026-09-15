"""M51 lock: per-branch injection caps bound any single branch, and the branch
loss carries an absolute-scale anchor (the 2970 explosion post-mortem).

The failure it locks out: the branch-balance loss is scale-free (log-ratios),
so a uniform growth of ALL branches was invisible while the residual stream
ran away to ~1e16 (gradient to the trunk -> 0 -> deadlock at step 2970).
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


def _forward(m):
    x = torch.randint(1, SMALL['vocab'], (1, 16))
    h = m.embed_tokens(x)
    out, st, gs, _ = m(h, None, step=1, tokens=x)
    return x, h, out


def _blow_up_first_conv(m, factor=1e6):
    with torch.no_grad():
        for p in m.layers[0].conv.parameters():
            p.mul_(factor)


def _capture_layer0(m):
    cap = {}

    def _hook(mod, inp, out):
        cap['h'] = (out[0] if isinstance(out, tuple) else out).detach()

    m.layers[0].register_forward_hook(_hook)
    return cap


def test_branch_cap_bounds_injection():
    m = _model(branch_cap=1.0).eval()
    _blow_up_first_conv(m)
    cap = _capture_layer0(m)
    _forward(m)
    assert float(cap['h'].abs().amax()) < 1e3, \
        'branch cap did not bound a 1e6-scale conv injection'


def test_branch_cap_disabled_explodes():
    m = _model(branch_cap=0.0).eval()
    _blow_up_first_conv(m)
    cap = _capture_layer0(m)
    _forward(m)
    assert float(cap['h'].abs().amax()) > 1e4, \
        'control failed: the uncapped injection should be huge'


def _branch_aux(anchor_w):
    m = _model(branch_balance_weight=0.1, branch_var_anchor=anchor_w).train()
    x, h, out = _forward(m)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    _ = float(aux['branch'])          # seed the anchor reference (healthy state)
    for layer in m.layers:
        for nm in ('_cache_conv_out', '_cache_bind_out', '_cache_mirror_out'):
            t = getattr(layer, nm, None)
            if t is not None:
                setattr(layer, nm, t * 1e4)   # uniform growth of ALL branches
    ce2, aux2 = m.compute_losses(out, x, h_emb=h)
    return float(aux2['branch'])


def test_branch_anchor_penalizes_uniform_growth():
    no_anchor = _branch_aux(0.0)
    with_anchor = _branch_aux(0.5)
    assert with_anchor > no_anchor + 10.0, \
        f'anchor blind to a uniform x1e4 branch growth: {no_anchor} -> {with_anchor}'
