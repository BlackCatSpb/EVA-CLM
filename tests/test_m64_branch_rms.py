"""M64.8r3 lock: the per-branch RMS telemetry must survive the cached-losses
reassignment (the round-3 verifier: `stack._cached_losses = {...}` overwrote
the first landing's writes at losses.py:349)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def test_branch_rms_reaches_the_cached_losses():
    torch.manual_seed(0)
    m = EVAStack(EVAConfig(**{**SMALL, 'branch_balance_weight': 0.1})).train()
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=1, tokens=x)
    m.compute_losses(out, x, h_emb=h)
    cl = m._cached_losses
    for k in ('branch_r_conv', 'branch_r_bind', 'branch_r_mirror'):
        assert k in cl, f'{k} lost (the reassignment overwrote it)'
        assert cl[k] > 0.0 and torch.isfinite(torch.tensor(cl[k]))
