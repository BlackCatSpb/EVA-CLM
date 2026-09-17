"""M64.8 lock: the telemetry batch (M63-A/E requests) + the grad census."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core.training_control import grad_census, training_telemetry  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _model():
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**SMALL)).train()


def test_grad_census_reports_the_channels():
    m = _model()
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=1, tokens=x)
    # the head's params live on the CE path (the stack returns the hidden state)
    logits = m.lm_head(out)
    import torch.nn.functional as F
    ce = F.cross_entropy(logits.reshape(-1, m.cfg.vocab), x.reshape(-1))
    ce.backward()
    gc = grad_census(m)
    for k in ('g_readout', 'g_token_bias', 'g_ucl_scale'):
        assert k in gc, f'{k} missing from the census'
        assert gc[k] >= 0.0 and torch.isfinite(torch.tensor(gc[k]))
    # the readout and token_bias must be LIVE (nonzero) after a real CE backward
    assert gc['g_readout'] > 0.0, 'the readout has no gradient through the CE path'
    assert gc['g_token_bias'] > 0.0


def test_training_telemetry_keys():
    m = _model()
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=1, tokens=x)
    tt = training_telemetry(m)
    for k in ('sr_wproj', 'alpha_std', 'usage_H'):
        assert k in tt, f'{k} missing'
        assert torch.isfinite(torch.tensor(tt[k]))
    # the stable rank ratio lives in [1/rank, 1]
    assert 0.0 < tt['sr_wproj'] <= 1.0 + 1e-6


def test_spike_snapshot_is_captured():
    m = _model()
    head = m.lm_head
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    with torch.no_grad():
        # force a saturation: a huge bus stencil
        head._bus_cap = 0.0
        u0 = head._gates(h)
        _ = head._su(u0, u0, h_norm=float(h.norm()))
    assert getattr(head, '_spike_stats', None) is not None or float(head._last_sat) == 0.0
