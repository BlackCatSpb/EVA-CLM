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
    for k in ('g_readout', 'g_token_bias', 'g_ucl_scale', 'g_phantom_basis',
              'g_lacuna_w', 'g_log_eta'):
        assert k in gc, f'{k} missing from the census'
        assert gc[k] >= 0.0 and torch.isfinite(torch.tensor(gc[k]))
    # the readout and token_bias must be LIVE (nonzero) after a real CE backward
    assert gc['g_readout'] > 0.0, 'the readout has no gradient through the CE path'
    assert gc['g_token_bias'] > 0.0


def test_grad_census_covers_the_bank_when_enabled():
    """M64.8r2 (the review): the census test must build the memory bank, or a
    bank path drift passes silently."""
    torch.manual_seed(0)
    m = EVAStack(EVAConfig(**{**SMALL, 'memory_bank': True})).train()
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=1, tokens=x)
    logits = m.lm_head(out)
    import torch.nn.functional as F
    F.cross_entropy(logits.reshape(-1, m.cfg.vocab), x.reshape(-1)).backward()
    gc = grad_census(m)
    for k in ('g_wk', 'g_wv', 'g_wq', 'g_wo', 'g_l1_proj'):
        assert k in gc, f'{k} missing (the bank path drifted?)'


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


def test_spike_snapshot_is_captured_on_a_real_saturation():
    """M64.8r2: the first version's test was false-green (bus_cap=0 disabled the
    stencil and sat stayed 0, so the assert passed trivially). Feed a huge zt
    directly into _su — the snapshot must appear with its counter."""
    m = _model()
    head = m.lm_head
    h = torch.randn(1, 4, m.cfg.D)
    with torch.no_grad():
        zt = torch.full((1, 4, head.K), 50.0)      # |u| >> 12 -> sat = 1
        u, _ = head._su(zt, zt, h_norm=float(h.norm()))
    assert float(head._last_sat) > 0.0, 'the forced saturation did not register'
    sp = getattr(head, '_spike_stats', None)
    assert sp is not None and sp['sat'] > 0.0
    assert sp['u_max'] > 12.0 and sp['n'] >= 1
    # the counter advances on the next spike (a stale snapshot is detectable)
    n0 = sp['n']
    with torch.no_grad():
        head._su(zt, zt, h_norm=float(h.norm()))
    assert head._spike_stats['n'] == n0 + 1
