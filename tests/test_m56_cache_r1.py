"""M56 locks: the R1 scheduled sampling is live again — the one-step-stale
logits from observe_output feed the cache, which stores/attends the COMPRESSED
logits with the configured probability (the inference-mode alignment).

Flow: forward -> observe_output(head(out)) -> the NEXT forward's cache.
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


def _hq(m):
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    return x, m.embed_tokens(x)


def _step(m, step):
    x, h = _hq(m)
    out, st, gs, _ = m(h, None, step=step, tokens=x)
    m.observe_output(m.lm_head(out))          # M56: stash the 1-step logits
    return out


def test_r1_sampling_stores_compressed_logits():
    m = _model(logit_cache_scheduled_sampling=1.0).train()
    _step(m, 1)                               # the first step: h-mode (no stash yet)
    assert len(m.logit_cache.cache._h_cache) >= 1
    _step(m, 2)                               # the stash is in -> the inference mode
    assert len(m.logit_cache.cache._logit_cache) >= 1, 'R1 never stored the compressed logits'


def test_r1_off_keeps_the_h_mode():
    m = _model(logit_cache_scheduled_sampling=0.0).train()
    _step(m, 1)
    _step(m, 2)
    assert len(m.logit_cache.cache._logit_cache) == 0, 'the h-mode run stored compressed logits'
    assert len(m.logit_cache.cache._h_cache) >= 1


def test_first_step_without_logits_is_safe():
    m = _model(logit_cache_scheduled_sampling=1.0).train()
    x, h = _hq(m)
    out, st, gs, _ = m(h, None, step=1, tokens=x)   # no observe_output -> no stash
    assert torch.isfinite(out).all()


def test_eval_keeps_the_h_mode_and_does_not_crash():
    m = _model(logit_cache_scheduled_sampling=1.0).train()
    _step(m, 1)
    m.eval()
    with torch.no_grad():
        x, h = _hq(m)
        out, st, gs, _ = m(h, None, step=2, tokens=x)
    assert torch.isfinite(out).all()
    assert len(m.logit_cache.cache._h_cache) >= 1
