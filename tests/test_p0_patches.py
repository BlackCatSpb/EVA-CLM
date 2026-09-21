"""P0-1 (F2) and P0-5 (F1a) — the proposed-patch acceptance locks.

P0-1: the learnable VSA ladder is clamped at the single tau_s entry point
(tau_s >= 0.5*k) so the tail-referenced fp32 scan can never overflow into
inf*0 = NaN; leaving the safe zone is telemetered (`scan_floor_bound`).

P0-5c: the AR (L=1) sentence-ring accumulates the sentence and pushes the pool
on SEP instead of pooling single tokens.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.block import EVABlock                    # noqa: E402
from core.config import EVAConfig                  # noqa: E402
from core.logit_cache import LogitAttention, LogitCache  # noqa: E402
from core.training_control import training_telemetry     # noqa: E402


def _block():
    cfg = EVAConfig(D=64, vocab=64)
    cfg.gradient_checkpointing = False
    torch.manual_seed(0)
    b = EVABlock(cfg, 0)
    b.train()
    return b


def test_p0_1_drifted_tau_is_clamped_and_flagged():
    b = _block()
    with torch.no_grad():
        b._vsa_tau_log.fill_(-50.0)      # tau_s -> e^-50 (the NaN regime)
    out = b(torch.randn(1, 8, 64))
    res = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(res).all(), 'the tau_s clamp must keep the scan finite'
    assert bool(b._scan_floor_bound), 'leaving the safe zone must be telemetered'


def test_p0_1_healthy_tau_is_untouched():
    b = _block()
    out = b(torch.randn(1, 8, 64))
    res = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(res).all()
    assert not bool(b._scan_floor_bound), 'the healthy ladder must not trip the clamp'


def test_p0_1_telemetry_reports_and_resets_the_flag():
    b = _block()
    with torch.no_grad():
        b._vsa_tau_log.fill_(-50.0)
    b(torch.randn(1, 8, 64))
    assert bool(b._scan_floor_bound)

    class _M:
        layers = [b]

    tt = training_telemetry(_M())
    assert tt.get('scan_floor_bound') == 1, tt
    assert not bool(b._scan_floor_bound), 'the flag is sticky per log interval'


def test_p0_5c_ar_sentence_accumulator_pools_on_sep():
    torch.manual_seed(0)
    att = LogitAttention(D=16, V=64, n_heads=2, kv_dim=8, codes=None,
                         sentence_ring=True, ms_spans=())
    cache = LogitCache(V=64, D=16, max_entries=8, ms_spans=())
    h = torch.randn(1, 1, 16)
    for tok in (7, 9, 11, 2):           # three tokens + SEP
        cache.store(h, training=True)   # the wrapper does this before attending
        att(h, cache, training=True, tokens=torch.tensor([[tok]]))
    assert len(cache._kv_sent) == 1, 'exactly one sentence pool is pushed on SEP'
    assert int(cache._sent_lens[-1]) == 4, cache._sent_lens
    # the accumulator is drained after the push
    assert float(att._sent_acc_n.item()) == 0.0
