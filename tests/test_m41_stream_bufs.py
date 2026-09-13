"""M41: external audit claimed the mirror grow-only stream buffers LEAK.
Verified: no leak (the plain-attr consumers hold no view; replaced buffers die
by refcount — weakrefs prove it here), but the audit's real target — an
unbounded grow-only footprint — gets a correct policy: batch change resizes
EXACTLY, document reset (stack.reset_cache) shrinks back to baseline."""
import gc
import os
import sys
import weakref

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack  # noqa: E402


def _model(seq_len=64):
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, seq_len=seq_len, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=False, explicit_reasoning=False,
                    unified_concept_layer=False)
    torch.manual_seed(0)
    return EVAStack(cfg).train(), cfg


def _fwd(m, cfg, B, L):
    x = torch.randint(1, cfg.vocab, (B, L))
    h = m.embed_tokens(x)
    with torch.no_grad():
        m(h, None, step=1, tokens=x)


def test_no_accumulation_old_buffers_die():
    m, cfg = _model()
    mir = m.layers[0].mirror
    refs = []
    for L in (128, 256, 512):
        refs.append(weakref.ref(mir._cached_hp_buf))
        _fwd(m, cfg, 1, L)
        gc.collect()
    # every superseded buffer must be dead by the end (refcount, not GC cycle)
    live = weakref.ref(mir._cached_hp_buf)
    assert all(r() is None for r in refs), 'a replaced stream buffer survived'
    assert live() is mir._cached_hp_buf
    assert mir._cached_hp_buf.shape == (1, 512, mir._cached_hp_buf.shape[2],
                                        mir._cached_hp_buf.shape[3])


def test_batch_change_resizes_exactly_then_reset_shrinks():
    m, cfg = _model(seq_len=64)
    mir = m.layers[0].mirror
    _fwd(m, cfg, 1, 64)
    assert tuple(mir._cached_hp_buf.shape)[:2] == (1, 64)
    _fwd(m, cfg, 1, 128)                      # grow -> exactly L
    assert mir._cached_hp_buf.shape[1] == 128
    _fwd(m, cfg, 2, 64)                       # batch change -> EXACT (B, L)
    assert tuple(mir._cached_hp_buf.shape)[:2] == (2, 64), \
        'grow-only max() resurrected: batch change must re-size exactly'
    m.reset_cache()
    assert tuple(mir._cached_hp_buf.shape)[:2] == (1, 64)   # baseline at doc reset
