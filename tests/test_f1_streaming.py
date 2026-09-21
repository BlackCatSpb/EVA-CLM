"""F1 (math audit): the streaming (autoregressive) path must reproduce the
training-time signals.

Before the fix:
  * the mirror's `hp_prev` was built WINDOW-INTERNALLY
    (cat[zeros, hp[:, :-1]]) — at L=1 it is identically ZERO, so
    pred_k = 0 and the prediction error froze at a constant (measured 0.25);
  * the bipolar positional mask was indexed `[:, :L]` — at L=1 every token
    read position 0, so the phase information disappeared (hp diverged ~150%).

The fix carries the last hp and the window-relative phase across eval calls
(training is bit-for-bit unchanged: the teacher-forced window form).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mirror import GroupedCognitiveMirror  # noqa: E402


def _mk(seq_len: int = 8):
    torch.manual_seed(0)
    m = GroupedCognitiveMirror(D=64, G=2, k=16, n_layers=2, seq_len=seq_len)
    m.eval()
    return m


def _raw_hp(m, h):
    return torch.einsum('blgd,gdk->blgk', h.reshape(1, 1, m.G, m.d), m.W_proj)


def test_f1_streaming_carries_hp_prev_and_the_phase():
    m = _mk(seq_len=8)
    h = torch.randn(1, 1, 64)
    mem = torch.randn(1, 1, 64)
    raw = _raw_hp(m, h)
    with torch.no_grad():
        m(h, mem)
        c1 = m._hp_prev_cache.clone()
        m(h, mem)
        c2 = m._hp_prev_cache.clone()
    # the cached hp is the MASKED hp of the token, and the mask phase advanced
    assert torch.allclose(c1, raw * m._pos_id_buf[:, 0:1], rtol=1e-5, atol=1e-6)
    assert torch.allclose(c2, raw * m._pos_id_buf[:, 1:2], rtol=1e-5, atol=1e-6)
    assert m._stream_phase == 2


def test_f1_streaming_pred_error_is_not_frozen():
    m = _mk(seq_len=8)
    mem = torch.randn(1, 1, 64)
    outs = []
    with torch.no_grad():
        for t in range(3):
            h = torch.randn(1, 1, 64)
            outs.append(m(h, mem)[0].clone())
    # with hp_prev carried, identical-shaped inputs no longer produce the
    # degenerate constant response of the zero-prediction regime
    assert not torch.allclose(outs[1], outs[2])


def test_f1_training_keeps_the_teacher_forced_form():
    m = _mk(seq_len=8)
    m.train()
    h = torch.randn(1, 4, 64)
    mem = torch.randn(1, 4, 64)
    m(h, mem)
    assert getattr(m, '_hp_prev_cache', None) is None, \
        'training must not use the eval-side hp carry'
    assert int(getattr(m, '_stream_phase', 0)) == 0, \
        'training must not advance the eval-side phase'


def test_f1_phase_wraps_at_seq_len():
    m = _mk(seq_len=3)
    h = torch.randn(1, 1, 64)
    mem = torch.randn(1, 1, 64)
    with torch.no_grad():
        for _ in range(5):
            m(h, mem)
    assert m._stream_phase == 5 % 3
