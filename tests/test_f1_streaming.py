"""F1 (math audit): the AR (L=1) streaming path must reproduce the training
signals.

Before the fix:
  * the mirror's `hp_prev` was built WINDOW-INTERNALLY
    (cat[zeros, hp[:, :-1]]) — at L=1 it is identically ZERO, so
    pred_k = 0 and the prediction error froze at a constant (measured 0.25);
  * the bipolar positional mask was indexed `[:, :L]` — at L=1 every token
    read position 0, so the phase information disappeared (hp diverged ~150%);
  * the embedding's sentence masks degenerated to rel≡0 / bos≡True;
  * the sentence-ring pooled single tokens as "sentences".

The fix carries the last hp, the window-relative mask phase, the
sentence-relative counter and the sentence pool — all gated on `_ar_mode`
(L=1 only). Training and windowed eval are bit-for-bit unchanged.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.embedding import PartitionedEmbedding   # noqa: E402
from core.mirror import GroupedCognitiveMirror    # noqa: E402


def _mk(seq_len: int = 8):
    torch.manual_seed(0)
    m = GroupedCognitiveMirror(D=64, G=2, k=16, n_layers=2, seq_len=seq_len)
    m.eval()
    m._ar_mode = True
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
        c1 = m._stream_hp_prev.clone()
        m(h, mem)
        c2 = m._stream_hp_prev.clone()
    # the carried hp is the MASKED hp of the token, and the phase advanced
    assert torch.allclose(c1, raw * m._pos_id_buf[:, 0:1], rtol=1e-5, atol=1e-6)
    assert torch.allclose(c2, raw * m._pos_id_buf[:, 1:2], rtol=1e-5, atol=1e-6)
    assert int(m._pos_ptr.item()) == 2


def test_f1_training_keeps_the_teacher_forced_form():
    m = _mk(seq_len=8)
    m.train()
    m._ar_mode = True          # even with the flag on, training must not carry
    h = torch.randn(1, 4, 64)
    mem = torch.randn(1, 4, 64)
    m(h, mem)
    assert float(m._stream_hp_prev.abs().max()) == 0.0, \
        'training must not use the eval-side hp carry'
    assert int(m._pos_ptr.item()) == 0, \
        'training must not advance the eval-side phase'


def test_f1_phase_wraps_at_seq_len():
    m = _mk(seq_len=3)
    h = torch.randn(1, 1, 64)
    mem = torch.randn(1, 1, 64)
    with torch.no_grad():
        for _ in range(5):
            m(h, mem)
    assert int(m._pos_ptr.item()) == 5
    # the mask index used is (ptr % seq_len): the 5th call used index 1
    assert torch.allclose(m._stream_hp_prev, _raw_hp(m, h) * m._pos_id_buf[:, 1:2],
                          rtol=1e-5, atol=1e-6)


def test_f1_reset_stream_bufs():
    m = _mk(seq_len=8)
    h = torch.randn(1, 1, 64)
    mem = torch.randn(1, 1, 64)
    with torch.no_grad():
        m(h, mem)
    m.reset_stream_bufs()
    assert float(m._stream_hp_prev.abs().max()) == 0.0
    assert int(m._pos_ptr.item()) == 0


def test_f1_sent_masks_stream_contract():
    from core.config import EVAConfig
    cfg = EVAConfig(D=64, vocab=64)
    emb = PartitionedEmbedding(cfg)
    emb.eval()
    emb._ar_mode = True
    SEP = torch.tensor([[2]])
    T = torch.tensor([[7]])
    seq = [T, T, T, SEP, T]
    rels, bosses = [], []
    with torch.no_grad():
        for t in seq:
            sep, bos, rel = emb.sent_masks_stream(t)
            rels.append(int(rel.item()))
            bosses.append(bool(bos.item()))
    # rel walks 0,1,2,3(SEP = its sentence's length-1) then resets to 0
    assert rels == [0, 1, 2, 3, 0], rels
    assert bosses == [True, False, False, False, True], bosses
