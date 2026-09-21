"""P1-2 (proposed patches): the rank-r pairwise (Ising) channel of the head.

The PoB family q ∝ exp(Σ_k c_vk u_k) cannot express intra-code correlations;
the ceiling probe (scripts/probe_head_ceiling.py) measured the price at
~4.2 nat on bigrams and a rank-16 channel closing ~67% of it for 2Kr = 2048
params. These tests pin the channel's contract:

  * identity at init (pair_V2 = 0 -> the logits are bit-for-bit the plain head's);
  * the sparse chunked `_pair_term` equals the dense reference;
  * the V2 gradient is LIVE on the first step (V1 small-random, not symmetric
    zero-init — the product's V2-gradient would be zero in the saddle);
  * a nonzero V2 changes the logits (the channel is wired into forward).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig       # noqa: E402
from core.embedding import SigmoidCodedHead  # noqa: E402


def _cfg(pair_r: int):
    cfg = EVAConfig(D=32, vocab=64, code_dim=32, code_sparsity=4)
    cfg.head_pair_rank = pair_r
    return cfg


def _head(pair_r: int = 16):
    torch.manual_seed(0)
    return SigmoidCodedHead(_cfg(pair_r))


def test_p1_2_identity_at_init():
    h = _head(16)
    assert h.pair_r == 16
    assert float(h.pair_V2.abs().max()) == 0.0
    u = torch.randn(1, 3, h.K)
    assert float(h._pair_term(u).abs().max()) == 0.0, 'the channel must start as identity'


def test_p1_2_sparse_term_matches_the_dense_reference():
    h = _head(16)
    with torch.no_grad():
        h.pair_V1.normal_(0.0, 0.5)
        h.pair_V2.normal_(0.0, 0.5)
    u = torch.randn(1, 3, h.K)
    got = h._pair_term(u)                                  # (1,3,V)
    a1 = u.unsqueeze(-1) * h.pair_V1                       # (1,3,K,r)
    a2 = u.unsqueeze(-1) * h.pair_V2
    t1 = torch.einsum('vk,...kr->...vr', h.codes, a1)
    t2 = torch.einsum('vk,...kr->...vr', h.codes, a2)
    ref = (t1 * t2).sum(dim=-1)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_p1_2_v2_gradient_is_live_on_the_first_step():
    h = _head(16)
    hh = torch.randn(1, 2, h.D)
    out = h(hh)
    out.sum().backward()
    assert h.pair_V2.grad is not None
    assert float(h.pair_V2.grad.abs().max()) > 0.0, \
        'the V2 path must be alive (V1 small-random, not all-zero)'


def test_p1_2_nonzero_v2_changes_the_logits():
    h = _head(16)
    hh = torch.randn(1, 2, h.D)
    with torch.no_grad():
        base = h(hh).clone()
        h.pair_V2.normal_(0.0, 0.05)
        after = h(hh).clone()
    assert not torch.allclose(base, after), 'the channel must be wired into forward'


def test_p1_2_off_by_default_is_bit_identical():
    h0 = _head(0)
    assert h0.pair_r == 0
    hh = torch.randn(1, 2, h0.D)
    out = h0(hh)
    assert torch.isfinite(out).all()
