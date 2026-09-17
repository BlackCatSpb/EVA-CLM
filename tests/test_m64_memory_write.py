"""M64 lock: the L2 memory bank's write projections are LIVE (the M63-D audit).

The old `write()` ran W_k/W_v under @torch.no_grad and copied the result into
the buffers, so the keys/vals the READ consumed were detached constants:
W_k/W_v never received a gradient (measured: p.grad is None for the whole
run) and the read degenerated into a per-forward bias (cos(r_t, r_t') ~ 1.0).

The first landing of this fix was REJECTED by the R1/R2/R3 review: the write
was still dead through the production call-site (StreamingMemoryBank.forward
wrapped the whole write loop in `with torch.no_grad()`), while these tests
called L2Bank.write directly — a false green. The call-site is fixed and
test_stack_path_write_is_live() goes through the stack. Also from the review:
the per-forward effective store is cleared on every `write=True` call (a
forward without a write must fall back to the buffer, never consume a stale
graph), the stash chains across the multiple writes of one forward (the
commit detaches, so only the last write's graph used to survive), the rows
are cast to the buffer dtype (AMP), and the novelty multiplier was removed
(LayerNorm annihilated it to ~1e-4 of the W_k gradient).
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core.memory_bank import L2Bank  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _bank():
    torch.manual_seed(0)
    return L2Bank(D=64, bridge_dim=8, n_slots=4)


def test_write_projections_get_gradients():
    """The M63-D regression: the write-side projections must learn."""
    b = _bank()
    emb = torch.randn(64, requires_grad=True)
    b.write(emb)
    out = b.read(torch.randn(1, 3, 64))
    out.sum().backward()
    for name, p in (('W_k', b.W_k.weight), ('W_v', b.W_v.weight)):
        assert p.grad is not None, f'{name} got no gradient (the M63-D dead write)'
        assert float(p.grad.abs().sum()) > 0.0, f'{name} gradient is zero'


def test_the_write_input_is_in_the_graph():
    """The gradient must reach the written summary through the read."""
    b = _bank()
    emb = torch.randn(64, requires_grad=True)
    b.write(emb)
    out = b.read(torch.randn(1, 2, 64))
    out.sum().backward()
    assert emb.grad is not None and float(emb.grad.abs().sum()) > 0.0, \
        'the write is detached from the read (the old no_grad path)'


def test_the_buffer_is_committed_detached():
    b = _bank()
    emb = torch.randn(64)
    b.write(emb)
    assert not b.keys.requires_grad and not b.vals.requires_grad
    want_k = F.normalize(b.W_k(emb), dim=-1) * torch.sigmoid(b.key_log_scale)
    assert torch.allclose(b.keys[0], want_k.detach(), atol=1e-6), \
        'the persistent buffer does not carry the committed write'


def test_read_falls_back_to_the_buffer_without_a_write():
    b = _bank()
    b.keys.data[0] = F.normalize(torch.randn(8), dim=-1) * 0.5
    b._keys_eff = None
    b._vals_eff = None
    out = b.read(torch.randn(1, 2, 64))
    assert torch.isfinite(out).all()


def test_the_effective_store_chains_across_writes():
    """R2: a forward writes once per SEP position; the commit detaches, so the
    effective store must chain from the PREVIOUS effective store (before this
    only the last write's graph survived)."""
    b = _bank()
    e1 = torch.randn(64, requires_grad=True)
    e2 = torch.randn(64, requires_grad=True)
    b.write(e1)
    b.write(e2)
    out = b.read(torch.randn(1, 2, 64))
    out.sum().backward()
    assert e1.grad is not None and float(e1.grad.abs().sum()) > 0.0, \
        'the first write was cut out of the graph by the second commit'
    assert e2.grad is not None and float(e2.grad.abs().sum()) > 0.0


def test_stack_path_write_is_live():
    """R2: the PRODUCTION call-site must carry the gradient (the direct-method
    test missed the no_grad wrapper around the write loop).

    Note the cold start: the fusion's last layer is zero-init (identity-safe
    by design), so at init the read path's gradients are EXACTLY zero while
    W2 itself receives a nonzero gradient (see the next test) — the wake-up
    path the live run followed (fusion.2 std 0.0148 by step 7315). We wake it
    explicitly here to test the production write path itself.
    """
    torch.manual_seed(0)
    m = EVAStack(EVAConfig(**{**SMALL, 'memory_bank': True,
                              'maturation_enabled': False})).train()
    bank = m.memory_bank
    with torch.no_grad():                 # wake the cold start
        bank.fusion[-1].weight.normal_(0.0, 0.01)
    x = torch.randint(3, SMALL['vocab'], (1, 12))
    x[0, 5] = 2                      # a SEP boundary (tokens == 2)
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=1, tokens=x)
    loss = out.float().pow(2).mean()
    loss.backward()
    l2 = bank.l2
    assert l2._keys_eff is not None and l2._keys_eff.requires_grad, \
        'the production write produced a detached store (the no_grad call-site)'
    assert l2.W_k.weight.grad is not None, 'W_k is still dead through the stack'
    assert float(l2.W_k.weight.grad.abs().sum()) > 0.0
    assert l2.W_v.weight.grad is not None
    assert float(l2.W_v.weight.grad.abs().sum()) > 0.0


def test_the_cold_start_wakes_through_the_fusion():
    """The zero-init fusion makes the read path silent at init — but W2 gets a
    nonzero gradient, which is the wake-up path (documented M63-D behaviour)."""
    torch.manual_seed(0)
    m = EVAStack(EVAConfig(**{**SMALL, 'memory_bank': True,
                              'maturation_enabled': False})).train()
    bank = m.memory_bank
    x = torch.randint(3, SMALL['vocab'], (1, 12))
    x[0, 5] = 2
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=1, tokens=x)
    out.float().pow(2).mean().backward()
    w2 = bank.fusion[-1].weight.grad
    assert w2 is not None and float(w2.abs().sum()) > 0.0, \
        'the wake-up path is broken: W2 gets no gradient at the cold start'
    assert float(bank.l2.W_k.weight.grad.abs().sum()) == 0.0, \
        'the cold start is not cold (fusion.2 is expected to be zero-init)'


def test_clear_effective_drops_the_stash():
    b = _bank()
    b.write(torch.randn(64))
    assert b._keys_eff is not None
    b.clear_effective()
    assert b._keys_eff is None and b._vals_eff is None
