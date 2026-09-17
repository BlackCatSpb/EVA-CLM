"""M64 lock: the L2 memory bank's write projections are LIVE (the M63-D audit).

The old `write()` ran W_k/W_v under @torch.no_grad and copied the result into
the buffers, so the keys/vals the READ consumed were detached constants:
W_k/W_v never received a gradient (measured: p.grad is None for the whole
run) and the read degenerated into a per-forward bias (cos(r_t, r_t') ~ 1.0
between positions). M64 makes the write functional: the effective store
(keys_eff/vals_eff) carries the graph, the read consumes it, the persistent
buffer is committed with the detached copy, and the novelty gate (a dead
diagnostic before) scales the written value so the CE can teach it.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.memory_bank import L2Bank  # noqa: E402


def _bank():
    torch.manual_seed(0)
    return L2Bank(D=64, bridge_dim=8, n_slots=4)


def test_write_projections_get_gradients():
    """The M63-D regression: all three write-side modules must learn."""
    b = _bank()
    emb = torch.randn(64, requires_grad=True)
    b.write(emb)
    out = b.read(torch.randn(1, 3, 64))
    out.sum().backward()
    for name, p in (('W_k', b.W_k.weight), ('W_v', b.W_v.weight),
                    ('novelty_gate', b.novelty_gate[0].weight)):
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
