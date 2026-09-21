"""F3 (math audit): the ties must be IN-GRAPH, not value-only.

Before the fix:
  * `BottleneckBind._tie_hook` copied W_proj.weight.data into the W_out
    Parameter under no_grad on every forward: the output path's gradient
    (measured ||dL/dW_out||≈51.5) was written into W_out and then ERASED by the
    next copy — it never reached W_proj; W_out's Adam moments were dead weight.
  * `GroupedCognitiveMirror` (tie_mirror_proj, production) synced the W_out
    BUFFER from W_proj under no_grad: the "K-space autoencoder" was true by
    value, false by gradient.

These tests pin the fix: the tied parameter's gradient equals the untied
model's (W_proj + W_out) gradient — i.e. the output path is in the graph.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.bind import BottleneckBind            # noqa: E402
from core.config import EVAConfig               # noqa: E402
from core.mirror import GroupedCognitiveMirror  # noqa: E402


def _bind(tie: bool):
    cfg = EVAConfig(D=64, vocab=64)
    cfg.bind_twist_mode = 'off'      # the tied path of the legacy diagonal mode
    cfg.tie_bind = tie
    torch.manual_seed(0)
    return BottleneckBind(D=64, K=32, cfg=cfg)


def _sync_untied_bind(m):
    with torch.no_grad():
        m.W_out.copy_(m.W_proj.weight)


def test_f3_bind_tie_matches_the_untied_gradient():
    tied, untied = _bind(True), _bind(False)
    untied.load_state_dict(tied.state_dict(), strict=False)   # same values
    _sync_untied_bind(untied)                                 # W_out = W_proj^T
    h = torch.randn(1, 5, 64, requires_grad=True)
    out_t = tied(h)
    out_t.sum().backward()
    out_u = untied(h)
    out_u.sum().backward()
    assert torch.allclose(out_t.detach(), out_u.detach(), rtol=1e-5, atol=1e-6)
    g_expected = untied.W_proj.weight.grad + untied.W_out.grad
    assert torch.allclose(tied.W_proj.weight.grad, g_expected, rtol=1e-4, atol=1e-6), \
        'the tied W_proj gradient must include the output path'


def test_f3_bind_forward_uses_w_proj_not_w_out():
    m = _bind(True)
    m._hook.remove()                 # stop the value mirror; W_out is now unused
    h = torch.randn(1, 5, 64)
    with torch.no_grad():
        m.W_out.fill_(123.0)
    out = m(h)
    with torch.no_grad():
        m.W_out.zero_()
    assert torch.allclose(out, m(h), rtol=1e-6, atol=1e-6), \
        'the tied forward must not read W_out'


def _mirror(tie: bool):
    torch.manual_seed(0)
    return GroupedCognitiveMirror(D=64, G=2, k=16, n_layers=2,
                                  tie_mirror_proj=tie, seq_len=8)


def test_f3_mirror_tie_matches_the_untied_gradient():
    tied, untied = _mirror(True), _mirror(False)
    untied.load_state_dict(tied.state_dict(), strict=False)   # W_out = W_proj^T
    h = torch.randn(1, 4, 64, requires_grad=True)
    mem = torch.randn(1, 4, 64)
    torch.manual_seed(1)
    out_t = tied(h, mem)[0]
    out_t.sum().backward()
    torch.manual_seed(1)
    out_u = untied(h, mem)[0]
    out_u.sum().backward()
    assert torch.allclose(out_t.detach(), out_u.detach(), rtol=1e-5, atol=1e-6)
    # W_out is (G,k,d) while W_proj is (G,d,k): the tie is W_out = W_proj^T,
    # so the untied output-path gradient must be permuted back.
    g_expected = untied.W_proj.grad + untied.W_out.grad.permute(0, 2, 1)
    assert torch.allclose(tied.W_proj.grad, g_expected, rtol=1e-4, atol=1e-6), \
        'the tied mirror W_proj gradient must include the reconstruction path'
