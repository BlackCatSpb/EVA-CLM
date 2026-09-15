"""M52a locks: the head's saturation wall (P1), the emphasis gain (P2), the
prior-free emphasis (P3), the straight-through tau clamp (P4).

The audit it answers (hybrid_gate, 2970 post-mortem):
  * dL/dz through sigma(u) is ~9.4e-14 at u=30 (logsigmoid floor) -> the head
    can freeze; the wall's gradient is linear in the excess.
  * the emphasis read zt (prior-shifted) -> corr(bit_bias, bonus) = +0.90.
  * exp(log_temp).clamp(0.1,10) froze log_temp outside the rails.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adaptive_gate import AdaptiveGate, hybrid_gate  # noqa: E402
from core.config import EVAConfig  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**{**SMALL, **kw}))


def _hq(m):
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    return x, m.embed_tokens(x)


def test_u_wall_keeps_gradient_alive_at_saturation():
    m = _model(head_u_wall=1e-3, head_u_wall_u0=6.0).train()
    with torch.no_grad():                       # drive the bits into saturation
        m.lm_head.readout.mul_(60.0)
        m.lm_head.bit_bias.add_(20.0)
    x, h = _hq(m)
    out, st, gs, _ = m(h, None, step=1, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    assert 'head_wall' in aux, 'the wall did not fire at saturated bits'
    assert float(aux['head_wall']) > 0.0
    m.zero_grad(set_to_none=True)
    aux['head_wall'].backward()
    g = m.lm_head.readout.grad
    assert g is not None and float(g.abs().max()) > 0.0, 'wall gradient dead'


def test_u_wall_off_by_default_in_small_stacks():
    m = _model(head_u_wall=0.0).train()
    x, h = _hq(m)
    out, st, gs, _ = m(h, None, step=1, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    assert 'head_wall' not in aux


def test_emphasis_gain_is_one_at_init_and_forward_unchanged():
    m = _model().eval()
    assert float(m.lm_head.emphasis_gain) == 1.0
    zt = torch.randn(1, 4, m.lm_head.K)
    u, _ = m.lm_head._su(zt)
    r = torch.softmax(zt, -1)
    want = zt + torch.log1p(r) - math.log1p(1.0 / m.lm_head.K)
    assert torch.allclose(u, want, atol=1e-6), 'forward drifted from the old form'


def test_emphasis_is_prior_free():
    m = _model().eval()
    z_data = torch.randn(1, 4, m.lm_head.K)
    zt1 = z_data + torch.randn(1, 4, m.lm_head.K)
    zt2 = zt1 + 3.0                              # a prior shift
    u1, _ = m.lm_head._su(zt1, z_data)
    u2, _ = m.lm_head._su(zt2, z_data)
    assert torch.allclose(u2 - u1, zt2 - zt1, atol=1e-6), \
        'bit_bias leaked into the emphasis'


def test_tau_st_clamp_keeps_gradient_at_the_rail():
    g = AdaptiveGate(4)
    with torch.no_grad():
        g.log_tau.fill_(3.0)                     # tau = 20 -> rail 10
    out = g(torch.randn(2, 4))
    out.sum().backward()
    assert g.log_tau.grad is not None
    assert float(g.log_tau.grad.abs().max()) > 0.0, 'tau gradient frozen at the rail'


def test_hybrid_gate_gain_and_emph_logits_api():
    z = torch.randn(2, 5)
    e = torch.randn(2, 5)
    u0, _ = hybrid_gate(z, 1.0, log=True)
    u1, _ = hybrid_gate(z, 1.0, log=True, emph_logits=e)
    assert not torch.allclose(u0, u1), 'emph_logits ignored'
    u2, _ = hybrid_gate(z, 1.0, log=True, emph_logits=e, gain=2.0)
    c = math.log1p(1.0 / 5)
    r = torch.softmax(e, -1)
    assert torch.allclose(u2, z + 2.0 * (torch.log1p(r) - c), atol=1e-6)
