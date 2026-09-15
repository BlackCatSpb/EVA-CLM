"""M52b locks: the lacuna residual (orthogonal to the readout), the phantom
channel (identity at init, lacuna-gated, gradient alive), the learnable
exploration noise.

Design (EVA-Ai transplanted to hidden states): the lacuna = the part of h that
is invisible to the known bits; the phantom basis reads it, gated by its
magnitude — the head's "potential state" for unfamiliar concepts.
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


def test_lacuna_is_orthogonal_to_every_readout_direction():
    m = _model().eval()
    h = torch.randn(2, 8, m.cfg.D)
    zt, zd, e_l = m.lm_head._gates(h, return_data=True)
    K, d = m.lm_head.K, m.cfg.D // m.lm_head.K
    proj = (e_l.reshape(2, 8, K, d) * m.lm_head.readout).sum(-1)
    assert float(proj.abs().max()) < 1e-5, 'the known bits can see the lacuna'


def test_lacuna_scalar_zero_on_familiar_positive_on_novel():
    m = _model().eval()
    K, D = m.lm_head.K, m.cfg.D
    z = torch.randn(2, 8, K)
    h_fam = (z.unsqueeze(-1) * m.lm_head.readout).reshape(2, 8, D)
    h_nov = h_fam + 2.0 * torch.randn(2, 8, D)
    ell = {}
    for nm, h in (('fam', h_fam), ('nov', h_nov)):
        _, _, e_l = m.lm_head._gates(h, return_data=True)
        ell[nm] = float(e_l.norm(dim=-1).mean() / (h.norm(dim=-1).mean() + 1e-6))
    assert ell['fam'] < 1e-4, f'in-span state shows a lacuna: {ell}'
    assert ell['nov'] > 0.05, f'novel direction invisible: {ell}'


def test_phantom_channel_is_identity_at_init():
    m = _model().eval()
    h = torch.randn(2, 8, m.cfg.D)
    out1 = m.lm_head(h)
    m.lm_head.Kp = 0                                  # disable the phantom path
    out2 = m.lm_head(h)
    m.lm_head.Kp = 32
    assert torch.allclose(out1, out2, atol=0.0), 'phantom channel is not identity at init'
    assert float(m.lm_head.phantom_mix.abs().max()) == 0.0


def test_phantom_gradient_alive_and_lacuna_aux_fires():
    m = _model().train()
    x, h = _hq(m)
    out, st, gs, _ = m(h, None, step=1, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    assert 'phantom_l1' in aux, 'the phantom L1 aux did not fire'
    m.zero_grad(set_to_none=True)
    ce.backward()
    assert m.lm_head.phantom_mix.grad is not None
    assert float(m.lm_head.phantom_mix.grad.abs().max()) > 0.0, 'mix gradient dead'
    # once the mix is nonzero, the phantom basis itself must learn
    with torch.no_grad():
        m.lm_head.phantom_mix.normal_(0.0, 0.01)
    x2, h2 = _hq(m)
    out2, _, _, _ = m(h2, None, step=2, tokens=x2)
    ce2, _ = m.compute_losses(out2, x2, h_emb=h2)
    m.zero_grad(set_to_none=True)
    ce2.backward()
    assert float(m.lm_head.phantom_basis.grad.abs().max()) > 0.0, 'basis gradient dead'


def test_noise_eta_is_learnable_and_bounded():
    import torch.nn as nn
    m = _model()
    assert isinstance(m.lm_head.log_eta, nn.Parameter)
    eta = float(torch.exp(m.lm_head.log_eta).clamp(0.0, 0.2))
    assert 0.0 < eta <= 0.2


def test_phantom_telemetry_exists():
    m = _model().train()
    x, h = _hq(m)
    _ = m.lm_head(h)
    assert hasattr(m.lm_head, '_last_lacuna') and hasattr(m.lm_head, '_last_p')
    assert float(m.lm_head._last_lacuna) >= 0.0
