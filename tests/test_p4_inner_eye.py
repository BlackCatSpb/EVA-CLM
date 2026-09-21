"""P4-1: InnerEye — обучаемая интроспекция (shared inner eye).

Контракты: (1) identity на init (zero-init выхода ⇒ forward побитово прежний);
(2) wake-up path — градиент в верхний слой с шага 0, в нижний — с шага 1
(тот же контракт, что phantom_mix); (3) ограниченный авторитет — выход
RMS-нормирован и зажат ±10 при любом раздутии весов (урок run A2).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig      # noqa: E402
from core.inner_eye import InnerEye    # noqa: E402
from core.stack import EVAStack        # noqa: E402


def _cfg(**kw):
    base = dict(D=64, vocab=128, n_layers=2, logit_cache_enabled=False,
                gradient_checkpointing=False)
    base.update(kw)
    return EVAConfig(**base)


def test_identity_at_init():
    torch.manual_seed(1)
    m_off = EVAStack(_cfg())
    m_off.eval()
    sd = {k: v.clone() for k, v in m_off.state_dict().items()}
    cfg_on = _cfg()
    cfg_on.inner_eye = True
    torch.manual_seed(2)
    m_on = EVAStack(cfg_on)
    missing, unexpected = m_on.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all('inner_eye' in k for k in missing), missing
    m_on.eval()
    x = torch.randint(3, 128, (1, 12))
    with torch.no_grad():
        o1 = m_off(m_off.embed_tokens(x), None, adaptive=False, step=10, tokens=x)[0]
        o2 = m_on(m_on.embed_tokens(x), None, adaptive=False, step=10, tokens=x)[0]
    assert torch.equal(o1, o2), 'InnerEye не identity на init'
    assert m_on.inner_eye is not None
    assert len(m_on.inner_eye.expert_bias) == max(
        int(l.mirror.G) for l in m_on.layers)


def test_grad_flows_wake_up_path():
    cfg = _cfg()
    cfg.inner_eye = True
    torch.manual_seed(3)
    m = EVAStack(cfg)
    m.train()
    ie = m.inner_eye
    x = torch.randint(3, 128, (1, 12))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, adaptive=False, step=1, tokens=x)
    ce, _ = m.compute_losses(out, x, h_emb=h)
    ce.backward()
    assert ie.net[-1].weight.grad is not None
    assert float(ie.net[-1].weight.grad.abs().sum()) > 0.0
    assert ie.expert_bias.grad is not None
    # шаг 0: нижний слой нулевой (верхний нулевой по init — градиент вниз не течёт)
    g0 = float(ie.net[0].weight.grad.abs().sum()) if ie.net[0].weight.grad is not None else 0.0
    assert g0 == 0.0, f'нижний слой не должен учиться на шаге 0 (g={g0})'
    # шаг 1 (после сдвига верхнего слоя): градиент доходит и вниз
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        ie.net[-1].weight.add_(0.01)
    h2 = m.embed_tokens(x)          # свежий граф (первый backward его освободил)
    out, *_ = m(h2, None, adaptive=False, step=2, tokens=x)
    ce, _ = m.compute_losses(out, x, h_emb=h2)
    ce.backward()
    assert float(ie.net[0].weight.grad.abs().sum()) > 0.0, 'wake-up path мёртв'


def test_bounded_authority():
    ie = InnerEye(G=8, hidden=32)
    feats = torch.randn(1, 4, 8, 12) * 10.0
    with torch.no_grad():
        for p in ie.net.parameters():
            p.mul_(1000.0)
    ie.train()
    for _ in range(300):
        o = ie(feats, 0.5)
        assert torch.isfinite(o).all()
    assert float(o.abs().max()) <= 10.0 + 1e-5
    assert float(o.abs().mean()) < 12.0


def test_g_bias_slicing():
    ie = InnerEye(G=32)
    for G in (8, 16, 32):
        o = ie(torch.zeros(1, 2, G, 12), 0.3)
        assert o.shape == (1, 2, G)
