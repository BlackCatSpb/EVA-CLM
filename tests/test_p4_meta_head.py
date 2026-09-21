"""P4-2: MetaHead — зонд читаемости внутренних сигналов из h.

Контракты: (1) зонд вне CE-графа (forward ствола побитово не меняется);
(2) при meta_head_grad=False градиент терма не касается ствола (проверка через
autograd.grad); (3) терм реально тренирует параметры зонда через balancer
(PROBE_AUX-маршрут: прямой градиент — align/bypass обнулили бы его, т.к.
gce=None); (4) nan-маска (недоступный сигнал не ломает loss).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig              # noqa: E402
from core.meta_head import MetaHead, meta_loss  # noqa: E402
from core.stack import EVAStack                # noqa: E402
from core.training_control import LossBalancer  # noqa: E402


def _cfg(**kw):
    base = dict(D=64, vocab=128, n_layers=2, logit_cache_enabled=False,
                gradient_checkpointing=False)
    base.update(kw)
    return EVAConfig(**base)


def _run(m, step=1):
    x = torch.randint(3, 128, (1, 12))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, adaptive=False, step=step, tokens=x)
    return m.compute_losses(out, x, h_emb=h)


def test_probe_is_forward_neutral():
    torch.manual_seed(1)
    m_off = EVAStack(_cfg())
    m_off.eval()
    sd = {k: v.clone() for k, v in m_off.state_dict().items()}
    cfg_on = _cfg()
    cfg_on.meta_head = True
    torch.manual_seed(2)
    m_on = EVAStack(cfg_on)
    missing, unexpected = m_on.load_state_dict(sd, strict=False)
    assert not unexpected
    assert missing and all(k.startswith('meta_head.') for k in missing), missing
    m_on.eval()
    x = torch.randint(3, 128, (1, 12))
    with torch.no_grad():
        o1 = m_off(m_off.embed_tokens(x), None, adaptive=False, step=10, tokens=x)[0]
        o2 = m_on(m_on.embed_tokens(x), None, adaptive=False, step=10, tokens=x)[0]
    assert torch.equal(o1, o2), 'MetaHead изменил forward ствола'


def test_grad_does_not_touch_trunk_when_probe_mode():
    cfg = _cfg()
    cfg.meta_head = True                    # meta_head_grad=False (default)
    torch.manual_seed(3)
    m = EVAStack(cfg)
    m.train()
    ce, aux = _run(m)
    assert 'meta_read' in aux and torch.isfinite(aux['meta_read'])
    trunk = [p for n, p in m.named_parameters() if not n.startswith('meta_head')]
    g = torch.autograd.grad(aux['meta_read'], trunk, allow_unused=True,
                            retain_graph=True)
    assert all(x is None for x in g), 'зонд течёт в ствол при grad=False'
    head_g = torch.autograd.grad(aux['meta_read'], m.meta_head.net[-1].weight,
                                 allow_unused=True, retain_graph=True)[0]
    assert head_g is not None and float(head_g.abs().sum()) > 0.0


def test_probe_term_trains_through_balancer():
    cfg = _cfg()
    cfg.meta_head = True
    torch.manual_seed(4)
    m = EVAStack(cfg)
    m.train()
    ce, aux = _run(m)
    bal = LossBalancer(align=True, align_every=1)
    params = [p for p in m.parameters() if p.requires_grad]
    bal.backward(ce, aux, params, step=1, phase_model=m)
    g = m.meta_head.net[-1].weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0, \
        'PROBE_AUX-маршрут не доносит градиент до зонда'
    gb = m.meta_head.net[-1].bias.grad
    assert gb is not None and float(gb.abs().sum()) > 0.0


def test_nan_mask_keeps_loss_finite():
    ie = MetaHead(8, hidden=8)
    pred = ie(torch.randn(1, 4, 8))
    nan = torch.full((1, 4), float('nan'))
    tgt = torch.stack([nan, nan, nan, nan, nan, torch.ones(1, 4)], dim=-1)
    loss = meta_loss(pred, tgt)
    assert torch.isfinite(loss)
    all_nan = meta_loss(pred, torch.full((1, 4, 6), float('nan')))
    assert float(all_nan.detach()) == 0.0
