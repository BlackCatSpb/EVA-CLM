"""P4-3: межшкальный детектор противоречий (ContradictionField).

Ключевой тест — синтетика: поток из двух режимов (a, затем −a); χ_time обязан
подскочить на смене режима и остаться у нуля на однородном потоке. Плюс:
forward-нейтральность флага (детектор — монитор), конечность на мёртвой памяти,
lazy-init лестницы и влияние rel-руки на логиты.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.block import EVABlock            # noqa: E402
from core.config import EVAConfig          # noqa: E402
from core.stack import EVAStack            # noqa: E402


def _cfg(**kw):
    base = dict(D=64, vocab=128, n_layers=1, logit_cache_enabled=False,
                gradient_checkpointing=False)
    base.update(kw)
    return EVAConfig(**base)


def _blk(flag=True):
    cfg = _cfg()
    cfg.contradiction_field = flag
    torch.manual_seed(0)
    return EVABlock(cfg, 0)


def test_chi_time_spikes_at_regime_shift():
    blk = _blk(True)
    blk.eval()
    a = torch.randn(64)
    b = -a                       # cos(a, b) = −1: полная смена режима
    vals = []
    st = None
    with torch.no_grad():
        for t in range(160):
            v = a if t < 80 else b
            out, st = blk(v.view(1, 1, 64), st)
            vals.append(float(blk._chi_time.mean()))
    base = sum(vals[40:80]) / 40
    after = vals[80:130]
    assert max(after) > base + 0.5, f'сдвиг не замечен: base={base:.4f} max={max(after):.4f}'
    assert min(after[:10]) > base, 'первые позиции после сдвига не выше базы'
    assert max(after) > 2.0 * base + 1e-3


def test_chi_zero_and_finite_on_dead_memory():
    blk = _blk(True)
    blk.eval()
    with torch.no_grad():
        blk(torch.zeros(1, 8, 64))
    assert torch.isfinite(blk._chi_time).all()
    assert float(blk._chi_time.abs().max()) == 0.0


def test_detector_is_forward_neutral():
    torch.manual_seed(1)
    m_off = EVAStack(_cfg(n_layers=2))
    m_off.eval()                          # eval: без train-шума фантома
    sd = {k: v.clone() for k, v in m_off.state_dict().items()}
    cfg_on = _cfg(n_layers=2)
    cfg_on.contradiction_field = True
    torch.manual_seed(2)
    m_on = EVAStack(cfg_on)
    m_on.load_state_dict(sd)
    m_on.eval()
    x = torch.randint(3, 128, (1, 12))
    with torch.no_grad():
        o1 = m_off(m_off.embed_tokens(x), None, adaptive=False, step=10, tokens=x)[0]
        o2 = m_on(m_on.embed_tokens(x), None, adaptive=False, step=10, tokens=x)[0]
    assert torch.equal(o1, o2), 'детектор изменил forward (должен быть монитором)'
    assert m_on.layers[0]._chi_time is not None
    assert m_off.layers[0]._chi_time is None
    assert torch.isfinite(m_on.layers[0]._chi_time).all()


def test_chi_ladder_lazy_init():
    cfg = _cfg()
    cfg.contradiction_field = True
    torch.manual_seed(3)
    m = EVAStack(cfg)
    m.train()
    head = m.lm_head
    assert head.Kp > 0, 'фантом-канал выключен — лестница не обновляется'
    x = torch.randint(3, 128, (1, 12))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, adaptive=False, step=1, tokens=x)
    m.compute_losses(out, x, h_emb=h)     # голова вызывается здесь (_phantom_mix)
    lad = head._chi_ladder
    assert float(lad.sum()) > 0.0, 'lazy-init лестницы не сработал'
    before = lad.clone()
    out, *_ = m(h, None, adaptive=False, step=2, tokens=x)
    m.compute_losses(out, x, h_emb=h)
    assert not torch.equal(before, lad), 'лестница не движется в training'


def test_temper_rel_arm_changes_logits():
    cfg = _cfg()
    cfg.head_temper_rel = True
    torch.manual_seed(4)
    m = EVAStack(cfg)
    m.eval()
    head = m.lm_head
    h = torch.randn(1, 4, 64)
    with torch.no_grad():
        head._mem_dir = torch.randn(1, 4, 64)
        head._chi_in = torch.full((1, 4), 0.5)
        head._chi_ladder.fill_(0.5)
        l_on = head(h)
        head._temper_rel = False
        l_off = head(h)
    assert torch.isfinite(l_on).all()
    assert not torch.allclose(l_on, l_off), 'rel-рука не влияет на логиты'
