# -*- coding: utf-8 -*-
"""T9.9 лок: границы предложений как первоклассный сигнал для всей модели.

Раньше SEP(id=2) читал только банк памяти; ствол/голова/кэш видели плоский
поток. Теперь эмбеддинг добавляет три сигнала (все zero-init ⇒ на старте
forward бит-в-бит прежний): sent_eos_emb (на SEP), sent_bos_emb (первый токен
предложения / позиция 0), sent_pos_emb (позиция внутри предложения).

Run: python -m pytest tests/test_t9_sent_boundary.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack


def _cfg(sent=True):
    return EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                     vocab=400, save_dir='.', logit_cache_enabled=False,
                     memory_bank=False, intent_bridge=True, vsa_decay_floor_k=2.0,
                     gradient_checkpointing=False, sent_boundary_emb=sent)


def _tokens_with_seps():
    torch.manual_seed(7)
    x = torch.randint(3, 400, (1, 24))
    x[:, 7] = 2
    x[:, 15] = 2
    x[:, 23] = 2
    return x


def test_masks_correct():
    torch.manual_seed(0)
    m = EVAStack(_cfg()).eval()
    x = _tokens_with_seps()
    sep, bos, rel = m.embed.sent_masks(x)
    assert sep.sum().item() == 3 and bool(sep[0, 7]) and bool(sep[0, 15])
    # rel: позиция внутри предложения (0 = первый токен; SEP — конец своего)
    assert int(rel[0, 0]) == 0 and int(rel[0, 6]) == 6
    assert int(rel[0, 7]) == 7 and int(rel[0, 8]) == 0 and int(rel[0, 14]) == 6
    assert int(rel[0, 15]) == 7 and int(rel[0, 16]) == 0
    # bos: позиция 0 + первые токены после SEP (rel==0)
    assert bool(bos[0, 0]) and bool(bos[0, 8]) and bool(bos[0, 16])
    assert not bool(bos[0, 3]) and not bool(bos[0, 7])


def test_zero_init_identity_then_learns():
    torch.manual_seed(0)
    m_on = EVAStack(_cfg(True)).train()
    torch.manual_seed(0)
    m_off = EVAStack(_cfg(False)).train()
    x = _tokens_with_seps()
    with torch.no_grad():
        h_on = m_on.embed(x)
        h_off = m_off.embed(x)
    assert torch.equal(h_on, h_off), 'zero-init сигналы не бит-в-бит нейтральны'
    # после шага обучения сигналы получают градиент и начинают менять forward
    # (h пересчитываем С графом — h_on снят под no_grad)
    h_g = m_on.embed(x)
    out, *_ = m_on(h_g, None, step=1, tokens=x)
    ce, _ = m_on.compute_losses(out, x, h_emb=h_g)
    ce.backward()
    for name in ('sent_eos_emb', 'sent_bos_emb', 'sent_pos_emb.weight'):
        p = dict(m_on.embed.named_parameters())[name]
        assert p.grad is not None and float(p.grad.norm()) > 0, f'{name}: мёртвый градиент'
    with torch.no_grad():
        m_on.embed.sent_pos_emb.weight[1].fill_(0.01)
        h2 = m_on.embed(x)
    assert not torch.equal(h2, h_off), 'сигнал не влияет на forward'


def test_ab_switch_off_is_previous_behavior():
    torch.manual_seed(0)
    m = EVAStack(_cfg(False)).train()
    assert getattr(m.embed, 'sent_eos_emb', None) is None
    x = _tokens_with_seps()
    h = m.embed(x)   # не должно падать и не содержать сигналов
    assert h.shape == (1, 24, 128)


def test_stack_forward_with_signal():
    torch.manual_seed(0)
    m = EVAStack(_cfg(True)).train()
    x = _tokens_with_seps()
    h = m.embed(x)
    out, st, gs, _ = m(h, None, step=1, tokens=x)
    assert out.shape == h.shape and torch.isfinite(out).all()


def test_rel_pos_clamped():
    torch.manual_seed(0)
    m = EVAStack(_cfg()).eval()
    x = torch.randint(3, 400, (1, 200))   # без SEP: rel = 0..199 → clamp 63
    _, _, rel = m.embed.sent_masks(x)
    assert int(rel.max()) == m.embed._sent_pos_max - 1
