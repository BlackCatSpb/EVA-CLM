# -*- coding: utf-8 -*-
"""T9.9 шаг 2 лок: sentence-level наблюдения фантома (пулы предложений).

R1/R2-ревью: ассерты через ШПИОН на observe (формы входов), а не счётчик
вызовов `_obs` (он инкрементится на вызов, не на юнит).

Run: python -m pytest tests/test_t9_phantom_sentence.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack


def _model(sent=True):
    torch.manual_seed(0)
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=True, vsa_decay_floor_k=2.0,
                    gradient_checkpointing=False,
                    phantom_sentence_level=sent)
    return cfg, EVAStack(cfg).train()


def _tokens(cfg, L=24):
    torch.manual_seed(7)
    x = torch.randint(3, cfg.vocab, (1, L))
    x[:, 7] = 2
    x[:, 15] = 2
    x[:, 23] = 2
    return x


def _spy(m):
    calls = []
    orig = m.lm_head.phantom_bank.observe

    def spy(e, ell, thr):
        calls.append((tuple(e.shape), tuple(ell.shape)))
        return orig(e, ell, thr)

    m.lm_head.phantom_bank.observe = spy
    return calls


def test_sentence_level_observes_pooled_units():
    """В sentence-режиме observe получает ПУЛЫ: 2-D вход (юниты, D), число
    юнитов ≤ числу сегментов; позиционные (3-D) вызовы не наблюдаются вовсе."""
    cfg, m = _model(True)
    hd = m.lm_head
    calls = _spy(m)
    step0 = int(hd.phantom_after) + 10
    x = _tokens(cfg)
    for i in range(4):
        h = m.embed(x)
        m(h, None, step=step0 + i, tokens=x)
        hd._pb_step.fill_(0)
    assert calls, 'observe не вызывался'
    for eshape, lshape in calls:
        assert len(eshape) == 2, f'не пул (e {eshape}) — позиционное наблюдение'
        assert eshape[0] == 3 and eshape[1] == 256, f'пул не по 3 сегментам: {eshape}'
        assert lshape == (3,), f'салиентность не по юнитам: {lshape}'


def test_per_position_mode_off():
    cfg, m = _model(False)
    hd = m.lm_head
    assert hd._phantom_sent_level is False
    calls = _spy(m)
    step0 = int(hd.phantom_after) + 10
    x = _tokens(cfg)
    h = m.embed(x)
    m(h, None, step=step0, tokens=x)
    hd._pb_step.fill_(0)
    m(h, None, step=step0 + 1, tokens=x)
    assert any(len(es) == 3 for es, _ in calls), 'per-position режим не наблюдал'


def test_spike_sentence_observed():
    """Спайк: занижаем базовую шкалу ⇒ сегменты наблюдаются как ЕДИНИЦЫ;
    принятых юнитов ≤ числу предложений за вызовы."""
    cfg, m = _model(True)
    hd = m.lm_head
    _pb = hd.phantom_bank
    step0 = int(hd.phantom_after) + 10
    x = _tokens(cfg)
    h = m.embed(x)
    for i in range(3):
        m(h, None, step=step0 + i, tokens=x)
        hd._pb_step.fill_(0)
    before = int(_pb._obs.item())
    with torch.no_grad():
        hd.ell_ladder[-1].mul_(0.2)
    for i in range(4):
        m(h, None, step=step0 + 5 + i, tokens=x)
        hd._pb_step.fill_(0)
    assert int(_pb._obs.item()) > before, 'спайк-сегменты не наблюдались'
    assert int(_pb._obs.item()) <= 3 * 8, 'наблюдений больше, чем сегментов'
