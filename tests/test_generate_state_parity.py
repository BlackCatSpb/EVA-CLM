"""P0-6 (F1b) state parity for BOTH generation paths (generate + smart_generate).

Баг, который лок'ится: smart_generate (путь --smart) не получил фикс P0-6 и
пере-подавал скользящее окно с несённым состоянием — VSA/банк/UCL/кэш получали
~L записей на токен; observe_output получал скрытые состояния вместо логитов;
AR-режим не выставлялся. Тест инструментирует model.forward и observe_output и
требует: (1) prefill = одно окно промпта; (2) после него КАЖДЫЙ вызов — ровно
L=1; (3) нумерация шагов от base_step; (4) observe_output получает логиты
(last dim == vocab), а не скрытые.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


class _Tok:
    """Мини-стаб токенизатора: 4 токена + SEP (id=2) добавит сам генератор."""

    def encode(self, s):
        class _E:
            ids = [5, 6, 7, 8]
        return _E()

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(i) for i in ids)


def _model():
    cfg = EVAConfig(D=64, vocab=128, n_layers=2, logit_cache_enabled=True,
                    gradient_checkpointing=False)
    torch.manual_seed(0)
    return EVAStack(cfg)


def _instrument(model):
    calls, obs = [], []
    orig = model.forward

    def wrapped(h, *a, **kw):
        # только верхнеуровневые вызовы: триада делает внутренние ре-проходы
        # (self.forward) с тем же окном — это by design, не генерация
        if int(kw.get('_triad_depth', 0)) == 0:
            calls.append((int(h.shape[1]), int(kw.get('step', -1))))
        return orig(h, *a, **kw)

    model.forward = wrapped
    orig_obs = model.observe_output

    def obs_wrapped(x):
        obs.append(tuple(x.shape))
        return orig_obs(x)

    model.observe_output = obs_wrapped
    return calls, obs


def test_smart_generate_is_prefill_plus_l1(monkeypatch):
    import scripts.generate as G
    import scripts.smart_controller as SC
    monkeypatch.setattr(G, 'load_russian_tokenizer', lambda *a, **k: _Tok())
    m = _model()
    calls, obs = _instrument(m)
    ctrl = SC.SmartController(m, 128, reasoning_on=False)
    _, dec = SC.smart_generate(m, 'привет', ctrl, max_new_tokens=3, base_step=100)
    lens = [c[0] for c in calls]
    steps = [c[1] for c in calls]
    assert lens[0] == 5, f'prefill должен быть одним окном промпта: {lens}'
    assert lens[1:] == [1, 1, 1], f'декод обязан быть L=1 (баг: пере-подача окна): {lens}'
    assert steps == [100, 101, 102, 103], f'нумерация шагов от base_step: {steps}'
    assert all(s[-1] == 128 for s in obs), \
        f'observe_output обязан получать логиты (vocab=128), а не скрытые: {obs}'
    assert len(dec) == 3


def test_plain_generate_is_prefill_plus_l1(monkeypatch):
    import scripts.generate as G
    monkeypatch.setattr(G, 'load_russian_tokenizer', lambda *a, **k: _Tok())
    m = _model()
    calls, obs = _instrument(m)
    G.generate(m, 'привет', max_new_tokens=3, temperature=1.0, top_k=8,
               sampler=None, base_step=100)
    lens = [c[0] for c in calls]
    steps = [c[1] for c in calls]
    assert lens[0] == 5, lens
    assert lens[1:] == [1, 1, 1], lens
    assert steps == [100, 101, 102, 103], steps
    assert all(s[-1] == 128 for s in obs), obs
