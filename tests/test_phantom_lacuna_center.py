"""M65 (аудит 19360 §3): центрирование лакуны (`phantom_lacuna_ema`).

Контракт: при >0 из НАБЛЮДАЕМОГО вектора вычитается EMA общего режима лакуны
(структурная константа, ph_cos_p50=0.98), салиентность/порог не трогаются;
EMA двигается только в training на каденции наблюдений (M8-доктрина).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


def _model(ema: float):
    cfg = EVAConfig(D=64, vocab=128, n_layers=2, logit_cache_enabled=False,
                    gradient_checkpointing=False)
    cfg.phantom_lacuna_ema = ema
    torch.manual_seed(0)
    m = EVAStack(cfg)
    m.train()
    return m


def _spy(head):
    calls = []
    orig = head.phantom_bank.observe

    def _ob(e, l, t):
        calls.append(e.detach().float().clone())
        return orig(e, l, t)

    head.phantom_bank.observe = _ob
    return calls


def _mix(head, e, h):
    u = torch.zeros(e.shape[0], e.shape[1], head.K)
    head._phantom_mix(u, e, h)


def test_centering_removes_common_mode_and_updates_ema():
    m = _model(0.9)
    head = m.lm_head
    calls = _spy(head)
    torch.manual_seed(5)
    v = torch.randn(1, 1, 64)
    h = v.expand(1, 8, 64).contiguous()
    e = 2.0 * v.expand(1, 8, 64).contiguous() + 0.01 * torch.randn(1, 8, 64)

    _mix(head, e, h)
    assert len(calls) == 1, 'наблюдение обязано сработать (первый forward)'
    raw_n = float(e.norm(dim=-1).mean())
    obs_n = float(calls[0].norm(dim=-1).mean())
    assert obs_n < raw_n * 0.05, f'центрирование не сняло общий режим: {obs_n:.3f} vs {raw_n:.3f}'
    _v = v.reshape(-1)
    _em = head.lacuna_mode_ema.reshape(-1)
    cos = float((_v @ _em) / (_v.norm() * _em.norm() + 1e-9))
    assert cos > 0.99, f'EMA обязана указывать на общий режим: cos={cos:.4f}'
    assert head._last_lacuna_centered_rel < 0.05
    assert head._last_lacuna_mode_norm > 0.0

    # второй вызов: ветка EMA-обновления (не lazy-init), режим не сбивается
    # (каденция наблюдений — раз в phantom_every forward'ов, сбрасываем счётчик)
    head._pb_step.fill_(0)
    _mix(head, e, h)
    assert len(calls) == 2
    _em2 = head.lacuna_mode_ema.reshape(-1)
    cos2 = float((_em @ _em2) / (_em.norm() * _em2.norm() + 1e-9))
    assert cos2 > 0.99, f'EMA обязана стабилизироваться: cos={cos2:.4f}'


def test_single_observation_falls_back_to_ema():
    """M65b: при одном наблюдении среднее батча == само наблюдение —
    центрирование обнулило бы его; обязан работать EMA-фолбэк."""
    m = _model(0.9)
    head = m.lm_head
    calls = _spy(head)
    torch.manual_seed(7)
    v = torch.randn(1, 1, 64)
    h = v.expand(1, 1, 64).contiguous()
    e1 = 2.0 * v.expand(1, 1, 64).contiguous()
    _mix(head, e1, h)                       # инициализирует EMA
    assert len(calls) == 1
    head._pb_step.fill_(0)
    e2 = e1 + 0.3 * torch.randn(1, 1, 64)
    _mix(head, e2, h)                       # одно наблюдение -> EMA-фолбэк
    assert len(calls) == 2
    obs = calls[1]
    assert float(obs.norm()) > 0.0, 'одиночное наблюдение не должно обнуляться'
    assert float(obs.norm()) < float(e2.norm()), 'центрирование обязано убрать часть'


def test_off_is_identity_and_eval_does_not_move_ema():
    m = _model(0.0)
    head = m.lm_head
    calls = _spy(head)
    torch.manual_seed(6)
    v = torch.randn(1, 1, 64)
    h = v.expand(1, 8, 64).contiguous()
    e = 2.0 * v.expand(1, 8, 64).contiguous()

    _mix(head, e, h)
    assert len(calls) == 1
    assert torch.allclose(calls[0], e, atol=1e-6), 'при 0 наблюдение — сырая лакуна'
    assert float(head.lacuna_mode_ema.abs().sum()) == 0.0

    m.eval()
    _mix(head, e, h)
    assert len(calls) == 1, 'в eval наблюдений нет (M8-доктрина)'
    assert float(head.lacuna_mode_ema.abs().sum()) == 0.0
