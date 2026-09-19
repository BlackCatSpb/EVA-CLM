# -*- coding: utf-8 -*-
"""T9 locks: ковариационная память (порт EVA-Ai/FCP).

- chunked-скан ≡ одношаговая рекуррентность (streaming-контракт EVA);
- τ-затухание: больший τ → память держится дольше;
- градиенты текут во все проекции;
- zero-init W_out ⇒ старт как residual (бит-в-бит).

Run: python -m pytest tests/test_t9_cov_memory.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.cov_memory import CovarianceMemory


def _m(tau=64.0, chunk=16, seed=0):
    torch.manual_seed(seed)
    m = CovarianceMemory(D=32, n_heads=2, head_dim=8, tau=tau, chunk=chunk)
    with torch.no_grad():
        m.W_out.weight.normal_(0.0, 0.3)   # снять zero-init для содержательных тестов
    return m


def test_chunked_scan_matches_streaming():
    m = _m(chunk=16)
    x = torch.randn(1, 37, 32)
    y_full, st = m(x, None)
    # пошагово
    ys = []
    s = None
    for t in range(x.shape[1]):
        yt, s = m.step(x[:, t:t + 1], s)
        ys.append(yt)
    y_step = torch.cat(ys, dim=1)
    d = (y_full - y_step).abs().max().item()
    assert d < 1e-4, f'скан ≠ рекуррентность: max|Δ|={d:.2e}'
    # состояние совпадает
    ds = (st - s).abs().max().item()
    assert ds < 1e-4, f'состояние после скана ≠ после шагов: max|Δ|={ds:.2e}'


def test_zero_init_is_identity():
    torch.manual_seed(0)
    m = CovarianceMemory(D=32, n_heads=2, head_dim=8, tau=64.0, chunk=16)
    x = torch.randn(1, 20, 32)
    y, _ = m(x, None)
    assert float(y.abs().max().detach()) == 0.0, 'zero-init W_out не даёт бит-в-бит residual'


def test_tau_controls_retention():
    """Больший τ → медленнее затухание → дальний вклад заметнее."""
    x = torch.randn(1, 24, 32)
    y_fast, _ = _m(tau=8.0).forward(x, None)
    y_slow, _ = _m(tau=512.0).forward(x, None)
    assert not torch.allclose(y_fast, y_slow), 'τ не влияет на память'
    # энергия состояния: медленная память сохраняет больше
    _, s_fast = _m(tau=8.0).forward(x, None)
    _, s_slow = _m(tau=512.0).forward(x, None)
    assert float(s_slow.norm()) > float(s_fast.norm()), 'τ=512 должен удерживать больше'


def test_gradients_flow():
    m = _m()
    x = torch.randn(1, 24, 32, requires_grad=False)
    y, _ = m(x, None)
    y.pow(2).mean().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f'{name}: нет градиента'
    assert float(m.k_proj.weight.grad.norm()) > 0
    assert float(m.q_proj.weight.grad.norm()) > 0
    assert float(m.W_read.weight.grad.norm()) > 0
