# -*- coding: utf-8 -*-
"""T9 locks: ковариационная память (порт EVA-Ai/FCP).

- chunked-скан ≡ одношаговая рекуррентность (streaming-контракт EVA) при B=1 и B>1;
- handoff forward→step и step→forward (prefill ↔ токенный цикл);
- 4-D и 5-D (legacy) state; None после тёплого = холодный старт;
- τ-затухание: больший τ → память держится дольше; τ клампится TAU_MIN;
- градиенты текут во все проекции; state детачится (BPTT не течёт);
- zero-init выхода ⇒ бит-в-бит residual.

Run: python -m pytest tests/test_t9_cov_memory.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.cov_memory import CovarianceMemory
from core.tau_api import TAU_MIN


def _m(tau=64.0, chunk=16, seed=0, rank=16, heads=2, hd=8, D=32):
    torch.manual_seed(seed)
    m = CovarianceMemory(D=D, n_heads=heads, head_dim=hd, tau=tau, chunk=chunk, rank=rank)
    with torch.no_grad():
        _out = m.W_out_b.weight if m.W_out_b is not None else m.W_out.weight
        _out.normal_(0.0, 0.3)   # снять zero-init для содержательных тестов
    return m


def test_chunked_scan_matches_streaming():
    m = _m(chunk=16)
    x = torch.randn(1, 37, 32)
    y_full, st = m(x, None)
    ys, s = [], None
    for t in range(x.shape[1]):
        yt, s = m.step(x[:, t:t + 1], s)
        ys.append(yt)
    y_step = torch.cat(ys, dim=1)
    d = (y_full - y_step).abs().max().item()
    assert d < 1e-4, f'скан ≠ рекуррентность: max|Δ|={d:.2e}'
    ds = (st - s).abs().max().item()
    assert ds < 1e-4, f'состояние после скана ≠ после шагов: max|Δ|={ds:.2e}'


def test_batch2_step_matches_forward():
    m = _m(chunk=16)
    x = torch.randn(3, 21, 32)
    y_full, st = m(x, None)
    ys, s = [], None
    for t in range(x.shape[1]):
        yt, s = m.step(x[:, t:t + 1], s)
        ys.append(yt)
    y_step = torch.cat(ys, dim=1)
    assert y_full.shape == y_step.shape == (3, 21, 32)
    assert st.shape == s.shape == (3, 2, 8, 8)
    d = (y_full - y_step).abs().max().item()
    assert d < 1e-4, f'B>1: скан ≠ рекуррентность: max|Δ|={d:.2e}'


def test_handoff_forward_step():
    m = _m(chunk=8)
    x = torch.randn(2, 20, 32)
    y_full, st_full = m(x, None)
    y_a, s = m(x[:, :8], None)          # prefill-чанк
    y_b, s = m.step(x[:, 8:9], s)       # токенный цикл
    assert y_b.shape == (2, 1, 32) and s.shape == (2, 2, 8, 8)
    d = (y_full[:, :9] - torch.cat([y_a, y_b], dim=1)).abs().max().item()
    assert d < 1e-4, f'handoff forward→step: max|Δ|={d:.2e}'


def test_state_none_after_warm_is_cold():
    m = _m()
    x = torch.randn(1, 12, 32)
    _, s_warm = m(x, None)
    y_cold1, s_cold1 = m(x[:, :4], None)
    y_cold2, s_cold2 = m(x[:, :4], None)
    assert torch.equal(s_cold1, s_cold2), 'None после тёплого не даёт холодный старт'
    assert not torch.equal(s_warm, s_cold1)
    assert torch.equal(y_cold1, y_cold2)


def test_legacy_5d_state_accepted():
    m = _m()
    x = torch.randn(1, 10, 32)
    _, st = m(x, None)
    _, st4 = m(x, st)                    # тёплый 4-D
    _, st5 = m(x, st.unsqueeze(1))       # legacy (B,1,H,Dh,Dh)
    assert torch.equal(st4, st5), '5-D state не совместим с 4-D'


def test_state_detached():
    m = _m()
    x = torch.randn(1, 10, 32, requires_grad=True)
    y, st = m(x, None)
    assert not st.requires_grad, 'state не детачится (BPTT потечёт через шаги)'
    y.sum().backward()
    assert x.grad is not None


def test_zero_init_is_identity():
    torch.manual_seed(0)
    m = CovarianceMemory(D=32, n_heads=2, head_dim=8, tau=64.0, chunk=16, rank=16)
    x = torch.randn(1, 20, 32)
    y, _ = m(x, None)
    assert float(y.abs().max().detach()) == 0.0, 'zero-init W_out_b не даёт бит-в-бит residual'
    torch.manual_seed(0)
    m0 = CovarianceMemory(D=32, n_heads=2, head_dim=8, tau=64.0, chunk=16, rank=0)
    y0, _ = m0(x, None)
    assert float(y0.abs().max().detach()) == 0.0, 'zero-init W_out (full) не даёт бит-в-бит residual'


def test_tau_clamped_to_ladder_floor():
    m = CovarianceMemory(D=32, n_heads=2, head_dim=8, tau=0.5, chunk=16, rank=16)
    assert m.tau == TAU_MIN, f'τ ниже флора не клампится: {m.tau}'


def test_tau_controls_retention():
    """Больший τ → медленнее затухание → энергия состояния держится дольше."""
    x = torch.randn(1, 24, 32)
    _, s_fast = _m(tau=8.0).forward(x, None)
    _, s_slow = _m(tau=512.0).forward(x, None)
    assert float(s_slow.norm()) > float(s_fast.norm()), 'τ=512 должен удерживать больше'


def test_per_head_decay_diverges():
    """Per-head w_d: после обучения гейтов головы затухают по-разному."""
    m = _m(heads=2, hd=8, D=32)
    with torch.no_grad():
        m.w_d[:, 0].fill_(1.0)
        m.w_d[:, 1].fill_(-1.0)
    x = torch.randn(1, 16, 32)
    ld = m._log_decay(x)                 # (B, L, H)
    assert ld.shape == (1, 16, 2)
    assert not torch.allclose(ld[..., 0], ld[..., 1]), 'per-head затухание не различается'


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
