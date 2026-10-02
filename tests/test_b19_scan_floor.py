"""B19: the log-space scan floor equals the external maximum(decay, d_s^k).

Плюс замок корневого фикса bind: градиент по decay СТРОГО НИЖЕ пола не ноль
(STE), forward при этом бит-в-бит жёсткий кламп (decay<=1, паддинг точен).
История: сглаженный softplus-пол был отклонён адверсариальным аудитом — у
медленных шкал rest лежит в ~2e-4 над полом, сглаживание делало log_a>0
(decay>1, рост памяти 28.9x/512 токенов при tau=4111)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.block import _scan_chunk, _soft_floor   # noqa: E402


def test_ste_floor_forward_is_hard_clamp_and_grad_is_transparent():
    torch.manual_seed(3)
    # область скана: log(decay) <= 0 (decay = clamp(..., 0.01, 1.0))
    x = -torch.rand(4096, dtype=torch.float64) * 3.0
    x.requires_grad_(True)
    floor = torch.tensor(-0.05, dtype=torch.float64)
    y = _soft_floor(x, floor)
    ref = torch.maximum(x.detach(), floor)
    assert torch.equal(y.detach(), ref), 'forward не жёсткий кламп (бит-в-бит)'
    # decay<=1: log_a = max(log(decay), floor) <= 0 при floor<0
    assert float(y.detach().max()) <= 0.0
    # градиент прозрачен ВЕЗДЕ, включая строго ниже пола (старый баг: 0)
    w = torch.randn_like(y)
    (y * w).sum().backward()
    assert torch.allclose(x.grad, w, atol=0, rtol=0), 'STE-градиент не identity'
    below = x.detach() < floor
    assert bool(below.any()), 'тест не задел область ниже пола'
    assert float(x.grad[below].abs().sum()) > 0.0, \
        'градиент строго ниже пола мёртв — регрессия STE'


def test_log_floor_equals_external_max():
    torch.manual_seed(0)
    B, CH, S, Dh = 1, 16, 4, 64
    d_s = torch.tensor([0.70, 0.90, 0.99, 0.9993])          # per-scale nominal
    decay = (d_s.view(1, 1, S, 1) * torch.rand(B, CH, S, Dh) * 1.4).clamp(0.01, 1.0)
    k = 2.0
    floored = torch.maximum(decay, d_s.view(1, 1, S, 1).pow(k))
    # b differs -> recompute with SAME b
    b = torch.randn(B, CH, S, Dh)
    a1 = _scan_chunk(b, decay, floor_log=(k * d_s.clamp(min=1e-6).log()).view(1, 1, S, 1))[1]
    a2 = _scan_chunk(b, floored)[1]
    rel = float((a1 - a2).norm() / a2.norm())
    # external max is fp32, the in-scan floor is fp64 — agree to fp32 round-off
    assert rel < 1e-5, f'log floor != external max: rel {rel:.2e}'
