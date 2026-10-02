"""tau_compression: roundtrip-контракты и STE (M65-opt).

Ранее модуль не имел прямых тестов. Здесь же зафиксирован найденный баг:
обе STE-формулы были перевёрнуты (`q + (x−q).detach()`) — forward возвращал
НЕквантованные значения, backward был ноль. Канон `x + (q−x).detach()`:
forward = квантованное, backward = identity.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import core.tau_compression as tc      # noqa: E402


def test_uniform8_roundtrip_bound():
    torch.manual_seed(0)
    t = torch.randn(64, 64)
    idx, t_min, scale = tc.compress_uniform8(t)
    rec = tc.decompress_uniform8(idx, t_min, scale, t.shape, t.dtype)
    assert rec.shape == t.shape and rec.dtype == t.dtype
    err = float((rec - t).abs().max())
    assert err <= scale / 2 + 1e-6, f'err={err} bound={scale/2}'


def test_uniform8_constant_tensor_is_exact():
    t = torch.full((4, 4), 2.5)
    idx, t_min, scale = tc.compress_uniform8(t)
    assert idx is None and scale == 0.0
    rec = tc.decompress_uniform8(idx, t_min, scale, t.shape, t.dtype)
    assert torch.allclose(rec, t)


def test_sparse_topk_roundtrip_keeps_top_entries_exactly():
    torch.manual_seed(0)
    t = torch.randn(2, 4, 256)      # decompress ожидает (B, L, V)
    k = 16
    idx_pos, idx_vals, meta = tc.compress_sparse_topk(t, k=k)
    rec = tc.decompress_sparse_topk(idx_pos, idx_vals, meta, t.shape, t.dtype)
    _vals, pos = t.topk(k, dim=-1)   # компрессор берёт top-k по ЗНАЧЕНИЮ
    q_rec = rec.gather(-1, pos)
    q_ref = t.gather(-1, pos)
    assert float((q_rec - q_ref).abs().max()) <= float(meta[1]) / 2 + 1e-5
    mask = torch.ones_like(t, dtype=torch.bool)
    mask.scatter_(-1, pos, False)
    # B1: вне top-k — конечный хвост vmin−2 (не −inf, не нули)
    assert torch.allclose(rec[mask], torch.full_like(rec[mask], float(meta[2]))), \
        'вне top-k обязан быть хвост meta[2]'


def test_delta_roundtrip_bound():
    torch.manual_seed(0)
    cur = torch.randn(128)
    cached = cur + 0.5 * torch.randn(128)   # |delta| < 3 (внутри clamp'а)
    idx, d_min, scale = tc.compress_delta(cur, cached)
    rec = tc.decompress_delta(idx, d_min, scale, cached, cur.dtype)
    err = float((rec - cur).abs().max())
    assert err <= scale / 2 + 1e-5, f'err={err} bound={scale/2}'


def test_ste_uniform_forward_is_quantized_backward_is_identity():
    torch.manual_seed(0)
    x = torch.randn(256, requires_grad=True)
    idx, t_min, scale, vals = tc.compress_uniform8_ste(x)
    assert vals.requires_grad, 'STE: граф порван (был баг перевёрнутой формулы)'
    q = idx.float() * scale + t_min
    assert torch.allclose(vals.detach(), q, atol=1e-6), \
        'STE forward обязан быть КВАНТОВАННЫМ (был неквантованный)'
    vals.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x)), 'STE backward != identity'


def test_ste_sparse_forward_is_quantized_backward_is_identity():
    torch.manual_seed(0)
    x = torch.randn(4, 64, requires_grad=True)
    idx_pos, idx_vals, meta, vals = tc.compress_sparse_topk_ste(x, k=8)
    assert vals.requires_grad, 'STE: граф порван'
    q = idx_vals.float() * meta[1] + meta[0]
    assert torch.allclose(vals.detach(), q, atol=1e-6), \
        'STE forward обязан быть квантованным'
    vals.sum().backward()
    # STE разреженный по построению: identity на top-k позициях, 0 вне
    pos = idx_pos.long()
    g_top = x.grad.gather(1, pos)
    assert torch.allclose(g_top, torch.ones_like(g_top)), 'STE backward != identity на top-k'
    mask = torch.ones_like(x, dtype=torch.bool)
    mask.scatter_(1, pos, False)
    assert float(x.grad[mask].abs().max()) == 0.0, 'вне top-k градиента быть не должно'
