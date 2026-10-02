"""M65-opt (агент B): FFT-свёртка вместо gather+einsum в bind.

Замер агента B: 37.0 -> 0.74 ms/слой (50x), освобождение ~18.9MB/слой графа.
Индекс `_circ_conv_idx` = (n−t)%K — это СВЁРТКА: irfft(rfft(a)·rfft(b)).
(Прототип B с flip+roll проверял корреляцию (t+n)%K — к этим сайтам неприменим.)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.bind import _hrr_conv, TrajectorySpiralBind   # noqa: E402
from core.config import EVAConfig                        # noqa: E402


def _conv_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    K = a.shape[-1]
    idx = torch.tensor([[(n - t) % K for n in range(K)] for t in range(K)],
                       dtype=torch.long)
    return torch.einsum('...t,...tn->...n', a, b[..., idx])


def test_fft_conv_matches_gather_einsum_fwd_and_grad():
    torch.manual_seed(0)
    for dt, tol in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
        for shape in ((7, 32), (3, 5, 32), (2, 4, 3, 32)):
            a = torch.randn(*shape, dtype=dt, requires_grad=True)
            b = torch.randn(*shape, dtype=dt, requires_grad=True)
            ref = _conv_ref(a, b)
            out = _hrr_conv(a, b)
            assert out.shape == ref.shape and out.dtype == ref.dtype
            assert torch.allclose(out, ref, atol=tol, rtol=0), \
                (shape, dt, float((out - ref).abs().max()))
            w = torch.randn_like(ref)
            (ref * w).sum().backward()
            ga, gb = a.grad.clone(), b.grad.clone()
            a.grad = b.grad = None
            (_hrr_conv(a, b) * w).sum().backward()
            assert torch.allclose(a.grad, ga, atol=tol * 10, rtol=0), (shape, dt, 'ga')
            assert torch.allclose(b.grad, gb, atol=tol * 10, rtol=0), (shape, dt, 'gb')


def test_fft_conv_half_cast_path():
    torch.manual_seed(1)
    a = torch.randn(5, 32, dtype=torch.float16)
    b = torch.randn(5, 32, dtype=torch.float16)
    out = _hrr_conv(a, b)
    assert out.dtype == torch.float16
    ref = _conv_ref(a.float(), b.float())
    assert torch.allclose(out.float(), ref, atol=1e-2, rtol=0)


def test_trajectory_spiral_forward_uses_fft_and_flows_grad():
    # live-режим (bind_twist_mode='trajectory_spiral'): forward конечен, форма
    # верна, градиент по h течёт (интеграционный smoke поверх _hrr_conv)
    cfg = EVAConfig(D=128, n_layers=2, mlp_groups=4, code_dim=16,
                    code_sparsity=4, vocab=256, bind_K=32, save_dir='.',
                    bind_twist_mode='trajectory_spiral', bind_traj_dims=3,
                    bind_twist_gate=True)
    torch.manual_seed(0)
    m = TrajectorySpiralBind(128, 32, cfg).train()
    h = torch.randn(1, 24, 128, requires_grad=True)
    out, new_traj, coh = m(h)
    assert out.shape == (1, 24, 128)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert h.grad is not None and float(h.grad.abs().sum()) > 0.0
    assert float(out.norm()) > 0.0
