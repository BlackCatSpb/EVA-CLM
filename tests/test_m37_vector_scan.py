"""M37: the vectorized chunk scan is identical to the python-loop oracle —
outputs AND gradients — at the production fast floor, and stays fp32."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.block import _scan_chunk, _scan_chunks  # noqa: E402


def _oracle(b, d, floor, chunk=32):
    outs = [_scan_chunk(b[:, s:s + chunk], d[:, s:s + chunk], floor_log=floor)
            for s in range(0, b.shape[1], chunk)]
    return (torch.cat([o[0] for o in outs], 1),
            torch.cat([o[1] for o in outs], 1),
            torch.cat([o[2] for o in outs], 1))


def test_vectorized_matches_loop_outputs_and_grads():
    for L in (64, 96, 224):                      # exact multiples of 32
        torch.manual_seed(0)
        d_s = torch.tensor([0.3997, 0.7951, 0.9443, 0.9858])
        floor = (2.0 * d_s.log()).view(1, 1, 4, 1)
        b = torch.randn(1, L, 4, 128, requires_grad=True)
        d = (d_s.view(1, 1, 4, 1) * torch.rand(1, L, 4, 128) * 1.4).clamp(.01, 1.).requires_grad_()
        bi = b.detach().clone().requires_grad_()
        di = d.detach().clone().requires_grad_()
        v_intra, v_final, v_cum = _scan_chunks(b, d, floor_log=floor)
        o_intra, o_final, o_cum = _oracle(bi, di, floor)
        assert torch.allclose(v_intra, o_intra, atol=1e-4), L
        assert v_final.shape == o_final.shape == (1, L // 32, 4, 128)
        assert torch.allclose(
            v_final, torch.stack([o_intra[:, s + 31] for s in range(0, L, 32)], 1), atol=1e-4)
        (v_intra.pow(2).mean() + v_final.pow(2).mean()).backward()
        (o_intra.pow(2).mean() + o_final.pow(2).mean()).backward()
        assert torch.isfinite(b.grad).all() and torch.isfinite(d.grad).all()
        assert torch.allclose(b.grad, bi.grad, atol=1e-4)
        assert torch.allclose(d.grad, di.grad, atol=1e-5)


def test_ragged_length_pads_not_leaks():
    torch.manual_seed(1)
    d_s = torch.tensor([0.4, 0.8, 0.94, 0.985])
    floor = (2.0 * d_s.log()).view(1, 1, 4, 1)
    b = torch.randn(1, 70, 4, 32)               # not a multiple of 32
    d = (d_s.view(1, 1, 4, 1) * torch.rand(1, 70, 4, 32) * 1.4).clamp(.01, 1.)
    intra, final, cum = _scan_chunks(b, d, floor_log=floor)
    ref = _scan_chunk(b[:, :32], d[:, :32], floor_log=floor)[0]
    assert torch.allclose(intra[:, :32], ref, atol=1e-4)
    tail = _scan_chunk(b[:, 64:], d[:, 64:], floor_log=floor)[0]
    assert torch.allclose(intra[:, 64:], tail, atol=1e-4)
    assert torch.isfinite(intra).all() and torch.isfinite(final).all()
    assert intra.dtype == torch.float32
