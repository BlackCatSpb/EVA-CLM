"""EXT/аудит: полный линейный readout головы (`head_read_full`) — A/B-рука.

Контракты: (1) при включении forward ИДЕНТИЧЕН блочно-диагональному чтению
(readout_full инициализируется точно из блока, вне — нули); (2) вне-блочные
компоненты получают градиент (обучаемы); (3) e_l (лакуна) — корректный
ортогональный остаток h − z·Wᵀ; (4) 163,840 новых параметров, resume-safe
(оптимизатор пропускает по именам).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


def _model(full: bool):
    cfg = EVAConfig(D=64, vocab=128, n_layers=2, logit_cache_enabled=False,
                    gradient_checkpointing=False)
    cfg.head_read_full = full
    torch.manual_seed(0)
    return EVAStack(cfg)


def test_init_is_block_diagonal_copy_and_forward_identical():
    m_on = _model(True)
    m_off = _model(False)
    # те же веса ствола/головы (readout_full отсутствует в off-модели)
    sd = {k: v for k, v in m_on.state_dict().items() if 'readout_full' not in k}
    miss, unexp = m_off.load_state_dict(sd, strict=False)
    assert not unexp, unexp
    W = m_on.lm_head.readout_full.data
    R = m_on.lm_head.readout.data
    d, K = int(m_on.lm_head.readout.shape[-1]), m_on.lm_head.K
    # 1. вне блока — ровно нули; в блоке — копия readout
    off_block = W.clone()
    for k in range(K):
        off_block[k * d:(k + 1) * d, k] = 0.0
    assert float(off_block.abs().sum()) == 0.0, 'вне-блочные компоненты должны быть нулями'
    for k in range(K):
        assert torch.equal(W[k * d:(k + 1) * d, k], R[k]), 'блок должен точно копировать readout'
    # 2. forward идентичен (та же h -> те же логиты)
    h = torch.randn(1, 8, 64)
    m_on.eval()
    m_off.eval()
    with torch.no_grad():
        l_on = m_on.lm_head(h)
        l_off = m_off.lm_head(h)
    assert torch.allclose(l_on, l_off, rtol=1e-5, atol=1e-6), \
        f'forward должен совпадать: max|d|={float((l_on - l_off).abs().max()):.2e}'


def test_off_block_components_get_gradient():
    m = _model(True)
    m.train()
    h = torch.randn(1, 8, 64)
    x = torch.randint(3, 128, (1, 8))
    out, *_ = m(h, None, adaptive=False, step=1, tokens=x)
    ce, _ = m.compute_losses(out, x, h_emb=h)
    ce.backward()
    g = m.lm_head.readout_full.grad
    assert g is not None
    d, K = int(m.lm_head.readout.shape[-1]), m.lm_head.K
    off = g.clone()
    for k in range(K):
        off[k * d:(k + 1) * d, k] = 0.0
    assert float(off.abs().sum()) > 0.0, 'вне-блочные компоненты обязаны обучаться'
    assert float(g.abs().sum()) > 0.0


def test_lacuna_is_orthogonal_residual():
    m = _model(True)
    m.eval()
    h = torch.randn(1, 6, 64)
    with torch.no_grad():
        zt, z_data, e_l = m.lm_head._gates(h, return_data=True)
        z = h.reshape(-1, 64) @ m.lm_head.readout_full
        rec = z @ m.lm_head.readout_full.T
        e_ref = h.reshape(-1, 64) - rec
    assert torch.allclose(e_l.reshape(-1, 64), e_ref, rtol=1e-5, atol=1e-6)
    # e_l ортогонален строкам W (по построению остатка проекции)
    W = m.lm_head.readout_full
    dot = (e_l.reshape(-1, 64) @ W)
    assert float(dot.abs().max()) < 1e-4, f'остаток должен быть ⟂ W: {float(dot.abs().max()):.2e}'
