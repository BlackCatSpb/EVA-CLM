"""B19: the log-space scan floor. ИСТОРИЯ: жёсткий clamp_min(log_a, k*log(d_s))
давал точное равенство внешнему max(decay, d_s^k), но обнулял градиент bind
у медленных лестниц (frac_clamped=1.0, якобиан 0). Корневой фикс — мягкий пол
_soft_floor (softplus-колено, T=0.01): значение не ниже жёсткого пола,
отличие <= T*ln2 (~0.7%), выше колена — бит-в-бит жёсткий пол."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.block import _scan_chunk, _soft_floor   # noqa: E402


def test_soft_floor_semantics():
    torch.manual_seed(0)
    B, CH, S, Dh = 1, 16, 4, 64
    d_s = torch.tensor([0.70, 0.90, 0.99, 0.9993])
    decay = (d_s.view(1, 1, S, 1) * torch.rand(B, CH, S, Dh) * 1.4).clamp(0.01, 1.0)
    k = 2.0
    hard = torch.maximum(decay, d_s.view(1, 1, S, 1).pow(k))
    soft = torch.exp(_soft_floor(torch.log(decay.clamp(min=1e-6)),
                                 (k * d_s.clamp(min=1e-6).log()).view(1, 1, S, 1)))
    # 1) никогда не ниже жёсткого пола
    assert bool((soft >= hard - 1e-6).all()), 'мягкий пол ниже жёсткого'
    # 2) отклонение от жёсткого max ограничено T*ln2 (~0.7%)
    assert float((soft / hard - 1.0).abs().max()) < 0.008
    # 3) выше колена (decay > d_s^k * e^{0.2}) — точное равенство
    high = decay > d_s.view(1, 1, S, 1).pow(k) * 1.2214
    assert torch.allclose(soft[high], hard[high], atol=1e-6), 'выше колена не точно'
    # 4) и главное: градиент по decay НЕ ноль, когда decay СТОИТ на жёстком
    # полу (ровно зажатый случай, который раньше давал якобиан 0)
    d2 = (d_s.view(1, 1, S, 1).pow(2.0) * torch.ones(B, CH, S, Dh)).requires_grad_()
    b = torch.randn(B, CH, S, Dh)
    fl = (k * d_s.clamp(min=1e-6).log()).view(1, 1, S, 1)
    out = _scan_chunk(b, d2, floor_log=fl)[0]
    (out * torch.randn_like(out)).sum().backward()
    assert d2.grad is not None and float(d2.grad.abs().sum()) > 0.0, \
        'градиент по decay у пола мёртв — регрессия _soft_floor'
