"""M65 (аудит 19360 §1): ST-кламп log-odds головы (`head_u_clamp`).

Контракты: (1) 0 = выключено (forward бит-в-бит как раньше); (2) ниже рейла
кламп не меняет forward; (3) на рейле forward ограничен |u|≤U; (4) backward —
identity: у правильно насыщенного бита параметрический CE-градиент оживает
(~4e-4 против ~e^-34); (5) `_last_u`/`_last_sat` (стена и спайк-телеметрия)
читают СЫРОЙ u.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


def _model(clamp: float):
    cfg = EVAConfig(D=64, vocab=128, n_layers=2, logit_cache_enabled=False,
                    gradient_checkpointing=False)
    cfg.head_u_clamp = clamp
    torch.manual_seed(0)
    return EVAStack(cfg)


def _u_of(head, h, gain_zero=True):
    """u (log-odds) головы для данного h; emphasis_gain=0 изолирует путь битов."""
    with torch.no_grad():
        if gain_zero:
            head.emphasis_gain.data.zero_()
        zt, zd, _ = head._gates(h, return_data=True)
        u, _ = head._su(zt, zd)
    return u


def test_zero_is_off_and_small_h_is_identical():
    m0, m8 = _model(0.0), _model(8.0)
    m8.load_state_dict(m0.state_dict(), strict=False)
    torch.manual_seed(1)
    h = torch.randn(1, 4, 64) * 0.05            # u заведомо < 8
    u0 = _u_of(m0.lm_head, h)
    u8 = _u_of(m8.lm_head, h)
    assert torch.allclose(u0, u8, atol=1e-6), 'ниже рейла кламп обязан не влиять'
    assert float(u0.abs().max()) < 8.0


def test_rail_bounds_forward():
    m8 = _model(8.0)
    torch.manual_seed(2)
    h = torch.randn(1, 4, 64) * 200.0           # спайк: сырой u >> 8
    u8 = _u_of(m8.lm_head, h)
    assert float(u8.abs().max()) <= 8.0 + 1e-3, 'forward обязан быть ограничен'
    m0 = _model(0.0)
    u0 = _u_of(m0.lm_head, h)
    assert float(u0.abs().max()) > 12.0, 'сырой u должен насыщать (тест-предусловие)'


def test_ste_revives_parameter_gradient_at_rail():
    m8, m0 = _model(8.0), _model(0.0)
    m8.load_state_dict(m0.state_dict(), strict=False)
    torch.manual_seed(3)
    h = torch.randn(1, 4, 64) * 200.0

    def param_grad(m):
        m.lm_head.zero_grad(set_to_none=True)
        m.lm_head.emphasis_gain.data.zero_()     # только путь битов
        zt, zd, _ = m.lm_head._gates(h, return_data=True)
        u, _ = m.lm_head._su(zt, zd)
        # gain=0 ⇒ u = z (сырой zt); метки — знак СЫРОГО z, маска — только
        # насыщенные биты: у правильно насыщенного бита CE-градиент = σ'(u) → 0
        # без клампа, σ'(±8)≈3.4e-4 с клампом.
        c = (zt.detach() > 0).float()
        sel = (zt.detach().abs() > 20.0).float()
        loss = -(sel * (c * F.logsigmoid(u) + (1 - c) * F.logsigmoid(-u))).sum()
        loss.backward()
        return float(m.lm_head.readout.grad.abs().max())

    g8, g0 = param_grad(m8), param_grad(m0)
    assert g8 > 1e-6, f'кламп обязан оживить градиент: {g8:.2e}'
    assert g0 < g8 * 1e-3, f'без клампа градиент мёртв: {g0:.2e} vs {g8:.2e}'


def test_last_u_and_sat_are_raw():
    m8 = _model(8.0)
    m8.train()
    torch.manual_seed(4)
    h = torch.randn(1, 4, 64) * 200.0
    with torch.no_grad():
        m8.lm_head.emphasis_gain.data.zero_()
        zt, zd, _ = m8.lm_head._gates(h, return_data=True)
        m8.lm_head._su(zt, zd)
    assert float(m8.lm_head._last_u.abs().max()) > 12.0, \
        'стена/телеметрия обязаны видеть СЫРОЙ u'
    assert float(m8.lm_head._last_sat) > 0.0
