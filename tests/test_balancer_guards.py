"""Блок 1 (закрытие очереди перед A100): гарды LossBalancer.

(a) NaN/Inf-гард: не-конечный per-parameter вклад aux/bypass/safety
    отбрасывается (вклад = 0), счётчик n_nonfinite; p.grad не травится.
(b) Bypass-бонд: combined (CE+aux+bypass) per-parameter ≤ 2·‖g_CE‖
    (до фикса адверсариальная сумма достигала 4·‖g_CE‖).
(c) cos не искажается eps-полом при gau≈0 (+1e-8 убран из nb); пороги
    scale_min_ratio не сломаны.
(d) last_scale сбрасывается в None на невалидном align-замере.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.training_control import LossBalancer  # noqa: E402


def _toy():
    a = torch.nn.Parameter(torch.tensor(1.0))
    b = torch.nn.Parameter(torch.tensor(1.0))
    return a, b


INF = float('inf')
NAN = float('nan')


# ── (a) NaN/Inf-гард ─────────────────────────────────────────────────────────

def test_align_inf_aux_contribution_is_dropped():
    """inf-градиент aux не должен попасть в p.grad: параметр получает CE.

    Контракт (задача блока 1): не-конечный gau ⇒ вклад 0 ДЛЯ ЭТОГО ПАРАМЕТРА
    целиком (aux сходится в один граф-сумму; отделить здоровое слагаемое того
    же параметра без 10-кратного роста traversals нельзя). На a (aux не
    зависит) CE-градиент сохраняется."""
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    ce = a ** 2 + b ** 2
    lb.backward(ce, {'good': 0.05 * b ** 2, 'bad': b * INF}, [a, b], step=0)
    assert lb.last_path == 'align'
    assert lb.n_nonfinite >= 1, 'non-finite aux contribution was not counted'
    assert torch.isfinite(b.grad).all(), f'inf leaked into p.grad: {b.grad}'
    # per-param drop: b получает только CE (finite aux + inf в одной сумме)
    assert abs(float(b.grad) - 2.0) < 1e-5, f'inf aux leaked: {float(b.grad)}'
    assert abs(float(a.grad) - 2.0) < 1e-5


def test_align_nan_aux_contribution_is_dropped():
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    ce = a ** 2 + b ** 2
    lb.backward(ce, {'bad': b * NAN}, [a, b], step=0)
    assert lb.n_nonfinite >= 1
    assert torch.isfinite(b.grad).all() and torch.isfinite(a.grad).all()
    assert abs(float(b.grad) - 2.0) < 1e-5, 'CE gradient must survive alone'


def test_cheap_inf_aux_value_is_dropped():
    """cheap-путь: inf-ЗНАЧЕНИЕ aux исключается из total (один backward)."""
    lb = LossBalancer(align=True, align_every=0, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)  # seed
    s = float(lb.scale_ema)
    a.grad = b.grad = None
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2, 'bad': b * INF},
                [a, b], step=1)
    assert lb.last_path == 'balance'
    assert lb.n_nonfinite == 1
    assert abs(float(b.grad) - (2.0 + s * 0.2)) < 1e-3, \
        f'inf aux value poisoned the cheap total: {float(b.grad)}'


def test_bypass_inf_is_dropped():
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    ce = a ** 2 + b ** 2
    lb.backward(ce, {'x': 0.05 * b ** 2, 'gradalign': b * INF}, [a, b], step=0)
    assert lb.n_nonfinite >= 1
    assert torch.isfinite(b.grad).all()
    assert abs(float(b.grad) - 2.1) < 1e-5


def test_safety_inf_is_dropped():
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    ce = a ** 2 + b ** 2
    lb.backward(ce, {'x': 0.05 * b ** 2, 'head_wall': b * INF}, [a, b], step=0)
    assert lb.n_nonfinite >= 1
    assert torch.isfinite(b.grad).all()
    assert abs(float(b.grad) - 2.1) < 1e-5


def test_finite_path_stays_bit_identical():
    """Гард не меняет числа на нормальном пути (счётчик нулевой)."""
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)
    assert lb.n_nonfinite == 0
    assert abs(float(a.grad) - 2.0) < 1e-6 and abs(float(b.grad) - 2.2) < 1e-5


# ── (b) bypass-бонд: combined ≤ 2‖CE‖ ────────────────────────────────────────

def test_bypass_combined_per_parameter_bound():
    """Адверсариал: aux и bypass по 100×CE на параметре. До фикса ‖итог‖ ≤ 4‖CE‖
    (aux ≤ ‖CE‖ поверх CE, bypass ≤ ‖p.grad‖ поверх обоих); теперь ≤ 2‖CE‖."""
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    ce = a ** 2 + b ** 2
    lb.backward(ce, {'x': 100.0 * b ** 2, 'gradalign': 100.0 * b ** 2},
                [a, b], step=0)
    assert lb.last_path == 'align'
    gce_b = 2.0                      # d(a^2+b^2)/db at b=1
    assert abs(float(b.grad)) <= 2.0 * gce_b + 1e-5, \
        f'combined bound 2‖CE‖ broken: {float(b.grad)}'
    assert abs(float(b.grad)) > gce_b + 1e-3, 'aux/bypass contributed nothing'
    assert abs(float(a.grad)) <= 2.0 * 2.0 + 1e-5
    assert abs(float(a.grad)) > 2.0 - 1e-3   # CE-only on a (no aux path)


def test_bypass_under_bound_is_not_rescaled():
    """Малый bypass (Δ < ‖CE‖): рескейл неактивен, вклад ровно как добавлялся."""
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    ce = a ** 2 + b ** 2
    lb.backward(ce, {'gradalign': 0.1 * b ** 2}, [a, b], step=0)
    # aux пуст ⇒ bypass-ветка «CE+B14»: b = 0.1*2b = 0.2; sc=min(1, 2/0.2)=1
    assert abs(float(b.grad) - 2.2) < 1e-5, f'under-bound rescaled: {float(b.grad)}'
    assert lb.last_path == 'align'


def test_bypass_combined_bound_holds_across_params():
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    ce = 3.0 * a ** 2 + 5.0 * b ** 2
    lb.backward(ce, {'x': 50.0 * a ** 2, 'gradalign': 30.0 * b ** 2},
                [a, b], step=0)
    for p, expect in ((a, 6.0), (b, 10.0)):
        gn = expect                      # d(ce)/dp at p=1
        assert abs(float(p.grad)) <= 2.0 * gn + 1e-5, \
            f'bound broken on {p.shape}: {float(p.grad)} vs 2·{gn}'


# ── (c) cos без eps-искажения, пороги scale_min_ratio ────────────────────────

def test_cos_not_distorted_when_aux_grad_is_tiny():
    """gau ~1e-12: истинный cos = 1.0. Со старым +1e-8 в nb выходило ~1e-4."""
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'tiny': 1e-12 * b ** 2}, [a, b], step=0)
    assert lb.last_cos is not None
    assert lb.last_cos > 0.999, f'cos distorted by eps floor: {lb.last_cos}'


def test_scale_min_ratio_gate_still_rejects_noise():
    """Порог не сломан: nb/na ~1e-12 < scale_min_ratio => scale не сеется
    (порог работает на СЫРОМ nb — +1e-8 удалён только из cos)."""
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'tiny': 1e-12 * b ** 2}, [a, b], step=0)
    assert lb.scale_ema is None, f'noise measurement seeded s={lb.scale_ema}'
    # чуть ниже порога (nb/na = 0.049 < 0.05) — тоже отказ
    lb3 = LossBalancer(align=True, eval_interval=100)
    a3, b3 = _toy()
    lb3.backward(a3 ** 2 + b3 ** 2, {'x': 0.049 * b3 ** 2}, [a3, b3], step=0)
    assert lb3.scale_ema is None, 'sub-threshold measurement seeded the scale'
    # масштаб порядка порога (nb/na = 0.05 ровно) принимается (cap scale_max=10)
    lb2 = LossBalancer(align=True, eval_interval=100)
    a2, b2 = _toy()
    lb2.backward(a2 ** 2 + b2 ** 2, {'x': 0.05 * b2 ** 2}, [a2, b2], step=0)
    assert lb2.scale_ema is not None
    assert lb2.scale_ema <= lb2.scale_max + 1e-9 and lb2.last_scale is not None


# ── (d) last_scale сбрасывается на невалидном замере ─────────────────────────

def test_last_scale_reset_on_invalid_measurement():
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)
    assert lb.last_scale is not None
    s_ema = float(lb.scale_ema)
    a.grad = b.grad = None
    # невалидный замер: nb/na ~1e-12 — scale не обновляется и last_scale чист
    lb.backward(a ** 2 + b ** 2, {'tiny': 1e-12 * b ** 2}, [a, b], step=1)
    assert lb.last_scale is None, 'stale last_scale not reset'
    assert abs(float(lb.scale_ema) - s_ema) < 1e-12, 'EMA touched by invalid measurement'


def test_zero_aux_grad_is_invalid_and_safe():
    """nb == 0 (нулевой вспомогательный градиент) не делит на ноль и не сеет."""
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    zero_grad_aux = (b * 0.0).sum() * 0.0 + 0.0 * b   # value 0, grad 0
    lb.backward(a ** 2 + b ** 2, {'z': zero_grad_aux}, [a, b], step=0)
    assert lb.last_scale is None and lb.scale_ema is None
    assert lb.last_cos == 0.0
