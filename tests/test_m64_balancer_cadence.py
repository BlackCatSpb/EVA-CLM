"""M64.4 lock: the LossBalancer align cadence (M63-F's #1 cost lever).

The align path costs THREE graph traversals per step (CE / aux / bypass —
with gradient checkpointing that is ~3 recomputes) — the single largest
structural cost of the run (the step is latency-bound at ~0.3% of the TF32
peak). `align_every=k>1` runs the alignment on every k-th step and the cheap
ONE-backward normalized total (Kendall & Gal style: each aux normalized by its
running EMA, the block scaled to track |CE|) on the rest; 0 = never align.
Default 1 keeps the historical behaviour exactly.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core.training_control import LossBalancer  # noqa: E402


def _toy():
    a = torch.nn.Parameter(torch.tensor(1.0))
    b = torch.nn.Parameter(torch.tensor(1.0))
    return a, b


def test_default_is_every_step_align():
    lb = LossBalancer(align=True, eval_interval=100)
    assert lb.align_every == 1
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)
    assert lb.last_path == 'align'
    assert lb.last_cos is not None, 'the align path must measure the geometry'


def test_cadence_selects_the_paths():
    lb = LossBalancer(align=True, align_every=4, eval_interval=100)
    a, b = _toy()
    paths = []
    for step in range(8):
        a.grad = b.grad = None
        lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=step)
        paths.append(lb.last_path)
    assert paths == ['align', 'balance', 'balance', 'balance',
                     'align', 'balance', 'balance', 'balance']


def test_align_every_zero_is_balance_after_the_seed():
    """k=0: the first call still aligns (the scale seed), the rest are cheap."""
    lb = LossBalancer(align=True, align_every=0, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)
    assert lb.last_path == 'align' and lb.scale_ema is not None
    a.grad = b.grad = None
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=1)
    assert lb.last_path == 'balance'
    assert lb.last_cos is None


def test_the_cheap_path_is_the_measured_scale_single_backward():
    """The balance path: one backward of ce + s * sum(aux), s = the EMA of the
    scale measured by the last align pass."""
    lb = LossBalancer(align=True, align_every=0, eval_interval=100)
    a, b = _toy()
    # the first step always aligns (seeds scale_ema) even with align_every=0?
    # no: align=False-style never-align has no measurement -> the first call
    # runs the align path (the seeding rule), then the cheap ones follow.
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)
    assert lb.last_path == 'align', 'the first call must seed scale_ema'
    assert lb.scale_ema is not None
    s = float(lb.scale_ema)
    a.grad = b.grad = None
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=1)
    assert lb.last_path == 'balance'
    # d(ce)/db = 2; d(aux)/db = 0.2; the aux enters with the measured s
    assert abs(float(b.grad) - (2.0 + s * 0.2)) < 1e-3


def test_the_cheap_path_cannot_explode_on_zero_crossing_terms():
    """R1's P12/P3b adversarial case: an aux term passing through zero (value
    exactly 0, or sign-cancelling) must NOT blow up the gradient. The rejected
    value-EMA normalization gave 1e8x the CE gradient through the 1e-8 floors;
    the measured-scale path is gradient-geometry based."""
    lb = LossBalancer(align=True, align_every=0, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)  # seed
    a.grad = b.grad = None
    # an aux term that is EXACTLY zero at b=1 but has a live gradient
    lb.backward(a ** 2 + b ** 2, {'z': (b - 1.0) * 0.0 + (b - 1.0)},
                [a, b], step=1)
    assert lb.last_path == 'balance'
    ratio = abs(float(b.grad)) / 2.0
    assert ratio < 10.0, f'the cheap path exploded: {float(b.grad)} vs CE 2.0'


def test_the_cheap_path_cannot_explode_on_sign_cancellation():
    """R1's P3a: aux terms that cancel in the value sum must not blow up."""
    lb = LossBalancer(align=True, align_every=0, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)  # seed
    a.grad = b.grad = None
    lb.backward(a ** 2 + b ** 2, {'p': b - 1.0, 'n': -(b - 1.0)}, [a, b], step=1)
    assert lb.last_path == 'balance'
    ratio = abs(float(b.grad)) / 2.0
    assert ratio < 10.0, f'the cheap path exploded: {float(b.grad)} vs CE 2.0'


def test_the_cheap_path_freezes_the_gradalign_target():
    """R3: the gradalign hook must not overwrite its CE-only target with the
    combined gradient on the cheap steps."""
    lb = LossBalancer(align=True, align_every=2, eval_interval=100)

    class _L:
        _ga_record = True

    class _M:
        layers = [_L()]

    m = _M()
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0, phase_model=m)
    assert m.layers[0]._ga_record is True, 'the align path must restore the flag'
    a.grad = b.grad = None
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=1, phase_model=m)
    assert lb.last_path == 'balance'
    assert m.layers[0]._ga_record is True, 'the cheap path must restore the flag'


def test_the_align_path_keeps_the_per_parameter_bound():
    """The align path is unchanged: the aux is sign-masked and bounded by
    ||g_CE|| per parameter."""
    lb = LossBalancer(align=True, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 100.0 * b ** 2}, [a, b], step=0)
    assert lb.last_path == 'align'
    # raw aux grad at b=1 is 200; the bound adds at most ||g_CE|| = 2 -> 4 total
    assert abs(float(b.grad)) <= 4.0 + 1e-5, f'the bound is broken: {float(b.grad)}'
    assert abs(float(b.grad)) > 2.0 + 1e-3, 'the aux contributed nothing'


def test_config_knob_defaults_to_one():
    cfg = EVAConfig(n_layers=2, D=512, mlp_groups=4, code_dim=16,
                    code_sparsity=4, vocab=1820)
    assert cfg.balancer_align_every == 1


def test_state_dict_roundtrips_the_scale_seed():
    lb = LossBalancer(align=True, align_every=8, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)  # align: seeds
    assert lb.scale_ema is not None
    sd = lb.state_dict()
    assert sd.get('align_every') == 8
    assert sd.get('scale_ema') is not None
    lb2 = LossBalancer(align=True, eval_interval=100)
    lb2.load_state_dict(sd)
    assert abs(lb2.scale_ema - sd['scale_ema']) < 1e-12
