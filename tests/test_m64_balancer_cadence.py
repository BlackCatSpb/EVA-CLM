"""M64.4 lock: the LossBalancer align cadence (M63-F's #1 cost lever).

The align path costs THREE graph traversals per step (CE / aux / bypass —
with gradient checkpointing that is ~3 recomputes) — the single largest
structural cost of the run (the step is latency-bound at ~0.3% of the TF32
peak). `align_every=k>1` runs the alignment on every k-th step and a cheap
ONE-backward total `ce + s*sum(aux)` on the rest, where `s` is the EMA of the
align-measured scale (gradient geometry). `align_every=0` = never align after
the first seeding call; default 1 keeps the historical behaviour exactly.

The first landing used a value-EMA normalization and was REJECTED by the R1
review (it amplified gradients as 1/|v_i|: 1e8x on a zero-crossing term).
The rework (round 3) additionally: seeds/updates `s` only from valid
measurements (nb >= scale_min_ratio*na), hard-caps `s`, falls back to the
align path when the measurement is noise-level, and freezes the gradalign
hook on the cheap steps (with a falsifiable during-backward test).
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
    """R1's P12 adversarial case: an aux term passing through zero (value
    exactly 0, live gradient) must NOT blow up the gradient. The rejected
    value-EMA normalization gave 2.0e8 vs CE=2 (1e8x) through the 1e-8 floors;
    the measured-scale path is gradient-geometry based."""
    lb = LossBalancer(align=True, align_every=0, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)  # seed
    assert lb.scale_ema is not None
    a.grad = b.grad = None
    # an aux term that is EXACTLY zero at b=1 but has a live gradient
    lb.backward(a ** 2 + b ** 2, {'z': (b - 1.0) * 0.0 + (b - 1.0)},
                [a, b], step=1)
    assert lb.last_path == 'balance'
    ratio = abs(float(b.grad)) / 2.0
    assert ratio < 10.0, f'the cheap path exploded: {float(b.grad)} vs CE 2.0'


def test_the_cheap_path_cannot_explode_on_sign_cancellation():
    """R1's exact P3a: aux terms that cancel in the value sum must not blow up.
    The value-EMA path gave 2e6x here (ema_A -> 1e-8); the measured-scale path
    is independent of the loss values."""
    lb = LossBalancer(align=True, align_every=0, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)  # seed
    a.grad = b.grad = None
    # exact cancellation of the VALUES: p + n = eps*b^2 -> ema_A ~ 1e-8 (old bomb)
    lb.backward(a ** 2 + b ** 2, {'p': b, 'n': -b + 0.01 * b ** 2}, [a, b], step=1)
    assert lb.last_path == 'balance'
    ratio = abs(float(b.grad)) / 2.0
    assert ratio < 5.0, f'the cheap path exploded: {float(b.grad)} vs CE 2.0'


def test_a_noise_level_measurement_does_not_seed_the_scale():
    """R1-verify: an align step whose aux gradient is a negligible fraction of
    the CE one (nb/na < scale_min_ratio) must NOT seed the scale (the ratio
    na/nb is noise there — the measured counterexample seeded s=9.9e5)."""
    lb = LossBalancer(align=True, align_every=4, eval_interval=100)
    a, b = _toy()
    # a tiny aux: d(aux)/db = 1e-6 * 2 * b, nb/na ~ 1e-6 < 0.05
    lb.backward(a ** 2 + b ** 2, {'x': 1e-6 * b ** 2}, [a, b], step=0)
    assert lb.last_path == 'align'
    assert lb.scale_ema is None, f'the noise measurement seeded s={lb.scale_ema}'
    # and the next step must keep aligning (uncertainty -> align), not explode
    a.grad = b.grad = None
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=1)
    assert lb.last_path == 'align'


def test_the_measured_scale_is_hard_capped():
    lb = LossBalancer(align=True, align_every=0, eval_interval=100, scale_max=2.0)
    a, b = _toy()
    # a small-but-valid aux (nb/na ~ 0.07) with a perfect cos -> raw scale ~14
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)
    assert lb.last_path == 'align'
    assert lb.scale_ema is not None and lb.scale_ema <= 2.0 + 1e-9, \
        f'the scale cap is not applied: {lb.scale_ema}'


def test_the_cheap_path_freezes_the_gradalign_target():
    """R3-verify: the freeze must be visible DURING the backward (the previous
    version of this test was false-green — it only checked the after-state)."""
    lb = LossBalancer(align=True, align_every=2, eval_interval=100)

    class _L:
        _ga_record = True
        seen = None

    class _M:
        layers = [_L()]

    m = _M()
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0, phase_model=m)
    assert m.layers[0]._ga_record is True
    a.grad = b.grad = None
    # a hook on the aux tensor reads the flag WHILE the backward runs
    aux = 0.1 * b ** 2
    aux.register_hook(lambda g: setattr(m.layers[0], 'seen', m.layers[0]._ga_record) or g)
    lb.backward(a ** 2 + b ** 2, {'x': aux}, [a, b], step=1, phase_model=m)
    assert lb.last_path == 'balance'
    assert m.layers[0].seen is False, \
        'the hook was NOT frozen during the cheap backward (false green before)'
    assert m.layers[0]._ga_record is True, 'the flag was not restored'


def test_align_false_backward_is_the_documented_raw_sum():
    """R1-verify: the align=False legacy path is the raw weighted sum (s=1)."""
    lb = LossBalancer(align=False, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)
    assert lb.last_path == 'balance'
    assert abs(float(a.grad) - 2.0) < 1e-3
    assert abs(float(b.grad) - (2.0 + 0.2)) < 1e-3, f'raw sum expected: {float(b.grad)}'


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
