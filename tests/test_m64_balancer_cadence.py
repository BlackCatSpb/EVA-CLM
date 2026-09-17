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


def test_align_every_zero_is_pure_balance():
    lb = LossBalancer(align=True, align_every=0, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)
    assert lb.last_path == 'balance'
    assert lb.last_cos is None


def test_the_cheap_path_is_the_normalized_single_backward():
    """The balance path: one backward of ce + beta * sum(v/ema_v)."""
    lb = LossBalancer(align=True, align_every=0, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)
    # first call: ema_ce = |ce| = 2, ema_aux = 0.1, A = 0.1/0.1 = 1, beta = 2
    # b.grad = d(ce)/db + beta * d(aux)/db / ema_aux = 2 + 2*0.2/0.1 = 6
    assert abs(float(a.grad) - 2.0) < 1e-3
    assert abs(float(b.grad) - 6.0) < 1e-2


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


def test_state_dict_roundtrips_the_emas():
    lb = LossBalancer(align=True, align_every=0, eval_interval=100)
    a, b = _toy()
    lb.backward(a ** 2 + b ** 2, {'x': 0.1 * b ** 2}, [a, b], step=0)  # fills EMAs
    sd = lb.state_dict()
    assert sd.get('align_every') == 0
    assert sd['ema_ce'] is not None and sd['ema_aux']
    lb2 = LossBalancer(align=True, eval_interval=100)
    lb2.load_state_dict(sd)
    assert lb2.ema_ce == sd['ema_ce']
    assert lb2.ema_aux == sd['ema_aux']
