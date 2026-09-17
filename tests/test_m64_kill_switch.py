"""M64.12 lock: the aux kill-switch (round-robin geometry + Schmitt trigger)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core.training_control import LossBalancer, AuxKillSwitch  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def test_measure_only_never_disables():
    ks = AuxKillSwitch(['a', 'b'], disable=False)
    lb = LossBalancer(eval_interval=100)
    a = torch.nn.Parameter(torch.tensor(1.0))
    for _ in range(6):
        ks.measure(lb, a ** 2, {'a': a ** 2, 'b': 0.0 * a}, [a])
    assert ks.disabled() == set(), 'measure-only must never disable'
    assert ks.last, 'the measurement returned nothing'


def test_schmitt_with_dwell_and_revival():
    ks = AuxKillSwitch(['a'], eps_off=0.5, eps_on=0.9, dwell=2, per_call=1, disable=True)
    lb = LossBalancer(eval_interval=100)
    p = torch.nn.Parameter(torch.tensor(1.0))
    # an aux that FIGHTS the CE (same parameter: cos = -1 -> proj = 0)
    for _ in range(2):
        ks.measure(lb, p ** 2, {'a': -(p ** 2)}, [p])
    assert ks.disabled() == {'a'}, f'the dwell did not trip: {ks.state}'
    # a friendly aux (cos = 1, ratio = 1 -> proj = 1) revives it
    for _ in range(3):
        ks.measure(lb, p ** 2, {'a': (p ** 2)}, [p])
    assert ks.disabled() == set(), 'the revival failed'


def test_the_balancer_filters_the_disabled_terms():
    lb = LossBalancer(eval_interval=100, kill_terms=['a'], kill_disable=True)
    p = torch.nn.Parameter(torch.tensor(1.0))
    q = torch.nn.Parameter(torch.tensor(1.0))
    for _ in range(2):
        lb.measure_kill(p ** 2, {'a': -(q ** 2)}, [p, q])
    assert lb.kill.disabled() == {'a'}
    p.grad = q.grad = None
    lb.backward(p ** 2, {'a': -(q ** 2)}, [p, q], step=0)
    assert p.grad is not None
    assert q.grad is None or float(q.grad.abs().sum()) == 0.0, \
        'the disabled term still contributed a gradient'


def test_the_switch_state_roundtrips():
    lb = LossBalancer(eval_interval=100, kill_terms=['a'], kill_disable=True)
    p = torch.nn.Parameter(torch.tensor(1.0))
    for _ in range(2):
        lb.measure_kill(p ** 2, {'a': -(p ** 2)}, [p])
    sd = lb.state_dict()
    assert sd.get('kill') is not None
    lb2 = LossBalancer(eval_interval=100, kill_terms=['a'], kill_disable=True)
    lb2.load_state_dict(sd)
    assert lb2.kill.disabled() == {'a'}


def test_config_defaults_off():
    cfg = EVAConfig(**SMALL)
    assert cfg.aux_kill_switch is False and cfg.aux_kill_disable is False
