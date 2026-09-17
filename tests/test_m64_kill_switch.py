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


def test_measure_kill_freezes_the_gradalign_hook():
    """R1/R2 round-1 blocker: the measurement's aux traversals fired the
    gradalign hook and overwrote its CE-only target (the M64.4 defect class)."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    m = EVAStack(EVAConfig(**{**SMALL, 'gradalign_weight': 0.3,
                              'maturation_enabled': False})).train()
    lb = LossBalancer(eval_interval=100, kill_terms=['a'], kill_disable=False)
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=1, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    # a real backward first: the hook records the CE-only target
    ce.backward(retain_graph=True)
    tgt_before = [getattr(l, '_gradalign_tgt', None) for l in m.layers]
    tgt_before = [None if t is None else t.clone() for t in tgt_before]
    # the measurement must NOT touch the target (the hook is frozen)
    lb.measure_kill(ce, {'a': (aux.get('pred') if isinstance(aux.get('pred'), torch.Tensor)
                               else sum(v for v in aux.values() if isinstance(v, torch.Tensor)))},
                    m.parameters(), phase_model=m)
    for i, (b, a) in enumerate(zip(tgt_before, [getattr(l, '_gradalign_tgt', None) for l in m.layers])):
        if b is None or a is None:
            continue
        assert torch.equal(b, a), f'layer {i}: the measurement overwrote the target'
    assert all(getattr(l, '_ga_record', True) for l in m.layers), 'the flag was not restored'


def test_t12_hysteresis_does_not_oscillate():
    """R3 round-1 (T12): a proj hovering at the threshold must not flap."""
    ks = AuxKillSwitch(['a'], eps_off=0.5, eps_on=0.9, dwell=2, per_call=1, disable=True)
    lb = LossBalancer(eval_interval=100)
    p = torch.nn.Parameter(torch.tensor(1.0))
    q = torch.nn.Parameter(torch.tensor(1.0))
    # alternate a hostile and a friendly aux: the Schmitt band (0.5..0.9) must
    # keep it off once tripped (no oscillation)
    for i in range(6):
        aux = -(p ** 2) if i % 2 == 0 else 0.5 * (p ** 2)
        ks.measure(lb, p ** 2, {'a': aux}, [p])
    sw = ks.state['a']['switches']
    assert sw <= 2, f'the switch oscillated: {sw} switches'
