"""M64 locks: the phantom bank's lifecycle arithmetic (the M63-C findings).

Two structural defects were measured on the live run:
1. the hard merge=0.7 is UNREACHABLE on D=2560 lacuna residuals (max
   confidence = conf_init for the whole run => not a single merge), so every
   unmatched observation evicted a slot — the bank was a snapshot of the last
   <=16 residuals, not a memory;
2. the decay ran per training FORWARD (~8 head forwards per step with the
   knowledge/reasoning passes), so the slot life (~100 steps) sat below the
   confirmation time (~200 steps) by arithmetic.

M64 adds the soft route (merge_lo, a similarity-weighted EMA + partial bump),
drops unmatched observations when the bank is full (no eviction churn), and
moves the decay to the observe cadence.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.phantom import PhantomBank  # noqa: E402
from core.config import EVAConfig  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _unit(D, seed):
    torch.manual_seed(seed)
    v = torch.randn(1, D)
    return v / v.norm()


def test_no_churn_on_random_residuals():
    """F3 (M63-C): independent random residuals must NOT churn the bank."""
    b = PhantomBank(n_slots=4, D=64)
    torch.manual_seed(0)
    for _ in range(200):
        b.observe(torch.randn(8, 64), torch.full((8,), 5.0), 0.1)
    st = b.stats()
    assert st['obs'] == 200
    assert st['births'] <= 4, f'the bank churned: {st}'


def test_recurring_direction_reaches_confirmation():
    """F4 (M63-C): a recurring direction must be able to CONFIRM."""
    b = PhantomBank(n_slots=4, D=64)
    d = _unit(64, 1)
    for _ in range(30):
        e = d + 0.01 * torch.randn(4, 64)
        b.observe(e, torch.full((4,), 5.0), 0.1)
    st = b.stats()
    assert st['confirmed'] >= 1, f'no confirmation: {st}'
    assert st['merged'] >= 5, f'the recurrence was not merged: {st}'


def test_soft_route_accumulates_partial_confidence():
    """A moderately similar direction (cos in [merge_lo, merge)) bumps conf."""
    b = PhantomBank(n_slots=4, D=64, merge_lo=0.2, merge=0.9)
    d = _unit(64, 2)
    b.observe(d, torch.tensor([5.0]), 0.1)          # birth, conf = conf_init
    c0 = float(b.confidence[0])
    # a direction at ~cos 0.5: mix d with an orthogonal unit vector
    o = _unit(64, 3)
    o = o - (o @ d.T) * d
    o = o / o.norm()
    e = (0.866 * d + 0.5 * o)
    b.observe(e, torch.tensor([5.0]), 0.1)
    c1 = float(b.confidence[0])
    assert c1 > c0, 'the soft route did not bump the confidence'
    assert b.stats()['merged'] >= 1


def test_decay_is_per_observe_not_per_forward():
    b = PhantomBank(n_slots=2, D=8, decay=0.5)
    d = _unit(8, 3)
    b.observe(d, torch.tensor([5.0]), 0.1)           # birth: conf 0.5
    c0 = float(b.confidence[0])
    b.observe(-d, torch.tensor([5.0]), 0.1)          # opposite: no merge; decays once
    c1 = float(b.confidence[0])
    assert abs(c1 - c0 * 0.5) < 1e-6, f'not one decay per observe: {c0} -> {c1}'


def test_config_knob_reaches_the_bank():
    torch.manual_seed(0)
    m = EVAStack(EVAConfig(**{**SMALL, 'head_phantom_merge_lo': 0.33}))
    assert abs(m.lm_head.phantom_bank.merge_lo - 0.33) < 1e-12
