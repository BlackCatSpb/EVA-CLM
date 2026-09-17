"""M64 locks: the phantom bank's lifecycle arithmetic (the M63-C findings).

Two structural defects were measured on the live run:
1. the hard merge=0.7 is UNREACHABLE on D=2560 lacuna residuals (max
   confidence = conf_init for the whole run => not a single merge), so every
   unmatched observation evicted a slot — the bank was a snapshot of the last
   <=16 residuals, not a memory;
2. the decay ran per training FORWARD (~8 head forwards per step with the
   knowledge/reasoning passes), so the slot life (~100 steps) sat below the
   confirmation time (~200 steps) by arithmetic.

The first landing (soft route + drop + per-observe decay) was REJECTED by the
R1/R2 review: `count > 3` in the archive condition + drop froze the bank on
its first <=16 residuals FOREVER (12/16 slots of the live checkpoint had
count=1), `merge_lo=0.25` was uncalibrated (the null best-cos on D=2560 is
~0.06 mean / 0.09 max), and the decay was still tied to the forward count.
This version: no archive grace (conf_init > archive, so a fresh slot can never
archive immediately), merge_lo=0.2 (between the null and the recurring mode),
decay 0.99 per observe (~69 observes ~ 287 steps at the observed cadence),
telemetry (dropped/archived/cos percentiles), and the tests cover the freeze
regression and the late-recurrence spec.
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
    """F3 (M63-C): independent random residuals must not churn the bank.

    D=512 is used deliberately: the null best-cos scales as ~1/sqrt(D), so the
    D=64 geometry of a naive test has a ~0.125 std and false-merges random
    noise through the soft route (the R1 review measured 4/4 false confirms).
    The production geometry is D=2560 (null max ~0.09).
    """
    b = PhantomBank(n_slots=4, D=512)
    torch.manual_seed(0)
    for _ in range(200):
        b.observe(torch.randn(8, 512), torch.full((8,), 5.0), 0.1)
    st = b.stats()
    assert st['obs'] == 200
    # the recycling is archival-driven; 12 was the measured value with zero
    # margin (R3 nit) -> a loose bound still catches a churn (births ~= obs)
    assert st['births'] <= 24, f'the bank churned: {st}'
    assert st['merged'] == 0, f'random noise was merged (merge_lo too low): {st}'
    assert st['archived'] >= 1, 'the neglected slots never archived (the freeze)'


def test_immortal_slots_archive_after_neglect():
    """R2/R3 regression: the old `count > 3` grace made never-recurring slots
    (count=1) immortal — the live checkpoint had 12/16 of them."""
    b = PhantomBank(n_slots=4, D=512)
    d = _unit(512, 1)
    b.observe(d, torch.tensor([5.0]), 0.1)          # one birth, count=1
    assert b.stats()['phantoms'] == 1
    # orthogonal (random) observations must eventually retire it
    torch.manual_seed(3)
    for _ in range(300):
        b.observe(torch.randn(4, 512), torch.full((4,), 5.0), 0.1)
    st = b.stats()
    assert st['archived'] >= 1, f'the count=1 slot is immortal: {st}'
    assert st['births'] > 1, 'no recycling after the archive'


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


def test_late_recurrence_confirms_after_prefill():
    """The M63-C F4 spec: a PREFILLED bank + a recurrence arriving later must
    still confirm (the first landing failed this: the prefilled slots blocked
    the new direction forever — drop with no reachable archival)."""
    b = PhantomBank(n_slots=4, D=512)
    torch.manual_seed(7)
    for _ in range(8):                               # prefill with random
        b.observe(torch.randn(4, 512), torch.full((4,), 5.0), 0.1)
    torch.manual_seed(9)
    for _ in range(90):                              # the prefilled fade out
        b.observe(torch.randn(4, 512), torch.full((4,), 5.0), 0.1)
    assert b.stats()['archived'] >= 1, 'the prefilled slots never retired'
    d = _unit(512, 11)
    for _ in range(60):                              # the late recurrence
        b.observe(d + 0.005 * torch.randn(4, 512), torch.full((4,), 5.0), 0.1)
    st = b.stats()
    assert st['confirmed'] >= 1, f'the late recurrence never confirmed: {st}'


def test_soft_route_accumulates_partial_confidence():
    """A moderately similar direction (cos in [merge_lo, merge)) bumps conf."""
    b = PhantomBank(n_slots=4, D=64, merge_lo=0.2, merge=0.9)
    d = _unit(64, 2)
    b.observe(d, torch.tensor([5.0]), 0.1)          # birth, conf = conf_init
    c0 = float(b.confidence[0])
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
