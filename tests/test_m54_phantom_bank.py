"""M54 locks: the phantom-concept bank (the EVA-Ai lacuna lifecycle at
hidden-state level) + the head telemetry (M53b)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core.phantom import PhantomBank  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**{**SMALL, **kw}))


def _unit(D, seed):
    torch.manual_seed(seed)
    v = torch.randn(1, D)
    return v / v.norm()


def test_bank_births_merges_and_confirms():
    b = PhantomBank(n_slots=4, D=64)
    d1 = _unit(64, 1)
    ell_hi = torch.tensor([0.5])
    assert b.observe(d1, ell_hi, 0.1) == 1
    assert b.stats()['phantoms'] == 1
    # the same direction merges and raises confidence
    for _ in range(6):
        b.observe(d1 + 0.01 * torch.randn(1, 64), ell_hi, 0.1)
    st = b.stats()
    assert st['phantoms'] == 1, 'the same phantom was not merged'
    assert st['confirmed'] == 1, f"not confirmed: {st}"
    # a different direction births a second phantom
    b.observe(_unit(64, 2), ell_hi, 0.1)
    assert b.stats()['phantoms'] == 2


def test_bank_respects_threshold():
    b = PhantomBank(n_slots=4, D=64)
    assert b.observe(_unit(64, 3), torch.tensor([0.01]), 0.1) == 0
    assert b.stats()['phantoms'] == 0


def test_bank_archives_faded_phantoms():
    b = PhantomBank(n_slots=4, D=64, decay=0.5, archive=0.25)
    d = _unit(64, 4)
    b.decay_rate = 1.0                       # M64: the fade rides in observe now
    for _ in range(5):                       # count > 3 so archival can apply
        b.observe(d, torch.tensor([0.5]), 0.1)
    assert b.stats()['phantoms'] == 1
    b.decay_rate = 0.5
    for _ in range(12):
        b.decay()                            # 0.7 * 0.5^12 << 0.25
    # the lifecycle check runs inside observe: a new (opposite) lacuna triggers it
    b.observe(-d, torch.tensor([0.5]), 0.1)
    st = b.stats()
    assert st['phantoms'] == 1, f'the faded phantom was not archived: {st}'
    assert float(b.directions[0].norm()) == 0.0, 'the archived slot was not freed'


def test_bank_state_rides_in_checkpoint():
    b = PhantomBank(n_slots=4, D=64)
    sd = b.state_dict()
    for k in ('directions', 'confidence', 'count', 'filled'):
        assert k in sd, f'{k} missing from the bank state'


def test_head_telemetry_reports_lacuna_srl_phantoms():
    m = _model(head_phantom_every=1, head_srl=True,
               head_srl_after=0, head_phantom_after=0).train()
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    out, st, gs, _ = m(h, None, step=1, tokens=x)
    with torch.no_grad():
        m.lm_head.ell_ladder[-1].mul_(0.2)  # T9.7: гейт читает середину VSA-лестницы      # M55b: a relative spike makes the bank observe
    out, st, gs, _ = m(h, None, step=2, tokens=x)
    tel = m.head_telemetry()
    assert 'lacuna' in tel and tel['lacuna'] >= 0.0
    assert 'srl_conf' in tel and 'srl_expl' in tel
    assert 'ph_phantoms' in tel and 'ph_confirmed' in tel
    assert tel['ph_obs'] >= 1, 'the bank never observed'


def test_confirmed_directions_are_unit():
    b = PhantomBank(n_slots=4, D=64)
    d = _unit(64, 5)
    for _ in range(7):   # M64: the per-observe decay 0.999 -> 7 bumps for 0.75
        b.observe(d, torch.tensor([0.5]), 0.1)
    cd = b.confirmed_directions()
    assert cd.shape[0] == 1
    assert abs(float(cd.norm()) - 1.0) < 1e-4
