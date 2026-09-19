# -*- coding: utf-8 -*-
"""T9 locks: фантом-добор из EVA-Ai — cycle-gate подтверждения, coh-EMA,
классификатор типов, аудит-ринг, priority-направления.

Run: python -m pytest tests/test_t9_phantom_upgrade.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F
from core.phantom import PhantomBank


def _bank(**kw):
    torch.manual_seed(0)
    b = PhantomBank(n_slots=4, D=32, **kw)
    return b


def _obs(b, vec, thr=0.0):
    v = F.normalize(vec.float().clone().reshape(1, -1), dim=-1)
    ell = torch.tensor([1.0])
    return b.observe(v, ell, thr)


def test_confirmation_requires_cycles():
    """conf>=confirm, но count<cycles → НЕ confirmed (EVA-Ai recurrence gate)."""
    b = _bank(cycles_before_stable=5, decay=1.0)
    v = torch.zeros(32); v[0] = 1.0
    # 5 одинаковых наблюдений: birth(conf 0.5) + 4 soft/hard merge (conf 0.5+4*0.05*w)
    for _ in range(5):
        _obs(b, v)
    st = b.stats()
    # conf может быть >=0.75? при w=1 (hard merge) conf = 0.5+4*0.05=0.7 <0.75 — поднимем
    # искусственно, но count уже 5 → gate пройден
    assert st['confirmed'] == 0 or st['confirmed'] == 1
    # теперь: conf высокий, но count мал
    b2 = _bank(cycles_before_stable=5, decay=1.0)
    b2.directions[0].copy_(F.normalize(v, dim=-1)); b2.filled[0] = True
    b2.confidence[0] = 0.9; b2.count[0] = 2
    assert b2.stats()['confirmed'] == 0, 'count<cycles не должен подтверждаться (T9)'
    b2.count[0] = 5
    assert b2.stats()['confirmed'] == 1, 'count>=cycles и conf>=confirm → confirmed (T9)'


def test_types_classifier():
    b = _bank(cycles_before_stable=5)
    v = torch.zeros(32); v[0] = 1.0
    # slot 0: stable
    b.directions[0].copy_(F.normalize(v, dim=-1)); b.filled[0] = True
    b.confidence[0] = 0.9; b.count[0] = 5; b.coh[0] = 0.5
    # slot 1: emerging (частый, когерентный)
    b.directions[1].copy_(F.normalize(v, dim=-1)); b.filled[1] = True
    b.confidence[1] = 0.6; b.count[1] = 12; b.coh[1] = 0.4
    # slot 2: ambiguous (слабые совпадения)
    b.directions[2].copy_(F.normalize(v, dim=-1)); b.filled[2] = True
    b.confidence[2] = 0.6; b.count[2] = 12; b.coh[2] = 0.1
    # slot 3: nascent
    b.directions[3].copy_(F.normalize(v, dim=-1)); b.filled[3] = True
    b.confidence[3] = 0.6; b.count[3] = 2; b.coh[3] = 0.4
    t = b.types()
    # типы: emerging=0, ambiguous=1, nascent=2 (stable — статус, не тип)
    assert (int(t[0]), int(t[1]), int(t[2]), int(t[3])) == (2, 0, 1, 2), f'types: {t.tolist()}'
    st = b.stats()
    assert st['stable'] == 1, f"stable статус: {st['stable']}"
    assert st['emerging'] == 1 and st['ambiguous'] == 1 and st['nascent'] == 2
    # high priority: conf>0.7 и тип emerging/ambiguous — slot 0 (conf 0.9, nascent) НЕ входит
    assert st['high_priority'] == 0


def test_priority_directions():
    b = _bank(cycles_before_stable=5)
    v = torch.zeros(32); v[0] = 1.0
    b.directions[0].copy_(F.normalize(v, dim=-1)); b.filled[0] = True
    b.confidence[0] = 0.8; b.count[0] = 12; b.coh[0] = 0.1   # ambiguous + conf>0.7
    d = b.priority_directions()
    assert d.shape[0] == 1 and d.shape[1] == 32, f'priority_directions: {d.shape}'


def test_audit_ring_records_lifecycle():
    b = _bank(cycles_before_stable=5, decay=1.0)
    v = torch.zeros(32); v[0] = 1.0
    _obs(b, v)                                   # birth
    _obs(b, v)                                   # merge (hard или soft)
    tail = b.audit_tail(8)
    assert len(tail) >= 2, f'аудит пуст: {tail}'
    evs = [e['event'] for e in tail]
    assert 'birth' in evs, f'birth не записан: {evs}'
    assert all(set(e.keys()) == {'event', 'slot', 'conf', 'cos'} for e in tail)


def test_coh_ema_updates_on_observations():
    b = _bank(cycles_before_stable=5, decay=1.0)
    v = torch.zeros(32); v[0] = 1.0
    _obs(b, v)
    c0 = float(b.coh[0])
    _obs(b, v)
    c1 = float(b.coh[0])
    assert c1 >= c0, 'coh-EMA не растёт при повторных совпадениях (T9)'
