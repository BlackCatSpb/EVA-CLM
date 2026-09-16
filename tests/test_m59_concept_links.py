"""M59 locks: the two concept systems are linked (the operator's insight) and
the UCL's self-closure has an experiment lever.

A. the head's confirmed phantoms birth UCL concepts (birth_from_direction);
B. the UCL's active directions steer the head's phantom basis;
C. the phantom channel grows in place (capacity allocated once, the active
   count rises, NO shape ever changes, the forward is unchanged at the growth);
+ the read-scale floor (ucl_read_scale_floor) and the UCL telemetry keys.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core.migrate import migrate_state_dict  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**{**SMALL, **kw}))


def test_ucl_read_scale_floor_holds_then_releases():
    m = _model(ucl_read_scale_floor=0.5, ucl_read_scale_floor_until=100).train()
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    with torch.no_grad():
        m(h, None, step=0, tokens=x)
        assert float(m.concept_layer._last_scale) >= 0.5, 'floor not applied'
        m(h, None, step=200, tokens=x)
        assert float(m.concept_layer._last_scale) < 0.5, 'floor never released'


def test_link_a_external_birth_from_a_direction():
    m = _model().train()
    ucl = m.concept_layer
    d = F.normalize(torch.randn(m.cfg.D), dim=0)
    assert ucl.birth_from_direction(d, confidence=0.7) is True
    used = (ucl.concept_count > 0).nonzero()
    assert used.numel() >= 1
    i = int(used[0].item())
    v = ucl.concept_vals[i]
    assert float(F.cosine_similarity(v, d, dim=0)) > 0.99, 'the value is not the direction'
    k = ucl.concept_keys[i]
    k_want = F.normalize(ucl.q_proj(d).reshape(-1), dim=-1)
    assert torch.allclose(k, k_want, atol=1e-5), 'the key is not q_proj(direction)'
    assert abs(float(ucl.concept_confidence[i]) - 0.7) < 1e-6


def test_link_b_ucl_directions_steer_the_phantom_basis():
    m = _model(head_phantom_every=1).train()
    ucl = m.concept_layer
    d = F.normalize(torch.randn(m.cfg.D), dim=0)
    assert ucl.birth_from_direction(d, confidence=0.9)
    m.lm_head._ext_phantom_dirs = ucl.active_directions(min_conf=0.5)
    m.lm_head._pb_active = True
    with torch.no_grad():
        m.lm_head._pb_step.zero_()                 # force the steering cadence
    h = torch.randn(1, 8, m.cfg.D)
    b0 = m.lm_head.phantom_basis.data[0].clone()
    m.lm_head(h)
    b1 = m.lm_head.phantom_basis.data[0]
    assert not torch.allclose(b0, b1), 'the UCL direction did not steer the basis'
    assert (torch.cosine_similarity(b1, d, dim=0)
            > torch.cosine_similarity(b0, d, dim=0))


def test_link_c_growth_is_in_place_and_forward_neutral():
    m = _model(head_phantom_bits=8, head_phantom_max=12).eval()
    head = m.lm_head
    assert head.phantom_basis.shape[0] == 12, 'the capacity is not allocated upfront'
    assert int(head._kp_active) == 8
    h = torch.randn(1, 8, m.cfg.D)
    with torch.no_grad():
        lg0 = head(h)
        dirs = F.normalize(torch.randn(10, m.cfg.D), dim=-1)
        grew = head.grow_phantom_bits(dirs)
        assert grew == 4, f'expected 4 (the capacity), got {grew}'
        assert int(head._kp_active) == 12
        lg1 = head(h)
    assert torch.allclose(lg0, lg1, atol=1e-5), 'the growth changed the forward'


def test_link_c_growth_skips_covered_directions():
    m = _model(head_phantom_bits=4, head_phantom_max=16).eval()
    head = m.lm_head
    with torch.no_grad():
        # the existing active rows are 'covered' - asking for them again is a no-op
        covered = head.phantom_basis.data[:4].clone()
        assert head.grow_phantom_bits(covered) == 0
        # a fresh direction is added
        fresh = F.normalize(torch.randn(1, m.cfg.D), dim=-1)
        assert head.grow_phantom_bits(fresh) == 1
        assert int(head._kp_active) == 5


def test_migrate_pads_a_smaller_phantom_capacity():
    m = _model(head_phantom_bits=8, head_phantom_max=16)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    small = None
    for k in list(sd):
        if k.endswith('phantom_basis'):
            sd[k] = sd[k][:8].clone()              # simulate the old 8-row ckpt
            small = sd[k].clone()
        elif k.endswith('phantom_mix'):
            sd[k] = sd[k][:, :8].clone()
    sd.pop('_kp_active', None)
    new, changed = migrate_state_dict(sd, m)
    assert changed >= 2
    assert new['lm_head.phantom_basis'].shape[0] == 16
    assert new['lm_head.phantom_mix'].shape[1] == 16
    # the old prefix survives, the new rows are zero
    assert torch.allclose(new['lm_head.phantom_basis'][:8], small)
    assert float(new['lm_head.phantom_basis'][8:].abs().max()) == 0.0


def test_ucl_telemetry_reports_the_scale():
    m = _model().train()
    dg = m.concept_layer.get_diagnostics()
    for k in ('concept_read_scale', 'concept_write_alpha', 'concept_scale_floor',
              'concept_maturity'):
        assert k in dg, f'{k} missing from the UCL diagnostics'
