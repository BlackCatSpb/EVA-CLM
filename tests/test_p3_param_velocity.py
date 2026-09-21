"""P3-4: ParamVelocity — the parameter-efficiency KPI."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.param_velocity import ParamVelocity  # noqa: E402


class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(8, 8)
        self.tiny = nn.Parameter(torch.zeros(4))          # near-zero init -> skipped
        self.frozen = nn.Parameter(torch.randn(4), requires_grad=False)


def test_velocity_moves_freezes_and_skips_tiny():
    torch.manual_seed(0)
    m = _M()
    pv = ParamVelocity(m, frac=0.5, min_coords=4)
    pv.sample(m)
    with torch.no_grad():
        m.lm_head.weight.add_(0.1)                        # move
        m.tiny.add_(1.0)                                  # from a zero norm
    pv.sample(m)
    assert pv.vel.get('lm_head.weight', 0.0) > 0.0
    assert 'tiny' not in pv.vel, 'a near-zero-norm param has no meaningful rel velocity'
    assert 'frozen' not in pv.idx, 'requires_grad=False params are not tracked'
    rep = pv.report()
    assert rep.get('head', 0.0) > 0.0
    sd = pv.state_dict()
    pv2 = ParamVelocity(m, frac=0.5, min_coords=4)
    pv2.load_state_dict(sd)
    assert pv2.vel == pv.vel


def test_buckets():
    assert ParamVelocity._bucket('lm_head.readout') == 'head'
    assert ParamVelocity._bucket('layers.0.mirror.W_proj') == 'mirror'
    assert ParamVelocity._bucket('layers.3.mlp.gate') == 'mlp'
    assert ParamVelocity._bucket('layers.3.bind.W_proj.weight') == 'bind'
    assert ParamVelocity._bucket('logit_cache.attention.q_proj.weight') == 'cache'
    assert ParamVelocity._bucket('lm_head.pair_V1') == 'newborn'
    assert ParamVelocity._bucket('layers.0.mirror.phantom_basis') == 'newborn'
    assert ParamVelocity._bucket('layers.5.whatever') == 'trunk'
