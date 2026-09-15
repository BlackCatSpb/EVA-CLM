"""M55a locks: P5 (the dead base removed from the normalized path), P6 (the
saturation counter), the head<->memory contradiction tempering, the
lacuna-driven memory-search broadening."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core.memory_bank import _memory_attention  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**{**SMALL, **kw}))


def _hq(m):
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    return x, m.embed_tokens(x)


def test_p5_base_is_dead_in_the_normalized_path():
    m = _model().eval()
    h = torch.randn(1, 8, m.cfg.D)
    zt, zd, e_l = m.lm_head._gates(h, return_data=True)
    u, base = m.lm_head._su(zt, zd)
    lg = m.lm_head(h)
    raw = u @ m.lm_head.codes.T + m.lm_head.token_bias
    want = raw - raw.logsumexp(-1, keepdim=True)
    assert torch.allclose(lg, want, atol=1e-5), 'the base is not dead / the forward drifted'


def test_p6_saturation_counter():
    m = _model(head_u_wall=1e-3).train()
    x, h = _hq(m)
    m(h, None, step=1, tokens=x)
    sat = float(m.lm_head._last_sat)
    assert 0.0 <= sat <= 1.0
    with torch.no_grad():
        m.lm_head.readout.mul_(80.0)
    x2, h2 = _hq(m)
    m(h2, None, step=2, tokens=x2)
    assert float(m.lm_head._last_sat) > 0.5, 'saturated bits not counted'


def test_memory_search_broadens_with_temperature():
    q = torch.randn(2, 4, 32)
    k = F.normalize(torch.randn(8, 32), dim=-1)
    a1 = _memory_attention(q, k, torch.tensor(1.0), 32, True, None)
    a3 = _memory_attention(q, k, torch.tensor(3.0), 32, True, None)
    ent = lambda a: float(-(a * (a + 1e-9).log()).sum(-1).mean())   # noqa: E731
    assert ent(a3) > ent(a1), 'a higher temp did not broaden the retrieval'


def test_head_memory_conflict_tempering():
    m = _model(memory_bank=True, head_temper=True, head_temper_after=0).train()
    m.memory_bank._min_write_maturation = 0.0   # the maturation ramp is data-driven
    x, h = _hq(m)
    out, st, gs, _ = m(h, None, step=3000, tokens=x)
    assert getattr(m.lm_head, '_mem_dir', None) is not None, 'the stack never passed the memory read'
    assert hasattr(m.lm_head, '_last_conflict'), 'the conflict was not measured'
    assert 0.0 <= float(m.lm_head._last_conflict)
    # a big k must change the logits vs k=0
    m.lm_head.temper_k = 0.0
    x2, h2 = _hq(m)
    out0, _, _, _ = m(h2, None, step=3001, tokens=x2)
    m.lm_head.temper_k = 5.0
    m.lm_head._mem_dir = torch.randn_like(m.lm_head._mem_dir)
    x3, h3 = _hq(m)
    out5, _, _, _ = m(h3, None, step=3002, tokens=x3)
    assert not torch.allclose(out0, out5), 'the tempering had no effect'


def test_telemetry_has_sat_and_conflict():
    m = _model(memory_bank=True, head_temper=True, head_temper_after=0).train()
    m.memory_bank._min_write_maturation = 0.0
    x, h = _hq(m)
    m(h, None, step=3000, tokens=x)
    tel = m.head_telemetry()
    assert 'sat' in tel and 'conflict' in tel
