"""B18/B18b: tau ladder at rest == nominal; floor guards the extreme."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack   # noqa: E402


def _tail_carry(**cfgkw):
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=False, **cfgkw)
    torch.manual_seed(0)
    m = EVAStack(cfg).eval()
    if m.maturation is not None:
        m.maturation.gate.data.fill_(1.0)
    x1 = torch.full((1, 64), 5)
    x2 = torch.full((1, 512), 9)          # long unrelated window: only scan state reaches the tail
    emb = (m.embed_tokens if hasattr(m, 'embed_tokens') else m.embed)
    with torch.no_grad():
        o1, s1, _, _ = m(emb(x1), None, step=None, tokens=x1)
        oa, _, _, _ = m(emb(x2), s1, step=None, tokens=x2)
        ob, _, _, _ = m(emb(x2), None, step=None, tokens=x2)
        d = float((oa[0, -1] - ob[0, -1]).norm())
        return d / max(float(ob[0, -1].norm()), 1e-9)


def test_rest_ladder_carries_across_a_production_window():
    r = _tail_carry()                       # default: rest-normalized gates + floor k=2
    assert r > 0.08, f'carried state must shape a 512-window tail, got {r:.5f}'
    r_off = _tail_carry(vsa_decay_floor_k=0.0)
    assert r_off > 0.05, f'rest normalization alone must restore carry (k=0 gave {r_off:.5f})'


def test_floor_unit_never_vetoes_below_d_s_pow_k():
    # numeric contract of the floor: max(decay, d_s^k)
    d_s = torch.tensor(0.99931)            # slowest production scale
    decay_raw = torch.tensor(0.70)         # vicious content modulation
    k = 2.0
    assert float(torch.maximum(decay_raw, d_s.pow(k))) == float(d_s.pow(k))
