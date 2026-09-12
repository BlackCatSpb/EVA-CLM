"""B17: migrate_state_dict must not drop CURRENT logit-cache projections.

Audit 05 found that the legacy decision-#3 cleanup matched k_proj_l /
v_proj_l / logit_to_hidden BY NAME — the same names the code-space
replacement later reused — so EVERY resume wiped those live weights back to
init. Gate is now shape-identity against the live model.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack                    # noqa: E402
from core.migrate import migrate_state_dict             # noqa: E402


def _model():
    cfg = EVAConfig(n_layers=1, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=300, save_dir='.', memory_bank=False, intent_bridge=False)
    torch.manual_seed(0)
    return EVAStack(cfg)


def test_live_cache_projections_survive_migrate():
    m = _model()
    sd = m.state_dict()
    k = 'logit_cache.attention.k_proj_l.weight'
    assert k in sd, 'model lacks the live projection the test is about'
    out, _n = migrate_state_dict(dict(sd), m)
    assert k in out, 'B17 regression: live K-shaped projection dropped by name'
    assert torch.equal(out[k], sd[k])


def test_legacy_v_shaped_projection_is_still_dropped():
    m = _model()
    sd = dict(m.state_dict())
    k = 'logit_cache.attention.k_proj_l.weight'
    sd[k] = torch.zeros(sd[k].shape[0], 300)      # legacy (D, V) input dim
    out, _n = migrate_state_dict(sd, m)
    assert k not in out, 'legacy V-dim tensor must be dropped (re-init safe: zero gate)'
