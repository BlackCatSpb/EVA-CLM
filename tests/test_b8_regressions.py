"""B8: locks for audit-02a HIGH/MED findings (docs/audit/02a_code_geometry.md)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack                              # noqa: E402
from core.training_control import (codebook_fingerprint,          # noqa: E402
                                   verify_identity_resume)


def _mini(**kw):
    base = dict(n_layers=1, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                vocab=400, save_dir='.')
    base.update(kw)
    cfg = EVAConfig(**base)
    torch.manual_seed(0)
    return EVAStack(cfg).train()


def test_f2a02_fingerprint_deterministic_and_geometry_sensitive():
    a = codebook_fingerprint(_mini())
    b = codebook_fingerprint(_mini())
    assert a == b and a != 'none'
    c = codebook_fingerprint(_mini(code_dim=32))
    assert c != a


def test_f2a03_identity_mismatch_is_fatal_nonidentity_is_not():
    m = _mini()
    fp = codebook_fingerprint(m)
    with pytest.raises(RuntimeError, match='identity tensors'):
        verify_identity_resume(m, {'code_fp': fp}, ['embed.basis', 'misc.thing'])
    with pytest.raises(RuntimeError, match='fingerprint'):
        verify_identity_resume(m, {'code_fp': 'bogus'}, [])
    assert verify_identity_resume(m, {'code_fp': fp}, ['misc.thing']) == fp
    assert verify_identity_resume(m, {}, ['misc.thing']) == fp  # legacy: warn only


def test_f2a04_embed_rope_footgun_retired():
    with pytest.raises(NotImplementedError):
        EVAConfig(embed_rope=True)


def test_f2a08_no_double_tanh_in_inference_profile_path():
    import inspect
    from core.logit_cache import LogitAttention
    src = inspect.getsource(LogitAttention.forward)
    assert 'torch.tanh(cached' not in src, 'outer tanh restored -> double-squash'
