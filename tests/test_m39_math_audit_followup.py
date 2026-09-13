"""M39 locks (math-analysis follow-up): twin_free assert, retired dead code,
single-source LLRD default, and the README's corrected claims stay corrected."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_e10_twin_free_requires_k64():
    with pytest.raises(ValueError, match='code_dim'):
        EVAConfig(codebook='twin_free', code_dim=32)
    assert EVAConfig(codebook='twin_free', code_dim=64).code_dim == 64
    assert EVAConfig().codebook == 'legacy'          # default unaffected


def test_e9_no_dead_branch_after_adamp_return():
    src = open(os.path.join(ROOT, 'core', 'eva_optim.py'),
               encoding='utf-8', errors='replace').read()
    assert 'uf = u.reshape(-1)' not in src
    i = src.find('def _adamp_project')
    seg = src[i:src.find('def _resolve_role')]
    tail = seg[seg.rindex('return u'):]
    assert tail.strip() == 'return u'


def test_r5_cli_index_llrd_retired_by_default():
    src = open(os.path.join(ROOT, 'scripts', 'train.py'),
               encoding='utf-8', errors='replace').read()
    assert "'--llrd', type=float, default=1.0" in src
    assert 'tau-LLRD (single source' in src


def test_readme_corrected_markers():
    src = open(os.path.join(ROOT, 'README.md'),
               encoding='utf-8', errors='replace').read()
    # retired lies must not return
    for bad in ('fp64-кумулянты', 'готовность_моста',
                '(l−1)/(L−1)·(1+0.3·devₗ)'):
        assert bad not in src, bad
    # the truth markers
    assert 'cumsum(Δ₀·softplus(dev_eff))' in src
    assert 'τ₀ ≈ 9.5' in src and 'e^{−k/τ_s}' in src
    assert 'Полный bind' in src and 'изометрична именно поворотная' in src
