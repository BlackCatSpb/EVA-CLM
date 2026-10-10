"""M39 locks (math-analysis follow-up): twin_free assert, retired dead code,
single-source LLRD default, and the README's corrected claims stay corrected.

Батч 6 (source-локи -> AST): код-локи переведены на AST-узлы (последний
return, отсутствие имени uf, точный вызов add_argument с help-строкой).
README — текстовый артефакт (не парсится как Python): остаётся строгим
raw-поиском фраз; ослабления нет.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402

import _srclock as srclock  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_e10_twin_free_requires_k64():
    with pytest.raises(ValueError, match='code_dim'):
        EVAConfig(codebook='twin_free', code_dim=32)
    assert EVAConfig(codebook='twin_free', code_dim=64).code_dim == 64
    assert EVAConfig().codebook == 'legacy'          # default unaffected


def test_e9_no_dead_branch_after_adamp_return():
    src = os.path.join(ROOT, 'core', 'eva_optim.py')
    fn = srclock.find_def(src, '_adamp_project')
    assert fn is not None, '_adamp_project vanished'
    # мёртвый хвост ловится точнее подстрочного лока: последний стейтмент
    # функции обязан быть ровно `return u` (любой код после return сделал бы
    # последним другой узел)
    assert srclock.unparse(fn.body[-1]) == 'return u', \
        f'dead code after return u: last stmt is {srclock.unparse(fn.body[-1])!r}'
    # удалённый мёртвый дубль `uf = u.reshape(-1)` — имя uf в AST не существует
    assert not srclock.has_name(src, 'uf'), 'dead `uf` branch restored'


def test_r5_cli_index_llrd_retired_by_default():
    src = os.path.join(ROOT, 'scripts', 'train.py')
    assert srclock.has_call(src, 'add_argument', args=["'--llrd'"],
                            kwargs={'type': 'float', 'default': '1.0'}), \
        'the --llrd default changed (index LLRD must stay retired)'
    assert any('tau-LLRD (single source' in v for v in srclock.str_values(src)), \
        'the single-source LLRD help note vanished'


def test_readme_corrected_markers():
    src = srclock.read(os.path.join(ROOT, 'README.md'))
    # retired lies must not return
    for bad in ('fp64-кумулянты', 'готовность_моста',
                '(l−1)/(L−1)·(1+0.3·devₗ)'):
        assert bad not in src, bad
    # the truth markers
    assert 'cumsum(Δ₀·softplus(dev_eff))' in src
    assert 'τ₀ ≈ 9.5' in src and 'e^{−k/τ_s}' in src
    assert 'Полный bind' in src and 'зометрична именно поворотная' in src
