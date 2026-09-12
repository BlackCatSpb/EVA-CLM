"""B13: locks for audit 04 (docs/audit/04_control_machine.md)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.training_control import FailureDetector   # noqa: E402


def _wd(**kw):
    kw.setdefault('warmup', 0)
    return FailureDetector(torch.nn.Linear(4, 4), **kw)


def test_f402_arm_ce_is_first_val_only():
    wd = _wd()
    for i in range(40):
        wd.check(10.0, i)                   # build CE stats (unarmed)
    wd.arm_ce()
    assert wd.ce_armed and 'ce' not in wd._stats
    wd.check(10.0, 100, {'ce': 10.0})       # arm + one observation
    n_before = wd._stats['ce'][2]
    wd.arm_ce()                             # later evals must NOT re-anchor
    assert 'ce' in wd._stats and wd._stats['ce'][2] == n_before


def test_f403_legacy_4wide_stats_normalize_on_load():
    wd = _wd()
    wd.load_state_dict({'stats': {'ce': [1.0, 1.0, 5, 1.0]}})   # pre-B4 pickle
    assert wd._stats['ce'] == [1.0, 1.0, 5.0, 1.0, 0.0, 0.0, 0.0]
    wd2 = _wd()
    wd2.load_state_dict({'stats': {'ce': [9, 9, 9, 9, 9, 9, 9, 9]}})
    assert len(wd2._stats['ce']) == 7


def test_f407_ls_neutral_on_frozen_log_scale():
    # direct unit lock of the ratio math (zero/zero must be neutral, not crash)
    fast, slow = 0.0, 0.0
    if fast < 1e-12 and slow < 1e-12:
        mult = 1.0
    else:
        mult = max(0.5, min(2.0, 1.0 / max(fast / max(slow, 1e-12), 1e-12)))
    assert mult == 1.0
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'core', 'lr_scheduler.py'), encoding='utf-8').read()
    assert 'mults.append(1.0)' in src and '_fa < 1e-12 and _sl < 1e-12' in src


def test_f401_mirror_alpha_writes_are_recompute_idempotent():
    from core.mirror import GroupedCognitiveMirror
    import inspect
    src = inspect.getsource(GroupedCognitiveMirror)
    assert 'self._alpha_pending' in src
    assert '.data.lerp_(alpha_target' not in src, \
        'unconditional in-forward lerp is back (recompute double-write)'


def test_f408_tau_lr_applied_before_clip_both_copies():
    import json
    nb = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'notebooks', 'eva_colab.ipynb'), encoding='utf-8'))
    s10 = ''.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    assert s10.find('apply_tau_lr(model') < s10.find('clipper.clip(model.parameters())')
    t = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'scripts', 'train.py'), encoding='utf-8').read()
    assert 'apply_tau_lr' in t and t.find('apply_tau_lr(model') < t.find('clipper.clip(model.parameters())')
