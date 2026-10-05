"""Regression checks for scripts/t4_bounded_ab.py error handling (post-T4 crash).

Covers:
  * fmt_num on None/NaN/Inf/str -- no TypeError, no fake zeros;
  * run_arm early-NaN stop -> status='failed_nan', steps_done=0, NaN NOT
    zero-substituted (metrics stay n/a);
  * log/print paths when train_ce / eval_ce are None (the Colab T4
    ``TypeError: unsupported format string passed to NoneType`` crash);
  * summarize_ab on failed_nan arms (advice text, no exception);
  * AMP defaults: --ab fp32 unless explicit --amp; --dry-run keeps the fp16
    attempt unless --no-amp (checked with IS_CUDA monkeypatched True).

Run: python scripts/test_t4_bounded_ab_errors.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import time
import types
import unittest.mock

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_SPEC = importlib.util.spec_from_file_location(
    't4_bounded_ab', os.path.join(_HERE, 't4_bounded_ab.py'))
t4 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(t4)


def _fake_data(vocab=256, n=4096):
    rng = np.random.default_rng(0)
    return {
        'train': [rng.integers(0, vocab, size=n, dtype=np.uint16) for _ in range(2)],
        'hold': rng.integers(0, vocab, size=n, dtype=np.uint16),
        'train_files': ['a.bin', 'b.bin'],
        'holdout_file': 'h.bin',
    }


def _args(steps, eval_every):
    return types.SimpleNamespace(steps=steps, eval_every=eval_every, seed=7,
                                 windows=1, cap=4096)


def test_fmt_num():
    assert t4.fmt_num(None) == 'n/a'
    assert t4.fmt_num(float('nan')) == 'nan'
    assert t4.fmt_num(float('inf')) == 'inf'
    assert t4.fmt_num(3.14159, 2) == '3.14'
    assert t4.fmt_num('junk') == 'n/a'


def test_early_nan_stop():
    data = _fake_data()
    args = _args(steps=20, eval_every=10)
    calls = {'n': 0}

    def fake_step(*a, **k):
        calls['n'] += 1
        return 'nonfinite_loss', float('nan')

    buf = io.StringIO()
    with unittest.mock.patch.object(t4, 'train_step', fake_step):
        with contextlib.redirect_stdout(buf):
            res = t4.run_arm('smoke', True, args, data, time.time() + 120, False)
    out = buf.getvalue()
    assert res['failed_nan'] is True, res
    assert res['status'] == 'failed_nan', res
    assert res['steps_done'] == 0, res
    assert res['nan_step'] == 0 and res['nan_status'] == 'nonfinite_loss', res
    assert res['train_ce_first50'] is None and res['train_ce_last50'] is None
    assert res['train_ce_min'] is None
    assert calls['n'] == 1, calls
    assert 'FAILED_NAN' in out
    assert 'advice' in out
    assert 'TypeError' not in out


def test_none_formatting_paths():
    data = _fake_data()
    args = _args(steps=2, eval_every=1)

    def fake_step(*a, **k):
        return 'ok', float('nan')

    def fake_eval(*a, **k):
        return None

    buf = io.StringIO()
    with unittest.mock.patch.object(t4, 'train_step', fake_step), \
            unittest.mock.patch.object(t4, 'evaluate_mini', fake_eval):
        with contextlib.redirect_stdout(buf):
            res = t4.run_arm('smoke', False, args, data, time.time() + 120, False)
    out = buf.getvalue()
    assert res['steps_done'] == 2, res
    assert res['history'] and res['history'][0]['eval_ce'] is None, res
    assert 'eval_ce=n/a' in out, out
    assert 'train_ce=nan' in out, out

    results = {'use_amp': False, 'arms': {'bounded': res, 'baseline': res}}
    with tempfile.TemporaryDirectory() as td:
        txt = os.path.join(td, 'summary.txt')
        text, delta = t4.summarize_ab(results, args, txt)
        assert os.path.exists(txt)
    assert delta is None
    assert 'n/a' in text


def test_summary_failed_nan_advice():
    base = {
        'params': 1, 'init_fp': 'x', 'steps_done': 0, 'steps_target': 20,
        'stopped_early': False, 'status': 'failed_nan', 'failed_nan': True,
        'use_amp': True, 's_per_step': None, 'tok_per_s': None,
        'train_ce_first50': None, 'train_ce_last50': None, 'train_ce_min': None,
        'history': [], 'cfg': {'D': 512},
        'data': {'train_files': ['a'], 'holdout_file': 'h'},
    }
    results = {'use_amp': True,
               'arms': {'bounded': dict(base), 'baseline': dict(base)}}
    args = _args(20, 10)
    with tempfile.TemporaryDirectory() as td:
        text, delta = t4.summarize_ab(results, args, os.path.join(td, 's.txt'))
    assert delta is None
    assert 'EARLY-NaN STOP' in text
    assert '--no-amp' in text
    assert 'no A/B verdict' in text


def test_amp_defaults():
    calls = []

    def fake_ab(args, use_amp, scale='ab'):
        calls.append(('ab', use_amp, args.amp, args.no_amp))
        return 0

    def fake_dry(args, use_amp):
        calls.append(('dry', use_amp, args.amp, args.no_amp))
        return 0

    old_ab, old_dry, old_cuda = t4.run_ab, t4.run_dry_run, t4.IS_CUDA
    t4.run_ab, t4.run_dry_run, t4.IS_CUDA = fake_ab, fake_dry, True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            t4.main(['--ab', '--steps', '1'])
            t4.main(['--ab', '--amp', '--steps', '1'])
            t4.main(['--dry-run', '--steps', '1'])
            t4.main(['--dry-run', '--no-amp', '--steps', '1'])
    finally:
        t4.run_ab, t4.run_dry_run, t4.IS_CUDA = old_ab, old_dry, old_cuda
    assert calls == [
        ('ab', False, False, False),
        ('ab', True, True, False),
        ('dry', True, False, False),
        ('dry', False, False, True),
    ], calls


def run_all():
    tests = [test_fmt_num, test_early_nan_stop, test_none_formatting_paths,
             test_summary_failed_nan_advice, test_amp_defaults]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f'[PASS] {fn.__name__}', flush=True)
        except Exception as e:
            failed += 1
            print(f'[FAIL] {fn.__name__}: {type(e).__name__}: {e}', flush=True)
    print(f'[test_t4] {len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(run_all())
