"""M23+M24 locks: the run never auto-stops; NaN is treated, not fatal.

Operator policy (live step-1 NaN incident, explicit): "remove everything that
stops training — I watch the log." M24 makes D6+ LOG-ONLY in both copies.
M23's nan-guard resets poisoned streaming state instead of feeding it back.
Remaining protections: M14 vetoes (bad batches never reach weights), B3
best.pt discipline (checkpoint advances only on improving isolated eval).
"""
import json
import os
import re

import pytest

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(R, 'notebooks', 'eva_colab.ipynb')
TR = os.path.join(R, 'scripts', 'train.py')


def _cell10():
    nb = json.load(open(NB, encoding='utf-8'))
    return ''.join(nb['cells'][10]['source'])


class TestRunNeverStops:
    def test_notebook_has_no_alarm_stop(self):
        s = _cell10()
        assert '_alarm_stop' not in s
        assert 'STOPPED ON ALARM' not in s
        assert 'sys.exit' not in s
        assert re.search(r"if _rb:[\s\S]{0,1600}?\[ALARM:LOG\]", s)

    def test_train_has_no_alarm_stop(self):
        t = open(TR, encoding='utf-8', errors='replace').read()
        assert 'STOPPED ON ALARM' not in t
        assert 'sys.exit(2)' not in t
        assert '[ALARM:LOG]' in t

    def test_sensor_message_says_log_only(self):
        tc = open(os.path.join(R, 'core', 'training_control.py'),
                  encoding='utf-8', errors='replace').read()
        assert 'log-only, operator supervises' in tc
        assert 'stop on confirmed second' not in tc


class TestNaNGuard:
    def test_guard_resets_stream_state_in_both(self):
        for src in (_cell10(), open(TR, encoding='utf-8', errors='replace').read()):
            i = src.find('[nan-guard]')
            assert i > 0
            blk = src[i:i + 700]
            for var in ('state = None', 'gs = None', 'intent_state = None'):
                assert var in blk, f'guard must reset {var}'
            assert 'continue' in blk
            assert 'break' not in blk.split('continue')[1][:200]  # not a loop-exit
