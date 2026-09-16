"""M24→M28 locks: the run never auto-stops, and (since M28, operator policy)
the ONLY training interventions live in core.adaptation — LossBalancer, AGC
(incl. non-finite gradient drop), the LR controller and depth. The loops must
not veto, guard, skip, anneal or damp anything: spikes train through and the
operator watches the log."""
import json
import os

import pytest

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(R, 'notebooks', 'eva_colab.ipynb')
TR = os.path.join(R, 'scripts', 'train.py')


def _cell10():
    nb = json.load(open(NB, encoding='utf-8'))
    for c in nb['cells']:
        s = ''.join(c.get('source', []))
        if 'TRAINING LOOP' in s:
            return s
    raise AssertionError('training-loop cell not found')


class TestLoopIsPure:
    FORBIDDEN = ('[veto', 'veto:soft', 'nan-guard', '[update-skip]',
                 'watchdog.check', 'phase_scales', '_anneal', '_ce_uni',
                 'arm_ce', 'mean_mirror_scale', 'Soft EOS-aware',
                 'memgov', '_v16',
                 'if math.isfinite(disp_loss)', 'STOPPED ON ALARM',
                 'ALARM:LOG', '[ALARM:WARN]')

    def _srcs(self):
        return {'cell10': _cell10(), 'train.py': open(TR, encoding='utf-8', errors='replace').read()}

    def test_no_loop_interventions(self):
        for name, src in self._srcs().items():
            for bad in self.FORBIDDEN:
                assert bad not in src, f'{name} still contains {bad!r}'

    def test_adaptation_calls_survive(self):
        for name, src in self._srcs().items():
            for keep in ('balancer.backward', 'clipper.clip', 'apply_tau_lr',
                        'scheduler.step()', 'release_step_graph'):
                assert keep in src, f'{name} lost required {keep!r}'

    def test_run_never_stops(self):
        s = _cell10()
        assert '_alarm_stop' not in s
        assert 'sys.exit' not in s
        t = self._srcs()['train.py']
        assert 'sys.exit(2)' not in t
