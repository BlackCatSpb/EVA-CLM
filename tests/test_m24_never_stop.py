"""M24→M28 locks: the run never auto-stops, and (since M28, operator policy)
the ONLY training interventions live in core.adaptation — LossBalancer, AGC
(incl. non-finite gradient drop), the LR controller and depth. The loops must
not veto, guard, skip, anneal or damp anything: spikes train through and the
operator watches the log.

Батч 6 (source-локи -> AST): обязательные вызовы контура адаптации проверяются
по AST-узлам (`has_call`). ЗАПРЕЩЁННЫЕ маркеры остаются строгим raw-поиском:
это маркеры-строки/комментарии из удалённых блоков, AST их не видит вовсе и
был бы СЛАБЕЕ (миграция невозможна без ослабления смысла).
"""
import json
import os

import _srclock as srclock

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
        return {'cell10': _cell10(), 'train.py': srclock.read(TR)}

    def test_no_loop_interventions(self):
        for name, src in self._srcs().items():
            for bad in self.FORBIDDEN:
                assert bad not in src, f'{name} still contains {bad!r}'

    def test_adaptation_calls_survive(self):
        for name, src in self._srcs().items():
            for keep in ('balancer.backward', 'clipper.clip', 'apply_tau_lr',
                         'scheduler.step', 'release_step_graph'):
                assert srclock.has_call(src, keep), f'{name} lost required {keep!r}'

    def test_run_never_stops(self):
        s = _cell10()
        assert '_alarm_stop' not in s
        assert not srclock.has_call(s, 'sys.exit')
        t = self._srcs()['train.py']
        assert not srclock.has_call(t, 'sys.exit', args=['2'])
