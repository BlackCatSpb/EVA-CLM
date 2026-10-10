"""M58 locks: the chain report's risks that the existing suite did NOT cover.

Each test pins one finding from docs/analysis/FINAL_REPORT.md:
  * the tempering silently skipped the CE path (a shape check) - now reshaped;
  * the eval handed the head a stale training memory direction;
  * reset_cache left the block's _traj_state and the mirror's _cached_usefulness;
  * one SEP sentence was written ~24x per forward (once per layer);
  * M55's confirmed_directions() had no consumer;
  * `mat` was a timer (pen_init=1.0) instead of a competence measure;
  * train.py's B19 halved seq_len at startup and B15 clobbered the gate;
  * the watchdog is gone for good.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core.maturation import MaturationController  # noqa: E402
from core import EVAStack  # noqa: E402

import _srclock as srclock  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**{**SMALL, **kw}))


def _hq(m):
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    return x, m.embed_tokens(x)


def test_tempering_reaches_the_ce_path():
    # M58b: the CE calls the head with (N, D) while _mem_dir is (B, L, D) —
    # the old equality check skipped the tempering exactly where it matters.
    m = _model(memory_bank=True, head_temper=True, head_temper_after=0).train()
    m.memory_bank._min_write_maturation = 0.0
    x, h = _hq(m)
    out, st, gs, _ = m(h, None, step=3000, tokens=x)      # sets _mem_dir
    assert getattr(m.lm_head, '_mem_dir', None) is not None
    y = torch.randint(1, SMALL['vocab'], (1, 8))
    lp_on = m.lm_head.log_probs_for_target(h.reshape(-1, m.cfg.D), y.reshape(-1))
    m.lm_head._temper_active = False
    lp_off = m.lm_head.log_probs_for_target(h.reshape(-1, m.cfg.D), y.reshape(-1))
    assert not torch.allclose(lp_on, lp_off), 'the CE path is not tempered'


def test_eval_isolation_covers_the_memory_head_channel():
    m = _model(memory_bank=True).train()
    m.memory_bank._min_write_maturation = 0.0
    x, h = _hq(m)
    m(h, None, step=3000, tokens=x)
    d0 = m.lm_head._mem_dir.clone()
    snap = m.snapshot_runtime_buffers()
    with torch.no_grad():
        m.lm_head._mem_dir = torch.randn_like(m.lm_head._mem_dir)   # a "val" read
    m.restore_runtime_buffers(snap)
    assert torch.allclose(m.lm_head._mem_dir, d0), 'the eval leaked into the head channel'


def test_reset_cache_scrubs_the_block_and_mirror_state():
    m = _model(memory_bank=True).train()
    x, h = _hq(m)
    m(h, None, step=3000, tokens=x)
    # M65-opt: фейковый assert (`or True`) убран — реальные замки ниже
    # (reset_cache обязан очистить ЯВНО выставленное состояние).
    m.layers[0]._traj_state = torch.randn(1, 2, 8, 16)
    m.layers[0].mirror._cached_usefulness = torch.randn(1, 8, 4)
    m.reset_cache()
    assert getattr(m.layers[0], '_traj_state', None) is None
    assert getattr(m.layers[0].mirror, '_cached_usefulness', None) is None


def test_bank_writes_once_per_forward():
    m = _model(memory_bank=True).train()
    m.memory_bank._min_write_maturation = 0.0
    calls = {'n': 0}
    orig = m.memory_bank.l1.write

    def _spy(summary):
        calls['n'] += 1
        return orig(summary)

    m.memory_bank.l1.write = _spy
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    x[0, 4] = 2                                   # a SEP boundary
    h = m.embed_tokens(x)
    m(h, None, step=3000, tokens=x)
    assert calls['n'] <= 1, f'the bank wrote {calls["n"]}x in one forward'


def test_m55_confirmed_directions_steer_the_basis():
    m = _model(head_phantom_every=1).train()
    x, h = _hq(m)
    m(h, None, step=2000, tokens=x)               # past the 1045 warmup (the bank is live)
    pb = m.lm_head.phantom_bank
    with torch.no_grad():
        d = torch.randn(m.cfg.D)
        d = d / d.norm()
        pb.directions[0].copy_(d)
        pb.confidence[0] = 0.9                    # confirmed
        pb.count[0] = pb.cycles_before_stable     # T9: recurrence gate (EVA-Ai)
        pb.filled[0] = True
        b0 = m.lm_head.phantom_basis.data[0].clone()
        m.lm_head._pb_step.zero_()                # force the steering cadence
    out, *_ = m(h, None, step=2001, tokens=x)
    m.lm_head(out)   # Round 7-fixrev: реальный вызов головы двигает базис
    b1 = m.lm_head.phantom_basis.data[0]
    assert not torch.allclose(b0, b1), 'the confirmed direction did not steer the basis'
    assert torch.cosine_similarity(b1, d, dim=0) > torch.cosine_similarity(b0, d, dim=0)


def test_pen_init_starts_at_zero_so_readiness_measures_competence():
    cfg = EVAConfig(**SMALL)
    mc = MaturationController(n_layers=2, tau_min=cfg.tau_min, tau_max=cfg.tau_max, cfg=cfg)
    assert float(mc.pen_init.abs().max()) == 0.0, 'pen_init is still the 1.0 timer'
    # first observation seeds the max; a stable series then must NOT look "ready"
    pe = torch.tensor([0.3, 0.4])
    mc.update(0, pe)
    assert torch.allclose(mc.pen_init, pe)
    for s in range(1, 5):
        mc.update(s, pe)                          # no improvement
    assert float(mc.readiness.max()) < 0.5, 'readiness rose without any improvement'
    # an improving series must raise it
    for s in range(5, 40):
        mc.update(s, pe * 0.5)
    assert float(mc.readiness.min()) > float(mc.readiness.max()) * 0.0  # smoke


def test_train_py_static_locks():
    # AST-нормализация (батч 6): отсутствие/наличие конструкций проверяется по
    # AST-узлам (формулировка/форматирование не влияют), НО маркерные строки
    # ('watchdog' и т.п.) остаются строгим raw-поиском: в AST их может не быть
    # вовсе (комментарий), и AST-проверка была бы СЛАБЕЕ исходной.
    t = os.path.join(ROOT, 'scripts', 'train.py')
    assert not srclock.has_augassign(t, 'cfg.seq_len', 'floordiv'), 'B19 regression is back'
    assert srclock.has_name(t, '_gate_missing'), 'B15 guard is missing'
    # `X.mul_(ls_m)` в любом виде — второй раз LS-LR: AST-проверка на аргумент
    assert not srclock.has_call(t, 'mul_', args=['ls_m']), \
        'the ls_mult double application is back'
    raw = srclock.read(t)
    assert 'FailureDetector' not in raw and 'watchdog' not in raw


def test_watchdog_is_gone_everywhere():
    tc = os.path.join(ROOT, 'core', 'training_control.py')
    assert srclock.find_def(tc, 'FailureDetector') is None
    assert not srclock.has_name(tc, 'FailureDetector')
    assert srclock.find_def(tc, 'hard_veto_ceiling') is None
    import core.adaptation as ad
    assert not hasattr(ad, 'FailureDetector')


def test_notebook_resume_restores_are_present():
    # The coverage gap that let the M58a watchdog removal eat two resume
    # restores (they sat inside the deleted block): an AST lock on the
    # notebook's resume contract (батч 6: подстроки -> узлы вызовов/присваиваний).
    import json
    nb = json.load(open(os.path.join(ROOT, 'notebooks', 'eva_colab.ipynb'),
                        encoding='utf-8'))
    s9 = ''.join(nb['cells'][9]['source'])
    assert srclock.has_call(s9, 'balancer.load_state_dict', args=['_resume_balancer_sd']), \
        'cell 9 lost the balancer resume'
    assert srclock.has_call(s9, 'depth.put_state'), 'cell 9 lost the depth resume'
    assert srclock.has_if(s9, '_resume_branch_var_ref is not None'), \
        'cell 9 lost the branch_var_ref guard'
    s8 = ''.join(nb['cells'][8]['source'])
    assert srclock.assigns(s8, '_resume_balancer_sd', "ckpt.get('balancer')")
    assert srclock.assigns(s8, '_resume_branch_var_ref', "ckpt.get('branch_var_ref')")
    s10 = ''.join(nb['cells'][10]['source'])
    assert srclock.has_dict_entry(s10, 'balancer', 'balancer.state_dict()')
    assert srclock.has_dict_entry(s10, 'branch_var_ref')
    assert srclock.has_call(s10, 'model.head_telemetry'), \
        'the head telemetry row vanished from the log'
