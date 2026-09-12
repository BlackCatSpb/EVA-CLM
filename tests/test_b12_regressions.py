"""B12: locks for audit 03 (docs/audit/03_losses_optimization.md)."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack                      # noqa: E402
from core.training_control import LossBalancer            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stack():
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=False)
    torch.manual_seed(0)
    return EVAStack(cfg).train(), cfg


# F3-07: legacy scheduler state without the damp counter must not crash
def test_f307_damp_resume_from_legacy_state():
    from core.lr_scheduler import MirrorLRScheduler
    m, cfg = _stack()
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    s = MirrorLRScheduler(m, opt, base_lr=1e-3, cfg=cfg)
    s.report_val_loss(10.0)
    # simulate resume from a pre-B10 pickle: keep _best_val_loss, drop the counter
    del s._lr_damp_steps
    s.report_val_loss(20.0)            # regression path touches the counter
    s.report_val_loss(20.0)            # plateau path too
    assert hasattr(s, '_lr_damp_steps')  # lock = no AttributeError (the crash)


# F3-01: empty validation pool is NaN, never a flattering 0.0
def test_f301_empty_val_pool_is_not_a_perfect_score():
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('_tr_b12', os.path.join(ROOT, 'scripts', 'train.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m, cfg = _stack()
    for hn in (0, 1, None):
        v = mod.evaluate(m, [], cfg, 'cpu', hold_n=hn)
        assert math.isnan(v), f'empty pool must be NaN (F3-01), got {v} (hold_n={hn})'


# F3-04 (B6 refinement): priority terms must survive the max_terms cap
def test_ggeo_priority_not_alphabetical():
    ce = torch.tensor(1.0, requires_grad=True)
    aux = {n: (2.0 + i) * ce for i, n in enumerate(
        ['zeta', 'yard', 'delta', 'balance', 'branch', 'diversity',
         'pred', 'bridge_conn', 'xray', 'whiskey'])}
    geo = LossBalancer(align=False).grad_geometry(ce, aux, [ce], max_terms=6)
    assert set(geo) >= {'bridge_conn', 'pred', 'diversity', 'branch', 'balance'}, geo


# B12: ggeo recompute-freeze guards the trajectory step counter
def test_ggeo_freeze_blocks_step_count_side_effects():
    from core.bind import TrajectorySpiralBind
    cfg = EVAConfig(n_layers=1, D=128, code_dim=16, code_sparsity=4, vocab=200,
                    save_dir='.', bind_twist_mode='trajectory_spiral')
    torch.manual_seed(0)
    b = TrajectorySpiralBind(cfg.D, cfg.bind_K, cfg)
    x = torch.randn(1, 8, cfg.D)
    b._step_count += 0                      # materialize
    b._ggeo_freeze = True
    n0 = int(b._step_count.clone().cpu().numpy()[0]) if torch.is_tensor(b._step_count) else b._step_count
    b(x, None)
    n1 = int(b._step_count.clone().cpu().numpy()[0]) if torch.is_tensor(b._step_count) else b._step_count
    assert n1 == n0, 'frozen recompute still ticked the counter'
    b._ggeo_freeze = False
    b(x, None)
    n2 = int(b._step_count.clone().cpu().numpy()[0]) if torch.is_tensor(b._step_count) else b._step_count
    assert n2 == n1 + 1


# source locks: train.py by-name restore + post-construction CLI overrides
def test_train_py_wiring_locks():
    t = open(os.path.join(ROOT, 'scripts', 'train.py'), encoding='utf-8').read()
    assert "_restore_optimizer(optimizer, model, ckpt['optimizer'])" in t
    assert "cfg.warmup_steps = args.warmup" in t
    assert "float('nan')" in t


# F3-05: u_gate threshold rebased to the post-B10 pen scale
def test_f305_uncertainty_threshold_rebased():
    from core.concept_layer import UnifiedConceptLayer
    src = open(os.path.join(ROOT, 'core', 'concept_layer.py'), encoding='utf-8').read()
    assert 'log_tau_uncert = nn.Parameter(torch.tensor(-0.6931))' in src
    import math as _m
    assert abs(_m.exp(-0.6931) - 0.5) < 1e-3     # half-open at typical surprise
