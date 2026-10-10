"""B10: locks for audit 02b (docs/audit/02b_layer_dynamics.md)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack          # noqa: E402
from core.bind import TrajectorySpiralBind    # noqa: E402

import _srclock as srclock                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg():
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', bind_twist_mode='trajectory_spiral',
                    logit_cache_enabled=False, memory_bank=False, intent_bridge=False)
    torch.manual_seed(0)
    return cfg


def test_b10_in_graph_shift_is_mode_identical():
    cfg = _cfg()
    b = TrajectorySpiralBind(cfg.D, cfg.bind_K, cfg)
    x = torch.randn(1, 32, cfg.D)
    b.train()
    o_train, t_train = b(x, None)[0], b(x, None)[1]
    b.eval()
    with torch.no_grad():
        o_eval, t_eval = b(x, None)[0], b(x, None)[1]
    assert torch.allclose(o_train, o_eval, atol=1e-5), 'train/eval spiral split'
    assert (t_train - t_eval).abs().max() < 1e-5


def test_b10_eval_cannot_kill_the_trajectory_carry():
    cfg = _cfg()
    m = EVAStack(cfg).train()
    blk = m.layers[0]
    m.train()
    m(m.embed(torch.randint(0, cfg.vocab, (1, 32))), None, step=0)
    cached = getattr(blk, '_traj_state', None)
    assert cached is not None and cached.abs().max() > 1e-6, 'no traj carry written'
    m.eval()
    with torch.no_grad():
        m(m.embed(torch.randint(0, cfg.vocab, (1, 32))), None)
    after = getattr(blk, '_traj_state', None)
    assert torch.equal(after, cached), 'eval mutated the training traj cache'


def test_b10_pen_is_normalized_surprise():
    cfg = _cfg()
    m = EVAStack(cfg).train()
    m(m.embed(torch.randint(0, cfg.vocab, (1, 32))), None, step=0)
    pen = m.layers[0].mirror._cached_pred_error_norm
    assert pen is not None
    hi = float(pen.max())
    assert hi < 2.0, f'pen back at inflated scale (max {hi:.2f}) -> ladder killer'
    f = 1.0 - (torch.sigmoid(pen) - torch.sigmoid(torch.zeros_like(pen)))
    assert float(f.min()) > 0.75, f'pen_decay_factor pinned at {float(f.min()):.2f}'

def test_b10_concept_layer_index_copy_is_dtype_explicit():
    # Батч 6: inspect.getsource(модуля) + построчный текст -> AST-вызовы
    # `X.index_copy(0, ...)`: третий аргумент обязан содержать `.to(` (явный
    # dtype). Строже текста: ловит и многострочные вызовы, не зависти от
    # того, на какой строке оказался 'index_copy(0, it'.
    import ast
    src = os.path.join(ROOT, 'core', 'concept_layer.py')
    checked = 0
    bad = []
    for site in srclock.call_sites(src, 'index_copy'):
        if len(site.args) < 3:
            continue
        first = site.args[0]
        if not (isinstance(first, ast.Constant) and first.value == 0):
            continue
        checked += 1
        third = srclock.unparse(site.args[2])
        if '.to(' not in third:
            bad.append(srclock.unparse(site))
    assert checked >= 4, f'index_copy(0, ...) calls vanished? checked={checked}'
    assert not bad, f'regression: dtype-less index_copy: {bad}'

