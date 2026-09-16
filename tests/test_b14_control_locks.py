"""B14: locks for audit 04 remaining MEDIUMs (docs/audit/04_control_machine.md)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mini():
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=False)
    torch.manual_seed(0)
    return EVAStack(cfg).train(), cfg


def test_f411_m33_eval_writes_caches_snapshot_isolates():
    # M33 SUPERSEDES the F4-11 mode gate: streaming caches (hp/pen/pred_k) are
    # FORWARD INPUTS — erasing them on eval made validation a different model
    # (+~12 nats measured on step-1045 best.pt). Parity now: eval writes.
    # Isolation moved to where it belongs: the runtime snapshot covers the
    # caches and the evaluator restores them byte-exact after the val pass.
    m, cfg = _mini()
    x = torch.randint(3, cfg.vocab, (1, 32))
    m.train()
    m(m.embed_tokens(x) if hasattr(m, 'embed_tokens') else m.embed(x), None, step=0, tokens=x)
    mir = m.layers[0].mirror
    sentinel = torch.full_like(mir._cached_pred_error_norm, 7.77)
    mir._cached_pred_error_norm = sentinel
    snap = m.snapshot_runtime_buffers()
    m.eval()
    with torch.no_grad():
        inp = m.embed_tokens(x) if hasattr(m, 'embed_tokens') else m.embed(x)
        m(inp, None, adaptive=False, tokens=x)
    assert not torch.equal(mir._cached_pred_error_norm, sentinel), \
        'M33 parity broken: eval no longer updates streaming caches'
    m.restore_runtime_buffers(snap)
    assert torch.equal(mir._cached_pred_error_norm, sentinel), \
        'M33: snapshot/restore must bring the train-document cache back byte-exact'


def test_f410_bypass_only_path_is_bounded():
    from core.training_control import LossBalancer
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 2, bias=False)
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    ce = ((lin(x) - y) ** 2).mean()
    params = list(lin.parameters())
    g_ce = torch.autograd.grad(2 * ce, params, retain_graph=True)
    cn = float(torch.stack([g.norm() for g in g_ce]).norm())
    lb = LossBalancer(align=True)
    lb.zero = None
    lin.zero_grad(set_to_none=True)
    big = 100.0 * ce                      # huge bypass-only ledger
    lb.backward(ce, {'gradalign': big}, params, phase_model=None)
    got = float(torch.stack([p.grad.norm() for p in params]).norm())
    assert got <= 2.05 * cn, f'bypass-only aux unbounded: {got:.3f} vs CE {cn:.3f}'


def test_f406_depth_integrator_roundtrip():
    from core.adaptation import DepthController
    m, cfg = _mini()
    d = DepthController(m, n_layers=2, init_k=1, unfreeze_inc=1,
                        warmup_steps=0, eval_interval=2, max_depth=2)
    d.update(10, 5.0)
    d.update(12, 5.0)
    st = d.get_state()
    assert st['last_depth_step'] != -10 ** 9 or st['val_ema'] is not None
    d2 = DepthController(m, n_layers=2, init_k=1, unfreeze_inc=1,
                         warmup_steps=0, eval_interval=2, max_depth=2)
    d2.put_state(st)
    assert d2._val_ema == d._val_ema and d2._last_depth_step == d._last_depth_step

