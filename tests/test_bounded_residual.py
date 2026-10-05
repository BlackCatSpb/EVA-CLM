# -*- coding: utf-8 -*-
"""Stage 3: bounded residual (OPT-IN) locks.

(i)   bounded_residual=False => forward БИТ-В-БИТ равен старому пути;
(ii)  bounded_residual=True  => нормы потока ограничены, W_g/b_g градиенты живые;
(iii) snapshot/restore + state_dict-резюм не ломаются.

Эталон: st1b_bounded.py (bounded_v2), калиброванный стенд scripts/bench_calibrated.py.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


def _cfg(**kw):
    base = dict(n_layers=2, D=64, mlp_groups=2, code_dim=16, code_sparsity=4,
                vocab=256, memory_bank=True, mem_min_write_mat=0.0,
                gradient_checkpointing=False, logit_cache_enabled=False,
                maturation_enabled=False, seq_len=16, batch_size=1)
    base.update(kw)
    return EVAConfig(**base)


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(_cfg(**kw)).train()


def _tokens(L=16, V=256, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (1, L), generator=g)


def _fwd(m, x, scale=1.0, seed=1234):
    # train-forward has stochastic noise (block.py noise_scale, head spike
    # gen): the parity tests reseed both generators (same as the stand).
    torch.manual_seed(seed)
    _ng = getattr(m.lm_head, '_noise_gen', None)
    if _ng is not None:
        _ng.manual_seed(seed)
    h = m.embed_tokens(x) * scale
    out, _, _, _ = m(h, step=1, tokens=x, adaptive=False)
    return out


def test_off_is_bit_identical():
    m_off = _model(bounded_residual=False)
    m_on = _model(bounded_residual=True)
    # bounded-параметры созданы последними => core-веса обеих сборок равны
    sd_off, sd_on = m_off.state_dict(), m_on.state_dict()
    assert 'memory_bank.gate_W' in sd_on and 'bounded_post_norm_w.0' in sd_on
    assert 'memory_bank.gate_W' not in sd_off
    for k in set(sd_off) & set(sd_on):
        assert torch.equal(sd_off[k], sd_on[k]), f'core weight diverged: {k}'
    # тот же core, ветка выключена на лету => старый путь
    m_on._bounded_residual = False
    m_on.memory_bank._bounded_residual = False
    x = _tokens()
    y_off = _fwd(m_off, x)
    y_prev = _fwd(m_on, x)
    assert torch.equal(y_off, y_prev), 'flag-off output is not bit-identical'


def _flow_norms(model, x, scale=1.0):
    """Per-layer post-norm flow norms (stack telemetry, Stage 3)."""
    model._bounded_flow = []
    _fwd(model, x, scale=scale)
    return list(model._bounded_flow)


def test_on_bounds_stream_and_gate_grads_alive():
    m = _model(bounded_residual=True)
    # stand calibration: the fusion head is zero-init (old path is a no-op at
    # init too); without it unit(fused)=0 and the gate is dead by construction.
    with torch.no_grad():
        _std = 1.0 / (m.cfg.D ** 0.5)
        m.memory_bank.fusion[-1].weight.normal_(0.0, _std)
        m.memory_bank.fusion[-1].bias.normal_(0.0, _std)
    x = _tokens()
    w_max = float(m.bounded_post_norm_w[0].detach().max())
    bound = w_max * (m.cfg.D ** 0.5) + 1e-3
    norms = _flow_norms(m, x)
    assert norms and max(norms) <= bound, (norms, bound)
    assert min(norms) > 0.1
    # boundedness is scale-free: 1000x input stays under the same bound
    assert max(_flow_norms(m, x, scale=1000.0)) <= bound

    # telemetry: Delta/h ~ O(1e-2) on the calibrated scale
    m.zero_grad(set_to_none=True)
    m.memory_bank._bounded_log = []
    h = m.embed_tokens(x) * 30.0
    out, _, _, _ = m(h, step=1, tokens=x, adaptive=False)
    out.pow(2).mean().backward()
    log = m.memory_bank._bounded_log
    assert log, 'bounded telemetry is empty'
    assert all(entry['ratio'] < 0.5 for entry in log), log
    gw, gb = m.memory_bank.gate_W.grad, m.memory_bank.gate_b.grad
    assert gw is not None and gb is not None
    assert torch.isfinite(gw).all() and float(gw.norm()) > 0.0
    assert torch.isfinite(gb).all() and float(gb.norm()) > 0.0


def test_scalar_gate_arm_bounds_too():
    m = _model(bounded_residual=True, bounded_residual_gate=False)
    x = _tokens()
    assert all(torch.isfinite(t).all() for t in [_fwd(m, x, scale=100.0)])


def test_snapshot_restore_and_state_dict_resume():
    # snapshot/restore is the eval-isolation contract (M8): eval + no_grad.
    m = _model(bounded_residual=True).eval()
    x = _tokens()
    with torch.no_grad():
        y0 = _fwd(m, x)
        snap = m.snapshot_runtime_buffers()
        y1 = _fwd(m, x)
        m.restore_runtime_buffers(snap)
        y2 = _fwd(m, x)
    assert torch.equal(y1, y2), 'snapshot/restore changed the forward'

    src = _model(bounded_residual=True).eval()
    m2 = _model(bounded_residual=True).eval()
    m2.load_state_dict(src.state_dict(), strict=True)
    assert torch.equal(m2.memory_bank.gate_W, src.memory_bank.gate_W)
    assert torch.equal(m2.bounded_post_norm_w[0], src.bounded_post_norm_w[0])
    src.memory_bank.reset()
    m2.memory_bank.reset()
    with torch.no_grad():
        assert torch.equal(_fwd(src, x), _fwd(m2, x)), 'resume roundtrip diverged'


def test_gc_true_bounded_residual_train_step_and_gate_grads():
    """M65-opt x Stage 3: gradient_checkpointing=True (recompute) must keep the
    bounded path trainable — finite output, alive W_g/b_g and post-norm grads.
    OFF under gc=True is bit-identical to OFF under gc=False (same seed);
    ON+gc=True is checked by finiteness + nontrivial grads (the forward itself
    is not bit-comparable across gc: recompute + stochastic block internals)."""
    x = _tokens()
    off_gc = _model(bounded_residual=False, gradient_checkpointing=True)
    off_no = _model(bounded_residual=False, gradient_checkpointing=False)
    assert torch.equal(_fwd(off_gc, x), _fwd(off_no, x)), \
        'flag-off + gc=True diverged from gc=False'
    m = _model(bounded_residual=True, gradient_checkpointing=True)
    # stand calibration (as in the non-gc test): the fusion head is zero-init;
    # without it unit(fused)=0 and the gate is dead by construction.
    with torch.no_grad():
        _std = 1.0 / (m.cfg.D ** 0.5)
        m.memory_bank.fusion[-1].weight.normal_(0.0, _std)
        m.memory_bank.fusion[-1].bias.normal_(0.0, _std)
    m._bounded_flow = []
    torch.manual_seed(1234)
    _ng = getattr(m.lm_head, '_noise_gen', None)
    if _ng is not None:
        _ng.manual_seed(1234)
    h = m.embed_tokens(x) * 30.0
    out, _, _, _ = m(h, step=1, tokens=x, adaptive=False)   # train-forward
    assert torch.isfinite(out).all()
    assert len(m._bounded_flow) == m.cfg.n_layers, \
        'post-norm flow telemetry is empty under gc=True'
    m.zero_grad(set_to_none=True)
    out.pow(2).mean().backward()
    for name, p in (('gate_W', m.memory_bank.gate_W),
                    ('gate_b', m.memory_bank.gate_b),
                    ('bounded_post_norm_w.0', m.bounded_post_norm_w[0])):
        g = p.grad
        assert g is not None, f'{name}: grad is None under gc=True'
        assert torch.isfinite(g).all() and float(g.norm()) > 0.0, \
            f'{name}: dead grad under gc=True (norm={float(g.norm()):.3g})'
