# -*- coding: utf-8 -*-
"""Очередь A100 блок 3: memory-only read (v4) + stability-guard (OPT-IN).

Контракты:
(a) memory_only_read=False => forward БИТ-В-БИТ старый путь; mo-параметров нет;
(b) ON => вклад ограничен по построению (unit·tanh, ||Δu||<=1) и норм-капом
    относительно логитов, градиенты mo_proj/mo_scale живые;
(c) guard срабатывает на синтетически патологическом вкладе (аттенюация/кап,
    NaN/Inf -> нулевой вклад + счётчик);
(d) snapshot/restore и state_dict-резюм не сломаны.

Мотивация: старый v4 без guard коллапсировал на s0 стенда
(scripts/bench_calibrated.py, 3000 шагов): eval 7.2->21.3 за ~250 шагов,
||Δz|| 0.07->0.40 (норма mo_proj 3.7->5.16) при g_bind 0.5->20.
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
    torch.manual_seed(seed)
    _ng = getattr(m.lm_head, '_noise_gen', None)
    if _ng is not None:
        _ng.manual_seed(seed)
    h = m.embed_tokens(x) * scale
    out, _, _, _ = m(h, step=1, tokens=x, adaptive=False)
    return out


def _calibrate_bank(m):
    """Стенд-калибровка: fusion zero-init => unit(fused)=0 и путь мёртв."""
    with torch.no_grad():
        _std = 1.0 / (m.cfg.D ** 0.5)
        m.memory_bank.fusion[-1].weight.normal_(0.0, _std)
        m.memory_bank.fusion[-1].bias.normal_(0.0, _std)


# ─── (a) OFF бит-в-бит ──────────────────────────────────────────────────────
def test_off_is_bit_identical_and_creates_no_params():
    m_off = _model(memory_only_read=False)
    m_on = _model(memory_only_read=True)
    sd_off, sd_on = m_off.state_dict(), m_on.state_dict()
    assert 'lm_head.mo_proj' in sd_on and 'lm_head.mo_scale' in sd_on
    assert 'lm_head.mo_proj' not in sd_off
    assert 'lm_head.mo_scale' not in sd_off
    assert getattr(m_off.lm_head, '_mo_on', False) is False
    assert m_off.lm_head.mo_proj is None and m_off.lm_head.mo_scale is None
    # mo-параметры созданы ЛОКАЛЬНЫМ RNG => core-веса обеих сборок равны
    for k in set(sd_off) & set(sd_on):
        assert torch.equal(sd_off[k], sd_on[k]), f'core weight diverged: {k}'
    # ON-модель с выключенным на лету флагом == OFF БИТ-В-БИТ (включая голову)
    m_on.lm_head._mo_on = False
    m_on.memory_bank._memory_only_read = False
    x = _tokens()
    y_off = _fwd(m_off, x)
    l_off = m_off.lm_head(y_off)
    y_prev = _fwd(m_on, x)
    l_prev = m_on.lm_head(y_prev)
    assert torch.equal(y_off, y_prev), 'flag-off stack output is not bit-identical'
    assert torch.equal(l_off, l_prev), 'flag-off head logits are not bit-identical'
    # OFF-путь не пишет mo-телеметрию и не ставит _mo_read
    assert m_off.lm_head._mo_stats == {}
    assert m_off.lm_head._mo_read is None


# ─── (b) ON ограничен, градиенты живые ──────────────────────────────────────
def test_on_bounds_contribution_and_grads_alive():
    m = _model(memory_only_read=True, bounded_residual=True)
    _calibrate_bank(m)
    m.lm_head.mo_scale.data.fill_(0.5)
    x = _tokens()
    out = _fwd(m, x)
    assert torch.isfinite(out).all()
    st = m.lm_head._mo_stats
    assert st and st['mo_gate'] > 0.0
    # структурная граница: unit·tanh => ||Δu|| <= |tanh(scale)|
    _gate = float(torch.tanh(m.lm_head.mo_scale.detach()).abs())
    assert st['mo_norm'] <= _gate + 1e-5
    # масштабная инвариантность: x1000 вход не взрывает вклад
    out_big = _fwd(m, x, scale=1000.0)
    assert torch.isfinite(out_big).all()
    st_big = m.lm_head._mo_stats
    assert st_big['mo_norm'] <= _gate + 1e-5
    # градиенты пути живые (CE-путь: голова вызывается по h стека)
    m.zero_grad(set_to_none=True)
    m.lm_head.mo_scale.data.fill_(0.5)
    out = _fwd(m, x)
    logits = m.lm_head(out)
    logits.pow(2).mean().backward()
    for name, p in (('mo_proj', m.lm_head.mo_proj),
                    ('mo_scale', m.lm_head.mo_scale)):
        g = p.grad
        assert g is not None, f'{name}: grad is None'
        assert torch.isfinite(g).all(), f'{name}: grad non-finite'
        assert float(g.norm()) > 0.0, f'{name}: dead grad'


# ─── (c) guard на патологическом вкладе (sabotage) ──────────────────────────
def test_guard_caps_synthetic_pathological_contribution():
    m = _model(memory_only_read=True)
    head = m.lm_head
    # синтетический раздутый вклад: ||Δ||=1000 при капе ~0.01
    dang = torch.full((1, 4, head.K), 1000.0)
    ref = torch.full((1, 4, 1), 0.01)
    n_before = head._mo_guard_n
    out_d, att = head._mo_cap_apply(dang, ref)
    cap = head._mo_cap_ratio * ref + head._mo_cap_floor
    got = out_d.detach().norm(dim=-1, keepdim=True)
    assert bool((got <= cap + 1e-6).all()), (float(got.max()), float(cap.max()))
    assert float(att.max()) < 1.0, 'attenuation did not engage'
    assert head._mo_guard_n == n_before + 1, 'guard counter did not increment'


def test_guard_caps_model_level_when_cap_tightened():
    m = _model(memory_only_read=True)
    head = m.lm_head
    _calibrate_bank(m)
    head.mo_scale.data.fill_(5.0)
    head._mo_cap_ratio = 0.0
    head._mo_cap_floor = 0.01          # жёсткий кап: ||Δlogits|| <= 0.01
    x = _tokens()
    out = _fwd(m, x)
    assert torch.isfinite(out).all()
    st = head._mo_stats
    assert st['mo_logit_norm'] <= 0.01 + 1e-4, st
    assert st['mo_att'] < 1.0
    assert st['mo_guard_n'] >= 1
    assert st['mo_norm'] <= 1.0 + 1e-5


def test_guard_zeros_nonfinite_read():
    m = _model(memory_only_read=True)
    head = m.lm_head
    _calibrate_bank(m)
    x = _tokens()
    _fwd(m, x)                          # ставит _mo_read из банка
    assert head._mo_read is not None
    n_before = head._mo_nan_n
    with torch.no_grad():
        head.mo_proj.fill_(float('inf'))
    out = head(m.embed_tokens(x))       # прямой вызов головы, без stack.forward
    assert torch.isfinite(out).all(), 'non-finite contribution leaked'
    assert head._mo_nan_n == n_before + 1, 'NaN counter did not increment'


# ─── (d) snapshot/restore/resume ────────────────────────────────────────────
def test_snapshot_restore_and_state_dict_resume():
    m = _model(memory_only_read=True, bounded_residual=True).eval()
    _calibrate_bank(m)
    m.lm_head.mo_scale.data.fill_(0.5)
    x = _tokens()
    with torch.no_grad():
        _fwd(m, x)
        snap = m.snapshot_runtime_buffers()
        y1 = _fwd(m, x)
        m.restore_runtime_buffers(snap)
        y2 = _fwd(m, x)
        # include_caches=False (best.pt-контракт) не должен ломаться
        snap2 = m.snapshot_runtime_buffers(include_caches=False)
        m.restore_runtime_buffers(snap2)
        y3 = _fwd(m, x)
    assert torch.equal(y1, y2), 'snapshot/restore changed the forward'
    assert torch.isfinite(y3).all()

    src = _model(memory_only_read=True).eval()
    m2 = _model(memory_only_read=True).eval()
    m2.load_state_dict(src.state_dict(), strict=True)
    assert torch.equal(m2.lm_head.mo_proj, src.lm_head.mo_proj)
    assert torch.equal(m2.lm_head.mo_scale, src.lm_head.mo_scale)
    src.memory_bank.reset()
    m2.memory_bank.reset()
    with torch.no_grad():
        assert torch.equal(_fwd(src, x), _fwd(m2, x)), 'resume roundtrip diverged'


def test_on_gc_true_train_step_finite_and_grads():
    """Блок 2 x блок 3: recompute-путь не ломает memory-only read."""
    m = _model(memory_only_read=True, bounded_residual=True,
               gradient_checkpointing=True)
    _calibrate_bank(m)
    m.lm_head.mo_scale.data.fill_(0.5)
    x = _tokens()
    out = _fwd(m, x)
    assert torch.isfinite(out).all()
    logits = m.lm_head(out)
    logits.pow(2).mean().backward()
    for name, p in (('mo_proj', m.lm_head.mo_proj),
                    ('mo_scale', m.lm_head.mo_scale)):
        g = p.grad
        assert g is not None and torch.isfinite(g).all()
        assert float(g.norm()) > 0.0, f'{name}: dead grad under gc=True'


def test_requires_memory_bank_and_supported_head():
    import pytest
    with pytest.raises(ValueError):
        EVAStack(_cfg(memory_only_read=True, memory_bank=False))
