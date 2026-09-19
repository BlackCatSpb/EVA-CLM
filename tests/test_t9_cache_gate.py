# -*- coding: utf-8 -*-
"""T9.5 лок: логит-кэш (двусторонний KV-аналог) — гейт, телеметрия, ценз.

ЗАМЕР (реальный ckpt): σ(bias)=4.5e-5 — ЛОЖНЫЙ индикатор; weight-терм даёт
фактический mean-гейт 0.043 уже на 250 шагах (|g| веса гейта≈7) — кэш
приоткрыт и учится сам. Рампа (A/B-рука, по умолчанию 0) — опциональное
ускорение открытия; инвариант training-only (generate передаёт step в eval).

Run: python -m pytest tests/test_t9_cache_gate.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack


def _model(ramp=0, final=-2.0):
    torch.manual_seed(0)
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=True,
                    memory_bank=True, intent_bridge=True, vsa_decay_floor_k=2.0,
                    gradient_checkpointing=False,
                    logit_cache_gate_ramp=ramp,
                    logit_cache_gate_bias_final=final)
    return cfg, EVAStack(cfg)


def _bias(m):
    return float(m.logit_cache.attention.cache_gate[-2].bias.detach())


def _tokens(cfg, L=16):
    torch.manual_seed(7)
    x = torch.randint(3, cfg.vocab, (1, L)); x[:, ::11] = 2
    return x


def test_ramp_monotone_and_final():
    cfg, m = _model(ramp=4000, final=-2.0)
    m.train()
    sched = m.logit_cache.set_gate_schedule
    sched(0, ramp=4000, bias_final=-2.0)
    assert abs(_bias(m) - (-10.0)) < 1e-9, 'шаг 0: гейт не на identity'
    sched(2000, ramp=4000, bias_final=-2.0)
    b_mid = _bias(m)
    assert -10.0 < b_mid < -2.0, f'середина рампы вне (−10,−2): {b_mid}'
    sched(4000, ramp=4000, bias_final=-2.0)
    assert abs(_bias(m) - (-2.0)) < 1e-9, 'конец рампы не на bias_final'
    # после рампы расписание НЕ трогает параметр (учится сам)
    with torch.no_grad():
        m.logit_cache.attention.cache_gate[-2].bias.fill_(-1.234)
    sched(9000, ramp=4000, bias_final=-2.0)
    assert abs(_bias(m) - (-1.234)) < 1e-6, 'после рампы параметр затёрт'


def test_ramp_off_is_closed_arm():
    cfg, m = _model(ramp=0)
    m.train()
    for s in (0, 2000, 50000):
        m.logit_cache.set_gate_schedule(s, ramp=0)
    assert abs(_bias(m) - (-10.0)) < 1e-9, 'ramp=0 не сохраняет closed-arm'


def test_stack_applies_ramp_in_train():
    """R1/R3-лок: вызов из стека (удаление вызова ломает этот тест)."""
    cfg, m = _model(ramp=4000, final=-2.0)
    m.train()
    x = _tokens(cfg)
    h = m.embed(x)
    with torch.no_grad():
        m(h, None, step=2000, tokens=x)
    assert abs(_bias(m) - (-6.0)) < 1e-6, \
        f'стек не применил расписание: bias={_bias(m)} (ожидалось −6)'


def test_eval_does_not_touch_gate():
    """R1/R2/R3-лок: generate передаёт step в eval — bias обязан выжить."""
    cfg, m = _model(ramp=4000, final=-2.0)
    m.eval()
    with torch.no_grad():
        m.logit_cache.attention.cache_gate[-2].bias.fill_(-2.0)
    x = _tokens(cfg)
    h = m.embed(x)
    with torch.no_grad():
        m(h, None, step=1, tokens=x)
    assert abs(_bias(m) - (-2.0)) < 1e-6, \
        f'eval затёр обученный bias гейта: {_bias(m)}'


def test_sigma_grows_monotonically():
    cfg, m = _model(ramp=4000, final=-2.0)
    m.train()
    sig = []
    for s in range(0, 4001, 500):
        m.logit_cache.set_gate_schedule(s, ramp=4000, bias_final=-2.0)
        sig.append(float(torch.sigmoid(m.logit_cache.attention.cache_gate[-2].bias)))
    assert all(b > a for a, b in zip(sig, sig[1:])), f'σ не растёт: {sig}'


def test_identity_at_step0_forward():
    cfg, m = _model()
    m.train()
    x = _tokens(cfg)
    h = m.embed(x)
    with torch.no_grad():
        m(h, None, step=0, tokens=x)
    lc = m.logit_cache
    assert lc.attention._last_gate_mean is not None
    assert float(lc.attention._last_gate_mean) < 1e-3, \
        f'гейт на шаге 0 не identity: {lc.attention._last_gate_mean}'


def test_census_and_telemetry_see_cache():
    from core.training_control import grad_census, training_telemetry
    cfg, m = _model()
    m.train()
    x = _tokens(cfg)
    h = m.embed(x)
    out, *_ = m(h, None, step=500, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    ce.backward()
    gc = grad_census(m)
    for k in ('g_cache_gate', 'g_cache_gate_w', 'g_cache_attn', 'g_logit_to_hidden'):
        assert k in gc, f'{k} отсутствует в цензе'
        assert gc[k] is None or gc[k] >= 0.0
    tt = training_telemetry(m)
    for k in ('cache_gate', 'cache_gate_bias', 'cache_read_ratio'):
        assert k in tt, f'{k} отсутствует в телеметрии: {sorted(tt)}'
    assert 0.0 < tt['cache_gate_bias'] < 1.0
    assert tt['cache_gate'] >= 0.0 and tt['cache_read_ratio'] >= 0.0
