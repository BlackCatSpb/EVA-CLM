"""M55b locks: the RELATIVE (self-calibrating) lacuna novelty, the selective
gate/bank, and the RNG hygiene (the exploration noise must not shift the
training's stochastic ops — the head runs inside the stack's forward)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**{**SMALL, **kw}))


def _hq(m):
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    return x, m.embed_tokens(x)


def test_absolute_lacuna_is_nearly_constant_but_relative_is_not():
    m = _model(head_phantom_every=1).train()
    x, h = _hq(m)
    m(h, None, step=1, tokens=x)
    ell_abs = float(m.lm_head._last_lacuna)
    rel = float(m.lm_head._last_lacuna_rel)
    # the absolute ell sits near the sqrt(1 - K/D) floor for ANY state...
    assert ell_abs > 0.5, f'absolute ell unexpectedly small: {ell_abs}'
    # ...but the self-calibrated ratio is ~1 on the running level
    assert 0.9 < rel < 1.1, f'relative novelty not calibrated: {rel}'
    assert float(m.lm_head._last_lacuna_gate) < 0.5, 'the gate is open on the running level'


def test_gate_opens_on_a_relative_spike():
    m = _model(head_phantom_every=1).train()
    x, h = _hq(m)
    m(h, None, step=1, tokens=x)
    g_closed = float(m.lm_head._last_lacuna_gate)
    # T9.7: гейт читает СЕРЕДИНУ VSA-лестницы (дроп-ин к прежней ell_ema ~100);
    # спайк симулируем тем же приёмом — вдвое занижаем рабочий уровень.
    _mid = m.lm_head.ell_ladder.numel() // 2
    with torch.no_grad():
        m.lm_head.ell_ladder[_mid].mul_(0.5)
    m(h, None, step=2, tokens=x)
    assert float(m.lm_head._last_lacuna_rel) > 1.5
    assert float(m.lm_head._last_lacuna_gate) > 0.9, 'the gate did not open on a spike'
    assert float(m.lm_head._last_lacuna_gate) > g_closed


def test_bank_is_selective_on_the_relative_novelty():
    m = _model(head_phantom_every=1).train()
    m.lm_head._pb_active = True
    x, h = _hq(m)
    for s in range(5):
        m(h, None, step=s, tokens=x)
    # ell_rel ~ 1 < thr 1.1 -> the bank must not observe
    assert int(m.lm_head.phantom_bank._obs) == 0, 'the bank observed the running level'


def test_noise_does_not_advance_the_global_rng():
    m = _model().train()
    x, h = _hq(m)
    h = m.embed_tokens(x)
    zt, zd, e_l = m.lm_head._gates(h, return_data=True)
    u, _ = m.lm_head._su(zt, zd)
    torch.manual_seed(0)
    s1 = torch.get_rng_state()
    _ = m.lm_head._phantom_mix(u, e_l, h)
    s2 = torch.get_rng_state()
    assert torch.equal(s1, s2), 'the exploration noise shifted the global RNG'


def test_noise_generator_is_device_local():
    m = _model().train()
    x, h = _hq(m)
    _ = m.lm_head._phantom_mix(torch.zeros(1, 8, m.lm_head.K), h, h)
    assert m.lm_head._noise_gen is not None
    assert str(m.lm_head._noise_gen.device) == str(h.device)


def test_ell_ema_does_not_drift_at_eval():
    m = _model(head_phantom_every=1).eval()
    x, h = _hq(m)
    m(h, None, step=1, tokens=x)          # lazy init happens in eval too
    e0 = float(m.lm_head.ell_ema)
    assert e0 > 0.0, 'the eval did not even initialize the EMA'
    for _ in range(5):
        m(h, None, step=2, tokens=x)
    assert float(m.lm_head.ell_ema) == e0, 'the eval drifted the training statistic'
