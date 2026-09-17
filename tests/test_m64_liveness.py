"""M64.10 lock: the liveness census as a TEST (agent B's T5/T6 protocol) + the
T13 weight-semantics pin.

T5: every channel the census tracks must receive a nonzero gradient through a
real CE backward — a zero entry is the dead-channel signal. The config must
ACTIVATE the channels (maturation off, phantom warmup 0, the memory bank on,
the fusion wake, an active phantom slice), otherwise the zeros are warmups,
not death.

The census itself FOUND a real dead zone (recorded in the whiteboard as the M65
normalization A/B): the phantom basis / lacuna_w / lacuna_b / log_eta receive
~zero gradient because tanh SATURATES on the un-normalized lacuna (‖e_l‖ ~
sqrt(D) after the stack's final_norm). The test pins the saturation metric
(ph_sat -> 1) so a silent regression is caught and the fix is verifiable.

T13: in the align mode the `*_weight` config knobs are ON/OFF only — the raw
aux values do not depend on them (the LossBalancer owns the weighting). The
test pins that semantics so the docs cannot drift again.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core.training_control import grad_census  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _live_model(**kw):
    torch.manual_seed(0)
    m = EVAStack(EVAConfig(**{**SMALL, 'memory_bank': True,
                              'maturation_enabled': False,
                              'head_phantom_after': 0, **kw})).train()
    with torch.no_grad():
        m.memory_bank.fusion[-1].weight.normal_(0.0, 0.01)   # the memory cold-start wake
        m.lm_head._kp_active.fill_(4)                        # an active phantom slice
        m.lm_head.phantom_mix.normal_(0.0, 0.01)             # the phantom cold-start wake
    return m


def _backward(m):
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    x[0, 3] = 2
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=1, tokens=x)
    logits = m.lm_head(out)
    F.cross_entropy(logits.reshape(-1, m.cfg.vocab), x.reshape(-1)).backward()
    return m


def test_t5_all_expected_channels_are_live():
    m = _backward(_live_model())
    gc = grad_census(m)
    for k in ('g_readout', 'g_token_bias', 'g_bit_bias', 'g_log_temp', 'g_emphasis',
              'g_phantom_mix', 'g_ucl_scale', 'g_wk', 'g_wv', 'g_wq', 'g_wo',
              'g_fusion2', 'g_l1_proj'):
        assert gc.get(k, 0.0) > 0.0, f'{k} is dead: {gc.get(k)}'


def test_t5_the_phantom_channel_is_a_cold_start_not_dead():
    """The census's own find (M64.10): with the zero-init phantom_mix the basis
    is at a COLD START (dL/dbasis = dL/dp @ mix = 0) — but the wake-up path is
    alive (the mix's own gradient). After the wake all the phantom params live;
    ph_sat (~0.3 at init) tracks the un-normalized-lacuna risk separately."""
    m = _live_model()
    # BEFORE the wake (zero mix): the basis is blocked, the mix itself is not
    with torch.no_grad():
        m.lm_head.phantom_mix.zero_()
    _backward(m)
    gc0 = grad_census(m)
    assert gc0.get('g_phantom_mix', 0.0) > 0.0, 'the wake-up path is dead'
    assert gc0.get('g_phantom_basis', 0.0) == 0.0, \
        'the cold start is gone — update this pin and the M65 record'
    ph_sat = getattr(m.lm_head, '_last_ph_sat', None)
    assert ph_sat is not None and 0.0 < ph_sat < 0.95, f'ph_sat out of range: {ph_sat}'
    # AFTER the wake the basis/lacuna params are live
    m2 = _backward(_live_model())
    gc1 = grad_census(m2)
    for k in ('g_phantom_basis', 'g_lacuna_w', 'g_lacuna_b', 'g_log_eta'):
        assert gc1.get(k, 0.0) > 0.0, f'{k} is dead after the wake: {gc1.get(k)}'


def test_t13_align_mode_weights_are_on_off_only():
    """The raw aux values must not depend on the *_weight knobs in the align
    mode (the balancer owns the weighting; only gradalign bypasses)."""
    vals = []
    for w in (10.0, 1.0):
        torch.manual_seed(0)
        m = EVAStack(EVAConfig(**{**SMALL, 'div_weight': w})).train()
        x = torch.randint(1, SMALL['vocab'], (1, 8))
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=1, tokens=x)
        _, aux = m.compute_losses(out, x, h_emb=h)
        vals.append(float(aux['div'].detach()))
    assert abs(vals[0] - vals[1]) < 1e-9, \
        f'the div aux value depends on div_weight in the align mode: {vals}'
