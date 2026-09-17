"""M64.7 lock: the readout LR override (the unfreeze A/B arm).

The M63-A audit measured the head's readout (= embed.basis, tied) moving only
-1.9% in 7315 steps under the λ⁻² damp (0.296 at λ_d=3) — while the head is
the LM bottleneck (CE code-only 12.84 > bias-only 8.65). `readout_lr_mult > 0`
overrides the damp for the A/B; the default 0 keeps the historical behaviour.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adaptation import build_optimizer, _role_lr_mult  # noqa: E402
from core.config import EVAConfig  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def test_role_mult_default_keeps_the_lambda_damp():
    lam = 1.8393  # lambda_d(3)
    got = _role_lr_mult('lm_head.readout', lam)
    assert abs(got - lam ** -2) < 1e-9
    assert got < 0.31, 'the historical damp must stay in place by default'


def test_role_mult_override_reaches_the_readout():
    lam = 1.8393
    for name in ('lm_head.readout', 'lm_head.proj', 'embed.basis', 'embed.weight'):
        assert abs(_role_lr_mult(name, lam, 1.0) - 1.0) < 1e-9, name
    # a non-readout name is untouched by the override (its own role applies)
    for name in ('layers.0.mlp.W_gate', 'layers.0.mirror.W_proj'):
        a = _role_lr_mult(name, lam, 1.0)
        b = _role_lr_mult(name, lam, 0.0)
        assert abs(a - b) < 1e-12, f'{name}: the override leaked ({a} vs {b})'


def test_build_optimizer_applies_the_override():
    torch.manual_seed(0)
    m = EVAStack(EVAConfig(**SMALL))
    lam = 1.8393
    o_default = build_optimizer(m, 1e-3, llrd_decay=1.0, lam=lam)
    o_unfreeze = build_optimizer(m, 1e-3, llrd_decay=1.0, lam=lam, readout_lr_mult=1.0)
    # find the group holding the readout — in the sigmoid_coded head the readout
    # IS embed.basis (the tied parameter, M63-A)
    def _readout_lr(opt):
        for g in opt.param_groups:
            names = [n for n, p in m.named_parameters()
                     if any(p is q for q in g['params'])]
            if any(n == 'embed.basis' for n in names):
                return g['lr']
        raise AssertionError('no readout group')
    lr_def = _readout_lr(o_default)
    lr_unf = _readout_lr(o_unfreeze)
    assert abs(lr_def - 1e-3 * lam ** -2) < 1e-12, f'default: {lr_def}'
    assert abs(lr_unf - 1e-3) < 1e-12, f'unfreeze: {lr_unf}'


def test_config_knob_defaults_off():
    cfg = EVAConfig(**SMALL)
    assert cfg.readout_lr_mult == 0.0
