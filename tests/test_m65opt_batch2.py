"""M65-opt батч 2: поведенческие замки hot-path оптимизаций.

Каждая оптимизация без замка — регрессия в будущем:
  1. τ-снимок TauConfig.update() обновляется на каждом forward и совпадает
     с тензорами (и блок реально следует за полем, а не застывает);
  2. lazy-init лакуны: ненулевой буфер (резюм) НЕ переинициализируется;
  3. батчевая телеметрия losses даёт те же значения, что прямые тензоры.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


def _model():
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16,
                    code_sparsity=4, vocab=256, logit_cache_enabled=False,
                    gradient_checkpointing=False, save_dir='.')
    torch.manual_seed(0)
    return EVAStack(cfg).train()


def test_tau_snapshot_matches_tensors_and_follows_the_field():
    m = _model()
    x = torch.randint(1, 256, (1, 16))
    m(m.embed_tokens(x), None, step=5, tokens=x)
    tc = m.tau_config
    ref = tc.tau_norm.detach().tolist()
    assert len(tc.tau_norm_py) == len(ref)
    assert all(abs(a - b) < 1e-6 for a, b in zip(tc.tau_norm_py, ref)), \
        'снимок tau_norm рассинхронизирован с тензором'
    blk = m.layers[0]
    assert abs(blk._tau_norm - tc.tau_norm_py[0]) < 1e-9

    # сдвигаем τ-поле и проверяем, что снимок и блок остаются согласованными
    # (замок против протухшего снимка; величина сдвига не важна)
    with torch.no_grad():
        tc._tau_dev.fill_(5.0)
    m(m.embed_tokens(x), None, step=6, tokens=x)
    ref2 = tc.tau_norm.detach().tolist()
    assert all(abs(a - b) < 1e-6 for a, b in zip(tc.tau_norm_py, ref2)), \
        'снимок tau_norm протух после сдвига поля'
    assert abs(blk._tau_norm - tc.tau_norm_py[0]) < 1e-9
    with torch.no_grad():
        tc._tau_dev.fill_(0.0)
    m(m.embed_tokens(x), None, step=7, tokens=x)
    assert all(abs(a - b) < 1e-6
               for a, b in zip(tc.tau_norm_py, tc.tau_norm.detach().tolist()))


def test_lacuna_ema_resume_is_not_reinitialized():
    m = _model()
    head = m.lm_head
    with torch.no_grad():
        head.ell_ema.fill_(0.42)
        head.ell_ladder.fill_(0.42)
    head._ell_ema_ready = False            # как после резюма ядра
    x = torch.randint(1, 256, (1, 16))
    m(m.embed_tokens(x), None, step=5, tokens=x)
    assert head._ell_ema_ready
    assert abs(float(head.ell_ema) - 0.42) < 0.05, \
        'восстановленная EMA лакуны была переинициализирована'


def test_cached_losses_batch_matches_direct_tensors():
    m = _model()
    x = torch.randint(1, 256, (1, 16))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=5, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    cl = m._cached_losses
    keys = ('ce', 'ce_raw', 'pred', 'gate_l1', 'reinforce', 'balance', 'div',
            'alpha_novelty', 'signal_ent', 'ls_reg', 'decorr')
    assert set(keys) <= set(cl), f'потеряны ключи: {set(keys) - set(cl)}'
    for k in keys:
        v = cl[k]
        assert isinstance(v, float) and v == v, f'{k}: не число ({v!r})'
    assert abs(cl['ce'] - float(ce.detach())) < 1e-5
    # branch-RMS телеметрия (батч 3×N -> один tolist)
    for k in ('branch_r_conv', 'branch_r_bind', 'branch_r_mirror'):
        if k in cl:
            assert isinstance(cl[k], float) and cl[k] == cl[k]
