# -*- coding: utf-8 -*-
"""T9 root-fix лок: общий τ-модуль НЕ регистрируется подмодулем потребителей.

Корневая причина мёртвой τ-лестницы (найдена 2026-09-19): EVABlock присваивал
общий tau_config как атрибут nn.Module ⇒ регистрация подмодулем ⇒ параметры τ
попадали в layer.parameters() и в state_dict (layers.N.tau_config.*), а
set_active_depth(k) (DepthController, итерирует ТОЛЬКО model.layers) вызывал
requires_grad_(False) на _tau_dev при заморозке слоёв k..n−1 ⇒ g_tau_dev=0,
_tau_dev=0.0000 в чекпойнтах за 250+ шагов.

MemoryBank/Maturation исправлены тем же приёмом, но это гигиена state_dict
(их параметр-обходы не морозятся depth-контроллером) — причиной были блоки.

Run: python -m pytest tests/test_t9_tau_registration.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack
from core.adaptation import set_active_depth, build_optimizer


def _model(n_layers=3):
    torch.manual_seed(0)
    cfg = EVAConfig(n_layers=n_layers, D=128, mlp_groups=4, code_dim=16,
                    code_sparsity=4, vocab=400, save_dir='.',
                    logit_cache_enabled=False, memory_bank=True,
                    intent_bridge=True, vsa_decay_floor_k=2.0,
                    gradient_checkpointing=False)
    return cfg, EVAStack(cfg).train()


def test_tau_not_in_layer_parameters():
    cfg, m = _model()
    td = m.tau_config._tau_dev
    for i, layer in enumerate(m.layers):
        assert all(p is not td for p in layer.parameters()), \
            f'L{i}: _tau_dev попал в layer.parameters() (регистрация подмодуля)'
    assert all(p is not td for p in m.memory_bank.parameters()), \
        'memory_bank: _tau_dev попал в parameters()'
    assert all(p is not td for p in m.maturation.parameters()), \
        'maturation: _tau_dev попал в parameters()'


def test_state_dict_tau_keys_canonical():
    """Ожидаем ровно два имени ОДНОГО тензора: канонический
    `tau_config._tau_dev` и алиас стека `_tau_l_dev` (stack.py:136, та же
    память). Вложенных `layers.N.tau_config.*` быть не должно."""
    cfg, m = _model()
    sd = m.state_dict()
    # '_tau_l_dev' не оканчивается на '_tau_dev' — ловим алиас явно
    dev_keys = sorted(k for k in sd if k == '_tau_l_dev' or k.endswith('_tau_dev'))
    assert dev_keys == ['_tau_l_dev', 'tau_config._tau_dev'], \
        f'набор τ-ключей изменился: {dev_keys}'
    assert sd['_tau_l_dev'].data_ptr() == sd['tau_config._tau_dev'].data_ptr(), \
        'алиас и канон указывают на разные тензоры'
    dup = [k for k in sd if '.tau_config.' in k and not k.startswith('tau_config.')]
    assert not dup, f'вложенные регистрации tau_config: {dup[:5]}'


def test_set_active_depth_keeps_tau_trainable():
    cfg, m = _model(n_layers=3)
    td = m.tau_config._tau_dev
    assert td.requires_grad
    set_active_depth(m, 1)          # морозим слои 1..n−1
    assert td.requires_grad, 'set_active_depth заморозил _tau_dev (root-баг)'
    for p in m.layers[1].parameters():
        assert not p.requires_grad, 'слой 1 не заморожен (контроль)'


def test_tau_grad_lives_after_depth_freeze():
    cfg, m = _model(n_layers=3)
    set_active_depth(m, 1)
    torch.manual_seed(7)
    x = torch.randint(3, cfg.vocab, (1, 32)); x[:, ::11] = 2
    h = m.embed(x)
    out, st, gs, r = m(h, None, step=1, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    ce.backward()
    td = m.tau_config._tau_dev
    assert td.grad is not None and float(td.grad.norm()) > 0, \
        'g_tau_dev=0 после заморозки глубины — путь τ мёртв'


def test_optimizer_has_tau_once():
    cfg, m = _model(n_layers=2)
    opt = build_optimizer(m, base_lr=6e-4)
    td = m.tau_config._tau_dev
    n = sum(1 for g in opt.param_groups for p in g['params'] if p is td)
    assert n == 1, f'_tau_dev в оптимизаторе {n} раз (ожидалось 1)'


def test_census_distinguishes_none_from_zero():
    """R3-лок: grad_census обязан различать «вне графа» (None) и «ноль» (0.0)."""
    from core.training_control import grad_census
    cfg, m = _model(n_layers=2)
    set_active_depth(m, 1)
    torch.manual_seed(7)
    x = torch.randint(3, cfg.vocab, (1, 16)); x[:, ::11] = 2
    h = m.embed(x)
    out, st, gs, r = m(h, None, step=1, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    ce.backward()
    gc = grad_census(m)
    assert gc.get('g_tau_dev') is not None and gc['g_tau_dev'] > 0, \
        'живой τ-путь дал None/0 в цензе'
    # замороженный параметр слоя (вне графа) должен быть None, не 0.0
    m.tau_config._tau_dev.requires_grad_(False)
    m.zero_grad(set_to_none=True)
    h2 = m.embed(x)
    out2, _, _, _ = m(h2, None, step=1, tokens=x)
    ce2, _ = m.compute_losses(out2, x, h_emb=h2)
    ce2.backward()
    gc2 = grad_census(m)
    assert gc2.get('g_tau_dev') is None, \
        f'замороженный τ: ценз вернул {gc2.get("g_tau_dev")!r} вместо None'
