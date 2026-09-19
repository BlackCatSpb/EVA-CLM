# -*- coding: utf-8 -*-
"""T9.6 лок: safety-термы (стена головы) не гейтятся CE-градиентом.

Находка: при насыщении |u|>17 CE-градиент ≈1e-13, и батчер множит aux на
clamp(‖gce‖/‖b‖,max=1) + sign-маску ⇒ head_wall глушился ровно при насыщении
(замер: wall=250, u_max рос 197→592). Safety получает свой градиент со
масштабом 1.0 без маски/бонда; профиль relu(|u|−u0)² самоограничен.
Покрыты все три пути батчера: cheap, align (пустой aux), align (с живым aux).

Run: python -m pytest tests/test_t9_safety_wall.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack
from core.training_control import LossBalancer


def _model(u0=0.0):
    torch.manual_seed(0)
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=True, vsa_decay_floor_k=2.0,
                    gradient_checkpointing=False,
                    head_u_wall=1e-3, head_u_wall_u0=u0)
    return cfg, EVAStack(cfg).train()


def _wall_and_param(m):
    """Живая стена (u0=0 ⇒ активна) и её градиент по readout."""
    torch.manual_seed(7)
    x = torch.randint(3, m.cfg.vocab, (1, 16)); x[:, ::11] = 2
    h = m.embed(x)
    out, st, gs, _ = m(h, None, step=1, tokens=x)
    m.observe_output(m.lm_head(out))
    ce, aux = m.compute_losses(out, x, h_emb=h)
    p = m.lm_head.readout
    wall = aux['head_wall']
    g_wall = torch.autograd.grad(wall, [p], retain_graph=True, allow_unused=True)[0]
    return ce, aux, p, wall, g_wall


def test_safety_aux_declared():
    assert 'head_wall' in LossBalancer.SAFETY_AUX
    assert 'head_wall' not in LossBalancer.BYPASS_AUX


def test_align_empty_path():
    """align + только стена: p.grad == ∂wall/∂p (CE≈0, aux-пусто после pop)."""
    cfg, m = _model()
    bal = LossBalancer(align=True, eval_interval=440, align_every=1)
    ce, aux, p, wall, g_wall = _wall_and_param(m)
    assert float(g_wall.norm()) > 0
    _ce0 = (p * 0.0).sum()
    bal.backward(_ce0, {'head_wall': wall}, [p], phase_model=m, step=1)
    assert p.grad is not None
    assert torch.allclose(p.grad, g_wall, atol=1e-6), \
        f'align-empty: стена не дошла ({(p.grad - g_wall).abs().max()})'


def test_align_normal_path():
    """align + живой второй aux: стена добавляется без маски/бонда."""
    cfg, m = _model()
    bal = LossBalancer(align=True, eval_interval=440, align_every=1)
    ce, aux, p, wall, g_wall = _wall_and_param(m)
    _ce0 = (p * 0.0).sum()
    _dummy = (p * 2.0).sum()          # живой aux (второй тензор ⇒ обычный путь)
    bal.backward(_ce0, {'head_wall': wall, 'dummy': _dummy}, [p],
                 phase_model=m, step=1)
    assert p.grad is not None
    # CE=0 ⇒ sign-маска обнуляет dummy; safety добавляет ровно ∂wall/∂p
    assert torch.allclose(p.grad, g_wall, atol=1e-6), \
        f'align-normal: стена искажена ({(p.grad - g_wall).abs().max()})'


def test_cheap_path():
    """cheap (align=False): стена добавляется поверх s-масштабированного aux."""
    cfg, m = _model()
    bal = LossBalancer(align=False, eval_interval=440)
    ce, aux, p, wall, g_wall = _wall_and_param(m)
    _ce0 = (p * 0.0).sum()
    _dummy = (p * 2.0).sum()
    bal.backward(_ce0, {'head_wall': wall, 'dummy': _dummy}, [p],
                 phase_model=m, step=1)
    assert p.grad is not None
    expect = g_wall + 2.0            # dummy (s=1 в legacy align=False) + стена
    assert torch.allclose(p.grad, expect, atol=1e-6), \
        f'cheap: {p.grad.norm()} vs {expect.norm()}'


def test_caller_dict_not_mutated():
    """R2-лок: backward не съедает head_wall из словаря вызывающего (лог)."""
    cfg, m = _model()
    bal = LossBalancer(align=True, eval_interval=440, align_every=1)
    ce, aux, p, wall, g_wall = _wall_and_param(m)
    _ce0 = (p * 0.0).sum()
    d = {'head_wall': wall, 'dummy': (p * 2.0).sum()}
    bal.backward(_ce0, d, [p], phase_model=m, step=1)
    assert 'head_wall' in d, 'aux_dict вызывающего мутирован (wall потерян в логе)'


def test_safety_flag_off_restores_gating():
    """A/B-ручка: safety_aux=() ⇒ стена снова гейтится (нулевой вклад при CE=0)."""
    cfg, m = _model()
    bal = LossBalancer(align=True, eval_interval=440, align_every=1, safety_aux=())
    ce, aux, p, wall, g_wall = _wall_and_param(m)
    _ce0 = (p * 0.0).sum()
    bal.backward(_ce0, {'head_wall': wall}, [p], phase_model=m, step=1)
    assert p.grad is None or float(p.grad.norm()) == 0.0, \
        'при safety_aux=() стена не должна проходить'


def test_safety_zero_below_threshold():
    cfg, m = _model(u0=1e6)          # порог недостижим
    torch.manual_seed(7)
    x = torch.randint(3, cfg.vocab, (1, 16))
    h = m.embed(x)
    out, *_ = m(h, None, step=1, tokens=x)
    m.observe_output(m.lm_head(out))
    ce, aux = m.compute_losses(out, x, h_emb=h)
    assert 'head_wall' not in aux, 'стена активна ниже порога'


def test_gradalign_target_untouched_in_cheap():
    """Safety-backward не перезаписывает gradalign-цель (хук морозится)."""
    cfg, m = _model()
    bal = LossBalancer(align=False, eval_interval=440)
    ce, aux, p, wall, g_wall = _wall_and_param(m)
    blk = m.layers[0]
    sentinel = torch.full((1,), 123.0)
    blk._gradalign_tgt = sentinel
    _ce0 = (p * 0.0).sum()
    bal.backward(_ce0, {'head_wall': wall, 'dummy': (p * 2.0).sum()}, [p],
                 phase_model=m, step=1)
    assert blk._gradalign_tgt is sentinel, \
        'safety-backward перезаписал gradalign-цель'
    assert getattr(blk, '_ga_record', True) is True, '_ga_record не восстановлен'
