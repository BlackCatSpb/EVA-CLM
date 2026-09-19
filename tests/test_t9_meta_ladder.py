# -*- coding: utf-8 -*-
"""T9.7 лок: VSA-лестница противоречий головы + адаптивный порог фантома.

Находка: ph_obs=0 во всех прогонах — константа head_phantom_thr=1.1 выше
диапазона lacuna_rel (~1.00-1.02), фантом никогда не наблюдал, лестница
«фантом↔UCL» мертва с рождения. Фикс: 4-шкальная VSA-лестница (tau_api.
VSA_LADDER) + порог наблюдения = p95 салиентности в скользящем окне (не
может «застрять» выше сигнала по построению).

Run: python -m pytest tests/test_t9_meta_ladder.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack
from core import tau_api


def _model():
    torch.manual_seed(0)
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=True, vsa_decay_floor_k=2.0,
                    gradient_checkpointing=False)
    return cfg, EVAStack(cfg).train()


def _head(m):
    return m.lm_head


def test_ladder_scales_from_tau_api():
    cfg, m = _model()
    hd = _head(m)
    tau = hd._ladder_tau
    assert tau.numel() == len(tau_api.VSA_LADDER), 'лестница не 4-шкальная'
    for i, t in enumerate(tau_api.VSA_LADDER):
        assert abs(float(tau[i]) - t) < 1e-6, f'шкала {i} ≠ {t}'
    d = hd._ladder_decay
    assert abs(float(d[0]) - torch.exp(torch.tensor(-1.0 / tau_api.VSA_LADDER[0]))) < 1e-6
    assert float(d[0]) < float(d[-1]), 'быстрая шкала должна затухать быстрее'


def test_ladder_updates_in_training_only():
    cfg, m = _model()
    hd = _head(m)
    x = torch.randint(3, cfg.vocab, (1, 16))
    m.eval()
    with torch.no_grad():
        h = m.embed(x)
        m(h, None, step=1, tokens=x)
    lvl_eval = hd.ell_ladder.clone()
    m.train()
    torch.manual_seed(7)
    x2 = torch.randint(3, cfg.vocab, (1, 16))
    h2 = m.embed(x2)
    out, *_ = m(h2, None, step=1, tokens=x2)
    assert not torch.equal(lvl_eval, hd.ell_ladder), 'лестница не обновляется в train'


def test_fast_scale_reacts_faster():
    """Спайк: быстрая шкала реагирует сильнее (медленная почти не движется)."""
    cfg, m = _model()
    hd = _head(m)
    x = torch.randint(3, cfg.vocab, (1, 16))
    h = m.embed(x)
    m(h, None, step=1, tokens=x)
    base = hd.ell_ladder.clone()
    with torch.no_grad():
        hd.ell_ladder.mul_(hd._ladder_decay).add_(
            (1.0 - hd._ladder_decay) * (float(base.mean()) * 10.0))
    after = hd.ell_ladder.clone()
    dev_fast = float(after[0]) - float(base[0])
    dev_slow = float(after[-1]) - float(base[-1])
    assert dev_fast > 0 and dev_slow > 0, 'шкалы не отреагировали'
    assert dev_fast > 10.0 * dev_slow, \
        f'быстрая шкала должна реагировать много сильнее: {dev_fast:.4g} vs {dev_slow:.4g}'


def test_stationary_input_observes_rarely():
    """R3-лок селективности: наблюдение по СТАРОЙ шкале (база×1.1) — на
    стационарном входе почти нет наблюдений (quantile-порог давал 16/24)."""
    cfg, m = _model()
    hd = _head(m)
    _pb = hd.phantom_bank
    step0 = int(hd.phantom_after) + 10
    torch.manual_seed(1)
    x = torch.randint(3, cfg.vocab, (1, 16))
    h0 = m.embed(x)
    for i in range(24):
        m(h0, None, step=step0 + i, tokens=x)
        hd._pb_step.fill_(0)
    assert int(_pb._obs.item()) <= 3, \
        f'стационарный вход дал {int(_pb._obs.item())} наблюдений (ожидаемо ≤3)'


def test_spike_above_slow_base_observes():
    """Спайк относительно базы: занижаем СТАРУЮ шкалу ⇒ салиентность >1.1 ⇒
    наблюдение срабатывает (и порог — константа phantom_thr, не quantile)."""
    cfg, m = _model()
    hd = _head(m)
    _pb = hd.phantom_bank
    step0 = int(hd.phantom_after) + 10
    torch.manual_seed(2)
    x = torch.randint(3, cfg.vocab, (1, 16))
    h = m.embed(x)
    for i in range(4):
        m(h, None, step=step0 + i, tokens=x)
        hd._pb_step.fill_(0)
    with torch.no_grad():
        hd.ell_ladder[-1].mul_(0.2)      # база занижена ⇒ спайк
    m(h, None, step=step0 + 5, tokens=x)
    hd._pb_step.fill_(0)
    assert int(_pb._obs.item()) > 0, 'наблюдение не сработало на спайке'
    assert hd._meta_thr is not None and abs(hd._meta_thr - hd.phantom_thr) < 1e-9


def test_telemetry_keys():
    from core.training_control import training_telemetry
    cfg, m = _model()
    hd = _head(m)
    hd._pb_active = True
    x = torch.randint(3, cfg.vocab, (1, 16))
    h = m.embed(x)
    m(h, None, step=1, tokens=x)
    tt = training_telemetry(m)
    assert 'meta_lad' in tt, f'нет meta_lad в tele: {sorted(tt)}'
    assert tt['meta_lad'].count('|') == 3, f'meta_lad не 4-шкальный: {tt["meta_lad"]}'
