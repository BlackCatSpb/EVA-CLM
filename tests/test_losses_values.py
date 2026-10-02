"""losses.compute_losses: ЗНАЧЕНИЯ, а не только формы (M65-opt).

Модуль не имел прямых тестов (аудит: ноль импортов в tests/) — проверялся
лишь косвенно через стек. Замки: ce_raw = ручная маскированная CE того же
пути головы; aux-значения конечны и в смысловых диапазонах; CE падает на
градиентных шагах (реальный обучающий сигнал, а не только формы).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


def _mini():
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16,
                    code_sparsity=4, vocab=256, save_dir='.',
                    logit_cache_enabled=False, gradient_checkpointing=False)
    cfg.intent_bridge = False          # точное сравнение без bus_bias
    torch.manual_seed(0)
    m = EVAStack(cfg)
    m.eval()                           # без train-обновлений телеметрии
    return m


def test_ce_raw_matches_manual_masked_mean():
    m = _mini()
    x = torch.randint(1, 256, (1, 16))
    with torch.no_grad():
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=5, tokens=x)
        _ce, aux = m.compute_losses(out, x, h_emb=h)
        lp = m.lm_head.log_probs_for_target(out.reshape(-1, out.shape[-1]),
                                            x.reshape(-1), bus_bias=None)
    mask = (x.reshape(-1) != 0).float()
    ref = (-lp * mask).sum() / mask.sum().clamp(min=1)
    got = m._cached_losses['ce_raw']
    assert abs(got - float(ref)) < 1e-4, f'ce_raw {got} != ручная {float(ref)}'


def test_aux_terms_finite_and_in_semantic_ranges():
    m = _mini()
    x = torch.randint(1, 256, (1, 16))
    with torch.no_grad():
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=5, tokens=x)
        m.compute_losses(out, x, h_emb=h)
    cl = m._cached_losses
    for k in ('ce', 'ce_raw', 'pred', 'gate_l1', 'reinforce', 'balance',
              'div', 'alpha_novelty', 'signal_ent', 'ls_reg', 'decorr'):
        v = cl[k]
        assert v == v and abs(v) < 1e4, f'{k}={v}'
    assert cl['signal_ent'] <= 0.0, 'энтропия сигналов обязана быть <= 0'
    # div — ОТРИЦАНИЕ дисперсии (min loss = max diversity, см. losses.py:345)
    assert cl['div'] <= 0.0, 'div — знаковая конвенция -Var, обязана быть <= 0'
    assert cl['balance'] >= 0.0, 'balance — MSE, обязана быть >= 0'


def test_ce_decreases_on_gradient_steps():
    m = _mini()
    m.train()
    # lr из рабочего режима (0.05 на этом стеке расходится — проверено)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    x = torch.randint(1, 256, (1, 16))
    ce0 = None
    ces = []
    for i in range(20):
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=5 + i, tokens=x)
        ce, _aux = m.compute_losses(out, x, h_emb=h)
        ces.append(float(ce.detach()))
        if ce0 is None:
            ce0 = ces[0]
        opt.zero_grad()
        ce.backward()
        opt.step()
    assert min(ces) < ce0, f'CE не улучшилась на градиентных шагах: {ce0:.4f} -> {min(ces):.4f}'
