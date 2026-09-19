# -*- coding: utf-8 -*-
"""T9 блок-локи: врезка CovarianceMemory в EVABlock.

- zero-risk: cov_memory=True при zero-init выхода даёт бит-в-бит тот же выход
  (ветвь создаётся последней ⇒ RNG-поток ствола не сдвинут — без транспланта);
- state-контракт: cov off ⇒ 5-кортеж (чекпойнты не меняются); cov on ⇒ 6-й
  элемент; cov on принимает старый 5-элементный state (холодный cov-старт);
- liveness: после снятия zero-init выход меняется, градиенты текут во все
  проекции ветви; cov-состояние едет в state[5] и меняется между шагами;
- порядок resume в scripts/train.py: восстановленный stream_state не затирается
  (R2: мёртвый warm-resume).

Run: python -m pytest tests/test_t9_cov_block.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack


def _cfg(cov=False):
    return EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                     vocab=400, save_dir='.', logit_cache_enabled=False,
                     memory_bank=True, intent_bridge=True, explicit_reasoning=False,
                     vsa_decay_floor_k=2.0, cov_memory=cov,
                     cov_memory_heads=4, cov_memory_head_dim=32,
                     cov_memory_chunk=16, cov_memory_rank=16)


def _tokens(vocab=400, L=32):
    torch.manual_seed(7)
    x = torch.randint(3, vocab, (1, L)); x[:, ::11] = 2
    return x


def _unzero(m):
    for l in m.layers:
        cm = getattr(l, 'cov_memory', None)
        if cm is not None:
            with torch.no_grad():
                _p = cm.W_out_b.weight if cm.W_out_b is not None else cm.W_out.weight
                _p.normal_(0.0, 0.05)


def test_zero_risk_init_bit_identical():
    # один сид, без транспланта: ветвь создаётся последней и не сдвигает RNG
    torch.manual_seed(0)
    base = EVAStack(_cfg(cov=False)).train()
    torch.manual_seed(0)
    cov = EVAStack(_cfg(cov=True)).train()
    x = _tokens()
    with torch.no_grad():
        h = base.embed(x)
        torch.manual_seed(123)          # dropout-RNG: одинаковый старт для обоих forward
        o_base, _, _, _ = base(h, None, step=1, tokens=x)
        torch.manual_seed(123)
        o_cov, _, _, _ = cov(h, None, step=1, tokens=x)
    assert torch.equal(o_base, o_cov), 'zero-init выхода не бит-в-бит residual'


def test_state_contract_5_vs_6():
    x = _tokens()
    torch.manual_seed(0)
    off = EVAStack(_cfg(cov=False)).eval()
    with torch.no_grad():
        h = off.embed(x)
        _, st_off, _, _ = off(h, None, step=1, tokens=x)
    assert len(st_off[0]) == 5, f'cov off: контракт 5-кортежа нарушен ({len(st_off[0])})'
    torch.manual_seed(0)
    on = EVAStack(_cfg(cov=True)).eval()
    with torch.no_grad():
        h2 = on.embed(x)
        _, st_on, _, _ = on(h2, None, step=1, tokens=x)
        assert len(st_on[0]) == 6, f'cov on: нет 6-го элемента ({len(st_on[0])})'
        # legacy-конверт (5 элементов) принимается: cov-ветвь стартует холодной
        legacy = tuple(st_off[0][:5])
        st_leg = [legacy] + list(st_on[1:])
        _, st_new, _, _ = on(h2, st_leg, step=2, tokens=x)
        assert len(st_new[0]) == 6 and st_new[0][5] is not None


def test_liveness_grads_flow():
    torch.manual_seed(0)
    m = EVAStack(_cfg(cov=True)).train()
    _unzero(m)
    x = _tokens()
    h = m.embed(x)
    out, st, gs, r = m(h, None, step=1, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    ce.backward()
    got = {}
    for l in m.layers:
        cm = l.cov_memory
        _op = cm.W_out_b.weight if cm.W_out_b is not None else cm.W_out.weight
        got.setdefault('k', []).append(float(cm.k_proj.weight.grad.norm()))
        got.setdefault('q', []).append(float(cm.q_proj.weight.grad.norm()))
        got.setdefault('read', []).append(float(cm.W_read.weight.grad.norm()))
        got.setdefault('out', []).append(float(_op.grad.norm()))
    for key, vals in got.items():
        assert all(v > 0 for v in vals), f'{key}: мёртвый градиент {vals}'
    assert st[0][5] is not None, 'cov-состояние не вернулось в state[5]'


def test_state_threads_and_resets():
    torch.manual_seed(0)
    m = EVAStack(_cfg(cov=True)).eval()
    _unzero(m)
    x = _tokens()
    h = m.embed(x)
    seen = []   # state-аргумент каждого вызова ветви (None ⇒ холодный старт)
    def _hook(mod, inp):
        st = inp[1] if len(inp) > 1 else None
        seen.append(st)
    m.layers[0].cov_memory.register_forward_pre_hook(_hook)
    with torch.no_grad():
        o1, st1, gs1, _ = m(h, None, step=1, tokens=x)
        s1 = st1[0][5]
        assert s1 is not None and s1.ndim == 4, f'cov state shape: {None if s1 is None else s1.shape}'
        assert seen and seen[0] is None, 'state=None: ветвь не стартовала с нуля'
        n_first = len(seen)
        o2, st2, gs2, _ = m(h, st1, global_state=gs1, step=2, tokens=x)
        s2 = st2[0][5]
        assert s2 is not None and s2.shape == s1.shape, 'состояние не протянулось'
        assert len(seen) > n_first and seen[n_first] is not None, \
            'state=st1: состояние не дошло до ветви в следующем forward'
        assert not torch.equal(s1, s2), 'состояние не обновилось между шагами'


def test_train_resume_order_not_wiped():
    """R2-лок: восстановленный stream_state не затирается до цикла обучения."""
    p = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'train.py')
    src = open(p, encoding='utf-8').read()
    i_restore = src.index("_tstate(ckpt['stream_state']")
    i_loop = src.index('# Training loop')
    seg = src[i_restore:i_loop]
    assert 'state = None' not in seg, 'stream_state затирается после восстановления'
    assert 'gs = None' not in seg, 'stream_gs затирается после восстановления'
    i_decl = src.index('state = None')
    assert i_decl < i_restore, 'state объявляется после resume-блока'
