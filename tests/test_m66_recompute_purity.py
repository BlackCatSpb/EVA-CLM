"""Блок 2 (закрытие очереди перед A100): корневой recompute-фикс gc=True.

Замок контракта: при gradient_checkpointing=True recompute-проход обязан быть
ЧИСТЫМ — не менять running-состояние (EMA/счётчики/controller-записи), чтобы
градиенты и состояние совпадали с gc=False до fp-уровня (фактически бит-в-бит
на CPU: замер 0.0 maxabs после фикса против 5.9e-3/6.8e-3 до него).

Модель — мутационно-богатый мини-конфиг: private_mem, meta_trust, memory_bank,
intent_bridge, UCL, variable_precision, inner_eye (shared EMA + S0-читатели
`_gate_ema`/`_delta_var`/`_pm_coh`/`_private_mem`/`alpha_diag`).

Саботаж-проверка (temp-копия, отключение фикса -> тест краснеет) выполнена
скриптом %TEMP%/opencode/sabotage_block2.py: `_rc = False` в core/block.py
возвращает расхождение 5.9e-3 > порога 1e-6 -> RED (см. REVISION_NOTES_block2.md).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402

THR = 1e-6


def _cfg(gc):
    return EVAConfig(
        n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
        vocab=256, seq_len=32, save_dir='.', logit_cache_enabled=False,
        memory_bank=True, intent_bridge=True, unified_concept_layer=True,
        private_mem=True, meta_trust=True, variable_precision=True,
        explicit_reasoning=False, head_lacuna=False, inner_eye=True,
        gradient_checkpointing=gc)


def _model(gc, seed=0):
    torch.manual_seed(seed)
    return EVAStack(_cfg(gc)).train()


def _tokens(seed=7, B=2, L=16, V=256):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (B, L), generator=g)


def _run_arm(gc, steps):
    """steps x (forward, backward) без optimizer.step — чистое сравнение градиентов
    и накопленного состояния двух arm'ов с одинаковым сидом."""
    m = _model(gc)
    x = _tokens()
    state = [None] * len(m.layers)
    gs = None
    rb = rc = None
    ce = None
    for t in range(steps):
        torch.manual_seed(100 + t)
        h = m.embed_tokens(x)
        out, state, gs, (rb, rc) = m(h, state, global_state=gs, step=t + 1,
                                     tokens=x, reasoning_buffer=rb,
                                     reasoning_count=rc)
        ce, _ = m.compute_losses(out, x, h_emb=h)
        m.zero_grad(set_to_none=True)
        ce.backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()
             if p.grad is not None}
    bufs = {n: b.detach().clone() for n, b in m.named_buffers()}
    return m, grads, bufs, float(ce.detach())


def _grad_maxabs(ga, gb):
    keys = sorted(set(ga) & set(gb))
    assert keys, 'нет общих параметров с градиентом — тест слепой'
    worst, wk = 0.0, None
    for k in keys:
        d = float((ga[k].double() - gb[k].double()).abs().max())
        if d > worst:
            worst, wk = d, k
    return worst, wk, keys


def test_m66_checkpointed_gradients_bit_match_single_step():
    """gc=True vs gc=False на одном сиде: single-step градиенты совпадают
    (post-fix замер 0.0; pre-fix 5.9e-3)."""
    _, g_off, _, _ = _run_arm(False, 1)
    _, g_on, _, _ = _run_arm(True, 1)
    worst, wk, keys = _grad_maxabs(g_off, g_on)
    assert worst < THR, f'gc on/off расходятся: maxabs={worst:.3e} @ {wk}'
    # не вакуумный тест: градиенты ненулевые
    assert max(float(g.abs().max()) for g in g_off.values()) > 0.0


def test_m66_recompute_does_not_advance_state_multi_step():
    """4 шага с переносом состояния: градиенты и ВСЁ running-состояние
    (буферы) бит-идентичны gc=False. Pre-fix: 70/149 буферов расходились
    (_mlp_cnt +4, pen/EMA-сдвиги), градиенты 6.8e-3."""
    m_off, g_off, b_off, _ = _run_arm(False, 4)
    m_on, g_on, b_on, _ = _run_arm(True, 4)
    worst, wk, _ = _grad_maxabs(g_off, g_on)
    assert worst < THR, f'4-step градиенты расходятся: maxabs={worst:.3e} @ {wk}'
    diff = [k for k in sorted(set(b_off) & set(b_on))
            if not torch.equal(b_off[k], b_on[k])]
    assert not diff, f'recompute сдвинул состояние: {diff[:8]}'
    # счётчики шагов: ровно один инкремент за шаг в обоих режимах
    for i, l in enumerate(m_on.layers):
        assert int(l._mlp_cnt.item()) == 4, f'L{i} _mlp_cnt != 4 (двойной re-execute)'
        assert int(l.bind._step_count.item()) == 4, f'L{i} bind._step_count != 4'
        assert int(l.mirror._fwd_count.item()) == 4, f'L{i} _fwd_count != 4'
    # alpha_diag: controller-flush исполняется раз за шаг (F4-01)
    for i in range(len(m_on.layers)):
        assert torch.equal(m_on.layers[i].mirror.alpha_diag,
                           m_off.layers[i].mirror.alpha_diag), \
            f'L{i} alpha_diag разошёлся (recompute продублировал flush)'


def test_m66_eval_snapshot_roundtrip_under_gc():
    """Снимок/восстановление runtime не сломаны фиксом: после тренировочного
    шага с gc=True snapshot->restore бит-точен, а eval-forward не течёт в
    train-состояние после restore."""
    m, _, _, _ = _run_arm(True, 2)
    snap = m.snapshot_runtime_buffers()
    before = {k: v.detach().clone() for k, v in m.named_buffers()}
    x = _tokens()
    m.eval()
    with torch.no_grad():
        h = m.embed_tokens(x)
        m(h, None, step=99, tokens=x)
    m.train()
    m.restore_runtime_buffers(snap)
    for k, v in m.named_buffers():
        assert torch.equal(v, before[k]), f'{k}: restore после eval не бит-точен'


def _run_multipass(gc):
    """Балансерный паттерн: 3x autograd.grad (aux-проходы) + backward за шаг."""
    m = _model(gc)
    x = _tokens()
    torch.manual_seed(100)
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=1, tokens=x)
    ce, _ = m.compute_losses(out, x, h_emb=h)
    params = [p for p in m.parameters() if p.requires_grad]
    for _ in range(3):
        torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
    m.zero_grad(set_to_none=True)
    ce.backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()
             if p.grad is not None}
    bufs = {n: b.detach().clone() for n, b in m.named_buffers()}
    return grads, bufs


def test_m66_multi_autograd_pass_bit_parity():
    """Повторные recompute'ы в одном шаге (CE + aux autograd.grad) обязаны
    видеть S0 первого прохода, а не значения, обновлённые hook'ами предыдущих
    проходов: без протокола `_prev_grad_norm` drift 2.2e-3 на 2-м проходе."""
    g_off, b_off = _run_multipass(False)
    g_on, b_on = _run_multipass(True)
    worst, wk, _ = _grad_maxabs(g_off, g_on)
    assert worst < THR, f'multi-pass градиенты расходятся: maxabs={worst:.3e} @ {wk}'
    diff = [k for k in sorted(set(b_off) & set(b_on))
            if not torch.equal(b_off[k], b_on[k])]
    assert not diff, f'multi-pass сдвинул состояние: {diff[:8]}'
