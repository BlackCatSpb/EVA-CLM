"""M65-OPT-2: замки на критичные находки аудита агентов (A1-A9, B1, D-blind).

Каждый тест воспроизводит находку и фиксирует фикс:
  A1 head_mode != sigmoid_coded падал (_st unbound);
  A2 резюм терял non-persistent runtime (27 буферов) — snapshot в чекпоинт;
  A5 gradient checkpointing мутировал EMA в backward (recompute-примесь);
  A7 B*L==1 -> diversity NaN; A8 пустой батч; A9 ms-аккумуляторы через clear();
  B1 микро-стена покупала третий backward;
  D: PAD-маска coded-CE, all-PAD NaN, restore None-состояния, emphasis
  data-части, τ-температура сигналов, строгость blacklist, ёмкость ring.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


def _cfg(**kw):
    base = dict(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                vocab=256, logit_cache_enabled=False,
                gradient_checkpointing=False, save_dir='.')
    base.update(kw)
    return EVAConfig(**base)


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(_cfg(**kw))


# ── A1: головы без srl_on падали на каждом forward ──
def test_head_modes_partitioned_and_cognitive_forward():
    for hm in ('partitioned', 'cognitive_coded'):
        m = _model(head_mode=hm).train()
        x = torch.randint(1, 256, (1, 8))
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=5, tokens=x)
        assert torch.isfinite(out).all(), hm
        ce, _ = m.compute_losses(out, x, h_emb=h)
        ce.backward()
        assert m.lm_head is not None


# ── A2: end-to-end резюм (state_dict + runtime + RNG) бит-в-бит ──
def test_resume_with_runtime_state_is_bit_equal():
    torch.manual_seed(1)
    x = torch.randint(1, 256, (1, 16))
    a = _model().train()
    h = a.embed_tokens(x)
    a(h, None, step=1, tokens=x)
    sd = {k: v.detach().clone() for k, v in a.state_dict().items()}
    rt = a.snapshot_runtime_buffers()
    rng = torch.get_rng_state()
    h2 = a.embed_tokens(x)
    out_a, *_ = a(h2, None, step=2, tokens=x)
    ref = {k: v.detach().clone() for k, v in a.state_dict().items()}

    torch.manual_seed(2)          # другой сид — всё должно восстановиться
    b = _model().train()
    b.load_state_dict(sd, strict=False)
    b.restore_runtime_buffers(rt)
    torch.set_rng_state(rng)
    h2b = b.embed_tokens(x)
    out_b, *_ = b(h2b, None, step=2, tokens=x)
    assert torch.equal(out_a, out_b), 'выход после резюма разошёлся'
    for k, v in ref.items():
        w = b.state_dict()[k]
        if v.is_floating_point():
            assert torch.equal(v, w), f'{k} разошёлся после резюма'


# ── A5: recompute-чистота (gc=True) ──
def _iso_model(gc):
    return _model(gradient_checkpointing=gc, explicit_reasoning=False,
                  bridge_conn=0.0, unified_concept_layer=False,
                  private_mem=False, memory_bank=False, intent_bridge=False,
                  variable_precision=False, head_lacuna=False).train()


def test_checkpoint_recompute_does_not_mutate_ema():
    for gc in (False, True):
        m = _iso_model(gc)
        x = torch.randint(1, 256, (2, 16))
        ema = m.layers[0].mirror._signal_norm_ema
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=6, tokens=x)
        e1 = ema.detach().clone()          # ПОСЛЕ forward, ДО backward
        ce, _ = m.compute_losses(out, x, h_emb=h)
        ce.backward()
        assert torch.equal(ema, e1), f'gc={gc}: backward мутировал EMA'


def test_checkpoint_gradients_match_uncheckpointed():
    grads = {}
    for gc in (False, True):
        m = _iso_model(gc)
        torch.manual_seed(7)
        x = torch.randint(1, 256, (2, 16))
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=5, tokens=x)
        ce, _ = m.compute_losses(out, x, h_emb=h)
        ce.backward()
        grads[gc] = {n: p.grad.detach().clone()
                     for n, p in m.named_parameters() if p.grad is not None}
    keys = set(grads[False]) & set(grads[True])
    worst = 0.0
    for k in keys:
        d = float((grads[False][k] - grads[True][k]).abs().max())
        worst = max(worst, d)
    # gc=False (чистый путь) обязан быть точным; gc=True: остаток pen-пути
    # 6.9e-6 (замер агента A), корневой фикс — в очереди (см. xfail ниже).
    # Примесь ДО фикса зеркала была 5.8e-3 — замок держит её подавленной.
    assert worst < 1e-4, f'градиенты gc on/off расходятся: {worst:.2e}'


# ── A7/A8/A9, B1 ──
def test_bl1_diversity_is_finite():
    m = _model().train()
    x = torch.randint(1, 256, (1, 1))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=5, tokens=x)
    m.compute_losses(out, x, h_emb=h)
    for k in ('ce', 'div', 'balance', 'signal_ent'):
        v = m._cached_losses[k]
        assert v == v, f'{k} = NaN при B*L==1'


def test_empty_batch_does_not_crash():
    m = _model()
    with torch.no_grad():
        h = m.embed_tokens(torch.zeros(0, 4, dtype=torch.long))
    assert h.shape[0] == 0


def test_ms_accumulators_cleared_on_cache_clear():
    # fixrev-7: precondition-assert + реальный путь нового документа.
    # Аккумуляторы _ms_acc_* живут на LogitAttention (не на .cache);
    # старый тест смотрел .cache и проходил пусто (guard -> return).
    m = _model(logit_cache_enabled=True)
    att = getattr(m.logit_cache, 'attention', None)
    assert att is not None, 'precondition: attention не построен'
    acc = getattr(att, '_ms_acc_n', None)
    assert acc is not None, 'precondition: _ms_acc_n отсутствует — тест слепой'
    acc[8] = 5
    m.logit_cache.clear()              # граница документа (модульный clear)
    assert len(acc) == 0, 'многоразрешающие аккумуляторы пережили clear()'
    acc[8] = 5
    m.reset_cache()                    # путь stack.reset_cache
    assert len(acc) == 0, 'reset_cache не почистил attention-аккумуляторы'


def test_head_wall_not_emitted_for_micro_values():
    # fixrev-7: безусловный сценарий вместо `if 'head_wall' in aux`. Wall
    # читает _last_u в losses; в eval головной forward его НЕ перезаписывает
    # (embedding.py: training-only), поэтому оба режима задаются явно.
    m = _model().eval()
    x = torch.randint(1, 256, (1, 16))
    with torch.no_grad():
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=5, tokens=x)
        u0 = float(m.cfg.head_u_wall_u0)
        m.lm_head._last_u = torch.full((1, 16, 4), u0 + 1e-4)
        _ce, aux = m.compute_losses(out, x, h_emb=h)
        assert 'head_wall' not in aux, 'микро-стена купила третий backward'
        m.lm_head._last_u = torch.full((1, 16, 4), u0 + 6.0)
        _ce, aux = m.compute_losses(out, x, h_emb=h)
    assert 'head_wall' in aux, 'реальное насыщение не эмитит стену'
    assert float(aux['head_wall'].detach()) > 1e-6


# ── D: слепые зоны, доказанные мутационным тестированием ──
def test_pad_excluded_from_coded_ce():
    m = _model()
    x = torch.randint(1, 256, (1, 8))
    x[0, 0] = 0                                   # PAD!
    with torch.no_grad():
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=5, tokens=x)
        m.compute_losses(out, x, h_emb=h)
        lp = m.lm_head.log_probs_for_target(out.reshape(-1, out.shape[-1]),
                                            x.reshape(-1), bus_bias=None)
    mask = (x.reshape(-1) != 0).float()
    ref = float((-lp * mask).sum() / mask.sum())
    assert abs(m._cached_losses['ce_raw'] - ref) < 1e-4


def test_all_pad_batch_ce_finite():
    m = _model()
    x = torch.randint(1, 256, (1, 8))
    y = torch.zeros_like(x)
    with torch.no_grad():
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=5, tokens=x)
        ce, _ = m.compute_losses(out, y, h_emb=h)
    assert torch.isfinite(ce), 'полностью PAD-батч дал NaN/Inf'


def test_restore_returns_none_state():
    m = _model()
    m._last_bus = None
    snap = m.snapshot_runtime_buffers()
    m._last_bus = torch.ones(3)
    m.restore_runtime_buffers(snap)
    assert m._last_bus is None, 'restore не вернул None-состояние'


def test_signal_weights_apply_tau_signal():
    m = _model()
    mir = m.layers[0].mirror
    with torch.no_grad():
        mir._tau_norm_layer = 0.5
        n_sig = mir._signal_log_weights.numel()
        mir._signal_log_weights.copy_(
            torch.linspace(-1.0, 1.0, n_sig))
        w = mir._signal_weights()
        tau = float(mir._tau_signal_used)
    assert tau > 0.0
    ref = torch.sigmoid(mir._signal_log_weights / tau)
    assert torch.allclose(w, ref, atol=1e-6), 'τ-температура сигналов не применена'


def test_blacklist_threshold_is_strict():
    from core.birth_ledger import BirthLedger
    led = BirthLedger()
    d = torch.zeros(8)
    d[0] = 1.0
    q = torch.zeros(8)
    q[0] = 0.9
    q[1] = (1 - 0.81) ** 0.5
    q = q / q.norm()
    cos = float(abs(d @ q))
    led.blacklist.append(dict(d=d, until=1000))
    assert led.allow_birth(q, step=500, thr=cos) is True, 'cos==thr обязан НЕ блокировать'
    assert led.allow_birth(q, step=500, thr=cos - 1e-6) is False


def test_bind_pen_dead_in_clean_path():
    # ИСПРАВЛЕНО (корневой фикс _soft_floor в core/block.py): жёсткий
    # clamp_min(log_a, k*log(d_s)) в _scan_chunks у медленной лестницы
    # (d_s -> 1) клампил ВСЕ входы (frac_clamped=1.0) -> локальный якобиан
    # d(combined)/d(decay) == 0 -> bind-параметры последнего слоя получали
    # ровно нулевой градиент в чистом gc=False пути. Мягкий пол возвращает
    # градиент (softplus-колено), forward меняется <= ~0.7% на зажатых входах.
    # Этот тест — замок на отсутствие регрессии: он ОБЯЗАН проходить.
    cfg = _cfg(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4,
               vocab=1820, gradient_checkpointing=False,
               intent_bridge=True, memory_bank=True, bridge_conn=0.1,
               unified_concept_layer=True, explicit_reasoning=True)
    torch.manual_seed(0)
    m = EVAStack(cfg).train()
    for b in m.layers:
        with torch.no_grad():
            b.precision_gate.gate.bias.fill_(3.0)
    opt = torch.optim.SGD(m.parameters(), lr=0.02)
    state = None
    for it in range(4):
        x = torch.randint(1, m.cfg.vocab, (1, 64))
        x[:, 31] = 2
        with torch.no_grad():
            h0 = m.embed_tokens(x)
            o0, _, _, _ = m(h0, state, step=20000 + it, tokens=x)
            m.observe_output(m.lm_head(o0))
        h = m.embed_tokens(x)
        out, state, gs, r = m(h, state, step=20000 + it, tokens=x)
        loss, aux = m.compute_losses(out, x, h_emb=h)
        total = loss + sum(v for v in aux.values() if isinstance(v, torch.Tensor))
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        m._reasoning_buffer, m._reasoning_count = r
    dead = [n for n, pp in m.named_parameters()
            if n.startswith('layers.1.') and n.split('.')[-1] in
            ('w_d', 'b_d', 'w_d_pen')
            and (pp.grad is None or float(pp.grad.abs().sum()) == 0.0)]
    assert not dead, f'bind-параметры мёртвы в чистом пути: {dead}'
