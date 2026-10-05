"""fixrev: замки на 7 находок глубокой ревизии EVA-CLM.

1. _stats_freeze (stack._head_frozen) глушит ВЕСЬ run-state головы, не только
   ell_ema/лестницы: банк фантомов, _pb_step, _sal_ring/_sal_ptr/_sal_q,
   _meta_thr, phantom_basis.
2. A7-div: corr нормируется тем же знаменателем, что и std (population, ÷N) —
   диагональ ровно 1, N=2 больше не завышает в 2×.
3. bounded_residual init детерминирован (без torch.randn) — post-build RNG
   ON == OFF.
4. logit_cache._ss_gen уезжает в snapshot_runtime_buffers/restore (резюм не
   replay'ит последовательность scheduled sampling с начала).
5. Мёртвый код удалён (_circ_conv_idx, mirror._snapshot/_restore_fwd_buffers);
   n_unconnected выведен в телеметрию (bal_unc).
6. ON/OFF-резюм bounded gate логирует warning; bounded_residual при
   memory_bank=None предупреждает.
"""
import math
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.config import EVAConfig          # noqa: E402
from core.stack import EVAStack            # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(**kw):
    base = dict(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                vocab=256, logit_cache_enabled=False,
                gradient_checkpointing=False, save_dir='.',
                explicit_reasoning=False, intent_bridge=False,
                memory_bank=False)
    base.update(kw)
    return EVAConfig(**base)


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(_cfg(**kw))


# ── 1: полный freeze run-state головы ───────────────────────────────────────
def test_stats_freeze_covers_head_observation_run_state():
    m = _model(head_phantom_every=1).train()
    head = m.lm_head
    head.phantom_after = 0
    head._pb_active = True
    head._thr_mode = 'static'          # observe обязан принимать наблюдения
    head.phantom_thr = -1.0
    D = m.cfg.D
    calls = {'n': 0}
    head.phantom_bank.confirmed_directions = lambda: (
        calls.__setitem__('n', calls['n'] + 1)
        or torch.ones(4, D))

    x = torch.randint(1, 256, (1, 8))
    h = m.embed_tokens(x)
    m.lm_head(h)                       # прямой train-вызов головы: _pb_step 0 -> 1
    assert int(head._pb_step) == 1
    assert int(head.phantom_bank._obs) == 1
    snap_basis = head.phantom_basis.detach().clone()
    snap_ring = head._sal_ring.clone()
    snap_ptr = int(head._sal_ptr)
    snap_thr = head._meta_thr
    snap_q = head._sal_q
    head._pb_step.fill_(3)             # 3%4 != 0; без фикса 4-й вызов двинул бы basis

    m._last_conf(h)                    # валидационные вызовы (triad/knowledge)
    m._knowledge_signal(h)
    assert int(head._pb_step) == 3, 'freeze: _pb_step сдвинулся'
    assert int(head.phantom_bank._obs) == 1, 'freeze: _pb.observe сработал'
    assert torch.equal(head._sal_ring, snap_ring), 'freeze: _sal_ring двинулся'
    assert int(head._sal_ptr) == snap_ptr, 'freeze: _sal_ptr двинулся'
    assert head._meta_thr == snap_thr, 'freeze: _meta_thr двинулся'
    assert head._sal_q == snap_q, 'freeze: _sal_q двинулся'
    assert torch.equal(head.phantom_basis.detach(), snap_basis), \
        'freeze: phantom_basis обновлён валидацией'

    m.lm_head(h)                       # прямой train-вызов двигает run-state
    assert int(head._pb_step) == 4
    assert int(head.phantom_bank._obs) == 2
    assert int(head._sal_ptr) == snap_ptr + 1
    m.lm_head(h)                       # _pb_step=4 -> 4%4==0: basis EMA
    assert not torch.equal(head.phantom_basis.detach(), snap_basis)


# ── 2: A7-div population-нормировка ─────────────────────────────────────────
def _diversity(m, group_out):
    x = torch.randint(1, 256, (1, group_out.shape[1]))
    h = m.embed_tokens(x)
    out, *_ = m(h, None, step=5, tokens=x)
    for l in m.layers:
        l.mlp._cached_group_out = group_out
    _ce, aux = m.compute_losses(out, x, h_emb=h)
    return float(aux.get('diversity', 0.0))


def _go_from_matrix(A):
    """(N,G) неотрицательная матрица -> group_out (1,N,G,G), у которого
    group_out.norm(-1) == A (норма сохраняет только модуль, знак задаётся
    ориентацией колонки)."""
    N, G = A.shape
    go = torch.zeros(1, N, G, G)
    for g in range(G):
        go[0, :, g, g] = A[:, g]
    return go


def _alpha_mean(m):
    return sum(1.0 - math.exp(-float(m.tau_config.tau_l[i].detach())
                              / float(m.tau_config.tau_min))
               for i in range(m.cfg.n_layers)) / m.cfg.n_layers


def test_diversity_corr_population_normalization_n2():
    m = _model()
    G = m.cfg.mlp_groups
    A = torch.zeros(2, G)
    A[0, ::2] = 1.0
    A[1, 1::2] = 1.0                      # N=2: колонки -> ±[1,-1]
    got = _diversity(m, _go_from_matrix(A))
    exp = ((G - 1) / G) * _alpha_mean(m)  # |corr|=1: (G-1)/G, не ×N/(N-1)
    assert abs(got - exp) < 1e-5, f'N=2: {got} != {exp}'


def test_diversity_corr_population_normalization_n16():
    m = _model()
    G = m.cfg.mlp_groups
    H = torch.ones(1, 1)
    for _ in range(4):
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    A = (H[:, 1:1 + G] + 1.0) / 2.0       # баланс 0/1 -> corr колонок = I
    got = _diversity(m, _go_from_matrix(A))
    assert got < 1e-6, f'N=16: corr колонок обязан быть I, div={got}'


# ── 3: детерминированный init bounded residual ──────────────────────────────
def _bcfg(**kw):
    base = dict(n_layers=2, D=64, mlp_groups=2, code_dim=16, code_sparsity=4,
                vocab=256, memory_bank=True, mem_min_write_mat=0.0,
                gradient_checkpointing=False, logit_cache_enabled=False,
                maturation_enabled=False, seq_len=16, batch_size=1)
    base.update(kw)
    return EVAConfig(**base)


def test_bounded_residual_init_consumes_no_global_rng():
    torch.manual_seed(123)
    EVAStack(_bcfg(bounded_residual=False))
    s_off = torch.get_rng_state()
    torch.manual_seed(123)
    m_on = EVAStack(_bcfg(bounded_residual=True))
    s_on = torch.get_rng_state()
    assert torch.equal(s_off, s_on), 'ON-сборка сдвинула глобальное RNG'
    torch.manual_seed(123)
    m_on2 = EVAStack(_bcfg(bounded_residual=True))
    assert torch.equal(m_on.memory_bank.gate_W, m_on2.memory_bank.gate_W)
    assert torch.equal(m_on.memory_bank.gate_b, m_on2.memory_bank.gate_b)
    assert torch.equal(m_on.bounded_post_norm_w[0], m_on2.bounded_post_norm_w[0])


# ── 4: _ss_gen roundtrip через снимок ──────────────────────────────────────
def test_ss_gen_state_roundtrips_snapshot():
    m = _model(logit_cache_enabled=True).train()
    lc = m.logit_cache
    g = torch.Generator()
    g.manual_seed(0)
    lc._ss_gen = g
    for _ in range(3):
        torch.rand(1, generator=g)
    snap = m.snapshot_runtime_buffers()
    ref = g.get_state().clone()        # состояние НА момент снимка
    for _ in range(7):
        torch.rand(1, generator=g)
    assert not torch.equal(g.get_state(), ref)
    m.restore_runtime_buffers(snap)
    assert torch.equal(lc._ss_gen.get_state(), ref), 'состояние _ss_gen не вернулось'
    # восстановленная последовательность детерминирована
    a = torch.rand(1, generator=lc._ss_gen).clone()
    m.restore_runtime_buffers(snap)
    b = torch.rand(1, generator=lc._ss_gen).clone()
    assert torch.equal(a, b), 'после restore последовательность не воспроизводима'


# ── 5: мёртвый код / телеметрия ─────────────────────────────────────────────
def test_dead_code_removed_and_unconnected_telemetered():
    from core.bind import TrajectorySpiralBind
    from core.mirror import GroupedCognitiveMirror
    cfg = EVAConfig(D=64, n_layers=1, mlp_groups=2, code_dim=8,
                    code_sparsity=2, vocab=64, bind_K=16, save_dir='.')
    b = TrajectorySpiralBind(64, 16, cfg)
    names = dict(b.named_buffers())
    assert '_circ_conv_idx' not in names, 'мёртвый _circ_conv_idx не удалён'
    assert '_circ_corr_idx' in names, '_circ_corr_idx (живой) удалён'
    assert not hasattr(GroupedCognitiveMirror, '_snapshot_fwd_buffers')
    assert not hasattr(GroupedCognitiveMirror, '_restore_fwd_buffers')
    src = open(os.path.join(REPO, 'scripts', 'train.py'),
               encoding='utf-8').read()
    assert 'bal_unc' in src, 'n_unconnected не выведен в телеметрию'


# ── 6: предупреждения ON/OFF и memory_bank=None ────────────────────────────
def test_bounded_gate_mismatch_warning_helper():
    from core.ckpt_io import warn_bounded_gate_mismatch
    msgs = []
    keys = warn_bounded_gate_mismatch(
        ['memory_bank.gate_W', 'bounded_post_norm_w.0', 'embed.codes'],
        ['memory_bank.gate_b'], log=msgs.append)
    assert set(keys) == {'memory_bank.gate_W', 'bounded_post_norm_w.0',
                         'memory_bank.gate_b'}
    assert msgs and 'bounded_residual' in msgs[0]
    assert warn_bounded_gate_mismatch([], [], log=msgs.append) == []
    assert len(msgs) == 1, 'warning без несовпадений не нужен'


def test_off_ckpt_into_on_model_warns():
    from core.ckpt_io import warn_bounded_gate_mismatch
    torch.manual_seed(0)
    off = EVAStack(_bcfg(bounded_residual=False))
    on = EVAStack(_bcfg(bounded_residual=True))
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    hits = warn_bounded_gate_mismatch(missing, unexpected, log=lambda *_: None)
    assert any(k.endswith('gate_W') for k in hits)
    assert any(k.endswith('gate_b') for k in hits)
    assert any('bounded_post_norm_w' in k for k in hits)


def test_bounded_without_bank_warns_and_disables():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m = EVAStack(EVAConfig(n_layers=1, D=64, mlp_groups=2, code_dim=16,
                               code_sparsity=4, vocab=64, save_dir='.',
                               logit_cache_enabled=False,
                               gradient_checkpointing=False,
                               memory_bank=False, bounded_residual=True))
    assert m._bounded_residual is False
    assert any('bounded_residual' in str(x.message) for x in w), \
        'memory_bank=None молча съел флаг'
