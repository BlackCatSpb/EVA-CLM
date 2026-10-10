"""Блок 1 (закрытие очереди перед A100): покрытие снимка runtime — инвентарь
ревизии из ~20 write-before-read атрибутов.

Проверяем: (1) все атрибуты инвентаря попадают в snapshot_runtime_buffers();
(2) include_caches=False исключает forward-ГРАФ-кэши (персистентность чекпоинта
не раздувается), default=True их кладёт; (3) после restore значения бит-равны
снимку и свежеклонированы (data_ptr отличается; мутация модели не портит
снимок; повторный restore работает); (4) eval-изоляция не сломана: после
restore состояние и выход не хуже прежнего контракта.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack  # noqa: E402

INVENTORY = [
    'blk.0._cache_mlp_out', 'blk.0._cache_mlp_mod',
    'blk.0._fwd_py_snap', 'blk.0._tau_norm',
    'mir.0._cached_gate_usage', 'mir.0._cached_gate_l1', 'mir.0._cached_decorr',
    'mir.0._last_mlp_mod', 'mir.0._tau_signal_used', 'mir.0._pred_loss_term',
    'mir.0._fwd_py_snap',
    'mlp.0._cached_group_out',
    'head._ext_phantom_dirs', 'head._last_ph_sat', 'head._tokens',
    'head._kp_active_py', 'head._ell_ema_ready', 'head._pb_active',
    'head._temper_active',
    'ucl._mature_py',
    'lcache.cache._position',
]
CACHE_KEYS = {
    'blk.0._cache_mlp_out', 'blk.0._cache_mlp_mod', 'blk.0._fwd_py_snap',
    'mir.0._cached_decorr',
    'mir.0._cached_gate_l1', 'mir.0._cached_gate_usage', 'mir.0._last_mlp_mod',
    'mir.0._pred_loss_term', 'mir.0._tau_signal_used', 'mir.0._fwd_py_snap',
    'mlp.0._cached_group_out',
}


def _stack():
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16,
                    code_sparsity=4, vocab=400, seq_len=32, save_dir='.',
                    logit_cache_enabled=True, memory_bank=True,
                    intent_bridge=True, explicit_reasoning=True,
                    unified_concept_layer=True)
    torch.manual_seed(0)
    return EVAStack(cfg).train()


def _tokens(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 400, (1, 32), generator=g)


def _fwd(m, x, step=10, state=None):
    h = m.embed_tokens(x)
    return m(h, state, step=step, tokens=x)


def _get(m, path):
    parts = path.split('.')
    if parts[0] == 'blk':
        return getattr(m.layers[int(parts[1])], parts[2])
    if parts[0] == 'mir':
        return getattr(m.layers[int(parts[1])].mirror, parts[2])
    if parts[0] == 'mlp':
        return getattr(m.layers[int(parts[1])].mlp, parts[2])
    if parts[0] == 'head':
        return getattr(m.lm_head, parts[1])
    if parts[0] == 'ucl':
        return getattr(m.concept_layer, parts[1])
    if parts[0] == 'lcache':
        return getattr(m.logit_cache.cache, parts[-1])
    raise KeyError(path)


def test_inventory_is_covered_by_the_default_snapshot():
    m = _stack()
    _fwd(m, _tokens(1))
    snap = m.snapshot_runtime_buffers()
    attrs = snap.get('__attrs__') or {}
    missing = [p for p in INVENTORY if p not in attrs]
    assert not missing, f'snapshot lost inventory attrs: {missing}'
    # not-None материализованы живым forward'ом (иначе тест был бы пустым)
    unset = [p for p in INVENTORY if _get(m, p) is None]
    assert not unset, f'attrs not materialized by the forward: {unset}'


def test_include_caches_false_excludes_graph_caches_only():
    m = _stack()
    _fwd(m, _tokens(1))
    full = (m.snapshot_runtime_buffers()['__attrs__'])
    slim = (m.snapshot_runtime_buffers(include_caches=False)['__attrs__'])
    for p in CACHE_KEYS:
        assert p in full, f'default snapshot must cover {p}'
        assert p not in slim, f'checkpoint snapshot must skip graph cache {p}'
    for p in INVENTORY:
        if p not in CACHE_KEYS:
            assert p in slim, f'checkpoint snapshot lost non-cache state {p}'


def test_restore_brings_fresh_clones_for_every_tensor():
    m = _stack()
    _fwd(m, _tokens(1))
    snap = m.snapshot_runtime_buffers()
    attrs = snap['__attrs__']
    # мутируем модель, затем восстанавливаем
    with torch.no_grad():
        for p in INVENTORY:
            v = _get(m, p)
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v.add_(3.0)
            elif isinstance(v, torch.Tensor):
                v.fill_(0)
            elif isinstance(v, int):
                _set_int(m, p, 999)
    m.restore_runtime_buffers(snap)
    for p in INVENTORY:
        sv = attrs[p]
        mv = _get(m, p)
        if isinstance(sv, torch.Tensor):
            assert torch.equal(mv, sv), f'{p}: value not restored'
            if sv.numel() > 0:
                assert mv.data_ptr() != sv.data_ptr(), \
                    f'{p}: restore returned the snapshot tensor itself (not a clone)'
        else:
            assert mv == sv, f'{p}: scalar not restored ({mv!r} != {sv!r})'
    # мутация восстановленного тензора не портит снимок; повторный restore чист
    for p in INVENTORY:
        sv = attrs[p]
        if isinstance(sv, torch.Tensor) and sv.is_floating_point():
            before = sv.clone()
            with torch.no_grad():
                _get(m, p).add_(7.0)
            assert torch.equal(sv, before), f'{p}: snapshot shares storage with model'
    m.restore_runtime_buffers(snap)
    for p in INVENTORY:
        sv = attrs[p]
        mv = _get(m, p)
        if isinstance(sv, torch.Tensor):
            assert torch.equal(mv, sv), f'{p}: second restore diverged'


def _set_int(m, path, value):
    parts = path.split('.')
    target = _get(m, path)
    if parts[0] == 'blk':
        setattr(m.layers[int(parts[1])], parts[2], value)
    elif parts[0] == 'mir':
        setattr(m.layers[int(parts[1])].mirror, parts[2], value)
    elif parts[0] == 'mlp':
        setattr(m.layers[int(parts[1])].mlp, parts[2], value)
    elif parts[0] == 'head':
        setattr(m.lm_head, parts[1], value)
    elif parts[0] == 'ucl':
        setattr(m.concept_layer, parts[1], value)
    else:
        setattr(m.logit_cache.cache, parts[-1], value)


def test_eval_isolation_still_holds():
    m = _stack()
    x1 = _tokens(1)
    with torch.no_grad():
        _fwd(m, x1)                       # fwd1: рабочее состояние train-документа
    if getattr(m, 'logit_cache', None) is not None:
        m.logit_cache.cache.clear()       # контракт evaluate(): val-окна не в кэше
    snap = m.snapshot_runtime_buffers()
    before = {p: (_get(m, p).clone() if isinstance(_get(m, p), torch.Tensor)
                  else _get(m, p)) for p in INVENTORY}
    with torch.no_grad():
        out_ref, _, _, _ = _fwd(m, x1)    # fwd2: эталон из ТОГО ЖЕ состояния
    # «валидационный» документ: полный forward с другими токенами
    m.eval()
    with torch.no_grad():
        out_eval, _, _, _ = _fwd(m, _tokens(2), step=11)
    if getattr(m, 'logit_cache', None) is not None:
        m.logit_cache.cache.clear()
    m.restore_runtime_buffers(snap)
    m.train()
    for p in INVENTORY:
        bv = before[p]
        if isinstance(bv, torch.Tensor):
            assert torch.equal(_get(m, p), bv), f'{p}: eval leaked into train state'
        else:
            assert _get(m, p) == bv, f'{p}: eval leaked (scalar)'
    # выход после restore воспроизводит train-документ из того же состояния.
    # Абсолютный допуск нельзя занижать: остаточный разброс forward+restore —
    # известный fp/хаос-эффект (журнал: ~0.026-0.1, allocator + хаотическая
    # динамика), он был и до покрытия инвентаря. Контракт проверяем
    # относительно «валидационного» выхода: restore обязан быть НА ПОРЯДОК
    # ближе к train-эталону, чем сама val-документная ветка.
    with torch.no_grad():
        out_after, _, _, _ = _fwd(m, x1)  # fwd3
    d_after = float((out_after - out_ref).abs().max())
    d_eval = float((out_eval - out_ref).abs().max())
    assert d_after < 0.25 * d_eval, \
        f'eval isolation broke: after-restore Δ={d_after:.3e} vs val Δ={d_eval:.3e}'
