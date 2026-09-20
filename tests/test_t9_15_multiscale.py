"""T9.15: многоразрешающий кэш — пулы K/V на фиксированных шкалах τ.

Проверяем: (1) выключенная рука — no-op; (2) атомы младшей шкалы и bottom-up
сборка старших (×4); (3) значения пулов = средние; (4) причинность на уровне
пулов (только завершённые блоки); (5) кольца/лимиты/clear; (6) чтение и
градиентный путь; (7) B>1 не смешивает батчи.
"""
import torch

from core.logit_cache import LogitCache, LogitAttention


def _fwd(att, c, h):
    """store() метаданных -> attention: тот же порядок, что в wrapper.augment."""
    c.store(h, training=True, novelty=1.0)
    return att(h, c, training=True, tokens=None)


def _mk(D=16, kv=8, spans=(8, 32), ms_max=8, max_entries=8):
    att = LogitAttention(D=D, V=64, n_heads=2, kv_dim=kv, codes=None,
                         sentence_ring=False, ms_spans=spans)
    c = LogitCache(V=64, D=D, max_entries=max_entries, ms_spans=spans,
                   ms_max=ms_max)
    return att, c


def test_ms_off_is_noop():
    att = LogitAttention(D=16, V=64, n_heads=2, kv_dim=8, codes=None,
                         sentence_ring=False, ms_spans=())
    assert att.ms_spans == () and att.ms_emb is None
    c = LogitCache(V=64, D=16, max_entries=4, ms_spans=())
    assert c.ms_spans == () and c._kv_ms == {}
    # forward не должен ничего создавать
    h = torch.randn(1, 16, 16)
    _fwd(att, c, h)
    assert c._kv_ms == {} and att._last_ms_mass is None


def test_ms_atoms_bottom_up_counts():
    torch.manual_seed(0)
    att, c = _mk(spans=(8, 32), ms_max=8)
    h = torch.randn(1, 16, 16)              # окно 16 = 2 атома по 8
    _fwd(att, c, h)
    assert len(c._kv_ms[8]) == 2            # 2 завершённых атома
    assert len(c._kv_ms[32]) == 0           # 2 < 4 — старшая ещё не собрана
    att(h, c, training=True, tokens=None)   # ещё 2 атома → ×4 → пул 32
    assert len(c._kv_ms[8]) == 4
    assert len(c._kv_ms[32]) == 1
    assert c._ms_lens[32][0] == 32          # счётчик шкалы — в токенах


def test_ms_pool_values_are_means():
    torch.manual_seed(1)
    att, c = _mk(spans=(8,), ms_max=8)
    h = torch.randn(1, 8, 16)
    _fwd(att, c, h)
    # ожидаемый атом = среднее k_new по 8 токенам
    k_new = att.k_norm(att.k_proj_h(h)).detach()
    exp = k_new.mean(dim=1, keepdim=True)
    got = c._kv_ms[8][0][0]
    assert torch.allclose(got, exp, atol=1e-5), (got - exp).abs().max()


def test_ms_causality_completed_only():
    torch.manual_seed(2)
    att, c = _mk(spans=(8,), ms_max=8)
    h = torch.randn(1, 20, 16)              # 20 = 2 полных атома + 4 токена
    _fwd(att, c, h)
    assert len(c._kv_ms[8]) == 2            # только завершённые 0-7 и 8-15
    k_new = att.k_norm(att.k_proj_h(h)).detach()
    assert torch.allclose(c._kv_ms[8][1][0], k_new[:, 8:16].mean(1, True),
                          atol=1e-5)


def test_ms_ring_cap_and_clear():
    torch.manual_seed(3)
    att, c = _mk(spans=(8,), ms_max=3, max_entries=3)
    h = torch.randn(1, 8, 16)
    for _ in range(6):
        _fwd(att, c, h)
    assert len(c._kv_ms[8]) == 3            # ms_max держит
    assert c.size_mb() > 0
    c.clear()
    assert c._kv_ms[8] == [] and c._ms_lens[8] == []


def test_ms_read_and_gradient_path():
    torch.manual_seed(4)
    att, c = _mk(spans=(8,), ms_max=8)
    h = torch.randn(1, 16, 16, requires_grad=True)
    out = _fwd(att, c, h)
    assert out.shape == h.shape
    # чтение пулов шло: телеметрия массы уровня заполнена
    assert att._last_ms_mass is not None and att._last_ms_mass.shape == (1,)
    out.pow(2).mean().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    # level-эмбеддинг получает градиент (иначе выбор уровня не обучается)
    assert att.ms_emb.weight.grad is not None


def test_ms_batch_guard_b1():
    torch.manual_seed(5)
    att, c = _mk(spans=(8,), ms_max=8)
    h = torch.randn(2, 16, 16)
    _fwd(att, c, h)
    # при смене B кольца пересобираются, а не смешиваются
    h1 = torch.randn(1, 16, 16)
    _fwd(att, c, h1)
    for _k, _v in c._kv_ms[8]:
        assert _k.shape[0] == 1
