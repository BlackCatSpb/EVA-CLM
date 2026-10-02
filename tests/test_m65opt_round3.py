"""M65-opt round 3 (адверсариальный аудит): замки на исправленные дефекты.

1. Снимок/restore: тип-сохранение (tuple/dict), свежие клоны, устойчивость
   к укороченным спискам (раньше IndexError), отсутствие шаринга ссылок.
2. STE-пол скана: forward бит-в-бит жёсткий кламп, градиент прозрачен
   (test_b19_scan_floor.py).
3. bind_traj_dims=1 не падает (test_m65opt_fft_bind.py).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.stack import EVAStack        # noqa: E402
from core.config import EVAConfig      # noqa: E402


def _mini():
    torch.manual_seed(0)
    cfg = EVAConfig(D=64, n_layers=2, mlp_groups=2, code_dim=8, code_sparsity=2,
                    vocab=16, save_dir='.', logit_cache_enabled=False,
                    gradient_checkpointing=False, intent_bridge=True,
                    bridge_conn=0.1)
    return EVAStack(cfg).eval()


def test_snapshot_preserves_types_and_clones_values():
    m = _mini()
    assert m.bridge is not None and m.lm_head is not None
    m._last_salience = (torch.zeros(4), torch.ones(3))          # main-attrs, tuple
    m._reasoning_buffer = [torch.zeros(6)]                      # main-attrs, list
    m.lm_head._spike_stats = {'a': torch.zeros(5), 'b': 'scalar'}   # head, dict
    m.layers[0].mirror._cached_concept_dendrogram = (torch.zeros(2),)  # mirror, tuple
    m.bridge._preds = [torch.zeros(7)]                          # bridge, list
    snap = m.snapshot_runtime_buffers()
    snap_clean = m.snapshot_runtime_buffers()      # для проверки restore ниже
    ex = snap['__attrs__']

    # снимок не шарит ссылки: мутация снимка не трогает модель
    ex['_last_salience'][0].add_(100.0)
    ex['head._spike_stats']['a'].add_(100.0)
    assert float(m._last_salience[0].abs().sum()) == 0.0, 'tuple расшарен'
    assert float(m.lm_head._spike_stats['a'].abs().sum()) == 0.0, 'dict расшарен'
    assert isinstance(ex['_last_salience'], tuple), 'снимок потерял тип tuple'
    assert isinstance(ex['head._spike_stats'], dict), 'снимок потерял тип dict'

    # мутация модели не трогает снимок
    m._last_salience[1].add_(7.0)
    m.bridge._preds[0].add_(9.0)
    assert float(ex['_last_salience'][1].sum()) == 3.0, 'снимок мутирован'
    assert float(ex['bridge._preds'][0].abs().sum()) == 0.0

    m.restore_runtime_buffers(snap_clean)
    assert isinstance(m._last_salience, tuple), 'restore потерял тип tuple'
    assert isinstance(m.lm_head._spike_stats, dict), 'restore потерял тип dict'
    assert isinstance(m._reasoning_buffer, list), 'restore потерял тип list'
    assert isinstance(m.bridge._preds, list)
    assert torch.equal(m._last_salience[1], torch.ones(3)), 'tuple не восстановлен'
    assert torch.equal(m.lm_head._spike_stats['a'], torch.zeros(5)), 'dict не восстановлен'
    assert m.lm_head._spike_stats['b'] == 'scalar'
    assert torch.equal(m.bridge._preds[0], torch.zeros(7)), 'bridge._preds не восстановлен'
    assert torch.equal(m.layers[0].mirror._cached_concept_dendrogram[0],
                       torch.zeros(2)), 'dendrogram не восстановлен'
    assert torch.equal(m._reasoning_buffer[0], torch.zeros(6))


def test_restore_shortened_and_lengthened_lists_no_crash():
    m = _mini()
    m._reasoning_buffer = [torch.zeros(3), torch.ones(3)]
    snap = m.snapshot_runtime_buffers()
    m._reasoning_buffer = [torch.full((3,), 5.0)]          # укорочен — раньше IndexError
    m.restore_runtime_buffers(snap)
    assert len(m._reasoning_buffer) == 2
    assert torch.equal(m._reasoning_buffer[0], torch.zeros(3))
    # удлинённый текущий список тоже не ломает restore
    m._reasoning_buffer = [torch.zeros(3), torch.zeros(3), torch.zeros(3)]
    m.restore_runtime_buffers(snap)
    assert len(m._reasoning_buffer) == 2
    assert torch.equal(m._reasoning_buffer[1], torch.ones(3))
