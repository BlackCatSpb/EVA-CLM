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


def test_head_lacuna_off_with_memory_bank_and_temper_no_crash():
    # Round 5 (аудит): head_lacuna=False (Kp=0) + head_temper=True +
    # memory_bank=True падало AttributeError (_temper_rel) на
    # step>=head_temper_after. Комбинация валидная.
    torch.manual_seed(0)
    cfg = EVAConfig(D=64, n_layers=2, mlp_groups=2, code_dim=8, code_sparsity=2,
                    vocab=16, save_dir='.', logit_cache_enabled=False,
                    gradient_checkpointing=False, head_lacuna=False,
                    head_temper=True, head_temper_after=0, memory_bank=True,
                    intent_bridge=True, bridge_conn=0.1)
    m = EVAStack(cfg).train()
    x = torch.randint(1, 16, (1, 12))
    h = m.embed_tokens(x)
    out, state, gs, r = m(h, None, step=2000, tokens=x)
    assert torch.isfinite(out).all()
    loss, aux = m.compute_losses(out, x, h_emb=h)
    assert torch.isfinite(loss)


def test_restore_never_shares_storage_with_snapshot():
    # Round 4 (саботаж S3): restore-путь не должен отдавать модели тензоры
    # снимка — ни через _restore_value, ни через list-ветку с иной формой.
    m = _mini()
    m._reasoning_buffer = [torch.zeros(3)]                 # форма 3
    snap = m.snapshot_runtime_buffers()
    m._reasoning_buffer = [torch.full((5,), 7.0)]          # та же длина, форма 5
    m.restore_runtime_buffers(snap)
    assert m._reasoning_buffer[0].shape == (3,), 'значение не восстановлено'
    assert torch.equal(m._reasoning_buffer[0], torch.zeros(3))
    # нет шаринга хранилищ
    assert m._reasoning_buffer[0].data_ptr() != \
        snap['__attrs__']['_reasoning_buffer'][0].data_ptr(), 'list-элемент расшарен'
    m._reasoning_buffer[0].add_(11.0)
    assert float(snap['__attrs__']['_reasoning_buffer'][0].abs().sum()) == 0.0, \
        'мутация модели испортила снимок'
    # тензорный атрибут: тоже свежий клон
    m._last_logits = torch.zeros(4)
    snap2 = m.snapshot_runtime_buffers()
    m.restore_runtime_buffers(snap2)
    assert m._last_logits.data_ptr() != \
        snap2['__attrs__']['_last_logits'].data_ptr(), 'тензорный атрибут расшарен'


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
