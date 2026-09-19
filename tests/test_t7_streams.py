# -*- coding: utf-8 -*-
"""T7 locks: единый контракт холодного рестарта стримов (reset_streams),
тёплый resume intent-потока и его покрытие snapshot/restore.

Run: python -m pytest tests/test_t7_streams.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack


def _mini(**kw):
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=600, save_dir='.', intent_bridge=True, bridge_conn=0.1,
                    logit_cache_enabled=False, memory_bank=False, **kw)
    torch.manual_seed(0)
    return EVAStack(cfg).train()


def test_reset_streams_clears_all():
    """reset_streams: intent/bus/salience → None, bridge-stream → 0 (T7)."""
    m = _mini()
    m._intent_stream = [torch.ones(1, 1, m._n_experts, m._K_max)
                        for _ in range(2)]
    m._last_bus = torch.ones(3)
    m._last_salience = torch.ones(2)
    assert m.bridge is not None
    m.bridge.bridge_stream.fill_(1.0)
    m.reset_streams()
    assert m._intent_stream is None, 'intent-поток пережил границу документа (T7)'
    assert m._last_bus is None, 'bus пережил границу документа (T7)'
    assert m._last_salience is None, 'salience пережил границу документа (T7)'
    assert float(m.bridge.bridge_stream.abs().max()) == 0.0, \
        'bridge-stream не обнулён (T7)'


def test_forward_fills_intent_stream_and_snapshot_covers_it():
    """forward заполняет intent-поток; snapshot/restore (eval-изоляция) его покрывает."""
    m = _mini()
    x = torch.randint(1, 600, (1, 16))
    h = m.embed_tokens(x)
    m(h.clone(), None, step=5, tokens=x)
    assert isinstance(m._intent_stream, list) and len(m._intent_stream) == 2, \
        'forward не заполнил _intent_stream (T7)'
    snap = m.snapshot_runtime_buffers()
    m.reset_streams()
    assert m._intent_stream is None
    m.restore_runtime_buffers(snap)
    assert isinstance(m._intent_stream, list), \
        'snapshot/restore не покрывает _intent_stream (T7: eval-изоляция)'


def test_intent_stream_roundtrip_serialization():
    """T7: список тензоров intent-потока переживает detach→cpu→to(device) roundtrip."""
    m = _mini()
    x = torch.randint(1, 600, (1, 16))
    h = m.embed_tokens(x)
    m(h.clone(), None, step=5, tokens=x)
    saved = [s.detach().cpu() for s in m._intent_stream]
    m._intent_stream = None
    m._intent_stream = [s.to('cpu') for s in saved]
    assert all(torch.equal(a, b) for a, b in zip(saved, m._intent_stream))
