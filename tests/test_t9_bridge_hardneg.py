# -*- coding: utf-8 -*-
"""T9 locks: hard-negative mining контрастива моста (порт FCF).

Run: python -m pytest tests/test_t9_bridge_hardneg.py -q
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.bridge import SemanticBridge


def _bridge(k, D=64, dim=32, n_layers=2):
    cfg = type('C', (), {'bridge_hard_neg_k': k})()
    torch.manual_seed(0)
    return SemanticBridge(D=D, n_layers=n_layers, bridge_dim=dim, cfg=cfg)


def _run(b, B=1, L=48, D=64, vocab=600):
    torch.manual_seed(1)
    emb = torch.nn.Embedding(vocab, D)
    x = torch.randint(1, vocab, (B, L))
    y = torch.randint(1, vocab, (B, L))
    # пробы — выходы реального probe (градиент должен течь в его параметры)
    b._preds = [b.probe(torch.randn(B, L, D)) for _ in range(b.n_layers)]
    return b.loss(y, lambda t: emb(t))


def test_hard_neg_default_off_is_full_pool_regime():
    b = _bridge(0)
    l = _run(b)
    assert l is not None and torch.isfinite(l), 'loss не конечен'
    assert float(l.detach()) > 3.0, f'k=0 должен быть многоклассовым режимом (~ln Nq), got {float(l.detach())}'


def test_hard_neg_k1_is_binary_regime():
    b = _bridge(1)
    l = _run(b)
    assert l is not None and torch.isfinite(l)
    # CE по [позитив + 1 трудный негатив] — бинарный режим (≈ln2·масштаб temp),
    # заметно ниже многоклассового полного пула
    assert float(l.detach()) < 2.5, f'k=1 должен схлопнуться к бинарному CE, got {float(l.detach())}'


def test_hard_neg_gradients_flow_to_probe_and_temp():
    b = _bridge(8)
    l = _run(b)
    l.backward()
    assert b.nce_log_temp.grad is not None and torch.isfinite(b.nce_log_temp.grad).all()
    assert any(p.grad is not None and float(p.grad.norm()) > 0
               for p in b.probe.parameters()), 'probe не получает градиент (T9)'
