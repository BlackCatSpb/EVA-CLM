# -*- coding: utf-8 -*-
"""T9.9 шаг 2 лок: sentence-ring кэша (пулы K/V предложений, второй уровень).

Run: python -m pytest tests/test_t9_sent_ring.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack


def _model(ring=True):
    torch.manual_seed(0)
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=True,
                    logit_cache_kv_dim=64, logit_cache_sentence_ring=ring,
                    memory_bank=False, intent_bridge=True, vsa_decay_floor_k=2.0,
                    gradient_checkpointing=False)
    return cfg, EVAStack(cfg).train()


def _tokens(cfg, L=24):
    torch.manual_seed(7)
    x = torch.randint(3, cfg.vocab, (1, L))
    x[:, 7] = 2
    x[:, 15] = 2
    x[:, 23] = 2
    return x


def test_sentence_ring_fills_on_seps():
    cfg, m = _model(True)
    x = _tokens(cfg)
    h = m.embed(x)
    m(h, None, step=1, tokens=x)
    c = m.logit_cache.cache
    assert len(c._kv_sent) == 3, f'ring: {len(c._kv_sent)} (ожидалось 3 SEP)'
    for k, v in c._kv_sent:
        assert k.shape == (1, 1, 64) and v.shape == (1, 1, 64), 'не пулы предложений'
    assert c._sent_lens == [8, 8, 8], f'длины сегментов: {c._sent_lens}'
    # второй forward: ещё 3 (сегменты окна 2)
    h2 = m.embed(x)
    m(h2, None, step=2, tokens=x)
    assert len(c._kv_sent) == 6


def test_ring_read_increases_m():
    """Структурный лок чтения: M (последняя ось attn-весов) = токены окна +
    записи sentence-ring (вклад гейта ~4.5e-5 — по выходу не отличить)."""
    cfg, m = _model(True)
    att = m.logit_cache.attention
    c = m.logit_cache.cache
    c.push_kv_sent(torch.randn(1, 1, 64), torch.randn(1, 1, 64), 8)
    c.push_kv_sent(torch.randn(1, 1, 64), torch.randn(1, 1, 64), 8)
    x = _tokens(cfg)
    h = m.embed(x)
    out, w = att(h, c, training=True, return_attention=True, tokens=x)
    # окно (24) + ring до вызова (2) + предложения текущего окна (3 SEP) = 29
    assert w.shape[-1] == 24 + 2 + 3, f'M={w.shape[-1]} (ring не подключён?)'
    assert len(c._kv_sent) == 5


def test_ring_off_switch():
    cfg, m = _model(False)
    x = _tokens(cfg)
    h = m.embed(x)
    m(h, None, step=1, tokens=x)
    assert len(m.logit_cache.cache._kv_sent) == 0, 'ring пишется при выключенном флаге'


def test_ring_clear_and_evict():
    cfg, m = _model(True)
    c = m.logit_cache.cache
    c.max_entries = 2
    x = _tokens(cfg)
    for s in range(1, 4):
        h = m.embed(x)
        m(h, None, step=s, tokens=x)
    assert len(c._kv_sent) <= 2, f'эвикция ring: {len(c._kv_sent)}'
    assert len(c._sent_lens) == len(c._kv_sent), 'метаданные ring рассинхронизированы'
    c.clear()
    assert c._kv_sent == [] and c._sent_lens == []


def test_ring_memory_small():
    cfg, m = _model(True)
    x = _tokens(cfg)
    for s in range(1, 6):
        h = m.embed(x)
        m(h, None, step=s, tokens=x)
    mb = m.logit_cache.cache.size_mb(training=True)
    # 5 окон (16? окон по 24 токена) + ring: при kv=64 всё мало
    assert mb < 1.0, f'память кэша: {mb}MB'
