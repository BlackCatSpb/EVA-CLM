# -*- coding: utf-8 -*-
"""T9.8 лок: кэш логитов — выход оператора, а не состояние.

1. Мёртвое h-кольцо снято (252MB при 64×384×D): хранились тензоры h, которые
   модель никогда не читала (`retrieve(training=True)` жил только в dev-скрипте);
   метаданные (lens/scores) для M34-удержания сохранены.
2. Low-rank K/V: kv_dim>0 — пространство внимания обеих сторон (трейн/инференс),
   0 = D (прежнее поведение).

Run: python -m pytest tests/test_t9_cache_operator.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack


def _model(kv_dim=0):
    torch.manual_seed(0)
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=True,
                    logit_cache_kv_dim=kv_dim, memory_bank=False,
                    intent_bridge=True, vsa_decay_floor_k=2.0,
                    gradient_checkpointing=False)
    return cfg, EVAStack(cfg).train()


def test_h_ring_removed_metadata_kept():
    cfg, m = _model()
    c = m.logit_cache.cache
    assert not hasattr(c, '_h_cache'), 'мёртвое h-кольцо не снято'
    x = torch.randint(3, cfg.vocab, (1, 16))
    h = m.embed(x)
    m(h, None, step=1, tokens=x)
    assert len(c._h_lens) == 1 and len(c._h_scores) == 1, 'метаданные не пишутся'
    assert len(c._kv_h) == 1, 'K/V-кольцо не заполнилось'
    assert c.retrieve(training=True) is None, 'retrieve(training=True) должен быть None'


def test_low_rank_shapes_and_params():
    cfg, m = _model(kv_dim=64)
    att = m.logit_cache.attention
    assert att.kv_dim == 64 and att.head_dim == 64 // 8
    assert att.k_proj_h.weight.shape == (64, 256), 'k_proj_h не low-rank'
    assert att.q_proj.weight.shape == (64, 256)
    assert att.out_proj.weight.shape == (256, 64)
    assert att.k_norm.weight.numel() == 64
    assert att.pos_enc.weight.shape[1] == 64
    n_att = sum(p.numel() for p in att.parameters())
    cfg2, m2 = _model(kv_dim=0)
    n_full = sum(p.numel() for p in m2.logit_cache.attention.parameters())
    assert n_att < n_full / 3, f'low-rank не сэкономил: {n_att} vs {n_full}'


def test_low_rank_forward_and_ring():
    cfg, m = _model(kv_dim=64)
    x = torch.randint(3, cfg.vocab, (1, 16))
    h = m.embed(x)
    out, *_ = m(h, None, step=1, tokens=x)
    assert out.shape == h.shape
    k, v = m.logit_cache.cache._kv_h[0]
    assert k.shape[-1] == 64 and v.shape[-1] == 64, 'в кольце не low-rank K/V'
    mb = m.logit_cache.cache.size_mb(training=True)
    # 1 окно × 16 токенов × 2 × 64 × 4B = 8KB
    assert mb < 0.05, f'size_mb завышен: {mb}'


def test_full_rank_default_unchanged():
    cfg, m = _model(kv_dim=0)
    att = m.logit_cache.attention
    assert att.kv_dim == 256 and att.head_dim == 32
    assert att.k_proj_h.weight.shape == (256, 256)
    x = torch.randint(3, cfg.vocab, (1, 16))
    h = m.embed(x)
    out, *_ = m(h, None, step=1, tokens=x)
    assert out.shape == h.shape


def test_eviction_with_metadata_only():
    cfg, m = _model()
    c = m.logit_cache.cache
    c.max_entries = 4
    x = torch.randint(3, cfg.vocab, (1, 8))
    h = m.embed(x)
    for s in range(1, 8):
        m(h, None, step=s, tokens=x)
    assert len(c._h_lens) <= 4 and len(c._kv_h) <= 4, \
        f'эвикция не работает: lens={len(c._h_lens)} kv={len(c._kv_h)}'
    assert len(c._kv_h) == len(c._h_lens), 'кольцо K/V рассинхронизировано'


def test_inference_store_evicts_with_horizon():
    """R2-блокер-лок: >=2 подряд store(training=False) при horizon>0 не падает
    (старый call-site _evict(self._logit_cache, ...) давал int+dict)."""
    from core.logit_cache import LogitCache
    c = LogitCache(50, 8, 64, n_scales=4, horizon_tokens=128)
    for i in range(4):
        c.store(torch.randn(1, 32, 50), training=False, novelty=1.0)
    assert len(c._l_lens) == len(c._l_scores) >= 1
    assert len(c) >= 1
    c.clear()
    assert not c._l_lens and not c._l_scores
