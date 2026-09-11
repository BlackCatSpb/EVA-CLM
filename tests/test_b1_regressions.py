# -*- coding: utf-8 -*-
"""B1 regression locks — every test pins a fix from the agent-audit batch 1.
Run: python -m pytest tests/test_b1_regressions.py -q"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F
from core.config import EVAConfig
from core.stack import EVAStack
from core.vsa_utils import sparse_block_codes


def _mini(**kw):
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=600, save_dir='.', **kw)
    torch.manual_seed(0)
    return EVAStack(cfg).train()


def test_b1_state_actually_carries():
    m = _mini(logit_cache_enabled=False, memory_bank=False)
    seen = []
    import core.block as BD
    orig = BD.EVABlock.forward

    def spy(self, h, state=None, *a, **k):
        seen.append(state is not None and any(s is not None for s in state))
        return orig(self, h, state, *a, **k)
    BD.EVABlock.forward = spy
    try:
        x = torch.randint(1, 600, (1, 16))
        h = m.embed_tokens(x)
        _, st, _, _ = m(h.clone(), None, step=5, tokens=x)
        seen.clear()
        m(h.clone(), st, step=6, tokens=x)
    finally:
        BD.EVABlock.forward = orig
    assert seen and all(seen), 'carried state still discarded (B1.1 regression)'


def test_b1_short_seq_streaming_conv():
    m = _mini(logit_cache_enabled=False, memory_bank=False)
    x = torch.randint(1, 600, (1, 8))
    h = m.embed_tokens(x)
    _, st, _, _ = m(h.clone(), None, step=5, tokens=x)
    for t in range(6):
        out, st, _, _ = m(h.clone(), st, step=6 + t, tokens=x)
    assert torch.isfinite(out).all(), 'streaming conv broke on L<kernel carry (B1)'


def test_b1_reset_cache_recomputes_structural_and_scrubs_holders():
    m = _mini()
    x = torch.randint(1, 600, (1, 16))
    m(m.embed_tokens(x), None, step=5, tokens=x)
    tc = m.tau_config
    with torch.no_grad():
        tc._log_tau_min.fill_(float('nan'))
        tc._log_tau_range.fill_(float('nan'))
        m.layers[0].mirror._cached_hp = torch.full((1, 3, 1), float('nan'))
    m.reset_cache()
    assert torch.isfinite(tc._log_tau_min).all() and torch.isfinite(tc._log_tau_range).all()
    assert abs(float(tc._log_tau_min) - math.log(tc.tau_min)) < 1e-5
    assert m.layers[0].mirror._cached_hp is None
    h = m.embed_tokens(x)
    o, *_ = m(h.clone(), None, step=9, tokens=x)          # the old scrub crashed the NEXT forward
    assert torch.isfinite(o).all()


def test_b1_reset_clears_logit_cache_lists():
    m = _mini()
    if m.logit_cache is not None:
        x = torch.randint(1, 600, (1, 16))
        m(m.embed_tokens(x), None, step=5, tokens=x)
        assert len(m.logit_cache.cache) > 0
        m.reset_cache()
        assert len(m.logit_cache.cache) == 0


def test_b1_snapshot_covers_nonbuffer_streaming_attrs():
    m = _mini()
    x = torch.randint(1, 600, (1, 16))
    m(m.embed_tokens(x), None, step=5, tokens=x)
    snap = m.snapshot_runtime_buffers()
    assert '__attrs__' in snap
    bus0 = getattr(m, '_last_bus', None)
    if isinstance(bus0, torch.Tensor):
        key = bus0.clone()
        m(m.embed_tokens(x), None, step=6, tokens=x)      # mutates
        m.restore_runtime_buffers(snap)
        assert torch.equal(m._last_bus, key), 'B1: _last_bus leaked past eval restore'


def test_b1_ucl_shape_guard_and_bounded_injection_and_identity_init():
    m = _mini(memory_bank=False)
    x = torch.randint(1, 600, (1, 8))
    m(m.embed_tokens(x), None, step=5, tokens=x)          # caches hp/pen at (1,8)
    m.eval()
    with torch.no_grad():
        x2 = torch.randint(1, 600, (1, 24))               # eval with different shape
        o, *_ = m(m.embed_tokens(x2), None, adaptive=False, tokens=x2)
    assert torch.isfinite(o).all(), 'UCL consumed stale-shape cache (B1.3)'
    m.train()
    cl = m.concept_layer
    assert float(torch.sigmoid(cl.read_scale.detach())) < 0.03, 'UCL not identity-at-init'
    h = torch.randn(1, 8, 256)
    hp = torch.randn(1, 8, m.layers[0].mirror.G, m.layers[0].mirror.k)
    with torch.no_grad():
        out = cl(h, hp=hp, pen=torch.rand(1, 8) + 0.5, resvar=0.2, mat_gate=1.0,
                 allow_write=False, gate=None, tau_norm=0.5)
    ratio = float(out.norm(dim=-1).max() / (h.norm(dim=-1).max() + 1e-9))
    assert ratio <= 0.26, f'UCL injection amplitude unbounded: {ratio:.2f}×‖h‖'


def test_b1_cascade_finite_at_init():
    from core.bind import BottleneckBind
    cfg = EVAConfig(bind_twist_mode='cascade', code_dim=16, code_sparsity=4)
    b = BottleneckBind(D=64, K=16, cfg=cfg)
    o = b(torch.randn(1, 8, 64))
    assert torch.isfinite(o).all(), 'cascade mode still NaN at init (B1.5)'


def test_b1_codes_guard_cache_and_nonpersistent():
    import pytest
    with pytest.raises(ValueError):
        sparse_block_codes(2_000_000, 32, 6)              # was raw IndexError
    a = sparse_block_codes(500, 16, 4)
    b = sparse_block_codes(500, 16, 4)
    assert torch.equal(a, b)
    m = _mini()
    sd = m.state_dict()
    assert not any(k.endswith('.codes') or k == 'codes' for k in sd), 'codes still persisted'


def test_b1_cache_inference_path_alive_and_discriminative():
    from core.logit_cache import LogitCacheAttention
    torch.manual_seed(0)
    codes = sparse_block_codes(64, 16, 4)
    a = LogitCacheAttention(D=64, V=64, n_layers=2, max_entries=4, n_heads=4,
                            codes=codes, sparsity=4)
    lg1 = torch.randn(1, 8, 64)
    lg2 = torch.randn(1, 8, 64)
    a.eval()
    with torch.no_grad():
        a.cache.store(lg1, training=False)
        h = torch.randn(1, 8, 64)
        out, _ = a(h, lg2, training=False)
    assert torch.isfinite(out).all(), 'inference-mode cache still crashes (B1.6)'
    p1, p2 = a.attention.bit_profile(lg1), a.attention.bit_profile(lg2)
    cos = float(F.cosine_similarity(p1.flatten(), p2.flatten(), dim=0))
    assert cos < 0.999, 'profiles collapsed — fill/disc riminability regression'
    rec = a.cache.retrieve(n=1, training=False)
    assert torch.isfinite(rec).all(), '-inf fill leaked into reconstruction'
    k0, k4 = a.cache.get_k(0), None
    with torch.no_grad():
        a.cache.vsa_scales.fill_(4.0)
        hi = a.cache.get_k(0)
        a.cache.vsa_scales.fill_(-4.0)
        lo = a.cache.get_k(0)
    assert hi > lo and lo >= 8, 'vsa scales not driving k (B1)'


def test_b1_hp_buf_grows_with_sequence():
    m = _mini()
    x8 = torch.randint(1, 600, (1, 8))
    m(m.embed_tokens(x8), None, step=5, tokens=x8)
    x64 = torch.randint(1, 600, (1, 64))                  # curriculum growth beyond seq_len
    o, *_ = m(m.embed_tokens(x64), None, step=6, tokens=x64)
    assert torch.isfinite(o).all() and o.shape[1] == 64


def test_b1_build_without_intent_bridge():
    m = _mini(intent_bridge=False)
    x = torch.randint(1, 600, (1, 16))
    o, *_ = m(m.embed_tokens(x), None, step=5, tokens=x)
    assert torch.isfinite(o).all()


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as e:
            fails += 1
            print(f'FAIL {fn.__name__}: {repr(e)[:200]}')
    print(f'\n{len(fns)-fails}/{len(fns)} passed')
    sys.exit(1 if fails else 0)
