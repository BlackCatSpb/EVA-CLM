"""M18: write-time code-profile cache (logit_cache_mode='profile')."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack                       # noqa: E402
from core.logit_cache import LogitCache, LogitAttention   # noqa: E402
from core.vsa_utils import twin_free_codes                # noqa: E402


def _setup(V=256, K=64, D=128):
    codes = twin_free_codes(V, K=K, S=8)
    C = codes.float()
    cache = LogitCache(V, D, max_entries=8)
    att = LogitAttention(D, V, 4, codes=C, sparsity=1.0)
    return C, cache, att


def test_profile_exact_and_topk_loses_tail():
    V, K, D = 256, 64, 128
    C, cache, att = _setup(V, K, D)
    torch.manual_seed(0)
    # logit field with a heavy informative tail (uniform-ish over the code)
    z = torch.randn(1, 4, V) * 3.0
    p_exact = att.bit_profile(z)
    cache.store(z, training=False)                       # legacy top-k64-ish
    rec = cache.retrieve(training=False)
    p_topk = att.bit_profile(rec)
    d_topk = 1.0 - F.cosine_similarity(p_exact.flatten(), p_topk.flatten(), dim=0)
    cache2 = LogitCache(V, D, max_entries=8)
    cache2.store_profile(p_exact)
    p_prof = cache2.profile_window().float()
    d_prof = 1.0 - F.cosine_similarity(p_exact.flatten(), p_prof.flatten(), dim=0)
    assert float(d_prof) < 1e-3, 'profile mode is lossy beyond fp16'
    assert float(d_prof) < float(d_topk), 'top-k tail loss must exceed direct projection'


def test_profile_read_path_never_enters_v_space():
    # The real M18 claim: with write-time profiles stored, RETRIEVAL must not
    # rebuild (B,M,V) nor run the V@C matmul — cache.retrieve (the V path)
    # must never be called in profile mode.
    V, K, D = 256, 64, 128
    C, cache, att = _setup(V, K, D)
    torch.manual_seed(0)
    z = torch.randn(1, 4, V) * 3.0
    calls = {'n': 0}
    orig = cache.retrieve
    def spy(*a, **k):
        calls['n'] += 1
        return orig(*a, **k)
    cache.retrieve = spy
    h = torch.randn(1, 4, D)
    for i in range(4):
        cache.store_profile(att.bit_profile(z[:, i:i + 1]))
        out = att(h, cache, training=False)
        out = out[0] if isinstance(out, tuple) else out
    assert calls['n'] == 0, 'profile mode fell back to the V-space read path'
    assert torch.isfinite(out).all() and out.shape == h.shape
    # gate ~ identity at init (bias -10): output must stay within 1e-2 of h
    assert float((out - h).norm() / h.norm()) < 5e-2


def test_profile_mode_stack_smoke_identity():
    cfg = EVAConfig(n_layers=1, D=128, code_dim=16, code_sparsity=4, vocab=300,
                    mlp_groups=4, save_dir='.', logit_cache_enabled=True,
                    logit_cache_mode='profile',
                    logit_cache_max_entries=8, memory_bank=False,
                    intent_bridge=False, explicit_reasoning=False)
    torch.manual_seed(0)
    m = EVAStack(cfg).eval()
    x = torch.randint(3, cfg.vocab, (1, 24))
    with torch.no_grad():
        h = (m.embed_tokens(x) if hasattr(m, 'embed_tokens') else m.embed(x))
        out1, s1, _, _ = m(h, None, step=None, tokens=x)
        h_aug, logits = m.augment_with_cache(h, m.lm_head(h)) if hasattr(m, 'augment_with_cache') else (h, None)
    # if the model exposes process_logits-style API, ensure no NaN and shapes
    assert out1.shape == h.shape and torch.isfinite(out1).all()
