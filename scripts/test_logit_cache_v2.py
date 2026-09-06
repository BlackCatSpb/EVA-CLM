"""
Test: Per-scale LogitCache with 4 VSA scales.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logit_cache_v2 import PerScaleLogitCache, PerScaleLogitAttention, PerScaleCacheAttention


def test_per_scale_cache():
    """Test per-scale compression."""
    print("=" * 60)
    print("TEST: Per-scale compression")
    print("=" * 60)

    V = 65536  # Real vocab size
    cache = PerScaleLogitCache(V, max_tokens=1000, n_scales=4)

    # Create dummy logits
    logits = torch.randn(1, 1, V)

    # Store
    cache.store(logits)

    print(f"Cache entries: {len(cache)}")
    print(f"Cache size: {cache.size_mb():.4f} MB")
    print(f"Cache size bytes: {cache.size_bytes()}")

    # Per-scale sizes
    for i in range(4):
        scale_size = cache.size_bytes_per_scale(i)
        k = cache._get_k(i)
        print(f"  Scale {i} (k={k}): {scale_size} bytes ({scale_size/1024:.2f} KB)")

    # Original size
    orig_bytes = logits.numel() * 4
    print(f"\nOriginal: {orig_bytes / 1024:.2f} KB")
    print(f"Total compressed: {cache.size_bytes() / 1024:.2f} KB")
    print(f"Ratio: {orig_bytes / max(cache.size_bytes(), 1):.1f}x")

    print("PASSED\n")
    return True


def test_per_scale_attention():
    """Test per-scale attention."""
    print("=" * 60)
    print("TEST: Per-scale attention")
    print("=" * 60)

    D = 256
    V = 1000
    n_scales = 4
    n_heads = 4

    attention = PerScaleLogitAttention(D, V, n_scales, n_heads)
    cache = PerScaleLogitCache(V, max_tokens=100, n_scales=n_scales)

    # Fill cache
    for i in range(5):
        cache.store(torch.randn(1, 1, V))

    print(f"Cache has {len(cache)} entries")

    # Process token
    h = torch.randn(1, 1, D)
    output, attn_weights = attention(h, cache, return_attention=True)

    print(f"Input shape: {h.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention is dict: {isinstance(attn_weights, dict)}")

    # Check output
    if output.shape == h.shape and not torch.isnan(output).any():
        print("PASSED\n")
        return True
    else:
        print("FAILED\n")
        return False


def test_per_scale_cache_attention():
    """Test combined module."""
    print("=" * 60)
    print("TEST: PerScaleCacheAttention")
    print("=" * 60)

    D = 256
    V = 1000
    n_layers = 24
    n_heads = 4

    module = PerScaleCacheAttention(D, V, n_layers, max_tokens=100, n_heads=n_heads)

    # Process 10 tokens
    for i in range(10):
        h = torch.randn(1, 1, D) * 0.1
        logits = torch.randn(1, 1, V) * 0.1
        h_out, logits_out = module(h, logits, use_cache=True)

    print(f"Cache entries: {len(module.cache)}")
    print(f"Cache size: {module.cache.size_mb():.4f} MB")

    # Per-scale breakdown
    for i in range(4):
        scale_size = module.cache.size_bytes_per_scale(i)
        k = module.cache._get_k(i)
        print(f"  Scale {i} (k={k}): {scale_size} bytes")

    # Check for NaN
    if not torch.isnan(h_out).any():
        print("PASSED\n")
        return True
    else:
        print("FAILED: NaN in output\n")
        return False


def test_size_projection():
    """Project size for 1M tokens."""
    print("=" * 60)
    print("SIZE PROJECTION")
    print("=" * 60)

    V = 65536
    cache = PerScaleLogitCache(V, max_tokens=1000, n_scales=4)

    # Store 100 entries to estimate
    for i in range(100):
        cache.store(torch.randn(1, 1, V))

    per_token = cache.size_bytes() / 100

    print(f"Per-token size: {per_token:.0f} bytes ({per_token/1024:.2f} KB)")
    print(f"\nProjected sizes:")
    for n in [10_000, 100_000, 1_000_000]:
        mb = per_token * n / (1024 * 1024)
        print(f"  {n:>10,} tokens: {mb:>8.1f} MB")

    # Compare with KV-cache
    D = 2560
    n_layers = 24
    kv_per_token = 2 * n_layers * D * 2  # K+V, fp16
    print(f"\nKV-cache comparison:")
    print(f"  KV per token: {kv_per_token} bytes ({kv_per_token / 1024:.2f} KB)")
    print(f"  1M tokens:    {kv_per_token * 1_000_000 / (1024*1024):.2f} GB")
    print(f"  Compression ratio: {kv_per_token / per_token:.0f}x")

    print("\nPASSED\n")
    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("PER-SCALE LOGITCACHE TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_per_scale_cache,
        test_per_scale_attention,
        test_per_scale_cache_attention,
        test_size_projection,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 60)
