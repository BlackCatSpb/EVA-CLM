"""
Test: LogitCache and LogitAttention modules.
Verifies basic functionality before integration into EVAStack.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logit_cache import LogitCache, LogitAttention, LogitCacheAttention


def test_logit_cache():
    """Test LogitCache compression/decompression."""
    print("=" * 60)
    print("TEST: LogitCache")
    print("=" * 60)

    V = 75138  # vocab size
    cache = LogitCache(V, max_tokens=1000, n_layers=24)

    # Create dummy logits
    logits = torch.randn(1, 1, V)

    # Store
    cache.store(logits)
    print(f"Cache size: {len(cache)} entries, {cache.size_mb():.4f} MB")

    # Retrieve
    retrieved = cache.get_recent(1)
    print(f"Retrieved shape: {retrieved.shape}")

    # Check compression ratio
    orig_bytes = logits.numel() * 4
    comp_bytes = cache.size_bytes()
    ratio = orig_bytes / max(comp_bytes, 1)
    print(f"Original: {orig_bytes / 1024:.2f} KB")
    print(f"Compressed: {comp_bytes / 1024:.2f} KB")
    print(f"Ratio: {ratio:.1f}x")

    # Store multiple entries
    for i in range(10):
        cache.store(torch.randn(1, 1, V))
    print(f"\nAfter 10 entries:")
    print(f"Cache size: {len(cache)} entries, {cache.size_mb():.4f} MB")
    print(f"Ratio: {orig_bytes / max(cache.size_bytes() / 10, 1):.1f}x per entry")

    print("PASSED\n")
    return True


def test_logit_attention():
    """Test LogitAttention module."""
    print("=" * 60)
    print("TEST: LogitAttention")
    print("=" * 60)

    D = 2560  # hidden dim
    V = 75138  # vocab size
    n_heads = 8

    attention = LogitAttention(D, V, n_heads)
    cache = LogitCache(V, max_tokens=100, n_layers=24)

    # Fill cache with some entries
    for i in range(5):
        cache.store(torch.randn(1, 1, V))

    # Create dummy hidden state
    h = torch.randn(1, 10, D)  # (B, L, D)

    # Forward
    output, attn_weights = attention(h, cache, return_attention=True)

    print(f"Input shape: {h.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn_weights.shape}")
    print(f"Cache entries: {len(cache)}")

    # Verify shapes
    assert output.shape == h.shape, f"Output shape mismatch: {output.shape} != {h.shape}"
    assert attn_weights.shape[0] == h.shape[0], "Batch dim mismatch"
    assert attn_weights.shape[1] == h.shape[1], "Sequence dim mismatch"
    assert attn_weights.shape[2] == len(cache), "Cache dim mismatch"

    print("PASSED\n")
    return True


def test_logit_cache_attention():
    """Test LogitCacheAttention combined module."""
    print("=" * 60)
    print("TEST: LogitCacheAttention")
    print("=" * 60)

    D = 2560  # hidden dim
    V = 75138  # vocab size
    n_layers = 24
    n_heads = 8

    module = LogitCacheAttention(D, V, n_layers, max_tokens=1000, n_heads=n_heads)

    # Create dummy data
    h = torch.randn(1, 10, D)  # (B, L, D)
    logits = torch.randn(1, 10, V)  # (B, L, V)

    # Forward
    h_out, logits_out = module(h, logits, use_cache=True)

    print(f"Hidden input: {h.shape}")
    print(f"Hidden output: {h_out.shape}")
    print(f"Logits output: {logits_out.shape}")
    print(f"Cache size: {len(module.cache)} entries")
    print(f"Cache MB: {module.cache.size_mb():.4f} MB")

    # Verify shapes
    assert h_out.shape == h.shape, f"Hidden shape mismatch: {h_out.shape} != {h.shape}"
    assert logits_out.shape == logits.shape, f"Logits shape mismatch"

    # Test without cache
    h_out2, logits_out2 = module(h, logits, use_cache=False)
    assert torch.allclose(h_out2, h), "Without cache should return original h"
    assert torch.allclose(logits_out2, logits), "Without cache should return original logits"

    print("PASSED\n")
    return True


def test_cache_size_projection():
    """Project cache size for 1M tokens."""
    print("=" * 60)
    print("CACHE SIZE PROJECTION")
    print("=" * 60)

    V = 75138
    cache = LogitCache(V, max_tokens=1_000_000)

    # Simulate storing entries
    logits = torch.randn(1, 1, V)

    # Store 100 entries to estimate
    for i in range(100):
        cache.store(logits)

    avg_bytes = cache.size_bytes() / 100
    avg_kb = avg_bytes / 1024
    avg_mb = avg_bytes / (1024 * 1024)

    print(f"Average entry size: {avg_bytes:.0f} bytes ({avg_kb:.2f} KB)")
    print(f"\nProjected sizes:")
    print(f"  1K tokens:   {avg_mb * 1000:.2f} MB")
    print(f"  10K tokens:  {avg_mb * 10000:.2f} MB")
    print(f"  100K tokens: {avg_mb * 100000:.2f} MB")
    print(f"  1M tokens:   {avg_mb * 1000000:.2f} MB")

    # Compare with KV-cache
    D = 2560
    n_layers = 24
    kv_per_token = 2 * n_layers * D * 2  # K+V, fp16
    print(f"\nKV-cache comparison (D={D}, layers={n_layers}):")
    print(f"  KV per token: {kv_per_token} bytes ({kv_per_token / 1024:.2f} KB)")
    print(f"  1M tokens:    {kv_per_token * 1000000 / (1024*1024):.2f} GB")
    print(f"  Compression ratio: {kv_per_token / avg_bytes:.0f}x")

    print("\nPASSED\n")
    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("LOGIT CACHE MODULE TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_logit_cache,
        test_logit_attention,
        test_logit_cache_attention,
        test_cache_size_projection,
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
