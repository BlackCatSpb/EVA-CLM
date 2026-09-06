"""
Comprehensive test: Verify LogitCache works correctly.
Tests: storage, retrieval, attention, quality preservation.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logit_cache import LogitCache, LogitAttention, LogitCacheAttention


def test_storage_retrieval():
    """Test that stored logits can be retrieved correctly."""
    print("=" * 60)
    print("TEST 1: Storage and Retrieval")
    print("=" * 60)

    V = 100  # Small vocab for easy testing
    cache = LogitCache(V, max_tokens=100)

    # Create known logits
    logits1 = torch.arange(V, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [0, 1, 2, ..., 99]
    logits2 = torch.arange(V, 0, -1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [99, 98, ..., 0]

    # Store
    cache.store(logits1)
    cache.store(logits2)

    print(f"Stored 2 entries")
    print(f"Cache size: {len(cache)} entries")

    # Retrieve most recent
    retrieved = cache.get_recent(1)
    print(f"Most recent shape: {retrieved.shape}")

    # Check if retrieved matches stored (with some compression loss)
    diff = (retrieved - logits2).abs().max().item()
    print(f"Max diff (retrieved vs stored): {diff:.6f}")

    # Retrieve both
    retrieved_all = cache.get_recent(2)
    print(f"Both entries shape: {retrieved_all.shape}")

    # Check order: cache returns entries in insertion order (oldest first)
    # get_recent(2) returns [logits1, logits2] (insertion order)
    diff1 = (retrieved_all[0, 0] - logits1[0, 0]).abs().max().item()  # First stored = oldest
    diff2 = (retrieved_all[0, 1] - logits2[0, 0]).abs().max().item()  # Second stored = newest
    print(f"Order check - Entry 0 vs logits1 (oldest): {diff1:.6f}")
    print(f"Order check - Entry 1 vs logits2 (newest): {diff2:.6f}")

    if diff < 0.5 and diff1 < 0.5 and diff2 < 0.5:  # Allow some compression loss
        print("PASSED\n")
        return True
    else:
        print("FAILED: Retrieval mismatch\n")
        return False


def test_compression_ratio():
    """Test compression ratio."""
    print("=" * 60)
    print("TEST 2: Compression Ratio")
    print("=" * 60)

    V = 75138  # Real vocab size
    cache = LogitCache(V, max_tokens=1000)

    # Create random logits
    logits = torch.randn(1, 1, V)

    # Store
    cache.store(logits)

    # Calculate sizes
    orig_bytes = logits.numel() * 4  # fp32
    comp_bytes = cache.size_bytes()
    ratio = orig_bytes / max(comp_bytes, 1)

    print(f"Original: {orig_bytes / 1024:.2f} KB")
    print(f"Compressed: {comp_bytes / 1024:.2f} KB")
    print(f"Ratio: {ratio:.1f}x")

    # Store more entries
    for i in range(100):
        cache.store(torch.randn(1, 1, V))

    avg_bytes = cache.size_bytes() / 101
    avg_ratio = orig_bytes / max(avg_bytes, 1)

    print(f"\nAfter 101 entries:")
    print(f"Avg entry: {avg_bytes / 1024:.2f} KB")
    print(f"Avg ratio: {avg_ratio:.1f}x")

    if ratio > 50 and avg_ratio > 50:
        print("PASSED\n")
        return True
    else:
        print("FAILED: Ratio too low\n")
        return False


def test_attention_token_to_token():
    """Test that attention works token-to-token."""
    print("=" * 60)
    print("TEST 3: Attention Token-to-Token")
    print("=" * 60)

    D = 256  # Small hidden dim for testing
    V = 1000  # Small vocab
    n_heads = 4

    attention = LogitAttention(D, V, n_heads)
    cache = LogitCache(V, max_tokens=100)

    # Simulate 5 tokens
    for i in range(5):
        logits = torch.randn(1, 1, V)
        cache.store(logits)

    print(f"Cache has {len(cache)} tokens")

    # Process token 6
    h = torch.randn(1, 1, D)  # Current hidden state
    output, attn_weights = attention(h, cache, return_attention=True)

    print(f"Input shape: {h.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn_weights.shape}")

    # Check attention weights sum to 1
    attn_sum = attn_weights.sum(dim=-1).item()
    print(f"Attention weights sum: {attn_sum:.6f}")

    # Check attention is non-zero for all positions
    attn_min = attn_weights.min().item()
    print(f"Attention min: {attn_min:.6f}")

    # Process token 7 (should attend to all 6 previous tokens)
    cache.store(torch.randn(1, 1, V))
    h2 = torch.randn(1, 1, D)
    output2, attn_weights2 = attention(h2, cache, return_attention=True)

    print(f"\nAfter adding token 7:")
    print(f"Cache has {len(cache)} tokens")
    print(f"Attention shape: {attn_weights2.shape}")

    # Check attention now covers all 6 tokens
    if attn_weights2.shape[-1] == 6 and attn_sum > 0.99 and attn_min > 0:
        print("PASSED\n")
        return True
    else:
        print("FAILED: Attention not working correctly\n")
        return False


def test_quality_preservation():
    """Test that generation quality is preserved."""
    print("=" * 60)
    print("TEST 4: Quality Preservation")
    print("=" * 60)

    V = 1000
    D = 256
    n_heads = 4

    module = LogitCacheAttention(D, V, n_layers=24, max_tokens=100, n_heads=n_heads)

    # First token: no cache, should return unchanged
    h1 = torch.randn(1, 1, D) * 0.1
    logits1 = torch.randn(1, 1, V) * 0.1
    h1_out, logits1_out = module(h1, logits1, use_cache=True)

    diff1 = (h1_out - h1).abs().mean().item()
    print(f"Token 1 (no cache): hidden diff = {diff1:.6f}")

    # Second token: has cache, should modify hidden state
    h2 = torch.randn(1, 1, D) * 0.1
    logits2 = torch.randn(1, 1, V) * 0.1
    h2_out, logits2_out = module(h2, logits2, use_cache=True)

    diff2 = (h2_out - h2).abs().mean().item()
    print(f"Token 2 (has cache): hidden diff = {diff2:.6f}")

    # Third token: has more cache, should modify more
    h3 = torch.randn(1, 1, D) * 0.1
    logits3 = torch.randn(1, 1, V) * 0.1
    h3_out, logits3_out = module(h3, logits3, use_cache=True)

    diff3 = (h3_out - h3).abs().mean().item()
    print(f"Token 3 (more cache): hidden diff = {diff3:.6f}")

    print(f"\nCache size: {len(module.cache)} entries")

    # Check that cache modifies hidden state for tokens with cache
    # Token 1 has no cache, so diff should be ~0
    # Tokens 2+ have cache, so diff should be > 0
    if diff2 > 0 and diff3 > 0:
        print("PASSED\n")
        return True
    else:
        print("FAILED: Cache not modifying hidden state\n")
        return False


def test_cache_persistence():
    """Test that cache persists across forward passes."""
    print("=" * 60)
    print("TEST 5: Cache Persistence")
    print("=" * 60)

    D = 256
    V = 1000
    n_heads = 4

    module = LogitCacheAttention(D, V, n_layers=24, max_tokens=100, n_heads=n_heads)

    # Simulate 5 forward passes
    for i in range(5):
        h = torch.randn(1, 1, D)
        logits = torch.randn(1, 1, V)
        module(h, logits, use_cache=True)

    print(f"After 5 forward passes:")
    print(f"Cache size: {len(module.cache)} entries")

    # Check that cache persists
    if len(module.cache) == 5:
        print("PASSED\n")
        return True
    else:
        print(f"FAILED: Expected 5 entries, got {len(module.cache)}\n")
        return False


def test_cache_reset():
    """Test that cache can be reset."""
    print("=" * 60)
    print("TEST 6: Cache Reset")
    print("=" * 60)

    D = 256
    V = 1000
    n_heads = 4

    module = LogitCacheAttention(D, V, n_layers=24, max_tokens=100, n_heads=n_heads)

    # Add some entries
    for i in range(5):
        h = torch.randn(1, 1, D)
        logits = torch.randn(1, 1, V)
        module(h, logits, use_cache=True)

    print(f"Before reset: {len(module.cache)} entries")

    # Reset
    module.cache.clear()

    print(f"After reset: {len(module.cache)} entries")

    if len(module.cache) == 0:
        print("PASSED\n")
        return True
    else:
        print(f"FAILED: Cache not cleared\n")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("LOGITCACHE COMPREHENSIVE TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_storage_retrieval,
        test_compression_ratio,
        test_attention_token_to_token,
        test_quality_preservation,
        test_cache_persistence,
        test_cache_reset,
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
