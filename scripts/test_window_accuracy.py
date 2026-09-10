"""
Test: Accuracy comparison — 512 window vs full attention.
"""
import sys, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logit_cache_v2 import PerScaleLogitCache, PerScaleLogitAttention


def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    a_flat = a.view(-1).float()
    b_flat = b.view(-1).float()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-8)).item()


def top1_accuracy(original_logits, recovered_logits):
    """Top-1 prediction accuracy."""
    orig_top1 = original_logits.argmax(dim=-1)
    recov_top1 = recovered_logits.argmax(dim=-1)
    return (orig_top1 == recov_top1).float().mean().item()


def top5_accuracy(original_logits, recovered_logits):
    """Top-5 prediction accuracy."""
    orig_top5 = original_logits.topk(5, dim=-1).indices
    recov_top5 = recovered_logits.topk(5, dim=-1).indices
    matches = (orig_top5 == recov_top5).any(dim=-1)
    return matches.float().mean().item()


def test_window_vs_full(n_tokens):
    """Compare window=512 vs full attention."""
    D = 256
    V = 1000

    cache_full = PerScaleLogitCache(V, max_tokens=n_tokens + 10, n_scales=4)
    cache_win = PerScaleLogitCache(V, max_tokens=n_tokens + 10, n_scales=4)

    attn_full = PerScaleLogitAttention(D, V, n_scales=4, n_heads=4)
    attn_win = PerScaleLogitAttention(D, V, n_scales=4, n_heads=4)

    # Store same tokens in both caches
    torch.manual_seed(42)
    stored_logits = []
    for i in range(n_tokens):
        logits = torch.randn(1, 1, V)
        stored_logits.append(logits.clone())
        cache_full.store(logits.clone())
        cache_win.store(logits.clone())

    # Retrieve last logits for comparison
    last_logits = stored_logits[-1]

    # Test 1: Retrieve quality (decompress accuracy)
    full_retrieved = cache_full.retrieve_scale(0, n=n_tokens)
    win_retrieved = cache_win.retrieve_scale(0, n=min(n_tokens, 512))

    # Cosine similarity
    full_cos = cosine_sim(last_logits, full_retrieved[:, -1:, :])
    win_cos = cosine_sim(last_logits, win_retrieved[:, -1:, :])

    # Test 2: Attention quality (output similarity)
    h = torch.randn(1, 1, D)
    torch.manual_seed(123)

    with torch.no_grad():
        out_full, attn_full_w = attn_full(h, cache_full, return_attention=True)
        torch.manual_seed(123)
        out_win, attn_win_w = attn_win(h, cache_win, return_attention=True)

    output_cos = cosine_sim(out_full, out_win)

    # Test 3: Top-1 accuracy
    top1 = top1_accuracy(out_full, out_win)
    top5 = top5_accuracy(out_full, out_win)

    return {
        'n': n_tokens,
        'full_cos': full_cos,
        'win_cos': win_cos,
        'output_cos': output_cos,
        'top1': top1,
        'top5': top5,
        'full_attn_tokens': attn_full_w[0].shape[-1] if attn_full_w[0] is not None else 0,
        'win_attn_tokens': attn_win_w[0].shape[-1] if attn_win_w[0] is not None else 0,
    }


print("=" * 80)
print("ACCURACY COMPARISON: 512 WINDOW vs FULL ATTENTION")
print("=" * 80)
print()

results = []
for n in [100, 500, 1000, 5000, 10000, 50000, 100000]:
    print(f"Testing {n:,} tokens...")
    r = test_window_vs_full(n)
    results.append(r)
    print(f"  Full attention: {r['full_attn_tokens']} tokens")
    print(f"  Window attention: {r['win_attn_tokens']} tokens")
    print(f"  Cosine sim (retrieve): {r['full_cos']:.6f} (full) vs {r['win_cos']:.6f} (window)")
    print(f"  Cosine sim (output): {r['output_cos']:.6f}")
    print(f"  Top-1 accuracy: {r['top1']:.4f}")
    print(f"  Top-5 accuracy: {r['top5']:.4f}")
    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"{'Tokens':<12} {'Full Attn':<12} {'Win Attn':<12} {'Cos Sim':<12} {'Top-1':<10} {'Top-5'}")
print("-" * 70)
for r in results:
    print(f"{r['n']:<12,} {r['full_attn_tokens']:<12} {r['win_attn_tokens']:<12} {r['output_cos']:<12.6f} {r['top1']:<10.4f} {r['top5']:.4f}")

print()
print("INTERPRETATION:")
print("  - Cosine similarity: >0.99 = negligible difference")
print("  - Top-1 accuracy: >0.99 = predictions match")
print("  - Top-5 accuracy: >0.99 = top choices match")
