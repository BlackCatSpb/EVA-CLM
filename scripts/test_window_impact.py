"""
Test: Accuracy impact of 512 window vs full attention.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logit_cache_v2 import PerScaleLogitCache, PerScaleLogitAttention


def test_window_impact(n_tokens):
    D = 256
    V = 1000
    MAX_POS = 1024  # pos_enc limit

    attn = PerScaleLogitAttention(D, V, n_scales=4, n_heads=4)
    cache = PerScaleLogitCache(V, max_tokens=n_tokens + 10, n_scales=4)

    torch.manual_seed(42)
    for i in range(n_tokens):
        logits = torch.randn(1, 1, V)
        cache.store(logits)

    h = torch.randn(1, 1, D)

    # Full: retrieve ALL (capped by pos_enc)
    full_n = min(n_tokens, MAX_POS)
    original_retrieve = cache.retrieve_scale
    cache.retrieve_scale = lambda idx, n=None: original_retrieve(idx, n=full_n)
    with torch.no_grad():
        out_full, attn_full = attn(h, cache, return_attention=True)
    cache.retrieve_scale = original_retrieve

    # Window: 512
    win_n = min(512, n_tokens)
    with torch.no_grad():
        out_win, attn_win = attn(h, cache, return_attention=True)

    cos_sim = torch.nn.functional.cosine_similarity(
        out_full.view(-1).float(), out_win.view(-1).float(), dim=0
    ).item()

    top1 = (out_full.argmax(-1) == out_win.argmax(-1)).float().mean().item()
    top5 = (out_full.topk(5, -1).indices == out_win.topk(5, -1).indices).any(-1).float().mean().item()

    return full_n, win_n, cos_sim, top1, top5


print("=" * 75)
print("512 WINDOW vs FULL ATTENTION")
print("=" * 75)
print()
print(f"{'Tokens':<10} {'Full':<8} {'Win':<8} {'Cosine':<12} {'Top-1':<10} {'Top-5'}")
print("-" * 60)
for n in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    full, win, cos, t1, t5 = test_window_impact(n)
    marker = " <-- window starts here" if n > 512 else ""
    print(f"{n:<10,} {full:<8} {win:<8} {cos:<12.6f} {t1:<10.4f} {t5:.4f}{marker}")
