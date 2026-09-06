"""
Test: EVAStack integration with PerScaleCacheAttention.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.serialization import add_safe_globals
from core import EVAConfig, EVAStack
from core.config import WideBindConfig
from core.logit_cache_v2 import PerScaleCacheAttention
add_safe_globals([EVAConfig, WideBindConfig])


def load_model():
    ckpt = torch.load(r"C:\Users\black\OneDrive\Desktop\EVA CLM\checkponts\best 19.pt",
                      map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    model = EVAStack(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().float()
    return model, cfg


def test_forward_with_vsa_cache():
    """Test forward with VSA-driven cache."""
    print("=" * 60)
    print("TEST: Forward with VSA-driven cache")
    print("=" * 60)

    model, cfg = load_model()

    # Enable VSA-driven cache
    model.logit_cache = PerScaleCacheAttention(
        D=cfg.D, V=cfg.vocab, n_layers=cfg.n_layers,
        max_tokens=1000, n_heads=8,
    )

    # Forward
    tokens = torch.randint(0, cfg.vocab, (1, 10))
    h = model.embed_tokens(tokens)

    with torch.no_grad():
        out, state, _, _ = model(h, None, adaptive=False, step=0, tokens=tokens)

    logits = model.lm_head(out)
    h_aug, logits_out = model.process_with_cache(out, logits, use_cache=True)

    print(f"Input: {tokens.shape}")
    print(f"Hidden: {out.shape}")
    print(f"Augmented: {h_aug.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Cache entries: {len(model.logit_cache.cache)}")
    print(f"Cache size: {model.logit_cache.cache.size_mb():.4f} MB")

    # VSA scales
    vsa_scales = model.logit_cache.cache.vsa_scales
    print(f"VSA scales: {vsa_scales.data.tolist()}")
    print(f"K values: {[model.logit_cache.cache.get_k(i) for i in range(4)]}")

    # Per-scale breakdown
    for i in range(4):
        k = model.logit_cache.cache.get_k(i)
        size = model.logit_cache.cache.size_bytes_per_scale(i)
        print(f"  Scale {i}: k={k}, size={size} bytes")

    if h_aug.shape == out.shape and not torch.isnan(h_aug).any():
        print("PASSED\n")
        return True
    else:
        print("FAILED\n")
        return False


def test_generation_with_vsa_cache():
    """Test generation with VSA-driven cache."""
    print("=" * 60)
    print("TEST: Generation with VSA-driven cache")
    print("=" * 60)

    model, cfg = load_model()

    # Enable VSA-driven cache
    model.logit_cache = PerScaleCacheAttention(
        D=cfg.D, V=cfg.vocab, n_layers=cfg.n_layers,
        max_tokens=1000, n_heads=8,
    )

    # Generation loop
    tokens = [torch.randint(0, cfg.vocab, (1,)).item()]
    state = None

    for step in range(10):
        ctx = torch.tensor([tokens[-128:]], dtype=torch.long)
        h = model.embed_tokens(ctx)

        with torch.no_grad():
            out, state, _, _ = model(h, state, adaptive=False, step=step, tokens=ctx)

        logits = model.lm_head(out)
        h_aug, logits_out = model.process_with_cache(out, logits, use_cache=True)

        next_token = logits_out[:, -1, :].argmax(dim=-1).item()
        tokens.append(next_token)

    print(f"Generated {len(tokens)} tokens")
    print(f"Cache size: {model.logit_cache.cache.size_mb():.4f} MB")
    print(f"VSA scales: {model.logit_cache.cache.vsa_scales.data.tolist()}")
    print(f"K values: {[model.logit_cache.cache.get_k(i) for i in range(4)]}")

    if not any(torch.isnan(h_aug).any() for _ in [1]):
        print("PASSED\n")
        return True
    else:
        print("FAILED\n")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("EVASTACK + PER-SCALE CACHE TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_forward_with_vsa_cache,
        test_generation_with_vsa_cache,
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
