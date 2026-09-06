"""
Test: EVAStack integration with LogitCache.
Verifies that the model works with cache enabled/disabled.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.serialization import add_safe_globals
from core import EVAConfig, EVAStack
from core.config import WideBindConfig
from core.logit_cache import LogitCacheAttention
add_safe_globals([EVAConfig, WideBindConfig])


def load_model():
    ckpt_path = r"C:\Users\black\OneDrive\Desktop\EVA CLM\checkponts\best 19.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    model = EVAStack(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().float()
    return model, cfg


def test_forward_without_cache():
    """Test forward pass without cache (baseline)."""
    print("=" * 60)
    print("TEST: Forward without cache (baseline)")
    print("=" * 60)

    model, cfg = load_model()

    # Disable cache
    model.logit_cache = None

    # Forward pass
    tokens = torch.randint(0, cfg.vocab, (1, 10))
    h = model.embed_tokens(tokens)

    with torch.no_grad():
        out, state, _, _ = model(h, None, adaptive=False, step=0, tokens=tokens)

    # Get logits
    logits = model.lm_head(out)

    print(f"Input shape: {tokens.shape}")
    print(f"Hidden shape: {out.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Cache size: {model.cache_size_mb():.4f} MB")

    print("PASSED\n")
    return True


def test_forward_with_cache():
    """Test forward pass with cache enabled."""
    print("=" * 60)
    print("TEST: Forward with cache enabled")
    print("=" * 60)

    model, cfg = load_model()

    # Enable cache
    model.logit_cache = LogitCacheAttention(
        D=cfg.D,
        V=cfg.vocab,
        n_layers=cfg.n_layers,
        max_tokens=1000,
        n_heads=8,
    )

    # Forward pass
    tokens = torch.randint(0, cfg.vocab, (1, 10))
    h = model.embed_tokens(tokens)

    with torch.no_grad():
        out, state, _, _ = model(h, None, adaptive=False, step=0, tokens=tokens)

    # Get logits
    logits = model.lm_head(out)

    # Process with cache
    h_augmented, logits_out = model.process_with_cache(out, logits, use_cache=True)

    print(f"Input shape: {tokens.shape}")
    print(f"Hidden shape: {out.shape}")
    print(f"Augmented shape: {h_augmented.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Cache size: {model.cache_size_mb():.4f} MB")
    print(f"Cache entries: {len(model.logit_cache.cache)}")

    # Verify shapes
    assert h_augmented.shape == out.shape, f"Shape mismatch: {h_augmented.shape} != {out.shape}"
    assert logits_out.shape == logits.shape, "Logits shape mismatch"

    print("PASSED\n")
    return True


def test_generation_with_cache():
    """Test generation loop with cache."""
    print("=" * 60)
    print("TEST: Generation with cache")
    print("=" * 60)

    model, cfg = load_model()

    # Enable cache
    model.logit_cache = LogitCacheAttention(
        D=cfg.D,
        V=cfg.vocab,
        n_layers=cfg.n_layers,
        max_tokens=1000,
        n_heads=8,
    )

    # Generation loop
    tokens = [torch.randint(0, cfg.vocab, (1,)).item()]  # Start token
    state = None

    for step in range(10):
        ctx = torch.tensor([tokens[-128:]], dtype=torch.long)
        h = model.embed_tokens(ctx)

        with torch.no_grad():
            out, state, _, _ = model(h, state, adaptive=False, step=step, tokens=ctx)

        # Get logits
        logits = model.lm_head(out)

        # Process with cache
        h_augmented, logits_out = model.process_with_cache(out, logits, use_cache=True)

        # Sample next token
        next_token = logits_out[:, -1, :].argmax(dim=-1).item()
        tokens.append(next_token)

    print(f"Generated {len(tokens)} tokens")
    print(f"Cache size: {model.cache_size_mb():.4f} MB")
    print(f"Cache entries: {len(model.logit_cache.cache)}")

    print("PASSED\n")
    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("EVASTACK + LOGITCACHE INTEGRATION TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_forward_without_cache,
        test_forward_with_cache,
        test_generation_with_cache,
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
