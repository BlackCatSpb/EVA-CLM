"""
Test: EVAStack integration with unified LogitCache (dual-mode).
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.serialization import add_safe_globals
from core import EVAConfig, EVAStack
from core.config import WideBindConfig
add_safe_globals([EVAConfig, WideBindConfig])


def load_model():
    ckpt = torch.load(r"C:\Users\black\OneDrive\Desktop\EVA CLM\checkponts\best 19.pt",
                      map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    # Enable logit cache
    cfg.logit_cache_enabled = True
    cfg.logit_cache_max_tokens = 1000
    cfg.logit_cache_n_heads = 8
    model = EVAStack(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    return model, cfg


def test_training_mode():
    """Test training mode: stores h, gradient flows."""
    print("=" * 60)
    print("TEST 1: TRAINING mode (store h)")
    print("=" * 60)

    model, cfg = load_model()
    model.train()

    tokens = torch.randint(0, cfg.vocab, (1, 32))
    h = model.embed_tokens(tokens)
    h.requires_grad_(True)
    h.retain_grad()  # needed for non-leaf tensor

    out, state, _, _ = model(h, None, adaptive=False, step=0, tokens=tokens)
    logits = model.lm_head(out)

    # Forward with cache (training mode)
    h_aug, logits_out = model.process_with_cache(out, logits, use_cache=True)

    # Compute loss
    loss = h_aug.sum()
    loss.backward()

    # Check gradient on input h (the key: does gradient flow through cache?)
    h_has_grad = h.grad is not None and not torch.isnan(h.grad).any()

    print(f"Output shape: {h_aug.shape}")
    print(f"Cache size: {model.cache_size_mb():.4f} MB (training mode = h storage)")
    print(f"Cache entries: {len(model.logit_cache.cache)}")
    print(f"Input h gradient flows: {h_has_grad}")
    if h.grad is not None:
        print(f"Gradient norm: {h.grad.norm().item():.6f}")

    if h_has_grad and not torch.isnan(h_aug).any():
        print("PASSED\n")
        return True
    else:
        print("FAILED\n")
        return False


def test_inference_mode():
    """Test inference mode: stores compressed logits."""
    print("=" * 60)
    print("TEST 2: INFERENCE mode (store logits)")
    print("=" * 60)

    model, cfg = load_model()
    model.eval()

    tokens = torch.randint(0, cfg.vocab, (1, 128))
    h = model.embed_tokens(tokens)

    with torch.no_grad():
        out, state, _, _ = model(h, None, adaptive=False, step=0, tokens=tokens)
        logits = model.lm_head(out)

        # Forward with cache (inference mode)
        h_aug, logits_out = model.process_with_cache(out, logits, use_cache=True)

    cache_size = model.cache_size_mb()

    print(f"Output shape: {h_aug.shape}")
    print(f"Cache size: {cache_size:.4f} MB (inference mode = compressed logits)")
    print(f"Cache entries: {len(model.logit_cache.cache)}")

    if not torch.isnan(h_aug).any():
        print("PASSED\n")
        return True
    else:
        print("FAILED\n")
        return False


def test_generation():
    """Test generation with cache."""
    print("=" * 60)
    print("TEST 3: Generation with cache")
    print("=" * 60)

    model, cfg = load_model()
    model.eval()

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

    cache_size = model.cache_size_mb()

    print(f"Generated {len(tokens)} tokens")
    print(f"Cache size: {cache_size:.4f} MB")

    if not any(torch.isnan(h_aug).any() for _ in [1]):
        print("PASSED\n")
        return True
    else:
        print("FAILED\n")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("EVASTACK + UNIFIED LOGIT CACHE TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_training_mode,
        test_inference_mode,
        test_generation,
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
