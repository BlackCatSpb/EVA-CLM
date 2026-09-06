"""
Test: Gradient flow through LogitCache (dual-mode).
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logit_cache import LogitCache, LogitAttention, LogitCacheAttention


def test_gradient_flow_training():
    """Test that gradients flow through cache in training mode (store h)."""
    print("=" * 60)
    print("TEST 1: Gradient flow in TRAINING mode (store h)")
    print("=" * 60)

    D = 256
    V = 1000
    cache = LogitCache(V, D, max_tokens=100, n_scales=4)
    attention = LogitAttention(D, V, n_heads=4)

    # Create input with gradient
    h = torch.randn(1, 10, D, requires_grad=True)

    # Store in cache
    cache.store(h, training=True)

    # Retrieve from cache
    cached = cache.retrieve(n=5, training=True)

    # Compute loss
    loss = cached.sum()

    # Backward
    loss.backward()

    # Check gradient
    has_grad = h.grad is not None and not torch.isnan(h.grad).any()

    print(f"Input requires_grad: {h.requires_grad}")
    print(f"Cached shape: {cached.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient exists: {h.grad is not None}")
    print(f"Gradient has NaN: {torch.isnan(h.grad).any().item() if h.grad is not None else 'N/A'}")
    print(f"Gradient norm: {h.grad.norm().item():.6f}" if h.grad is not None else "N/A")

    if has_grad:
        print("PASSED: Gradient flows through training cache\n")
        return True
    else:
        print("FAILED: No gradient through training cache\n")
        return False


def test_gradient_flow_inference():
    """Test that gradients DON'T flow in inference mode (store logits)."""
    print("=" * 60)
    print("TEST 2: No gradient in INFERENCE mode (store logits)")
    print("=" * 60)

    D = 256
    V = 1000
    cache = LogitCache(V, D, max_tokens=100, n_scales=4)

    logits = torch.randn(1, 10, V, requires_grad=True)

    # Store in cache (inference mode - compressed)
    cache.store(logits, training=False)

    # Retrieve from cache
    cached = cache.retrieve(n=5, training=False)

    print(f"Input requires_grad: {logits.requires_grad}")
    print(f"Cached shape: {cached.shape}")
    print(f"Cached requires_grad: {cached.requires_grad}")

    # In inference mode, cached data is detached (compressed)
    if not cached.requires_grad:
        print("PASSED: Inference cache is detached (no gradient)\n")
        return True
    else:
        print("FAILED: Inference cache has gradient (should be detached)\n")
        return False


def test_gradient_through_attention():
    """Test that gradients flow through LogitAttention in training mode."""
    print("=" * 60)
    print("TEST 3: Gradient through LogitAttention (training)")
    print("=" * 60)

    D = 256
    V = 1000
    cache = LogitCache(V, D, max_tokens=100, n_scales=4)
    attention = LogitAttention(D, V, n_heads=4)

    # Create input with gradient
    h = torch.randn(1, 10, D, requires_grad=True)

    # Store in cache
    cache.store(h, training=True)

    # Attend to cache
    output = attention(h, cache, training=True)

    # Compute loss
    loss = output.sum()

    # Backward
    loss.backward()

    # Check gradient
    has_grad = h.grad is not None and not torch.isnan(h.grad).any()

    print(f"Input requires_grad: {h.requires_grad}")
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient exists: {h.grad is not None}")
    print(f"Gradient norm: {h.grad.norm().item():.6f}" if h.grad is not None else "N/A")

    if has_grad:
        print("PASSED: Gradient flows through attention\n")
        return True
    else:
        print("FAILED: No gradient through attention\n")
        return False


def test_gradient_through_full_module():
    """Test gradient flow through LogitCacheAttention (full module)."""
    print("=" * 60)
    print("TEST 4: Gradient through LogitCacheAttention (full)")
    print("=" * 60)

    D = 256
    V = 1000
    module = LogitCacheAttention(D, V, max_tokens=100, n_heads=4)

    # Create inputs with gradients
    h = torch.randn(1, 10, D, requires_grad=True)
    logits = torch.randn(1, 10, V, requires_grad=True)

    # Forward (training mode)
    module.train()
    h_aug, logits_out = module(h, logits, training=True, use_cache=True)

    # Compute loss
    loss = h_aug.sum()

    # Backward
    loss.backward()

    # Check gradients
    h_has_grad = h.grad is not None and not torch.isnan(h.grad).any()
    logits_has_grad = logits.grad is not None  # logits shouldn't have grad in training mode
    module_has_grad = any(p.grad is not None for p in module.parameters())

    print(f"h requires_grad: {h.requires_grad}")
    print(f"h_aug shape: {h_aug.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"h gradient: {h_has_grad}")
    print(f"Module gradient: {module_has_grad}")

    if h_has_grad and module_has_grad:
        print("PASSED: Gradient flows through full module\n")
        return True
    else:
        print("FAILED: No gradient through full module\n")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("LOGIT CACHE GRADIENT FLOW TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_gradient_flow_training,
        test_gradient_flow_inference,
        test_gradient_through_attention,
        test_gradient_through_full_module,
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
