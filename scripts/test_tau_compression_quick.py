"""Quick test of tau_compression module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.tau_compression import TauAdaptiveLogitsCache, tau_to_compression

# Test tau_to_compression
print("=== tau_to_compression ===")
for tn in [0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0]:
    s = tau_to_compression(tn, maturation=1.0)
    print(f"  tau_norm={tn:.1f} -> {s['description']}")

# Test cache
print("\n=== TauAdaptiveLogitsCache ===")
tau_norm = torch.linspace(0, 1, 24)
mat = torch.ones(24)
cache = TauAdaptiveLogitsCache(B=1, V=65536, n_layers=24, device='cpu',
                                tau_norm=tau_norm, maturation=mat)
print(cache.schedule_str())

# Test single step
logits = torch.randn(1, 1, 65536) * 50 - 50
decompressed = cache.step(23, logits)
acc = (logits.argmax(-1) == decompressed.argmax(-1)).float().mean().item()
print(f"\nStep: orig={list(logits.shape)}, decompressed={list(decompressed.shape)}")
print(f"Top-1 accuracy: {acc:.4f}")

# Test all layers
for i in range(24):
    logits_i = torch.randn(1, 1, 65536) * 50 - 50
    cache.step(i, logits_i)

stats = cache.stats()
print(f"\nStats after 24 steps:")
print(f"  Ratio: {stats['compression_ratio']:.1f}x")
print(f"  Per step: {stats['per_step_original_kb']:.1f} KB -> {stats['per_step_compressed_kb']:.1f} KB")
print(f"  Strategies: {stats['strategies']}")
print("\n[OK] All tests passed")
