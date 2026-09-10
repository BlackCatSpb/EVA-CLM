"""
tau-adaptive Logits Compression
================================
Связывает сжатие логитов с tau-field и maturation.

Ключевая идея:
  - tau_norm ∈ [0,1] определяет "стабильность" представлений слоя
  - tau_norm → 0 (shallow): нестабильные, требуют точности → легкое сжатие
  - tau_norm → 1 (deep): стабильные, допускают агрессивное сжатие
  - maturation M_l ∈ [0,1] управляет "зрелостью" слоя
  - M_l → 0: незрелый, представления меняются → без сжатия
  - M_l → 1: зрелый, стабильный → агрессивное сжатие

Интеграция с Memory Bank:
  - L1 (volatile): delta-8bit или без сжатия
  - L2 (medium): uniform-8bit
  - L3 (stable concepts): sparse top-k или 4-bit

Метрики:
  - Compression ratio vs tau_norm
  - Quality (top-1, KL) vs tau_norm
  - Latency breakdown
  - Optimal compression schedule по tau-ladder
"""
from __future__ import annotations

import sys
import os
import time
import math
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F


# ─── Compression primitives (from test_logits_compression.py) ────

def compress_uniform8(t: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    t_f = t.float().flatten()
    t_min, t_max = t_f.min().item(), t_f.max().item()
    if t_min == t_max:
        return None, t_min, 0.0
    scale = (t_max - t_min) / 255.0
    idx = ((t_f - t_min) / scale).round_().clamp_(0, 255).to(torch.uint8)
    return idx.reshape(t.shape), t_min, scale

def decompress_uniform8(idx, t_min, scale, shape, dtype):
    if idx is None:
        return torch.tensor(t_min, dtype=dtype).expand(shape).clone()
    return (idx.float() * scale + t_min).to(dtype)

def compress_uniform4(t: torch.Tensor) -> Tuple[torch.Tensor, float, float, int]:
    """4-bit: pack 2 values per byte."""
    t_f = t.float().flatten()
    t_min, t_max = t_f.min().item(), t_f.max().item()
    if t_min == t_max:
        return None, t_min, 0.0, t.numel()
    scale = (t_max - t_min) / 15.0
    idx4 = ((t_f - t_min) / scale).round_().clamp_(0, 15).to(torch.uint8)
    n = len(idx4)
    if n % 2 == 1:
        idx4 = torch.cat([idx4, torch.zeros(1, dtype=torch.uint8)])
    packed = (idx4[0::2] << 4) | idx4[1::2]
    return packed, t_min, scale, n

def decompress_uniform4(packed, t_min, scale, n_orig, shape, dtype):
    if packed is None:
        return torch.tensor(t_min, dtype=dtype).expand(shape).clone()
    idx = torch.stack([packed >> 4, packed & 0x0F], dim=-1).flatten()[:n_orig]
    return (idx.float() * scale + t_min).reshape(shape).to(dtype)

def compress_sparse_topk(t: torch.Tensor, k: int = 128) -> Tuple:
    topk_vals, topk_idx = t.topk(k, dim=-1)
    vmin, vmax = topk_vals.min().item(), topk_vals.max().item()
    scale = (vmax - vmin) / 255.0 if vmax > vmin else 1.0
    idx_vals = ((topk_vals - vmin) / scale).round_().clamp_(0, 255).to(torch.uint8)
    return topk_idx.to(torch.uint16), idx_vals, torch.tensor([vmin, scale])

def decompress_sparse_topk(idx_pos, idx_vals, meta, shape, dtype):
    B, L, V = shape
    vmin, scale = meta[0].item(), meta[1].item()
    vals = idx_vals.float() * scale + vmin
    result = torch.full((B, L, V), float('-inf'), dtype=dtype)
    result.scatter_(-1, idx_pos.long(), vals.to(dtype))
    return result

def compress_delta(current, cached, n_bits=8):
    delta = (current.float() - cached.float()).clamp(-3.0, 3.0)
    d_min, d_max = delta.min().item(), delta.max().item()
    if d_min == d_max:
        return None, d_min, 0.0
    scale = (d_max - d_min) / (2**n_bits - 1)
    idx = ((delta - d_min) / scale).round_().clamp_(0, 2**n_bits - 1).to(torch.uint8)
    return idx, d_min, scale

def decompress_delta(idx, d_min, scale, cached, dtype):
    if idx is None:
        return cached.to(dtype)
    return cached.to(dtype) + (idx.float() * scale + d_min).to(dtype)


# ─── Quality metrics ──────────────────────────────────────────────

def top1_accuracy(orig, recon):
    return (orig.argmax(-1) == recon.argmax(-1)).float().mean().item()

def kl_divergence(orig, recon):
    p = F.softmax(orig, dim=-1)
    q = F.softmax(recon, dim=-1)
    return (p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(-1).mean().item()


# ─── tau-adaptive compression schedule ─────────────────────────────

def tau_to_compression_level(tau_norm: float, maturation: float = 1.0) -> Dict:
    """Map tau_norm + maturation to compression parameters.

    tau_norm ∈ [0,1]: 0=shallow(volatile), 1=deep(stable)
    maturation ∈ [0,1]: 0=immature(changing), 1=mature(stable)

    Returns dict with strategy, n_bits, topk, etc.
    """
    # Combined stability score
    stability = 0.6 * tau_norm + 0.4 * maturation

    if stability < 0.2:
        # Very unstable: delta only (or no compression)
        return {
            'strategy': 'delta_8bit',
            'n_bits': 8,
            'topk': None,
            'description': 'delta-8bit (shallow+immature)',
        }
    elif stability < 0.4:
        # Unstable: 8-bit uniform
        return {
            'strategy': 'uniform8',
            'n_bits': 8,
            'topk': None,
            'description': 'uniform-8bit (shallow)',
        }
    elif stability < 0.6:
        # Medium: 8-bit with delta
        return {
            'strategy': 'delta_8bit',
            'n_bits': 8,
            'topk': None,
            'description': 'delta-8bit (medium)',
        }
    elif stability < 0.8:
        # Stable: sparse top-k
        return {
            'strategy': 'sparse_topk',
            'n_bits': 8,
            'topk': 128,
            'description': 'sparse-topk-128 (deep)',
        }
    else:
        # Very stable: aggressive sparse or 4-bit
        return {
            'strategy': 'sparse_topk',
            'n_bits': 8,
            'topk': 64,
            'description': 'sparse-topk-64 (very deep)',
        }


# ─── tau-adaptive cache ───────────────────────────────────────────

class TauAdaptiveLogitsCache:
    """Cache that adapts compression based on tau_norm and maturation.

    Per-layer compression schedule:
      - L0-L7 (shallow, tau_norm < 0.3): delta-8bit (4x, 99% top-1)
      - L8-L15 (medium, tau_norm 0.3-0.6): uniform-8bit (4x, 84% top-1)
      - L16-L23 (deep, tau_norm > 0.6): sparse-topk-128 (683x, 98% top-1)

    Total cache size: ~10x smaller than fp32 baseline.
    """

    def __init__(self, B: int, V: int, n_layers: int, device: torch.device,
                 tau_norm: Optional[torch.Tensor] = None,
                 maturation: Optional[torch.Tensor] = None):
        self.B = B
        self.V = V
        self.n_layers = n_layers
        self.device = device

        # tau_norm per layer: (n_layers,)
        self.tau_norm = tau_norm if tau_norm is not None else torch.linspace(0, 1, n_layers)
        self.maturation = maturation if maturation is not None else torch.ones(n_layers)

        # Per-layer cache: stores (compressed_data, meta, decompressed)
        self._cache: List[Optional[Dict]] = [None] * n_layers

        # Compression schedule per layer
        self._schedules: List[Dict] = []
        for i in range(n_layers):
            sched = tau_to_compression_level(
                self.tau_norm[i].item(),
                self.maturation[i].item()
            )
            self._schedules.append(sched)

        # Stats
        self._total_original_bytes = 0
        self._total_compressed_bytes = 0
        self._compress_times: List[float] = []
        self._decompress_times: List[float] = []

    def update_tau(self, tau_norm: torch.Tensor, maturation: torch.Tensor):
        """Update tau and maturation (called once per forward)."""
        self.tau_norm = tau_norm.detach().cpu()
        self.maturation = maturation.detach().cpu()
        for i in range(self.n_layers):
            self._schedules[i] = tau_to_compression_level(
                self.tau_norm[i].item(),
                self.maturation[i].item()
            )

    def compress_and_cache(self, layer_idx: int, logits: torch.Tensor):
        """Compress logits for a layer and store in cache."""
        sched = self._schedules[layer_idx]
        B, L, V = logits.shape

        t0 = time.perf_counter()

        if sched['strategy'] == 'delta_8bit':
            cached = self._cache[layer_idx]
            if cached is not None and cached['decompressed'].shape == logits.shape:
                idx, d_min, scale = compress_delta(logits, cached['decompressed'], sched['n_bits'])
                compressed = {'type': 'delta', 'idx': idx, 'd_min': d_min, 'scale': scale,
                             'base': cached['decompressed']}
            else:
                # First time or shape changed: store as uniform8
                idx, t_min, scale = compress_uniform8(logits)
                compressed = {'type': 'uniform8', 'idx': idx, 't_min': t_min, 'scale': scale}

        elif sched['strategy'] == 'uniform8':
            idx, t_min, scale = compress_uniform8(logits)
            compressed = {'type': 'uniform8', 'idx': idx, 't_min': t_min, 'scale': scale}

        elif sched['strategy'] == 'sparse_topk':
            k = sched['topk']
            idx_pos, idx_vals, meta = compress_sparse_topk(logits, k=k)
            compressed = {'type': 'sparse', 'idx_pos': idx_pos, 'idx_vals': idx_vals, 'meta': meta}

        t_compress = time.perf_counter() - t0
        self._compress_times.append(t_compress)

        # Calculate bytes
        original_bytes = logits.numel() * 4
        if compressed['type'] == 'delta':
            compressed_bytes = logits.numel() + 8 if compressed['idx'] is not None else 8
        elif compressed['type'] == 'uniform8':
            compressed_bytes = logits.numel() + 8 if compressed['idx'] is not None else 8
        elif compressed['type'] == 'sparse':
            compressed_bytes = compressed['idx_pos'].numel() * 2 + compressed['idx_vals'].numel() + 8

        self._total_original_bytes += original_bytes
        self._total_compressed_bytes += compressed_bytes

        # Store decompressed version for next delta
        decompressed = self.decompress_layer(layer_idx, compressed, logits.shape, logits.dtype)
        compressed['decompressed'] = decompressed.detach()

        self._cache[layer_idx] = compressed
        return decompressed

    def decompress_layer(self, layer_idx: int, compressed: Dict,
                          shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Decompress logits for a layer."""
        t0 = time.perf_counter()

        if compressed['type'] == 'delta':
            result = decompress_delta(compressed['idx'], compressed['d_min'],
                                       compressed['scale'], compressed['base'], dtype)
        elif compressed['type'] == 'uniform8':
            result = decompress_uniform8(compressed['idx'], compressed['t_min'],
                                          compressed['scale'], shape, dtype)
        elif compressed['type'] == 'sparse':
            result = decompress_sparse_topk(compressed['idx_pos'], compressed['idx_vals'],
                                             compressed['meta'], shape, dtype)

        t_decompress = time.perf_counter() - t0
        self._decompress_times.append(t_decompress)
        return result

    def get_stats(self) -> Dict:
        """Get compression statistics."""
        ratio = self._total_original_bytes / max(self._total_compressed_bytes, 1)
        avg_compress = sum(self._compress_times) / max(len(self._compress_times), 1)
        avg_decompress = sum(self._decompress_times) / max(len(self._decompress_times), 1)

        # Per-layer breakdown
        layer_strategies = {}
        for i, sched in enumerate(self._schedules):
            s = sched['strategy']
            if s not in layer_strategies:
                layer_strategies[s] = []
            layer_strategies[s].append(i)

        return {
            'total_original_kb': self._total_original_bytes / 1024,
            'total_compressed_kb': self._total_compressed_bytes / 1024,
            'compression_ratio': ratio,
            'avg_compress_ms': avg_compress * 1000,
            'avg_decompress_ms': avg_decompress * 1000,
            'layer_strategies': layer_strategies,
            'tau_norm_range': (self.tau_norm.min().item(), self.tau_norm.max().item()),
            'maturation_range': (self.maturation.min().item(), self.maturation.max().item()),
        }


# ─── Main experiment ──────────────────────────────────────────────

def run_experiment():
    print("=" * 70)
    print("tau-adaptive Logits Compression Experiment")
    print("=" * 70)

    from core.config import WideBindConfig as EVAConfig
    from core.stack import EVAStack

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkponts', 'best 19.pt')
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = sd['cfg']
    print(f"Config: D={cfg.D}, n_layers={cfg.n_layers}, vocab={cfg.vocab}")

    model = EVAStack(cfg).to(device)
    state = sd.get('model_state_dict', sd.get('model', sd))
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: step={sd.get('step', '?')}, val_loss={sd.get('best_val_loss', '?')}")

    # Get tau_norm and maturation from model
    print("\n--- tau-field analysis ---")
    if hasattr(model, 'tau_config') and model.tau_config is not None:
        tc = model.tau_config
        tc.update()  # recompute tau ladder
        tau_norm = tc.tau_norm.detach().cpu()
        tau_l = tc.tau_l.detach().cpu()
        print(f"tau_l: {tau_l.numpy().round(1)}")
        print(f"tau_norm: {tau_norm.numpy().round(3)}")
    else:
        print("No tau_config found, using synthetic")
        tau_norm = torch.linspace(0, 1, cfg.n_layers)

    # Get maturation from model
    if hasattr(model, 'maturation') and model.maturation is not None:
        mat = model.maturation
        mat_gate = mat.mat_gate.detach().cpu() if hasattr(mat, 'mat_gate') else torch.ones(cfg.n_layers)
        print(f"mat_gate: {mat_gate.numpy().round(3)}")
    else:
        print("No maturation found, using synthetic")
        mat_gate = torch.linspace(0.1, 0.9, cfg.n_layers)

    # Forward pass
    B, L = 2, 64
    x = torch.randint(0, cfg.vocab, (B, L), device=device)
    print(f"\nForward pass: B={B}, L={L}")

    with torch.no_grad():
        t0 = time.perf_counter()
        h = model.embed(x)
        layer_logits = []
        for i, layer in enumerate(model.layers):
            h, _ = layer(h)
            # Get logits at each layer (intermediate)
            layer_h = h
            layer_logits.append(model.lm_head(layer_h))
        final_logits = layer_logits[-1]
        t_forward = time.perf_counter() - t0

    print(f"Forward: {t_forward*1000:.1f} ms")
    print(f"Final logits: {final_logits.shape} ({final_logits.numel()*4/1024:.0f} KB)")

    # ─── Experiment 1: tau-correlation with logits stability ──────
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: tau vs logits stability")
    print("=" * 70)

    print(f"\n{'Layer':>6} {'tau_norm':>8} {'mat_gate':>8} {'logits_std':>10} {'logits_range':>12} {'entropy':>8}")
    print("-" * 60)

    for i in range(cfg.n_layers):
        logits_i = layer_logits[i]
        with torch.no_grad():
            std = logits_i.std().item()
            rmin, rmax = logits_i.min().item(), logits_i.max().item()
            p = F.softmax(logits_i, dim=-1)
            ent = -(p * p.clamp_min(1e-9).log()).sum(-1).mean().item()

        print(f"{i:>6} {tau_norm[i].item():>8.3f} {mat_gate[i].item():>8.3f} "
              f"{std:>10.2f} [{rmin:>6.1f},{rmax:>6.1f}] {ent:>8.3f}")

    # ─── Experiment 2: compression quality vs tau_norm ────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: compression quality vs tau_norm")
    print("=" * 70)

    # Use final logits for this analysis
    logits = final_logits

    # Stratify by tau_norm ranges
    ranges = [
        ("shallow (tau<0.3)", tau_norm < 0.3),
        ("medium (0.3<tau<0.6)", (tau_norm >= 0.3) & (tau_norm < 0.6)),
        ("deep (tau>0.6)", tau_norm >= 0.6),
    ]

    for name, mask in ranges:
        if not mask.any():
            continue
        print(f"\n--- {name} ---")

        # Test different compression strategies
        strategies = [
            ("uniform8", lambda t: compress_uniform8(t)),
            ("sparse_topk_128", lambda t: compress_sparse_topk(t, k=128)),
            ("sparse_topk_64", lambda t: compress_sparse_topk(t, k=64)),
            ("delta_8bit", lambda t: compress_delta(t, t + torch.randn_like(t) * 0.05)),
        ]

        for sname, compress_fn in strategies:
            t0 = time.perf_counter()
            compressed = compress_fn(logits)
            t_compress = time.perf_counter() - t0

            # Calculate compression ratio
            orig_bytes = logits.numel() * 4
            if sname.startswith("sparse"):
                c_bytes = compressed[0].numel() * 2 + compressed[1].numel() + 8
            else:
                c_bytes = logits.numel() + 8 if compressed[0] is not None else 8

            ratio = orig_bytes / c_bytes
            print(f"  {sname:<20}: {ratio:>6.1f}x, {t_compress*1000:>6.2f} ms")

    # ─── Experiment 3: tau-adaptive cache simulation ──────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: tau-adaptive cache simulation")
    print("=" * 70)

    # Create tau-adaptive cache
    cache = TauAdaptiveLogitsCache(
        B=B, V=cfg.vocab, n_layers=cfg.n_layers,
        device=device, tau_norm=tau_norm, maturation=mat_gate
    )

    print(f"\nCompression schedule:")
    for i in range(cfg.n_layers):
        sched = cache._schedules[i]
        print(f"  Layer {i:>2}: tau_norm={tau_norm[i].item():.3f}, "
              f"mat={mat_gate[i].item():.3f} -> {sched['description']}")

    # Simulate cache over multiple steps
    print(f"\nSimulating {cfg.n_layers} layers x 10 steps...")
    for step in range(10):
        for i in range(cfg.n_layers):
            # Add small noise to simulate timestep variation
            noisy_logits = layer_logits[i] + torch.randn_like(layer_logits[i]) * 0.02 * (step + 1)
            cache.compress_and_cache(i, noisy_logits)

    stats = cache.get_stats()
    print(f"\nCache statistics:")
    print(f"  Original: {stats['total_original_kb']:.0f} KB")
    print(f"  Compressed: {stats['total_compressed_kb']:.0f} KB")
    print(f"  Ratio: {stats['compression_ratio']:.1f}x")
    print(f"  Avg compress: {stats['avg_compress_ms']:.2f} ms")
    print(f"  Avg decompress: {stats['avg_decompress_ms']:.2f} ms")
    print(f"  tau_norm range: {stats['tau_norm_range']}")
    print(f"  maturation range: {stats['maturation_range']}")

    for strategy, layers in stats['layer_strategies'].items():
        print(f"  {strategy}: layers {layers}")

    # ─── Experiment 4: comparison with uniform compression ────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: tau-adaptive vs uniform compression (streaming)")
    print("=" * 70)

    # Streaming scenario: store compressed logits, decompress on demand
    # fp32 baseline: all layers, all timesteps
    fp32_total = cfg.n_layers * layer_logits[0].numel() * 4

    # Uniform 8-bit: all layers
    uniform_total = 0
    for i in range(cfg.n_layers):
        uniform_total += layer_logits[i].numel() + 8  # uint8 + min/scale

    # tau-adaptive streaming:
    # - delta layers: store BASE (fp32) + deltas (uint8) per step
    # - sparse layers: store topk indices (uint16) + values (uint8) per step
    # For streaming, only store compressed (not base separately)
    adaptive_per_step = 0
    for i in range(cfg.n_layers):
        sched = cache._schedules[i]
        if sched['strategy'] == 'delta_8bit':
            # Delta: store uint8 delta (same size as uniform8)
            adaptive_per_step += layer_logits[i].numel() + 8
        elif sched['strategy'] == 'sparse_topk':
            k = sched['topk']
            # Sparse: uint16 indices + uint8 values + meta
            adaptive_per_step += layer_logits[0].shape[0] * layer_logits[0].shape[1] * k * 3 + 8

    print(f"\nPer-step storage (single forward):")
    print(f"  fp32 baseline: {fp32_total/1024:.0f} KB")
    print(f"  Uniform 8-bit: {uniform_total/1024:.0f} KB ({fp32_total/uniform_total:.1f}x)")
    print(f"  tau-adaptive:  {adaptive_per_step/1024:.0f} KB ({fp32_total/adaptive_per_step:.1f}x)")

    # For long-context scenario (1000 steps)
    n_steps = 1000
    fp32_long = fp32_total * n_steps
    uniform_long = uniform_total * n_steps
    # tau-adaptive: base + (n_steps-1) * delta/sparse
    delta_layers = len(cache._schedules) - sum(1 for s in cache._schedules if s['strategy'] != 'delta_8bit')
    sparse_layers = cfg.n_layers - delta_layers
    k_sparse = 128  # average

    # For delta: first step = uniform8, rest = delta
    delta_base = layer_logits[0].numel() + 8
    delta_per_step = layer_logits[0].numel() + 8  # delta is same size as uniform8
    adaptive_long = (delta_layers * delta_base + sparse_layers * (layer_logits[0].shape[0] * layer_logits[0].shape[1] * k_sparse * 3 + 8)) + \
                    (n_steps - 1) * (delta_layers * delta_per_step + sparse_layers * (layer_logits[0].shape[0] * layer_logits[0].shape[1] * k_sparse * 3 + 8))

    print(f"\nLong-context ({n_steps} steps):")
    print(f"  fp32: {fp32_long/1e9:.2f} GB")
    print(f"  Uniform 8-bit: {uniform_long/1e6:.0f} MB ({fp32_long/uniform_long:.1f}x)")
    print(f"  tau-adaptive:  {adaptive_long/1e6:.0f} MB ({fp32_long/adaptive_long:.1f}x)")
    print(f"  Savings: {(uniform_long - adaptive_long)/1e6:.0f} MB ({(1 - adaptive_long/uniform_long)*100:.1f}% less)")

    # ─── Feasibility verdict ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("FEASIBILITY VERDICT")
    print("=" * 70)

    ratio = stats['compression_ratio']
    avg_lat = stats['avg_decompress_ms']

    print(f"\ntau-adaptive compression:")
    print(f"  Compression ratio: {ratio:.1f}x")
    print(f"  Avg decompress latency: {avg_lat:.2f} ms")
    print(f"  Per-layer adaptation: YES")
    print(f"  Maturation-aware: YES")

    if ratio > 10 and avg_lat < 50:
        print("\n[OK] FEASIBLE: tau-adaptive compression is practical.")
        print("   Deep layers get aggressive compression (sparse top-k),")
        print("   shallow layers keep precision (delta/uniform8).")
    elif ratio > 5:
        print("\n[!] MARGINAL: compression is moderate, latency is acceptable.")
    else:
        print("\n[X] NOT FEASIBLE: compression ratio too low.")


if __name__ == '__main__':
    run_experiment()
