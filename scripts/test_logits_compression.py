"""
FCF-CPR Online Logits Compression Experiment
==============================================
Эксперимент: можно ли адаптировать FCF-CPR для онлайн сжатия/распаковки логитов.

Цель: определить, реально ли сжимать логиты (B, L, V) в реальном времени
для кеширования, с приемлемым качеством и минимальным latency overhead.

Стратегии:
1. Uniform 8-bit per-tensor (базовый FCF-CPR)
2. Per-channel 8-bit (row-wise квантизация)
3. Sparse top-k (хранить только top-k логитов)
4. Delta compression (разность между текущими и кешем)
5. Mixed: 8-bit + delta для статичных + дельта для динамических

Метрики:
- Compression ratio (bytes original / bytes compressed)
- Latency (compress + decompress time)
- Quality: top-1 accuracy, KL divergence, perplexity delta
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


# --- Compression primitives --------------------------------------

def compress_uniform8(t: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """Per-tensor uniform 8-bit quantization."""
    t_f = t.float().flatten()
    t_min = t_f.min().item()
    t_max = t_f.max().item()
    if t_min == t_max:
        return None, t_min, 0.0
    scale = (t_max - t_min) / 255.0
    idx = ((t_f - t_min) / scale).round_().clamp_(0, 255).to(torch.uint8)
    return idx.reshape(t.shape), t_min, scale


def decompress_uniform8(idx: Optional[torch.Tensor], t_min: float, scale: float,
                         shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """Restore from uniform 8-bit."""
    if idx is None:
        return torch.tensor(t_min, dtype=dtype).expand(shape).clone()
    return (idx.float() * scale + t_min).to(dtype)


def compress_per_channel8(t: torch.Tensor, dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel uniform 8-bit (row-wise for 2D)."""
    t_f = t.float()
    n_ch = t_f.shape[dim]
    mins, scales = [], []
    parts = []
    for i in range(n_ch):
        sl = t_f.select(dim, i)
        s_min = sl.min().item()
        s_max = sl.max().item()
        mins.append(s_min)
        if s_min == s_max:
            scales.append(0.0)
            parts.append(torch.full(sl.shape, 0, dtype=torch.uint8, device=t.device))
        else:
            sc = (s_max - s_min) / 255.0
            scales.append(sc)
            idx = ((sl - s_min) / sc).round_().clamp_(0, 255).to(torch.uint8)
            parts.append(idx.unsqueeze(dim))
    indices = torch.cat(parts, dim=dim)
    return indices, torch.tensor(mins), torch.tensor(scales)


def decompress_per_channel8(indices: torch.Tensor, mins: torch.Tensor, scales: torch.Tensor,
                              orig_shape: torch.Size, dtype: torch.dtype, dim: int = -1) -> torch.Tensor:
    """Restore from per-channel 8-bit."""
    n_ch = mins.shape[0]
    # Determine which dim has n_ch channels
    if dim < 0:
        dim = len(orig_shape) + dim
    restored = torch.zeros(orig_shape, dtype=dtype)
    for i in range(n_ch):
        if scales[i] == 0.0:
            other_dims = [s for j, s in enumerate(orig_shape) if j != dim]
            sl = torch.full(other_dims, mins[i].item(), dtype=dtype)
        else:
            sl = indices.select(dim, i).float() * scales[i] + mins[i]
        restored.select(dim, i).copy_(sl)
    return restored


def compress_sparse_topk(logits: torch.Tensor, k: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Store only top-k logits per position + their indices."""
    B, L, V = logits.shape
    topk_vals, topk_idx = logits.topk(k, dim=-1)  # (B, L, k)
    # Quantize values to 8-bit
    vmin = topk_vals.min().item()
    vmax = topk_vals.max().item()
    scale = (vmax - vmin) / 255.0 if vmax > vmin else 1.0
    idx_vals = ((topk_vals - vmin) / scale).round_().clamp_(0, 255).to(torch.uint8)
    # Store indices as uint16 (V <= 65536)
    idx_pos = topk_idx.to(torch.uint16)
    return idx_pos, idx_vals, torch.tensor([vmin, scale])


def decompress_sparse_topk(idx_pos: torch.Tensor, idx_vals: torch.Tensor,
                            meta: torch.Tensor, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """Restore full logits from sparse top-k (rest = -inf)."""
    B, L, V = shape
    k = idx_pos.shape[-1]
    vmin, scale = meta[0].item(), meta[1].item()
    vals = idx_vals.float() * scale + vmin
    result = torch.full((B, L, V), float('-inf'), dtype=dtype)
    result.scatter_(-1, idx_pos.long(), vals.to(dtype))
    return result


def compress_delta(current: torch.Tensor, cached: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, float, float]:
    """Compress difference between current and cached logits."""
    delta = (current.float() - cached.float()).clamp(-3.0, 3.0)  # clip delta
    d_min = delta.min().item()
    d_max = delta.max().item()
    if d_min == d_max:
        return None, d_min, 0.0
    scale = (d_max - d_min) / (2 ** n_bits - 1)
    idx = ((delta - d_min) / scale).round_().clamp_(0, 2 ** n_bits - 1).to(torch.uint8)
    return idx, d_min, scale


def decompress_delta(idx: Optional[torch.Tensor], d_min: float, scale: float,
                      cached: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Restore logits from delta + cache."""
    if idx is None:
        return cached.to(dtype)
    delta = (idx.float() * scale + d_min).to(dtype)
    return cached.to(dtype) + delta


# --- Quality metrics ----------------------------------------------

def top1_accuracy(orig: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """How often top-1 token matches."""
    orig_top1 = orig.argmax(dim=-1)
    recon_top1 = reconstructed.argmax(dim=-1)
    return (orig_top1 == recon_top1).float().mean().item()


def kl_divergence(orig: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """KL(p_orig || p_recon) averaged over batch and position."""
    p = F.softmax(orig, dim=-1)
    q = F.softmax(reconstructed, dim=-1)
    kl = (p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(dim=-1)
    return kl.mean().item()


def perplexity_delta(orig: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """CE(orig) - CE(reconstructed) — positive = quality loss."""
    p = F.softmax(orig, dim=-1)
    ce_orig = -(p * orig.clamp_min(1e-9).log()).sum(dim=-1).mean()
    ce_recon = -(p * reconstructed.clamp_min(1e-9).log()).sum(dim=-1).mean()
    return (ce_recon - ce_orig).item()


# --- Main experiment ----------------------------------------------

def run_experiment():
    print("=" * 70)
    print("FCF-CPR Online Logits Compression Experiment")
    print("=" * 70)

    # Load model
    from core.config import WideBindConfig as EVAConfig
    from core.stack import EVAStack

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Try loading checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkponts', 'best 19.pt')
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Use config from checkpoint if available
        if 'cfg' in sd and sd['cfg'] is not None:
            cfg = sd['cfg']
            print(f"Config from checkpoint: D={cfg.D}, n_layers={cfg.n_layers}, vocab={cfg.vocab}")
        else:
            cfg = EVAConfig()
            print(f"Default config: D={cfg.D}, n_layers={cfg.n_layers}, vocab={cfg.vocab}")
        model = EVAStack(cfg).to(device)
        state = sd.get('model_state_dict', sd.get('model', sd))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded: missing={len(missing)}, unexpected={len(unexpected)}")
        model.eval()
        print("Checkpoint loaded successfully")
    else:
        print("No checkpoint found, using random weights")
        cfg = EVAConfig()
        model = EVAStack(cfg).to(device)
        model.eval()

    # Generate sample input
    B, L = 2, 64
    x = torch.randint(0, cfg.vocab, (B, L), device=device)

    print(f"\nInput: B={B}, L={L}, seq={x.shape}")

    # Forward pass to get logits
    print("\nRunning forward pass...")
    with torch.no_grad():
        t0 = time.perf_counter()
        h = model.embed(x)
        for layer in model.layers:
            h, _ = layer(h)
        logits = model.lm_head(h)  # (B, L, V)
        t_forward = time.perf_counter() - t0

    print(f"Forward time: {t_forward*1000:.1f} ms")
    print(f"Logits shape: {logits.shape} ({logits.numel():,} elements)")
    print(f"Logits size (fp32): {logits.numel() * 4 / 1024:.1f} KB")

    # --- Analyze logits distribution ------------------------------
    print("\n" + "-" * 70)
    print("LOGITS DISTRIBUTION ANALYSIS")
    print("-" * 70)

    with torch.no_grad():
        l_flat = logits.float().flatten()
        print(f"  Range: [{l_flat.min().item():.4f}, {l_flat.max().item():.4f}]")
        print(f"  Mean: {l_flat.mean().item():.4f}")
        print(f"  Std:  {l_flat.std().item():.4f}")
        print(f"  |mean/std|: {abs(l_flat.mean().item() / l_flat.std().item()):.4f}")

        # Entropy of softmax distribution
        p = F.softmax(logits, dim=-1)
        ent = -(p * p.clamp_min(1e-9).log()).sum(dim=-1)
        print(f"  Entropy: mean={ent.mean().item():.4f}, std={ent.std().item():.4f}")
        print(f"  Max entropy (log V): {math.log(cfg.vocab):.4f}")

        # Sparsity: how many logits are close to -inf
        threshold = logits.max(dim=-1, keepdim=True).values - 10.0
        sparse_mask = logits < threshold
        sparsity = sparse_mask.float().mean().item()
        print(f"  Sparsity (within 10 of max): {sparsity:.4f}")

        # Per-position variance
        pos_var = logits.var(dim=-1).mean().item()
        print(f"  Per-position variance: {pos_var:.4f}")

    # --- Test compression strategies ------------------------------
    print("\n" + "-" * 70)
    print("COMPRESSION STRATEGIES")
    print("-" * 70)

    original_bytes = logits.numel() * 4  # fp32
    results = []

    # Strategy 1: Uniform 8-bit per-tensor
    print("\n[1] Uniform 8-bit per-tensor")
    with torch.no_grad():
        t0 = time.perf_counter()
        idx, t_min, scale = compress_uniform8(logits)
        recon1 = decompress_uniform8(idx, t_min, scale, logits.shape, logits.dtype)
        t_compress = time.perf_counter() - t0

        c_bytes = logits.numel() + 8  # uint8 + min + scale
        ratio = original_bytes / c_bytes
        acc = top1_accuracy(logits, recon1)
        kl = kl_divergence(logits, recon1)
        pd = perplexity_delta(logits, recon1)
        print(f"  Compressed: {c_bytes/1024:.1f} KB (ratio: {ratio:.2f}x)")
        print(f"  Latency: {t_compress*1000:.2f} ms")
        print(f"  Top-1 accuracy: {acc:.4f}")
        print(f"  KL divergence: {kl:.6f}")
        print(f"  Perplexity delta: {pd:.4f}")
        results.append(('uniform8', ratio, t_compress, acc, kl, pd))

    # Strategy 2: Per-channel 8-bit
    print("\n[2] Per-channel 8-bit (row-wise)")
    with torch.no_grad():
        t0 = time.perf_counter()
        idx2, mins2, scales2 = compress_per_channel8(logits, dim=-1)
        recon2 = decompress_per_channel8(idx2, mins2, scales2, logits.shape, logits.dtype, dim=-1)
        t_compress = time.perf_counter() - t0

        c_bytes = logits.numel() + logits.shape[-1] * 8  # uint8 + per-channel min/scale
        ratio = original_bytes / c_bytes
        acc = top1_accuracy(logits, recon2)
        kl = kl_divergence(logits, recon2)
        pd = perplexity_delta(logits, recon2)
        print(f"  Compressed: {c_bytes/1024:.1f} KB (ratio: {ratio:.2f}x)")
        print(f"  Latency: {t_compress*1000:.2f} ms")
        print(f"  Top-1 accuracy: {acc:.4f}")
        print(f"  KL divergence: {kl:.6f}")
        print(f"  Perplexity delta: {pd:.4f}")
        results.append(('per_channel8', ratio, t_compress, acc, kl, pd))

    # Strategy 3: Sparse top-k (k=64)
    print("\n[3] Sparse top-k (k=64)")
    with torch.no_grad():
        t0 = time.perf_counter()
        idx_pos, idx_vals, meta3 = compress_sparse_topk(logits, k=64)
        recon3 = decompress_sparse_topk(idx_pos, idx_vals, meta3, logits.shape, logits.dtype)
        t_compress = time.perf_counter() - t0

        c_bytes = idx_pos.numel() * 2 + idx_vals.numel() + 8  # uint16 idx + uint8 vals + meta
        ratio = original_bytes / c_bytes
        acc = top1_accuracy(logits, recon3)
        kl = kl_divergence(logits, recon3)
        pd = perplexity_delta(logits, recon3)
        print(f"  Compressed: {c_bytes/1024:.1f} KB (ratio: {ratio:.2f}x)")
        print(f"  Latency: {t_compress*1000:.2f} ms")
        print(f"  Top-1 accuracy: {acc:.4f}")
        print(f"  KL divergence: {kl:.6f}")
        print(f"  Perplexity delta: {pd:.4f}")
        results.append(('sparse_topk_64', ratio, t_compress, acc, kl, pd))

    # Strategy 4: Sparse top-k (k=128)
    print("\n[4] Sparse top-k (k=128)")
    with torch.no_grad():
        t0 = time.perf_counter()
        idx_pos, idx_vals, meta4 = compress_sparse_topk(logits, k=128)
        recon4 = decompress_sparse_topk(idx_pos, idx_vals, meta4, logits.shape, logits.dtype)
        t_compress = time.perf_counter() - t0

        c_bytes = idx_pos.numel() * 2 + idx_vals.numel() + 8
        ratio = original_bytes / c_bytes
        acc = top1_accuracy(logits, recon4)
        kl = kl_divergence(logits, recon4)
        pd = perplexity_delta(logits, recon4)
        print(f"  Compressed: {c_bytes/1024:.1f} KB (ratio: {ratio:.2f}x)")
        print(f"  Latency: {t_compress*1000:.2f} ms")
        print(f"  Top-1 accuracy: {acc:.4f}")
        print(f"  KL divergence: {kl:.6f}")
        print(f"  Perplexity delta: {pd:.4f}")
        results.append(('sparse_topk_128', ratio, t_compress, acc, kl, pd))

    # Strategy 5: Delta compression (simulate cache)
    print("\n[5] Delta compression (vs cached)")
    with torch.no_grad():
        # Simulate a "cached" version (slightly perturbed)
        cached = logits + torch.randn_like(logits) * 0.1

        t0 = time.perf_counter()
        idx5, d_min, scale5 = compress_delta(logits, cached, n_bits=8)
        recon5 = decompress_delta(idx5, d_min, scale5, cached, logits.dtype)
        t_compress = time.perf_counter() - t0

        c_bytes = logits.numel() + 8 if idx5 is not None else 8
        ratio = original_bytes / c_bytes
        acc = top1_accuracy(logits, recon5)
        kl = kl_divergence(logits, recon5)
        pd = perplexity_delta(logits, recon5)
        print(f"  Compressed: {c_bytes/1024:.1f} KB (ratio: {ratio:.2f}x)")
        print(f"  Latency: {t_compress*1000:.2f} ms")
        print(f"  Top-1 accuracy: {acc:.4f}")
        print(f"  KL divergence: {kl:.6f}")
        print(f"  Perplexity delta: {pd:.4f}")
        results.append(('delta_8bit', ratio, t_compress, acc, kl, pd))

    # Strategy 6: Delta compression (exact cache — should be near-zero)
    print("\n[6] Delta compression (exact cache — best case)")
    with torch.no_grad():
        t0 = time.perf_counter()
        idx6, d_min, scale6 = compress_delta(logits, logits, n_bits=8)
        recon6 = decompress_delta(idx6, d_min, scale6, logits, logits.dtype)
        t_compress = time.perf_counter() - t0

        c_bytes = 8 if idx6 is not None else 8  # constant delta = just meta
        ratio = original_bytes / c_bytes
        acc = top1_accuracy(logits, recon6)
        kl = kl_divergence(logits, recon6)
        pd = perplexity_delta(logits, recon6)
        print(f"  Compressed: {c_bytes/1024:.1f} KB (ratio: {ratio:.2f}x)")
        print(f"  Latency: {t_compress*1000:.2f} ms")
        print(f"  Top-1 accuracy: {acc:.4f}")
        print(f"  KL divergence: {kl:.6f}")
        print(f"  Perplexity delta: {pd:.4f}")
        results.append(('delta_exact', ratio, t_compress, acc, kl, pd))

    # Strategy 7: Mixed — 4-bit quantization
    print("\n[7] Uniform 4-bit per-tensor")
    with torch.no_grad():
        t0 = time.perf_counter()
        l_flat = logits.float().flatten()
        l_min = l_flat.min().item()
        l_max = l_flat.max().item()
        scale4 = (l_max - l_min) / 15.0
        idx4 = ((l_flat - l_min) / scale4).round_().clamp_(0, 15).to(torch.uint8)
        # Pack 2 indices per byte
        idx_packed = (idx4[0::2].to(torch.uint8) << 4) | idx4[1::2].to(torch.uint8)
        t_compress = time.perf_counter() - t0

        # Unpack and dequantize
        idx_up = torch.stack([idx_packed >> 4, idx_packed & 0x0F], dim=-1).flatten()[:logits.numel()]
        recon7 = (idx_up.float() * scale4 + l_min).reshape(logits.shape).to(logits.dtype)
        t_decompress = time.perf_counter() - t0

        c_bytes = math.ceil(logits.numel() / 2) + 8
        ratio = original_bytes / c_bytes
        acc = top1_accuracy(logits, recon7)
        kl = kl_divergence(logits, recon7)
        pd = perplexity_delta(logits, recon7)
        print(f"  Compressed: {c_bytes/1024:.1f} KB (ratio: {ratio:.2f}x)")
        print(f"  Latency: {t_compress*1000:.2f} ms")
        print(f"  Top-1 accuracy: {acc:.4f}")
        print(f"  KL divergence: {kl:.6f}")
        print(f"  Perplexity delta: {pd:.4f}")
        results.append(('uniform4', ratio, t_compress, acc, kl, pd))

    # --- Summary --------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Strategy':<20} {'Ratio':>8} {'Latency':>10} {'Top-1':>8} {'KL':>10} {'PPL d':>8}")
    print("-" * 70)
    for name, ratio, lat, acc, kl, pd in results:
        print(f"{name:<20} {ratio:>7.2f}x {lat*1000:>8.2f}ms {acc:>7.4f} {kl:>9.6f} {pd:>7.4f}")

    print(f"\nOriginal: {original_bytes/1024:.1f} KB ({logits.numel():,} elements)")
    print(f"Vocab: {cfg.vocab}, D: {cfg.D}")

    # --- Analysis for cache use case ------------------------------
    print("\n" + "-" * 70)
    print("CACHE USE CASE ANALYSIS")
    print("-" * 70)

    # Typical cache scenario: B=1, L=1 (single token generation)
    B_cache, L_cache = 1, 1
    logits_cache = logits[:B_cache, :L_cache, :]  # (1, 1, V)
    cache_bytes_fp32 = logits_cache.numel() * 4
    print(f"\nSingle token logits: {logits_cache.numel():,} elements = {cache_bytes_fp32/1024:.1f} KB (fp32)")

    # What's the overhead per step?
    print(f"\nPer-step overhead for different strategies:")
    for name, ratio, lat, acc, kl, pd in results:
        c_size = cache_bytes_fp32 / ratio
        print(f"  {name:<20}: {c_size/1024:.2f} KB, {lat*1000:.2f} ms, top-1={acc:.4f}")

    # --- Feasibility verdict --------------------------------------
    print("\n" + "=" * 70)
    print("FEASIBILITY VERDICT")
    print("=" * 70)

    best_acc = max(r[3] for r in results)
    best_ratio = max(r[1] for r in results)
    fastest = min(r[2] for r in results)

    print(f"\nBest accuracy: {best_acc:.4f}")
    print(f"Best compression: {best_ratio:.2f}x")
    print(f"Fastest: {fastest*1000:.2f} ms")

    if best_acc >= 0.99:
        print("\n[OK] FEASIBLE: Online logits compression is practical.")
        print("   Quality loss is minimal for cache compression use case.")
    elif best_acc >= 0.95:
        print("\n[!]  MARGINAL: Online compression is possible but with quality trade-offs.")
        print("   Consider hybrid approach: compress only for long contexts.")
    else:
        print("\n[X] NOT FEASIBLE: Quality loss too high for online compression.")
        print("   Consider alternative: only cache hidden states, not logits.")


if __name__ == '__main__':
    run_experiment()
