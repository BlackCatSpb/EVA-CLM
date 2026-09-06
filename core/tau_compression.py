"""
tau_compression.py — tau-adaptive logits compression for EVA-CLM.

Compression strategy is modulated by tau_norm and maturation:
  - shallow layers (low tau): delta-8bit (4x, high fidelity)
  - deep layers (high tau): sparse top-k (683-1365x, extreme compression)

No retraining needed — purely post-hoc compression of model outputs.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─── Compression primitives ──────────────────────────────────────

def compress_uniform8(t: torch.Tensor) -> Tuple[Optional[torch.Tensor], float, float]:
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


def compress_sparse_topk(t: torch.Tensor, k: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    scale = (d_max - d_min) / (2 ** n_bits - 1)
    idx = ((delta - d_min) / scale).round_().clamp_(0, 2 ** n_bits - 1).to(torch.uint8)
    return idx, d_min, scale


def decompress_delta(idx, d_min, scale, cached, dtype):
    if idx is None:
        return cached.to(dtype)
    return cached.to(dtype) + (idx.float() * scale + d_min).to(dtype)


# ─── tau-adaptive compression schedule ─────────────────────────────

def tau_to_compression(tau_norm: float, maturation: float = 1.0) -> Dict:
    """Map tau_norm + maturation to compression parameters.

    Returns: {strategy, n_bits, topk, description}
    """
    stability = 0.6 * tau_norm + 0.4 * maturation

    if stability < 0.3:
        return {'strategy': 'uniform8', 'n_bits': 8, 'topk': None,
                'description': 'uniform-8bit (volatile)'}
    elif stability < 0.6:
        return {'strategy': 'sparse_topk', 'n_bits': 8, 'topk': 128,
                'description': 'sparse-topk-128 (medium)'}
    else:
        return {'strategy': 'sparse_topk', 'n_bits': 8, 'topk': 64,
                'description': 'sparse-topk-64 (stable)'}


# ─── tau-adaptive logits cache ────────────────────────────────────

class TauAdaptiveLogitsCache:
    """Per-layer logits cache with tau-modulated compression.

    Usage:
        cache = TauAdaptiveLogitsCache(B=1, V=65536, n_layers=24, device='cuda',
                                        tau_norm=tau_norm, maturation=mat_gate)

        # At each generation step:
        for i, layer in enumerate(model.layers):
            layer_logits = lm_head(layer_out)
            decompressed = cache.step(i, layer_logits)

        # Stats
        print(cache.stats())
    """

    def __init__(self, B: int, V: int, n_layers: int, device: torch.device,
                 tau_norm: Optional[torch.Tensor] = None,
                 maturation: Optional[torch.Tensor] = None):
        self.B = B
        self.V = V
        self.n_layers = n_layers
        self.device = device

        self.tau_norm = (tau_norm if tau_norm is not None
                         else torch.linspace(0, 1, n_layers)).cpu()
        self.maturation = (maturation if maturation is not None
                           else torch.ones(n_layers)).cpu()

        # Per-layer compressed cache
        self._cache: List[Optional[Dict]] = [None] * n_layers

        # Compression schedule per layer
        self._schedules = [
            tau_to_compression(self.tau_norm[i].item(), self.maturation[i].item())
            for i in range(n_layers)
        ]

        # Stats
        self._total_orig = 0
        self._total_comp = 0
        self._n_steps = 0

    def update_tau(self, tau_norm: torch.Tensor, maturation: torch.Tensor):
        """Update per-layer tau and maturation (call once per forward)."""
        self.tau_norm = tau_norm.detach().cpu()
        self.maturation = maturation.detach().cpu()
        for i in range(self.n_layers):
            self._schedules[i] = tau_to_compression(
                self.tau_norm[i].item(), self.maturation[i].item())

    def step(self, layer_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """Compress logits, store in cache, return decompressed version.

        Accepts logits of shape (V,), (1, V), or (B, L, V).
        Returns tensor of same shape as input (decompressed).
        """
        # Normalize to (1, 1, V) for compression
        orig_shape = logits.shape
        if logits.dim() == 1:
            logits_3d = logits.unsqueeze(0).unsqueeze(0)
        elif logits.dim() == 2:
            logits_3d = logits.unsqueeze(1)
        else:
            logits_3d = logits

        sched = self._schedules[layer_idx]
        B, L, V = logits_3d.shape

        compressed = self._compress(layer_idx, logits_3d, sched)
        decompressed = self._decompress(layer_idx, compressed, logits_3d.shape, logits_3d.dtype)

        # Stats
        orig_bytes = logits_3d.numel() * 4
        comp_bytes = self._compressed_bytes(compressed)
        self._total_orig += orig_bytes
        self._total_comp += comp_bytes
        self._n_steps += 1

        self._cache[layer_idx] = compressed

        # Restore original shape
        return decompressed.reshape(orig_shape)

    def _compress(self, layer_idx: int, logits: torch.Tensor, sched: Dict) -> Dict:
        strategy = sched['strategy']

        if strategy == 'uniform8':
            idx, t_min, scale = compress_uniform8(logits)
            return {'type': 'u8', 'idx': idx, 'min': t_min, 'scale': scale,
                    'shape': logits.shape, 'dtype': logits.dtype}

        elif strategy == 'sparse_topk':
            k = sched['topk']
            idx_pos, idx_vals, meta = compress_sparse_topk(logits, k=k)
            return {'type': 'sp', 'pos': idx_pos, 'vals': idx_vals, 'meta': meta,
                    'shape': logits.shape, 'dtype': logits.dtype}

        else:
            # Fallback: uniform8
            idx, t_min, scale = compress_uniform8(logits)
            return {'type': 'u8', 'idx': idx, 'min': t_min, 'scale': scale,
                    'shape': logits.shape, 'dtype': logits.dtype}

    def _decompress(self, layer_idx: int, compressed: Dict,
                     shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        if compressed['type'] == 'u8':
            return decompress_uniform8(compressed['idx'], compressed['min'],
                                        compressed['scale'], shape, dtype)
        elif compressed['type'] == 'sp':
            return decompress_sparse_topk(compressed['pos'], compressed['vals'],
                                           compressed['meta'], shape, dtype)
        return torch.zeros(shape, dtype=dtype)

    def _compressed_bytes(self, c: Dict) -> int:
        if c['type'] == 'u8':
            return c['idx'].numel() + 8 if c['idx'] is not None else 8
        elif c['type'] == 'sp':
            return c['pos'].numel() * 2 + c['vals'].numel() + 8
        return 0

    def stats(self) -> Dict:
        ratio = self._total_orig / max(self._total_comp, 1)
        per_step_orig = self._total_orig / max(self._n_steps, 1)
        per_step_comp = self._total_comp / max(self._n_steps, 1)

        strategy_counts = {}
        for s in self._schedules:
            name = s['strategy']
            strategy_counts[name] = strategy_counts.get(name, 0) + 1

        return {
            'compression_ratio': ratio,
            'per_step_original_kb': per_step_orig / 1024,
            'per_step_compressed_kb': per_step_comp / 1024,
            'n_steps': self._n_steps,
            'strategies': strategy_counts,
            'tau_norm_range': (self.tau_norm.min().item(), self.tau_norm.max().item()),
        }

    def schedule_str(self) -> str:
        lines = []
        for i in range(self.n_layers):
            s = self._schedules[i]
            lines.append(f"  L{i:>2}: tau={self.tau_norm[i].item():.3f} "
                        f"mat={self.maturation[i].item():.3f} -> {s['description']}")
        return '\n'.join(lines)
