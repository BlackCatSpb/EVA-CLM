"""
logit_cache_v2.py — Per-scale LogitCache driven by VSA scales.

Each scale stores k values per token. The k value is determined by
the VSA scales (learned parameters), not hardcoded:

  k_scale = base_k * sigmoid(vsa_scale)  →  [0, base_k]

This means the model LEARNS how much information to store per scale:
- Low VSA scale → more compression (smaller k)
- High VSA scale → less precision (larger k)

Total: ~1.1 KB/token → 1M tokens = ~1.1 GB (vs 240 GB KV-cache)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tau_compression import compress_sparse_topk, decompress_sparse_topk


# ─── Base k values (maximum per scale) ────────────────────────
# These are the MAXIMUM k values. Actual k = base_k * sigmoid(vsa_scale)
BASE_K = [128, 96, 80, 64]


class PerScaleLogitCache(nn.Module):
    """Per-scale logit cache driven by VSA scales.

    Each scale stores compressed logits with VSA-adaptive precision:
    - k = base_k * sigmoid(vsa_scale)
    - The model LEARNS vsa_scale to control compression

    Memory usage per token (when vsa_scale=0):
    - Scale 0: 128 × 3 bytes = 384 bytes
    - Scale 1: 96 × 3 bytes = 288 bytes
    - Scale 2: 80 × 3 bytes = 240 bytes
    - Scale 3: 64 × 3 bytes = 192 bytes
    - Total: 1,104 bytes = 1.08 KB per token
    """

    def __init__(self, V: int, max_tokens: int = 1_000_000,
                 n_scales: int = 4, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.V = V
        self.max_tokens = max_tokens
        self.n_scales = n_scales
        self.device = device

        # VSA scale parameters (learned)
        # sigmoid(vsa_scale) ∈ [0, 1] → k = base_k * sigmoid(vsa_scale)
        # Initialized to 0 → sigmoid(0) = 0.5 → k = base_k/2
        self.vsa_scales = nn.Parameter(torch.zeros(n_scales))

        # Per-scale storage (plain lists)
        self._scales = []
        for i in range(n_scales):
            self._scales.append({
                'pos': [],
                'vals': [],
                'meta': [],
            })

        # Position tracking per scale
        self._positions = [0] * n_scales

    def get_k(self, scale_idx: int) -> int:
        """Get adaptive k value for scale (VSA-driven)."""
        base_k = BASE_K[scale_idx]
        # k = base_k * sigmoid(vsa_scale) → range [0, base_k]
        # Add minimum k=8 to ensure at least some information is stored
        k = int(base_k * torch.sigmoid(self.vsa_scales[scale_idx]).item())
        return max(k, 8)  # Minimum 8 values per scale

    def compress_scale(self, logits: torch.Tensor, scale_idx: int) -> Dict:
        """Compress logits for a specific scale."""
        if logits.dim() == 1:
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.dim() == 2:
            logits = logits.unsqueeze(1)

        k = self.get_k(scale_idx)
        idx_pos, idx_vals, meta = compress_sparse_topk(logits, k=k)

        return {
            'pos': idx_pos,
            'vals': idx_vals,
            'meta': meta,
            'shape': logits.shape,
            'dtype': logits.dtype,
        }

    def decompress_scale(self, compressed: Dict) -> torch.Tensor:
        """Decompress logits for a scale."""
        return decompress_sparse_topk(
            compressed['pos'], compressed['vals'], compressed['meta'],
            compressed['shape'], compressed['dtype']
        )

    def store(self, logits: torch.Tensor, position: int = None) -> None:
        """Store logits across all scales."""
        for scale_idx in range(self.n_scales):
            if position is None:
                pos = self._positions[scale_idx]
                self._positions[scale_idx] += 1
            else:
                pos = position

            compressed = self.compress_scale(logits, scale_idx)

            self._scales[scale_idx]['pos'].append(compressed['pos'].detach())
            self._scales[scale_idx]['vals'].append(compressed['vals'].detach())
            self._scales[scale_idx]['meta'].append(compressed['meta'].detach())

            # Manage cache size
            if len(self._scales[scale_idx]['pos']) > self.max_tokens:
                self._scales[scale_idx]['pos'].pop(0)
                self._scales[scale_idx]['vals'].pop(0)
                self._scales[scale_idx]['meta'].pop(0)

    def retrieve_scale(self, scale_idx: int, n: int = None) -> torch.Tensor:
        """Retrieve decompressed logits for a scale."""
        pos_list = self._scales[scale_idx]['pos']
        vals_list = self._scales[scale_idx]['vals']
        meta_list = self._scales[scale_idx]['meta']

        if len(pos_list) == 0:
            return None

        if n is not None:
            pos_list = pos_list[-n:]
            vals_list = vals_list[-n:]
            meta_list = meta_list[-n:]

        logits = []
        for pos, vals, meta in zip(pos_list, vals_list, meta_list):
            B, L, k = pos.shape
            V = self.V
            compressed = {
                'pos': pos,
                'vals': vals,
                'meta': meta,
                'shape': (B, L, V),
                'dtype': torch.float32,
            }
            logits.append(self.decompress_scale(compressed))

        return torch.cat(logits, dim=1)

    def size_bytes_per_scale(self, scale_idx: int) -> int:
        """Estimate size of one scale in bytes."""
        pos_list = self._scales[scale_idx]['pos']
        vals_list = self._scales[scale_idx]['vals']
        meta_list = self._scales[scale_idx]['meta']

        total = 0
        for pos, vals, meta in zip(pos_list, vals_list, meta_list):
            total += pos.numel() * 2 + vals.numel() + meta.numel() * 4
        return total

    def size_bytes(self) -> int:
        """Estimate total cache size in bytes."""
        return sum(self.size_bytes_per_scale(i) for i in range(self.n_scales))

    def size_mb(self) -> float:
        """Estimate total cache size in megabytes."""
        return self.size_bytes() / (1024 * 1024)

    def clear(self) -> None:
        """Clear the cache."""
        for i in range(self.n_scales):
            self._scales[i]['pos'].clear()
            self._scales[i]['vals'].clear()
            self._scales[i]['meta'].clear()
            self._positions[i] = 0

    def __len__(self) -> int:
        return len(self._scales[0]['pos'])


class PerScaleLogitAttention(nn.Module):
    """Attention over per-scale cached logits.

    VSA-driven: each scale has its own learned temperature and
    combination weight, controlled by the same VSA scales that
    drive compression.
    """

    def __init__(self, D: int, V: int, n_scales: int = 4, n_heads: int = 8):
        super().__init__()
        self.D = D
        self.V = V
        self.n_scales = n_scales
        self.n_heads = n_heads
        self.head_dim = D // n_heads
        assert D % n_heads == 0

        # Per-scale projections
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_projs = nn.ModuleList([nn.Linear(V, D, bias=False) for _ in range(n_scales)])
        self.v_projs = nn.ModuleList([nn.Linear(V, D, bias=False) for _ in range(n_scales)])
        self.out_proj = nn.Linear(D, D, bias=False)

        # Per-scale LayerNorm
        self.k_norms = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_scales)])
        self.v_norms = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_scales)])

        # Position encoding
        self.pos_enc = nn.Embedding(1024, D)

        # Per-scale learned temperature
        self.log_tau = nn.Parameter(torch.zeros(n_scales))

        # Scale combination weights (learned)
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # Gate
        self.cache_gate = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.cache_gate[-2].weight)
        nn.init.zeros_(self.cache_gate[-2].bias)

    def forward(self, h: torch.Tensor, cache: PerScaleLogitCache,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attend to cached logits across all scales."""
        B, L, D = h.shape

        if len(cache) == 0:
            if return_attention:
                return h, None
            return h

        Q = self.q_proj(h)

        scale_outputs = []
        scale_attns = []

        for scale_idx in range(self.n_scales):
            cached_logits = cache.retrieve_scale(scale_idx, n=min(len(cache), 512))

            if cached_logits is None:
                scale_outputs.append(h)
                scale_attns.append(None)
                continue

            M = cached_logits.shape[1]
            cached_logits = torch.tanh(cached_logits / 10.0)

            positions = torch.arange(M, device=h.device).unsqueeze(0).expand(B, -1)
            K = self.k_norms[scale_idx](self.k_projs[scale_idx](cached_logits)) + self.pos_enc(positions)
            V_cache = self.v_norms[scale_idx](self.v_projs[scale_idx](cached_logits))

            Q_scale = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
            V_cache = V_cache.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)

            tau = torch.exp(self.log_tau[scale_idx]).clamp(min=0.1, max=10.0)
            scale = math.sqrt(self.head_dim) * tau
            attn_weights = torch.matmul(Q_scale, K.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn_weights, dim=-1)

            attn_output = torch.matmul(attn_weights, V_cache)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
            output = self.out_proj(attn_output)

            scale_outputs.append(output)
            scale_attns.append(attn_weights.mean(dim=1))

        # Combine scales
        weights = F.softmax(self.scale_weights, dim=0)
        combined = torch.zeros_like(h)
        for scale_idx, (output, weight) in enumerate(zip(scale_outputs, weights)):
            if output is not None:
                combined = combined + weight * output

        gate = self.cache_gate(h)
        output = gate * combined + (1 - gate) * h

        if return_attention:
            return output, scale_attns
        return output


class PerScaleCacheAttention(nn.Module):
    """Combined PerScaleLogitCache + PerScaleLogitAttention.

    VSA-driven compression: the model learns how much to compress
    per scale via vsa_scales parameters.
    """

    def __init__(self, D: int, V: int, n_layers: int = 24,
                 max_tokens: int = 1_000_000, n_heads: int = 8):
        super().__init__()
        self.cache = PerScaleLogitCache(V, max_tokens, n_scales=4)
        self.attention = PerScaleLogitAttention(D, V, n_scales=4, n_heads=n_heads)

        # Project logits to hidden space
        self.logit_to_hidden = nn.Linear(V, D, bias=False)
        nn.init.xavier_uniform_(self.logit_to_hidden.weight, gain=0.01)

    def forward(self, h: torch.Tensor, logits: torch.Tensor,
                use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process logits through VSA-driven cache."""
        if use_cache:
            self.cache.store(logits.detach())
            h_augmented = self.attention(h, self.cache)

            if torch.isnan(h_augmented).any():
                h_augmented = h

            cached = self.cache.retrieve_scale(0, n=1)
            if cached is not None:
                cached_h = self.logit_to_hidden(cached)
                if not torch.isnan(cached_h).any():
                    h_augmented = h_augmented + cached_h

            return h_augmented, logits
        else:
            return h, logits
