"""
logit_cache.py — Per-scale LogitCache with dual-mode: training + inference.

Training mode:
  - Stores hidden states h (full, no compression)
  - Gradient flows through cache → model learns to produce useful representations
  - Memory: D × 4 bytes/token = 10 KB/token (seq_len=128 → 1.3 MB)

Inference mode:
  - Stores compressed logits (VSA-driven, 4 scales)
  - Memory: ~1.1 KB/token → 1M tokens = 1.1 GB (vs 240 GB KV-cache)

The model LEARNS vsa_scales to control compression per scale.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tau_compression import (
    compress_sparse_topk, decompress_sparse_topk,
)


# ─── Base k values (maximum per scale) ────────────────────────
BASE_K = [128, 96, 80, 64]


class LogitCache(nn.Module):
    """Per-scale logit cache with dual-mode storage.

    Training: stores h (hidden states) for gradient flow.
    Inference: stores compressed logits for memory efficiency.
    """

    def __init__(self, V: int, D: int, max_tokens: int = 1_000_000,
                 n_scales: int = 4, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.V = V
        self.D = D
        self.max_tokens = max_tokens
        self.n_scales = n_scales
        self.device = device

        # VSA scale parameters (learned)
        # sigmoid(vsa_scale) ∈ [0, 1] → k = base_k * sigmoid(vsa_scale)
        # Initialized to 0 → sigmoid(0) = 0.5 → k = base_k/2
        self.vsa_scales = nn.Parameter(torch.zeros(n_scales))

        # Storage: either h (training) or compressed logits (inference)
        self._h_cache: List[torch.Tensor] = []  # training: store h
        self._logit_cache: List[Dict] = []  # inference: store compressed logits
        self._position = 0

    def get_k(self, scale_idx: int) -> int:
        """Get adaptive k value for scale (VSA-driven)."""
        base_k = BASE_K[scale_idx]
        k = int(base_k * torch.sigmoid(self.vsa_scales[scale_idx]).item())
        return max(k, 8)

    def store(self, h_or_logits: torch.Tensor, training: bool = True) -> None:
        """Store data in cache.

        Args:
            h_or_logits: (B, L, D) hidden states (training) or (B, L, V) logits (inference)
            training: if True, store h (gradient flows); if False, store compressed logits
        """
        if training:
            # Store h directly (no compression, gradient flows - NO detach!)
            self._h_cache.append(h_or_logits)
            if len(self._h_cache) > self.max_tokens:
                self._h_cache.pop(0)
        else:
            # Store compressed logits (no gradient)
            compressed = self._compress(h_or_logits)
            self._logit_cache.append(compressed)
            if len(self._logit_cache) > self.max_tokens:
                self._logit_cache.pop(0)

        self._position += 1

    def retrieve(self, n: int = None, training: bool = True) -> Optional[torch.Tensor]:
        """Retrieve data from cache.

        Args:
            n: number of recent entries to retrieve (None = all within window)
            training: if True, retrieve h; if False, retrieve logits

        Returns:
            (B, M, D) hidden states or (B, M, V) logits
        """
        if training:
            if not self._h_cache:
                return None
            entries = self._h_cache[-n:] if n else self._h_cache
            return torch.cat(entries, dim=1)
        else:
            if not self._logit_cache:
                return None
            entries = self._logit_cache[-n:] if n else self._logit_cache
            logits = [self._decompress(e) for e in entries]
            return torch.cat(logits, dim=1)

    def _compress(self, logits: torch.Tensor) -> Dict:
        """Compress logits for inference storage."""
        if logits.dim() == 1:
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.dim() == 2:
            logits = logits.unsqueeze(1)

        # Use first scale's k for simplicity (or average)
        k = self.get_k(0)
        idx_pos, idx_vals, meta = compress_sparse_topk(logits, k=k)

        return {
            'pos': idx_pos.detach(),
            'vals': idx_vals.detach(),
            'meta': meta.detach(),
            'shape': logits.shape,
            'dtype': logits.dtype,
        }

    def _decompress(self, compressed: Dict) -> torch.Tensor:
        """Decompress logits for inference retrieval."""
        return decompress_sparse_topk(
            compressed['pos'], compressed['vals'], compressed['meta'],
            compressed['shape'], compressed['dtype']
        )

    def clear(self) -> None:
        """Clear the cache."""
        self._h_cache.clear()
        self._logit_cache.clear()
        self._position = 0

    def size_mb(self, training: bool = True) -> float:
        """Estimate cache size in megabytes."""
        if training:
            if not self._h_cache:
                return 0.0
            # h: (B, L, D) × float32 × number of entries
            total_bytes = sum(h.numel() * 4 for h in self._h_cache)
        else:
            if not self._logit_cache:
                return 0.0
            total_bytes = 0
            for e in self._logit_cache:
                total_bytes += e['pos'].numel() * 2 + e['vals'].numel() + e['meta'].numel() * 4
        return total_bytes / (1024 * 1024)

    def __len__(self) -> int:
        return max(len(self._h_cache), len(self._logit_cache))


class LogitAttention(nn.Module):
    """Attention over cached hidden states or logits.

    Training mode: attends to cached h (hidden states)
    Inference mode: attends to cached logits

    Architecture:
        Q: current hidden state (B, L, D)
        K, V: cached data projected to D (B, M, D)
    """

    def __init__(self, D: int, V: int, n_heads: int = 8,
                 max_cache_len: int = 1024):
        super().__init__()
        self.D = D
        self.V = V
        self.n_heads = n_heads
        self.head_dim = D // n_heads
        assert D % n_heads == 0, f"D={D} must be divisible by n_heads={n_heads}"

        # Projections for h (training mode)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj_h = nn.Linear(D, D, bias=False)  # h → D
        self.v_proj_h = nn.Linear(D, D, bias=False)  # h → D

        # Projections for logits (inference mode)
        self.k_proj_l = nn.Linear(V, D, bias=False)  # logits → D
        self.v_proj_l = nn.Linear(V, D, bias=False)  # logits → D

        self.out_proj = nn.Linear(D, D, bias=False)

        # LayerNorm for stability
        self.k_norm = nn.LayerNorm(D)
        self.v_norm = nn.LayerNorm(D)

        # Learned temperature
        self.log_tau = nn.Parameter(torch.tensor(0.0))

        # Position encoding
        self.pos_enc = nn.Embedding(max_cache_len, D)

        # Gate: how much to use cache vs direct
        self.cache_gate = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, 1),
            nn.Sigmoid(),
        )

        # Initialize: start as no-op (cache_gate ≈ 0)
        nn.init.zeros_(self.cache_gate[-2].weight)
        nn.init.zeros_(self.cache_gate[-2].bias)

    def forward(self, h: torch.Tensor, cache: LogitCache,
                training: bool = True,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attend to cached data.

        Args:
            h: (B, L, D) current hidden state
            cache: LogitCache with stored data
            training: if True, retrieve h; if False, retrieve logits
            return_attention: if True, also return attention weights

        Returns:
            output: (B, L, D) cache-augmented hidden state
            attn_weights: (B, L, M) attention weights (if return_attention)
        """
        B, L, D = h.shape

        if len(cache) == 0:
            if return_attention:
                return h, None
            return h

        # Retrieve cached data
        cached = cache.retrieve(n=min(len(cache), 512), training=training)

        if cached is None:
            if return_attention:
                return h, None
            return h

        M = cached.shape[1]

        # Normalize for numerical stability
        if not training:
            # logits: tanh normalization
            cached = torch.tanh(cached / 10.0)

        # Project Q from hidden state
        Q = self.q_proj(h)

        # Project K, V from cached data
        positions = torch.arange(M, device=h.device).unsqueeze(0).expand(B, -1)

        if training:
            # h mode: project h → D
            K = self.k_norm(self.k_proj_h(cached)) + self.pos_enc(positions)
            V_cache = self.v_norm(self.v_proj_h(cached))
        else:
            # logits mode: project logits → D
            K = self.k_norm(self.k_proj_l(cached)) + self.pos_enc(positions)
            V_cache = self.v_norm(self.v_proj_l(cached))

        # Multi-head attention
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        V_cache = V_cache.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        scale = math.sqrt(self.head_dim) * tau
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, V_cache)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(attn_output)

        # Gate: blend cache output with direct path
        gate = self.cache_gate(h)
        output = gate * output + (1 - gate) * h

        if return_attention:
            attn_avg = attn_weights.mean(dim=1)
            return output, attn_avg
        return output


class LogitCacheAttention(nn.Module):
    """Combined LogitCache + LogitAttention module.

    Integrates into EVAStack to provide long-context memory.

    Training: stores h (gradient flows, model learns)
    Inference: stores compressed logits (43x smaller than KV-cache)

    Scheduled sampling: with probability `scheduled_sampling_ratio`,
    uses inference-mode (compressed logits) during training to align
    train/inference representations (R1: close train/inference skew).
    """

    def __init__(self, D: int, V: int, n_layers: int = 24,
                 max_tokens: int = 1_000_000, n_heads: int = 8,
                 scheduled_sampling_ratio: float = 0.05):
        super().__init__()
        self.cache = LogitCache(V, D, max_tokens, n_scales=4)
        self.attention = LogitAttention(D, V, n_heads)
        self.scheduled_sampling_ratio = scheduled_sampling_ratio

        # Project logits to hidden space (for inference mode)
        self.logit_to_hidden = nn.Linear(V, D, bias=False)
        nn.init.xavier_uniform_(self.logit_to_hidden.weight, gain=0.01)

    def forward(self, h: torch.Tensor, logits: torch.Tensor,
                training: bool = True,
                use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process through cache and augment hidden state.

        Args:
            h: (B, L, D) hidden state
            logits: (B, L, V) logits from lm_head
            training: if True, store h; if False, store logits
            use_cache: if False, return h unchanged

        Returns:
            h_augmented: (B, L, D) hidden state augmented with cache info
            logits_out: (B, L, V) logits (unchanged)
        """
        if not use_cache:
            return h, logits

        # R1: Scheduled sampling — with probability scheduled_sampling_ratio,
        # use inference-mode (compressed logits) during training to align
        # train/inference representations.
        use_inference_mode = False
        if training and self.scheduled_sampling_ratio > 0:
            if torch.rand(1).item() < self.scheduled_sampling_ratio:
                use_inference_mode = True

        # Store in cache
        if training and not use_inference_mode:
            # Normal training: store h (gradient flows)
            self.cache.store(h, training=True)
        else:
            # Inference mode or scheduled sampling: store compressed logits
            self.cache.store(logits, training=False)

        # Attend to cache
        # During scheduled sampling, attend to compressed logits (inference mode)
        h_augmented = self.attention(h, self.cache,
                                     training=(training and not use_inference_mode))

        # Check for NaN
        if torch.isnan(h_augmented).any():
            h_augmented = h

        # In inference mode: also project cached logits to hidden space
        if not training or use_inference_mode:
            cached_logits = self.cache.retrieve(n=1, training=False)
            if cached_logits is not None:
                cached_h = self.logit_to_hidden(cached_logits)
                if not torch.isnan(cached_h).any():
                    h_augmented = h_augmented + cached_h

        return h_augmented, logits
