"""Streaming Memory Bank for EVA — hierarchical L1 + L2 + L3.

Architecture:
  L1 (immediate): rolling buffer of last K sentence embeddings
    - Simple ring buffer: overwrite oldest
    - No learning, just storage + attention read
    - Always active, no maturation gating

  L2 (learned): VSA-mediated memory bank with N slots
    - Write: at sentence boundaries (SEP token = 2)
    - Read: attention over slots at each token
    - Selective: novelty-based write gating
    - Simple ring buffer: overwrite oldest (or consumed slot from L3)
    - Differentiable: gradients flow through write/read
    - Keys normalized via F.normalize (sigmoid-weighted)

  L3 (concepts): emergent from L2 slot clustering
    - Cluster L2 keys by cosine similarity
    - Concept birth: when cluster confidence > threshold
    - Concept update: running mean of cluster members (sigmoid-weighted)
    - Read: attention over concepts (higher-level abstractions)
    - Long-range memory (tau ~ 500+)
    - CONSUMES L2 slots: when concept born, source L2 slot marked for overwrite

Integration:
  - Sits AFTER embedding, BEFORE first layer
  - Write at sentence boundaries (token == 2)
  - Read at every token position
  - NOT gated by maturation (always active)
  - Maturation controls depth of processing, not memory access

Flow:
  L1.write(summary)  -> overwrite oldest (fast)
  L2.write(summary)  -> overwrite oldest or consumed (fast)
  L3.write(l2_key)   -> if birth/update, mark L2 slot as consumed

Design principles from EVA:
  - Softmax-free (sigmoid attention) — regime B
  - Bridge dim matches bridge_dim config
  - Compatible with gradient checkpointing
  - Persistent buffers for inference
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_gate import hybrid_gate


def _memory_attention(q: torch.Tensor, k: torch.Tensor, temp: torch.Tensor,
                      bridge_dim: int, softmax_free: bool = True,
                      age_decay: torch.Tensor = None) -> torch.Tensor:
    """Compute attention weights for memory bank read using HYBRID approach.

    Hybrid: gate = sigmoid(scores) * (1 + softmax(scores / tau))

    Args:
        q: (B, L, bridge_dim) — query
        k: (n_slots, bridge_dim) — keys
        temp: scalar tensor — temperature (tau)
        bridge_dim: int — dimension for scaling
        softmax_free: bool — if True, use hybrid; if False, use softmax only
        age_decay: (n_slots,) optional — age-based decay for slots

    Returns:
        attn: (B, L, n_slots) — attention weights (sums to 1 per position)
    """
    scores = (q @ k.T) / math.sqrt(bridge_dim) * temp

    if softmax_free:
        attn = hybrid_gate(scores, temp)
    else:
        attn = F.softmax(scores, dim=-1)

    if age_decay is not None:
        attn = attn * age_decay.unsqueeze(0).unsqueeze(0)

    attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    attn = attn / attn_sum

    return attn


class L1Buffer(nn.Module):
    """Rolling buffer of last K sentence embeddings.

    Simple ring buffer: overwrite oldest slot.
    Fast, no cosine similarity checks.
    Keys normalized via F.normalize for consistent attention with L2/L3.
    Uses hybrid attention (sigmoid * (1 + softmax/tau)).
    """
    def __init__(self, D: int, bridge_dim: int, n_slots: int = 3,
                 softmax_free: bool = True, tau_prior: float = 0.5):
        super().__init__()
        self.D = D
        self.bridge_dim = bridge_dim
        self.n_slots = n_slots
        self._softmax_free = softmax_free

        # Project stored embeddings to bridge space (for keys)
        self.proj = nn.Linear(D, bridge_dim)
        # Query projection for attention read
        self.q_proj = nn.Linear(D, bridge_dim)
        # Output projection: buf is (n_slots, D), read output is (B, L, D)
        self.out_proj = nn.Linear(D, D)
        # Learnable temperature for attention (tau)
        # P0 FIX: initialize from τ-prior instead of frozen=1.0
        # L1 = fastest: low tau → more precision
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau_prior, 0.1))))
        self._init_log_tau = self.log_tau.data.clone()  # prior for regularization

        # Persistent buffer: (n_slots, D)
        self.register_buffer('buf', torch.zeros(n_slots, D), persistent=True)
        self.register_buffer('buf_age', torch.zeros(n_slots), persistent=True)
        self.register_buffer('_write_idx', torch.zeros(1, dtype=torch.long), persistent=True)
        self._n_overwrites = 0

    @torch.no_grad()
    def write(self, embedding: torch.Tensor) -> None:
        """Write sentence embedding to buffer. Overwrites oldest slot."""
        n_filled = min(self._write_idx.item(), self.n_slots)

        if n_filled < self.n_slots:
            slot = n_filled  # fill empty slots first
        else:
            slot = int(self.buf_age.argmax().item())  # oldest slot

        self.buf.data[slot] = embedding.detach().float()
        self.buf_age.data[slot] = 0.0
        # Age all other slots
        mask = torch.arange(self.n_slots, device=self.buf_age.device) != slot
        self.buf_age.data[mask] += 1.0
        self._n_overwrites += 1
        self._write_idx += 1

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from buffer using hybrid attention.

        query: (B, L, D) — current hidden state
        returns: (B, L, D) — memory read output
        """
        B, L, _ = query.shape
        q = self.q_proj(query)  # (B, L, bridge_dim)
        k = F.normalize(self.proj(self.buf), dim=-1)  # (n_slots, bridge_dim) — normalized!
        v = self.out_proj(self.buf)  # (n_slots, D)

        temp = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        age_decay = torch.exp(-0.01 * self.buf_age)  # age-based decay
        attn = _memory_attention(q, k, temp, self.bridge_dim, 
                                 self._softmax_free, age_decay)  # (B, L, n_slots)

        read = (attn @ v)  # (B, L, D)
        return read

    def get_stats(self) -> dict:
        return {
            'n_overwrites': self._n_overwrites,
            'fill_rate': min(self._write_idx.item(), self.n_slots) / self.n_slots,
        }

    def reset(self) -> None:
        self.buf.zero_()
        self.buf_age.zero_()
        self._write_idx.zero_()
        self._n_overwrites = 0


class L2Bank(nn.Module):
    """Learned memory bank with N slots.

    Simple ring buffer: overwrite oldest or consumed slot.
    Fast, no cosine similarity checks.
    Consumed slots (from L3 concept birth) are prioritized for overwrite.
    
    Keys normalized via F.normalize + tau-based scaling (hybrid approach).
    Values scaled via tau-based sigmoid (preserves magnitude info).
    Uses hybrid attention (sigmoid * (1 + softmax/tau)).
    """
    def __init__(self, D: int, bridge_dim: int, n_slots: int = 16,
                 softmax_free: bool = True, tau_prior: float = 1.0):
        super().__init__()
        self.D = D
        self.bridge_dim = bridge_dim
        self.n_slots = n_slots
        self._softmax_free = softmax_free

        self.W_k = nn.Linear(D, bridge_dim)
        self.W_v = nn.Linear(D, bridge_dim)
        self.W_o = nn.Linear(bridge_dim, D)

        self.q_proj = nn.Linear(D, bridge_dim)

        # LayerNorm for vals to prevent magnitude explosion
        self.val_norm = nn.LayerNorm(bridge_dim)

        self.keys = nn.Parameter(torch.randn(n_slots, bridge_dim) * 0.02)
        self.vals = nn.Parameter(torch.randn(n_slots, bridge_dim) * 0.02)

        self.novelty_gate = nn.Sequential(
            nn.Linear(D, bridge_dim),
            nn.GELU(),
            nn.Linear(bridge_dim, 1),
        )

        # P0 FIX: initialize from τ-prior instead of frozen=1.0
        # L2 = medium: balanced tau
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau_prior, 0.1))))
        self._init_log_tau = self.log_tau.data.clone()  # prior for regularization
        
        # Tau-based scaling for keys/vals (hybrid approach)
        # Keys: F.normalize + sigmoid(tau) for stable cosine similarity
        # Vals: sigmoid(tau) for bounded magnitude preservation
        self.key_log_scale = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self.val_log_scale = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        self.register_buffer('slot_age', torch.zeros(n_slots), persistent=True)
        self.register_buffer('slot_novelty', torch.ones(n_slots), persistent=True)
        self.register_buffer('slot_consumed', torch.zeros(n_slots, dtype=torch.bool), persistent=True)
        self.register_buffer('_write_idx', torch.zeros(1, dtype=torch.long), persistent=True)
        self._n_overwrites = 0
        self._n_consumed = 0

    @torch.no_grad()
    def write(self, embedding: torch.Tensor) -> int:
        """Write to bank. Returns slot index that was written.

        Prioritizes overwriting consumed slots (from L3 concept birth).
        Falls back to overwriting oldest slot.
        """
        novelty_score = torch.sigmoid(self.novelty_gate(embedding))

        n_filled = min(self._write_idx.item(), self.n_slots)

        with torch.no_grad():
            # Hybrid normalization: F.normalize + tau-based scaling
            # Keys: normalized for stable cosine similarity in L3
            raw_key = self.W_k(embedding.detach())
            new_key = F.normalize(raw_key, dim=-1) * torch.sigmoid(self.key_log_scale)
            
            # Vals: normalized + tau-scaled (matches key normalization for stable attention)
            raw_val = self.W_v(embedding.detach())
            new_val = F.normalize(raw_val, dim=-1) * torch.sigmoid(self.val_log_scale)

        if n_filled < self.n_slots:
            # Fill empty slot
            slot = n_filled
        else:
            # Prioritize overwriting consumed slots
            consumed_mask = self.slot_consumed
            if consumed_mask.any():
                # Pick oldest consumed slot
                consumed_ages = self.slot_age.clone()
                consumed_ages[~consumed_mask] = -1  # ignore non-consumed
                slot = int(consumed_ages.argmax().item())
                self._n_consumed += 1
            else:
                # Overwrite oldest slot
                slot = int(self.slot_age.argmax().item())

        self.keys.data[slot] = new_key
        self.vals.data[slot] = new_val
        self.slot_age.data[slot] = 0.0
        self.slot_novelty.data[slot] = novelty_score.item()
        self.slot_consumed.data[slot] = False  # clear consumed flag
        mask = torch.arange(self.n_slots, device=self.slot_age.device) != slot
        self.slot_age.data[mask] += 1.0
        self._n_overwrites += 1
        self._write_idx += 1
        return slot

    @torch.no_grad()
    def mark_consumed(self, slot: int) -> None:
        """Mark slot as consumed by L3 concept birth."""
        if 0 <= slot < self.n_slots:
            self.slot_consumed.data[slot] = True

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from bank using hybrid attention.

        query: (B, L, D)
        returns: (B, L, D)
        """
        B, L, _ = query.shape
        q = self.q_proj(query)  # (B, L, bridge_dim)
        k = self.keys  # (n_slots, bridge_dim)
        v = self.val_norm(self.vals)  # (n_slots, bridge_dim) — normalized for stability

        temp = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        age_decay = torch.exp(-0.01 * self.slot_age)
        attn = _memory_attention(q, k, temp, self.bridge_dim,
                                 self._softmax_free, age_decay)  # (B, L, n_slots)

        read = attn @ v  # (B, L, bridge_dim)
        return self.W_o(read)  # (B, L, D)

    def get_stats(self) -> dict:
        return {
            'n_overwrites': self._n_overwrites,
            'n_consumed': self._n_consumed,
            'fill_rate': min(self._write_idx.item(), self.n_slots) / self.n_slots,
            'novelty_mean': self.slot_novelty.mean().item(),
            'consumed_count': int(self.slot_consumed.sum().item()),
            'key_scale': torch.sigmoid(self.key_log_scale).item(),
            'val_scale': torch.sigmoid(self.val_log_scale).item(),
        }

    def reset(self) -> None:
        self.keys.data.zero_()
        self.vals.data.zero_()
        self.slot_age.zero_()
        self.slot_novelty.zero_()
        self.slot_consumed.zero_()
        self._write_idx.zero_()
        self._n_overwrites = 0
        self._n_consumed = 0


class L3Concepts(nn.Module):
    """Emergent concept layer — clusters L2 slots into higher-level abstractions.

    Mechanism:
      - At each boundary write, check L2 key against existing concepts
      - If cosine similarity > threshold: update concept (running mean)
      - If no match and confidence > threshold: birth new concept
      - Read: attention over concepts (higher-level than L2)

    Concept birth is triggered by:
      1. L2 key doesn't match any existing concept
      2. The new key is confident enough (from L2 novelty gate)
      3. An empty slot is available (or least-used concept is evicted)

    Concept death: unused concepts decay via age and get evicted.

    CONSUMES L2 slots: when concept is born/updated, the source L2 slot
    is marked as consumed (for cleanup in L2).
    
    Values normalized via F.normalize + tau-based scaling (hybrid approach).
    """
    def __init__(self, D: int, bridge_dim: int, n_concepts: int = 8,
                 birth_threshold: float = 0.7, update_momentum: float = 0.1,
                 softmax_free: bool = True, tau_prior: float = 2.0):
        super().__init__()
        self.D = D
        self.bridge_dim = bridge_dim
        self.n_concepts = n_concepts
        self._birth_threshold = birth_threshold
        self._update_momentum = update_momentum
        self._softmax_free = softmax_free

        # Concept keys/values (learnable)
        self.concept_keys = nn.Parameter(torch.randn(n_concepts, bridge_dim) * 0.02)
        self.concept_vals = nn.Parameter(torch.randn(n_concepts, bridge_dim) * 0.02)

        # Query projection for read
        self.q_proj = nn.Linear(D, bridge_dim)
        self.out_proj = nn.Linear(bridge_dim, D)

        # Temperature (tau)
        # P0 FIX: initialize from τ-prior instead of frozen=1.0
        # L3 = slowest: high tau → more diversity (broader attention over concepts)
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau_prior, 0.1))))
        self._init_log_tau = self.log_tau.data.clone()  # prior for regularization
        
        # Tau-based scaling for concept values (hybrid approach)
        # Vals: F.normalize + sigmoid(tau) for stable representation
        self.val_log_scale = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        # Tracking
        self.register_buffer('concept_age', torch.zeros(n_concepts), persistent=True)
        self.register_buffer('concept_count', torch.zeros(n_concepts), persistent=True)
        self.register_buffer('concept_confidence', torch.zeros(n_concepts), persistent=True)
        self._n_births = 0
        self._n_updates = 0

    @torch.no_grad()
    def write(self, l2_key: torch.Tensor, l2_val: torch.Tensor = None, confidence: float = 0.5) -> bool:
        """Try to write L2 key into a concept slot.

        l2_key: (bridge_dim,) — the L2 key that was written
        l2_val: (bridge_dim,) — the L2 value that was written (optional, defaults to l2_key)
        confidence: float — novelty confidence from L2 gate
        returns: True if wrote (updated or birthed)
        """
        # Use l2_val if provided, otherwise fallback to l2_key (backward compat)
        if l2_val is None:
            l2_val = l2_key
            
        # Normalize for cosine similarity
        key_n = F.normalize(l2_key.unsqueeze(0), dim=-1)  # (1, bridge_dim)
        concept_n = F.normalize(self.concept_keys.data, dim=-1)  # (n_concepts, bridge_dim)

        # Cosine similarity to existing concepts
        sims = (key_n @ concept_n.T).squeeze(0)  # (n_concepts,)

        # Find best match
        best_sim, best_idx = sims.max(0)
        best_idx = best_idx.item()
        best_sim = best_sim.item()

        # Update existing concept if similarity > threshold
        if best_sim > self._birth_threshold:
            alpha = self._update_momentum
            self.concept_keys.data[best_idx] = F.normalize(
                self.concept_keys.data[best_idx] * (1 - alpha) + l2_key * alpha, dim=-1)
            # Hybrid normalization for concept_vals: F.normalize + tau-based scaling
            raw_val = self.concept_vals.data[best_idx] * (1 - alpha) + l2_val * alpha
            self.concept_vals.data[best_idx] = F.normalize(raw_val, dim=-1) * torch.sigmoid(self.val_log_scale)
            self.concept_count.data[best_idx] += 1
            self.concept_confidence.data[best_idx] = (
                self.concept_confidence.data[best_idx] * 0.9 + best_sim * 0.1)
            self.concept_age.data[best_idx] = 0.0
            self._n_updates += 1
            return True

        # Birth: find empty slot or evict least-used concept
        empty = torch.nonzero(self.concept_count == 0)
        if empty.numel() > 0:
            idx = empty[0].item()
        elif confidence > self._birth_threshold:
            # Evict concept with lowest confidence * count (least established)
            utility = self.concept_confidence * torch.clamp(self.concept_count, min=1)
            idx = utility.argmin().item()
        else:
            return False

        # Birth or overwrite
        self.concept_keys.data[idx] = F.normalize(l2_key.unsqueeze(0), dim=-1).squeeze(0)
        # Hybrid normalization for concept_vals: use l2_val (not l2_key!)
        self.concept_vals.data[idx] = F.normalize(l2_val.unsqueeze(0), dim=-1).squeeze(0) * torch.sigmoid(self.val_log_scale)
        self.concept_age.data[idx] = 0.0
        self.concept_count.data[idx] = 1
        self.concept_confidence.data[idx] = confidence
        self._n_births += 1
        return True

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from concepts using attention.

        query: (B, L, D)
        returns: (B, L, D)
        """
        B, L, _ = query.shape
        q = self.q_proj(query)  # (B, L, bridge_dim)
        k = self.concept_keys  # (n_concepts, bridge_dim)
        v = self.concept_vals  # (n_concepts, bridge_dim)

        temp = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        age_decay = torch.exp(-0.005 * self.concept_age)  # slower decay than L2 (concepts are long-range)
        attn = _memory_attention(q, k, temp, self.bridge_dim,
                                 self._softmax_free, age_decay)  # (B, L, n_concepts)

        read = attn @ v  # (B, L, bridge_dim)
        return self.out_proj(read)  # (B, L, D)

    def get_active_concepts(self) -> int:
        """Number of concepts with count > 0."""
        return int((self.concept_count > 0).sum().item())

    def get_stats(self) -> dict:
        return {
            'n_births': self._n_births,
            'n_updates': self._n_updates,
            'n_active': self.get_active_concepts(),
            'confidence_mean': self.concept_confidence.mean().item(),
            'val_scale': torch.sigmoid(self.val_log_scale).item(),
        }

    def reset(self) -> None:
        self.concept_keys.data.zero_()
        self.concept_vals.data.zero_()
        self.concept_age.zero_()
        self.concept_count.zero_()
        self.concept_confidence.zero_()
        self._n_births = 0
        self._n_updates = 0


class StreamingMemoryBank(nn.Module):
    """Combined L1 + L2 + L3 memory bank for EVA.

    Flow:
      L1.write(summary)  -> overwrite oldest (fast)
      L2.write(summary)  -> overwrite oldest or consumed (fast)
      L3.write(l2_key)   -> if birth/update, mark L2 slot as consumed

    Integration points:
    - forward(h, tokens, step, mat_gate): read from L1+L2+L3 at each position
    - write_boundary(embedding): called at sentence boundaries
    - reset(): clear all memory (for new sequence)
    """
    def __init__(self, D: int, bridge_dim: int,
                 l1_slots: int = 3, l2_slots: int = 16,
                 l3_concepts: int = 8, l3_birth_threshold: float = 0.7,
                 min_write_maturation: float = 0.3,
                 softmax_free: bool = True,
                 cfg=None,
                 tau_config=None):
        super().__init__()
        self.D = D
        self.bridge_dim = bridge_dim
        self.cfg = cfg
        self._min_write_maturation = min_write_maturation
        self._softmax_free = softmax_free
        self.tau_config = tau_config

        # Compute τ-priors from tau_config if available
        if tau_config is not None:
            mem_tau = tau_config.mem_tau  # (3,) — [L1_tau, L2_tau, L3_tau]
            # Normalize to reasonable range for hybrid_gate temperature
            l1_tau_prior = (mem_tau[0] / tau_config.mem_tau_ref).clamp(0.1, 5.0).item()
            l2_tau_prior = (mem_tau[1] / tau_config.mem_tau_ref).clamp(0.1, 5.0).item()
            l3_tau_prior = (mem_tau[2] / tau_config.mem_tau_ref).clamp(0.1, 5.0).item()
        else:
            l1_tau_prior = 0.5  # L1 = fast (low tau → precision)
            l2_tau_prior = 1.0  # L2 = balanced
            l3_tau_prior = 2.0  # L3 = slow (high tau → diversity)

        # L1: rolling buffer (immediate, ~last K diverse sentences)
        self.l1 = L1Buffer(D, bridge_dim, n_slots=l1_slots, softmax_free=softmax_free,
                           tau_prior=l1_tau_prior)

        # L2: learned bank (short-term, ~N diverse slots)
        self.l2 = L2Bank(D, bridge_dim, n_slots=l2_slots, softmax_free=softmax_free,
                         tau_prior=l2_tau_prior)

        # L3: emergent concepts (long-range, ~8 concepts from L2 clustering)
        self.l3 = L3Concepts(D, bridge_dim, n_concepts=l3_concepts,
                             birth_threshold=l3_birth_threshold, softmax_free=softmax_free,
                             tau_prior=l3_tau_prior)

        # Fusion gate: combine L1 + L2 + L3 + current state
        self.fusion = nn.Sequential(
            nn.Linear(D * 4, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        # Gate init: start as no-op
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

        # Injection scale (starts small, grows if helpful)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))
        # U6: τ-consistent fusion: per-level importance scales with τ_norm
        self._fusion_tau_alpha = nn.Parameter(torch.zeros(4))  # learnable per-level τ-modulation

        # Track sentence boundaries
        self._in_sentence = True
        self._sent_start = 0

    def forward(self, h: torch.Tensor, tokens: torch.Tensor,
                step: int = None, mat_gate: float = None) -> torch.Tensor:
        """Read from memory at each position.

        h: (B, L, D) — current hidden state (after embedding)
        tokens: (B, L) — token ids (for boundary detection)
        step: current training step (for logging)
        mat_gate: float — maturation gate value (0-1), gates writes
        returns: (B, L, D) — memory-augmented hidden state
        """
        B, L, D = h.shape
        is_sep = (tokens == 2)  # SEP token = sentence boundary

        # Determine if writes are allowed
        _can_write = (mat_gate is None) or (mat_gate >= self._min_write_maturation)

        # Detect boundaries and write to all levels
        with torch.no_grad():
            for b in range(B):
                sent_start = 0
                for t in range(L):
                    if is_sep[b, t]:
                        summary = h[b, sent_start:t+1].mean(0)  # (D,)

                        # Write to L1 (always when allowed)
                        if _can_write:
                            self.l1.write(summary)

                        # Write to L2 (only when allowed)
                        l2_slot = -1
                        if _can_write:
                            l2_slot = self.l2.write(summary)

                        # Write to L3 (concept clustering from L2 key+val)
                        # If L3 writes (birth/update), mark L2 slot as consumed
                        if _can_write and l2_slot >= 0:
                            l2_key = self.l2.keys[l2_slot]  # (bridge_dim,)
                            l2_val = self.l2.val_norm(self.l2.vals[l2_slot])  # (bridge_dim,) — normalized
                            conf = self.l2.slot_novelty[l2_slot].item()
                            wrote_l3 = self.l3.write(l2_key, l2_val=l2_val, confidence=conf)
                            if wrote_l3:
                                # L3 consumed this L2 slot — mark for overwrite
                                self.l2.mark_consumed(l2_slot)

                        sent_start = t + 1

        # Read from all levels
        mem_l1 = self.l1.read(h)  # (B, L, D)
        mem_l2 = self.l2.read(h)  # (B, L, D)
        mem_l3 = self.l3.read(h)  # (B, L, D)

        # Fusion: combine current + L1 + L2 + L3
        combined = torch.cat([h, mem_l1, mem_l2, mem_l3], dim=-1)  # (B, L, 4D)
        fused = self.fusion(combined)  # (B, L, D)

        # Injection with bounded scale
        scale = torch.tanh(self.log_scale)  # in (-1, 1)

        # U6: τ-consistent fusion scaling — modulate injection by τ_norm
        if self.tau_config is not None and hasattr(self.tau_config, 'tau_norm'):
            tau_norm = self.tau_config.tau_norm.mean()
            # τ-low layers (fast, shallow): less memory injection
            # τ-high layers (slow, deep): more memory injection
            tau_fusion_scale = 0.3 + 0.7 * tau_norm.item()
            scale = scale * tau_fusion_scale

        # When maturation too low, bypass memory bank entirely (no-op)
        if not _can_write:
            return h

        return h + scale * fused

    def reset(self) -> None:
        """Clear all memory (for new sequence)."""
        self.l1.reset()
        self.l2.reset()
        self.l3.reset()

    def get_diagnostics(self) -> dict:
        """Return diagnostic info for logging."""
        l1s = self.l1.get_stats()
        l2s = self.l2.get_stats()
        l3s = self.l3.get_stats()
        return {
            'l1_write_idx': self.l1._write_idx,
            'l1_overwrites': l1s['n_overwrites'],
            'l1_fill': l1s['fill_rate'],
            'l2_write_idx': self.l2._write_idx,
            'l2_overwrites': l2s['n_overwrites'],
            'l2_consumed': l2s['n_consumed'],
            'l2_fill': l2s['fill_rate'],
            'l2_novelty_mean': l2s['novelty_mean'],
            'l2_age_mean': self.l2.slot_age.mean().item(),
            'l2_key_scale': l2s['key_scale'],
            'l2_val_scale': l2s['val_scale'],
            'l3_n_concepts': l3s['n_active'],
            'l3_n_births': l3s['n_births'],
            'l3_n_updates': l3s['n_updates'],
            'l3_confidence_mean': l3s['confidence_mean'],
            'l3_val_scale': l3s['val_scale'],
            'mem_scale': torch.tanh(self.log_scale).item(),
        }
