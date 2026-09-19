"""Streaming Memory Bank for EVA — hierarchical L1 + L2 (concepts: global UCL).

Architecture:
  L1 (immediate): rolling buffer of last K sentence embeddings
    - Simple ring buffer: overwrite oldest
    - No learning, just storage + attention read
    - Always active, no maturation gating

  L2 (learned): VSA-mediated memory bank with N slots
    - Write: at sentence boundaries (SEP token = 2)
    - Read: attention over slots at each token
    - Ring buffer: overwrite oldest (or consumed slot); M64.6: the
      novelty-gate MLP was removed (dead by construction, M63-D)
    - Differentiable: gradients flow through write/read (M64.3)
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
    # B3 (agent D): the old form multiplied raw scores by temp AND fed them to
    # hybrid_gate (which divides internally) — the two effects cancelled and
    # entropy moved 1.5% across the whole τ range: the learned temperature was
    # decorative. One consistent τ in both branches. T8: the clamp band
    # [0.1, 10] is NOT τ-derived (the τ-ladder is 8..512) — it is a DECLARED
    # calibrated band for learnable log_tau (registered in test_tau_lint).
    scores = (q @ k.T) / math.sqrt(bridge_dim)

    if softmax_free:
        attn = hybrid_gate(scores, temp)
    else:
        attn = F.softmax(scores / temp.clamp(min=0.05), dim=-1)

    if age_decay is not None:
        attn = attn * age_decay.unsqueeze(0).unsqueeze(0)

    attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    attn = attn / attn_sum

    return attn


class L1Buffer(nn.Module):
    """Rolling buffer of last K sentence embeddings.

    Simple ring buffer: overwrite oldest slot.
    Fast, no cosine similarity checks.
    Keys normalized via F.normalize for consistent attention with L2.
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

    def read(self, query: torch.Tensor, temp_k: float = 1.0) -> torch.Tensor:
        """Read from buffer using hybrid attention.

        query: (B, L, D) — current hidden state
        temp_k: M55a — the lacuna broadening (attn temp multiplier)
        returns: (B, L, D) — memory read output
        """
        B, L, _ = query.shape
        q = self.q_proj(query)  # (B, L, bridge_dim)
        k = F.normalize(self.proj(self.buf), dim=-1)  # (n_slots, bridge_dim) — normalized!
        v = self.out_proj(self.buf)  # (n_slots, D)

        temp = torch.exp(self.log_tau).clamp(min=0.1, max=10.0) * float(temp_k)
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
    Consumed slots are prioritized for overwrite.
    
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

        # B7 (audit 01, F-02): bank CONTENT is state, not weights. As
        # Parameters they sat in Adam (whose moments went stale under the
        # .data mutations), were invisible to eval snapshot/restore (F-03),
        # and document-rotation wiped values the optimizer was chasing.
        self.register_buffer('keys', torch.randn(n_slots, bridge_dim) * 0.02)
        self.register_buffer('vals', torch.randn(n_slots, bridge_dim) * 0.02)

        # (M64.6 TOMBSTONE: the `novelty_gate` MLP stood here — a 2-layer gate
        # whose score only fed the `slot_novelty` diagnostic. The M63-D audit
        # measured it dead by construction (no gradient path at all: the value
        # was never multiplied into the read, and the M64.4 round-1 attempt to
        # use it as a value scale was annihilated by val_norm's LayerNorm).
        # Removed with the slot_novelty buffer + the novelty_mean telemetry.)

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
        self.register_buffer('slot_consumed', torch.zeros(n_slots, dtype=torch.bool), persistent=True)
        self.register_buffer('_write_idx', torch.zeros(1, dtype=torch.long), persistent=True)
        self._n_overwrites = 0
        self._n_consumed = 0
        self._keys_eff = None   # M64: the graph-carrying store of this forward
        self._vals_eff = None

    def write(self, embedding: torch.Tensor) -> int:
        """Write to bank. Returns slot index that was written.

        Prioritizes overwriting consumed slots.
        Falls back to overwriting oldest slot.

        M64 (M63-D): the projections LIVE in the autograd graph now. The old
        body ran W_k/W_v under @torch.no_grad and copied the result into the
        buffers, so the keys/vals the READ consumed were detached constants:
        W_k/W_v never received a gradient (measured: p.grad is None for the
        whole run) and the read degenerated into a per-forward bias
        (cos(r_t, r_t') ~ 1.0 between positions). The write is functional now:
        the effective store (keys_eff/vals_eff) carries the graph, the read
        consumes it, and the persistent buffer is committed with the detached
        copy so the checkpoint stays correct.
        """
        n_filled = min(self._write_idx.item(), self.n_slots)

        # the slot choice is a state decision (ages/flags), not a gradient path
        with torch.no_grad():
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

        # M64: live projections — the CE gradient flows through the read's
        # attention into W_k/W_v. (R1/R2/R3 review: the novelty multiplier on
        # the value was removed — val_norm (LayerNorm) annihilated a per-slot
        # positive scalar to ~1e-4 of the W_k gradient, so it was inert by
        # construction and beyond the M63-D audit. The gate's fate is M64.6.)
        raw_key = self.W_k(embedding)
        new_key = F.normalize(raw_key, dim=-1) * torch.sigmoid(self.key_log_scale)
        raw_val = self.W_v(embedding)
        new_val = F.normalize(raw_val, dim=-1) * torch.sigmoid(self.val_log_scale)
        # AMP safety (R1): the buffer is fp32 while autocast may hand us bf16
        new_key = new_key.to(self.keys.dtype)
        new_val = new_val.to(self.vals.dtype)

        # out-of-place row replacement (index_copy, the UCL pattern): an
        # in-place `keys_eff[slot] = ...` on a non-grad constant silently
        # DETACHES the new row (autograd does not track in-place ops on
        # tensors that do not require grad) — measured: W_k.grad stayed None.
        # R2: chain from the CURRENT effective store, not from the detached
        # buffer — a forward writes once per SEP position, and each commit
        # detaches, so only the LAST write's graph survived before.
        _base_k = self._keys_eff if self._keys_eff is not None else self.keys
        _base_v = self._vals_eff if self._vals_eff is not None else self.vals
        _slot_t = torch.tensor([slot], device=self.keys.device, dtype=torch.long)
        keys_eff = _base_k.index_copy(0, _slot_t, new_key.unsqueeze(0))
        vals_eff = _base_v.index_copy(0, _slot_t, new_val.unsqueeze(0))
        self._keys_eff = keys_eff          # per-forward, consumed by read()
        self._vals_eff = vals_eff

        with torch.no_grad():
            self.keys.data.copy_(keys_eff.detach())
            self.vals.data.copy_(vals_eff.detach())
            self.slot_age.data[slot] = 0.0
            self.slot_consumed.data[slot] = False  # clear consumed flag
            mask = torch.arange(self.n_slots, device=self.slot_age.device) != slot
            self.slot_age.data[mask] += 1.0
            self._n_overwrites += 1
            self._write_idx += 1
        return slot

    @torch.no_grad()
    def mark_consumed(self, slot: int) -> None:
        """Mark slot as consumed (e.g. by concept promotion)."""
        if 0 <= slot < self.n_slots:
            self.slot_consumed.data[slot] = True

    def clear_effective(self) -> None:
        """M64 (R1/R2): drop the per-forward effective store.

        The stash carries the forward's graph; it must not survive into a
        context that did not write (eval snapshot/restore, reset_cache, a
        low-maturation forward with no write). The bank's forward clears it
        on every `write=True` call, and this hook covers the external flushes.
        """
        self._keys_eff = None
        self._vals_eff = None

    def read(self, query: torch.Tensor, temp_k: float = 1.0) -> torch.Tensor:
        """Read from bank using hybrid attention.

        query: (B, L, D)
        temp_k: M55a — the lacuna broadening (attn temp multiplier)
        returns: (B, L, D)
        """
        B, L, _ = query.shape
        q = self.q_proj(query)  # (B, L, bridge_dim)
        # M64 (M63-D): prefer the graph-carrying effective store from this
        # forward's write; fall back to the committed buffer (eval/fresh).
        k = getattr(self, '_keys_eff', None)
        if k is None:
            k = self.keys  # (n_slots, bridge_dim)
        v_raw = getattr(self, '_vals_eff', None)
        if v_raw is None:
            v_raw = self.vals
        v = self.val_norm(v_raw)  # (n_slots, bridge_dim) — normalized for stability

        temp = torch.exp(self.log_tau).clamp(min=0.1, max=10.0) * float(temp_k)
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
            'consumed_count': int(self.slot_consumed.sum().item()),
            'key_scale': torch.sigmoid(self.key_log_scale).item(),
            'val_scale': torch.sigmoid(self.val_log_scale).item(),
        }

    def reset(self) -> None:
        self.keys.data.zero_()
        self.vals.data.zero_()
        self.slot_age.zero_()
        self.slot_consumed.zero_()
        self._write_idx.zero_()
        self._n_overwrites = 0
        self._n_consumed = 0
        self._keys_eff = None   # M64: drop the stale effective store
        self._vals_eff = None


class StreamingMemoryBank(nn.Module):
    """Combined L1 + L2 working memory for EVA.

    Audit decision #2: the emergent-concept store is the SINGLE global
    UnifiedConceptLayer (stack.concept_layer). L3Concepts was a second,
    differently-parameterized concept system (config and the UCL docstring
    already declared UCL its replacement) — retired here: no duplicate
    buffers, one birth math, one checkpoint footprint. Long-range concepts
    enter the trunk only through the UCL.

    Flow:
      L1.write(summary)  -> overwrite oldest (fast, immediate)
      L2.write(summary)  -> ring-buffer slots (short-term; M64.6: no novelty gate)

    Integration points:
    - forward(h, tokens, step, mat_gate): read from L1+L2 at each position
    - reset(): clear all memory (for new sequence)
    """
    def __init__(self, D: int, bridge_dim: int,
                 l1_slots: int = 3, l2_slots: int = 16,
                 min_write_maturation: float = 0.3,
                 softmax_free: bool = True,
                 cfg=None,
                 tau_config=None):
        super().__init__()
        self.D = D
        self.bridge_dim = bridge_dim
        self.cfg = cfg
        # M55a: the lacuna-driven search broadening (a big lacuna widens retrieval).
        self.lacuna_k: float = float(getattr(cfg, 'mem_lacuna_k', 0.5))
        self._min_write_maturation = min_write_maturation
        self._softmax_free = softmax_free
        self.tau_config = tau_config

        # Compute τ-priors from tau_config if available
        if tau_config is not None:
            mem_tau = tau_config.mem_tau  # (3,) percentiles — the bank uses the fast pair
            # Normalize to reasonable range for hybrid_gate temperature
            l1_tau_prior = (mem_tau[0] / tau_config.mem_tau_ref).clamp(0.1, 5.0).item()
            l2_tau_prior = (mem_tau[1] / tau_config.mem_tau_ref).clamp(0.1, 5.0).item()
        else:
            l1_tau_prior = 0.5  # L1 = fast (low tau → precision)
            l2_tau_prior = 1.0  # L2 = balanced

        # L1: rolling buffer (immediate, ~last K diverse sentences)
        self.l1 = L1Buffer(D, bridge_dim, n_slots=l1_slots, softmax_free=softmax_free,
                           tau_prior=l1_tau_prior)

        # L2: learned bank (short-term, ~N diverse slots)
        self.l2 = L2Bank(D, bridge_dim, n_slots=l2_slots, softmax_free=softmax_free,
                         tau_prior=l2_tau_prior)

        # Fusion gate: combine L1 + L2 + current state
        self.fusion = nn.Sequential(
            nn.Linear(D * 3, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        # Gate init: start as no-op
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

        # Injection scale (starts small, grows if helpful)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))
        # U6: τ-consistent fusion: per-level importance scales with τ_norm
        self._fusion_tau_alpha = nn.Parameter(torch.zeros(3))  # learnable per-level τ-modulation

        # Track sentence boundaries
        self._in_sentence = True
        self._sent_start = 0

    def forward(self, h: torch.Tensor, tokens: torch.Tensor,
                step: int = None, mat_gate: float = None,
                lacuna: float = None, write: bool = True) -> torch.Tensor:
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
        # M33 (supersedes B7 F-03 mode gate): writes happen in BOTH regimes —
        # 'инференс = обучение' (README §1.4). Hold-out content can no longer
        # leak into the training bank: evaluate() snapshots/restores the bank
        # buffers and resets them per hold-out document (boundary semantics).
        _can_write = ((mat_gate is None) or
                      (mat_gate >= self._min_write_maturation))

        # Detect boundaries and write to all levels (M58b: the stack passes
        # write=False after the first active layer — one sentence is written
        # ONCE per forward, not once per layer).
        # M64 (R1/R2): NO outer no_grad — it made the L2 write projections dead
        # in the only production call-site (the unit tests called L2Bank.write
        # directly and missed it: W_k.grad was None through the stack while the
        # method-level test was green). L1Buffer.write is @torch.no_grad itself
        # (its content is state, its projection lives on the read side).
        if write:
            # R1/R2: the effective store is per-FORWARD. The first bank call of
            # a forward clears the previous stash, so a forward without a write
            # (low maturation) or an eval after training falls back to the
            # committed buffer instead of consuming yesterday's graph.
            self.l2.clear_effective()
        for b in range(B if write else 0):
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

                    sent_start = t + 1

        # Read from all levels (M55a: a big lacuna widens the search)
        _tk = 1.0 + self.lacuna_k * max(0.0, float(lacuna or 0.0))
        mem_l1 = self.l1.read(h, temp_k=_tk)  # (B, L, D)
        mem_l2 = self.l2.read(h, temp_k=_tk)  # (B, L, D)


        # U6 (audit M11 made it real): _fusion_tau_alpha is an actual
        # learnable PER-LEVEL modulation — exp(offset) on each [h, L1, L2]
        # block before fusion, zero-init ⇒ identity (checkpoint-safe). The
        # parameter previously existed, sat in the optimizer and modulated
        # nothing.
        alpha = self._fusion_tau_alpha.to(h.dtype).exp()        # (4,), ≈1 at init
        w = torch.stack([alpha[0] * h, alpha[1] * mem_l1,
                         alpha[2] * mem_l2], dim=-2)
        D = h.shape[-1]
        combined = w.reshape(*h.shape[:-1], 3 * D)               # (B, L, 3D)
        fused = self.fusion(combined)  # (B, L, D)

        # Injection with bounded scale
        scale = torch.tanh(self.log_scale)  # in (-1, 1)

        # τ-consistent injection schedule (tensor, no CPU sync):
        # τ-low layers (fast, shallow): less memory injection
        # τ-high layers (slow, deep): more memory injection
        if self.tau_config is not None and hasattr(self.tau_config, 'tau_norm'):
            tau_norm = self.tau_config.tau_norm.mean()
            scale = scale * (0.3 + 0.7 * tau_norm)

        # When maturation too low, bypass memory bank entirely (no-op)
        # M55a/M58b: the read direction for the head<->memory conflict
        # channel — cached in BOTH modes (the eval must not hand the head a
        # stale training direction; the eval snapshot covers it either way).
        self._last_read = fused.detach()
        if not _can_write:
            return h

        return h + scale * fused

    def reset(self) -> None:
        """Clear all memory (for new sequence)."""
        self.l1.reset()
        self.l2.reset()


    def get_diagnostics(self) -> dict:
        """Return diagnostic info for logging."""
        l1s = self.l1.get_stats()
        l2s = self.l2.get_stats()

        return {
            'l1_write_idx': self.l1._write_idx,
            'l1_overwrites': l1s['n_overwrites'],
            'l1_fill': l1s['fill_rate'],
            'l2_write_idx': self.l2._write_idx,
            'l2_overwrites': l2s['n_overwrites'],
            'l2_consumed': l2s['n_consumed'],
            'l2_fill': l2s['fill_rate'],
            'l2_age_mean': self.l2.slot_age.mean().item(),
            'l2_key_scale': l2s['key_scale'],
            'l2_val_scale': l2s['val_scale'],
            'mem_scale': torch.tanh(self.log_scale).item(),
        }

    # ─── tau-adaptive state compression ────────────────────────────

    def compress_state(self) -> None:
        """Compress internal state for memory-efficient storage.

        Called between generation steps to reduce memory footprint.
        Uses tau-adaptive compression: L1 (volatile) gets uniform8,
        L2 (stable) gets sparse top-k.
        """
        from .tau_compression import compress_uniform8, compress_sparse_topk

        # L1: volatile → uniform8 (4x compression)
        if not hasattr(self, '_l1_compressed'):
            self._l1_compressed = None
        buf = self.l1.buf.data
        idx, t_min, scale = compress_uniform8(buf)
        self._l1_compressed = {'idx': idx, 'min': t_min, 'scale': scale,
                               'shape': buf.shape, 'dtype': buf.dtype}

        # L2 keys: medium → sparse_topk-128
        if not hasattr(self, '_l2_keys_compressed'):
            self._l2_keys_compressed = None
        keys = self.l2.keys.data
        idx_pos, idx_vals, meta = compress_sparse_topk(keys.unsqueeze(0), k=min(128, keys.shape[-1]))
        self._l2_keys_compressed = {'pos': idx_pos, 'vals': idx_vals, 'meta': meta,
                                    'shape': keys.shape, 'dtype': keys.dtype}

        # L2 vals: medium → sparse_topk-128
        if not hasattr(self, '_l2_vals_compressed'):
            self._l2_vals_compressed = None
        vals = self.l2.vals.data
        idx_pos, idx_vals, meta = compress_sparse_topk(vals.unsqueeze(0), k=min(128, vals.shape[-1]))
        self._l2_vals_compressed = {'pos': idx_pos, 'vals': idx_vals, 'meta': meta,
                                    'shape': vals.shape, 'dtype': vals.dtype}

    def decompress_state(self) -> None:
        """Decompress internal state after compression.

        Called before reading to restore full precision.
        """
        from .tau_compression import decompress_uniform8, decompress_sparse_topk

        # L1
        if hasattr(self, '_l1_compressed') and self._l1_compressed is not None:
            c = self._l1_compressed
            self.l1.buf.data.copy_(
                decompress_uniform8(c['idx'], c['min'], c['scale'],
                                    c['shape'], c['dtype']))

        # L2 keys
        if hasattr(self, '_l2_keys_compressed') and self._l2_keys_compressed is not None:
            c = self._l2_keys_compressed
            decompressed = decompress_sparse_topk(c['pos'], c['vals'], c['meta'],
                                                   (1,) + c['shape'], c['dtype'])
            self.l2.keys.data.copy_(decompressed.squeeze(0))

        # L2 vals
        if hasattr(self, '_l2_vals_compressed') and self._l2_vals_compressed is not None:
            c = self._l2_vals_compressed
            decompressed = decompress_sparse_topk(c['pos'], c['vals'], c['meta'],
                                                   (1,) + c['shape'], c['dtype'])
            self.l2.vals.data.copy_(decompressed.squeeze(0))

    def compression_stats(self) -> dict:
        """Get compression statistics."""
        import sys
        orig_bytes = 0
        comp_bytes = 0

        # L1
        orig_bytes += self.l1.buf.numel() * 4
        if hasattr(self, '_l1_compressed') and self._l1_compressed is not None:
            c = self._l1_compressed
            comp_bytes += c['idx'].numel() + 8 if c['idx'] is not None else 8

        # L2 keys
        orig_bytes += self.l2.keys.numel() * 4
        if hasattr(self, '_l2_keys_compressed') and self._l2_keys_compressed is not None:
            c = self._l2_keys_compressed
            comp_bytes += c['pos'].numel() * 2 + c['vals'].numel() + 8

        # L2 vals
        orig_bytes += self.l2.vals.numel() * 4
        if hasattr(self, '_l2_vals_compressed') and self._l2_vals_compressed is not None:
            c = self._l2_vals_compressed
            comp_bytes += c['pos'].numel() * 2 + c['vals'].numel() + 8

        ratio = orig_bytes / max(comp_bytes, 1)
        return {
            'original_bytes': orig_bytes,
            'compressed_bytes': comp_bytes,
            'ratio': ratio,
            'original_kb': orig_bytes / 1024,
            'compressed_kb': comp_bytes / 1024,
        }
