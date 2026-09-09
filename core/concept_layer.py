"""
Unified Concept Layer — single concept system replacing CollectiveConceptLayer + L3Concepts.

Design principles:
  1. All thresholds derived from τ-field (no magic numbers)
  2. Continuous maturity (sigmoid, not binary)
  3. Gradient flow through write path (no @torch.no_grad on writes)
  4. Per-expert attention (preserves ensemble diversity from mirror)
  5. τ-driven novelty, confidence, and update momentum

Architecture:
  - S concept slots (keys in bridge_dim, vals in D)
  - Per-expert similarity (B, L, G, S) → gate-weighted → read (B, L, D)
  - Write at sentence boundaries with τ-gated novelty
  - Continuous maturity from residual variance CV
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedConceptLayer(nn.Module):
    """Unified concept layer with τ-driven thresholds and continuous maturity.

    Replaces both CollectiveConceptLayer (per-block) and L3Concepts (in memory_bank).
    Single global instance in the stack, called after embedding.
    """

    def __init__(
        self,
        D: int,
        k: int,
        bridge_dim: int = 256,
        S: int = 8,
        seed: int = 42,
        cfg: Any = None,
        softmax_free: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.softmax_free = softmax_free
        self.D = D
        self.k = k
        self.bridge_dim = bridge_dim
        self.S = S

        # ─── τ-параметры (все пороги через sigmoid(τ·x)) ───
        # Novelty GAP: concept is novel when cosine distance to the nearest
        # slot exceeds gap = sigmoid(log_tau_novelty_thr) (design principle #1).
        # Audit M6: the old novelty_score = σ(τ·(1−sim)) tested `> 0.5` —
        # σ(x)>0.5 ⇔ x>0 ⇔ sim<1 for ANY τ: a dead knob that never gated.
        self._log_tau_novelty_thr = nn.Parameter(torch.tensor(0.0))  # gap position
        # Birth confidence: sigmoid(τ_birth · confidence) → порог рождения
        self.log_tau_birth = nn.Parameter(torch.tensor(0.0))
        # Update momentum: α = 1/τ_update → чем выше τ, тем медленнее обновление
        self.log_tau_update = nn.Parameter(torch.tensor(0.0))
        # Maturity scale: sigmoid((1/cv - λ) · τ_mat) → непрерывная зрелость
        self.log_tau_maturity = nn.Parameter(torch.tensor(1.0))
        # Read temperature: controls attention sharpness over concepts
        self.log_tau_read = nn.Parameter(torch.tensor(1.0))
        # Birth gate: sigmoid(τ_gate · (gap - best_sim)) → вероятность рождения
        self.log_tau_gate = nn.Parameter(torch.tensor(0.0))

        # ─── Concept slots ───
        # STORAGE buffers, not Parameters: learning lives in the projections
        # via the FUNCTIONAL write path (design principle #3; audit M6: the
        # old `.data[...]` writes cut every gradient and the Parameters were
        # dead weight in the optimizer). Read + effective store carry grads.
        g = torch.Generator().manual_seed(seed)
        m_init = torch.randn(S, bridge_dim, generator=g)
        self.register_buffer('concept_keys', F.normalize(m_init, dim=-1))  # (S, bridge_dim)
        self.register_buffer('concept_vals', torch.randn(S, D, generator=g) * 0.02)  # (S, D)

        # ─── Projections ───
        # Write path: hp expert K-space (B,L,G,k) → shared (B,L,k)
        self.write_q_proj = nn.Linear(k, bridge_dim)  # k → bridge_dim (write key matching)
        self.write_v_proj = nn.Linear(k, D)            # k → D (write value storage)
        # Read path: h D-space (B,L,D)
        self.q_proj = nn.Linear(D, bridge_dim)  # D → bridge_dim (read query)
        self.out_proj = nn.Linear(D, D)          # D → D (read output gate)

        # ─── Learnable read scale ───
        self.read_scale = nn.Parameter(torch.tensor(0.0))

        # ─── Persistent state ───
        self.register_buffer('concept_age', torch.zeros(S))
        self.register_buffer('concept_count', torch.zeros(S))
        self.register_buffer('concept_confidence', torch.zeros(S))
        self.register_buffer('_mature', torch.tensor(0.5))  # continuous: [0, 1]
        self.register_buffer('_resvar_ema', torch.tensor(0.0))
        self.register_buffer('_resvar_var', torch.tensor(1.0))
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long))

        # ─── Diagnostic counters ───
        self.register_buffer('_n_births', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_n_updates', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_n_skipped', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_cached_birth_gate', torch.tensor(0.0), persistent=False)

        # ─── Uncertainty/contradiction gates (from System A) ───
        self.log_tau_uncert = nn.Parameter(torch.tensor(1.0))   # uncertainty threshold
        self.log_tau_contra = nn.Parameter(torch.tensor(1.0))   # contradiction threshold
        self.uncert_kappa = nn.Parameter(torch.tensor(3.0))     # sharpness
        # U7: τ-learned birth threshold
        self._log_tau_birth_thr = nn.Parameter(torch.tensor(0.0))   # sigmoid → base threshold
        self._log_tau_decay_thr = nn.Parameter(torch.tensor(0.0))   # sigmoid → decay rate
        self._tau_norm = None  # set by stack during init

    # ─────────────────── Maturity ───────────────────

    def _update_maturity(self, resvar: torch.Tensor | float) -> None:
        """Continuous maturity from residual variance coefficient of variation.

        mat = sigmoid((1/cv - λ) · τ_mat)
        cv = sqrt(var) / |ema| → low cv = stable = mature
        (audit M6: λ must be the λ_d VALUE from lambda_utils — the old code
        substituted the DIMENSION d (3) in place of λ≈1.839, mis-scaling the
        EMA rate and the neutral cv.)
        """
        if resvar is None:
            return
        from .lambda_utils import lambda_d as _lambda_d
        rv = resvar.detach().item() if isinstance(resvar, torch.Tensor) else float(resvar)
        d_lam = getattr(self.cfg, 'lambda_d', 3) if self.cfg is not None else 3
        lam = _lambda_d(d_lam)
        ema_rate = 1.0 / lam

        delta = rv - self._resvar_ema.item()
        self._resvar_ema.fill_(self._resvar_ema.item() + ema_rate * delta)
        self._resvar_var.mul_(1 - ema_rate).add_(delta * delta * ema_rate)

        cv = (self._resvar_var.item() ** 0.5) / (abs(self._resvar_ema.item()) + 1e-8)
        # Continuous maturity: sigmoid((1/cv - λ) · τ)
        tau_mat = torch.exp(self.log_tau_maturity).clamp(0.1, 10.0)
        mat_raw = (1.0 / max(cv, 1e-8) - lam) * tau_mat.item()
        self._mature.fill_(torch.sigmoid(torch.tensor(mat_raw)).item())

    @property
    def maturity(self) -> float:
        return self._mature.item()

    # ─────────────────── Write ───────────────────

    def _maybe_write(self, hp: torch.Tensor, pen: torch.Tensor, mat_gate: float,
                     keys: torch.Tensor, vals: torch.Tensor,
                     gate: torch.Tensor | None = None,
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """τ-gated concept write — FUNCTIONAL & DIFFERENTIABLE (design
        principle #3; audit M6: the old implementation wrote through
        `.data[...]`, cutting autograd from read to write_q_proj/write_v_proj
        — the whole write path was learning-dead).

        Discrete bookkeeping (slot assignment, counts, ages, eviction) stays
        no-grad buffers; the key/value blend into the EFFECTIVE store is a
        live graph so gradients reach the write projections and the expert
        gate (principle #4: caller-provided per-expert gate weights the
        shared representation instead of being ignored).

        Returns (write_event: (B,L) bool, best: (B,L) long,
                 keys_eff: (S,bridge_dim) live, vals_eff: (S,D) live)."""
        self._step += 1
        B, L, G, k = hp.shape
        device = hp.device
        zeros_ev = torch.zeros(B, L, dtype=torch.bool, device=device)
        zeros_best = torch.zeros(B, L, dtype=torch.long, device=device)
        if mat_gate < 0.1:
            return zeros_ev, zeros_best, keys, vals

        # Shared representation: PER-EXPERT GATED average (principle #4).
        if gate is not None and tuple(gate.shape) == (B, L, G):
            gate_w = gate.float()
        else:
            gate_w = torch.ones(B, L, G, device=device)
        gsum = gate_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        shared = (hp * gate_w.unsqueeze(-1)).sum(dim=-2) / gsum  # (B, L, k)

        q = self.write_q_proj(shared)                            # live
        q_n = F.normalize(q, dim=-1)                             # (B,L,bridge_dim)
        concept_n = F.normalize(keys, dim=-1)                     # (S,bridge_dim)
        sims = torch.einsum('blk,sk->bls', q_n, concept_n)       # (B,L,S) live
        with torch.no_grad():
            best = sims.argmax(dim=-1)
            best_sim = sims.max(dim=-1).values
        val_proj = self.write_v_proj(shared)                      # (B,L,D) live

        conf = torch.sigmoid(-pen)                                # (B,L)
        write_event = torch.zeros(B, L, dtype=torch.bool, device=device)

        # Update momentum α (learnable, live scalar in the blend)
        alpha = torch.sigmoid(-self.log_tau_update).clamp(0.001, 0.5)
        mat = self._mature.item()

        if mat >= 0.3:
            conf_floor = conf.median().clamp(min=0.01)
            upd_idx, upd_keys, upd_vals = [], [], []
            for s in range(self.S):
                mask = (best == s) & (conf >= conf_floor)
                if not bool(mask.any()):
                    continue
                new_key = F.normalize(q_n[mask].mean(dim=0), dim=-1)
                new_val = val_proj[mask].mean(dim=0)
                if int(self.concept_count[s].item()) < 3:
                    k_upd, v_upd = new_key, new_val
                else:
                    a = alpha
                    # .clone(): a detached VIEW of the buffer row keeps the
                    # base's version counter — the read commit (copy_) then
                    # bumps it and breaks MulBackward (M6 in-place error)
                    k_upd = F.normalize(keys[s].detach().clone() * (1 - a) + new_key * a, dim=-1)
                    v_upd = vals[s].detach().clone() * (1 - a) + new_val * a
                upd_idx.append(s)
                upd_keys.append(k_upd)
                upd_vals.append(v_upd)
                with torch.no_grad():
                    self.concept_count[s] += mask.sum()
                    self.concept_age[s] = 0.0
                    self._n_updates += 1
                write_event |= mask
            if upd_idx:
                # out-of-place index_copy: grad flows through the SOURCE
                # (in-place put into a non-grad clone drops the graph — M6)
                it = torch.tensor(upd_idx, device=device)
                keys = keys.detach().index_copy(0, it, torch.stack(upd_keys))
                vals = vals.detach().index_copy(0, it, torch.stack(upd_vals))

        # ─── Birth new concepts ───
        # Novelty: cosine distance to the nearest slot must exceed the
        # LEARNABLE gap sigmoid(_log_tau_novelty_thr) (audit M6: the old
        # `novelty_score > 0.5` was equivalent to sim<1 — the τ knob never
        # gated anything).
        gap = torch.sigmoid(self._log_tau_novelty_thr)
        base_thr = torch.sigmoid(self._log_tau_birth_thr).item()
        decay = torch.sigmoid(self._log_tau_decay_thr).item()
        tau_norm_val = self._tau_norm if self._tau_norm is not None else 0.5
        birth_thresh = base_thr * (1.0 - tau_norm_val * decay)
        novel = ((1.0 - best_sim) > gap) & (conf >= birth_thresh)

        if mat >= 0.1 and bool(novel.any()):
            with torch.no_grad():
                empty = torch.nonzero(self.concept_count == 0)
            if empty.numel() > 0:
                idx = int(empty[0].item())
            else:
                with torch.no_grad():
                    utility = self.concept_confidence * self.concept_count.clamp(min=1)
                    idx = int(utility.argmin().item())
            it = torch.tensor([idx], device=device)
            # keep the update pass' gradient alive: index_copy's self is only
            # detached when it is still the raw (non-grad) buffer
            _kb = keys if keys.requires_grad else keys.detach()
            _vb = vals if vals.requires_grad else vals.detach()
            keys = _kb.index_copy(
                0, it, F.normalize(q_n[novel].mean(dim=0), dim=-1).unsqueeze(0))
            vals = _vb.index_copy(0, it, val_proj[novel].mean(dim=0).unsqueeze(0))
            with torch.no_grad():
                self.concept_count[idx] = 1
                self.concept_age[idx] = 0.0
                self.concept_confidence[idx] = conf[novel].mean().detach()
                self._n_births += 1
            write_event |= novel
        else:
            with torch.no_grad():
                self._n_skipped += (~novel).sum()

        with torch.no_grad():
            self.concept_age += 1.0
        return write_event, best, keys, vals

    # ─────────────────── Read ───────────────────

    def forward(
        self,
        h: torch.Tensor,
        hp: torch.Tensor | None = None,
        pen: torch.Tensor | None = None,
        resvar: torch.Tensor | None = None,
        mat_gate: float = 1.0,
        allow_write: bool = True,
        gate: torch.Tensor | None = None,
        tau_norm: float | None = None,
    ) -> torch.Tensor:
        """Unified concept layer forward.

        h: (B, L, D) — hidden state after embedding
        hp: (B, L, G, k) — expert K-space states (from first mirror)
        pen: (B, L) — prediction error norm
        resvar: scalar — residual variance for maturity
        mat_gate: float — maturation gate from stack
        allow_write: bool — enable writing
        gate: (B, L, G) — expert gate weights

        Returns: (B, L, D) — concept-augmented hidden state
        """
        B, L, D = h.shape
        device = h.device

        # U7: store τ_norm for birth threshold
        if tau_norm is not None:
            self._tau_norm = tau_norm

        # Update maturity (continuous, τ-driven)
        if self.training:
            self._update_maturity(resvar)

        # Write concepts (τ-gated) — FUNCTIONAL: returns the effective store
        # for the read below so the write path keeps its gradients (M6).
        if allow_write and hp is not None and pen is not None:
            write_event, best, keys_eff, vals_eff = self._maybe_write(
                hp, pen, mat_gate, self.concept_keys, self.concept_vals, gate=gate)
        else:
            write_event = torch.zeros(B, L, dtype=torch.bool, device=device)
            best = torch.zeros(B, L, dtype=torch.long, device=device)
            keys_eff, vals_eff = self.concept_keys, self.concept_vals

        # ─── Read from concepts ───
        concept_n = F.normalize(keys_eff, dim=-1)     # (S, bridge_dim)
        concept_v = vals_eff                           # (S, D)

        # Query: project h to bridge space
        q = self.q_proj(h)  # (B, L, bridge_dim)
        q_n = F.normalize(q, dim=-1)

        # τ-driven attention over concepts
        tau_read = torch.exp(self.log_tau_read).clamp(0.1, 10.0)
        scores = torch.einsum('blk,sk->bls', q_n, concept_n) * tau_read  # (B, L, S)

        # Hybrid attention: sigmoid * (1 + softmax/τ) — regime B
        if self.softmax_free:
            attn = torch.sigmoid(scores) * (1.0 + F.softmax(scores / tau_read, dim=-1))
        else:
            attn = F.softmax(scores, dim=-1)

        # Normalize attention
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Read values
        read = torch.einsum('bls,sd->bld', attn, concept_v)  # (B, L, D)
        read = self.out_proj(read)  # (B, L, D)

        # ─── Gating: uncertainty + contradiction (from System A) ───
        if pen is not None:
            tau_uncert = torch.exp(self.log_tau_uncert).clamp(0.1, 10.0)
            tau_contra = torch.exp(self.log_tau_contra).clamp(0.1, 10.0)
            kappa = self.uncert_kappa.clamp(0.5, 10.0)

            # Uncertainty gate: high when prediction error is high (need help)
            u_gate = torch.sigmoid(kappa * (pen.unsqueeze(-1) - tau_uncert))

            # Contradiction gate: high when concept output aligns with current state
            out_n = F.normalize(read, dim=-1)
            h_n = F.normalize(h.detach(), dim=-1)
            cos_sim = (out_n * h_n).sum(dim=-1, keepdim=True)
            c_gate = torch.sigmoid(tau_contra * (cos_sim - 0.0))  # threshold=0 via τ
        else:
            u_gate = torch.ones(B, L, 1, device=device)
            c_gate = torch.ones(B, L, 1, device=device)

        # ─── Output ───
        scale = torch.sigmoid(self.read_scale)
        out = read * u_gate * c_gate * scale

        # Cache birth gate for diagnostics
        if write_event.any():
            self._cached_birth_gate.fill_(attn[write_event].mean().item())
        else:
            self._cached_birth_gate.mul_(0.99)  # decay

        # ─── Commit the effective store to persistent buffers (after the
        # read so gradients through keys_eff/vals_eff are preserved).
        # Skip when nothing was written: keys_eff IS the buffer then, and a
        # self copy_ would bump the version of the tensors the read saved
        # (in-place-modified-saved-variable autograd error, audit M6).
        if write_event.any():
            with torch.no_grad():
                self.concept_keys.copy_(keys_eff.detach())
                self.concept_vals.copy_(vals_eff.detach())

        return out

    # ─────────────────── Diagnostics ───────────────────

    def get_diagnostics(self) -> dict[str, float | int]:
        active = int((self.concept_count > 0).sum().item())
        return {
            'concept_maturity': self._mature.item(),
            'concept_n_active': active,
            'concept_n_births': int(self._n_births.item()),
            'concept_n_updates': int(self._n_updates.item()),
            'concept_birth_gate': self._cached_birth_gate.item(),
            'concept_novelty_gap': torch.sigmoid(self._log_tau_novelty_thr).item(),
            'concept_tau_birth': torch.exp(self.log_tau_birth).item(),
            'concept_tau_read': torch.exp(self.log_tau_read).item(),
            'concept_confidence_mean': self.concept_confidence.mean().item(),
        }

    @torch.no_grad()
    def birth_gate_mean(self) -> torch.Tensor:
        return self._cached_birth_gate
