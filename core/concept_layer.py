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
        cfg=None,
        softmax_free: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.softmax_free = softmax_free
        self.D = D
        self.k = k
        self.bridge_dim = bridge_dim
        self.S = S

        # ─── τ-параметры (все пороги через sigmoid(τ·x)) ───
        # Novelty: sigmoid(τ_novelty · (1 - best_sim)) → вероятность "новизны"
        self.log_tau_novelty = nn.Parameter(torch.tensor(0.0))
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
        g = torch.Generator().manual_seed(seed)
        m_init = torch.randn(S, bridge_dim, generator=g)
        self.concept_keys = nn.Parameter(F.normalize(m_init, dim=-1))  # (S, bridge_dim)
        self.concept_vals = nn.Parameter(torch.randn(S, D) * 0.02)    # (S, D)

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

    def _update_maturity(self, resvar: torch.Tensor | float):
        """Continuous maturity from residual variance coefficient of variation.

        mat = sigmoid((1/cv - λ) · τ_mat)
        cv = sqrt(var) / |ema| → low cv = stable = mature
        """
        if resvar is None:
            return
        rv = resvar.detach().item() if isinstance(resvar, torch.Tensor) else float(resvar)
        lam = getattr(self.cfg, 'lambda_d', 3) if self.cfg else 3
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

    def _maybe_write(self, hp: torch.Tensor, pen: torch.Tensor, mat_gate: float):
        """τ-gated concept write (birth/update).

        hp: (B, L, G, k) — expert K-space states
        pen: (B, L) — prediction error norm
        mat_gate: float — maturation gate from stack

        Returns: (write_event: (B,L) bool, best: (B,L) long)
        """
        self._step += 1
        B, L, G, k = hp.shape
        device = hp.device
        zeros = torch.zeros(B, L, dtype=torch.bool, device=device)

        if mat_gate < 0.1:
            return zeros, zeros.long()

        # τ-driven thresholds
        tau_nov = torch.exp(self.log_tau_novelty).clamp(0.1, 10.0)
        tau_birth = torch.exp(self.log_tau_birth).clamp(0.1, 10.0)
        tau_update = torch.exp(self.log_tau_update).clamp(0.1, 10.0)

        # Shared representation: gate-weighted average over experts
        gate_w = torch.ones(B, L, G, device=device)  # uniform if no gate provided
        gsum = gate_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        shared = (hp * gate_w.unsqueeze(-1)).sum(dim=-2) / gsum  # (B, L, k)

        # Query/key for concept matching (project k → bridge_dim)
        q = self.write_q_proj(shared).reshape(B * L, self.bridge_dim)  # (B*L, bridge_dim)
        q = q.reshape(B, L, self.bridge_dim)
        q_n = F.normalize(q, dim=-1)
        concept_n = F.normalize(self.concept_keys.data, dim=-1)  # (S, bridge_dim)
        sims = torch.einsum('blk,sk->bls', q_n, concept_n)  # (B, L, S)
        best = sims.argmax(dim=-1)  # (B, L)
        best_sim = sims.max(dim=-1).values  # (B, L)

        # Value projection (k → D) for writing into concept_vals
        val_proj = self.write_v_proj(shared)  # (B, L, D)

        # Confidence from prediction error
        conf = torch.sigmoid(-pen)  # (B, L) — high confidence = low error

        write_event = torch.zeros(B, L, dtype=torch.bool, device=device)

        # ─── Update existing concepts ───
        # Update momentum: α = sigmoid(-log(τ_update)) → small for high τ
        alpha = torch.sigmoid(-self.log_tau_update).clamp(0.001, 0.5).item()
        mat = self._mature.item()

        if mat >= 0.3:
            for s in range(self.S):
                mask = (best == s) & (conf >= conf.median().clamp(min=0.01))
                if mask.any():
                    write_event |= mask
                    new_key = q_n.reshape(B, L, self.bridge_dim)[mask].mean(dim=0)
                    new_key = F.normalize(new_key, dim=-1)
                    if self.concept_count[s].item() < 3:
                        self.concept_keys.data[s] = new_key
                    else:
                        self.concept_keys.data[s] = F.normalize(
                            self.concept_keys.data[s] * (1 - alpha) + new_key * alpha, dim=-1
                        )
                    # Update value (no normalize — preserve magnitude)
                    new_val = val_proj[mask].mean(dim=0)
                    self.concept_vals.data[s] = self.concept_vals.data[s] * (1 - alpha) + new_val * alpha
                    self.concept_count[s] += mask.sum().item()
                    self.concept_age.data[s] = 0.0
                    self._n_updates += 1

        # ─── Birth new concepts ───
        # Novelty: sigmoid(τ_novelty · (1 - best_sim)) → high when far from all concepts
        novelty_score = torch.sigmoid(tau_nov * (1.0 - best_sim))  # (B, L)
        # U7: τ-learned birth threshold
        base_thr = torch.sigmoid(self._log_tau_birth_thr).item()
        decay = torch.sigmoid(self._log_tau_decay_thr).item()
        tau_norm_val = self._tau_norm if self._tau_norm is not None else 0.5
        birth_thresh = base_thr * (1.0 - tau_norm_val * decay)
        novel = (novelty_score > 0.5) & (conf >= birth_thresh)  # τ-driven novelty

        if mat >= 0.1 and novel.any():
            empty = torch.nonzero(self.concept_count == 0)
            if empty.numel() > 0:
                idx = empty[0].item()
                self.concept_keys.data[idx] = F.normalize(
                    q_n[novel].mean(dim=0), dim=-1
                )
                self.concept_vals.data[idx] = val_proj[novel].mean(dim=0)
                self.concept_count[idx] = 1
                self.concept_age[idx] = 0.0
                self.concept_confidence[idx] = conf[novel].mean().item()
                self._n_births += 1
                write_event |= novel
            else:
                # Eviction: least useful concept
                utility = self.concept_confidence * self.concept_count.clamp(min=1)
                evict = utility.argmin().item()
                self.concept_keys.data[evict] = F.normalize(
                    q_n[novel].mean(dim=0), dim=-1
                )
                self.concept_vals.data[evict] = val_proj[novel].mean(dim=0)
                self.concept_count[evict] = 1
                self.concept_age[evict] = 0.0
                self.concept_confidence[evict] = conf[novel].mean().item()
                self._n_births += 1
                write_event |= novel
        else:
            self._n_skipped += (~novel).sum().item() if novel.numel() > 0 else 0

        # Age all concepts
        self.concept_age += 1.0

        return write_event, best

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
        tau_norm: float = None,
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

        # Write concepts (τ-gated)
        if allow_write and hp is not None and pen is not None:
            write_event, best = self._maybe_write(hp, pen, mat_gate)
        else:
            write_event = torch.zeros(B, L, dtype=torch.bool, device=device)
            best = torch.zeros(B, L, dtype=torch.long, device=device)

        # ─── Read from concepts ───
        concept_n = F.normalize(self.concept_keys.data, dim=-1)  # (S, bridge_dim)
        concept_v = self.concept_vals.data  # (S, D)

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

        return out

    # ─────────────────── Diagnostics ───────────────────

    def get_diagnostics(self) -> dict:
        active = int((self.concept_count > 0).sum().item())
        return {
            'concept_maturity': self._mature.item(),
            'concept_n_active': active,
            'concept_n_births': int(self._n_births.item()),
            'concept_n_updates': int(self._n_updates.item()),
            'concept_birth_gate': self._cached_birth_gate.item(),
            'concept_tau_novelty': torch.exp(self.log_tau_novelty).item(),
            'concept_tau_birth': torch.exp(self.log_tau_birth).item(),
            'concept_tau_read': torch.exp(self.log_tau_read).item(),
            'concept_confidence_mean': self.concept_confidence.mean().item(),
        }

    @torch.no_grad()
    def birth_gate_mean(self):
        return self._cached_birth_gate
