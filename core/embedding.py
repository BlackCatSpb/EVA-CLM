"""EVA: embedding module."""

from __future__ import annotations

import math, os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import EVAConfig
from .vsa_utils import zeckendorf_codes, sparse_block_codes
from .adaptive_gate import hybrid_gate


class RotaryEmbedding(nn.Module):
    # No fixed max_len: _build_cache grows on demand to the actual sequence
    # length (dynamic RoPE), so a length cap knob was dead — audit 2026-09.
    def __init__(self, D: int, theta: float = 1000000.0, scaling: float = 1.0) -> None:
        super().__init__()
        self.D: int = D
        self.theta: float = theta
        self.scaling: float = scaling
        half: int = D // 2
        freqs: torch.Tensor = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer('_freqs', freqs)
        self._max_cached: int = 0

    def _build_cache(self, L: int) -> None:
        if L <= self._max_cached:
            return
        t: torch.Tensor = torch.arange(L, dtype=torch.float32, device=self._freqs.device) / self.scaling
        angles: torch.Tensor = t[:, None] * self._freqs[None, :]
        self._cos_cached: torch.Tensor = angles.cos()
        self._sin_cached: torch.Tensor = angles.sin()
        self._max_cached = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        self._build_cache(L)
        cos: torch.Tensor = self._cos_cached[:L].to(x.dtype).to(x.device)
        sin: torch.Tensor = self._sin_cached[:L].to(x.dtype).to(x.device)
        x0: torch.Tensor = x[..., 0::2].contiguous()
        x1: torch.Tensor = x[..., 1::2].contiguous()
        out0: torch.Tensor = x0 * cos.unsqueeze(0) - x1 * sin.unsqueeze(0)
        out1: torch.Tensor = x0 * sin.unsqueeze(0) + x1 * cos.unsqueeze(0)
        out: torch.Tensor = torch.empty_like(x)
        out[..., 0::2] = out0
        out[..., 1::2] = out1
        return out

class ZeckendorfEmbedding(nn.Module):
    """Token -> D-space via Zeckendorf codes + learned projection.
    
    Legacy: проекция K→D через Linear. Ранг матрицы эмбеддингов ≤ K=23.
    """
    def __init__(self, cfg: EVAConfig) -> None:
        super().__init__()
        codes: torch.Tensor = zeckendorf_codes(cfg.vocab)
        K: int = codes.shape[1]
        self.register_buffer('codes', codes, persistent=False)
        self.proj: nn.Linear = nn.Linear(K, cfg.D, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.codes[tokens])



class PartitionedEmbedding(nn.Module):
    """Token -> D-space via partitioned sparse codes.
    
    D делится на K сегментов, K = D // seg_size (точное деление).
    Каждый бит кода получает свой сегмент: h = Σ z_k · w_k.
    
    K=32, S=6: C(32,6)=906192 ≥ V=50000. Ровно 6 активных бит на токен.
    Per-token: 6 × d = 6×112 = 672 dims (18.8%), детерминированно.
    
    Математические свойства:
      - rank(E) = 3584 (полный ранг)
      - Segment ↔ mirror group: 1:1 alignment (32×112)
      - Равномерная частота бит: ~19% каждый
      - K=32 → bind compression 32→16: ровно 2 сегмента на bind-канал
    """
    def __init__(self, cfg: EVAConfig) -> None:
        super().__init__()
        codes: torch.Tensor = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K: int = codes.shape[1]
        self.register_buffer('codes', codes, persistent=False)
        
        D: int = cfg.D
        assert D % self.K == 0, f'D={D} must be divisible by K={self.K}'
        d: int = D // self.K
        
        # Rank expansion: mixing matrix M (K×K) с ортогональной инициализацией
        # codes → sigmoid(M·codes) даёт плотные коэффициенты, каждый бит влияет на все сегменты
        self.embed_mix: nn.Parameter = nn.Parameter(torch.zeros(self.K, self.K))
        nn.init.orthogonal_(self.embed_mix)
        self.register_buffer('_mix_scale', torch.tensor(2.0), persistent=False)
        
        self.basis: nn.Parameter = nn.Parameter(torch.randn(self.K, d))
        nn.init.xavier_uniform_(self.basis, gain=0.5)
        self._rope_theta: float = getattr(cfg, 'rope_theta', 1000000.0)
        self._rope_scaling: float = getattr(cfg, 'rope_scaling', 1.0)
        self.rope: RotaryEmbedding = RotaryEmbedding(D, theta=self._rope_theta, scaling=self._rope_scaling)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Защита от токенов ≥ vocab (device-side assert в gather): фон-клип
        max_id = self.codes.shape[0] - 1
        if not bool(getattr(self, '_oor_warned', False)) and tokens.numel() > 0:
            # Audit M9: ReasoningTokens.THINK..END (65536+) are ≥ vocab and
            # used to collapse onto the last real token SILENTLY. Report the
            # first occurrence loudly (one-time; the clamp stays as guard).
            if bool((tokens > max_id).any().item()):
                self._oor_warned = True
                import warnings
                warnings.warn(
                    'token ids ≥ vocab detected — they are clamped to the last '
                    'vocabulary entry (reserved reasoning tokens are NOT wired '
                    'to any embedding row; check your data/tokenizer)',
                    RuntimeWarning, stacklevel=2)
        tokens = tokens.clamp(0, max_id)
        codes: torch.Tensor = self.codes[tokens]  # (B, L, K), sparse binary
        # Dense mixing: sigmoid(scale · M · codes) → каждый бит влияет на все сегменты
        codes = torch.sigmoid(codes @ self.embed_mix * self._mix_scale)
        B, L = tokens.shape
        # Внешнее произведение вместо einsum (стабильно под AMP на любых GPU)
        out: torch.Tensor = (codes.unsqueeze(-1) * self.basis.view(1, 1, self.K, -1)).reshape(B, L, -1)
        out = self.rope(out)
        return out



class LmHead(nn.Module):
    """D-space -> vocab logits via Zeckendorf code projection (legacy)."""
    def __init__(self, cfg: EVAConfig) -> None:
        super().__init__()
        codes: torch.Tensor = zeckendorf_codes(cfg.vocab)
        K: int = codes.shape[1]
        self.register_buffer('codes', codes, persistent=False)
        self.proj: nn.Linear = nn.Linear(cfg.D, K, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h) @ self.codes.T



class PartitionedHead(nn.Module):
    """D-space -> vocab logits via segment-addressed readout + per-token bias.
    
    h ∈ ℝᴰ → split по тем же K сегментам, что и в PartitionedEmbedding.
    Каждый сегмент h_k сравнивается со своим readout r_k:
        logit_v = Σ_k z_{vk} · ⟨h_k, r_k⟩ + b_v
    
    b_v — learnable per-token bias (token frequency prior).
    K=32: каждый сегмент выровнен с mirror group (1:1).
    
    Если embed_basis передан (PartitionedEmbedding.basis), readout делится с ним
    (weight tying encode/decode). Иначе — собственный readout.
    """
    def __init__(self, cfg: EVAConfig, embed_basis: Optional[nn.Parameter] = None) -> None:
        super().__init__()
        codes: torch.Tensor = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K: int = codes.shape[1]
        self.register_buffer('codes', codes, persistent=False)
        
        D: int = cfg.D
        assert D % self.K == 0
        d: int = D // self.K
        
        if embed_basis is not None:
            self.readout = embed_basis  # shared reference
        else:
            self.readout: nn.Parameter = nn.Parameter(torch.randn(self.K, d))
            nn.init.xavier_uniform_(self.readout, gain=0.5)
        self.token_bias: nn.Parameter = nn.Parameter(torch.zeros(cfg.vocab))
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        h_g: torch.Tensor = h.reshape(B, L, self.K, -1)  # (B, L, K, d)
        scores: torch.Tensor = (h_g * self.readout.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return scores @ self.codes.T + self.token_bias.unsqueeze(0).unsqueeze(0)


class SigmoidCodedHead(nn.Module):
    def __init__(self, cfg: EVAConfig, embed_basis: Optional[nn.Parameter] = None) -> None:
        super().__init__()
        codes: torch.Tensor = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K: int = codes.shape[1]
        self.S: int = cfg.code_sparsity
        self.register_buffer('codes', codes, persistent=False)
        D: int = cfg.D
        assert D % self.K == 0
        d: int = D // self.K
        if embed_basis is not None:
            self.readout = embed_basis
        else:
            self.readout: nn.Parameter = nn.Parameter(torch.randn(self.K, d))
            nn.init.xavier_uniform_(self.readout, gain=0.5)
        prop: torch.Tensor = codes.mean(dim=0)
        self.register_buffer('_prop', prop)
        # Code-prior init (the _prop buffer was computed and left dead): with
        # bit_bias = logit(p_active) the head starts at the unigram prior of
        # the sparse block code, not at an arbitrary 0.5 per bit.
        _p = prop.clamp(1e-7, 1 - 1e-7)
        self.bit_bias: nn.Parameter = nn.Parameter(torch.log(_p / (1 - _p)))
        self.log_temp: nn.Parameter = nn.Parameter(torch.zeros(self.K))
        self.token_bias: nn.Parameter = nn.Parameter(torch.zeros(cfg.vocab))
        self.normalize: bool = bool(getattr(cfg, 'head_normalize', True))

    def _gates(self, h: torch.Tensor, temp_factor: Optional[torch.Tensor] = None, bus_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(1)
            squeeze: bool = True
        else:
            squeeze = False
        B, L, D = h.shape
        h_g: torch.Tensor = h.reshape(B, L, self.K, -1)
        z: torch.Tensor = (h_g * self.readout.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        T: torch.Tensor = torch.exp(self.log_temp).clamp(0.1, 10.0)
        if temp_factor is not None:
            T = T * temp_factor
        zt: torch.Tensor = z / T + self.bit_bias
        if bus_bias is not None:
            # Phase-2 stencil: cross-layer gist biases the projector readout.
            # bus_bias shape matches zt's (B,L,K) or (N,1,K) -> broadcasts cleanly.
            zt = zt + bus_bias
        if squeeze:
            zt = zt.squeeze(1)
        return zt

    def _su(self, zt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Single per-bit temperature: T already scaled z in _gates, so the
        # emphasis softmax reads the SAME logits (passing tau=exp(log_temp)
        # here again divided twice: z/T/tau = z/T^2 — audit M1).
        return hybrid_gate(zt, 1.0, log=True)

    def forward(self, h: torch.Tensor, bus_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(1)
            squeeze: bool = True
        else:
            squeeze = False
        u, base = self._su(self._gates(h, bus_bias=bus_bias))
        logits: torch.Tensor = u @ self.codes.T + base[..., None] + self.token_bias
        if self.normalize:
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        if squeeze:
            logits = logits.squeeze(1)
        return logits

    def log_probs_for_target(self, h: torch.Tensor, targets: torch.Tensor, bus_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Training calls this with flattened (N, D) h (losses.compute_losses).
        # The old 2D branch returned raw[0, 0, targets]: EVERY position was
        # scored by hidden state #0's logits and the gradient for rows 1..N-1
        # was exactly zero (audit M1, proven: per-position grad = [75,0,...,0]).
        # token_bias also double-counted (forward already adds it before the
        # logsumexp normalization). Both removed: one gather, one bias.
        t: torch.Tensor = targets.reshape(-1)
        h2: torch.Tensor = h.reshape(-1, h.shape[-1])
        if bus_bias is not None and bus_bias.dim() == 3 and bus_bias.shape[0] != h2.shape[0]:
            bus_bias = bus_bias.reshape(h2.shape[0], 1, bus_bias.shape[-1])
        if self.normalize:
            logits: torch.Tensor = self.forward(h2, bus_bias=bus_bias)   # (N,V), 2D-in -> 2D-out
            return torch.gather(logits, 1, t[:, None]).squeeze(1)
        zt: torch.Tensor = self._gates(h2, bus_bias=bus_bias)             # (N,K)
        u, _base = self._su(zt)                                           # (N,K) log-odds
        c: torch.Tensor = self.codes[t].to(u.dtype)
        lp: torch.Tensor = (c * F.logsigmoid(u) + (1 - c) * F.logsigmoid(-u)).sum(-1)
        return lp + self.token_bias[t]


class CognitiveCodedHead(nn.Module):
    def __init__(self, cfg: EVAConfig, embed_basis: Optional[nn.Parameter] = None, k_mirror: int = 32) -> None:
        super().__init__()
        codes: torch.Tensor = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K: int = codes.shape[1]
        self.S: int = cfg.code_sparsity
        self.d: int = cfg.D // self.K
        self.vocab: int = cfg.vocab
        self.normalize: bool = bool(getattr(cfg, 'head_normalize', True))
        self._k_mirror: int = k_mirror
        self.register_buffer('codes', codes, persistent=False)
        if embed_basis is not None:
            self.readout = embed_basis
            self.tie_readout: bool = True
        else:
            self.readout: nn.Parameter = nn.Parameter(torch.randn(self.K, self.d))
            nn.init.xavier_uniform_(self.readout, gain=0.5)
            self.tie_readout = False
        self.log_temp_base: nn.Parameter = nn.Parameter(torch.zeros(self.K))
        self.w_res: nn.Parameter = nn.Parameter(torch.tensor(0.5))
        self.w_stab: nn.Parameter = nn.Parameter(torch.tensor(0.1))
        prop: torch.Tensor = codes.float().mean(dim=0).clamp(1e-7, 1 - 1e-7)
        self.bit_bias: nn.Parameter = nn.Parameter(torch.log(prop / (1 - prop)))
        self.W_q_prior: nn.Parameter = nn.Parameter(torch.randn(self.d, 1) * 0.01)
        self.W_k_prior: nn.Parameter = nn.Parameter(torch.randn(k_mirror, 1) * 0.01)
        self.alpha_prior: nn.Parameter = nn.Parameter(torch.tensor(0.2))
        self.w_prior_scale: nn.Parameter = nn.Parameter(torch.ones(1))
        self.beta_social: nn.Parameter = nn.Parameter(torch.tensor(0.1))
        self.w_energy: nn.Parameter = nn.Parameter(torch.tensor(0.1))
        self.resonance_floor: float = 0.5
        self.gamma: nn.Parameter = nn.Parameter(torch.tensor(0.1))
        self.W_code_mod: nn.Parameter = nn.Parameter(torch.randn(self.K, self.d, 1) * 0.01)
        self.token_shift_embed: nn.Embedding = nn.Embedding(cfg.vocab, 8)
        self.proj_shift: nn.Linear = nn.Linear(8, self.K, bias=False)
        nn.init.normal_(self.token_shift_embed.weight, std=0.01)
        self.token_bias: nn.Parameter = nn.Parameter(torch.zeros(cfg.vocab))
        self._pred_error: Optional[torch.Tensor] = None
        self._private_mem: Optional[torch.Tensor] = None
        self._trust_matrix: Optional[torch.Tensor] = None
        self._contra_graph: Optional[torch.Tensor] = None
        self._dominance: Optional[torch.Tensor] = None

    def set_cognitive_state(self, pred_error: Optional[torch.Tensor] = None, private_mem: Optional[torch.Tensor] = None,
                            trust_matrix: Optional[torch.Tensor] = None, contra_graph: Optional[torch.Tensor] = None, dominance: Optional[torch.Tensor] = None) -> None:
        self._pred_error = pred_error
        self._private_mem = private_mem
        self._trust_matrix = trust_matrix
        self._contra_graph = contra_graph
        self._dominance = dominance

    def _compute_z(self, h: torch.Tensor, B: int, L: int, device: torch.device,
                   bus_bias: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        h_g: torch.Tensor = h.reshape(B, L, self.K, self.d)
        z_raw: torch.Tensor = (h_g * self.readout.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        if self._pred_error is not None:
            pe: torch.Tensor = self._pred_error.float()
            e_pred: torch.Tensor = pe.mean(dim=(0, 1)) if pe.ndim > 1 else pe
        else:
            e_pred = torch.zeros(self.K, device=device, dtype=h.dtype)
        if self._private_mem is not None and self._private_mem.shape[1] == self._k_mirror:
            stab: torch.Tensor = self._private_mem.float().var(dim=1)
        else:
            stab = torch.zeros(self.K, device=device, dtype=h.dtype)
        tau: torch.Tensor = self.log_temp_base + self.w_res * e_pred - self.w_stab * stab
        T: torch.Tensor = torch.exp(tau).clamp(0.3, 5.0)
        if self._private_mem is not None and self._private_mem.shape[1] == self._k_mirror:
            pm: torch.Tensor = self._private_mem.float()
        else:
            pm = torch.zeros(self.K, self._k_mirror, device=device, dtype=h.dtype)
        key: torch.Tensor = pm @ self.W_k_prior
        query: torch.Tensor = torch.matmul(h_g, self.W_q_prior).squeeze(-1)
        attn: torch.Tensor = torch.matmul(query, key).squeeze(-1)
        prior: torch.Tensor = self.bit_bias + self.alpha_prior * torch.tanh(attn.unsqueeze(-1) * self.w_prior_scale)
        if self._dominance is not None:
            dom: torch.Tensor = self._dominance.float()
        else:
            dom = torch.ones(self.K, device=device, dtype=h.dtype)
        if self._contra_graph is not None:
            contra_avg: torch.Tensor = self._contra_graph.float().mean(dim=1)
        else:
            contra_avg = torch.zeros(self.K, device=device, dtype=h.dtype)
        social_bias: torch.Tensor = torch.tanh(self.beta_social * (dom - contra_avg))
        if self.tie_readout and self.readout.ndim == 2 and self.readout.shape[-1] == self.d:
            wb: torch.Tensor = self.readout.detach()
            energy: torch.Tensor = ((h_g - wb.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)
            res: torch.Tensor = (1.0 + self.w_energy * torch.tanh(-energy)).clamp(self.resonance_floor, 2.0)
        else:
            res = 1.0
        ctx: torch.Tensor = torch.tanh((h_g * self.W_code_mod.squeeze(-1).unsqueeze(0).unsqueeze(0)).sum(dim=-1))
        z: torch.Tensor = z_raw * res
        z = z / T.unsqueeze(0).unsqueeze(0)
        z = z * (1.0 + self.gamma * ctx)
        z = z + prior + social_bias.unsqueeze(0).unsqueeze(0)
        if bus_bias is not None:
            # intent-bus phase-2 stencil: per-bit bias, same convention as
            # SigmoidCodedHead._gates (applied before the base normalization).
            z = z + bus_bias.reshape(B, L, self.K)
        base: torch.Tensor = F.logsigmoid(-z).sum(dim=-1)
        return z, base

    def _shift_all(self) -> torch.Tensor:
        delta: torch.Tensor = self.proj_shift(self.token_shift_embed.weight)
        return (self.codes * delta).sum(dim=1)

    def _shift_targets(self, token_ids: torch.Tensor) -> torch.Tensor:
        delta: torch.Tensor = self.proj_shift(self.token_shift_embed(token_ids))
        c: torch.Tensor = self.codes[token_ids]
        return (c * delta).sum(dim=-1)

    def forward(self, h: torch.Tensor, bus_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        h2d = h.dim() == 2
        if h2d:
            h = h.unsqueeze(1)
        B, L, _ = h.shape
        z, base = self._compute_z(h, B, L, h.device, bus_bias=bus_bias)
        raw: torch.Tensor = z @ self.codes.T + base.unsqueeze(-1) + self.token_bias + self._shift_all()
        if self.normalize:
            raw = raw - raw.logsumexp(dim=-1, keepdim=True)
        return raw.squeeze(1) if h2d else raw

    def log_probs_for_target(self, h: torch.Tensor, targets: torch.Tensor, bus_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # losses.compute_losses flattens to (N,D): accept 2D/3D uniformly and
        # gather per position (audit M1 — this head used to crash on 2D).
        t: torch.Tensor = targets.reshape(-1)
        if h.dim() == 2:
            h = h.unsqueeze(1)
        B, L, _ = h.shape
        z, base = self._compute_z(h, B, L, h.device, bus_bias=bus_bias)
        c: torch.Tensor = self.codes[t].to(z.dtype)
        zf, basef = z.reshape(-1, self.K), base.reshape(-1)
        score: torch.Tensor = (c * zf).sum(dim=-1) + basef + self.token_bias[t] + self._shift_targets(t)
        if not self.normalize:
            return score
        logits: torch.Tensor = zf @ self.codes.T + basef.unsqueeze(-1) + self.token_bias + self._shift_all()
        logZ: torch.Tensor = logits.logsumexp(dim=-1)
        return score - logZ


# ─── Grouped Cognitive Mirror (32 эксперта) ────────────────────────────