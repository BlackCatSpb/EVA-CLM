"""WideBind: embedding module."""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import WideBindConfig
from .vsa_utils import zeckendorf_codes, sparse_block_codes
from .adaptive_gate import hybrid_gate


class RotaryEmbedding(nn.Module):
    def __init__(self, D, theta=1000000.0, scaling=1.0, max_len=65536):
        super().__init__()
        self.D = D
        self.theta = theta
        self.scaling = scaling
        half = D // 2
        freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer('_freqs', freqs)
        self._max_cached = 0

    def _build_cache(self, L):
        if L <= self._max_cached:
            return
        t = torch.arange(L, dtype=torch.float32, device=self._freqs.device) / self.scaling
        angles = t[:, None] * self._freqs[None, :]
        self._cos_cached = angles.cos()
        self._sin_cached = angles.sin()
        self._max_cached = L

    def forward(self, x):
        B, L, D = x.shape
        self._build_cache(L)
        cos = self._cos_cached[:L].to(x.dtype).to(x.device)
        sin = self._sin_cached[:L].to(x.dtype).to(x.device)
        x0 = x[..., 0::2].contiguous()
        x1 = x[..., 1::2].contiguous()
        out0 = x0 * cos.unsqueeze(0) - x1 * sin.unsqueeze(0)
        out1 = x0 * sin.unsqueeze(0) + x1 * cos.unsqueeze(0)
        out = torch.empty_like(x)
        out[..., 0::2] = out0
        out[..., 1::2] = out1
        return out

class ZeckendorfEmbedding(nn.Module):
    """Token -> D-space via Zeckendorf codes + learned projection.
    
    Legacy: проекция K→D через Linear. Ранг матрицы эмбеддингов ≤ K=23.
    """
    def __init__(self, cfg):
        super().__init__()
        codes = zeckendorf_codes(cfg.vocab)
        K = codes.shape[1]
        self.register_buffer('codes', codes)
        self.proj = nn.Linear(K, cfg.D, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, tokens):
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
    def __init__(self, cfg):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.register_buffer('codes', codes)
        
        D = cfg.D
        assert D % self.K == 0, f'D={D} must be divisible by K={self.K}'
        d = D // self.K
        
        # Rank expansion: mixing matrix M (K×K) с ортогональной инициализацией
        # codes → sigmoid(M·codes) даёт плотные коэффициенты, каждый бит влияет на все сегменты
        self.embed_mix = nn.Parameter(torch.zeros(self.K, self.K))
        nn.init.orthogonal_(self.embed_mix)
        self.register_buffer('_mix_scale', torch.tensor(2.0), persistent=False)
        
        self.basis = nn.Parameter(torch.randn(self.K, d))
        nn.init.xavier_uniform_(self.basis, gain=0.5)
        self._rope_theta = getattr(cfg, 'rope_theta', 1000000.0)
        self._rope_scaling = getattr(cfg, 'rope_scaling', 1.0)
        self.rope = RotaryEmbedding(D, theta=self._rope_theta, scaling=self._rope_scaling)
    
    def forward(self, tokens):
        # Защита от токенов ≥ vocab (device-side assert в gather): фон-клип
        tokens = tokens.clamp(0, self.codes.shape[0] - 1)
        codes = self.codes[tokens]  # (B, L, K), sparse binary
        # Dense mixing: sigmoid(scale · M · codes) → каждый бит влияет на все сегменты
        codes = torch.sigmoid(codes @ self.embed_mix * self._mix_scale)
        B, L = tokens.shape
        # Внешнее произведение вместо einsum (стабильно под AMP на любых GPU)
        out = (codes.unsqueeze(-1) * self.basis.view(1, 1, self.K, -1)).reshape(B, L, -1)
        out = self.rope(out)
        return out



class LmHead(nn.Module):
    """D-space -> vocab logits via Zeckendorf code projection (legacy)."""
    def __init__(self, cfg):
        super().__init__()
        codes = zeckendorf_codes(cfg.vocab)
        K = codes.shape[1]
        self.register_buffer('codes', codes)
        self.proj = nn.Linear(cfg.D, K, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, h):
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
    def __init__(self, cfg, embed_basis=None):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.register_buffer('codes', codes)
        
        D = cfg.D
        assert D % self.K == 0
        d = D // self.K
        
        if embed_basis is not None:
            self.readout = embed_basis  # shared reference
        else:
            self.readout = nn.Parameter(torch.randn(self.K, d))
            nn.init.xavier_uniform_(self.readout, gain=0.5)
        self.token_bias = nn.Parameter(torch.zeros(cfg.vocab))
    
    def forward(self, h):
        B, L, D = h.shape
        h_g = h.reshape(B, L, self.K, -1)  # (B, L, K, d)
        scores = (h_g * self.readout.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return scores @ self.codes.T + self.token_bias.unsqueeze(0).unsqueeze(0)


class SigmoidCodedHead(nn.Module):
    def __init__(self, cfg, embed_basis=None):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.S = cfg.code_sparsity
        self.register_buffer('codes', codes)
        D = cfg.D
        assert D % self.K == 0
        d = D // self.K
        if embed_basis is not None:
            self.readout = embed_basis
        else:
            self.readout = nn.Parameter(torch.randn(self.K, d))
            nn.init.xavier_uniform_(self.readout, gain=0.5)
        prop = codes.mean(dim=0)
        self.register_buffer('_prop', prop)
        self.bit_bias = nn.Parameter(torch.zeros(self.K))
        self.log_temp = nn.Parameter(torch.zeros(self.K))
        self.token_bias = nn.Parameter(torch.zeros(cfg.vocab))
        self.normalize = bool(getattr(cfg, 'head_normalize', True))

    def _gates(self, h, temp_factor=None, bus_bias=None):
        if h.dim() == 2:
            h = h.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        B, L, D = h.shape
        h_g = h.reshape(B, L, self.K, -1)
        z = (h_g * self.readout.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        T = torch.exp(self.log_temp).clamp_min(0.1)
        if temp_factor is not None:
            T = T * temp_factor
        zt = z / T + self.bit_bias
        if bus_bias is not None:
            # Phase-2 stencil: cross-layer gist biases the projector readout.
            # bus_bias shape matches zt's (B,L,K) or (N,1,K) -> broadcasts cleanly.
            zt = zt + bus_bias
        if squeeze:
            zt = zt.squeeze(1)
        return zt

    def _su(self, zt):
        tau = torch.exp(self.log_temp).clamp(0.1, 10.0)
        return hybrid_gate(zt, tau, log=True)

    def forward(self, h, bus_bias=None):
        if h.dim() == 2:
            h = h.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        u, base = self._su(self._gates(h, bus_bias=bus_bias))
        logits = u @ self.codes.T + base[..., None] + self.token_bias
        if self.normalize:
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        if squeeze:
            logits = logits.squeeze(1)
        return logits

    def log_probs_for_target(self, h, targets, bus_bias=None):
        if h.dim() == 2:
            h_2d = True
            h_in = h.unsqueeze(1)
        else:
            h_2d = False
            h_in = h
        if self.normalize:
            raw = self.forward(h_in, bus_bias=bus_bias)
            if h_2d:
                return raw[0, 0, targets] + self.token_bias[targets]
            idx = torch.arange(raw.shape[1], device=raw.device)
            return raw[:, idx, targets] + self.token_bias[targets]
        zt = self._gates(h_in, bus_bias=bus_bias)
        tau = torch.exp(self.log_temp).clamp(0.1, 10.0)
        gate = hybrid_gate(zt, tau)
        gate = gate.clamp(1e-7, 1 - 1e-7)
        ls = torch.log(gate)
        lms = torch.log(1 - gate)
        c = self.codes[targets].float()
        if h_2d:
            c = c.unsqueeze(1)
        logp = (c * ls).sum(-1) + ((1 - c) * lms).sum(-1)
        return logp + self.token_bias[targets]


class CognitiveCodedHead(nn.Module):
    def __init__(self, cfg, embed_basis=None, k_mirror=32):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.S = cfg.code_sparsity
        self.d = cfg.D // self.K
        self.vocab = cfg.vocab
        self.normalize = bool(getattr(cfg, 'head_normalize', True))
        self._k_mirror = k_mirror
        self.register_buffer('codes', codes)
        if embed_basis is not None:
            self.readout = embed_basis
            self.tie_readout = True
        else:
            self.readout = nn.Parameter(torch.randn(self.K, self.d))
            nn.init.xavier_uniform_(self.readout, gain=0.5)
            self.tie_readout = False
        self.log_temp_base = nn.Parameter(torch.zeros(self.K))
        self.w_res = nn.Parameter(torch.tensor(0.5))
        self.w_stab = nn.Parameter(torch.tensor(0.1))
        prop = codes.float().mean(dim=0).clamp(1e-7, 1 - 1e-7)
        self.bit_bias = nn.Parameter(torch.log(prop / (1 - prop)))
        self.W_q_prior = nn.Parameter(torch.randn(self.d, 1) * 0.01)
        self.W_k_prior = nn.Parameter(torch.randn(k_mirror, 1) * 0.01)
        self.alpha_prior = nn.Parameter(torch.tensor(0.2))
        self.w_prior_scale = nn.Parameter(torch.ones(1))
        self.beta_social = nn.Parameter(torch.tensor(0.1))
        self.w_energy = nn.Parameter(torch.tensor(0.1))
        self.resonance_floor = 0.5
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.W_code_mod = nn.Parameter(torch.randn(self.K, self.d, 1) * 0.01)
        self.token_shift_embed = nn.Embedding(cfg.vocab, 8)
        self.proj_shift = nn.Linear(8, self.K, bias=False)
        nn.init.normal_(self.token_shift_embed.weight, std=0.01)
        self.token_bias = nn.Parameter(torch.zeros(cfg.vocab))
        self._pred_error = None
        self._private_mem = None
        self._trust_matrix = None
        self._contra_graph = None
        self._dominance = None

    def set_cognitive_state(self, pred_error=None, private_mem=None,
                            trust_matrix=None, contra_graph=None, dominance=None):
        self._pred_error = pred_error
        self._private_mem = private_mem
        self._trust_matrix = trust_matrix
        self._contra_graph = contra_graph
        self._dominance = dominance

    def _compute_z(self, h, B, L, device):
        h_g = h.reshape(B, L, self.K, self.d)
        z_raw = (h_g * self.readout.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        if self._pred_error is not None:
            pe = self._pred_error.float()
            e_pred = pe.mean(dim=(0, 1)) if pe.ndim > 1 else pe
        else:
            e_pred = torch.zeros(self.K, device=device, dtype=h.dtype)
        if self._private_mem is not None and self._private_mem.shape[1] == self._k_mirror:
            stab = self._private_mem.float().var(dim=1)
        else:
            stab = torch.zeros(self.K, device=device, dtype=h.dtype)
        tau = self.log_temp_base + self.w_res * e_pred - self.w_stab * stab
        T = torch.exp(tau).clamp(0.3, 5.0)
        if self._private_mem is not None and self._private_mem.shape[1] == self._k_mirror:
            pm = self._private_mem.float()
        else:
            pm = torch.zeros(self.K, self._k_mirror, device=device, dtype=h.dtype)
        key = pm @ self.W_k_prior
        query = torch.matmul(h_g, self.W_q_prior).squeeze(-1)
        attn = torch.matmul(query, key).squeeze(-1)
        prior = self.bit_bias + self.alpha_prior * torch.tanh(attn.unsqueeze(-1) * self.w_prior_scale)
        if self._dominance is not None:
            dom = self._dominance.float()
        else:
            dom = torch.ones(self.K, device=device, dtype=h.dtype)
        if self._contra_graph is not None:
            contra_avg = self._contra_graph.float().mean(dim=1)
        else:
            contra_avg = torch.zeros(self.K, device=device, dtype=h.dtype)
        social_bias = torch.tanh(self.beta_social * (dom - contra_avg))
        if self.tie_readout and self.readout.ndim == 2 and self.readout.shape[-1] == self.d:
            wb = self.readout.detach()
            energy = ((h_g - wb.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)
            res = (1.0 + self.w_energy * torch.tanh(-energy)).clamp(self.resonance_floor, 2.0)
        else:
            res = 1.0
        ctx = torch.tanh((h_g * self.W_code_mod.squeeze(-1).unsqueeze(0).unsqueeze(0)).sum(dim=-1))
        z = z_raw * res
        z = z / T.unsqueeze(0).unsqueeze(0)
        z = z * (1.0 + self.gamma * ctx)
        z = z + prior + social_bias.unsqueeze(0).unsqueeze(0)
        base = F.logsigmoid(-z).sum(dim=-1)
        return z, base

    def _shift_all(self):
        delta = self.proj_shift(self.token_shift_embed.weight)
        return (self.codes * delta).sum(dim=1)

    def _shift_targets(self, token_ids):
        delta = self.proj_shift(self.token_shift_embed(token_ids))
        c = self.codes[token_ids]
        return (c * delta).sum(dim=-1)

    def forward(self, h):
        B, L, _ = h.shape
        z, base = self._compute_z(h, B, L, h.device)
        raw = z @ self.codes.T + base.unsqueeze(-1) + self.token_bias + self._shift_all()
        if self.normalize:
            raw = raw - raw.logsumexp(dim=-1, keepdim=True)
        return raw

    def log_probs_for_target(self, h, targets):
        B, L, _ = h.shape
        z, base = self._compute_z(h, B, L, h.device)
        c = self.codes[targets]
        score = (c * z).sum(dim=-1) + base + self.token_bias[targets] + self._shift_targets(targets)
        if not self.normalize:
            return score
        raw = self.forward(h)
        logZ = raw.logsumexp(dim=-1)
        return score - logZ


# ─── Grouped Cognitive Mirror (32 эксперта) ────────────────────────────
