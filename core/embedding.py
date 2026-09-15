"""EVA: embedding module."""

from __future__ import annotations

import math, os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import EVAConfig
from .vsa_utils import zeckendorf_codes, sparse_block_codes, build_codes
from .adaptive_gate import hybrid_gate
from .phantom import PhantomBank


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
        codes: torch.Tensor = build_codes(cfg)
        self.K: int = codes.shape[1]
        self.register_buffer('codes', codes, persistent=False)
        # B9 (audit 02a measured): sparse 0/1 codes share a large per-bit DC
        # (mean = S/K); the near-identity mix carries it into every embedding
        # -> mean pairwise cosine 0.950 (95% of an embedding is ONE shared
        # vector). Centering the codes removes it. Constant shift absorbed by
        # the head bias => roundtrip preserved (measured 1.000 either way).
        
        D: int = cfg.D
        assert D % self.K == 0, f'D={D} must be divisible by K={self.K}'
        d: int = D // self.K
        
        # Rank expansion: mixing matrix M (K×K) с ортогональной инициализацией
        # codes → sigmoid(M·codes) даёт плотные коэффициенты, каждый бит влияет на все сегменты
        self.embed_mix: nn.Parameter = nn.Parameter(torch.zeros(self.K, self.K))
        # B2 (audit A): with a random orthogonal M the embed→head roundtrip
        # measured top-1 = 0.000 EVEN AT T=1 (SNR 9e12): the bit-sum decoder
        # cannot invert random mixing, so the identity path starts dead and
        # must be un-learned from noise. Near-identity init: σ(2Mc) keeps the
        # code support (z_k high ⇔ k∈c) while the 0.05G perturbation breaks
        # exact code symmetry. Measured: T=1 roundtrip 0.000 → 1.000.
        nn.init.eye_(self.embed_mix)
        _mg = torch.Generator().manual_seed(7)
        with torch.no_grad():
            self.embed_mix.add_(torch.randn(self.K, self.K, generator=_mg) * 0.05)
        self.register_buffer('_mix_scale', torch.tensor(2.0), persistent=False)
        self.embed_center: bool = bool(getattr(cfg, 'embed_center', False))
        if self.embed_center:
            with torch.no_grad():
                _cs = codes[:8192].to(torch.float32)
                _act = torch.sigmoid(_cs @ self.embed_mix * self._mix_scale)
                self.register_buffer('_sig_mean', _act.mean(0, keepdim=True), persistent=False)
        # B9 (audit 02a): the 95% common-mode of the embedding is the DC of the
        # DENSE sigmoid coefficients (all-positive), NOT of the sparse codes —
        # centering codes left cos 0.935. Estimate the codebook-mean sigmoid
        # activation (fixed 8k-code sample) and subtract it per forward.
        
        _bgen = torch.Generator().manual_seed(5)
        self.basis: nn.Parameter = nn.Parameter(_orth_rows(torch.randn(self.K, d, generator=_bgen)))
        self._embed_rope_on = bool(getattr(cfg, 'embed_rope', False))  # B2
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
        if self.embed_center:
            # Adaptive: _sig_mean starts as the codebook mean of the initial
            # mix, then EMA-tracks the current coefficient mean as embed_mix
            # trains (a static estimate would go stale with the geometry).
            if self.training:
                with torch.no_grad():
                    self._sig_mean.mul_(0.999).add_(
                        codes.detach().reshape(-1, self.K).mean(0) * 0.001)
            codes = codes - self._sig_mean.to(codes.dtype)
        B, L = tokens.shape
        # Внешнее произведение вместо einsum (стабильно под AMP на любых GPU)
        out: torch.Tensor = (codes.unsqueeze(-1) * self.basis.view(1, 1, self.K, -1)).reshape(B, L, -1)
        # B2 (agent-A identity-path finding): RoPE-on-embedding is off by default.
        # In an attention-free trunk positions are intrinsic to the stream (conv,
        # scan read the time axis directly); the rope tag bought nothing and it
        # broke the basis tying: the head computes <R_t e, r>, scrambling the
        # code roundtrip (measured top-1 0.03). Kept behind cfg.embed_rope for A/B.
        if self._embed_rope_on:
            out = self.rope(out)
        return out



def _orth_rows(A: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """B2 (agent-A): segment-addressed code reads need <B_i, B_k> to vanish for
    i != k — otherwise the per-bit margin (z_hi - z_lo)*||B||^2 (~0.3) drowns in
    cross-talk (measured: tied-basis code top-1 = 0.03 at init). Orthonormal
    rows when K <= d (QR of the transposed block), unit rows otherwise."""
    K, d = A.shape
    if K <= d:
        Q, _R = torch.linalg.qr(A.T)
        A = Q.T.contiguous()
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return A * (gain ** 0.5)


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





def _readout_rotated(head: nn.Module, B: int, L: int, D: int) -> torch.Tensor:
    """B2 (agent-A roundtrip finding): the embedding applies RoPE AFTER
    z⊗basis, so the TIED readout sitting unrotated computes ⟨R·e, r⟩ — not
    position-invariant (measured code top-1 = 0.000 even at T=1). Rotating the
    readout with the SAME rope recovers ⟨R e, R r⟩ = ⟨e, r⟩ exactly because R
    is orthogonal; the code identity path is then alive at initialization."""
    r = head.readout                                   # (K, d)
    big = torch.zeros(1, L, D, device=r.device, dtype=r.dtype)
    big.view(1, L, head.K, -1).copy_(r.view(1, 1, head.K, -1))
    rp = getattr(head, '_embed_rope', None)
    if rp is None:
        return big.reshape(1, L, head.K, -1)          # broadcasts over B
    big = rp(big.expand(B, L, D))
    return big.reshape(B, L, head.K, -1)


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
    def __init__(self, cfg: EVAConfig, embed_basis: Optional[nn.Parameter] = None,
                 rope: Optional[nn.Module] = None) -> None:
        super().__init__()
        codes: torch.Tensor = build_codes(cfg)
        self.K: int = codes.shape[1]
        self.register_buffer('codes', codes, persistent=False)
        self._embed_rope = rope
        
        D: int = cfg.D
        assert D % self.K == 0
        d: int = D // self.K
        
        if embed_basis is not None:
            self.readout = embed_basis  # shared reference
        else:
            _rgen = torch.Generator().manual_seed(6)
            self.readout: nn.Parameter = nn.Parameter(_orth_rows(torch.randn(self.K, d, generator=_rgen)))
        self.token_bias: nn.Parameter = nn.Parameter(torch.zeros(cfg.vocab))
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        h_g: torch.Tensor = h.reshape(B, L, self.K, -1)  # (B, L, K, d)
        scores: torch.Tensor = (h_g * self.readout.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return scores @ self.codes.T + self.token_bias.unsqueeze(0).unsqueeze(0)


class SigmoidCodedHead(nn.Module):
    def __init__(self, cfg: EVAConfig, embed_basis: Optional[nn.Parameter] = None,
                 rope: Optional[nn.Module] = None) -> None:
        super().__init__()
        codes: torch.Tensor = build_codes(cfg)
        self.K: int = codes.shape[1]
        self.S: int = cfg.code_sparsity
        self._embed_rope = rope
        self.register_buffer('codes', codes, persistent=False)
        D: int = cfg.D
        self.D: int = D
        assert D % self.K == 0
        d: int = D // self.K
        if embed_basis is not None:
            self.readout = embed_basis
        else:
            _rgen = torch.Generator().manual_seed(6)
            self.readout: nn.Parameter = nn.Parameter(_orth_rows(torch.randn(self.K, d, generator=_rgen)))
        prop: torch.Tensor = codes.mean(dim=0)
        self.register_buffer('_prop', prop)
        # Code-prior init (the _prop buffer was computed and left dead): with
        # bit_bias = logit(p_active) the head starts at the unigram prior of
        # the sparse block code, not at an arbitrary 0.5 per bit.
        _p = prop.clamp(1e-7, 1 - 1e-7)
        self.bit_bias: nn.Parameter = nn.Parameter(torch.log(_p / (1 - _p)))
        self.log_temp: nn.Parameter = nn.Parameter(torch.zeros(self.K))
        # M52a: learnable gain on the softmax emphasis (init 1 = the old
        # behavior bit-for-bit; the model sizes the competition itself).
        self.emphasis_gain: nn.Parameter = nn.Parameter(torch.ones(1))
        # M52b: the lacuna + phantom channel. The lacuna is the part of h that
        # is orthogonal to EVERY readout direction (mathematically invisible to
        # the known bits; verified |<e_l, R_k>| ~ 7e-7). A separate phantom
        # basis reads it, gated by the lacuna magnitude; the zero-init mix makes
        # the forward start bit-identical to the known-bits-only head.
        _Kp = int(getattr(cfg, 'head_phantom_bits', 32)) if getattr(cfg, 'head_lacuna', True) else 0
        self.Kp: int = max(0, min(_Kp, D))
        if self.Kp > 0:
            _pgen = torch.Generator().manual_seed(7)
            self.phantom_basis: nn.Parameter = nn.Parameter(
                _orth_rows(torch.randn(self.Kp, D, generator=_pgen)))
            self.phantom_mix: nn.Parameter = nn.Parameter(torch.zeros(self.K, self.Kp))
            self.lacuna_w: nn.Parameter = nn.Parameter(torch.tensor(30.0))
            self.lacuna_b: nn.Parameter = nn.Parameter(torch.tensor(-3.0))
            # M55b: the lacuna self-calibration. The ABSOLUTE ell is ~0.97 for
            # any realistic state (the readout spans K of D dims, so the
            # orthogonal remainder always dominates); the gate must read the
            # RELATIVE novelty ell/EMA(ell). ell_ema is non-persistent (it
            # re-calibrates within ~1000 steps after a resume).
            self.lacuna_ema: float = float(getattr(cfg, 'head_lacuna_ema', 0.99))
            self.register_buffer('ell_ema', torch.zeros(1), persistent=False)
            self._noise_gen = None
            self.log_eta: nn.Parameter = nn.Parameter(torch.tensor(
                math.log(max(float(getattr(cfg, 'head_phantom_noise', 0.05)), 1e-4))))
            # M54: the phantom-concept bank (the EVA-Ai lacuna lifecycle at
            # hidden-state level). Buffers ride in the checkpoint.
            self.phantom_bank = PhantomBank(
                n_slots=int(getattr(cfg, 'head_phantom_slots', 16)), D=D)
            self.phantom_thr: float = float(getattr(cfg, 'head_phantom_thr', 0.1))
            self.phantom_every: int = max(1, int(getattr(cfg, 'head_phantom_every', 25)))
            self.register_buffer('_pb_step', torch.zeros(1, dtype=torch.long), persistent=False)
        # M53: the State Resolution Loop knobs (off by default).
        self.srl_on: bool = bool(getattr(cfg, 'head_srl', False))
        self.srl_steps: int = int(getattr(cfg, 'head_srl_steps', 3))
        self.srl_shortlist: int = int(getattr(cfg, 'head_srl_shortlist', 64))
        self.srl_expl_thr: float = float(getattr(cfg, 'head_srl_expl_thr', 0.7))
        # M53c: warmups — the stack flips these per forward from the live step.
        self.srl_after: int = int(getattr(cfg, 'head_srl_after', 1045))
        self.phantom_after: int = int(getattr(cfg, 'head_phantom_after', 1045))
        self._srl_active: bool = self.srl_on
        self._pb_active: bool = True
        # M55a: the contradiction tempering (head<->memory). The stack stashes
        # the memory read direction in _mem_dir; the head compares it with the
        # direction its own bits imply and softens the logits on a conflict.
        self.temper_on: bool = bool(getattr(cfg, 'head_temper', True))
        self.temper_k: float = float(getattr(cfg, 'head_temper_k', 0.5))
        self.temper_cos: float = float(getattr(cfg, 'head_temper_cos', 0.3))
        self.temper_after: int = int(getattr(cfg, 'head_temper_after', 1045))
        self._temper_active: bool = self.temper_on
        self.token_bias: nn.Parameter = nn.Parameter(torch.zeros(cfg.vocab))
        self.normalize: bool = bool(getattr(cfg, 'head_normalize', True))

    def _gates(self, h: torch.Tensor, temp_factor: Optional[torch.Tensor] = None,
               bus_bias: Optional[torch.Tensor] = None, return_data: bool = False):
        if h.dim() == 2:
            h = h.unsqueeze(1)
            squeeze: bool = True
        else:
            squeeze = False
        B, L, D = h.shape
        h_g: torch.Tensor = h.reshape(B, L, self.K, -1)
        z: torch.Tensor = (h_g * self.readout.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        # M52a: ST clamp — forward identical, backward identity (a tau at a
        # rail keeps a nonzero gradient; the old clamp froze log_temp).
        T: torch.Tensor = torch.exp(self.log_temp)
        T = T + (T.clamp(0.1, 10.0) - T).detach()
        if temp_factor is not None:
            T = T * temp_factor
        z_data: torch.Tensor = z / T          # M52a: the emphasis source
        zt: torch.Tensor = z_data + self.bit_bias
        if bus_bias is not None:
            # Phase-2 stencil: cross-layer gist biases the projector readout.
            # bus_bias shape matches zt's (B,L,K) or (N,1,K) -> broadcasts cleanly.
            zt = zt + bus_bias
        if return_data:
            # M52b: the lacuna — the per-block orthogonal residual. It is
            # invisible to the readout by construction (the known bits cannot
            # represent it), which is exactly why the phantom basis exists.
            e_l = h - (z.unsqueeze(-1) * self.readout).reshape(B, L, self.D)
            if squeeze:
                zt = zt.squeeze(1)
                z_data = z_data.squeeze(1)
                e_l = e_l.squeeze(1)
            return zt, z_data, e_l
        if squeeze:
            zt = zt.squeeze(1)
        return zt

    def _su(self, zt: torch.Tensor, z_data: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Single per-bit temperature: T already scaled z in _gates, so the
        # emphasis softmax reads the SAME logits (passing tau=exp(log_temp)
        # here again divided twice: z/T/tau = z/T^2 — audit M1).
        # M52a (P3): the emphasis reads the DATA part (z/T) only — the old form
        # used zt, so the static prior reinforced itself through the softmax
        # (measured corr(bit_bias, emphasis bonus) = +0.90).
        # M52a (P2): the gain (init 1) lets the model size the competition.
        u, base = hybrid_gate(zt, 1.0, log=True, emph_logits=z_data,
                              gain=self.emphasis_gain)
        if self.training:
            self._last_u = u        # M52a (P1): the saturation wall reads this
            self._last_sat = (u.detach().abs() > 12.0).float().mean()   # M55a (P6)
        return u, base

    def srl(self, u0: torch.Tensor, steps: int = None, tau0: float = 1.0,
            gamma: float = 0.6, alpha: float = 0.7, shortlist: int = None):
        """M53: State Resolution Loop — annealed EM over the code dictionary.

        E: p(v|u) ∝ exp( u·(2C_v−1)/τ_t )   the softmax commitment
        M: ĉ = Σ p·C_v                       the expected code
        refine: u ← u + α(logit(ĉ) − u)      known + the sigmoid's start
        τ_t = τ0·γ^t                         the resolution sharpens

        Classification (verified on the prototype: a clean code -> conf 1.0,
        expl 0.03 nats/bit; a mix of two codes -> conf 0.50; random noise ->
        expl 1.08 nats/bit):
          concept       : max_p > 0.9 and expl < thr
          contradiction : max_p <= 0.9 (a split posterior)
          lacuna        : expl >= thr (the bits must be rewritten to fit)

        The explanation cost = the NLL of the ORIGINAL bits under the found
        code, per bit — scale-free. The candidate set is a shortlist taken from
        the first pass (cheap; the refinement never leaves it).
        """
        steps = self.srl_steps if steps is None else int(steps)
        M = min(int(self.srl_shortlist if shortlist is None else shortlist), self.codes.shape[0])
        if steps <= 0 or M <= 0:
            z = u0.new_zeros(u0.shape[:-1])
            return u0, {'conf': z + 1.0, 'ent': z, 'expl': z}
        shape = u0.shape
        uf = u0.reshape(-1, shape[-1])
        C = self.codes.float()
        Sc = 2.0 * C - 1.0
        idx = (uf @ Sc.T).topk(M, dim=-1).indices          # (N,M) shortlist
        Cs = C[idx]                                        # (N,M,K)
        Ss = Sc[idx]
        tau = tau0
        for t in range(steps):
            p = F.softmax((uf.unsqueeze(1) * Ss).sum(-1) / max(tau, 0.05), dim=-1)
            chat = (p.unsqueeze(-1) * Cs).sum(1)
            uf = uf + alpha * (torch.logit(chat.clamp(1e-4, 1 - 1e-4)) - uf)
            tau = tau * gamma
        p = F.softmax((uf.unsqueeze(1) * Ss).sum(-1) / max(tau, 0.05), dim=-1)
        star = p.argmax(-1)
        cstar = Cs.gather(1, star[:, None, None].expand(-1, 1, C.shape[-1])).squeeze(1)
        a0 = torch.sigmoid(uf.new_zeros(()) + u0.reshape(-1, shape[-1]))
        nll = -(cstar * F.logsigmoid(u0.reshape(-1, shape[-1]))
                + (1 - cstar) * F.logsigmoid(-u0.reshape(-1, shape[-1]))).sum(-1) / C.shape[-1]
        info = {'conf': p.max(-1).values.reshape(shape[:-1]),
                'ent': (-(p * (p + 1e-9).log()).sum(-1)).reshape(shape[:-1]),
                'expl': nll.reshape(shape[:-1])}
        return uf.reshape(shape), info

    def _phantom_mix(self, u: torch.Tensor, e_l: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """M52b: lacuna -> potential state. The residual (optionally noised by
        the learnable eta — the EVA-Ai exploration) is read by the phantom
        basis, gated by the lacuna magnitude and mixed into the known bits.
        Zero-init mix => the forward is identity at init."""
        if self.Kp <= 0:
            return u
        _hn = h.reshape(e_l.shape).norm(dim=-1, keepdim=True) + 1e-6
        ell = e_l.norm(dim=-1, keepdim=True) / _hn
        # M55b: the RELATIVE novelty (self-calibrating): ~1 for the running
        # level, > 1 for a spike. The gate and the bank read this, not ell.
        with torch.no_grad():
            if float(self.ell_ema) <= 0.0:
                self.ell_ema.fill_(float(ell.detach().mean()))
            else:
                self.ell_ema.mul_(self.lacuna_ema).add_(
                    ell.detach().mean(), alpha=1.0 - self.lacuna_ema)
        ell_rel = (ell / (self.ell_ema + 1e-6)).clamp(0.0, 5.0)
        if self.training:
            _eta = torch.exp(self.log_eta).clamp(0.0, 0.2)
            e_in = e_l + _eta * self._noise_like(e_l)
        else:
            e_in = e_l
        p = torch.tanh(e_in @ self.phantom_basis.T)          # (...,Kp)
        g = torch.sigmoid(self.lacuna_w * (ell_rel - 1.0) + self.lacuna_b)
        p = p * g
        if self.training:
            self._last_lacuna = ell.detach().mean()
            self._last_lacuna_rel = ell_rel.detach().mean()
            self._last_lacuna_gate = g.detach().mean()
            self._last_p = p                                  # live: the L1 aux
            _pb = getattr(self, 'phantom_bank', None)
            if _pb is not None and getattr(self, '_pb_active', True):
                _pb.decay()
                if int(self._pb_step.item()) % self.phantom_every == 0:
                    _pb.observe(e_l, ell_rel, self.phantom_thr)
                self._pb_step += 1
        return u + p @ self.phantom_mix.T

    def _noise_like(self, x: torch.Tensor) -> torch.Tensor:
        """M55b: the exploration noise from a DEDICATED generator. The head is
        called INSIDE the stack's forward (_last_conf), so a global-RNG draw here
        shifted the scheduled sampling and the concept births of the same step —
        two runs with identical weights then diverged from step 0."""
        g = self._noise_gen
        if g is None or g.device != x.device:
            g = torch.Generator(device=x.device)
            g.manual_seed(1234)
            self._noise_gen = g
        return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)

    def forward(self, h: torch.Tensor, bus_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(1)
            squeeze: bool = True
        else:
            squeeze = False
        zt, z_data, e_l = self._gates(h, bus_bias=bus_bias, return_data=True)
        u, base = self._su(zt, z_data)
        u = self._phantom_mix(u, e_l, h)
        if self.srl_on and getattr(self, '_srl_active', True):
            u, _srl_info = self.srl(u)
            if self.training:
                self._last_srl = {k: v.detach().mean() for k, v in _srl_info.items()}
        # M55a (P5): `base` is EXACTLY dead in the normalized path (a constant
        # over the vocab cancels in the logsumexp — verified 0 gradient), so it
        # is only added when the raw logits are returned. The "unknown" channel
        # now lives in the lacuna (ell), not in this term.
        logits: torch.Tensor = (u @ self.codes.T
                                + (base[..., None] if not self.normalize else 0.0)
                                + self.token_bias)
        if self.temper_on and getattr(self, '_temper_active', True):
            _md = getattr(self, '_mem_dir', None)
            if (_md is not None and _md.shape[-1] == self.D
                    and tuple(_md.shape[:-1]) == tuple(u.shape[:-1])):
                _a = torch.sigmoid(u)
                _h_impl = (_a.unsqueeze(-1) * self.readout).reshape(*u.shape[:-1], self.D)
                _cos = F.cosine_similarity(_h_impl, _md.reshape(_h_impl.shape), dim=-1, eps=1e-6)
                _chi = F.relu(self.temper_cos - _cos).unsqueeze(-1)
                if self.training:
                    self._last_conflict = _chi.detach().mean()
                logits = logits / (1.0 + self.temper_k * _chi)
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
        zt, z_data, e_l = self._gates(h2, bus_bias=bus_bias, return_data=True)  # (N,K)
        u, _base = self._su(zt, z_data)                                         # (N,K) log-odds
        u = self._phantom_mix(u, e_l, h2)
        if self.srl_on and getattr(self, '_srl_active', True):
            u, _srl_info = self.srl(u)
            if self.training:
                self._last_srl = {k: v.detach().mean() for k, v in _srl_info.items()}
        c: torch.Tensor = self.codes[t].to(u.dtype)
        lp: torch.Tensor = (c * F.logsigmoid(u) + (1 - c) * F.logsigmoid(-u)).sum(-1)
        return lp + self.token_bias[t]


class CognitiveCodedHead(nn.Module):
    def __init__(self, cfg: EVAConfig, embed_basis: Optional[nn.Parameter] = None, k_mirror: int = 32,
                 rope: Optional[nn.Module] = None) -> None:
        super().__init__()
        codes: torch.Tensor = build_codes(cfg)
        self.K: int = codes.shape[1]
        self.S: int = cfg.code_sparsity
        self.d: int = cfg.D // self.K
        self.vocab: int = cfg.vocab
        self.normalize: bool = bool(getattr(cfg, 'head_normalize', True))
        self._k_mirror: int = k_mirror
        self.register_buffer('codes', codes, persistent=False)
        self._embed_rope = rope
        if embed_basis is not None:
            self.readout = embed_basis
            self.tie_readout: bool = True
        else:
            _rgen = torch.Generator().manual_seed(6)
            self.readout: nn.Parameter = nn.Parameter(_orth_rows(torch.randn(self.K, self.d, generator=_rgen)))
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