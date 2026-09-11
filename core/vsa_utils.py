"""EVA: vsa_utils module."""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import EVAConfig

def dct_basis(n):
    """DCT-II basis vectors of shape (n, n) — orthogonal rows."""
    k = torch.arange(n, dtype=torch.float32)
    v = k.unsqueeze(1) * (k.unsqueeze(0) + 0.5)
    basis = torch.cos(v * math.pi / n)
    basis[0, :] = basis[0, :] / math.sqrt(2)
    return basis * math.sqrt(2.0 / n)



def zeckendorf_codes(vocab=50000):
    """Fibonacci Zeckendorf binary codes for vocab tokens.
    Возвращает (V, K≈23) — длина кода зависит от vocab.
    """
    fib = [1, 2]
    while fib[-1] <= vocab:
        fib.append(fib[-1] + fib[-2])
    fib = fib[:-1]
    K = len(fib)
    codes = torch.zeros(vocab, K)
    for i in range(vocab):
        n = i + 1
        for j in range(K - 1, -1, -1):
            if n >= fib[j]:
                codes[i, j] = 1.0
                n -= fib[j]
    return codes



def fib_sigmoid_init(n, fib_vals=None):
    """Fibonacci-based sigmoid bias initialization.
    
    Returns bias tensor such that sigmoid(b_i) = fib_vals[i] / sum(fib_vals).
    Если fib_vals=None, используется ряд Фибоначчи длины n: [1,1,2,3,5,...].
    """
    if fib_vals is None:
        f = [1, 1]
        while len(f) < n:
            f.append(f[-1] + f[-2])
        fib_vals = f[:n]
    fib = torch.tensor(fib_vals, dtype=torch.float32)
    p = fib / fib.sum()
    bias = torch.log(p / (1 - p + 1e-10))
    return bias



_CODES_CACHE = {}


def twin_free_codes(vocab=65536, K=64, S=6, max_overlap=None, seed=42, batch=2048):
    """B2 (audit A): random constant-weight codes carry overlap-(S−1) 'twins'
    (215 504 pairs at K=32/S=6/65 536) — recall measured a CLIFF at T≈600
    while SNR≈3.6 still looked healthy: margin 1 bit, not noise, kills it.
    Greedy constant-weight packing with max pairwise overlap ≤ S−2 doubles the
    margin and halves mean intersection; measured knee T: 550 → ≥1200 at equal
    capacity. Deterministic (seed). Raises if the pool cannot fit — the K=32
    pool genuinely can't hold 65k with overlap ≤ S−2, so codebook='twin_free'
    goes together with code_dim=64 (C(64,6)=74.9M).
    """
    from math import comb
    if max_overlap is None:
        max_overlap = S - 2
    total = comb(K, S)
    if vocab > total:
        raise ValueError(f'twin_free_codes: vocab {vocab} > C({K},{S}) = {total}')
    key = ('tf', int(vocab), int(K), int(S), int(max_overlap), int(seed))
    if key in _CODES_CACHE:
        return _CODES_CACHE[key].clone()
    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros(0, K)
    empty_rounds = 0
    while acc.shape[0] < vocab:
        cand = torch.rand(batch, K, generator=g).argsort(dim=1)[:, :S]
        cm = torch.zeros(batch, K)
        cm.scatter_(1, cand, 1.0)
        ok = torch.ones(batch, dtype=torch.bool)
        if acc.shape[0]:
            for c0 in range(0, acc.shape[0], 4096):      # chunked screening: ≤2k×4k window
                ov = cm @ acc[c0:c0 + 4096].T
                ok &= (ov.max(dim=1).values <= max_overlap)
                if not bool(ok.any()):
                    break
        sub_i = torch.nonzero(ok).squeeze(1)
        if sub_i.numel() > 1:                            # intra-batch greedy
            sub = cm[sub_i]
            pw = sub @ sub.T
            pw.fill_diagonal_(K + 1)                    # keep self-overlap harmless for max? no: self=S ≤ S−2 false
            pw.fill_diagonal_(0)
            bad = pw.max(dim=1).values > max_overlap
            sub_i = sub_i[~bad]
        need = vocab - acc.shape[0]
        add = cm[sub_i[:need]] if sub_i.numel() else cm[:0]
        acc = torch.cat([acc, add], dim=0)
        if add.shape[0] == 0:
            empty_rounds += 1
            if empty_rounds > 30:
                break
    if acc.shape[0] < vocab:
        raise ValueError(f'twin_free_codes: only {acc.shape[0]}/{vocab} codes fit '
                         f'overlap≤{max_overlap} at K={K},S={S} — enlarge K')
    _CODES_CACHE[key] = acc
    return acc.clone()


def build_codes(cfg):
    """cfg.codebook dispatcher: 'legacy' (combinadic, balanced by construction)
    or 'twin_free' (B2 constant-weight packing, max overlap S−2)."""
    cb = getattr(cfg, 'codebook', 'legacy')
    if cb == 'twin_free':
        return twin_free_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
    return sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)


def sparse_block_codes(vocab=50000, K=32, S=6):
    """Sparse block codes: ровно S единиц из K на каждый токен.
    
    Использует комбинаторную систему счисления (combinadic) с
    фиксированной случайной перестановкой, чтобы все K бит были
    равномерно представлены среди vocab токенов.
    
    Гарантии:
      - C(K, S) ≥ vocab     (C(32,6)=906192 ≥ 50000 ✓)
      - Ровно S=6 активных бит на каждый токен
      - Каждый бит активен у ≈ vocab·S/K токенов (≈ 9375)
      - Детерминированность (seed=42)
    """
    from math import comb
    total = comb(K, S)
    if vocab > total:   # B1: loud instead of IndexError deep inside combinadic
        raise ValueError(f'sparse_block_codes: vocab {vocab} > C({K},{S}) = {total}; '
                         f'enlarge K or reduce S')
    key = (int(vocab), int(K), int(S))
    if key in _CODES_CACHE:
        return _CODES_CACHE[key].clone()
    # Фиксированная случайная перестановка всех C(K, S) индексов
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(42))
    codes = torch.zeros(vocab, K)
    for v in range(vocab):
        idx = int(perm[v].item())
        n = idx
        for i in range(S, 0, -1):
            c = i - 1
            while comb(c + 1, i) <= n:
                c += 1
            codes[v, c] = 1.0
            n -= comb(c, i)
    _CODES_CACHE[key] = codes
    return codes.clone()


# ─── VSA Prefix Scan ───────────────────────────────────────────────────


def vsa_prefix_scan(a, b, state=None):
    """Associative parallel prefix scan for VSA memory (chunked for stability).
    mem[t] = a[t] * mem[t-1] + b[t]  (element-wise)
    
    a: (B, L, D) or (B, L) — decay factors
    b: (B, L, D) — input increments
    state: (B, D) — initial state or None
    
    Returns: (B, L, D) full scan, (B, D) final state
    """
    B, L, D = b.shape
    if a.dim() == 2:
        a = a.unsqueeze(-1).expand(-1, -1, D)
    
    eps = 1e-6
    CHUNK = 32
    out = []
    s = state.clone() if state is not None else None
    for start in range(0, L, CHUNK):
        end = min(start + CHUNK, L)
        b_chunk = b[:, start:end]
        a_chunk = a[:, start:end]
        
        log_a_chunk = torch.log(a_chunk.clamp(min=eps)).double()
        log_cum_chunk = torch.cumsum(log_a_chunk, dim=1)
        cum_decay_chunk = torch.exp(log_cum_chunk)
        inv_cum_decay_chunk = 1.0 / cum_decay_chunk

        weighted = b_chunk.to(torch.double) * inv_cum_decay_chunk
        cum_weighted = torch.cumsum(weighted, dim=1)

        if s is not None:
            s = s.to(torch.double)
            result_chunk = cum_decay_chunk * s.unsqueeze(1) + cum_decay_chunk * cum_weighted
        else:
            result_chunk = cum_decay_chunk * cum_weighted

        out.append(result_chunk.to(b_chunk.dtype))
        s = result_chunk[:, -1]
    
    result = torch.cat(out, dim=1)
    return result, result[:, -1]


# ─── Embedding ──────────────────────────────────────────────────────────
