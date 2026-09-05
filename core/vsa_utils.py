"""WideBind: vsa_utils module."""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import WideBindConfig

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
    return codes


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
