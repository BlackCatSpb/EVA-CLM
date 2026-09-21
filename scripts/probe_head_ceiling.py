"""P1-1 (proposed patches): the CEILING of the product-of-Bernoulli head.

The coded head's log-family is q(v|u) ∝ exp(Σ_k c_vk u_k) — a rank-K log-linear
family over the codebook: it cannot express ANY intra-code correlation. Their
own measurement (CE code-only 12.84 > bias-only 8.65) says the bit path is not
yet paying for itself. Before extending the head, measure how much CE the
family loses IN PRINCIPLE at the bigram level:

    (1) unigram CE            — the bias-only floor
    (2) bigram CE (full)      — the exact conditional softmax over successors
    (3) bigram CE under PoB   — the best CE this family can reach (LBFGS over
                                u ∈ R^K per context; CE is convex in u)

    gap = (3) − (2)  = the price of bit independence.

Interpretation (the patch's acceptance): gap < 0.1 nat — the ceiling does not
bind, the trunk is the priority; gap > 0.3 nat — the pairwise channel (P1-2)
is on the critical path.

Memory note: the bigram table is NOT materialised (V² = 4.3e9). Only the
top-`n_ctx` contexts (by unigram frequency) are counted, one at a time.

Run:  python scripts/probe_head_ceiling.py --stream <token_stream.bin> [--n-ctx 64]
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig          # noqa: E402
from core.vsa_utils import build_codes     # noqa: E402


def load_tokens(path: str, max_tokens: int = 0) -> torch.Tensor:
    import numpy as np
    data = np.fromfile(path, dtype=np.uint16).astype(np.int64)
    if max_tokens and data.size > max_tokens:
        data = data[:max_tokens]
    return torch.from_numpy(data)


def top_contexts(tokens: torch.Tensor, vocab: int, n_ctx: int) -> torch.Tensor:
    cnt = torch.bincount(tokens, minlength=vocab)
    return cnt.argsort(descending=True)[:n_ctx]


def successor_row(tokens: torch.Tensor, ctx: int, vocab: int) -> torch.Tensor:
    """(V,) counts of successors of `ctx` (one pass, no V² materialisation)."""
    first, second = tokens[:-1], tokens[1:]
    sel = (first == int(ctx))
    if not bool(sel.any()):
        return torch.zeros(vocab, dtype=torch.float64)
    return torch.bincount(second[sel], minlength=vocab).double()


def fit_pob(target: torch.Tensor, codes: torch.Tensor, iters: int = 200,
            device: str = 'cpu'):
    """Best CE of the family q_u = softmax(C u) against `target` (a prob. row).

    CE(u) = -Σ_v p_v·(C u)_v + logsumexp(C u) — convex in u; LBFGS.
    Returns (ce_best, ce_at_u0) so the caller can see the gain.
    """
    C = codes.to(device)
    p = target.to(device)
    u = torch.zeros(C.shape[1], dtype=C.dtype, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([u], max_iter=iters, line_search_fn='strong_wolfe')

    def ce_of(uu):
        z = C @ uu
        return float(-(p * (z - torch.logsumexp(z, dim=-1))).sum())

    ce0 = ce_of(u.detach())

    def closure():
        opt.zero_grad()
        z = C @ u
        loss = -(p * (z - torch.logsumexp(z, dim=-1))).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return ce_of(u.detach()), ce0


def fit_pob_pair(target: torch.Tensor, codes: torch.Tensor, r: int = 16,
                 iters: int = 150, device: str = 'cpu'):
    """P1-2 ceiling: the rank-r pairwise (Ising) extension of the family.

        logits_v = (C u)_v + Σ_s (c_v·(u⊙α_s))·(c_v·(u⊙β_s))

    α_s, β_s ∈ R^K are the learned pair directions (2Kr params, r=16 → 2048).
    The bit vector is sparse (S active bits), so each factor is a gather of S
    terms — the dense C-form would cost r× the head. V1 starts small-random and
    V2 at zero (an all-zero init is a saddle: the product's V2-gradient is
    (c·(u⊙α))·(c·(u⊙dβ)) ≠ 0 with α ≠ 0).

    NOTE: the proposed-patches snippet `(z1 @ codes.T)` is dimensionally
    inconsistent (z ∈ R^r vs codes.T ∈ R^{K×V}); this is the corrected form.
    """
    C = codes.to(device)
    p = target.to(device)
    K = C.shape[1]
    S = int(C[0].sum().item())
    idx = C.nonzero()[:, 1].reshape(-1, S).to(device)     # (V, S) active bits
    g = torch.Generator().manual_seed(11)
    V1 = (torch.randn(K, r, generator=g) * 0.02).to(device).to(C.dtype).requires_grad_(True)
    V2 = torch.zeros(K, r, device=device, dtype=C.dtype, requires_grad=True)
    u = torch.zeros(K, device=device, dtype=C.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([u, V1, V2], max_iter=iters,
                            line_search_fn='strong_wolfe')

    def _logits():
        z = C @ u
        u_g = u[idx]                                      # (V, S)
        t1 = (u_g.unsqueeze(-1) * V1[idx]).sum(-2)        # (V, r)
        t2 = (u_g.unsqueeze(-1) * V2[idx]).sum(-2)
        return z + (t1 * t2).sum(-1)

    def ce_of():
        with torch.no_grad():
            lg = _logits()
            return float(-(p * (lg - torch.logsumexp(lg, dim=-1))).sum())

    def closure():
        opt.zero_grad()
        lg = _logits()
        loss = -(p * (lg - torch.logsumexp(lg, dim=-1))).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return ce_of()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream', required=True, help='uint16 token stream (.bin)')
    ap.add_argument('--n-ctx', type=int, default=64)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--max-tokens', type=int, default=50_000_000)
    ap.add_argument('--codebook', default='twin_free')
    ap.add_argument('--pair-r', type=int, default=0,
                    help='also fit the rank-r pairwise extension (P1-2 ceiling)')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = EVAConfig(code_dim=64, code_sparsity=6, vocab=65536,
                    codebook=args.codebook)
    codes = build_codes(cfg).double()
    print(f'codebook={args.codebook} codes={tuple(codes.shape)} device={dev}')

    toks = load_tokens(args.stream, args.max_tokens)
    print(f'tokens: {toks.numel():,}')

    ctx = top_contexts(toks, cfg.vocab, args.n_ctx)
    unigram = torch.bincount(toks, minlength=cfg.vocab).double()
    p_uni = unigram / unigram.sum().clamp_min(1.0)
    ce_uni = float(-(p_uni[p_uni > 0] * p_uni[p_uni > 0].log()).sum())
    print(f'unigram CE (bias-only floor): {ce_uni:.4f}')

    w_sum = 0.0
    ce_full = ce_pob = ce_pob0 = ce_pair = 0.0
    w = unigram[ctx].double()
    w = w / w.sum().clamp_min(1e-12)
    for i, c in enumerate(ctx.tolist()):
        row = successor_row(toks, c, cfg.vocab)
        tot = row.sum()
        if float(tot) < 8.0:
            continue
        p = row / tot
        lp = (p + 1e-12).log()
        ce_full += float(w[i]) * float(-(p * lp).sum())
        _ce, _ce0 = fit_pob(p, codes, iters=args.iters, device=dev)
        ce_pob += float(w[i]) * _ce
        ce_pob0 += float(w[i]) * _ce0
        if args.pair_r > 0:
            ce_pair += float(w[i]) * fit_pob_pair(p, codes, r=args.pair_r,
                                                  iters=args.iters, device=dev)
        w_sum += float(w[i])
        if (i + 1) % 16 == 0:
            print(f'  [{i + 1}/{len(ctx)}] bigram={ce_full / max(w_sum, 1e-9):.4f} '
                  f'pob={ce_pob / max(w_sum, 1e-9):.4f} (u=0: {ce_pob0 / max(w_sum, 1e-9):.4f})')
    if w_sum <= 0:
        print('no contexts with enough counts — increase --max-tokens')
        return
    ce_full /= w_sum
    ce_pob /= w_sum
    ce_pob0 /= w_sum
    print('-' * 62)
    print(f'weighted bigram CE (full softmax) : {ce_full:.4f}')
    print(f'weighted bigram CE (PoB, fitted)  : {ce_pob:.4f}')
    print(f'weighted bigram CE (PoB, u=0)     : {ce_pob0:.4f}   (= uniform over V)')
    print(f'price of bit independence (gap)   : {ce_pob - ce_full:+.4f} nat')
    print(f'head gain over u=0                : {ce_pob0 - ce_pob:+.4f} nat')
    if args.pair_r > 0:
        ce_pair /= w_sum
        print(f'weighted bigram CE (PoB+pair r={args.pair_r}): {ce_pair:.4f}')
        print(f'pair channel closes               : '
              f'{(ce_pob - ce_pair) / max(ce_pob - ce_full, 1e-9) * 100:.1f}% of the gap')
    print('-' * 62)
    gap = ce_pob - ce_full
    if gap < 0.1:
        print('VERDICT: the PoB ceiling does NOT bind (<0.1 nat) — the trunk is the priority.')
    elif gap > 0.3:
        print('VERDICT: the bit-independence gap is LARGE (>0.3 nat) — P1-2 (pair channel) is critical.')
    else:
        print('VERDICT: moderate gap (0.1..0.3 nat) — P1-2 is a candidate, not mandatory.')


if __name__ == '__main__':
    main()
