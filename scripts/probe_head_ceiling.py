"""P1-1 (proposed patches): the CEILING of the product-of-Bernoulli head.

The coded head's log-family is q(v|u) ∝ exp(Σ_k c_vk u_k) — a rank-K log-linear
family over the codebook: it cannot express ANY intra-code correlation. Before
extending the head, measure how much CE the family loses IN PRINCIPLE at the
bigram level:

    (1) unigram CE            — the bias-only floor
    (2) bigram CE (full)      — the exact conditional softmax over successors
    (3) bigram CE under PoB   — the best CE this family can reach (LBFGS over
                                u ∈ R^K per context; CE is convex in u)

    gap = (3) − (2)  = the price of bit independence.

**EXT_GEN_13640_RESPONSE §1 correction:** the PRODUCTION head is
`logit_v = (C u)_v + b_v` with a shared per-word bias b (token_bias, 65536 free
params) — the family {C u + b} CONTAINS the exact unigram (u=0, b=log p_uni), so
the u-only number (3) is NOT the production ceiling. With `--bias` the probe fits
the full family: a shared b + per-context u_c, by alternating convex LBFGS
(given b each u_c is independent; given the u_c's, b is one logistic regression
over the pooled contexts), and additionally reports the OPTIMUM FLIP-RATE
(the fraction of contexts where argmax(C u_c + b) ≠ argmax(b)) and the
context-vs-bias margins — the model-side analogue of BIAS-DECOMP for an
ideal head of this family.

Interpretation (the patch's acceptance): gap < 0.1 nat — the ceiling does not
bind, the trunk is the priority; gap > 0.3 nat — the pairwise channel (P1-2)
is on the critical path.

Memory note: the bigram table is NOT materialised (V² = 4.3e9). Only the
top-`n_ctx` contexts (by unigram frequency) are counted, one at a time.

Run:  python scripts/probe_head_ceiling.py --stream <token_stream.bin> [--n-ctx 64]
      python scripts/probe_head_ceiling.py --stream <bin> --bias [--pair-r 16]
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


def fit_pob_b(target: torch.Tensor, codes: torch.Tensor, b: torch.Tensor,
              iters: int = 200, device: str = 'cpu'):
    """Best CE of q = softmax(C u + b) for a FIXED shared bias b (per context)."""
    C = codes.to(device)
    p = target.to(device)
    bb = b.to(device)
    u = torch.zeros(C.shape[1], dtype=C.dtype, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([u], max_iter=iters, line_search_fn='strong_wolfe')

    def ce_of(uu):
        z = C @ uu + bb
        return float(-(p * (z - torch.logsumexp(z, dim=-1))).sum())

    def closure():
        opt.zero_grad()
        z = C @ u + bb
        loss = -(p * (z - torch.logsumexp(z, dim=-1))).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return ce_of(u.detach()), u.detach()


def fit_bias(ps, codes, us, weights, b0: torch.Tensor, iters: int = 40,
             device: str = 'cpu'):
    """Fit the SHARED bias b given the per-context u_c (one logistic regression
    over the pooled weighted contexts). Convex; LBFGS over V params."""
    C = codes.to(device)
    P = [p.to(device) for p in ps]
    U = [u.to(device) for u in us]
    W = [float(w) for w in weights]
    b = b0.to(device).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([b], max_iter=iters, line_search_fn='strong_wolfe')

    def pooled():
        loss = 0.0
        for w, p, u in zip(W, P, U):
            z = C @ u + b
            loss = loss + w * (-(p * (z - torch.logsumexp(z, dim=-1))).sum())
        return loss

    def closure():
        opt.zero_grad()
        loss = pooled()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(pooled()), b.detach()


def flip_and_margins(ps, codes, us, b, weights, device='cpu'):
    """The optimum flip-rate and the context-vs-bias margin quantiles:
    for each context, v*_bias = argmax(b), v*_ctx = argmax(C u + b);
    D_pos = full(v*_ctx) − full(v*_bias) (nat)."""
    C = codes.to(device)
    bb = b.to(device)
    flips, margins, wsum = 0, [], 0.0
    with torch.no_grad():
        v_bias = int(bb.argmax())
        b_top = float(bb[v_bias])
        for w, p, u in zip(weights, ps, us):
            z = C @ u.to(device) + bb
            v_ctx = int(z.argmax())
            if v_ctx != v_bias:
                flips += 1
            margins.append((float(z[v_ctx]) - float(z[v_bias]), float(w)))
            wsum += float(w)
    m = torch.tensor([x for x, _ in margins])
    ww = torch.tensor([w for _, w in margins])
    order = torch.argsort(m)
    ms, ws = m[order], ww[order]
    cw = torch.cumsum(ws, 0) / ws.sum().clamp_min(1e-12)

    def q(p):
        i = int(torch.searchsorted(cw, torch.tensor(float(p))))
        return float(ms[min(i, ms.numel() - 1)])
    return flips, len(margins), q(0.10), q(0.50), q(0.90)


def fit_pob_pair(target: torch.Tensor, codes: torch.Tensor, r: int = 16,
                 iters: int = 150, device: str = 'cpu',
                 b: torch.Tensor = None, V1=None, V2=None, fit_shared: bool = False):
    """P1-2 ceiling: the rank-r pairwise (Ising) extension of the family.

        logits_v = (C u)_v + b_v + Σ_s (c_v·(u⊙α_s))·(c_v·(u⊙β_s))

    α_s, β_s ∈ R^K are the learned pair directions (2Kr params, r=16 → 2048).
    The bit vector is sparse (S active bits), so each factor is a gather of S
    terms — the dense C-form would cost r× the head. V1 starts small-random and
    V2 at zero (an all-zero init is a saddle: the product's V2-gradient is
    (c·(u⊙α))·(c·(u⊙dβ)) ≠ 0 with α ≠ 0).

    With `--bias`, the shared (b, V1, V2) are held fixed here (fit_shared=False)
    and updated in the shared step; returns (ce, u).
    """
    C = codes.to(device)
    p = target.to(device)
    K = C.shape[1]
    S = int(C[0].sum().item())
    idx = C.nonzero()[:, 1].reshape(-1, S).to(device)     # (V, S) active bits
    g = torch.Generator().manual_seed(11)
    if V1 is None:
        V1 = (torch.randn(K, r, generator=g) * 0.02).to(device).to(C.dtype)
    if V2 is None:
        V2 = torch.zeros(K, r, device=device, dtype=C.dtype)
    bb = (torch.zeros(C.shape[0], device=device, dtype=C.dtype) if b is None
          else b.to(device))
    u = torch.zeros(K, device=device, dtype=C.dtype, requires_grad=True)
    params = [u]
    if fit_shared:
        V1 = V1.clone().requires_grad_(True)
        V2 = V2.clone().requires_grad_(True)
        bb = bb.clone().requires_grad_(True)
        params += [V1, V2, bb]
    opt = torch.optim.LBFGS(params, max_iter=iters,
                            line_search_fn='strong_wolfe')

    def _logits():
        z = C @ u + bb
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
    with torch.no_grad():
        return ce_of(), u.detach(), V1.detach(), V2.detach(), bb.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream', required=True, help='uint16 token stream (.bin)')
    ap.add_argument('--n-ctx', type=int, default=64)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--max-tokens', type=int, default=50_000_000)
    ap.add_argument('--codebook', default='twin_free')
    ap.add_argument('--pair-r', type=int, default=0,
                    help='also fit the rank-r pairwise extension (P1-2 ceiling)')
    ap.add_argument('--bias', action='store_true',
                    help='EXT §1: fit the FULL production family (shared token_bias '
                         'b + per-context u) and report the optimum flip-rate/margins')
    ap.add_argument('--alt', type=int, default=3,
                    help='alternations of the (u_c | b) and (b | u_c) convex fits')
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

    # собрать целевые строки (только контексты с достаточным числом наблюдений)
    ps, ws, ctx_kept = [], [], []
    w_all = unigram[ctx].double()
    w_all = w_all / w_all.sum().clamp_min(1e-12)
    ce_full = 0.0
    for i, c in enumerate(ctx.tolist()):
        row = successor_row(toks, c, cfg.vocab)
        tot = row.sum()
        if float(tot) < 8.0:
            continue
        p = row / tot
        lp = (p + 1e-12).log()
        ce_full += float(w_all[i]) * float(-(p * lp).sum())
        ps.append(p)
        ws.append(float(w_all[i]))
        ctx_kept.append(c)
    w_sum = sum(ws)
    if w_sum <= 0:
        print('no contexts with enough counts — increase --max-tokens')
        return
    ce_full /= w_sum
    print(f'contexts kept: {len(ps)}')

    if args.bias:
        # ── полная продовая семья: общий b + u_c на контекст (альтернация) ──
        b = p_uni.clamp_min(1e-12).log().clone()
        ce_bias_only = float(-(p_uni[p_uni > 0] * p_uni[p_uni > 0].log()).sum())
        us = [torch.zeros(codes.shape[1], dtype=codes.dtype) for _ in ps]
        V1 = V2 = None
        ce_fam = ce_bias_only
        for it in range(max(1, args.alt)):
            ce_acc = 0.0
            for k, p in enumerate(ps):
                if args.pair_r > 0:
                    _ce, _u, V1n, V2n, _b = fit_pob_pair(
                        p, codes, r=args.pair_r, iters=args.iters, device=dev,
                        b=b, V1=V1, V2=V2, fit_shared=False)
                    V1, V2 = V1n, V2n
                else:
                    _ce, _u = fit_pob_b(p, codes, b, iters=args.iters, device=dev)
                us[k] = _u
                ce_acc += ws[k] * _ce
            ce_fam = ce_acc / w_sum
            # общий шаг: b (и V1,V2 при pair)
            ce_pooled, b = fit_bias(ps, codes, us, ws, b, iters=40, device=dev)
            ce_pooled /= w_sum
            print(f'  alt[{it + 1}/{args.alt}] family CE={ce_fam:.4f} '
                  f'pooled={ce_pooled:.4f} flip-rate pending')
        ce_fam = ce_pooled if ce_pooled < ce_fam else ce_fam
        flips, n, p10, p50, p90 = flip_and_margins(ps, codes, us, b, ws, device=dev)
        print('-' * 62)
        print(f'weighted bigram CE (full softmax)        : {ce_full:.4f}')
        print(f'CE(bias-only) = unigram of the stream    : {ce_bias_only:.4f}')
        print(f'CE(PoB+bias family, fitted)              : {ce_fam:.4f}')
        if args.pair_r > 0:
            # NB: the shared (b, V1, V2) are NOT jointly optimized here — the
            # per-context u fit uses the shared params from the last alternation.
            # This is a CONSERVATIVE (upper-bound) estimate of the joint optimum.
            ce_pair = 0.0
            for k, p in enumerate(ps):
                _ce, _u, _v1, _v2, _b = fit_pob_pair(
                    p, codes, r=args.pair_r, iters=args.iters, device=dev,
                    b=b, V1=V1, V2=V2, fit_shared=False)
                ce_pair += ws[k] * _ce
            ce_pair /= w_sum
            print(f'CE(PoB+bias+pair r={args.pair_r}, shared fixed) : {ce_pair:.4f} '
                  f'(conservative)')
        print(f'optimum flip-rate (argmax(Cu+b) != argmax(b)): '
              f'{flips}/{n} = {100.0 * flips / max(n, 1):.1f}%')
        print(f'context-vs-bias margin D_pos (nat): p10={p10:+.2f} p50={p50:+.2f} p90={p90:+.2f}')
        print('-' * 62)
        gap = ce_fam - ce_full
        print(f'price of the (bias-inclusive) family gap : {gap:+.4f} nat')
        if flips / max(n, 1) >= 0.5:
            print('VERDICT: the family WITH bias flips the argmax in most contexts — '
                  'hedging is NOT forced by the family (the head/optimization is).')
        else:
            print('VERDICT: even the ideal family barely flips — the context statistics '
                  'themselves are weak on this stream.')
        return

    ce_pob = ce_pob0 = ce_pair = 0.0
    for i, p in enumerate(ps):
        _ce, _ce0 = fit_pob(p, codes, iters=args.iters, device=dev)
        ce_pob += ws[i] * _ce
        ce_pob0 += ws[i] * _ce0
        if args.pair_r > 0:
            _ce, _u, _v1, _v2, _b = fit_pob_pair(p, codes, r=args.pair_r,
                                                 iters=args.iters, device=dev)
            ce_pair += ws[i] * _ce
        if (i + 1) % 16 == 0:
            print(f'  [{i + 1}/{len(ps)}] bigram={ce_full:.4f} '
                  f'pob={ce_pob / w_sum:.4f} (u=0: {ce_pob0 / w_sum:.4f})')
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
