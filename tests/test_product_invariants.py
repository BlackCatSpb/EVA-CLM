"""Product invariants (M1: language head CE must train EVERY position).

Origin: 5-agent audit (A4+A5) — head returned `raw[0, 0, targets]` in the 2D
training path, so CE gradient touched only hidden state #0, and token_bias was
added twice (forward + gather) on top of a normalize branch. These tests lock
the correct semantics; every later milestone adds its own section here.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from core.config import EVAConfig
from core.stack import EVAStack

device = torch.device('cpu')
SMALL = dict(n_layers=1, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _stack(**kw):
    cfg = EVAConfig(**{**SMALL, **kw})
    torch.manual_seed(0)
    return EVAStack(cfg).to(device).train()


def _hq(m, B=2, L=7):
    torch.manual_seed(1)
    h = torch.randn(B, L, m.cfg.D, device=device, requires_grad=True)
    y = torch.randint(1, m.cfg.vocab, (B, L), device=device)
    return h, y


# ── M1.1 per-position CE gradient (both normalize modes) ─────────────────────
def test_head_ce_grad_covers_every_position():
    for norm in (True, False):
        m = _stack(head_normalize=norm)
        h, y = _hq(m)
        lp = m.lm_head.log_probs_for_target(h.reshape(-1, m.cfg.D), y.reshape(-1))
        assert lp.shape == (h.shape[0] * h.shape[1],), f'{norm}: shape {tuple(lp.shape)}'
        (-lp.sum()).backward()
        g = h.grad.abs().sum(dim=-1).reshape(-1)
        assert (g > 0).all(), f'normalize={norm}: zero-grad positions {(g == 0).nonzero().flatten().tolist()}'


# ── M1.2 lp == gather from forward (normalize), no double bias ───────────────
def test_head_lp_matches_forward_gather():
    m = _stack(head_normalize=True)
    h, y = _hq(m)
    logits = m.lm_head(h)                                # (B,L,V) normalized
    lp = m.lm_head.log_probs_for_target(h.reshape(-1, m.cfg.D), y.reshape(-1))
    ref = torch.gather(logits.reshape(-1, logits.shape[-1]), 1, y.reshape(-1, 1)).squeeze(1)
    assert torch.allclose(lp, ref, atol=1e-5), (lp - ref).abs().max().item()
    # bias single-count: uniform +0.5 bias leaves normalized log-probs unchanged
    with torch.no_grad():
        m.lm_head.token_bias += 0.5
    logits2 = m.lm_head(h)
    lp2 = m.lm_head.log_probs_for_target(h.reshape(-1, m.cfg.D), y.reshape(-1))
    assert torch.allclose(lp, lp2, atol=1e-4), 'double token_bias: CE shifted without learning'


# ── M1.3 factorized branch is a proper bit-likelihood, no (N,N) blowup ───────
def test_head_factorized_lp_semantics():
    m = _stack(head_normalize=False)
    h, y = _hq(m)
    lp = m.lm_head.log_probs_for_target(h.reshape(-1, m.cfg.D), y.reshape(-1))
    assert lp.dim() == 1 and lp.shape[0] == 14
    # аналитически: Σ log p(active) + log(1-p)(inactive) + token_bias
    zt = m.lm_head._gates(h.reshape(-1, m.cfg.D), bus_bias=None)
    u, _ = m.lm_head._su(zt)
    c = m.lm_head.codes[y.reshape(-1)].float()
    want = (c * F.logsigmoid(u) + (1 - c) * F.logsigmoid(-u)).sum(-1) + m.lm_head.token_bias[y.reshape(-1)]
    assert torch.allclose(lp, want, atol=1e-4)


# ── M1.4 emphasis (softmax boost) survives saturation — no clamp cliff ───────
def test_head_emphasis_not_saturated():
    m = _stack()
    zt = torch.full((1, 1, m.lm_head.K), 30.0)
    u, base = m.lm_head._su(zt)
    # old behavior: gate clamped to 1-eps => u == ~16.1 constant, grad(z)->~1
    assert float(u.max()) > 30.0, 'confidence+emphasis collapsed by clamp'
    z = torch.full((1, 1, m.lm_head.K), 30.0, requires_grad=True)
    u2, b2 = m.lm_head._su(z)
    u2.sum().backward()
    g = z.grad.abs().mean().item()
    assert 0.5 < g < 2.0, f'gradient through confident bits degenerate: {g}'


# ── M1.5 bit_bias init = logit(code prior) (the dead _prop buffer) ───────────
def test_bit_bias_prior_init():
    m = _stack()
    prop = m.lm_head.codes.float().mean(dim=0).clamp(1e-7, 1 - 1e-7)
    want = torch.log(prop / (1 - prop))
    assert torch.allclose(m.lm_head.bit_bias.detach(), want, atol=1e-5), \
        'bit_bias not initialized from code prior (_prop buffer was dead)'


# ── M1.6 losses.py: PAD always masked, EOS per mask_eos, surprisal wired ─────
def test_losses_masking_code_head():
    m = _stack()
    B, L = 2, 8
    h = torch.randn(B, L, m.cfg.D, device=device)
    y = torch.randint(1, m.cfg.vocab, (B, L), device=device)
    y[:, -1] = 2  # EOS
    y[0, -2] = 0  # PAD
    cfg = m.cfg
    cfg.mask_eos = False
    ce_open, _ = m.compute_losses(h, y)
    cfg.mask_eos = True
    ce_masked, _ = m.compute_losses(h, y)
    n_all, n_no_eos = B * L - 1, B * L - 1 - B  # PAD excluded in both
    # open-mask: EOS trained => mean over n_all; masked: over n_no_eos
    assert ce_open.item() != ce_masked.item(), 'mask_eos does nothing for the coded head'
    cfg.mask_eos = False
    cfg.surprisal_weight = 1.0
    ce_sw, _ = m.compute_losses(h, y)
    assert abs(ce_sw.item() - ce_open.item()) > 1e-6, 'surprisal_weight ignored by coded-head CE'
    cfg.surprisal_weight = 0.0


# ── M1.7 CognitiveCodedHead must accept the flat 2D training path ────────────
def test_cognitive_head_2d_path():
    m = _stack(head_mode='cognitive_coded')
    h, y = _hq(m, B=1, L=5)
    ce, _ = m.compute_losses(h, y)   # compute_losses flattens (N,D) — used to crash
    assert torch.isfinite(ce)
    ce.backward()
    g = h.grad.abs().sum(dim=-1).reshape(-1)
    assert (g > 0).all(), 'cognitive head: dead positions in CE'


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as e:
            fails += 1
            print(f'FAIL {fn.__name__}: {repr(e)[:220]}')
    print(f'\n{len(fns) - fails}/{len(fns)} passed')
    sys.exit(1 if fails else 0)
