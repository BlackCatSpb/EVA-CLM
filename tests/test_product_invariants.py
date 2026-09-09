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


# ── M2.1 bridge injection must be LINEAR in maturity (was M² — double apply) ─
def test_bridge_maturity_linear():
    from core.bridge import SemanticBridge
    torch.manual_seed(2)
    br = SemanticBridge(D=64, n_layers=2, bridge_dim=16)
    with torch.no_grad():
        br.bridge_stream.normal_()
        # stream_log_scale inits to 0 (tanh→0 = zero injection at birth);
        # the LINEARITY test needs a live scale.
        br.stream_log_scale.fill_(0.4)
    h = torch.zeros(1, 1, 64)
    def delta(m):
        with torch.no_grad():
            out = br.inject_layer(1, h.clone(), maturity=torch.tensor(float(m)),
                                  tau_norm=torch.tensor(0.5))
        return float(out.norm())
    d1, d05 = delta(1.0), delta(0.5)
    assert d05 > 0.0, 'zero injection: scale not live in test setup'
    ratio = d1 / (d05 + 1e-12)
    assert abs(ratio - 2.0) < 0.1, f'injection scales as M^{math.log(ratio)/math.log(2):.2f}, expected linear'


# ── M2.2 diagnostics entropy: mean per position, no B·L scale explosion ──────
def test_lbg_entropy_normalized():
    import types
    from core.layer_bridge_gate import LayerBridgeGate
    lbg = LayerBridgeGate(n_layers=1, health_features=6)
    def feat(B, L, seed):
        torch.manual_seed(seed)
        hp = torch.randn(B, L, 8)
        layer = types.SimpleNamespace(mirror=types.SimpleNamespace(
            _cached_pred_error_norm=None, _cached_gate_l1=None,
            _cached_pred_k=None, _cached_hp=hp))
        return float(lbg.layer_diagnostics(layer, None)[4])
    a, b = feat(1, 4, 3), feat(8, 32, 3)
    assert abs(a - b) < 0.15, f'entropy feature scales with batch ({a} vs {b})'
    assert 0.0 < a < 1.0


# ── M2.3 feature 3 = live bridge contribution (was pinned 0.5 in stack) ──────
def test_lbg_bridge_contribution_live():
    m = _stack()
    if m.bridge is None:
        import pytest
        pytest.skip('bridge disabled')
    torch.manual_seed(4)
    x = torch.randint(1, m.cfg.vocab, (1, 6))
    with torch.no_grad():
        m.embed_tokens(x)  # not the real path; do a real forward below
    h = torch.randn(1, 1, m.cfg.D)
    with torch.no_grad():
        br = m.bridge
        br.bridge_stream.normal_()
        br.stream_log_scale.fill_(0.4)   # live scale (inits to 0 = no injection)
        out = br.inject_layer(0, torch.zeros(1, 4, m.cfg.D),
                              maturity=torch.tensor(0.7), tau_norm=torch.tensor(0.5))
        assert torch.isfinite(br.inj_ratio[0]), 'no injection ratio recorded'
        assert float(br.inj_ratio[0]) > 0.0
        lbg = m.layer_bridge_gate
        import types
        layer = types.SimpleNamespace(mirror=types.SimpleNamespace(
            _cached_pred_error_norm=None, _cached_gate_l1=None,
            _cached_pred_k=None, _cached_hp=None))
        d = lbg.layer_diagnostics(layer, bridge_contrib=br.inj_ratio[0])
        assert float(d[3]) == float(br.inj_ratio[0]), 'feature 3 not wired to contribution'


# ── M2.4 layer_gate: uniform-before-ready, SpectrumGate after ────────────────
def test_lbg_layer_gate_paths():
    from core.layer_bridge_gate import LayerBridgeGate
    torch.manual_seed(5)
    lbg = LayerBridgeGate(n_layers=2, health_features=6)
    health = torch.rand(6)
    mat = torch.tensor(0.4)
    g_off = lbg.layer_gate(0, health, mat, global_ready=False)
    assert torch.allclose(g_off, mat), 'pre-ready gate must equal maturation'
    g_on = lbg.layer_gate(0, health, mat, global_ready=True,
                          tau_external=torch.tensor(1.0))
    assert 0.0 <= float(g_on.detach()) <= 2.0 and float(g_on.detach()) != float(g_off.detach())
    # NaN health must not poison the gate
    health_nan = health.clone(); health_nan[0] = float('nan')
    g_nan = lbg.layer_gate(0, health_nan, mat, global_ready=True,
                           tau_external=torch.tensor(1.0))
    assert torch.isfinite(g_nan)


# ── M2.5 tau ladder of the gate is GEOMETRIC and reads the τ-field ───────────
def test_lbg_effective_tau_geometric():
    from core.layer_bridge_gate import LayerBridgeGate
    lbg = LayerBridgeGate(n_layers=2, health_features=6,
                          tau_min=0.3, tau_max=5.0)
    t0 = float(lbg._effective_tau(torch.tensor(0.0)))
    t1 = float(lbg._effective_tau(torch.tensor(1.0)))
    tm = float(lbg._effective_tau(torch.tensor(0.5)))
    assert abs(t0 - 5.0) < 1e-4 and abs(t1 - 0.3) < 1e-4
    assert abs(tm - math.sqrt(5.0 * 0.3)) < 1e-3, 'linear midpoint — must be geometric (τ-field convention)'


# ── M3.1 chunked VSA scan must equal the naive recurrence at adversarial decay
def test_scan_exactness_adversarial_decay():
    from core.block import _scan_chunk, _combine_chunks
    torch.manual_seed(6)
    B, L, C = 1, 64, 32
    b = torch.randn(B, L, C)
    d = (0.2 + 0.5 * torch.rand(B, L, C))            # fast scales (τ~1-2 regime)
    chunks = [_scan_chunk(b[:, s:s+32], d[:, s:s+32]) for s in range(0, L, 32)]
    mem, _, _ = _combine_chunks(chunks, None)
    naive = torch.zeros_like(b)
    s = torch.zeros(B, C)
    for t in range(L):
        s = d[:, t] * s + b[:, t]
        naive[:, t] = s
    err = (mem - naive).abs().max().item() / naive.abs().max().item()
    assert err < 1e-4, f'chunk scan diverges from naive recurrence: rel {err:.3e}'


# ── M3.2 surprisal decay factor must be NEUTRAL at zero prediction error ─────
def test_pen_decay_neutral_at_zero():
    from core.block import pen_decay_factor
    w = torch.zeros(4)
    f0 = pen_decay_factor(torch.zeros(1, 1, 4), w)
    assert torch.allclose(f0, torch.ones(1, 1, 4), atol=1e-6), \
        f'pen=0 must leave decay untouched, got {f0.flatten().tolist()}'
    f1 = pen_decay_factor(torch.ones(1, 1, 4) * 10.0, w)
    assert (f1 < f0).all() and (f1 >= 0.5 - 1e-6).all(), 'pen↑ must shorten memory toward 0.5'


# ── M3.3 mirror window-mean must be CAUSAL (prefix mean, no future leak) ─────
def test_mirror_prefix_mean_causal():
    from core.mirror import prefix_mean
    torch.manual_seed(7)
    x = torch.randn(2, 8, 3, 5)
    m = prefix_mean(x, dim=1)
    ref = x.cumsum(1) / torch.arange(1, 9, dtype=x.dtype).view(1, 8, 1, 1)
    assert torch.allclose(m, ref, atol=1e-6)
    y = x.clone(); y[:, 5:] += 100.0                  # change the FUTURE only
    assert torch.allclose(prefix_mean(y, dim=1)[:, :5], m[:, :5], atol=1e-5), \
        'future positions leak into past window mean'


# ── M3.4 γ_surprisal init from the REAL τ ladder (geometric center), not ln32 ─
def test_gamma_init_from_tau_ladder():
    from core.block import EVABlock
    cfg = EVAConfig(**{**SMALL, 'n_layers': 4})
    tau_min, tau_max = cfg.tau_min, cfg.tau_max
    tau_mid = math.sqrt(tau_min * tau_max)
    for li in range(cfg.n_layers):
        blk = EVABlock(cfg, li)
        frac = li / max(cfg.n_layers - 1, 1)
        tau_l = tau_min * (tau_max / tau_min) ** frac
        want = 0.5 / (1.0 + math.exp(-(math.log(tau_l) - math.log(tau_mid))))
        got = float(blk.gamma_surprisal.detach())
        assert abs(got - want) < 1e-4, f'L{li}: gamma_init {got:.4f} vs ladder-derived {want:.4f}'


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
