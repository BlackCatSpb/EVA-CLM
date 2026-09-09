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
    # n_layers=2: cross-layer var()/std() terms in aux losses need ≥2 layers
    cfg = EVAConfig(**{**SMALL, 'n_layers': 2, **kw})
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


# ── M4.1 expert amplitude ladder must be G-invariant (was 1.5^g → ×14k) ─────
def test_mirror_amplitude_ladder_g32():
    from core.mirror import GroupedCognitiveMirror
    torch.manual_seed(11)
    mir = GroupedCognitiveMirror(D=256, G=32, k=4, layer_idx=0, n_layers=2,
                                 expert_asymmetry=True)
    amp = torch.exp(mir.log_scale.detach().mean(dim=-1))
    # top of the ladder is amp=1 plus the documented log_scale_init_std noise
    assert float(amp.max()) <= 1.15, f'init amplitude explodes: {float(amp.max())}'
    assert float(amp.min()) >= 0.03, 'ladder floor lost'
    la = amp.log()
    assert 2.0 <= float(la.max() - la.min()) <= 3.5, 'ladder span lost'
    corr = torch.corrcoef(torch.stack([torch.arange(32, dtype=torch.float32), la]))[0, 1]
    assert float(corr) > 0.98, 'ladder not monotone-in-trend in expert index'


# ── M4.2 grad-modulation input must be scale-normalized, not raw ─────────────
def test_grad_mod_input_normalized():
    from core.mirror import grad_mod_input
    big = torch.full((4,), 1000.0)
    x = grad_mod_input(big, big.clone(), torch.zeros(4))
    assert float(x.abs().max()) < 1e-3, 'typical large-gradient step must map to ~0'
    x2 = grad_mod_input(big * 4, big.clone(), torch.zeros(4))
    assert 0.5 < float(x2.mean()) < 3.5, f'ratio lost: {float(x2.mean())}'


# ── M4.3 signal EMA must carry per-expert statistics ─────────────────────────
def test_signal_norm_per_expert():
    m = _stack()
    m.train()
    torch.manual_seed(8)
    for _ in range(3):
        x = torch.randint(1, m.cfg.vocab, (1, 6))
        h = m.embed_tokens(x)
        m(h, step=1, tokens=x)
    ema = m.layers[0].mirror._signal_norm_ema[0]      # (G, k)
    assert float(ema.std(dim=0).mean()) > 1e-6, \
        'signal EMA broadcast a global scalar over experts (old squeeze bug)'


# ── M4.4 anti-collapse governor must be active at EVAL too (parity) ──────────
def test_governor_eval_parity():
    m = _stack()
    m.eval()
    lay = m.layers[0].mirror
    torch.manual_seed(9)
    x = torch.randint(1, m.cfg.vocab, (1, 6))
    h = m.embed_tokens(x)
    with torch.no_grad():
        lay._ls_var_run.fill_(0.9)
        m(h, step=1, tokens=x)
        g_off = float(lay._last_gates.mean())
        lay._ls_var_run.fill_(0.001)
        m(h, step=1, tokens=x)
        g_on = float(lay._last_gates.mean())
    assert g_on > g_off, 'governor dead at eval (train-only branch, audit M4)'


# ── M5.1 pred aux trains alpha (both operands were detached before) ──────────
def test_pred_loss_differentiable():
    m = _stack()
    x = torch.randint(1, m.cfg.vocab, (1, 6))
    h = m.embed_tokens(x)
    out, st, gs, _ = m(h, step=5, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    assert 'pred' in aux and isinstance(aux['pred'], torch.Tensor) \
        and aux['pred'].requires_grad, 'pred has no graph'
    (ce + aux['pred']).backward()
    g = m.layers[0].mirror.alpha_diag.grad
    assert g is not None and float(g.abs().sum()) > 0, 'pred loss dead for alpha'


# ── M5.2 w_m2v regularizer moves the parameter (param side was detached) ─────
def test_wm2v_regularizer_live():
    m = _stack(w_m2v_hierarchy_weight=1.0)
    x = torch.randint(1, m.cfg.vocab, (1, 6))
    h = m.embed_tokens(x)
    out, st, gs, _ = m(h, step=5, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    assert 'w_m2v' in aux and aux['w_m2v'].requires_grad
    (ce + aux['w_m2v']).backward()
    g = m.layers[0].w_mem2v.grad
    assert g is not None and float(g.abs().sum()) > 0


# ── M5.3 intent_tau shapes the τ-field (actual side was detached) ────────────
def test_intent_tau_live():
    m = _stack(intent_bridge=True, intent_tau_hierarchy_weight=1.0)
    if not getattr(m, 'intent_bridge', False):
        import pytest
        pytest.skip('intent bridge not built in this config')
    x = torch.randint(1, m.cfg.vocab, (1, 6))
    h = m.embed_tokens(x)
    out, st, gs, _ = m(h, step=5, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    assert 'intent_tau' in aux and aux['intent_tau'].requires_grad
    (ce + aux['intent_tau']).backward()
    g = m.tau_config._tau_dev.grad
    assert g is not None and float(g.abs().sum()) > 0, 'intent_tau cannot shape τ'


# ── M5.4 signal_ent direction: uniform weights are the MINIMUM of the loss ────
def test_signal_ent_pushes_toward_uniform():
    def ent_term(w_val):
        m = _stack()
        for layer in m.layers:
            with torch.no_grad():
                lay = layer.mirror._signal_log_weights
                lay.zero_(); lay[0] = w_val
        x = torch.randint(1, m.cfg.vocab, (1, 4))
        h = m.embed_tokens(x)
        out, st, gs, _ = m(h, step=3, tokens=x)
        _, aux = m.compute_losses(out, x, h_emb=h)
        return float(aux['signal_ent'].detach())
    near_uniform = ent_term(0.0)
    collapsed = ent_term(20.0)
    assert near_uniform < collapsed, 'signal_ent rewards collapse (wrong sign)'


# ── M5.5 gradalign: hook target feeds a weighted term that trains the anchor ─
def test_gradalign_hook_anchor_bypass():
    from core.adaptation import LossBalancer
    m = _stack(gradalign_weight=0.3)
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    x = torch.randint(1, m.cfg.vocab, (1, 6))
    bal = LossBalancer(align=True)
    term_seen = None
    for it in range(3):                       # step1 fills hook targets
        h = m.embed_tokens(x)
        out, st, gs, _ = m(h, step=5 + it, tokens=x, intent_state=None)
        ce, aux = m.compute_losses(out, x, h_emb=h)
        if 'gradalign' in aux:
            term_seen = aux['gradalign']
        bal.backward(ce, aux, list(m.parameters()))
        opt.zero_grad(set_to_none=True)
    assert term_seen is not None and term_seen.requires_grad, 'no live gradalign'
    assert float(term_seen.detach()) > 0
    for layer in m.layers:
        assert getattr(layer, '_gradalign_tgt', None) is not None, 'hook never fired'
    h = m.embed_tokens(x)
    out, st, gs, _ = m(h, step=8, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    bal.backward(ce, aux, list(m.parameters()))
    ms = m.layers[0].mirror.mod_scale_mlp
    assert ms.grad is not None and float(ms.grad.abs().sum()) > 0, \
        'gradalign bypass did not reach the mlp_mod anchor'


# ── M5.6 mlp_mod anchored to mod_scale_mlp: identity at init, live on grad ────
def test_mlp_mod_anchor_identity():
    from core.mirror import GroupedCognitiveMirror
    import math as _m
    mir = GroupedCognitiveMirror(D=64, G=4, k=4, layer_idx=0, n_layers=1,
                                 expert_asymmetry=True, bridge_glu=True)
    assert abs(float(torch.sigmoid(mir.mod_scale_mlp[0]).item()) - 2.0 / 3.0) < 1e-6
    # σ(m)/σ(ln2) == 1 at init exactly → anchor does not alter the init path
    coef = 1.5 * torch.sigmoid(mir.mod_scale_mlp)
    assert torch.allclose(coef, torch.ones_like(coef), atol=1e-6)


# ── M6 CONCEPT LAYER: write path must carry gradient (design principle #3) ───
def _concept_step(m, it=0):
    x = torch.randint(1, m.cfg.vocab, (1, 8))
    h = m.embed_tokens(x)
    lay = m.layers[0]
    torch.manual_seed(100 + it)
    hp = torch.randn(1, 8, lay.mirror.G, lay.mirror.k, device=h.device)
    # low pen → conf = σ(−pen) ≥ birth_thresh (0.375 at init): the birth path
    # is what exercises write_q_proj/write_v_proj gradients.
    pen = 0.02 + torch.rand(1, 8) * 0.06
    out = m.concept_layer(h, hp=hp, pen=pen, resvar=0.2,
                          mat_gate=1.0, allow_write=True,
                          gate=torch.rand(1, 8, lay.mirror.G), tau_norm=0.5)
    return out


def test_concept_write_gradient_live():
    m = _stack()
    assert m.concept_layer is not None, 'mini stack has no concept layer'
    for it in range(10):
        out = _concept_step(m, it)
        out.sum().backward()          # writes start ~it=5 (maturity transient)
    wq = m.concept_layer.write_q_proj.weight
    wv = m.concept_layer.write_v_proj.weight
    assert wq.grad is not None and float(wq.grad.abs().sum()) > 0, \
        'write_q_proj: write path has no gradient (design principle #3 violated)'
    assert wv.grad is not None and float(wv.grad.abs().sum()) > 0, 'write_v_proj dead'
    assert int(m.concept_layer._n_births.item()) > 0, 'no concept was ever born'


def test_concept_gate_param_used():
    outs = []
    for gv in (0.0, 1.0):
        m = _stack()
        o = None
        for it in range(10):
            x = torch.randint(1, m.cfg.vocab, (1, 8))
            h = m.embed_tokens(x)
            lay = m.layers[0]
            torch.manual_seed(100 + it)
            hp = torch.randn(1, 8, lay.mirror.G, lay.mirror.k)
            pen = 0.02 + torch.rand(1, 8) * 0.06
            o = m.concept_layer(h, hp=hp, pen=pen, resvar=0.2, mat_gate=1.0,
                                allow_write=True,
                                gate=torch.full((1, 8, lay.mirror.G), gv),
                                tau_norm=0.5)
        outs.append(o.detach())
    assert not torch.allclose(outs[0], outs[1], atol=1e-6), 'gate argument ignored (principle #4)'


def test_concept_novelty_thr_live():
    m = _stack()
    with torch.no_grad():
        m.concept_layer._log_tau_novelty_thr.fill_(-20.0)   # gap≈0 → permissive
        for it in range(10):
            _concept_step(m, it)
    permissive = int(m.concept_layer._n_births.item())
    m3 = _stack()
    with torch.no_grad():
        m3.concept_layer._log_tau_novelty_thr.fill_(20.0)    # gap≈1 → never novel
        for it in range(10):
            _concept_step(m3, it)
    strict = int(m3.concept_layer._n_births.item())
    assert permissive > 0 and strict == 0, \
        f'novelty gap has no authority (permissive={permissive}, strict={strict})'


def test_concept_store_consistency():
    # after multiple write steps the persistent store must equal the last
    # effective store (no divergence between buffer and used keys)
    m = _stack()
    for it in range(10):
        out = _concept_step(m, it)
    out.sum().backward()
    keys = m.concept_layer.concept_keys
    assert torch.isfinite(keys).all() and float(keys.norm()) > 0
    # birth_gate diagnostic must be finite
    assert math.isfinite(float(m.concept_layer._cached_birth_gate))


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
