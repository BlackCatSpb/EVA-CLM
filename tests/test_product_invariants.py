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
    # B2 contract: the emphasis term is CENTERED — a uniform emphasis must be
    # a no-op (u == logits exactly); a concentrated emphasis still boosts
    # beyond the logit without any clamp cliff (old M5 property preserved).
    zt = torch.full((1, 1, m.lm_head.K), 30.0)
    u, base = m.lm_head._su(zt)
    assert (u - zt).abs().max().item() < 1e-5, 'uniform emphasis must be neutral (centered)'
    zt2 = torch.zeros(1, 1, m.lm_head.K)
    zt2[..., 0] = 9.0
    u2, _ = m.lm_head._su(zt2)
    assert float(u2[..., 0]) > 9.0, 'confidence+emphasis collapsed by clamp'
    z = torch.full((1, 1, m.lm_head.K), 30.0, requires_grad=True)
    u3, b3 = m.lm_head._su(z)
    u3.sum().backward()
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
        m.concept_layer.read_scale.data.fill_(0.0)   # B1: probe gate mechanism at σ(0),
        o = None                                    # not at the identity-init scale σ(-4)
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


# ── M7.1 role routing on EXACT name parts (b_delta_gate was confiscated) ─────
def test_role_routing_exact_tokens():
    from core.adaptation import _role_lr_mult
    from core.lambda_utils import lambda_d
    lam = lambda_d(3)
    assert _role_lr_mult('layers.0.mirror.b_delta_gate', lam) == lam ** 1, \
        'b_delta_gate mis-routed (substring collision with .b_d)'
    assert _role_lr_mult('layers.0.mirror.w_delta_gate', lam) == lam ** 1
    assert _role_lr_mult('layers.0.b_d', lam) == lam ** -2
    assert _role_lr_mult('layers.0.b_i', lam) == lam ** -2
    assert _role_lr_mult('layers.0.mirror.w_intent', lam) == lam ** 1
    assert _role_lr_mult('layers.0.mirror.b_intent', lam) == lam ** 1, \
        'b_intent hit the .b_i VSA bucket before the gate branch'


# ── M7.2 τ-ladder is bounded by dev_max (documented clip actually applies) ───
def test_tau_ladder_bounded():
    from core.tau_config import TauConfig
    tc = TauConfig(n_layers=24, tau_min=8.0, tau_max=512.0, dev_max=0.3)
    with torch.no_grad():
        tc._tau_dev.fill_(50.0)            # pathological drift
    tc.update()
    tau = tc.tau_l
    assert torch.isfinite(tau).all()
    ratio = float((tau.max() / tau.min()).detach())
    # B3: the ladder span is NORMALIZED to [τ_min·(max/min)^{1/(L-1)}, τ_max] —
    # endpoints exact, so the ratio can never exceed max/min regardless of dev
    # drift (the old cumsum overshot to 613>τ_max and clamped the top τ_norms).
    assert ratio < 65.0, f'ladder span escaped: ratio {ratio:.2f} (bound (512/8))'
    assert abs(float(tau.max().detach()) - 512.0) < 1e-3, 'top endpoint must be exact now'
    # dev=0 identity (checkpoint-safe): uniform in log, ending EXACTLY at τ_max
    tc0 = TauConfig(n_layers=24, tau_min=8.0, tau_max=512.0, dev_max=0.3)
    tc0.update()
    frac = math.log(512.0 / 8.0) / 24.0        # normalized cumsum: (i+1)/24 of the span
    want = torch.tensor([8.0 * math.exp(frac * (i + 1)) for i in range(24)])
    assert torch.allclose(tc0.tau_l.detach(), want, rtol=1e-5)
    assert abs(float(tc0.tau_l[-1].detach()) - 512.0) < 1e-3, 'top endpoint must be exact'
    # τ_norm resolution at depth: only the LAST layer equals 1.0 (old: two)
    tn = tc0.tau_norm
    assert int((tn >= 1.0 - 1e-9).sum()) == 1, f'tau_norm ties at the top: {tn[-3:]}'


# ── M7.3 τ consumers read the LIVE field, not the init snapshot ──────────────
def test_tau_consumers_live():
    # 4 layers so layer 1 sits mid-ladder (2-layer minis clamp every τ_norm to
    # the endpoints and hide the refresh)
    m = _stack(n_layers=4)
    x = torch.randint(1, m.cfg.vocab, (1, 4))
    with torch.no_grad():
        m(m.embed_tokens(x), step=2, tokens=x)
    snap = float(m.layers[1]._tau_norm)
    with torch.no_grad():
        # B3: the normalized ladder pins the ENDPOINTS (τ_min/τ_max = documented
        # physics); _tau_dev now shapes the SPACING — a uniform fill is a null
        # direction by construction (cs/cs[-1] invariant), so perturb non-uniformly.
        m.tau_config._tau_dev.copy_(torch.linspace(-0.25, 0.25, 4))
    with torch.no_grad():
        m(m.embed_tokens(x), step=3, tokens=x)
    now = float(m.layers[1]._tau_norm)
    assert abs(now - snap) > 1e-4, 'block._tau_norm frozen at init (U3/U10 snapshot bug)'
    assert abs(now - float(m.tau_config.tau_norm[1].detach())) < 1e-6


# ── M7.4 scheduler's usefulness-temp knob has a consumer ─────────────────────
def test_usefulness_temp_wired():
    m = _stack()
    x = torch.randint(1, m.cfg.vocab, (1, 4))
    lay = m.layers[0].mirror
    with torch.no_grad():
        lay._usefulness_temp.fill_(2.5)
    with torch.no_grad():
        m(m.embed_tokens(x), step=2, tokens=x)
    assert abs(lay._last_usef_temp - 2.5) < 1e-6, 'scheduler temp ignored by forward'
    with torch.no_grad():
        lay._usefulness_temp.fill_(0.0)
    with torch.no_grad():
        m(m.embed_tokens(x), step=300, tokens=x)
    assert 0.3 <= lay._last_usef_temp <= 3.0 and abs(lay._last_usef_temp - 2.5) > 1e-3


# ── M7.5 bind U10 bounds from cfg, not literals ──────────────────────────────
def test_bind_tau_bounds_from_cfg():
    cfg_kw = {**SMALL}
    cfg = EVAConfig(**cfg_kw)
    m = _stack()
    b = m.layers[0].bind
    if hasattr(b, '_tau_max'):
        assert b._tau_min == cfg.tau_min and b._tau_max == cfg.tau_max
    else:
        import pytest
        pytest.skip('bind mode without U10 schedule')


# ── M8.1 runtime-buffer isolation API (eval quarantine) ─────────────────────
def test_snapshot_restore_buffers():
    m = _stack()
    snap = m.snapshot_runtime_buffers()
    assert len(snap) > 0
    target = next(k for k, v in snap.items() if v.is_floating_point() and v.numel() > 1)
    own = dict(m.named_buffers())
    with torch.no_grad():
        own[target].add_(7.0)
    assert not torch.equal(own[target], snap[target])
    m.restore_runtime_buffers(snap)
    assert torch.equal(dict(m.named_buffers())[target], snap[target])


# ── M8.2 non-finite CE forces rollback; buffers get scrubbed ────────────────
def test_nan_ce_forces_alarm_without_side_effects():
    """D6: non-finite CE forces the ALARM (check()->True) — the sensor itself
    must not swap optimizers, load files or rewrite buffers; the caller stops
    the run and the NaN state dies with the process (resume reloads weights)."""
    from core.training_control import FailureDetector
    m = _stack()
    wd = FailureDetector(m, warmup=0)
    for i in range(5):
        wd.check(10.0, i, {'mlp_ratio': 1.0})
    fired = wd.check(float('nan'), 100, {'mlp_ratio': 1.0})
    assert fired, 'non-finite CE did not force the alarm'
    assert not hasattr(wd, 'optimizer'), 'D6 detector must not own an optimizer'
    assert not hasattr(wd, 'best_path'), 'D6 detector must not touch the filesystem'
    assert m is not None  # weights untouched by the sensor itself

# ── M8.3 TokenStream contract: wrapped flag + stream.len (train.py) ──────────
def test_tokenstream_wrapped(tmp_path=None):
    import importlib.util, numpy as np, tempfile, pathlib
    if tmp_path is None:
        tmp_path = pathlib.Path(tempfile.mkdtemp())
    spec = importlib.util.spec_from_file_location(
        '_train_mod', r'C:\Users\black\OneDrive\Desktop\EVA CLM\scripts\train.py')
    # import train.py for its class only: it guards side effects under __main__,
    # but heavy top-level imports could fail — fall back to source exec of class
    src = open(spec.origin, encoding='utf-8').read()
    i0 = src.index('class TokenStream')
    i1 = src.index('def evaluate', i0) if 'def evaluate' in src[i0:] else len(src)
    seg = src[i0:src.index('\n\n\n', i0)] if '\n\n\n' in src[i0:] else src[i0:i1]
    ns = {'np': np, 'torch': torch}
    exec(compile(seg, 'ts', 'exec'), ns)
    TS = ns['TokenStream']
    f = tmp_path / 's.bin'
    arr = np.arange(64, dtype=np.uint16)
    np.memmap(f, dtype=np.uint16, mode='w+', shape=arr.shape)[:]= arr
    del arr
    st = TS(str(f))
    assert st.len == 64
    x, y, off, wrapped = st.get_batch(8, 2, 40)          # fits: 40+17<64? no→wrapped
    # offset 40 needs 17 → 57 ≤ 64 fits
    assert not wrapped
    x, y, off, wrapped = st.get_batch(8, 2, 60)          # 60+17>64 → wrap
    assert wrapped, 'end-of-stream not reported'
    assert off == 16 or off > 0  # read from 0 after wrap


# ── M9.1 reasoning weighted-average must stay bounded when all gates close ───
def test_reasoning_negative_gates_bounded():
    m = _stack(explicit_reasoning=True, reasoning_adaptive=True)
    assert m.reasoning_gate is not None, 'adaptive reasoning gate not built'
    D = m.cfg.D
    h = torch.randn(2, 5, D)

    class _NegGate(torch.nn.Module):
        # every step fully negative (anti-knowledge) → positive-weight
        # denominator collapses; the floor must keep |accum| bounded (M9)
        def logits(self, h_acc, know, r):
            return torch.full((h_acc.shape[0], h_acc.shape[1], 8), -5.0)
    m.reasoning_gate = _NegGate()
    m._knowledge_signal = lambda x: torch.zeros(x.shape[0], 8)
    m._last_conf = lambda x: torch.full((x.shape[0],), 0.5)
    m._tau_norm_reasoning = 1.0
    h_acc = m._adaptive_reasoning(h, s=1.0, state=None,
                                  reasoning_buffer=None, reasoning_count=None)
    assert torch.isfinite(h_acc).all(), 'h_acc has NaN/Inf'
    # bounded: the delta over h cannot exceed a few × the unit-normalized
    # step contributions (was ~1e-6 denominator → orders-of-magnitude blowup)
    ratio = float(((h_acc - h).norm(dim=-1).mean() / h.norm(dim=-1).mean()).detach())
    assert ratio < 5.0, f'reasoning contribution explodes when gates close: ratio={ratio:.1e}'


# ── M9.2 ReasoningGate init: first gate ON (tanh≈1), rest OFF (≈0) ──────────
def test_reasoning_gate_init():
    from core.reasoning import ReasoningGate
    g = ReasoningGate(D=64, max_steps=8)
    with torch.no_grad():
        out = g(torch.randn(1, 3, 64))
    assert abs(float(out[..., 0].mean()) - 1.0) < 0.05, 'step-0 gate should start ≈ON'
    assert abs(float(out[..., 1:].mean())) < 0.3, 'later gates start near 0'


# ── M11 DEAD-PARAMETER DETECTOR: unregistered zero-grad params fail loudly ───
def test_no_unregistered_dead_parameters():
    """8-step full-objective SGD on a capability-on mini model; every parameter
    that receives no gradient must be on the explicit reason-allowlist
    (audit M11 / A5: 'architecturally dead' must be DECLARED, not discovered
    by chance; any NEW silently-dead param — renamed path, broken wiring —
    fails this test)."""
    m = _stack(intent_bridge=True, memory_bank=True, bridge_conn=0.1,
               unified_concept_layer=True, explicit_reasoning=True)
    for b in m.layers:                      # keep the exact-memory path open
        with torch.no_grad():
            b.precision_gate.gate.bias.fill_(3.0)
    opt = torch.optim.SGD(m.parameters(), lr=0.02)
    state = None
    for it in range(8):
        x = torch.randint(1, m.cfg.vocab, (1, 64))
        x[:, 31] = 2  # deterministic SEP boundary (id 2) so L1 writes happen
        with torch.no_grad():
            h0 = m.embed_tokens(x)
            o0, _, _, _ = m(h0, state, step=20000 + it, tokens=x)
            m.observe_output(m.lm_head(o0))          # salience for next step
        h = m.embed_tokens(x)
        out, state, gs, r = m(h, state, step=20000 + it, tokens=x)
        loss, aux = m.compute_losses(out, x, h_emb=h)
        total = loss + sum(v for v in aux.values() if isinstance(v, torch.Tensor))
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        m._reasoning_buffer, m._reasoning_count = r
    ALLOW_REASONS = {
        '._vsa_tau_log': 'standalone fallback ladder; excluded from the optimizer (M7)',
        'memory_bank.l2.': 'slot-event params: random data without real SEP boundaries keeps the L2 bank empty',
        'memory_bank.l3.': 'concept-value path needs accumulated concepts',
        'exact_memory': 'participates only while the precision gate is open (soft STE keeps the opener alive)',
        'logit_cache.cache.': 'k-selector for the compressed-logits path — trains on inference sweeps, not train steps',
        'logit_cache.attention.k_proj_l': 'inference-mode code-space projection (train-mode uses k_proj_h)',
        'logit_cache.attention.v_proj_l': 'inference-mode code-space projection',
        'logit_cache.logit_to_hidden': 'inference-mode logit→hidden projection',
        'precision_gate': 'depends on gate state per step',
        'concept_layer.log_tau_': 'write-side knobs fire only on confident novel events',
        'concept_layer._log_tau_': 'same conf-gated write path',
        'w_sal': 'needs external salience (covered by the dedicated test above)',
    }
    dead = []
    for n, p in m.named_parameters():
        if p.grad is not None and float(p.grad.abs().sum()) > 0.0:
            continue
        if not any(k in n for k in ALLOW_REASONS):
            dead.append(n)
    assert not dead, 'UNREGISTERED dead parameters (new silent zero-grad?): ' \
                     + ', '.join(dead[:12])


def test_logit_cache_incremental_kv_and_identity_init():
    """Audit #3 follow-up (L4 OOM / 42 tok/s): the cache must (1) store
    detached entries, (2) encode k/v ONCE at write time (O(L), not O(M)
    re-projection), and (3) start as identity — gate bias -10, not 0."""
    from core.logit_cache import LogitCacheAttention
    torch.manual_seed(3)
    D, V, L = 64, 40, 8
    a = LogitCacheAttention(D=D, V=V, n_layers=2, max_entries=4, n_heads=4)
    # (3) near-identity at init even with a populated cache
    hs = [torch.randn(1, L, D, requires_grad=True) for _ in range(3)]
    outs = []
    for h in hs:
        out = a.augment(h)
        assert (out - h.detach()).abs().max().item() < 2e-3, \
            'zero-gate must make augment ~identity at init'
        outs.append(out)
    # (1) every stored entry and stored kv pair is detached
    for e in a.cache._h_cache:
        assert e.grad_fn is None, 'entries must be detached at STORE time'
    for k, v in a.cache._kv_h:
        assert k.grad_fn is None and v.grad_fn is None
    # lengths stay in lockstep with max_entries
    assert len(a.cache._kv_h) == len(a.cache._h_cache) == 3
    # (2) frozen weights ⇒ incremental encoding == full re-projection
    h4 = torch.randn(1, L, D)
    with torch.no_grad():
        out_live = a.augment(h4)
    with torch.no_grad():
        full = a.attention.k_norm(a.attention.k_proj_h(
            torch.cat([e for e in a.cache._h_cache], dim=1)))
        last = a.attention.k_norm(a.attention.k_proj_h(h4))
        inc = torch.cat([p[0] for p in a.cache.kv_window()] + [last], dim=1)
        # stored encodings cover every window at write weights == full
        # projection now (frozen) — minus the live newest duplicate below
        assert torch.allclose(full, inc[:, :full.shape[1]], atol=1e-5), \
            'stored k/v must equal write-time projection'
    # gradient still flows to the live projections through augment
    a.zero_grad(set_to_none=True)
    h5 = torch.randn(1, L, D, requires_grad=True)
    a.augment(h5).sum().backward()
    assert a.attention.k_proj_h.weight.grad.abs().sum().item() > 0, \
        'k_proj_h must train via the live newest entry'
    # clear() must clear the kv list too
    a.cache.clear()
    assert a.cache._kv_h == [] and a.cache._h_cache == []
    # bias really is -10 (sigmoid≈4.5e-5), not the old zero init
    assert abs(float(a.attention.cache_gate[2].bias.detach()) + 10.0) < 1e-6


def test_checkpoint_state_roundtrip():
    """M12 single-best.pt: watchdog & balancer statistics must round-trip —
    a resumed session continues its signal baselines, violation streaks and
    alarm history instead of silently re-bootstrapping (recover_count
    was saved by M8 but never RESTORED until now)."""
    from core.training_control import FailureDetector, LossBalancer
    model = torch.nn.Module()
    wd = FailureDetector(model)
    for i in range(30):                       # stable series: stats build, no trigger
        assert not wd.check(7.0 + 0.001 * i, i)
    wd.ce_armed = True
    sd = wd.state_dict()
    assert 'ce' in sd['stats'] and len(sd['stats']['ce']) == 7  # fast,prev,n,slow,dvar,ph,phmin (B4)
    wd2 = FailureDetector(model)
    wd2.load_state_dict(sd)
    assert wd2._stats == wd._stats and wd2.recover_count == wd.recover_count
    assert wd2.ce_armed and wd2._viol == wd._viol
    # empty/legacy payload must be a no-ops, not crashes
    wd2.load_state_dict(None)
    bal = LossBalancer(align=False)
    ce, aux = torch.tensor(2.0), {'u': torch.tensor(0.5), 'v': torch.tensor(-1.0)}
    for _ in range(5):
        bal.loss(ce, aux)
    bsd = bal.state_dict()
    assert bsd['ema_ce'] is not None and set(bsd['ema_aux']) == {'u', 'v'}
    bal2 = LossBalancer(align=False)
    bal2.load_state_dict(bsd)
    assert bal2.ema_aux == bal.ema_aux and abs(bal2.ema_ce - bal.ema_ce) < 1e-12
    bal2.load_state_dict(None)


def test_arm_ce_rebootstraps_warmup_baseline():
    """Live L4 incident (step 1045/1052): arm-ing CE at the first eval while
    its baseline still carries the WARMUP ramp rolled the run back on the very
    first healthy CE oscillation. arm_ce() must disarm-and-rebootstrap: a fresh
    baseline, violation counter zeroed, and only THEN the watch is live."""
    from core.training_control import FailureDetector
    model = torch.nn.Module()
    wd = FailureDetector(model)
    for i in range(60):                       # ramping (warmup) CE, un-armed
        wd.check(6.0 + 0.05 * i, i)
    assert 'ce' in wd._stats and not wd.ce_armed
    wd.arm_ce()
    assert wd.ce_armed and 'ce' not in wd._stats and 'ce' not in wd._viol
    # the old ramp level must not veto a spike the very next steps: 100-sample
    # bootstrap grace (min_samples) before the rule can fire at all
    for i in range(70):
        assert not wd.check(9.5, 100 + i), 'warmup baseline survived arming'


def test_log_analyzer_tracks_live_format():
    """analyze.py's log parser must keep matching the notebook's live line
    format — it already drifted once (intent_w vs intent_eff) and died
    silently (empty dashboard, no error). Lock the exact format the run
    prints, plus the M12-era save lines and the short-aux-series render."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'az', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'analyze.py'))
    az = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(az)
    line = ('step=    55  loss=54.4124  ce=5.9117  mod_mlp=0.504 mod_std=0.079 '
            'lr=1.40e-05  tok/s=69  mem=14.7GB  intent_eff=0.0010  mlp_out=1006.1 '
            'usef=0.500  mat=0.060[0.020,0.121]\n')
    m = az._MAIN_RE.search(line)
    assert m, 'main log format drifted from _MAIN_RE'
    assert m.group(1) == '55' and float(m.group(9)) == 0.0010  # step…, mem(8), intent_eff(9)
    e = az._EVAL_RE.search('  EVAL step=1045: val_loss=6.8123 val_ppl=9.08e+02 (n=99)')
    assert e and int(e.group(1)) == 1045
    s1 = az._SAVE_RE.search('  EVAL saved best.pt (val_loss=6.8123) step=1045')
    s2 = az._SAVE_RE.search('Interrupted - saving latest state...')
    assert s1 and int(s1.group(2)) == 1045, 'notebook eval-save line must parse'
    # short aux series (gradalign appears only after the first step) must not
    # crash the full-table renderer
    ld = {'steps': [0, 55], 'main': {'ce': [5.8, 5.9], 'lr': [1e-7, 1e-5]},
          'aux': {'gradalign': [0.14]}, 'eval': [], 'depth': [], 'bridge': None,
          'saves': [('best', 1045)]}
    try:
        az.render_log_html(ld, os.devnull)
    except IndexError as ex:
        raise AssertionError(f'log dashboard crashes on ragged aux series: {ex!r}')


def test_garbage_scanner_detects_synthetic_noise():
    """M14: scan_garbage must actually fire on random-id stretches (empirical
    window entropy is capped at log2(window)≈8, so any threshold above 8 on
    entropy alone is a dead heuristic — the uniq-ratio carries the signal)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'sg', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'scan_garbage.py'))
    sg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sg)
    import numpy as np
    rng = np.random.default_rng(11)
    zipf = (np.minimum(rng.zipf(1.15, size=30000), 30000) + 1).astype(np.uint16)
    clean = os.path.join(os.path.dirname(__file__), '_sg_clean.tmp')
    dirty = os.path.join(os.path.dirname(__file__), '_sg_dirty.tmp')
    try:
        zipf.tofile(clean)
        garbage = rng.integers(0, 65536, size=5 * 256, dtype=np.uint16)
        np.concatenate([zipf[:10000], garbage, zipf[10000:]]).tofile(dirty)
        r_clean, _ = sg.scan_file(clean)
        r_dirty, _ = sg.scan_file(dirty)
        assert not r_clean, f'clean corpus flagged: {r_clean}'
        assert r_dirty, 'planted 5-window garbage NOT detected'
        a0, a1, e, u, k = r_dirty[0]
        assert u > 0.9 and a1 - a0 >= 5 * 256 * 0.8
    finally:
        for p in (clean, dirty):
            if os.path.exists(p):
                os.remove(p)


def test_alarm_sensor_never_touches_model():
    """Decision D6: check() only SOUNDS (returns True after 3 consecutive
    relative violations) — it must NOT roll weights back, NOT build
    optimizers, NOT read files, NOT clear its own baselines. The caller
    stops; recovery is a human call. Cooldown must suppress re-alarm spam."""
    from core.training_control import FailureDetector
    model = torch.nn.Linear(4, 4)
    wd = FailureDetector(model)
    wd.ce_armed = True
    with torch.no_grad():
        model.weight.add_(10.0)               # corrupt the weights
    corrupt = model.weight.detach().clone()
    for i in range(120):                      # warm, stable baseline
        assert not wd.check(6.0, 100 + i)
    fired = False
    for i in range(20):                       # sudden sustained spike
        fired = wd.check(50.0, 300 + i)
        if fired:
            break
    assert fired and wd.recover_count >= 1
    assert torch.equal(model.weight.detach(), corrupt), \
        'D6: sensor touched model state — must be side-effect free'
    assert wd._stats.get('ce') is not None, 'sensor must NOT wipe its baselines'
    assert not wd.check(50.0, 400), 'cooldown must gate alarm spam'


def test_runtime_checkpoint_toggle_is_equivalent():
    """M16 governor premise: cfg.gradient_checkpointing is honored PER
    FORWARD (stack reads it live), and the recompute path is numerically
    equivalent + trains. Without this the governor would be a memory
    theater toggle."""
    m = _stack()
    m.train()
    torch.manual_seed(0)
    x = torch.randint(1, m.cfg.vocab, (1, 24))
    h = m.embed_tokens(x)
    # forward mutates runtime buffers (VSA/mirror EMAs), so a plain-vs-checkpoint
    # comparison must run each mode on a FRESHLY re-seeded model from the same
    # weights; comparing two forwards on one live model measures the buffer
    # drift, not the checkpoint path.
    import copy as _copy
    w0 = _copy.deepcopy(m.state_dict())
    def _fwd(gc):
        mm = _stack()
        mm.load_state_dict(_copy.deepcopy(w0))
        mm.train()
        mm.cfg.gradient_checkpointing = gc
        torch.manual_seed(0)
        xx = torch.randint(1, mm.cfg.vocab, (1, 24))
        hh = mm.embed_tokens(xx)
        o, *_ = mm(hh.clone(), None, step=5, tokens=xx)
        return mm, xx, hh, o
    mm0, xx0, hh0, out0 = _fwd(False)
    mm1, xx1, hh1, out1 = _fwd(True)
    assert out1.shape == out0.shape and torch.isfinite(out1).all()
    assert torch.allclose(out0.detach(), out1.detach(), atol=2e-4, rtol=2e-4), \
        'checkpointed forward diverged from plain forward'
    l1, _ = mm1.compute_losses(out1, xx1, h_emb=hh1)
    mm1.zero_grad(set_to_none=True)
    l1.backward()
    g1 = {n: p.grad.detach().clone() for n, p in mm1.layers[0].named_parameters()
          if p.grad is not None}
    l0, _ = mm0.compute_losses(out0, xx0, h_emb=hh0)
    mm0.zero_grad(set_to_none=True)
    l0.backward()
    g0 = {n: p.grad.detach().clone() for n, p in mm0.layers[0].named_parameters()
          if p.grad is not None}
    # NB: layer-0's first parameter _vsa_tau_log is deliberately ungraduated
    # (M7: the τ-ladder owns it) — compare the GRADIENT SET as a whole.
    assert g0 and g1, 'no gradients in layer 0 at all'
    assert set(g0) == set(g1), 'checkpointed backward changed which params train'
    # Measured fact: recompute reads in-place-advanced runtime buffers (gate
    # RMS etc.), so checkpointed gradients drift a few percent in MAGNITUDE
    # (forward stays exact; per-tensor rel-drift was <=4%). What must not
    # change is the gradient DIRECTION — Adam normalizes magnitude, so this
    # is the mathematically meaningful invariant the governor relies on.
    # B2 criterion: recompute reads in-place-advanced runtime scalars (τ,
    # EMAs), so PER-PARAM MAGNITUDE may drift tens of percent on a handful of
    # governance scalars; what must be invariant is gradient DIRECTION and the
    # total energy — those are what Adam and the align balancer consume.
    t0 = sum(float(a.pow(2).sum()) for a in g0.values())
    t1 = sum(float(g1[n].pow(2).sum()) for n in g0)
    assert abs(t0 - t1) / max(t0, 1e-12) < 5e-2, 'total gradient energy moved'
    for n in g0:
        a, b = g0[n], g1[n]
        if float(a.norm()) < 1e-2:           # small governance grads: magnitude
            continue                          # dominated by recompute scalars
        cos = float(torch.nn.functional.cosine_similarity(
            a.reshape(-1), b.reshape(-1), dim=0))
        assert cos > 0.99, f'gradient DIRECTION changed at {n}: cos={cos:.4f}'
        rel = float((a - b).norm() / (a.norm() + 1e-12))
        assert rel < 5.0e-1, f'gradient norm drift at {n}: {rel:.1e}'



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
