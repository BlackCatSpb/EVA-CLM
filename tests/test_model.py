"""Comprehensive tests for WideBind core components."""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import WideBindConfig
from core.lambda_utils import LambdaConfig
from core.model import (
    WideBindStack, WideBindBlock, GroupedCognitiveMirror, GroupedMLP,
    PartitionedEmbedding, PartitionedHead, LmHead,
    sparse_block_codes, dct_basis, vsa_prefix_scan,
)
from core.live_inference import LiveInference, MirrorMonitor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


# ─── Sparse Block Codes ──────────────────────────────────────────────

def test_sparse_codes_exact_sparsity():
    codes = sparse_block_codes(vocab=5000, K=32, S=6)
    assert codes.shape == (5000, 32)
    counts = codes.sum(dim=-1)
    assert (counts == 6).all()


def test_sparse_codes_bits_used():
    codes = sparse_block_codes(vocab=5000, K=32, S=6)
    freq = codes.sum(dim=0)
    assert (freq > 0).all()


def test_sparse_codes_deterministic():
    c1 = sparse_block_codes(vocab=100, K=32, S=6)
    c2 = sparse_block_codes(vocab=100, K=32, S=6)
    assert (c1 == c2).all()


def test_sparse_codes_combinadic_coverage():
    codes = sparse_block_codes(vocab=5000, K=32, S=6)
    seen = set()
    for v in range(5000):
        bits = tuple(codes[v].nonzero(as_tuple=True)[0].tolist())
        seen.add(bits)
    assert len(seen) == 5000


def test_sparse_codes_prefix_stable():
    small = sparse_block_codes(vocab=500, K=32, S=6)
    big = sparse_block_codes(vocab=1000, K=32, S=6)
    assert torch.equal(small, big[:500])


# ─── PartitionedEmbedding ──────────────────────────────────────────

def test_partitioned_embed_shape():
    cfg = WideBindConfig(D=512, code_dim=16, code_sparsity=4, vocab=1820)
    emb = PartitionedEmbedding(cfg)
    tokens = torch.randint(0, 1800, (2, 16))
    h = emb(tokens)
    assert h.shape == (2, 16, 512)


def test_partitioned_embed_gradient_grouping():
    cfg = WideBindConfig(D=512, code_dim=16, code_sparsity=4, vocab=1820)
    emb = PartitionedEmbedding(cfg)
    tokens = torch.randint(0, 1800, (4, 32))
    h = emb(tokens)
    loss = h.sum()
    loss.backward()
    codes = emb.codes[tokens]
    for k in range(emb.K):
        active = codes[:, :, k].sum().item() > 0
        grad_norm = emb.basis.grad[k].norm().item()
        if active:
            assert grad_norm > 0
        else:
            assert grad_norm == 0.0


def test_partitioned_embed_small_vocab():
    cfg = WideBindConfig(D=512, code_dim=16, code_sparsity=4, vocab=1800)
    emb = PartitionedEmbedding(cfg)
    tokens = torch.randint(0, 1800, (1, 8))
    h = emb(tokens)
    assert h.shape == (1, 8, 512)


def test_partitioned_embed_grad_nonzero_with_active_bits():
    cfg = WideBindConfig(D=512, code_dim=16, code_sparsity=4, vocab=1820)
    emb = PartitionedEmbedding(cfg)
    tokens = torch.tensor([[42]])
    h = emb(tokens)
    loss = h.sum()
    loss.backward()
    assert emb.basis.grad is not None
    assert emb.basis.grad.abs().sum().item() > 0


def test_partitioned_embed_fewer_params():
    cfg_dense = WideBindConfig(D=512, code_dim=16, code_sparsity=4, vocab=1820)
    emb = PartitionedEmbedding(cfg_dense)
    expected = 16 * (512 // 16)
    assert emb.basis.numel() == expected


# ─── PartitionedHead ───────────────────────────────────────────────

def test_partitioned_head_shape():
    cfg = WideBindConfig(D=512, code_dim=16, code_sparsity=4, vocab=1820)
    head = PartitionedHead(cfg)
    h = torch.randn(2, 16, 512)
    logits = head(h)
    assert logits.shape == (2, 16, 1820)


def test_partitioned_head_gradient_grouping():
    cfg = WideBindConfig(D=512, code_dim=16, code_sparsity=4, vocab=1820)
    head = PartitionedHead(cfg)
    h = torch.randn(4, 32, 512, requires_grad=True)
    logits = head(h)
    loss = logits.sum()
    loss.backward()
    for k in range(head.K):
        grad_norm = head.readout.grad[k].norm().item()
        assert grad_norm > 0


def test_partitioned_head_zero_h_gives_uniform_logits():
    cfg = WideBindConfig(D=512, code_dim=16, code_sparsity=4, vocab=1820)
    head = PartitionedHead(cfg)
    h = torch.zeros(1, 1, 512)
    logits = head(h)
    assert (logits == 0).all()


# ─── VSA Prefix Scan ─────────────────────────────────────────────

def test_vsa_scan_exact():
    B, L, D = 1, 4, 2
    a = torch.full((B, L, D), 0.5)
    b = torch.ones(B, L, D)
    out, final = vsa_prefix_scan(a, b)
    expected = torch.tensor([[[1.0, 1.0], [1.5, 1.5], [1.75, 1.75], [1.875, 1.875]]])
    assert torch.allclose(out, expected, atol=1e-5)
    assert torch.allclose(final, expected[:, -1:])


def test_vsa_scan_with_state():
    B, L, D = 1, 2, 1
    a = torch.full((B, L, D), 0.5)
    b = torch.ones(B, L, D)
    state = torch.full((B, D), 10.0)
    out, final = vsa_prefix_scan(a, b, state)
    expected_out = torch.tensor([[[6.0], [4.0]]])
    expected_final = torch.tensor([[4.0]])
    assert torch.allclose(out, expected_out, atol=1e-5)
    assert torch.allclose(final, expected_final, atol=1e-5)


def test_vsa_scan_batched():
    B, L, D = 3, 8, 5
    a = torch.rand(B, L, D)
    b = torch.rand(B, L, D)
    out, final = vsa_prefix_scan(a, b)
    assert out.shape == (B, L, D)
    assert final.shape == (B, D)
    mem = b[:, 0:1].clone()
    for t in range(1, L):
        mem = a[:, t:t+1] * mem + b[:, t:t+1]
    assert torch.allclose(final, mem[:, -1], atol=1e-5)


# ─── GroupedCognitiveMirror ──────────────────────────────────────────

def test_mirror_shape():
    D, G, k = 512, 4, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    B, L = 2, 16
    h = torch.randn(B, L, D)
    mem_all = torch.randn(B, L, D)
    out, mlp_mod, mem_mod, *_ = mirror(h, mem_all)
    assert out.shape == (B, L, D)
    assert mlp_mod.shape == (B, L, G) and mem_mod.shape == (B, L, G)


def test_mirror_alpha_is_diag():
    D, G, k = 512, 4, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    assert mirror.alpha_diag.shape == (G, k)
    assert mirror.alpha_diag.requires_grad
    a = mirror.alpha_diag.data
    assert (a > 0.6).all() and (a < 1.0).all()


def test_mirror_no_lo_hi_split():
    D, G, k = 512, 4, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    B, L = 2, 8
    h = torch.randn(B, L, D)
    mem_all = torch.randn(B, L, D)
    out, *_ = mirror(h, mem_all)
    assert out.shape == (B, L, D)


def test_mirror_skip_connection_preserves_gradient():
    D, G, k = 256, 4, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    B, L = 1, 4
    h = torch.randn(B, L, D)
    mem_all = torch.randn(B, L, D)
    out, *_ = mirror(h, mem_all)
    loss = out.sum()
    loss.backward()
    assert mirror.log_scale.grad is not None
    assert mirror.log_scale.grad.norm().item() > 0


def test_mirror_per_expert_gates():
    D, G, k = 512, 4, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    B, L = 4, 16
    h = torch.randn(B, L, D)
    mem_all = torch.randn(B, L, D)
    out = mirror(h, mem_all)
    gate = mirror._last_gates
    assert gate.shape == (G,)
    assert (gate >= 0).all() and (gate <= 1).all()


def test_mirror_grad_cache():
    D, G, k = 512, 4, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    grad_h = torch.randn(4, 16, D)
    mirror.cache_grad_norms(grad_h)
    norms = mirror._prev_grad_norm
    assert norms.shape == (G,)
    assert (norms >= 0).all()


def test_mirror_global_state():
    D, G, k = 512, 4, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    B, L = 2, 16
    h = torch.randn(B, L, D)
    mem_all = torch.randn(B, L, D)
    global_state = torch.randn(1, 1, D)
    out_with, *_ = mirror(h, mem_all, global_state)
    out_without, *_ = mirror(h, mem_all, global_state=None)
    assert out_with.shape == out_without.shape


def test_mirror_conv_smooth_all_channels_active():
    for _ in range(3):
        k = 4
        mirror = GroupedCognitiveMirror(D=512, G=4, k=k)
        w = mirror.conv_smooth.weight.data
        assert w.shape == (4 * k, 1, 3)
        assert w[:, 0, 1].eq(1.0).all()


def test_mirror_conv_smooth_produces_temporal_diff():
    G, k = 4, 4
    D = G * (512 // G)
    mirror = GroupedCognitiveMirror(D=D, G=G, k=k)
    B, L = 2, 32
    h = torch.randn(B, L, D).reshape(B, L, G, D // G)
    hp = torch.einsum('blgd,gdk->blgk', h, mirror.W_proj.data)
    hp_perm = hp.permute(0, 2, 3, 1).reshape(B, G * k, L)
    hp_pad = F.pad(hp_perm, (2, 0))
    hp_smooth = mirror.conv_smooth(hp_pad)[:, :, :L]
    hp_smooth_r = hp_smooth.reshape(B, G, k, L).permute(0, 3, 1, 2)
    diff = (hp_smooth_r[:, 1:] - hp[:, :-1]).abs().mean()
    assert diff < 1e-5


# ─── GroupedMLP ─────────────────────────────────────────────────────

def test_mlp_shape():
    D, G = 512, 4
    mlp = GroupedMLP(D, expand=4, groups=G)
    h = torch.randn(2, 16, D)
    out = mlp(h)
    assert out.shape == h.shape


def test_mlp_nonzero():
    D, G = 512, 4
    mlp = GroupedMLP(D, expand=4, groups=G)
    h = torch.randn(1, 4, D)
    out = mlp(h)
    assert out.abs().sum().item() > 0


# ─── WideBindStack (end-to-end) ─────────────────────────────────────

def test_stack_forward():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg).to(device)
    x = torch.randint(0, cfg.vocab, (2, 8), device=device)
    h = model.embed_tokens(x)
    out, state, global_state, _ = model(h)
    assert out.shape == h.shape
    assert len(state) == cfg.n_layers


def test_stack_forward_twice_with_state():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg).to(device)
    x = torch.randint(0, cfg.vocab, (1, 8), device=device)
    h = model.embed_tokens(x)
    out1, state1, gs1, _ = model(h)
    out2, state2, gs2, _ = model(h, state1, gs1)
    assert out2.shape == out1.shape


def test_stack_loss():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg).to(device)
    x = torch.randint(0, cfg.vocab, (2, 8), device=device)
    h = model.embed_tokens(x)
    out, _, _, _ = model(h)
    loss = model.compute_loss(out[:, :-1], x[:, 1:])
    assert loss.item() > 0
    loss.backward()
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert total_grad > 0


def test_stack_param_count():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg)
    n = model.param_count()
    assert n > 0


def test_stack_embed_alignment():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg)
    K = cfg.code_dim
    assert model.embed.K == K
    assert model.lm_head.K == K


def test_strict_false_compatibility():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg)
    old_sd = {k: v for k, v in model.state_dict().items()
              if not any(b in k for b in ['_last_gates', '_last_h_pool', '_prev_grad_norm', '_last_magnitude'])}
    model.load_state_dict(old_sd, strict=False)
    x = torch.randint(0, cfg.vocab, (1, 4))
    h = model.embed_tokens(x)
    out, _, _, _ = model(h)
    loss = model.compute_loss(out[:, :-1], x[:, 1:])
    assert not torch.isnan(loss)


# ─── DCT Basis ──────────────────────────────────────────────────────

def test_dct_basis_orthogonal():
    for n in [64, 128, 256]:
        V = dct_basis(n)
        product = V @ V.T
        diff = (product - torch.eye(n)).abs().max().item()
        assert diff < 1e-5, f'n={n}: max diff={diff}'


def test_dct_basis_first_row():
    V = dct_basis(512)
    expected = torch.full((512,), math.sqrt(2.0 / 512) / math.sqrt(2)) * 1.0
    assert torch.allclose(V[0], expected, atol=1e-6)


# ─── AdaptiveController ─────────────────────────────────────────────

def test_adaptive_controller_ranges():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg)
    from core.model import AdaptiveController
    expl, diff = AdaptiveController.stats(model.layers)
    assert 0 <= expl <= 1
    assert 0 <= diff <= 1
    b_i = AdaptiveController.b_i(model.layers)
    b_d = AdaptiveController.b_d(model.layers)
    assert -3 <= b_i <= 0
    assert 2 <= b_d <= 5
    scale = AdaptiveController.w_mem2v_scale(model.layers)
    assert 0.5 <= scale <= 1.0
    alpha = AdaptiveController.ema_alpha(model.layers)
    assert 0.90 <= alpha <= 0.995


# ─── Config integration tests ──────────────────────────────────────────

def test_config_adaptive_controller_thresholds():
    cfg = WideBindConfig(**SMALL, lambda_d_enabled=False,
                         exploration_threshold=0.5, differentiation_threshold=0.5)
    model = WideBindStack(cfg)
    h = torch.randn(1, 4, cfg.D)
    model(h)
    from core.model import AdaptiveController
    expl, diff = AdaptiveController.stats(model.layers,
        expl_thresh=cfg.exploration_threshold, diff_thresh=cfg.differentiation_threshold)
    assert 0 <= expl <= 1
    assert 0 <= diff <= 1


def test_config_init_values():
    k = 4
    cfg = WideBindConfig(**SMALL, mirror_k=k, mirror_k_staircase=False,
                         lambda_d_enabled=False,
                         log_scale_init_std=0.1,
                         w_d_init_std=0.5, conv_init_std=0.05)
    model = WideBindStack(cfg)
    m0 = model.layers[0].mirror
    assert m0.alpha_diag.shape == (cfg.mlp_groups, k)
    assert m0.tanh_bias.shape == (cfg.mlp_groups, k)
    w_d_std = model.layers[0].w_d.data.std().item()
    assert abs(w_d_std - 0.5) < 0.2
    conv_std = model.layers[0].conv.weight.data.std().item()
    assert conv_std > 0


def test_config_param_groups_multipliers():
    cfg = WideBindConfig(**SMALL, lambda_d_enabled=False,
                         lambda_lr_hierarchy=False, gate_lr_mult=3.0)
    model = WideBindStack(cfg)
    groups = model.param_groups(1e-4)
    param_to_name = {id(p): n for n, p in model.named_parameters()}
    found_gate = False
    for g in groups:
        for p in g['params']:
            name = param_to_name.get(id(p), '')
            if any(x in name for x in ['.w_gate', '.b_gate', '.log_skip']):
                assert abs(g['lr'] - 3e-4) < 1e-7
                found_gate = True
    assert found_gate


def test_lambda_d_hierarchy():
    cfg = WideBindConfig()
    lc = LambdaConfig(3)
    assert abs(cfg.exploration_threshold - lc.exploration_threshold) < 1e-6
    assert abs(cfg.differentiation_threshold - lc.differentiation_threshold) < 1e-6
    assert abs(cfg.ema_alpha_max - lc.ema_alpha_max) < 1e-6
    assert abs(cfg.gate_lr_mult - lc.gate_lr_mult) < 1e-6
    assert cfg.warmup_steps == lc.warmup_steps
    assert cfg.eval_interval == lc.eval_interval
    cfg2 = WideBindConfig(lambda_d_enabled=False)
    assert abs(cfg2.exploration_threshold - 0.25) < 1e-6
    assert abs(cfg2.ema_alpha_max - 0.99) < 1e-6
    assert cfg2.warmup_steps == 1000


# ─── LiveInference ─────────────────────────────────────────────────

def test_live_inference_basic():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg).to(device)
    model.eval()
    live = LiveInference(model, cfg)
    h = model.embed_tokens(torch.randint(0, cfg.vocab, (1, 4), device=device))
    out = live.respond(h)
    assert out.shape == (1, 4, cfg.D)


def test_live_inference_state_persists():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg).to(device)
    model.eval()
    live = LiveInference(model, cfg)
    h1 = model.embed_tokens(torch.randint(0, cfg.vocab, (1, 4), device=device))
    live.respond(h1)
    h2 = model.embed_tokens(torch.randint(0, cfg.vocab, (1, 4), device=device))
    live.respond(h2)
    assert live.layer_states is not None
    assert live.global_state is not None


def test_live_inference_think():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg).to(device)
    model.eval()
    live = LiveInference(model, cfg)
    with torch.no_grad():
        h = live.think(n_steps=5)
    assert h.shape == (1, 1, cfg.D)


def test_live_inference_think_persists():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg).to(device)
    model.eval()
    live = LiveInference(model, cfg)
    with torch.no_grad():
        live.think(n_steps=5)
    assert live.step > 0


def test_live_inference_reset():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg).to(device)
    model.eval()
    live = LiveInference(model, cfg)
    h = model.embed_tokens(torch.randint(0, cfg.vocab, (1, 4), device=device))
    live.respond(h)
    assert live.layer_states is not None
    live.reset_state()
    assert live.layer_states is None
    assert live.global_state is None


# ─── MirrorMonitor ────────────────────────────────────────────────

def test_mirror_monitor_trace():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg).to(device)
    model.eval()
    monitor = MirrorMonitor(model)
    x = torch.randint(0, cfg.vocab, (2, 8), device=device)
    h = model.embed_tokens(x)
    with torch.no_grad():
        model(h)
    monitor.capture()
    assert len(monitor.history['step']) == 1
    assert 'expert_gates' in monitor.history
    assert 'tau' in monitor.history
    summary = monitor.summary(window=1)
    assert 'expert_gates_mean' in summary


def test_mirror_monitor_rolling():
    cfg = WideBindConfig(**SMALL)
    model = WideBindStack(cfg).to(device)
    model.eval()
    monitor = MirrorMonitor(model, max_history=5)
    for _ in range(10):
        x = torch.randint(0, cfg.vocab, (1, 4), device=device)
        h = model.embed_tokens(x)
        with torch.no_grad():
            model(h)
        monitor.capture()
    assert len(monitor.history['step']) == 5


# ─── Alpha-specific tests ───────────────────────────────────────────

def test_alpha_gradient_stronger_than_wpred():
    D, G, k = 512, 4, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    B, L = 2, 16
    h = torch.randn(B, L, D)
    mem_all = torch.randn(B, L, D)
    out, *_ = mirror(h, mem_all)
    loss = out.sum() * 0.01
    loss.backward()
    assert mirror.alpha_diag.grad.norm().item() > 0


def test_alpha_deviation_on_structured_data():
    cfg = WideBindConfig(D=512, n_layers=2, mlp_groups=4, mirror_k=4,
                         code_dim=16, code_sparsity=4, vocab=1000)
    model = WideBindStack(cfg)
    opt = torch.optim.AdamW(model.param_groups(), lr=1e-3)
    for step in range(50):
        x = torch.randint(0, 100, (2, 8))
        h = model.embed_tokens(x)
        out, _, _, _ = model(h, None)
        loss = model.compute_loss(out, x)
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        idiff = torch.stack([
            (1.0 - l.mirror.alpha_diag.data).abs().mean()
            for l in model.layers
        ]).mean().item()
    assert idiff > 0


def test_no_lo_hi_split_grad_to_all_k():
    D, G, k = 512, 4, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    B, L = 1, 4
    h = torch.randn(B, L, D, requires_grad=True)
    mem_all = torch.randn(B, L, D)
    out, *_ = mirror(h, mem_all)
    loss = out.sum()
    loss.backward()
    assert mirror.W_proj.grad is not None
    assert mirror.W_out.grad is not None
    assert mirror.log_skip_alpha.grad is not None
    assert mirror.log_dvar_mod_scale.grad is not None
    assert mirror.log_grad_mod_scale.grad is not None


def test_D4096_G32_forward():
    cfg = WideBindConfig(n_layers=2, D=512, mlp_groups=4, mirror_k=4,
                          code_dim=16, code_sparsity=4, vocab=1820)
    model = WideBindStack(cfg)
    x = torch.randint(0, 100, (1, 4))
    h = model.embed_tokens(x)
    out, _, _, _ = model(h, None)
    assert out.shape == (1, 4, 512)
    n = model.param_count()
    assert n > 0


def test_gradient_grouping_demonstrable():
    cfg = WideBindConfig(D=512, code_dim=16, code_sparsity=4, vocab=1820)
    emb = PartitionedEmbedding(cfg)
    tokens = torch.tensor([[0, 1, 2, 42, 100, 500, 1000, 1500]])
    h = emb(tokens)
    loss = h.sum()
    loss.backward()
    for k in range(emb.K):
        is_active_anywhere = emb.codes[tokens][:, :, k].any().item()
        grad = emb.embed_mix.grad[k].norm().item()
        if not is_active_anywhere:
            assert grad == 0.0
        if is_active_anywhere:
            assert grad > 0


# ─── LayerBridgeGate ───────────────────────────────────────────────

def test_layer_bridge_gate_shape():
    from core.layer_bridge_gate import LayerBridgeGate
    n_layers = 2
    D = 512
    gate = LayerBridgeGate(n_layers=n_layers)
    layer_outputs = torch.randn(n_layers, 2, D)
    diagnostics = torch.randn(n_layers, 6)
    tau = torch.tensor([0.5, 0.8])
    bridge_input, gate_weights, gate_info = gate(layer_outputs, diagnostics, tau)
    assert bridge_input.shape == (2, D)
    assert gate_weights.shape[0] == n_layers
    assert isinstance(gate_info, dict)


def test_layer_bridge_gate_nan_control():
    from core.layer_bridge_gate import LayerBridgeGate
    n_layers = 2
    D = 512
    gate = LayerBridgeGate(n_layers=n_layers)
    layer_outputs = torch.randn(n_layers, 2, D)
    diagnostics = torch.randn(n_layers, 6)
    diagnostics[0, :3] = float('nan')
    tau = torch.tensor([0.5, 0.8])
    bridge_input, gate_weights, health_scores = gate(layer_outputs, diagnostics, tau)
    assert not torch.isnan(bridge_input).any(), 'NaN in bridge_input'


def test_layer_bridge_gate_explosion_control():
    from core.layer_bridge_gate import LayerBridgeGate
    n_layers = 2
    D = 512
    gate = LayerBridgeGate(n_layers=n_layers)
    layer_outputs = torch.randn(n_layers, 2, D)
    diagnostics = torch.full((n_layers, 6), 1e6)
    tau = torch.tensor([0.5, 0.8])
    bridge_input, gate_weights, health_scores = gate(layer_outputs, diagnostics, tau)
    assert not torch.isnan(bridge_input).any()


def test_maturation_no_warmup():
    from core.maturation import MaturationController
    cfg = WideBindConfig(**SMALL)
    mc = MaturationController(n_layers=cfg.n_layers, tau_min=1e-4, tau_max=1.0, cfg=cfg)
    assert not hasattr(mc, 'set_resume_step')
    assert not hasattr(mc, 'warmup_steps')
    gate = mc.step_gate(step=1000)
    assert gate.shape == (cfg.n_layers,)
    assert (gate >= 0).all() and (gate <= 1).all()


# ─── Run all ────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [fn for fn in dir() if fn.startswith('test_')]
    passed = 0
    failed = 0
    for name in tests:
        try:
            globals()[name]()
            print(f'  PASS  {name}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {name}: {e}')
            failed += 1
            import traceback
            traceback.print_exc()

    print(f'\n{passed}/{passed + failed} passed')
    sys.exit(0 if failed == 0 else 1)
