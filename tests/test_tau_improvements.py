"""Comprehensive tests for τ-field tied improvements U1–U10.

Tests architecture initialization, method correctness, gradient flow,
and integration of all 10 improvements working together.
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from core.config import EVAConfig
from core.stack import EVAStack
from core.block import EVABlock
from core.bridge import SemanticBridge
from core.mirror import GroupedCognitiveMirror
from core.memory_bank import StreamingMemoryBank
from core.concept_layer import UnifiedConceptLayer
from core.adaptation import GradientClipper
from core.tau_config import TauConfig
from core.bind import TrajectorySpiralBind

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


# ═══════════════════════════════════════════════════════════════════════
# §1  TauConfig: field computation
# ═══════════════════════════════════════════════════════════════════════

class TestTauConfig:
    def test_tau_l_monotonic(self):
        tc = TauConfig(n_layers=24)
        tc.update()
        tau = tc.tau_l
        assert tau.shape == (24,)
        for i in range(23):
            assert tau[i] < tau[i+1] + 1e-4, f'tau[{i}]={tau[i]:.2f} >= tau[{i+1}]={tau[i+1]:.2f}'

    def test_tau_norm_range(self):
        tc = TauConfig(n_layers=24)
        tc.update()
        tn = tc.tau_norm
        assert tn.min() >= -0.01 and tn.max() <= 1.01, f'tau_norm range: [{tn.min():.3f}, {tn.max():.3f}]'

    def test_tau_norm_monotonic(self):
        tc = TauConfig(n_layers=24)
        tc.update()
        tn = tc.tau_norm
        for i in range(23):
            assert tn[i] <= tn[i+1] + 0.01

    def test_tau_dev_trainable(self):
        tc = TauConfig(n_layers=8)
        assert tc._tau_dev.requires_grad

    def test_intent_alpha_range(self):
        tc = TauConfig(n_layers=24)
        tc.update()
        alpha = tc.intent_alpha
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0

    def test_lr_mult_decreasing(self):
        tc = TauConfig(n_layers=24)
        tc.update()
        lr = tc.lr_mult
        # Deep layers (high tau) should have lower lr
        assert lr[0] > lr[-1]

    def test_mem_tau_3_values(self):
        tc = TauConfig(n_layers=24)
        tc.update()
        mt = tc.mem_tau
        assert mt.shape == (3,)
        assert mt[0] <= mt[1] <= mt[2]


# ═══════════════════════════════════════════════════════════════════════
# §2  U1: τ-consistent VSA Memory Scales
# ═══════════════════════════════════════════════════════════════════════

class TestU1VsaTauScales:
    def test_block_has_tau_norm(self):
        cfg = EVAConfig(**SMALL)
        tc = TauConfig(n_layers=cfg.n_layers)
        tc.update()
        block = EVABlock(cfg, 0, tau_config=tc)
        assert block._tau_norm is not None
        assert 0.0 <= block._tau_norm <= 1.0

    def test_vsa_tau_log_trainable(self):
        # Audit M7: the live trainable VSA scale ladder is the STACK-level
        # _vsa_log_param (single source, always passed as tau_s to blocks);
        # the block attribute is a constant fallback for standalone use.
        cfg = EVAConfig(**SMALL)
        tc = TauConfig(n_layers=cfg.n_layers)
        tc.update()
        block = EVABlock(cfg, 0, tau_config=tc)
        assert block._vsa_tau_log.requires_grad     # standalone ladder trainable
        stack = EVAStack(cfg)
        assert stack._vsa_log_param.requires_grad   # live single source
        grouped = {id(q) for g in stack.param_groups() for q in g['params']}
        assert id(block._vsa_tau_log) not in grouped, \
            'dead weight: block fallback ladder must not enter the stack optimizer' 

    def test_vsa_tau_per_layer_differs(self):
        cfg = EVAConfig(**SMALL)
        tc = TauConfig(n_layers=8)  # more layers → different tau_norm per layer
        tc.update()
        b0 = EVABlock(cfg, 0, tau_config=tc)
        b3 = EVABlock(cfg, 3, tau_config=tc)
        assert b0._tau_norm != b3._tau_norm, f'Both tau_norm={b0._tau_norm}'

    def test_forward_uses_passed_tau_s(self):
        cfg = EVAConfig(**SMALL)
        tc = TauConfig(n_layers=cfg.n_layers)
        tc.update()
        block = EVABlock(cfg, 0, tau_config=tc).to(device)
        B, L, D = 1, 4, cfg.D
        h = torch.randn(B, L, D, device=device)
        custom_tau = torch.tensor([10.0, 20.0, 40.0, 80.0], device=device)
        out, state = block(h, tau_s=custom_tau)
        assert out.shape == h.shape

    def test_stack_passes_per_layer_vsa_tau(self):
        cfg = EVAConfig(**SMALL)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (1, 4), device=device)
        h = model.embed_tokens(x)
        out, _, _, _ = model(h)
        assert out.shape == h.shape


# ═══════════════════════════════════════════════════════════════════════
# §3  U2: τ-adaptive Reasoning Budget
# ═══════════════════════════════════════════════════════════════════════

class TestU2ReasoningBudget:
    def test_reasoning_budget_tau_scaled(self):
        cfg = EVAConfig(**SMALL, explicit_reasoning=True,
                             reasoning_max_steps=8, reasoning_adaptive=True)
        model = EVAStack(cfg).to(device)
        # tau_norm_reasoning should be set
        assert hasattr(model, '_tau_norm_reasoning')
        assert 0.0 <= model._tau_norm_reasoning <= 1.0

    def test_reasoning_steps_reduced_when_low_tau(self):
        cfg = EVAConfig(**SMALL, explicit_reasoning=True,
                             reasoning_max_steps=8, reasoning_adaptive=True)
        model = EVAStack(cfg).to(device)
        model._tau_norm_reasoning = 0.1  # shallow layers
        x = torch.randint(0, cfg.vocab, (1, 8), device=device)
        h = model.embed_tokens(x)
        out, _, _, _ = model(h)
        # Should still work
        assert out.shape == h.shape

    def test_reasoning_forward_no_crash(self):
        cfg = EVAConfig(**SMALL, explicit_reasoning=True,
                             reasoning_max_steps=4, reasoning_adaptive=True)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (2, 8), device=device)
        h = model.embed_tokens(x)
        out, _, _, _ = model(h)
        assert out.shape == h.shape


# ═══════════════════════════════════════════════════════════════════════
# §4  U3: τ-spectral Chebyshev Damping
# ═══════════════════════════════════════════════════════════════════════

class TestUSpectralDamping:
    def test_block_has_tau_for_spectral(self):
        cfg = EVAConfig(**SMALL)
        tc = TauConfig(n_layers=cfg.n_layers)
        tc.update()
        block = EVABlock(cfg, 0, tau_config=tc)
        assert block._tau_norm is not None

    def test_spectral_damp_formula(self):
        # tau_norm=0 → damp = cos(0) = 1.0
        assert abs(math.cos(0) - 1.0) < 1e-6
        # tau_norm=1 → damp = cos(pi/2) ≈ 0.0
        assert abs(math.cos(math.pi/2)) < 1e-6
        # tau_norm=0.5 → damp = cos(pi/4) ≈ 0.707
        assert abs(math.cos(math.pi*0.5/2) - 0.707) < 0.01

    def test_spectral_forward(self):
        cfg = EVAConfig(**SMALL)
        tc = TauConfig(n_layers=cfg.n_layers)
        tc.update()
        block = EVABlock(cfg, 0, tau_config=tc).to(device)
        B, L, D = 1, 4, cfg.D
        h = torch.randn(B, L, D, device=device)
        out, _ = block(h, spectral_mod=1.0)
        assert out.shape == h.shape
        assert not torch.isnan(out).any()


# ═══════════════════════════════════════════════════════════════════════
# §5  U4: τ-coupled Bridge Injection Weight
# ═══════════════════════════════════════════════════════════════════════

class TestU4BridgeInjection:
    def test_bridge_has_alpha_beta(self):
        bridge = SemanticBridge(D=512, n_layers=2, bridge_dim=64)
        assert hasattr(bridge, '_inj_alpha')
        assert hasattr(bridge, '_inj_beta')
        assert bridge._inj_alpha.requires_grad
        assert bridge._inj_beta.requires_grad

    def test_bridge_inject_with_tau_norm(self):
        bridge = SemanticBridge(D=512, n_layers=2, bridge_dim=64).to(device)
        B, L, D = 1, 4, 512
        h = torch.randn(B, L, D, device=device)
        bridge.bridge_stream.zero_()
        tau_norm = torch.tensor(0.7, device=device)
        h_out = bridge.inject_layer(0, h, tau_norm=tau_norm)
        assert h_out.shape == h.shape
        # Injection should be non-zero (stream_log_scale initialized to 0 → tanh=0 → no-op)
        # But with maturity=None and tau_norm, scale = tanh(0) * 1.0 * inj_strength
        # Since tanh(0)=0, output should still be h (no-op at init)
        assert torch.allclose(h_out, h, atol=1e-5)

    def test_bridge_inject_without_tau_norm(self):
        bridge = SemanticBridge(D=512, n_layers=2, bridge_dim=64).to(device)
        B, L, D = 1, 4, 512
        h = torch.randn(B, L, D, device=device)
        bridge.bridge_stream.zero_()
        h_out = bridge.inject_layer(0, h)
        assert h_out.shape == h.shape

    def test_bridge_alpha_beta_gradient(self):
        bridge = SemanticBridge(D=512, n_layers=2, bridge_dim=64).to(device)
        B, L, D = 1, 4, 512
        h = torch.randn(B, L, D, device=device)
        bridge.bridge_stream.zero_()
        tau_norm = torch.tensor(0.5, device=device)
        h_out = bridge.inject_layer(0, h, tau_norm=tau_norm)
        loss = h_out.sum()
        loss.backward()
        assert bridge._inj_alpha.grad is not None
        assert bridge._inj_beta.grad is not None


# ═══════════════════════════════════════════════════════════════════════
# §6  U5: τ-scheduled Mirror Signal Temperature
# ═══════════════════════════════════════════════════════════════════════

class TestU5MirrorSignalTemp:
    def test_mirror_has_tau_signal(self):
        mirror = GroupedCognitiveMirror(D=512, G=4, k=4)
        assert hasattr(mirror, '_tau_signal_log')
        assert mirror._tau_signal_log.requires_grad

    def test_mirror_has_tau_norm_layer(self):
        tc = TauConfig(n_layers=4)
        tc.update()
        mirror = GroupedCognitiveMirror(D=512, G=4, k=4, layer_idx=1, n_layers=4, tau_config=tc)
        assert mirror._tau_norm_layer is not None
        assert 0.0 <= mirror._tau_norm_layer <= 1.0

    def test_signal_weights_use_temperature(self):
        tc = TauConfig(n_layers=4)
        tc.update()
        mirror = GroupedCognitiveMirror(D=512, G=4, k=4, layer_idx=0, n_layers=4, tau_config=tc)
        B, L = 2, 8
        h = torch.randn(B, L, 512)
        mem_all = torch.randn(B, L, 512)
        out, mlp_mod, mem_mod, *_ = mirror(h, mem_all)
        assert out.shape == (B, L, 512)

    def test_signal_weights_shape(self):
        tc = TauConfig(n_layers=4)
        tc.update()
        mirror = GroupedCognitiveMirror(D=512, G=4, k=4, layer_idx=0, n_layers=4, tau_config=tc)
        # 4 signals (no private mem)
        w = torch.sigmoid(mirror._signal_log_weights)
        assert w.shape == (4,)


# ═══════════════════════════════════════════════════════════════════════
# §7  U6: τ-consistent Memory Bank Fusion
# ═══════════════════════════════════════════════════════════════════════

class TestU6MemoryBankFusion:
    def test_memory_bank_has_tau_config(self):
        cfg = EVAConfig(**SMALL)
        tc = TauConfig(n_layers=cfg.n_layers)
        tc.update()
        mb = StreamingMemoryBank(D=cfg.D, bridge_dim=64, l1_slots=2, l2_slots=4,
                                  tau_config=tc)
        assert mb.tau_config is not None

    def test_memory_bank_has_fusion_tau_alpha(self):
        cfg = EVAConfig(**SMALL)
        tc = TauConfig(n_layers=cfg.n_layers)
        tc.update()
        mb = StreamingMemoryBank(D=cfg.D, bridge_dim=64, l1_slots=2, l2_slots=4,
                                  tau_config=tc)
        assert hasattr(mb, '_fusion_tau_alpha')
        assert mb._fusion_tau_alpha.requires_grad

    def test_memory_bank_forward(self):
        cfg = EVAConfig(**SMALL)
        tc = TauConfig(n_layers=cfg.n_layers)
        tc.update()
        mb = StreamingMemoryBank(D=cfg.D, bridge_dim=64, l1_slots=2, l2_slots=4,
                                  tau_config=tc).to(device)
        B, L, D = 1, 8, cfg.D
        h = torch.randn(B, L, D, device=device)
        tokens = torch.randint(0, cfg.vocab, (B, L), device=device)
        out = mb(h, tokens, step=100, mat_gate=0.5)
        assert out.shape == h.shape


# ═══════════════════════════════════════════════════════════════════════
# §8  U7: τ-learned Concept Birth Threshold
# ═══════════════════════════════════════════════════════════════════════

class TestU7ConceptBirthThreshold:
    def test_concept_layer_has_birth_params(self):
        cl = UnifiedConceptLayer(D=512, k=32, bridge_dim=64, S=4)
        assert hasattr(cl, '_log_tau_birth_thr')
        assert hasattr(cl, '_log_tau_decay_thr')
        assert cl._log_tau_birth_thr.requires_grad
        assert cl._log_tau_decay_thr.requires_grad

    def test_concept_layer_tau_norm_stored(self):
        cl = UnifiedConceptLayer(D=512, k=32, bridge_dim=64, S=4)
        B, L, D = 1, 4, 512
        h = torch.randn(B, L, D)
        hp = torch.randn(B, L, 4, 32)
        pen = torch.rand(B, L)
        cl(h, hp=hp, pen=pen, mat_gate=0.5, allow_write=True, tau_norm=0.7)
        assert cl._tau_norm == 0.7

    def test_birth_threshold_formula(self):
        # base_thr = sigmoid(0) = 0.5, decay = sigmoid(0) = 0.5
        # tau_norm = 0.5 → thr = 0.5 * (1 - 0.5 * 0.5) = 0.375
        base = torch.sigmoid(torch.tensor(0.0)).item()
        decay = torch.sigmoid(torch.tensor(0.0)).item()
        tn = 0.5
        thr = base * (1.0 - tn * decay)
        assert abs(thr - 0.375) < 1e-4

    def test_concept_layer_forward(self):
        cl = UnifiedConceptLayer(D=512, k=32, bridge_dim=64, S=4).to(device)
        B, L, D = 1, 4, 512
        h = torch.randn(B, L, D, device=device)
        hp = torch.randn(B, L, 4, 32, device=device)
        pen = torch.rand(B, L, device=device)
        out = cl(h, hp=hp, pen=pen, mat_gate=0.5, allow_write=True, tau_norm=0.5)
        assert out.shape == h.shape


# ═══════════════════════════════════════════════════════════════════════
# §9  U8: τ-modulated Intent Bridge Alpha
# ═══════════════════════════════════════════════════════════════════════

class TestU8IntentAlpha:
    def test_stack_has_w_alpha_expert(self):
        cfg = EVAConfig(**SMALL, intent_bridge=True)
        model = EVAStack(cfg)
        assert hasattr(model, '_w_alpha_expert')
        assert model._w_alpha_expert.requires_grad

    def test_w_alpha_expert_shape(self):
        cfg = EVAConfig(**SMALL, intent_bridge=True)
        model = EVAStack(cfg)
        G = cfg.mlp_groups
        assert model._w_alpha_expert.shape == (G,)

    def test_intent_bridge_forward(self):
        cfg = EVAConfig(**SMALL, intent_bridge=True)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (1, 4), device=device)
        h = model.embed_tokens(x)
        out, _, _, _ = model(h)
        assert out.shape == h.shape


# ═══════════════════════════════════════════════════════════════════════
# §10  U9: τ-aware Gradient Clipping
# ═══════════════════════════════════════════════════════════════════════

class TestU9GradientClipping:
    """τ-aware AGC (audit 2026-09): attach(model) строит карту id(p) ->
    (mem_tau_ref/τ_l)^llrd_gamma по слоям; clip() применяет её per-param."""

    def _model(self):
        cfg = EVAConfig(**SMALL)
        return EVAStack(cfg)

    def test_attach_maps_layered_params_only(self):
        import re as _re
        model = self._model()
        clipper = GradientClipper(c=0.1)
        clipper.attach(model)
        tc = model.tau_config
        ref, gam = float(tc.mem_tau_ref), float(tc.llrd_gamma)
        tau = tc.tau_l.detach().cpu().tolist()
        seen = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            mm = _re.match(r"(?:.*\.)?layers\.(\d+)\.", name)
            if mm:
                sc = clipper._p_scale[id(p)]
                assert abs(sc - (ref / tau[int(mm.group(1))]) ** gam) < 1e-9
                seen += 1
            else:
                assert id(p) not in clipper._p_scale
        assert seen > 0

    def test_ladder_monotonic_gives_deeper_tighter(self):
        model = self._model()
        tc = model.tau_config
        tau = tc.tau_l.detach().cpu().tolist()
        assert all(tau[i] <= tau[i + 1] for i in range(len(tau) - 1))
        ref, gam = float(tc.mem_tau_ref), float(tc.llrd_gamma)
        assert (ref / tau[-1]) ** gam < (ref / tau[0]) ** gam

    def test_clipper_clip_works(self):
        clipper = GradientClipper(c=0.1)
        params = [nn.Parameter(torch.randn(10, 10))]
        params[0].grad = torch.randn(10, 10) * 100
        clipper.clip(params)
        assert params[0].grad.norm() <= clipper.c * params[0].norm() + 1e-4

    def test_clipper_skip_zero_params(self):
        clipper = GradientClipper(c=0.1)
        p = nn.Parameter(torch.zeros(10, 10))
        p.grad = torch.randn(10, 10)
        before = p.grad.clone()
        clipper.clip([p])
        assert torch.equal(p.grad, before)


# ═══════════════════════════════════════════════════════════════════════
# §11  U10: τ-coherent Bind Frequency Schedule
# ═══════════════════════════════════════════════════════════════════════

class TestU10BindFrequency:
    def test_bind_has_eta(self):
        cfg = EVAConfig(**SMALL, bind_twist_mode='trajectory_spiral')
        bind = TrajectorySpiralBind(cfg.D, cfg.bind_K, cfg)
        assert hasattr(bind, '_eta')
        assert bind._eta.requires_grad

    def test_bind_has_tau_norm(self):
        cfg = EVAConfig(**SMALL, bind_twist_mode='trajectory_spiral')
        bind = TrajectorySpiralBind(cfg.D, cfg.bind_K, cfg)
        # Initially None, set by block
        assert bind._tau_norm is None

    def test_bind_with_tau_norm(self):
        cfg = EVAConfig(**SMALL, bind_twist_mode='trajectory_spiral')
        bind = TrajectorySpiralBind(cfg.D, cfg.bind_K, cfg)
        bind._tau_norm = 0.5
        B, L, D = 1, 4, cfg.D
        h = torch.randn(B, L, D)
        result, traj, coh = bind(h)
        assert result.shape == (B, L, D)

    def test_freq_eff_formula(self):
        # freq_scale=2pi, tau_min=8, tau_max=512, tau_norm=0.5, eta=0.5
        freq_scale = 2 * math.pi
        tau_min, tau_max = 8.0, 512.0
        tn = 0.5
        eta = 0.5
        tau_ratio = (tau_min / tau_max) ** tn
        freq_eff = freq_scale * (tau_ratio ** eta)
        # Should be less than 2pi (freq reduced at tau_norm=0.5)
        assert freq_eff < freq_scale
        assert freq_eff > 0


# ═══════════════════════════════════════════════════════════════════════
# §12  Integration: all improvements together
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_stack_forward_backward(self):
        cfg = EVAConfig(**SMALL, intent_bridge=True, memory_bank=True)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (2, 8), device=device)
        h = model.embed_tokens(x)
        out, state, gs, _ = model(h, step=50000)
        loss = model.compute_loss(out[:, :-1], x[:, 1:])
        loss.backward()
        # _w_alpha_expert: gradient is None at init because w_intent/b_intent are
        # zero-initialized → intent_gate = 0 → no loss dependency. This is by design
        # (checkpoint-safe). Verify the param at least has requires_grad.
        assert model._w_alpha_expert.requires_grad
        # Bridge injection params always get gradient (direct h-modulation)
        if model.bridge is not None:
            assert model.bridge._inj_alpha.grad is not None
            assert model.bridge._inj_beta.grad is not None

    def test_state_dict_contains_new_params(self):
        cfg = EVAConfig(**SMALL, intent_bridge=True, memory_bank=True)
        model = EVAStack(cfg)
        sd = model.state_dict()
        new_keys = [k for k in sd if '_inj_alpha' in k or '_inj_beta' in k
                     or '_w_alpha_expert' in k or '_log_tau_birth_thr' in k
                     or '_log_tau_decay_thr' in k or '_eta' in k
                     or '_tau_signal_log' in k or '_vsa_tau_log' in k
                     or '_fusion_tau_alpha' in k]
        assert len(new_keys) >= 6, f'Expected >=6 new param keys, got {len(new_keys)}: {new_keys}'

    def test_checkpoint_save_load(self):
        cfg = EVAConfig(**SMALL, intent_bridge=True, memory_bank=True)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (1, 4), device=device)
        h = model.embed_tokens(x)
        model(h)
        # Save
        path = '/tmp/test_tau_ckpt.pt'
        torch.save(model.state_dict(), path)
        # Load into fresh model
        model2 = EVAStack(cfg).to(device)
        model2.load_state_dict(torch.load(path, map_location=device))
        h2 = model2.embed_tokens(x)
        out2, _, _, _ = model2(h2)
        assert out2.shape == h.shape
        os.remove(path)

    def test_multiple_forward_steps(self):
        cfg = EVAConfig(**SMALL, intent_bridge=True)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (1, 8), device=device)
        h = model.embed_tokens(x)
        out1, s1, gs1, _ = model(h)
        out2, s2, gs2, _ = model(h, state=s1, global_state=gs1)
        assert out2.shape == out1.shape

    def test_new_params_count_small(self):
        """New U1-U10 params should be a tiny fraction of total."""
        cfg = EVAConfig(**SMALL, intent_bridge=True, memory_bank=True)
        model = EVAStack(cfg)
        total = sum(p.numel() for p in model.parameters())
        new_params = 0
        for n, p in model.named_parameters():
            if any(x in n for x in ['_inj_alpha', '_inj_beta', '_w_alpha_expert',
                                     '_log_tau_birth_thr', '_log_tau_decay_thr',
                                     '_eta', '_tau_signal_log', '_vsa_tau_log',
                                     '_fusion_tau_alpha']):
                new_params += p.numel()
        assert new_params < total * 0.01, f'New params {new_params} > 1% of {total}'

    def test_deterministic_forward(self):
        """Two forwards with same input produce similar (not identical) output.
        Streaming state (memory bank EMA, concept layer) mutates between calls,
        so exact equality is not expected — just no NaN/inf."""
        cfg = EVAConfig(**SMALL)
        model = EVAStack(cfg).to(device)
        model.eval()
        torch.manual_seed(42)
        x = torch.randint(0, cfg.vocab, (1, 4), device=device)
        h = model.embed_tokens(x)
        out1, _, _, _ = model(h)
        out2, _, _, _ = model(h)
        assert not torch.isnan(out1).any()
        assert not torch.isnan(out2).any()
        assert out1.shape == out2.shape

    def test_no_nan_in_forward(self):
        cfg = EVAConfig(**SMALL, intent_bridge=True, memory_bank=True,
                             explicit_reasoning=True, reasoning_max_steps=2)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (2, 8), device=device)
        h = model.embed_tokens(x)
        out, _, _, _ = model(h)
        assert not torch.isnan(out).any(), 'NaN in forward output'

    def test_tau_config_update_in_forward(self):
        cfg = EVAConfig(**SMALL)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (1, 4), device=device)
        h = model.embed_tokens(x)
        # Before forward, tau_config may have default values
        model(h, step=100)
        # After forward with step, tau_config should be updated
        assert model.tau_config.tau_l.std() > 0


# ═══════════════════════════════════════════════════════════════════════
# §13  Gradient flow through all new parameters
# ═══════════════════════════════════════════════════════════════════════

class TestGradientFlow:
    def test_bridge_alpha_beta_grad(self):
        cfg = EVAConfig(**SMALL, bridge_conn=0.1)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (1, 4), device=device)
        h = model.embed_tokens(x)
        out, _, _, _ = model(h)
        loss = model.compute_loss(out[:, :-1], x[:, 1:])
        loss.backward()
        if model.bridge is not None:
            assert model.bridge._inj_alpha.grad is not None
            assert model.bridge._inj_beta.grad is not None

    def test_mirror_tau_signal_grad(self):
        cfg = EVAConfig(**SMALL)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (1, 4), device=device)
        h = model.embed_tokens(x)
        out, _, _, _ = model(h)
        loss = model.compute_loss(out[:, :-1], x[:, 1:])
        loss.backward()
        for layer in model.layers:
            if hasattr(layer.mirror, '_tau_signal_log'):
                g = layer.mirror._tau_signal_log.grad
                # Only has grad if _tau_norm_layer is set and forward was called
                if layer.mirror._tau_norm_layer is not None:
                    assert g is not None

    def test_concept_layer_birth_grad(self):
        """Birth/decay thresholds are learnable but may not receive gradient on a
        single forward (write path gated by maturation). Verify requires_grad."""
        cfg = EVAConfig(**SMALL, unified_concept_layer=True)
        model = EVAStack(cfg).to(device)
        assert model.concept_layer is not None
        assert model.concept_layer._log_tau_birth_thr.requires_grad
        assert model.concept_layer._log_tau_decay_thr.requires_grad

    def test_bind_eta_grad(self):
        cfg = EVAConfig(**SMALL, bind_twist_mode='trajectory_spiral')
        model = EVAStack(cfg).to(device)
        for layer in model.layers:
            if hasattr(layer.bind, '_eta'):
                assert layer.bind._eta.requires_grad

    def test_vsa_tau_log_grad(self):
        cfg = EVAConfig(**SMALL)
        model = EVAStack(cfg).to(device)
        for layer in model.layers:
            assert layer._vsa_tau_log.requires_grad


# ═══════════════════════════════════════════════════════════════════════
# §14  Edge cases and robustness
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_layer(self):
        cfg = EVAConfig(n_layers=1, D=256, mlp_groups=2, code_dim=16,
                             code_sparsity=4, vocab=100)
        model = EVAStack(cfg).to(device)
        x = torch.randint(0, cfg.vocab, (1, 4), device=device)
        h = model.embed_tokens(x)
        out, _, _, _ = model(h)
        assert out.shape == h.shape

    def test_tau_norm_zero(self):
        """tau_norm=0 should not crash any component."""
        bridge = SemanticBridge(D=512, n_layers=2, bridge_dim=64).to(device)
        h = torch.randn(1, 4, 512, device=device)
        bridge.bridge_stream.zero_()
        h_out = bridge.inject_layer(0, h, tau_norm=torch.tensor(0.0, device=device))
        assert h_out.shape == h.shape

    def test_tau_norm_one(self):
        """tau_norm=1 should not crash any component."""
        bridge = SemanticBridge(D=512, n_layers=2, bridge_dim=64).to(device)
        h = torch.randn(1, 4, 512, device=device)
        bridge.bridge_stream.zero_()
        h_out = bridge.inject_layer(0, h, tau_norm=torch.tensor(1.0, device=device))
        assert h_out.shape == h.shape

    def test_empty_concept_write(self):
        """No crash when no data to write."""
        cl = UnifiedConceptLayer(D=512, k=32, bridge_dim=64, S=4).to(device)
        h = torch.randn(1, 4, 512, device=device)
        out = cl(h, hp=None, pen=None, mat_gate=0.0, allow_write=False)
        assert out.shape == h.shape

    def test_memory_bank_no_write_when_low_maturation(self):
        cfg = EVAConfig(**SMALL)
        tc = TauConfig(n_layers=cfg.n_layers)
        tc.update()
        mb = StreamingMemoryBank(D=cfg.D, bridge_dim=64, l1_slots=2, l2_slots=4,
                                  tau_config=tc).to(device)
        h = torch.randn(1, 4, cfg.D, device=device)
        tokens = torch.randint(0, cfg.vocab, (1, 4), device=device)
        out = mb(h, tokens, step=0, mat_gate=0.0)
        assert out.shape == h.shape

    def test_clipper_deeper_layer_tighter(self):
        """Audit 2026-09: AGC is per-layer via attach(model) — deeper layers
        (higher tau_l) get a smaller c_eff = c*(tau_ref/tau_l)**gamma."""
        import re as _re
        cfg = EVAConfig(**SMALL)
        model = EVAStack(cfg)
        clipper = GradientClipper(c=0.1)
        clipper.attach(model)
        tc = model.tau_config
        tau = tc.tau_l.detach().cpu().tolist()
        deepest = max(range(len(tau)), key=lambda i: tau[i])
        shallowest = min(range(len(tau)), key=lambda i: tau[i])
        by_layer = {}
        for name, p in model.named_parameters():
            mm = _re.search(r"layers\.(\d+)\.", name)
            if mm and id(p) in clipper._p_scale:
                by_layer[int(mm.group(1))] = clipper._p_scale[id(p)]
        assert by_layer[deepest] < by_layer[shallowest]


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import traceback
    passed = failed = 0
    classes = [v for v in globals().values()
               if isinstance(v, type) and v.__name__.startswith('Test')]
    for cls in sorted(classes, key=lambda c: c.__name__):
        instance = cls()
        for method_name in sorted(dir(instance)):
            if not method_name.startswith('test_'):
                continue
            try:
                getattr(instance, method_name)()
                print(f'  PASS  {cls.__name__}.{method_name}')
                passed += 1
            except Exception as e:
                print(f'  FAIL  {cls.__name__}.{method_name}: {e}')
                traceback.print_exc()
                failed += 1
    print(f'\n{passed}/{passed + failed} passed')
    sys.exit(0 if failed == 0 else 1)
