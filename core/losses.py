from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_losses(stack, h, targets, pred_weight=None, h_emb=None):
    """Compute CE and auxiliary losses separately. Returns raw (unweighted) values.

    h_emb: (optional) эмбеддинг-вход для кодечной головы (двухконечное чтение).

    Returns:
        ce_loss: scalar, cross-entropy loss
        aux_dict: dict of named auxiliary losses (raw, unweighted).
    """
    if hasattr(stack.lm_head, 'log_probs_for_target'):
        B, L, D = h.shape
        bus_bias = None
        if stack.intent_bridge and getattr(stack, 'bus_head_proj', None) is not None \
                and stack._last_bus is not None:
            _bus = stack._last_bus.expand(B, L, -1, -1).reshape(B, L, -1)
            bus_bias = stack.bus_head_proj(_bus)            # (B, L, K_head)
            bus_bias = bus_bias.reshape(B * L, 1, -1)      # (N,1,K) align h(-1,D)
        log_probs = stack.lm_head.log_probs_for_target(
            h.reshape(-1, D), targets.reshape(-1), bus_bias=bus_bias)
        # Audit M1 (A4/A5): the coded-head branch ignored PAD/EOS masking and
        # surprisal weighting entirely — token 0 (PAD) was trained as a real
        # prediction. Same contract as the legacy branch below now applies to
        # BOTH paths. mask_eos getattr default matches the declared field
        # (False: Colab data is *_eos.bin — sentence ends are trained).
        ce = -log_probs
        flat_t = targets.reshape(-1)
        mask = flat_t != 0
        if getattr(stack.cfg, 'mask_eos', False):
            mask = mask & (flat_t != 2)
        mask_f = mask.to(ce.dtype)
        sw = getattr(stack.cfg, 'surprisal_weight', 0.0)
        if stack.training and sw > 0:
            with torch.no_grad():
                ce_ratio = ce / (ce.sum() / mask_f.sum().clamp(min=1) + 1e-8)
                w = torch.sigmoid(sw * 2.0 * (ce_ratio - 1.0))
            ce_loss = (ce * w * mask_f).sum() / mask_f.sum().clamp(min=1)
        else:
            ce_loss = (ce * mask_f).sum() / mask_f.sum().clamp(min=1)
    else:
        logits = stack.lm_head(h)
        ce = F.cross_entropy(logits.reshape(-1, stack.cfg.vocab),
                             targets.reshape(-1), reduction='none')
        mask = targets.reshape(-1) != 0
        if getattr(stack.cfg, 'mask_eos', False):  # default = declared field (False: EOS trained)
            mask = mask & (targets.reshape(-1) != 2)
        ce = ce * mask.float()
        sw = getattr(stack.cfg, 'surprisal_weight', 0.0)
        if stack.training and sw > 0:
            with torch.no_grad():
                ce_ratio = ce / (ce.mean() + 1e-8)
                w = torch.sigmoid(2.0 * (ce_ratio - 1.0))
            ce_loss = (ce * w).sum() / mask.sum().clamp(min=1)
        else:
            ce_loss = ce.sum() / mask.sum().clamp(min=1)
    pred_loss = 0.0
    n_pred = 0
    # Live per-layer self-prediction scalars (audit M5: the old _pred_cache
    # stored pre-detached tensors → 'pred' had zero gradient by construction).
    for layer in stack.layers:
        _pt = getattr(layer.mirror, '_pred_loss_term', None)
        if _pt is not None:
            pred_loss = pred_loss + _pt
            n_pred = n_pred + 1
    if n_pred > 0:
        pred_loss = pred_loss / n_pred
    
    gate_l1 = 0.0
    n_gates = 0
    for layer in stack.layers:
        g = getattr(layer.mirror, '_cached_gate_l1', None)
        if g is not None:
            gate_l1 = gate_l1 + g
            n_gates = n_gates + 1
    if n_gates > 0:
        gate_l1 = gate_l1 / n_gates
    
    reinforce_loss = 0.0
    n_reinf = 0
    for layer in stack.layers:
        u = getattr(layer.mirror, '_cached_usefulness', None)
        g = getattr(layer.mirror, '_cached_gate', None)
        if u is not None and g is not None:
            reinforce_loss = reinforce_loss + F.mse_loss(u, g)
            n_reinf = n_reinf + 1
    if n_reinf > 0:
        reinforce_loss = reinforce_loss / n_reinf
    
    balance_loss = 0.0
    n_bal = 0
    for layer in stack.layers:
        usage = getattr(layer.mirror, '_cached_gate_usage', None)
        if usage is not None:
            usage_p = usage / (usage.sum() + 1e-10)
            hhi = (usage_p ** 2).sum()
            norm = (hhi - 1.0 / usage.shape[-1]) / (1.0 - 1.0 / usage.shape[-1])
            balance_loss = balance_loss + norm.clamp(min=0)
            n_bal = n_bal + 1
    if n_bal > 0:
        balance_loss = balance_loss / n_bal
    
    diversity_loss = 0.0
    n_div = 0
    for i, layer in enumerate(stack.layers):
        group_out = getattr(layer.mlp, '_cached_group_out', None)
        if group_out is not None:
            B, L, G, d = group_out.shape
            y = group_out.norm(dim=-1).reshape(-1, G)
            # Scale-invariant: correlation matrix (column-standardized) → bounded
            # regardless of ‖y‖; raw covariance scaled as ‖y‖⁴ and exploded in A2.
            y = (y - y.mean(dim=0)) / (y.std(dim=0) + 1e-8)
            corr = y.T @ y / (y.shape[0] - 1 + 1e-10)
            div = F.mse_loss(corr, torch.eye(G, device=group_out.device))
            # τ-tied per-layer weight: intent_alpha = 1 − exp(−τ_l/τ_min)
            # (τ-field expresses exploration authority; deep layers explore more).
            alpha = 1.0 - torch.exp(-stack.tau_config.tau_l[i].detach() / stack.tau_config.tau_min)
            diversity_loss = diversity_loss + alpha * div
            n_div = n_div + 1
    if n_div > 0:
        diversity_loss = diversity_loss / n_div
    
    nuc_loss = 0.0
    n_nuc = 0
    for layer in stack.layers:
        bind_W = None
        if hasattr(layer, 'bind') and hasattr(layer.bind, 'W_proj'):
            bind_W = layer.bind.W_proj.weight
        if bind_W is not None and bind_W.ndim == 2:
            rank_ub = min(bind_W.shape[0], bind_W.shape[1])
            nuc_iters = max(1, int(math.sqrt(rank_ub)))
            v = torch.randn(bind_W.shape[1], nuc_iters, device=bind_W.device)
            Wv = bind_W @ v
            nuc = Wv.norm(dim=0).mean() * math.sqrt(bind_W.shape[1])
            nuc_loss = nuc_loss + nuc
            n_nuc = n_nuc + 1
    if n_nuc > 0:
        nuc_loss = nuc_loss / n_nuc
    
    orth_loss = 0.0
    n_orth = 0
    if getattr(stack.cfg, 'orth_weight', 1e-4) > 0:
        for layer in stack.layers:
            bind_W = None
            if hasattr(layer, 'bind') and hasattr(layer.bind, 'W_proj'):
                bind_W = layer.bind.W_proj.weight
            if bind_W is not None and bind_W.ndim == 2:
                W_hat = bind_W / bind_W.norm(dim=0, keepdim=True).clamp(min=1e-8)
                gram = W_hat.T @ W_hat
                orth = F.mse_loss(gram, torch.eye(gram.shape[0], device=gram.device))
                orth_loss = orth_loss + orth
                n_orth = n_orth + 1
    if n_orth > 0:
        orth_loss = orth_loss / n_orth
    
    w_m2v_loss = 0.0
    n_m2v = 0
    if getattr(stack.cfg, 'w_m2v_hierarchy_weight', 0.0) > 0:
        for i, layer in enumerate(stack.layers):
            wm = getattr(layer, 'w_mem2v', None)
            if wm is not None:
                tau_l_t = stack.tau_config.tau_l[i].detach()
                tau_mid_t = (stack.tau_config.tau_l[0].detach() * stack.tau_config.tau_l[-1].detach()).sqrt()
                target = getattr(stack.cfg, 'w_m2v_hierarchy_target', 1.0)
                target_m2v = target / (1.0 + torch.exp(-(tau_l_t.log() - tau_mid_t.log())))
                # audit M5: the parameter side was .detach()ed too → the loss
                # could never move w_mem2v (dead decoration). Target stays
                # detached (no co-adaptation), parameter side is live.
                w_m2v_loss = w_m2v_loss + (wm.mean() - target_m2v).pow(2)
                n_m2v = n_m2v + 1
        if n_m2v > 0:
            w_m2v_loss = w_m2v_loss / n_m2v
    # Intent Bridge: τ-ladder regularization (targets DETACHED to prevent co-adaptation)
    intent_tau_loss = 0.0
    n_it = 0
    if stack.intent_bridge and getattr(stack.cfg, 'intent_tau_hierarchy_weight', 0.0) > 0:
        tau_mid_t = (stack.tau_config.tau_l[0].detach() * stack.tau_config.tau_l[-1].detach()).sqrt()
        c_ema_t = (1.0 / math.sqrt(stack.cfg.D)) * tau_mid_t
        for i in range(len(stack.layers)):
            # audit M5: BOTH sides were detached → the regularizer could not
            # shape the τ-ladder it claims to regulate (dead decoration).
            # Actual side is live (gradient flows to tau_config._tau_dev);
            # the target keeps detached τ (no co-adaptation, per comment).
            tau_intent_l = stack.tau_config.tau_l[i]
            actual_alpha = torch.clamp(1.0 - c_ema_t / tau_intent_l, min=0.0)
            tgt = getattr(stack.cfg, 'intent_tau_hierarchy_target', 0.3)
            target_alpha = tgt / (1.0 + torch.exp(-(tau_intent_l.detach().log() - tau_mid_t.log())))
            intent_tau_loss = intent_tau_loss + (actual_alpha - target_alpha).pow(2)
            n_it += 1
        if n_it > 0:
            intent_tau_loss = intent_tau_loss / n_it
    branch_loss = 0.0
    n_branch = 0
    if getattr(stack.cfg, 'branch_balance_weight', 0.0) > 0:
        for layer in stack.layers:
            conv = getattr(layer, '_cache_conv_out', None)
            bnd = getattr(layer, '_cache_bind_out', None)
            mir = getattr(layer, '_cache_mirror_out', None)
            if conv is not None and bnd is not None and mir is not None:
                vc = conv.norm(dim=-1).var() + 1e-10
                vb = bnd.norm(dim=-1).var() + 1e-10
                vm = mir.norm(dim=-1).var() + 1e-10
                branch_loss = branch_loss + (torch.log(vc) - torch.log(vb)).pow(2)
                branch_loss = branch_loss + (torch.log(vc) - torch.log(vm)).pow(2)
                branch_loss = branch_loss + (torch.log(vb) - torch.log(vm)).pow(2)
                n_branch = n_branch + 3
        if n_branch > 0:
            branch_loss = branch_loss / n_branch
    
    signal_entropy = 0.0
    n_sig = 0
    for layer in stack.layers:
        w = torch.sigmoid(layer.mirror._signal_log_weights)
        p = w / (w.sum() + 1e-10)  # normalize for entropy
        # MINIMIZE −H ⇒ MAXIMIZE signal entropy (all mirror signals stay
        # in play). The old sign (+H) actively pushed the 5 learnable signal
        # weights toward one-hot collapse — opposite to gate_repulse/branch,
        # which use the −entropy convention (audit M5).
        signal_entropy = signal_entropy + (p * torch.log(p + 1e-10)).sum()
        n_sig = n_sig + 1
    if n_sig > 0:
        signal_entropy = signal_entropy / n_sig

    # ─── gradalign: gradient-reactive governance loss (AGENT_BRIEF §4) ───
    # Target: per-expert ‖∂CE/∂mlp_out‖ captured by a backward hook in the
    # block (free — audit M5: the loops recomputed it with an extra full
    # autograd.grad per step). Model: the per-expert MLP gate mean, mapped
    # through its own max so both sides are relative distributions. Weight
    # cfg.gradalign_weight is a REAL strength now (was on/off only), and the
    # term BYPASSES spectral alignment downstream (LossBalancer.BYPASS_AUX):
    # its purpose is a direct teaching signal for the gate path — zeroing it
    # when the summed aux gradient happens orthogonal to CE would delete the
    # mechanism the brief built it to fix.
    gradalign_term = 0.0
    _gaw = float(getattr(stack.cfg, 'gradalign_weight', 0.0) or 0.0)
    if _gaw > 0 and stack.training:
        _ga_sum, _ga_n = 0.0, 0
        for layer in stack.layers:
            _tgt = getattr(layer, '_gradalign_tgt', None)
            _mod = getattr(layer, '_cache_mlp_mod', None)
            if _tgt is None or _mod is None or not _mod.requires_grad:
                continue
            _gt = _tgt.float()
            _mt = _mod.float().mean(dim=(0, 1))
            _gtn = (_gt / (_gt.max() + 1e-8)).detach()
            _mtn = _mt / (_mt.max().detach() + 1e-8)
            _ga_sum = _ga_sum + (_mtn - _gtn).pow(2).mean()
            _ga_n += 1
        if _ga_n > 0:
            gradalign_term = _gaw * _ga_sum / _ga_n
    
    log_scale_reg = 0.0
    n_ls = 0
    for layer in stack.layers:
        ls = layer.mirror.log_scale
        excess = (ls - 2.3).clamp(min=0)
        log_scale_reg = log_scale_reg + excess.pow(2).mean()
        n_ls = n_ls + 1
    if n_ls > 0:
        log_scale_reg = log_scale_reg / n_ls
    
    # Diversity: per-layer log_scale variance (inter-expert + intra-expert)
    div_loss_raw = 0.0
    div_w = getattr(stack.cfg, 'div_weight', 0.0)
    if div_w > 0:
        for layer in stack.layers:
            ls = layer.mirror.log_scale
            d = ls.shape[-1]
            G = ls.shape[0]
            intra_weight = math.sqrt(d / G)
            div_loss_raw = div_loss_raw - (ls.sigmoid().var(dim=0).mean() + intra_weight * ls.sigmoid().var(dim=-1).mean())
        div_loss_raw = div_loss_raw / max(len(stack.layers), 1)

    # Gate repulsion: push gate variance up (inverse of balance)
    gate_repulse_loss = 0.0
    gate_rp_w = getattr(stack.cfg, 'gate_repulse_weight', 0.0)
    if gate_rp_w > 0:
        n_rp = 0
        for layer in stack.layers:
            gate_usage = getattr(layer.mirror, '_cached_gate_usage', None)
            if gate_usage is not None:
                # P2 FIX: use negative entropy instead of -var
                # -var pushes all experts to same activation (could be all-zero).
                # -entropy pushes toward uniform distribution over experts.
                gate_p = F.softmax(gate_usage, dim=0)
                gate_entropy = -(gate_p * (gate_p + 1e-8).log()).sum()
                gate_repulse_loss = gate_repulse_loss - gate_entropy
                n_rp += 1
        if n_rp > 0:
            gate_repulse_loss = gate_repulse_loss / n_rp

    # Alpha novelty: push per-expert alpha apart
    alpha_novelty_loss = 0.0
    alpha_nv_w = getattr(stack.cfg, 'alpha_novelty_weight', 0.0)
    if alpha_nv_w > 0:
        n_nv = 0
        for layer in stack.layers:
            ad = layer.mirror.alpha_diag
            if ad is not None:
                alpha_novelty_loss = alpha_novelty_loss - ad.mean(dim=-1).var()
                n_nv += 1
        if n_nv > 0:
            alpha_novelty_loss = alpha_novelty_loss / n_nv

    decorr_loss = 0.0
    n_decorr = 0
    for layer in stack.layers:
        d = getattr(layer.mirror, '_cached_decorr', None)
        if d is not None:
            decorr_loss = decorr_loss + d
            n_decorr = n_decorr + 1
    if n_decorr > 0:
        decorr_loss = decorr_loss / n_decorr
    
    stack._cached_losses = {
        'ce': ce_loss.item(),
        'pred': pred_loss.item() if isinstance(pred_loss, torch.Tensor) else pred_loss,
        'gate_l1': gate_l1.item() if isinstance(gate_l1, torch.Tensor) else gate_l1,
        'reinforce': reinforce_loss.item() if isinstance(reinforce_loss, torch.Tensor) else reinforce_loss,
        'balance': balance_loss.item() if isinstance(balance_loss, torch.Tensor) else balance_loss,
        'div': div_loss_raw.item() if isinstance(div_loss_raw, torch.Tensor) else div_loss_raw,
        'gate_repulse': gate_repulse_loss.item() if isinstance(gate_repulse_loss, torch.Tensor) else gate_repulse_loss,
        'alpha_novelty': alpha_novelty_loss.item() if isinstance(alpha_novelty_loss, torch.Tensor) else alpha_novelty_loss,
        'signal_ent': signal_entropy.item() if isinstance(signal_entropy, torch.Tensor) else signal_entropy,
        'ls_reg': log_scale_reg.item() if isinstance(log_scale_reg, torch.Tensor) else log_scale_reg,
        'decorr': decorr_loss.item() if isinstance(decorr_loss, torch.Tensor) else decorr_loss,
    }
    # ─── Layer Bridge Gate: log per-layer gate weights (SpectrumGate) ───
    # Also compute a differentiable diversity aux loss so log_tau gets gradient.
    _lbg_aux = {}
    _lbg_diversity_loss = 0.0
    if stack.layer_bridge_gate is not None and stack._layer_diagnostics:
        _gr = False
        if getattr(stack, 'maturation', None) is not None:
            _gr = stack.maturation.global_ready
        _gates_items = []
        _taus_items = []
        _gate_outputs = []  # differentiable — diversity loss needs grad through LBG params
        for l in range(len(stack.layers)):
            if l in stack._layer_diagnostics:
                _mat = stack.maturation.gate[l] if getattr(stack, 'maturation', None) is not None else torch.ones(1)
                if _gr:
                    _mat_tau = stack.layer_bridge_gate._effective_tau(_mat)
                    # Clone diagnostics with grad so gate output carries grad through log_tau
                    _diag_grad = stack._layer_diagnostics[l].clone().detach().requires_grad_(True)
                    _gated = stack.layer_bridge_gate.gates[l](_diag_grad, tau_external=_mat_tau)
                    _gate_outputs.append(_gated.mean())
                    _gates_items.append(_gated.mean().item())
                    _taus_items.append(_mat_tau.item())
                else:
                    _gates_items.append(_mat.item())
                    _taus_items.append(stack.layer_bridge_gate.tau_max)
            else:
                _gates_items.append(0.5)
                _taus_items.append(1.0)
        with torch.no_grad():
            _gates_t = torch.tensor(_gates_items)
            _gates_std = (_gates_t.std().item() if _gates_t.numel() > 1 else 0.0)
            stack._cached_losses['lbg_mean'] = _gates_t.mean().item()
            stack._cached_losses['lbg_std'] = _gates_std
            stack._cached_losses['lbg_min'] = _gates_t.min().item()
            stack._cached_losses['lbg_max'] = _gates_t.max().item()
            stack._cached_losses['lbg_tau'] = sum(_taus_items) / len(_taus_items) if _taus_items else 0.0
            stack._cached_losses['lbg_global_ready'] = 1.0 if _gr else 0.0
            _lbg_aux = {
                'layer_gate_mean': _gates_t.mean().item(),
                'layer_gate_std': _gates_std,
                'layer_gate_min': _gates_t.min().item(),
                'layer_gate_max': _gates_t.max().item(),
                'lbg_global_ready': 1.0 if _gr else 0.0,
            }
        # Diversity loss: encourage gate weights to spread across layers
        # (negative entropy = collapse to one layer = bad)
        if _gate_outputs and _gr:
            _gate_stack = torch.stack(_gate_outputs)
            _gate_p = torch.softmax(_gate_stack, dim=0)
            _gate_entropy = -(_gate_p * (_gate_p + 1e-8).log()).sum()
            _max_ent = math.log(len(_gate_outputs))
            _lbg_diversity_loss = (_max_ent - _gate_entropy).clamp(min=0) / _max_ent
        stack._layer_diagnostics = {}  # reset for next step
    # ─── Memory Bank diagnostics (consolidation stats) ───
    if stack.memory_bank is not None:
        try:
            _mbd = stack.memory_bank.get_diagnostics()
            stack._cached_losses['mb_l1_overwrites'] = _mbd['l1_overwrites']
            stack._cached_losses['mb_l2_overwrites'] = _mbd['l2_overwrites']
            stack._cached_losses['mb_l2_consumed'] = _mbd['l2_consumed']
            # decision #2: concept counters now come from the SINGLE concept
            # store (UnifiedConceptLayer) — the retired bank-L3 keys keep their
            # log labels so analyze.py dashboards stay readable.
            if getattr(stack, 'concept_layer', None) is not None:
                _ucld = stack.concept_layer.get_diagnostics()
                stack._cached_losses['mb_l3_births'] = _ucld['concept_n_births']
                stack._cached_losses['mb_l3_updates'] = _ucld['concept_n_updates']
            stack._cached_losses['mb_scale'] = _mbd['mem_scale']
        except Exception:
            pass
    pred_w_loss = 0.0
    n_pred_w = 0
    head = getattr(stack, 'lm_head', None)
    if head is not None and hasattr(head, 'pred_w'):
        pw = head.pred_w
        if pw.ndim == 2:
            pred_w_loss = F.mse_loss(pw, torch.eye(pw.shape[0], device=pw.device))
            n_pred_w += 1

    # Raw auxiliary losses — NO per-loss magic weights.  All weighting is
    # done principledly by the training LossBalancer (core.adaptation),
    # either via spectral gradient alignment (default) or magnitude
    # balancing.  Returning raw values also removes the previous
    # double-weighting bug (weights were baked here AND reapplied in the
    # training loop).
    aux_dict = {}
    aux_dict.update(_lbg_aux)
    if pred_w_loss != 0:
        aux_dict['pred_w'] = pred_w_loss
    if pred_loss != 0:
        aux_dict['pred'] = pred_loss
    if gate_l1 != 0:
        aux_dict['gate_l1'] = gate_l1
    if reinforce_loss != 0:
        aux_dict['reinforce'] = reinforce_loss
    if balance_loss != 0:
        aux_dict['balance'] = balance_loss
    if diversity_loss != 0:
        aux_dict['diversity'] = diversity_loss
    if nuc_loss != 0:
        aux_dict['nuc'] = nuc_loss * getattr(stack.cfg, 'nuclear_weight', 1e-5)
    if orth_loss != 0:
        aux_dict['orth'] = orth_loss
    if w_m2v_loss != 0:
        aux_dict['w_m2v'] = w_m2v_loss
    if intent_tau_loss != 0:
        aux_dict['intent_tau'] = intent_tau_loss
    if branch_loss != 0:
        aux_dict['branch'] = branch_loss
    if div_loss_raw != 0:
        aux_dict['div'] = div_loss_raw
    if gate_repulse_loss != 0:
        aux_dict['gate_repulse'] = gate_repulse_loss
    if alpha_novelty_loss != 0:
        aux_dict['alpha_novelty'] = alpha_novelty_loss
    if n_decorr > 0:
        aux_dict['decorr'] = decorr_loss
    if n_sig > 0:
        aux_dict['signal_ent'] = signal_entropy
    if isinstance(gradalign_term, torch.Tensor):
        aux_dict['gradalign'] = gradalign_term
    if log_scale_reg != 0:
        aux_dict['ls_reg'] = log_scale_reg
    # ─── Memory bank log_tau: regularize toward prior + enforce L1 < L2 ───
    # (decision #2: the L3 tier of the hierarchy is the UCL, which owns its
    #  own τ-knobs; the bank enforces only its remaining two levels)
    mem_tau_reg = 0.0
    if stack.memory_bank is not None:
        for level_name, mem_level in [('l1', stack.memory_bank.l1),
                                       ('l2', stack.memory_bank.l2)]:
            if hasattr(mem_level, 'log_tau') and mem_level.log_tau.requires_grad:
                # Soft weight-decay toward the initialized prior
                if hasattr(mem_level, '_init_log_tau'):
                    mem_tau_reg = mem_tau_reg + (mem_level.log_tau - mem_level._init_log_tau).pow(2).mean()
                else:
                    mem_tau_reg = mem_tau_reg + mem_level.log_tau.pow(2).mean() * 0.01
        # Soft penalty: L1_tau (fast) should not exceed L2_tau (slow)
        if (hasattr(stack.memory_bank.l1, 'log_tau') and hasattr(stack.memory_bank.l2, 'log_tau')
                and stack.memory_bank.l1.log_tau.requires_grad and stack.memory_bank.l2.log_tau.requires_grad):
            inversions = F.relu(stack.memory_bank.l1.log_tau - stack.memory_bank.l2.log_tau)
            mem_tau_reg = mem_tau_reg + inversions.mean() * 0.1
    if mem_tau_reg != 0:
        aux_dict['mem_tau_reg'] = mem_tau_reg
    # ─── Semantic Bridge aux loss (per-layer next-token embedding prediction) ───
    # Each layer's probe is self-supervised to predict the next token's
    # embedding (cosine). Dense, well-distributed gradient at every depth;
    # weights/balances the rest of the aux suite via the training LossBalancer.
    if stack.bridge is not None and stack.bridge._preds is not None:
        _bl = stack.bridge.loss(targets, stack.embed)
        if _bl is not None:
            aux_dict['bridge_conn'] = _bl
    # ─── _tau_dev regularization: prevent one-sided collapse ───
    # Soft centering keeps the tau ladder near-uniform unless gradient
    # strongly prefers compression. Without this, optimizer pushes dev
    # negative (all tau shorter), collapsing the hierarchy.
    if hasattr(stack, 'tau_config') and stack.tau_config is not None:
        dev = stack.tau_config._tau_dev
        if dev.requires_grad:
            # L2 penalty toward zero (uniform ladder)
            _tau_dev_reg = dev.pow(2).mean() * 0.01
            aux_dict['tau_dev_reg'] = _tau_dev_reg
    if _lbg_diversity_loss != 0:
        aux_dict['lbg_diversity'] = _lbg_diversity_loss
    return ce_loss, aux_dict
