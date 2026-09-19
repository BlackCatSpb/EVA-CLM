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
    ce_raw = None   # M46: set on the coded path; legacy falls back to ce_loss
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
        # M46: the UNWEIGHTED per-token CE — the honest comparable metric.
        # The training objective below is surprisal-WEIGHTED (w = sigmoid(...)
        # <= 1), so the logged train CE is systematically BELOW the eval CE by
        # ~1/mean(w) (measured x2.0-2.3 at sw=0.3). Comparing the weighted
        # train number against the unweighted val number manufactured an
        # 8-nat phantom gap; both are logged now (ce_raw rides _cached_losses).
        ce_raw = (ce * mask_f).sum() / mask_f.sum().clamp(min=1)
        if stack.training and sw > 0:
            with torch.no_grad():
                ce_ratio = ce / (ce.sum() / mask_f.sum().clamp(min=1) + 1e-8)
                w = torch.sigmoid(sw * 2.0 * (ce_ratio - 1.0))
            ce_loss = (ce * w * mask_f).sum() / mask_f.sum().clamp(min=1)
        else:
            ce_loss = ce_raw
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
    
    # (M64.6 TOMBSTONE: the `nuc` term stood here — a stable-rank regularizer.
    # Removed: the M63-E audit + the round-2/3 reviewers measured the GRADIENT
    # inert (4.9e-8 of the CE gradient — the 1e-5 weight times a RADIAL
    # direction: cos(g_nuc, W) = -1.0000, the detached sigma-max made
    # d penalty/dW parallel to W, i.e. ||W|| growth, not rank recovery).
    # The penalty itself was NOT zero: on the production checkpoint the mean
    # is ~0.28 (rank_ub=min(32,2560)=32; deep layers ~0.05-0.12, early
    # 0.57-0.83) — the reviewer's 0.52 came from a rank-64 model (rank_ub=64),
    # so the number is model-dependent. `nuclear_weight` is marked REMOVED in
    # the config; a stable-rank METRIC (no loss) is queued for M64.8.)

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
    _br_conv = _br_bind = _br_mirror = None   # M64.8r3: defined even when the term is off
    n_branch = 0
    if getattr(stack.cfg, 'branch_balance_weight', 0.0) > 0:
        # M51 (the 2970 explosion post-mortem): the three terms below are
        # log-RATIOS — scale-free, so a uniform x10/layer growth of ALL branches
        # was invisible to this loss (measured: the stream hit ~1e16 while
        # 'branch' stayed finite). The anchor pins the ABSOLUTE scale: each
        # branch's log-variance is pulled to a slow EMA of itself, seeded at
        # the first observation (a healthy fresh run), so gradual drift is
        # allowed while a runaway is penalized. The reference is detached.
        _anchor_w = float(getattr(stack.cfg, 'branch_var_anchor', 0.0))
        _ref = getattr(stack, '_branch_var_ref', None)
        _seen = []
        _rms = {'conv': [], 'bind': [], 'mirror': []}   # M64.8r2 (M63-E telemetry)
        for layer in stack.layers:
            conv = getattr(layer, '_cache_conv_out', None)
            bnd = getattr(layer, '_cache_bind_out', None)
            mir = getattr(layer, '_cache_mirror_out', None)
            if conv is not None and bnd is not None and mir is not None:
                vc = conv.norm(dim=-1).var() + 1e-10
                vb = bnd.norm(dim=-1).var() + 1e-10
                vm = mir.norm(dim=-1).var() + 1e-10
                with torch.no_grad():
                    _rms['conv'].append(float(vc.sqrt()))
                    _rms['bind'].append(float(vb.sqrt()))
                    _rms['mirror'].append(float(vm.sqrt()))
                branch_loss = branch_loss + (torch.log(vc) - torch.log(vb)).pow(2)
                branch_loss = branch_loss + (torch.log(vc) - torch.log(vm)).pow(2)
                branch_loss = branch_loss + (torch.log(vb) - torch.log(vm)).pow(2)
                n_branch = n_branch + 3
                if _anchor_w > 0.0:
                    _vs = torch.stack([vc, vb, vm])
                    if _ref is None:
                        _ref = _vs.detach().mean().clone()
                    branch_loss = branch_loss + _anchor_w * (torch.log(_vs) - torch.log(_ref)).pow(2).sum()
                    n_branch = n_branch + 3
                    _seen.append(_vs.detach().mean())
        if n_branch > 0:
            branch_loss = branch_loss / n_branch
        # M64.8r2 (M63-E): the per-branch RMS telemetry — the loss balances the
        # log-VARIANCES, the RMS is what the stream actually carries. NOTE: this
        # runs BEFORE the `stack._cached_losses = {...}` reassignment below, so
        # the values ride in locals and are attached after it (the first landing
        # wrote them into the dict that was then overwritten — the round-3
        # verifier caught the loss).
        if _rms['conv']:
            _br_conv = sum(_rms['conv']) / len(_rms['conv'])
            _br_bind = sum(_rms['bind']) / len(_rms['bind'])
            _br_mirror = sum(_rms['mirror']) / len(_rms['mirror'])
        else:
            _br_conv = _br_bind = _br_mirror = None
        if _anchor_w > 0.0 and _ref is not None:
            with torch.no_grad():
                _cm = torch.stack(_seen).mean() if _seen else _ref
                stack._branch_var_ref = (_ref * 0.999 + _cm * 0.001).detach()
    
    signal_entropy = 0.0
    n_sig = 0
    for layer in stack.layers:
        # B2 (audit C): the mirror's ACTUAL gate is σ(θ/τ_signal); the old
        # entropy regularized σ(θ) — a different, un-traded quantity.
        _tsl = getattr(layer.mirror, '_tau_signal_used', None)
        _th = layer.mirror._signal_log_weights if _tsl is None else layer.mirror._signal_log_weights / _tsl
        w = torch.sigmoid(_th)
        p = w / (w.sum() + 1e-10)  # normalize for entropy
        # MINIMIZE −H ⇒ MAXIMIZE signal entropy (all mirror signals stay
        # in play). The old sign (+H) actively pushed the 5 learnable signal
        # weights toward one-hot collapse — opposite to balance/branch,
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

    # (M64.6 TOMBSTONE: the `gate_repulse` term stood here. Removed: the M63-E
    # audit measured it SATURATED at its own optimum (−3.43 ≈ −ln32, within 1%
    # of uniform usage) with a ~1e-3 gradient, and it duplicated `balance`
    # (both target uniform expert usage). `gate_repulse_weight` is REMOVED in
    # the config.)

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
        'ce_raw': float(ce_raw.detach()) if ce_raw is not None else ce_loss.item(),  # M46
        'pred': pred_loss.item() if isinstance(pred_loss, torch.Tensor) else pred_loss,
        'gate_l1': gate_l1.item() if isinstance(gate_l1, torch.Tensor) else gate_l1,
        'reinforce': reinforce_loss.item() if isinstance(reinforce_loss, torch.Tensor) else reinforce_loss,
        'balance': balance_loss.item() if isinstance(balance_loss, torch.Tensor) else balance_loss,
        'div': div_loss_raw.item() if isinstance(div_loss_raw, torch.Tensor) else div_loss_raw,
        'alpha_novelty': alpha_novelty_loss.item() if isinstance(alpha_novelty_loss, torch.Tensor) else alpha_novelty_loss,
        'signal_ent': signal_entropy.item() if isinstance(signal_entropy, torch.Tensor) else signal_entropy,
        'ls_reg': log_scale_reg.item() if isinstance(log_scale_reg, torch.Tensor) else log_scale_reg,
        'decorr': decorr_loss.item() if isinstance(decorr_loss, torch.Tensor) else decorr_loss,
    }
    if _br_conv is not None:                      # M64.8r2: per-branch RMS
        stack._cached_losses['branch_r_conv'] = _br_conv
        stack._cached_losses['branch_r_bind'] = _br_bind
        stack._cached_losses['branch_r_mirror'] = _br_mirror
    # (M64.5 TOMBSTONE: the Layer Bridge Gate block stood here — the per-layer
    # SpectrumGate telemetry (lbg_*) + the lbg_diversity aux term. Removed with
    # the dead channel: the gate was computed and discarded, the term measured
    # 2.75e-6 with its graph touching only the gate's own log_tau, and the B3
    # parity incident broke the bridge (0.26 train vs 3.69 eval < chance). The
    # log contract change: lbg_tau/lbg_diversity/layer_gate_* no longer exist.)
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
    # (M64.6 TOMBSTONE: the `pred_w` branch stood here — it tested
    # `hasattr(head, 'pred_w')` on the head, which has never had that attribute:
    # a dead branch. Removed.)

    # Raw auxiliary losses — NO per-loss magic weights.  All weighting is
    # done principledly by the training LossBalancer (core.adaptation),
    # either via spectral gradient alignment (default) or magnitude
    # balancing.  Returning raw values also removes the previous
    # double-weighting bug (weights were baked here AND reapplied in the
    # training loop).
    aux_dict = {}
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
    # (M64.6: the `nuc` emission was removed with the inert term.)
    # (B2: always emit when computed — a first-call penalty of exactly 0
    #  from the raw power-iteration estimate silently removed the key.)
    if orth_loss != 0:
        aux_dict['orth'] = orth_loss
    if w_m2v_loss != 0:
        aux_dict['w_m2v'] = w_m2v_loss
    if intent_tau_loss != 0:
        aux_dict['intent_tau'] = intent_tau_loss
    # M52a (P1): soft wall on the head's bit log-odds (the 2970 post-mortem:
    # the sigma CE-gradient is ~1e-13 at |u|>17). The wall's gradient is LINEAR
    # in the excess (2w(|u|-u0)) and stays alive at any u — the only fast escape
    # (ST clamps rescue at 1e-7..1e-4/step, i.e. ~1e6-1e11 steps).
    # T9.6-ПОПРАВКА (замер 2026-09-19): «выход» был структурно ЗАКРЫТ —
    # как aux-терм стена гейтилась CE-градиентом (sign-маска + бонд ‖g_CE‖),
    # а при насыщении g_CE≈0 ⇒ стена глушилась ровно тогда, когда нужна
    # (wall=250, u_max рос 197→592). Исправлено: head_wall в SAFETY_AUX —
    # отдельный путь без маски/бонда (LossBalancer._add_safety).
    _pl1 = float(getattr(stack.cfg, 'head_phantom_l1', 0.0))
    if _pl1 > 0.0:
        _last_p = getattr(stack.lm_head, '_last_p', None)
        if _last_p is not None:
            _ph = _pl1 * _last_p.abs().mean()
            if float(_ph.detach()) != 0.0:
                aux_dict['phantom_l1'] = _ph
    _hw = float(getattr(stack.cfg, 'head_u_wall', 0.0))
    if _hw > 0.0:
        _u_last = getattr(stack.lm_head, '_last_u', None)
        if _u_last is not None:
            _u0 = float(getattr(stack.cfg, 'head_u_wall_u0', 6.0))
            _wall = _hw * F.relu(_u_last.abs() - _u0).pow(2).mean()
            if float(_wall.detach()) != 0.0:
                aux_dict['head_wall'] = _wall
    if branch_loss != 0:
        aux_dict['branch'] = branch_loss
    if div_loss_raw != 0:
        aux_dict['div'] = div_loss_raw
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
    return ce_loss, aux_dict
