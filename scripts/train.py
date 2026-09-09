"""
EVA training: streaming from token_stream_{GENRE}.bin files.
"""

import os, sys, math, time, json, glob, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import torch
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
from torch.serialization import add_safe_globals

from core import EVAConfig, EVAStack, MirrorLRScheduler


def _save_checkpoint_safely(state, path):
    """Write checkpoint to temp file then rename — prevents corruption on interrupt."""
    import tempfile, shutil
    tmp = path + '.tmp'
    torch.save(state, tmp)
    shutil.move(tmp, path)
try:
    from analyze import save_html_report as generate_report
except Exception:
    # Report generation is optional; training must run without it.
    generate_report = lambda *a, **k: None

add_safe_globals([EVAConfig])


def _detach_state(st):
    """Recursively detach a (possibly nested) state structure of tensors."""
    if st is None:
        return None
    if isinstance(st, torch.Tensor):
        return st.detach()
    if isinstance(st, (list, tuple)):
        return type(st)(_detach_state(x) for x in st)
    return st


class TokenStream:
    """Memory-mapped uint16 token stream; converted to torch.long per batch."""
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.len = len(self.data)
    def get_batch(self, seq_len, batch_size, offset, vocab=50000):
        needed = batch_size * seq_len + 1
        wrapped = offset + needed > self.len
        if wrapped:
            offset = 0
        chunk = self.data[offset:offset + needed]
        if vocab is not None:
            # uint16-файлы могут содержать токены ≥ vocab → device-side assert
            # в codes[tokens] (index out of bounds); клипим до безопасности.
            chunk = np.clip(chunk, 0, vocab - 1)
        x = torch.from_numpy(chunk[:batch_size * seq_len].reshape(batch_size, seq_len).copy())
        y = torch.from_numpy(chunk[1:batch_size * seq_len + 1].reshape(batch_size, seq_len).copy())
        # Audit M8: return an explicit `wrapped` flag. The old 3-tuple made the
        # caller's `if offset == 0` rotation branch unreachable (get_batch
        # already rewound internally and returned bs*seq ≠ 0), so mixed-stream
        # sampling ran exactly ONE stream for the whole run and never reset the
        # VSA/bridge/reasoning document state at boundaries.
        return x.long(), y.long(), offset + batch_size * seq_len, wrapped


def _opt_param_names(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [names[id(p)] for g in optimizer.param_groups for p in g['params']]


def _restore_optimizer(optimizer, model, ckpt_opt):
    """Restore AdamW state BY PARAMETER NAME.

    Positional load shifts state onto wrong params whenever the parameter
    list changed (freq_scale/bind_coh_gate/W_out+K added): index i in the old
    checkpoint no longer refers to the same parameter. We re-map by name:
    new checkpoints carry 'param_names' (order of param_groups); old
    checkpoints without it get a FRESH Adam (safe) instead of a broken
    positional restore.
    """
    old_names = ckpt_opt.get('param_names') if isinstance(ckpt_opt, dict) else None
    if old_names is None:
        print('  WARNING: checkpoint has no param_names — optimizer state NOT restored (fresh Adam)')
        return False
    names = {id(p): n for n, p in model.named_parameters()}
    pos = {id(p): i for i, p in enumerate(
        (p for g in optimizer.param_groups for p in g['params']))}
    new_sd = optimizer.state_dict()
    new_sd['state'] = {}
    old_state = ckpt_opt.get('state', {})
    old_groups = ckpt_opt.get('param_groups', [])
    # матчинг по имени: старое имя -> слот
    moved = skipped = 0
    for name, p in model.named_parameters():
        if name not in old_names:
            skipped += 1
            continue
        si = old_names.index(name)
        st = old_state.get(str(si)) if str(si) in old_state else old_state.get(si)
        if st is None:
            continue
        if tuple(st['exp_avg'].shape) != tuple(p.shape):
            # W_out +K (когерентность): первые строки те же — частичный restore
            if (name.endswith('bind.W_out')
                    and len(st['exp_avg'].shape) == 2
                    and st['exp_avg'].shape[1] == p.shape[1]
                    and st['exp_avg'].shape[0] < p.shape[0]):
                st = {k: (v.clone() if isinstance(v, torch.Tensor) and v.dim() == 2
                          and v.shape[0] == p.shape[0]
                          else (torch.ones(p.shape[0], v.shape[1], dtype=v.dtype, device=v.device)
                                if k == 'exp_avg_sq' and isinstance(v, torch.Tensor) and v.dim() == 2
                                else (torch.zeros(p.shape[0], v.shape[1], dtype=v.dtype, device=v.device)
                                      if isinstance(v, torch.Tensor) and v.dim() == 2 else v)))
                      for k, v in st.items()}
                for k in ('exp_avg', 'exp_avg_sq'):
                    v = st[k]
                    if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] < p.shape[0]:
                        pv = v.new_zeros(p.shape[0], v.shape[1])
                        pv[:v.shape[0]] = v
                        if k == 'exp_avg_sq':
                            pv[v.shape[0]:] = 1.0
                        st[k] = pv
                moved += 1
            else:
                skipped += 1
                print(f'  Opt state shape mismatch {name}: '
                      f'{tuple(st["exp_avg"].shape)} vs {tuple(p.shape)}')
                continue
        else:
            moved += 1
        new_sd['state'][pos[id(p)]] = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                       for k, v in st.items()}
    # LR из старого чекпоинта
    for gi in range(min(len(new_sd['param_groups']), len(old_groups))):
        if 'lr' in old_groups[gi]:
            new_sd['param_groups'][gi]['lr'] = old_groups[gi]['lr']
    optimizer.load_state_dict(new_sd)
    print(f'  Optimizer restored by name: {moved} slots, {skipped} skipped (new params)')
    return moved > 0


def train(cfg=None, resume_path=None):
    if cfg is None:
        cfg = EVAConfig()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32  # no AMP for stability
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Data
    print(f'Loading data from {cfg.data_dir}')
    stream_files = sorted(glob.glob(os.path.join(cfg.data_dir, 'token_stream_*_eos.bin')))
    if not stream_files:
        stream_files = sorted(glob.glob(os.path.join(cfg.data_dir, 'token_stream_*_clean.bin')))
    if not stream_files:
        stream_files = sorted(glob.glob(os.path.join(cfg.data_dir, 'token_stream_*.bin')))
    if not stream_files:
        raise FileNotFoundError(f'No token_stream_*.bin files in {cfg.data_dir}')
    
    streams = [TokenStream(f) for f in stream_files]
    total_tokens = sum(s.len for s in streams)
    print(f'Found {len(streams)} files, {total_tokens:,} total tokens')
    
    # Model (retry once on OOM вЂ” transient CUDA context cleanup)
    try:
        model = EVAStack(cfg).to(device)
    except RuntimeError as e:
        if 'out of memory' in str(e) and device == 'cuda':
            print('[EVA] OOM on first attempt, clearing cache and retrying...')
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1)
            model = EVAStack(cfg).to(device)
        else:
            raise
    n_params = model.param_count()
    print(f'Model: {n_params:,} params ({n_params/1e6:.2f}M)')
    if device == 'cuda':
        print(f'  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    
    # Phase tracking state (EMA-based adaptive threshold)
    model._phase_ratio_ema = [0.0] * cfg.n_layers
    model._phase_ratio_std = [1.0] * cfg.n_layers

    # Fix for "MLP asleep": boost deep-MLP gradients to counter vanishing gradient.
    model.apply_mlp_depth_gradient_boost()
    
    # Unified principled adaptation (core.adaptation) — single source of truth.
    # Replaces the old scattered guard: cosine/CosineWarmup LR, ReadinessActivator
    # fixed schedule, Watchdog ce>15, and the inline bypass/aligned aux weighting.
    from core.adaptation import (LossBalancer, DepthController, LRController,
                                 FailureDetector, GradientClipper,
                                 set_active_depth, build_optimizer)

    def _make_opt(lr):
        return build_optimizer(model, lr, llrd_decay=cfg.llrd,
                               weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
                               optimizer=getattr(cfg, 'optimizer', 'adamw'))

    optimizer = _make_opt(cfg.lr)
    # LR controller: linear warmup + mirror-adaptive multiplier + plateau damping.
    scheduler = LRController(model, optimizer, cfg=cfg)
    # Progressive unfreeze: validation-loss plateau drives capacity expansion.
    depth = DepthController(model, n_layers=cfg.n_layers, init_k=cfg.init_active_layers,
                            unfreeze_inc=4, eval_interval=cfg.eval_interval)
    # Aux-loss balancer: spectral alignment (bounds aux grad by ||g_CE||).
    balancer = LossBalancer(align=True, align_cap=10.0, eval_interval=cfg.eval_interval)
    # Statistical failure detector (3-sigma) + fresh Adam + LR rewind.
    watchdog = FailureDetector(model, scheduler, _make_opt,
                               os.path.join(cfg.save_dir, 'best.pt'), cfg.lr,
                               k_sigma=3.0, warmup=cfg.warmup_steps)
    # Adaptive gradient clipping (AGC, scale-free ratio). EVA-блоки
    # трансформероподобны (MLP + концепт-внимание) -> docstring рекомендует
    # c->0.1 для transformer-блоков (0.01 — режим ResNet из статьи).
    clipper = GradientClipper(c=0.1)
    # τ-aware AGC: per-layer c_eff = c·(τ_ref/τ_l)^γ from the model's τ-ladder.
    clipper.attach(model)

    # AMP (Automatic Mixed Precision)
    use_amp = getattr(cfg, 'use_amp', False) and device == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print('  AMP: ON (mixed precision)')

    print(f'Adaptation: LLRD(={cfg.llrd}) + mirror-adaptive LR + plateau depth '
          f'(init={cfg.init_active_layers}) + 3-sigma watchdog + AGC + spectral aux')
    
    # Resume
    start_step = 0
    best_val_loss = float('inf')
    if resume_path == 'auto':
        # Find latest checkpoint: interrupt > step_* > best
        ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, 'interrupt_step_*.pt')))
        if not ckpts:
            ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, 'step_*.pt')))
        if not ckpts:
            ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, 'best.pt')))
        if ckpts:
            resume_path = ckpts[-1]
            print(f'Auto-resuming from latest: {resume_path}')
    if resume_path and os.path.exists(resume_path):
        print(f'Resuming from {resume_path}')
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        sd = dict(ckpt['model'])
        from core.migrate import migrate_state_dict
        sd, n_migrated = migrate_state_dict(sd, model)
        if n_migrated:
            print(f'  MIGRATED {n_migrated} keys (W_out +K, bind_coh_gate=0, freq_scale=1.0) — старое поведение сохранено')
        # Filter size-mismatched keys (e.g. L2 slots changed 16->32)
        _model_sd = model.state_dict()
        _filtered = {}
        for k, v in sd.items():
            if k in _model_sd and _model_sd[k].shape != v.shape:
                print(f'  SKIP size-mismatch: {k} ckpt={list(v.shape)} model={list(_model_sd[k].shape)}')
            else:
                _filtered[k] = v
        missing, unexpected = model.load_state_dict(_filtered, strict=False)
        if getattr(cfg, 'reset_skip_alpha', False):
            nzero = 0
            for layer in model.layers:
                layer.mirror.log_skip_alpha.data.zero_()
                nzero += 1
            print(f'  reset_skip_alpha: zeroed log_skip_alpha in {nzero} mirror layers (SMF L0-fix)')
        # Fix for "MLP asleep": reopen the cognitive gate on resume. The REAL gate
        # is mirror.hybrid_gate (sigmoid+softmax) replacing frozen mod_scale_mlp.
        # On resume set hybrid_gate tau so the gate starts clearly open.
        if getattr(cfg, 'mlp_gate_b_init', 0.0) > 0:
            for layer in model.layers:
                layer.mlp.mlp_gate_b.data.fill_(cfg.mlp_gate_b_init)
                # Initialize hybrid gate tau (sigmoid+softmax temperature)
                tau_val = getattr(cfg, 'mlp_hybrid_gate_tau', 1.0)
                layer.mirror.hybrid_gate.log_tau.data.fill_(math.log(tau_val))
            print(f'  reopened cognitive gate: mlp_gate_b -> {cfg.mlp_gate_b_init}, '
                  f'hybrid_gate tau -> {tau_val:.3f}')
        # P0 FIX: Reset _loss_lr_factor on resume to prevent stuck half-LR
        if hasattr(model, '_loss_lr_factor'):
            old_f = model._loss_lr_factor
            model._loss_lr_factor = 1.0
            if old_f != 1.0:
                print(f'  Reset _loss_lr_factor: {old_f:.4f} -> 1.0')
        if missing:
            print(f'  Missing keys (new arch): {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys (old arch): {len(unexpected)}')
        # R6: Invalidate cache on resume (stale cache from old weights)
        if getattr(cfg, 'logit_cache_reset_on_resume', True):
            model.reset_cache()
            print('  Cache invalidated on resume (R6)')
        # Restore optimizer/scheduler from checkpoint for stable resume.
        optimizer = _make_opt(cfg.lr)
        if 'optimizer' in ckpt and ckpt['optimizer'] is not None and not args.no_save_optimizer:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                print('  Optimizer state restored (momentum preserved)')
            except Exception as e:
                print(f'  [warn] Could not restore optimizer state: {e} — using fresh Adam')
        scheduler = LRController(model, optimizer, cfg=cfg)
        if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
            print(f'  Scheduler state restored (step={ckpt["step"]})')
        else:
            scheduler.set_step(ckpt['step'])
            print(f'  Scheduler step set to {ckpt["step"]} (no saved state)')
        depth = DepthController(model, n_layers=cfg.n_layers, init_k=cfg.init_active_layers,
                                unfreeze_inc=4, eval_interval=cfg.eval_interval)
        _saved_depth = ckpt.get('active_depth', None)
        if _saved_depth is not None:
            depth.set_depth(_saved_depth)
        else:
            depth.set_depth(min(8 + (ckpt['step'] // 15000) * 4, cfg.n_layers))  # legacy fallback (pre-fix ckpts)
        watchdog = FailureDetector(model, scheduler, _make_opt,
                                   os.path.join(cfg.save_dir, 'best.pt'), cfg.lr,
                                   k_sigma=3.0, warmup=cfg.warmup_steps)
        print('  Optimizer/scheduler rebuilt FRESH (no momentum restore)')
        start_step = ckpt['step']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
    reasoning_enabled_step = ckpt.get('reasoning_enabled_step', 0) if resume_path and os.path.exists(resume_path) else 0
    
    # State for recurrent layers
    state = None
    gs = None
    rng = torch.Generator().manual_seed(42)
    
    # Training loop
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    stream_idx = 0
    offset = 0
    tokens_seen = 0
    t0 = time.time()
    
    print(f'Starting training from step {start_step}')
    print(f'Streams: {len(streams)} ({", ".join(f"{s.len:,}" for s in streams)} tokens)')
    print('Press Ctrl+C to save checkpoint and exit gracefully.')
    try:
        for step in range(start_step, cfg.max_steps):
            model.train()
            if model.explicit_reasoning:
                model.reasoning_enabled_step = reasoning_enabled_step
            
            # (document-boundary rotation moved to the read below, after
            # seq_len is chosen — audit M8: the old pre-curriculum
            # `if offset == 0` block was unreachable after step 0.)
            
            
            # в”Ђв”Ђв”Ђ Multi-scale seq curriculum: С‡РµСЂРµРґРѕРІР°РЅРёРµ РґР»РёРЅС‹ Р±Р°С‚С‡Р° РїРѕ РѕРєС‚Р°РІР°Рј П„ в”Ђв”Ђв”Ђ
            # L=64 (П„в‰¤32, РѕРєС‚Р°РІС‹ 0вЂ“13): 7/9 С€Р°РіРѕРІ
            # L=256 (П„в‰¤92, РѕРєС‚Р°РІС‹ 14вЂ“23): 1/9 С€Р°РіРѕРІ
            # L=512 (П„в‰¤149, РѕРєС‚Р°РІС‹ 24вЂ“31): 1/9 С€Р°РіРѕРІ
            # Multi-scale seq curriculum: τ-driven progression to longer sequences
            # Phase 1 (step < τ_short): short only (64) — stabilize basics
            # Phase 2 (τ_short ≤ step < τ_long): mix short + medium (64–256)
            # Phase 3 (step ≥ τ_long): full range (64–512) — long-range patterns
            tau_short = getattr(cfg, 'seq_curriculum_tau_short', 32000)
            tau_long = getattr(cfg, 'seq_curriculum_tau_long', 96000)
            if step < tau_short:
                seq_max = 64
            elif step < tau_long:
                seq_max = 256
            else:
                seq_max = 512
            seq_pool = [s for s in [64, 128, 256, 512] if s <= seq_max]
            seq_len = seq_pool[step % len(seq_pool)]
            
            # ─── Mixed stream sampling: document-boundary rotation (audit M8) ───
            # The OLD `if offset == 0` rotation was unreachable after step 0
            # (get_batch wrapped internally and returned a non-zero offset), so
            # one stream served the entire run and the streaming document state
            # (VSA memory, bridge, reasoning) never reset at boundaries.
            # Rotation now happens HERE, before the read: when the next batch
            # does not fit, switch to a random stream and reset state.
            _need = cfg.batch_size * seq_len + 1
            if offset == 0 or offset + _need > streams[stream_idx].len:
                # holdout: the LAST stream belongs to evaluate() (audit M8) —
                # the old sampler could pick it, so 'val' was in-train data.
                stream_idx = torch.randint(0, max(len(streams) - 1, 1), (1,), generator=rng).item()
                offset = 0
                state = None  # reset state on stream switch (document boundary)
                gs = None
                if model.bridge is not None:
                    model.bridge.bridge_stream.zero_()  # reset bridge memory at document boundary
                if getattr(model, 'memory_bank', None) is not None:
                    model.memory_bank.reset()  # reset streaming banks at document boundary
                if model.explicit_reasoning:
                    model.reset_reasoning()  # new document: new chain
            stream = streams[stream_idx]
            x, y, offset, _wrapped = stream.get_batch(seq_len, cfg.batch_size, offset, cfg.vocab)
            
            x, y = x.to(device), y.to(device)
            
            # ─── Forward (with optional AMP) ───
            with autocast('cuda', enabled=use_amp):
                h = model.embed_tokens(x)
            out, state, gs, _ = model(h, state, global_state=gs, step=step, tokens=x)
            # Salience must come from the HEAD logits (same contract as the
            # Colab loop & the stack docstring); passing the raw hidden state
            # silently fed non-logit values into the salience stats (audit M8).
            model.observe_output(model.lm_head(out))  # salience of THIS step -> next step's intent
            ce_loss, aux_dict = model.compute_losses(out, y, h_emb=h)

            ce_val = ce_loss.item()
            # Progressive unfreeze (validation-plateau driven)
            depth.update(step)
            # Statistical watchdog: CE explosion -> rollback + fresh Adam + LR rewind.
            # Audit M8: train.py never passed the protective metrics and never
            # armed CE (watchdog.ce_armed stays False) — the Colab loop has both;
            # here the watchdog was decoration. Metrics mirror the notebook.
            _mets = {}
            try:
                _dv = aux_dict.get('diversity')
                if _dv is not None and isinstance(_dv, torch.Tensor):
                    _mets['diversity'] = float(_dv.abs().item())
                _gl = aux_dict.get('gate_l1')
                if _gl is not None and isinstance(_gl, torch.Tensor):
                    _mets['gate_l1'] = float(_gl.item())
                with torch.no_grad():
                    _rr = [getattr(l, '_mlp_ratio', None) for l in model.layers]
                    if _rr and all(v is not None for v in _rr):
                        _mets['mlp_ratio'] = float(max(_rr))
                    _ige = [getattr(l.mirror, '_cached_ig_eff', None) for l in model.layers]
                    if _ige and all(v is not None for v in _ige):
                        _mets['ig_eff'] = float(sum(_ige) / len(_ige))
            except Exception:
                pass
            if watchdog.check(ce_val, step, _mets):
                optimizer = watchdog.optimizer
                cfg.lr = watchdog.base_lr
                state = None
                gs = None
                model.reset_cache()   # scrub NaN-poisoned runtime EMAs too (M8)
                optimizer.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                continue

            # ── Gradient-reactive governance loss ─────────────────────────
            # Moved into core.losses.compute_losses (audit M5): the target
            # ‖∂CE/∂mlp_out‖ per expert is captured by a backward HOOK in the
            # block during the regular pass (no extra autograd.grad = no
            # second backward per step), the term carries its real
            # cfg.gradalign_weight and BYPASSES spectral alignment via
            # LossBalancer.BYPASS_AUX. aux_dict['gradalign'] is already set.

            # NaN guard
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                raise RuntimeError(f'NaN/Inf CE loss at step {step}')

            state = _detach_state(state)
            if gs is not None:
                gs = gs.detach()

            # Principled aux balancing via spectral gradient alignment
            # (core.adaptation.LossBalancer). All aux weighting is data-derived —
            # no per-loss magic constants, and the aux gradient is bounded by
            # ||g_CE|| so it can never hijack the update. Under AMP the losses are
            # scaled so the grads survive GradScaler.unscale_/step.
            # P1 FIX: Anneal noisy auxiliary losses over training (τ-gated timescale)
            # branch, diversity add noise early → mask real signal.
            # Warm-restart: linearly ramp from 0→1 over τ_anneal steps (default 5000).
            if step > 0:
                _anneal_tau = getattr(cfg, 'aux_anneal_tau', 5000)
                _anneal = min(1.0, step / _anneal_tau)
                for _k in ('branch', 'diversity'):
                    if _k in aux_dict and isinstance(aux_dict[_k], torch.Tensor):
                        aux_dict[_k] = aux_dict[_k] * _anneal
            gscale = scaler.get_scale() if use_amp else 1.0
            ce_s = ce_loss * gscale
            aux_s = {k: (v * gscale if isinstance(v, torch.Tensor) else v)
                     for k, v in aux_dict.items()}
            balancer.backward(ce_s, aux_s, model.parameters())
            
            # Adaptive phase scaling: EMA-based mirror/base gradient balance
            phase_scales = []
            for i, layer in enumerate(model.layers):
                mirror_norm = 0.0
                base_norm = 0.0
                for p in layer.mirror_parameters:
                    if p.grad is not None:
                        mirror_norm += p.grad.norm().item() ** 2
                for p in layer.base_parameters:
                    if p.grad is not None:
                        base_norm += p.grad.norm().item() ** 2
                mirror_norm = mirror_norm ** 0.5
                base_norm = base_norm ** 0.5
                ratio = mirror_norm / (base_norm + 1e-8)
                ema = model._phase_ratio_ema[i]
                std = model._phase_ratio_std[i]
                model._phase_ratio_ema[i] = 0.99 * ema + 0.01 * ratio
                model._phase_ratio_std[i] = 0.99 * std + 0.01 * abs(ratio - ema)
                mir_s = max(0.2, min(2.0, 1.0 / (1.0 + math.exp(-(ratio - ema) / (std + 1e-8)))))
                phase_scales.append((mir_s, ratio))
                for p in layer.mirror_parameters:
                    if p.grad is not None:
                        p.grad *= mir_s
            mean_mirror_scale = sum(s[0] for s in phase_scales) / len(phase_scales)
            mean_ratio = sum(s[1] for s in phase_scales) / len(phase_scales)

            # Per-layer LS-based LR modulation (cfg.per_layer_ls_lr)
            ls_mults = getattr(scheduler, '_ls_mult', None)
            if ls_mults is not None:
                mirror_hi = getattr(cfg, 'ls_mirror_mult_max', 2.0)
                for i, layer in enumerate(model.layers):
                    ls_m = ls_mults[i]
                    for p in layer.base_parameters:
                        if p.grad is not None:
                            p.grad.mul_(ls_m)
                    if ls_m != 1.0:
                        mir_s = phase_scales[i][0]
                        total = max(0.2, min(mirror_hi, mir_s * ls_m))
                        for p in layer.mirror_parameters:
                            if p.grad is not None:
                                p.grad.mul_(total / mir_s)
            tokens_seen += cfg.batch_size * seq_len
            
            # Clip gradients (AGC — scale-free ratio, replaces magic grad_clip)
            # U9: keep the τ-AGC per-layer map fresh (τ ladder drifts slowly).
            if step % cfg.eval_interval == 0:
                clipper.attach(model)
            if use_amp:
                scaler.unscale_(optimizer)
            clipper.clip(model.parameters())

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                if getattr(cfg, 'optimizer', 'adamw') in ('eva', 'eva_proj'):
                    _t = {}
                    if model.bridge is not None:
                        _t["bridge"] = float(model.bridge.readiness())
                    if model.maturation is not None:
                        _rm = abs(float(model.maturation.gate.float().mean()))
                        _t["intent"] = _rm
                        _t["mem"] = _rm
                    optimizer.set_trust(_t)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if model.explicit_reasoning:
                reasoning_enabled_step += 1
            
            current_lr = scheduler.get_last_lr()[0]
            
            # в”Ђв”Ђв”Ђ Soft EOS-aware state reset: Р·Р°С‚СѓС…Р°РЅРёРµ РІРјРµСЃС‚Рѕ РѕР±РЅСѓР»РµРЅРёСЏ в”Ђв”Ђв”Ђ
            if (y[:, -1] == 2).any():
                if state is not None:
                    state = tuple(s * 0.1 for s in state)
                if gs is not None:
                    gs = gs * 0.1
            
            # Log
            if step % cfg.log_interval == 0:
                dt = time.time() - t0
                tok_s = tokens_seen / max(dt, 1e-8)
                lc = getattr(model, '_cached_losses', {})
                aux_str = ' '.join(f'{k}={v:.4f}' for k, v in lc.items())
                gate_str = ''
                rg = getattr(model, 'reasoning_gate', None)
                if rg is not None:
                    gates = getattr(model, '_reasoning_gates', None)
                    if gates:
                        gate_str = ' gates=' + str([round(g, 3) for g in gates])
                mod_scl = 0.0
                try:
                    mod_scl = torch.stack([torch.sigmoid(l.mirror.mod_scale_mlp).mean()
                                           for l in model.layers]).mean().item()
                except Exception:
                    pass
                print(f'  step={step:>6} loss={ce_loss.item():.4f} mod_mlp={mod_scl:.3f} lr={current_lr:.2e} '
                      f'tok/s={tok_s:.0f} stream={stream_idx} '
                      f'ms={mean_mirror_scale:.3f} mr={mean_ratio:.4f} | {aux_str}{gate_str}')
            
            # Eval
            if step > 0 and step % cfg.eval_interval == 0:
                val_loss = evaluate(model, streams, cfg, device)
                print(f'  EVAL step={step}: val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2e}')
                if device == 'cuda':
                    torch.cuda.empty_cache()
                scheduler.report_val_loss(val_loss)
                depth.update(step, val_loss)
                watchdog.ce_armed = True  # CE joins the watch once val is trusted (M8)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(cfg.save_dir, f'best.pt')
                    _save_checkpoint_safely({
                        'step': step,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict() if not args.no_save_optimizer else None,
                        'param_names': _opt_param_names(model, optimizer) if not args.no_save_optimizer else None,
                        'scheduler': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'cfg': cfg,
                        'reasoning_enabled_step': reasoning_enabled_step,
                        'active_depth': depth.active,
                    }, save_path)
                    print(f'  Saved best model to {save_path}')
                    generate_report(save_path)
            
            # Periodic step_*.pt checkpoints DISABLED: only best.pt is written (saves space).
    except KeyboardInterrupt:
        print('\n[EVA] Ctrl+C detected - keeping last best.pt (no separate checkpoint written)')
        print('[EVA] Exiting gracefully.')
        sys.exit(0)
    
    print('Training complete!')


@torch.no_grad()
def evaluate(model, streams, cfg, device):
    model.eval()
    # Audit M8: (a) `len(stream)` crashed — TokenStream exposes `.len` only,
    # so train.py died at its FIRST eval (eval_interval≈233); (b) val
    # forwards mutate the streaming runtime buffers (memory banks, intent bus,
    # EMAs) — snapshot/restore isolates validation documents from the training
    # working memory; (c) adaptive=False matches the notebook (controller
    # buffers must not learn from val); (d) fresh state per batch.
    _rt_snap = model.snapshot_runtime_buffers()
    if getattr(model, 'explicit_reasoning', False):
        model.reset_reasoning()
    total_loss = 0.0
    total_steps = 0
    
    # Use last stream for eval (hold-out, excluded from the train sampler)
    stream = streams[-1]
    offset = max(stream.len // 2, cfg.batch_size * cfg.seq_len + 1)
    
    for _ in range(min(100, stream.len // (cfg.batch_size * cfg.seq_len))):
        x, y, offset, wrapped = stream.get_batch(cfg.seq_len, cfg.batch_size, offset, cfg.vocab)
        if wrapped:
            break  # end of the hold-out document region — no wrapped re-read
        x, y = x.to(device), y.to(device)
        h = model.embed_tokens(x)
        out, _, _, _ = model(h, None, adaptive=False, tokens=x)
        loss = model.compute_loss(out, y, h_emb=h)
        total_loss += loss.item()
        total_steps += 1
    
    model.restore_runtime_buffers(_rt_snap)
    model.train()
    return total_loss / max(total_steps, 1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=24)
    parser.add_argument('--D', type=int, default=4096, help='model width')
    parser.add_argument('--vocab', type=int, default=50000)
    parser.add_argument('--mirror-k', type=int, default=32)
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--llrd', type=float, default=0.9, help='layer-wise LR decay per depth (deeper=smaller LR)')
    parser.add_argument('--init-active-layers', type=int, default=8, help='blocks trained from step 0 (rest frozen)')
    parser.add_argument('--stage-steps', type=int, default=15000, help='unlock next block every N steps (backstop)')
    parser.add_argument('--readiness-full', type=float, default=0.6, help='meta-maturity (differentiation) to unlock deepest block')
    parser.add_argument('--stage-mode', type=str, default='readiness', choices=['readiness', 'fixed'])
    parser.add_argument('--watchdog-ce', type=float, default=15.0, help='CE above this => rollback + fresh Adam')
    parser.add_argument('--recover-lr-mult', type=float, default=0.5)
    parser.add_argument('--recover-max', type=int, default=20)
    parser.add_argument('--no-grad-ckpt', action='store_true',
                        help='Disable gradient checkpointing (avoids CheckpointError if recompute mismatches)')
    parser.add_argument('--bind-K', type=int, default=64)
    parser.add_argument('--mlp-groups', type=int, default=8)
    parser.add_argument('--mlp-expand', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=5000)
    parser.add_argument('--scheduler', type=str, default='mirror', choices=['cosine', 'mirror'])
    parser.add_argument('--per-layer-ls-lr', action='store_true',
                        help='Per-layer LR modulation from fast/slow EMA var(log_scale)')
    parser.add_argument('--head', type=str, default='partitioned', choices=['partitioned', 'codec'],
                        help='LM head: partitioned (softmax-CE) or codec (SignedAmpCodec CE + W_pred + echo)')
    parser.add_argument('--amp-obj', type=str, default='ce', choices=['ce', 'mh'],
                        help='Codec objective: ce = one CE (confirmed recipe), mh = margin+hinge')
    parser.add_argument('--no-amp-pred', action='store_true',
                        help='Disable W_pred transition operator in codec head')
    parser.add_argument('--traj-manifold', action='store_true',
                        help='Trajectory: РјР°РЅРёС„РѕР»Рґ РїРµСЂРµС…РѕРґРѕРІ (TrajectoryManifoldBind)')
    parser.add_argument('--traj-beams', type=int, default=0, help='Manifold: С‡РёСЃР»Рѕ Р»СѓС‡РµР№ (0 = Р°РІС‚Рѕ = ceil(sqrt(buffer)))')
    parser.add_argument('--traj-buffer', type=int, default=1024, help='Manifold: Р±СѓС„РµСЂ РїРµСЂРµС…РѕРґРѕРІ')
    parser.add_argument('--traj-gain', type=float, default=0.05, help='Manifold: РјР°СЃС€С‚Р°Р± РІРєР»Р°РґР°')
    parser.add_argument('--reset-skip-alpha', action='store_true',
                        help='Zero log_skip_alpha in all mirror layers after resume (SMF L0-depth fix)')
    parser.add_argument('--no-save-optimizer', action='store_true',
                        help='Do NOT save optimizer state in checkpoints (avoids resume OOM on <=16GB GPU)')
    args = parser.parse_args()
    
    cfg = EVAConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_layers=args.n_layers,
        D=args.D,
        vocab=args.vocab,
        mirror_k=args.mirror_k,
        bind_K=args.bind_K,
        mlp_groups=args.mlp_groups,
        mlp_expand=args.mlp_expand,
        lr=args.lr,
        grad_clip=args.grad_clip,
        llrd=args.llrd,
        init_active_layers=args.init_active_layers,
        stage_steps=args.stage_steps,
        readiness_full=args.readiness_full,
        stage_mode=args.stage_mode,
        watchdog_ce=args.watchdog_ce,
        recover_lr_mult=args.recover_lr_mult,
        recover_max=args.recover_max,
        max_steps=args.max_steps,
        warmup_steps=args.warmup,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        scheduler=args.scheduler,
        per_layer_ls_lr=False,
        traj_manifold=args.traj_manifold,
        traj_beams=args.traj_beams,
        traj_buffer_size=args.traj_buffer,
        traj_gain=args.traj_gain,
        reset_skip_alpha=args.reset_skip_alpha,
        gradient_checkpointing=(not args.no_grad_ckpt),
    )
    
    train(cfg, resume_path=args.resume)
