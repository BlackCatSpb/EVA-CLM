"""
EVA training: streaming from token_stream_{GENRE}.bin files.
"""

import os, sys, math, time, json, glob, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import torch
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import CheckpointError as _CEr  # M22
import torch.nn.functional as F
import numpy as np
from torch.serialization import add_safe_globals

from core import EVAConfig, EVAStack, MirrorLRScheduler
from core.training_control import (codebook_fingerprint,
                                verify_identity_resume, apply_tau_lr,
                                grad_census, training_telemetry)  # M64.8


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
        self.path = path
        self.len = len(self.data)
    def get_batch(self, seq_len, batch_size, offset, vocab=None):
        needed = batch_size * seq_len + 1
        wrapped = offset + needed > self.len
        if wrapped:
            offset = 0
        chunk = self.data[offset:offset + needed]
        # B7 (audit 01, F-01): the old silent np.clip folded 1.3-7.8% of real
        # corpus ids (e.g. 25.5M tokens >= 50000 in FANTASY, incl. id 65535)
        # onto one junk token. A corpus/model mismatch is fatal and LOUD now.
        if vocab is not None:
            _hi = int(chunk.max(initial=0))
            if _hi >= vocab:
                raise ValueError(
                    f'TokenStream: token id {_hi} >= vocab {vocab} in {self.path!r} '
                    '- corpus and model vocabulary disagree (no silent clipping).')
        x = torch.from_numpy(chunk[:batch_size * seq_len].reshape(batch_size, seq_len).copy())
        y = torch.from_numpy(chunk[1:batch_size * seq_len + 1].reshape(batch_size, seq_len).copy())
        # Audit M8: return an explicit `wrapped` flag. The old 3-tuple made the
        # caller's `if offset == 0` rotation branch unreachable (get_batch
        # already rewound internally and returned bs*seq ≠ 0), so mixed-stream
        # sampling ran exactly ONE stream for the whole run and never reset the
        # VSA/bridge/reasoning document state at boundaries.
        return x.long(), y.long(), offset + batch_size * seq_len, wrapped


def _dstate(o):
    # B15 (audit 05 F5-07): the loop's streaming state (per-layer VSA,
    # global_state) lived OUTSIDE best.pt — a mid-document resume cold-started
    # the document (measured 0.37 nat first-window CE displacement). Deep
    # detach-to-CPU for save / device-move for load.
    if torch.is_tensor(o):
        return o.detach().cpu()
    if isinstance(o, (list, tuple)):
        return [_dstate(x) for x in o]
    return o

def _tstate(o, dev):
    if torch.is_tensor(o):
        return o.to(dev)
    if isinstance(o, (list, tuple)):
        return [_tstate(x, dev) for x in o]
    return o


def _opt_param_names(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [names[id(p)] for g in optimizer.param_groups for p in g['params']]


def _restore_optimizer(optimizer, model, ckpt_opt, param_names=None):
    """Restore AdamW state BY PARAMETER NAME.

    Positional load shifts state onto wrong params whenever the parameter
    list changed (freq_scale/bind_coh_gate/W_out+K added): index i in the old
    checkpoint no longer refers to the same parameter. We re-map by name:
    new checkpoints carry 'param_names' (order of param_groups); old
    checkpoints without it get a FRESH Adam (safe) instead of a broken
    positional restore.
    """
    # B15 (F5-03): param_names rides the CKPT top level (where the save
    # writes it), not the optimizer dict — the old lookup never fired and
    # every resume silently got fresh Adam moments with a 'by name' print.
    old_names = param_names if param_names is not None else (
        ckpt_opt.get('param_names') if isinstance(ckpt_opt, dict) else None)
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


def _pick_stream(stream_idx, n_pick, no_repeat, rng):
    """M62: choose the next genre stream.

    `no_repeat` (chunk rotation on) excludes the CURRENT stream: a switch into
    the same genre is a no-op for the novelty machinery (lacuna gate, phantom
    bank, UCL births) — it would look like a shift but carry no distribution
    change. The legacy path (rotation only on exhaustion) keeps the original
    uniform pick so old behaviour/checkpoints are unaffected.
    """
    n_pick = max(int(n_pick), 1)
    if no_repeat and n_pick > 1:
        p = int(torch.randint(0, n_pick - 1, (1,), generator=rng).item())
        return p if p < int(stream_idx) else p + 1
    return int(torch.randint(0, n_pick, (1,), generator=rng).item())


def train(cfg=None, resume_path=None):
    if cfg is None:
        cfg = EVAConfig()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:  # M16: expandable segments (env var alone is too late on some hosts)
        torch.cuda.memory._set_allocator_settings('expandable_segments:True')
    except Exception:
        pass
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
    # Audit M13: file-level hold-out (last 3 files when >=8 files exist,
    # else the single-stream M8 semantics). Excluded from the train sampler.
    _hold_n = 3 if len(streams) >= 8 else (1 if streams else 0)
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
                                 GradientClipper,
                                 set_active_depth, build_optimizer)

    def _make_opt(lr):
        return build_optimizer(model, lr, llrd_decay=cfg.llrd,
                               weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
                               optimizer=getattr(cfg, 'optimizer', 'adamw'),
                               readout_lr_mult=float(getattr(cfg, 'readout_lr_mult', 0.0) or 0.0))

    optimizer = _make_opt(cfg.lr)
    # LR controller: linear warmup + mirror-adaptive multiplier + plateau damping.
    scheduler = LRController(model, optimizer, cfg=cfg)
    # Progressive unfreeze: validation-loss plateau drives capacity expansion.
    depth = DepthController(model, n_layers=cfg.n_layers, init_k=cfg.init_active_layers,
                            unfreeze_inc=4, eval_interval=cfg.eval_interval)
    # Aux-loss balancer: spectral alignment (bounds aux grad by ||g_CE||).
    balancer = LossBalancer(align=True, align_cap=10.0, eval_interval=cfg.eval_interval,
                            align_every=int(1 if getattr(cfg, 'balancer_align_every', 1) is None
                                            else getattr(cfg, 'balancer_align_every', 1)),
                            kill_terms=list(LossBalancer.AUX_TERMS)
                            if getattr(cfg, 'aux_kill_switch', False) else None,
                            kill_disable=bool(getattr(cfg, 'aux_kill_disable', False)))
    # Adaptive gradient clipping (AGC, scale-free ratio). EVA-блоки
    # трансформероподобны (MLP + концепт-внимание) -> docstring рекомендует
    # c->0.1 для transformer-блоков (0.01 — режим ResNet из статьи).
    clipper = GradientClipper(c=0.1)
    # τ-aware AGC: per-layer c_eff = c·(τ_ref/τ_l)^γ from the model's τ-ladder.
    clipper.attach(model)

    # AMP (Automatic Mixed Precision)
    use_amp = getattr(cfg, 'use_amp', False) and device == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    # M42 (mirror of the notebook): one envelope builder + rolling latest.pt
    def _full_env(_step):
        model.flush_control_pending()
        return {
            'step': int(_step), 'model': model.state_dict(),
            'code_fp': codebook_fingerprint(model),
            'optimizer': optimizer.state_dict(),
            'param_names': _opt_param_names(model, optimizer),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': float(best_val_loss), 'cfg': cfg,
            'reasoning_enabled_step': reasoning_enabled_step,
            'active_depth': depth.active, 'depth_state': depth.get_state(),
            'balancer': balancer.state_dict(),
            # M58c: the M51 branch-anchor reference rides too — without it a
            # resume re-seeds the anchor from the (possibly drifted) current
            # branch variances instead of the original healthy scale.
            'branch_var_ref': (model._branch_var_ref.detach().cpu()
                               if getattr(model, '_branch_var_ref', None) is not None else None),
            'stream_idx': int(stream_idx), 'offset': int(offset),
            'rng': torch.get_rng_state(), 'data_rng': rng.get_state(),
            'stream_state': _dstate(state), 'stream_gs': _dstate(gs if gs is not None else None),
            'cuda_rng': torch.cuda.get_rng_state() if device == 'cuda' else None,
        }

    def _atomic42(env, name):
        p = os.path.join(cfg.save_dir, name); q = p + '.tmp'
        _save_checkpoint_safely(env, q)
        os.replace(q, p)
        return p
    if use_amp:
        print('  AMP: ON (mixed precision)')

    _llrd_note = ('' if abs(float(cfg.llrd) - 1.0) < 1e-9
                  else ' ACTIVE — double-LLRD, legacy!')
    print(f'Adaptation: tau-LLRD (single source; index llrd={cfg.llrd}{_llrd_note}) '
          f'+ mirror-adaptive LR + plateau depth (init={cfg.init_active_layers}) '
          f'+ AGC + spectral aux')
    
    # Resume
    start_step = 0
    best_val_loss = float('inf')
    resumed_offset = 0
    resumed_stream_idx = 0
    _m12_rng = None
    _m12_data_rng = None
    if resume_path == 'auto':
        # Find the checkpoint: interrupt > step_* > best (operator: the name is
        # best, as before — the M42 rolling latest.pt is gone: the poisoned
        # 2970 file was auto-resumed once and cost a whole session).
        ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, 'interrupt_step_*.pt')))
        if not ckpts:
            ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, 'step_*.pt')))
        if not ckpts:
            ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, 'best.pt')))
        if ckpts:
            resume_path = ckpts[-1]
            print(f'Auto-resuming from: {resume_path}')
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
        _skipped = []  # B8: identity drift is fatal, collect names
        _filtered = {}
        for k, v in sd.items():
            if k in _model_sd and _model_sd[k].shape != v.shape:
                print(f'  SKIP size-mismatch: {k} ckpt={list(v.shape)} model={list(_model_sd[k].shape)}')
                _skipped.append(k)
            else:
                _filtered[k] = v
        missing, unexpected = model.load_state_dict(_filtered, strict=False)
        verify_identity_resume(model, ckpt, _skipped)  # B15 (F5-01): it returns the fp, not a tuple
        if getattr(cfg, 'reset_skip_alpha', False):
            nzero = 0
            for layer in model.layers:
                layer.mirror.log_skip_alpha.data.zero_()
                nzero += 1
            print(f'  reset_skip_alpha: zeroed log_skip_alpha in {nzero} mirror layers (SMF L0-fix)')
        # Fix for "MLP asleep": reopen the cognitive gate on resume. The REAL gate
        # is mirror.hybrid_gate (sigmoid+softmax) replacing frozen mod_scale_mlp.
        # On resume set hybrid_gate tau so the gate starts clearly open.
        _gate_missing = any('mlp_gate_b' in _m for _m in missing)   # M58c (B15)
        if getattr(cfg, 'mlp_gate_b_init', 0.0) > 0 and _gate_missing:
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
                # B12 (F3-08): positional Adam restore shifts states onto wrong
                # tensors when param order changes; by-name restorer existed but
                # was dead code here (the notebook has always used it).
                _restore_optimizer(optimizer, model, ckpt['optimizer'],
                                 param_names=ckpt.get('param_names'))
                print('  Optimizer state restored BY NAME (momentum preserved)')
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
        print('  Optimizer/scheduler rebuilt FRESH (no momentum restore)')
        start_step = ckpt['step']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        depth.put_state(ckpt.get('depth_state'))  # B14 (F4-06)
        if ckpt.get('balancer') is not None:
            balancer.load_state_dict(ckpt['balancer'])
        if ckpt.get('branch_var_ref') is not None:      # M58c (M51 anchor)
            model._branch_var_ref = ckpt['branch_var_ref'].to(device)
        if ckpt.get('stream_state') is not None:   # B15 (F5-07): mid-document
            state = _tstate(ckpt['stream_state'], device)   # streaming continuity
            if ckpt.get('stream_gs') is not None:
                gs = _tstate(ckpt['stream_gs'], device)
        if ckpt.get('cuda_rng') is not None and device == 'cuda':   # B16
            torch.cuda.set_rng_state(ckpt['cuda_rng'])
        _m12_rng = ckpt.get('rng')
        _m12_data_rng = ckpt.get('data_rng')
        resumed_offset = int(ckpt.get('offset', 0) or 0)
        resumed_stream_idx = int(ckpt.get('stream_idx', 0) or 0)
    reasoning_enabled_step = ckpt.get('reasoning_enabled_step', 0) if resume_path and os.path.exists(resume_path) else 0
    
    # State for recurrent layers
    state = None
    gs = None
    rng = torch.Generator().manual_seed(42)
    if _m12_rng is not None:          # resume the dropout/sampling stream (M12)
        torch.set_rng_state(_m12_rng.cpu())  # map_location=device may have moved it to CUDA
    if _m12_data_rng is not None:     # resume the document-shuffle stream
        rng.set_state(_m12_data_rng.cpu())
    
    # Training loop
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    stream_idx = resumed_stream_idx   # continue the data cursor (audit M12)
    if stream_idx >= max(len(streams) - _hold_n, 1):   # pre-M13 cursor (M13)
        stream_idx, offset = 0, 0
    offset = resumed_offset
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
            # M62: additionally rotate every `stream_chunk_steps` steps (the
            # novelty machinery needs distribution SHIFTS to have a job: the
            # exhaustion-only cadence was ~1200 steps per genre, ~6 shifts per
            # 7.3k steps — nearly nothing to test the lacuna/phantom/UCL chain).
            _need = cfg.batch_size * seq_len + 1
            _chunk = int(getattr(cfg, 'stream_chunk_steps', 0) or 0)
            _rotate = (_chunk > 0 and step > 0 and step % _chunk == 0)
            if offset == 0 or offset + _need > streams[stream_idx].len or _rotate:
                # holdout: the LAST stream belongs to evaluate() (audit M8) —
                # the old sampler could pick it, so 'val' was in-train data.
                stream_idx = _pick_stream(stream_idx, len(streams) - _hold_n,
                                          _rotate, rng)
                offset = 0
                state = None  # reset state on stream switch (document boundary)
                gs = None
                if model.bridge is not None:
                    model.bridge.bridge_stream.zero_()  # reset bridge memory at document boundary
                if getattr(model, 'memory_bank', None) is not None:
                    model.memory_bank.reset()  # reset streaming banks at document boundary
                if getattr(model, 'logit_cache', None) is not None:
                    model.logit_cache.cache.clear()  # new document ⇒ empty cache (decision #3)
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

            depth.update(step)
            # ── Gradient-reactive governance loss ─────────────────────────
            # Moved into core.losses.compute_losses (audit M5): the target
            # ‖∂CE/∂mlp_out‖ per expert is captured by a backward HOOK in the
            # block during the regular pass (no extra autograd.grad = no
            # second backward per step), the term carries its real
            # cfg.gradalign_weight and BYPASSES spectral alignment via
            # LossBalancer.BYPASS_AUX. aux_dict['gradalign'] is already set.


            state = _detach_state(state)
            if gs is not None:
                gs = gs.detach()

            # Principled aux balancing via spectral gradient alignment
            # (core.adaptation.LossBalancer). All aux weighting is data-derived —
            # no per-loss magic constants, and the aux gradient is bounded by
            # ||g_CE|| so it can never hijack the update. Under AMP the losses are
            # scaled so the grads survive GradScaler.unscale_/step.
            gscale = scaler.get_scale() if use_amp else 1.0
            ce_s = ce_loss * gscale
            aux_s = {k: (v * gscale if isinstance(v, torch.Tensor) else v)
                     for k, v in aux_dict.items()}
            # M48: ggeo removed from the loop (operator directive) — pure
            # telemetry, 8 retained backward passes, M47 incident trigger.
            # The capability stays in core.adaptation (LossBalancer.
            # grad_geometry) for offline diagnostics.
            # M22: backward freeze-wrap removed (recompute must replay the
            # forward path-identically); CheckpointError self-heals.
            # M64.12: the kill-switch measurement needs the LIVE graph -> here,
            # before the backward (a no-op when aux_kill_switch is off)
            _ks = {}
            if balancer.kill is not None and step % max(cfg.log_interval, 1) == 0:
                _ks = balancer.measure_kill(ce_s, aux_s, model.parameters(), phase_model=model)
            try:
                balancer.backward(ce_s, aux_s, model.parameters(), phase_model=model,
                                  step=step)
            except _CEr as _ce15:
                print('  [ckpt-fallback]', str(_ce15)[:120], '- checkpointing OFF for the run')
                cfg.gradient_checkpointing = False
                ce_s = aux_s = ce_loss = aux_dict = None
                optimizer.zero_grad(set_to_none=True)
                # M64.4 (R2 review): an exception inside the align path can leave
                # the gradalign hook frozen — restore it on the fallback.
                for _l in getattr(model, 'layers', []):
                    if hasattr(_l, '_ga_record'):
                        _l._ga_record = True
                model.release_step_graph()   # M47: drop stale graph-pinned attrs
                if device == 'cuda':
                    torch.cuda.empty_cache()
                continue

            

            # M58c: the per-layer LS-LR modulation is applied ONCE, inside
            # apply_tau_lr below (the direct grad.mul_ here was a second copy:
            # ls_mult landed on the base parameters squared).
            tokens_seen += cfg.batch_size * seq_len
            # M64.8 (M63-E): the liveness census — captured AFTER backward,
            # BEFORE zero_grad (an absent/zero entry is the dead-channel signal).
            _gc = grad_census(model) if step % max(cfg.log_interval, 1) == 0 else None

            # Clip gradients (AGC — scale-free ratio, replaces magic grad_clip)
            # U9: keep the τ-AGC per-layer map fresh (τ ladder drifts slowly).
            if step % cfg.eval_interval == 0:
                clipper.attach(model)
            if use_amp:
                scaler.unscale_(optimizer)
            # B13 (F2B-07 drift closure): the notebook had tau-lr scaling,
            # train.py never applied it. Same law, same order: BEFORE the clip.
            ls_mults = getattr(scheduler, '_ls_mult', None)
            apply_tau_lr(model, getattr(model, 'tau_config', None), ls_mults)
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
                model.release_step_graph()   # M21 (probe-proven): _cache_*/_cached_* attrs pin the whole step graph between steps
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if model.explicit_reasoning:
                reasoning_enabled_step += 1
            
            current_lr = scheduler.get_last_lr()[0]
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
                    if gates is not None and torch.as_tensor(gates).numel():  # B15 (F5-05)
                        gate_str = ' gates=' + str([round(g, 3) for g in gates])
                mod_scl = 0.0
                try:
                    mod_scl = torch.stack([torch.sigmoid(l.mirror.mod_scale_mlp).mean()
                                           for l in model.layers]).mean().item()
                except Exception:
                    pass
            if step % max(cfg.log_interval, 1) == 0:
                # M58c: the head telemetry (lacuna/SRL/phantom/conflict/wall) was
                # visible only in the notebook and the analyzer — the CLI's log
                # showed nothing of the M52-M56 channels.
                try:   # M64.2r2: telemetry must never kill the run
                    _tel = model.head_telemetry() if hasattr(model, 'head_telemetry') else {}
                    _tel_str = ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                                        for k, v in _tel.items())
                except Exception as _te2:
                    _tel_str = f'skipped ({str(_te2)[:60]})'
                # M64.4 (R3 review): the balancer's cadence telemetry — without
                # it the A/B (align_every=8 vs 1) is indistinguishable in the log.
                _bal = (f'bal_a={getattr(balancer, "n_align", 0)} '
                        f'bal_b={getattr(balancer, "n_balance", 0)} '
                        f'bal_s={getattr(balancer, "scale_ema", None) if getattr(balancer, "scale_ema", None) is None else round(float(balancer.scale_ema), 5)} '
                        f'bal_sc={getattr(balancer, "last_scale", None) if getattr(balancer, "last_scale", None) is None else round(float(balancer.last_scale), 5)} '
                        f'bal_cos={getattr(balancer, "last_align_cos", None) if getattr(balancer, "last_align_cos", None) is None else round(float(balancer.last_align_cos), 4)}')
                print(f'  step={step:>6} loss={ce_loss.item():.4f} mod_mlp={mod_scl:.3f} lr={current_lr:.2e} '
                      f'tok/s={tok_s:.0f} stream={stream_idx} {_bal} '
                      f'{aux_str}{gate_str}')
                if _tel_str:
                    print(f'  head: {_tel_str}')
                if _ks:
                    print('  ks: ' + ' '.join(f'{k}={v:.4g}' for k, v in _ks.items())
                          + ' | ' + ' '.join(f'{k}={v}' for k, v in balancer.kill.stats().items()))
                # M64.8: the telemetry batch (stable rank / alpha std / usage H
                # / the spike snapshot) + the grad census (live != effective)
                try:
                    _tt = training_telemetry(model)
                    _tt.update(_gc or {})
                    if _tt:
                        print('  tele: ' + ' '.join(
                            f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                            for k, v in _tt.items()))
                except Exception as _te:
                    print(f'  tele: skipped ({str(_te)[:60]})')
            
            # Eval
            # M49: early measurement evals (see the notebook twin)
            _canonical_eval = (step > 0 and step % cfg.eval_interval == 0)
            _early_eval = (step > 0 and not _canonical_eval
                           and step < getattr(cfg, 'eval_early_until', 3000)
                           and step % getattr(cfg, 'eval_early_every', 250) == 0)
            if _canonical_eval or _early_eval:
                val_loss = evaluate(model, streams, cfg, device)
                if not math.isfinite(val_loss):   # B12 (F3-01): empty pool — skip controllers
                    print('  EVAL skipped: empty validation pool (NaN) — no reporting', flush=True)
                else:
                    print(f'  EVAL step={step}: val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2e}')
                    if device == 'cuda':
                        torch.cuda.empty_cache()

                if _canonical_eval:
                    depth.update(step, val_loss)
                    scheduler.report_val_loss(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.flush_control_pending()   # B16 (audit 05)
                    save_path = os.path.join(cfg.save_dir, f'best.pt')
                    _save_checkpoint_safely({
                        'step': step,
                        'model': model.state_dict(), 'code_fp': codebook_fingerprint(model),
                        'optimizer': optimizer.state_dict() if not args.no_save_optimizer else None,
                        'param_names': _opt_param_names(model, optimizer) if not args.no_save_optimizer else None,
                        'scheduler': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'cfg': cfg,
                        'reasoning_enabled_step': reasoning_enabled_step,
                        'active_depth': depth.active,
                        # M12: one checkpoint carries the FULL restart state
                        'depth_state': depth.get_state(),
                        'balancer': balancer.state_dict(),
                        'stream_idx': int(stream_idx), 'offset': int(offset),
                        'rng': torch.get_rng_state(), 'data_rng': rng.get_state(),
                        'stream_state': _dstate(state), 'stream_gs': _dstate(gs if gs is not None else None),
                        'cuda_rng': torch.cuda.get_rng_state() if device == 'cuda' else None,  # B16
                    }, save_path)
                    print(f'  Saved best model to {save_path}')
                    try:   # B15 (F5-04): save_html_report wants (ckpt, cfg, model, ...);
                        generate_report(save_path)   # calling it 1-arg crashed the run
                    except Exception as _re:         # right AFTER saving. Report is
                        print(f'  [report] skipped: {_re}')   # diagnostics, not training.

            
            # M42 flush (the rolling latest.pt is gone — operator: the
            # checkpoint name is best, as before). The 495-step flush of
            # non-continuity caches (logit cache, mirror stream caches,
            # allocator) stays: it bounds the ggeo-spike fragmentation creep
            # seen on the A100 cycle.
            if step > 0 and step % 495 == 0:
                model.reset_cache()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                print(f'  [flush] step={step}: non-continuity caches flushed')
    except KeyboardInterrupt:
        print('\n[EVA] Ctrl+C detected - keeping last best.pt (no separate checkpoint written)')
        print('[EVA] Exiting gracefully.')
        sys.exit(0)
    
    print('Training complete!')


@torch.no_grad()
def evaluate(model, streams, cfg, device, hold_n=None):
    if hold_n is None:  # M13 default (mirrors the train-sampler split)
        hold_n = 3 if len(streams) >= 8 else (1 if streams else 0)
    model.eval()
    # Audit M8: (a) `len(stream)` crashed — TokenStream exposes `.len` only,
    # so train.py died at its FIRST eval (eval_interval≈233); (b) val
    # forwards mutate the streaming runtime buffers (memory banks, intent bus,
    # EMAs) — snapshot/restore isolates validation documents from the training
    # working memory; (c) adaptive=False matches the notebook (controller
    # buffers must not learn from val); (d) fresh state per batch.
    _rt_snap = model.snapshot_runtime_buffers()
    _lc = getattr(model, 'logit_cache', None)
    if _lc is not None:
        _lc.cache.clear()  # val windows must not enter the train cache (decision #3)
    # M32: the chain is reset PER HOLD-OUT DOCUMENT below (a global reset
    # would let hold-out file 1's deliberation leak into file 2's).
    total_loss = 0.0
    total_steps = 0
    
    # Hold-out eval (audit M13): average over the last `_hold_n` files, each
    # read from its 3/4-region; per-file budget divides 100 batches.
    # B12 (F3-01): with an empty pool `streams[-1]` raised IndexError before
    # the NaN path could ever be reached; and 0-steps used to return 0.0.
    if not streams:
        model.train()
        model.restore_runtime_buffers(_rt_snap)
        return float('nan')
    eval_pool = streams[-hold_n:] if hold_n >= 1 else [streams[-1]]
    for stream in eval_pool:
        if stream.len < cfg.batch_size * cfg.seq_len + 1:
            continue
        # M32: each hold-out file is its own document — fresh deliberation
        # chain and fresh streaming state at its boundary, then windows CARRY
        # it, exactly mirroring the training sampler's document semantics.
        if getattr(model, 'explicit_reasoning', False):
            model.reset_reasoning()
        if getattr(model, 'memory_bank', None) is not None:   # M33: fresh bank per doc
            model.memory_bank.reset()
        est = ogs = None
        offset = max(stream.len // 2, cfg.batch_size * cfg.seq_len + 1)
        for _ in range(max(min(100 // max(hold_n, 1),
                               stream.len // (cfg.batch_size * cfg.seq_len)), 1)):
            x, y, offset, wrapped = stream.get_batch(cfg.seq_len, cfg.batch_size, offset, cfg.vocab)
            if wrapped:
                break  # end of the hold-out document region — no wrapped re-read
            x, y = x.to(device), y.to(device)
            h = model.embed_tokens(x)
            out, est, ogs, _ = model(h, est, global_state=ogs, adaptive=False, tokens=x)
            loss = model.compute_loss(out, y, h_emb=h)
            total_loss += loss.item()
            total_steps += 1
    
    if _lc is not None:
        _lc.cache.clear()
    model.restore_runtime_buffers(_rt_snap)
    model.train()
    # B12 (F3-01): an empty pool returned 0.0 — a PERFECT score that anchored
    # _best_val_loss=0 forever and ratcheted LR to its floor with an
    # unreachable recovery branch. NaN = 'no measurement'.
    return (total_loss / total_steps) if total_steps else float('nan')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=24)
    parser.add_argument('--D', type=int, default=4096, help='model width')
    parser.add_argument('--vocab', type=int, default=65536)
    parser.add_argument('--mirror-k', type=int, default=32)
    parser.add_argument('--llrd', type=float, default=1.0, help='RETIRED index-based per-depth LR decay — tau-LLRD (apply_tau_lr) is the single source (MATHEMATICAL_ANALYSIS R5); set !=1.0 only to reproduce legacy arms')
    parser.add_argument('--init-active-layers', type=int, default=8, help='blocks trained from step 0 (rest frozen)')
    parser.add_argument('--stage-steps', type=int, default=15000, help='unlock next block every N steps (backstop)')
    parser.add_argument('--readiness-full', type=float, default=0.6, help='meta-maturity (differentiation) to unlock deepest block')
    parser.add_argument('--stage-mode', type=str, default='readiness', choices=['readiness', 'fixed'])
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
        llrd=args.llrd,
        init_active_layers=args.init_active_layers,
        stage_steps=args.stage_steps,
        readiness_full=args.readiness_full,
        stage_mode=args.stage_mode,
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
    # B12 (F3-02): __post_init__ recomputes schedule fields from the lambda-domain
    # and clobbers the constructor kwargs (--warmup 500 arrived as 101). The CLI
    # is the user; it must win. Apply AFTER construction.
    cfg.warmup_steps = args.warmup
    cfg.log_interval = args.log_interval
    cfg.eval_interval = args.eval_interval

    # B19 fix (M58c): this block was a MISPLACED copy of the OOM reaction —
    # it ran at startup, so every run that already had gradient_checkpointing
    # on (the production default) silently HALVED cfg.seq_len (224 -> 112).
    # The config is the user's; the OOM ladder reacts at runtime if at all.
