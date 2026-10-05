"""T4 runner for EVA-CLM: bounded_residual production smoke + mid-scale A/B.

Modes:
  --dry-run  Production config taken from notebooks/eva_colab.ipynb cell 4
             (D=2560, 24 layers, seq 384, B=1, bounded_residual=True) with
             gradient_checkpointing=True and an AMP fp16 attempt. Runs a few
             steps on CUDA and reports peak VRAM, s/step, b_flow/b_drift and
             finiteness. If fp16 overflows on the first steps it auto-falls
             back to fp32 (recorded as amp_fallback=True in JSON). The point is
             to answer whether the production run fits a 16 GB T4 with gc=True
             and how fast it is (s/step > 30 => not worth T4 time).
  --ab       Mid-scale A/B: D=512, 8 layers, mlp_groups=8, seq 384, B=4,
             vocab=8192, gc=True, fp32 by doctrine (AMP only with explicit
             --amp), real data (3 train files + 1 hold-out). Arms:
             bounded_residual=True vs False (baseline), same data and seed, one
             run each. Metrics every ~200 steps: train CE, eval CE (2 windows),
             b_flow/b_drift (bounded), s/step. Results in JSON + a short txt
             comparison. If loss/grad is non-finite within the first
             NAN_EARLY_STEPS steps the arm stops immediately as status
             failed_nan (no zero-substitution) and the next arm runs.
  --smoke    Same pipeline as --ab at D=64, 2 layers, vocab=256, 20 steps --
             the CPU end-to-end check (bounded flag, telemetry, saving).

The mini arms accelerate maturation (matur_T0=0, T_delay=0, delta=1000,
mem_min_write_mat=0.1) like the calibrated stand scripts/bench_calibrated.py:
with the production T0=8000 a <=1200-step run never reaches the memory-bank
injection path, so bounded and baseline would be identical by construction.

Paths default to Colab (/content/...), falling back to <repo>/wb and
<repo>/logs/t4. CPU runs are fp32 (AMP is CUDA-only, opt-in via --amp;
--dry-run keeps its fp16 attempt). --dry-run requires CUDA.
Nothing is committed; outputs go to --out-dir/--out.
"""

from __future__ import annotations

import argparse
import gc as _gc
import glob
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch

torch.set_num_threads(4)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core import EVAConfig, EVAStack
from core.adaptation import (GradientClipper, build_optimizer,
                             nonfinite_gradient_names)
from core.training_control import LossBalancer, apply_tau_lr, training_telemetry

try:
    from torch.utils.checkpoint import CheckpointError
except Exception:
    CheckpointError = RuntimeError

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IS_CUDA = DEVICE == 'cuda'

NAN_EARLY_STEPS = 10  # non-finite loss/grad in the first N steps => failed_nan


def fmt_num(v, nd=4, none='n/a'):
    """Format a possibly-None/NaN/Inf number for logs without raising."""
    if v is None:
        return none
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return none


def nan_advice(use_amp, dry_run=False):
    """Human hint printed when an early non-finite loss/grad stops a run."""
    if use_amp:
        if dry_run:
            return 'AMP-overflow, попробуй без --amp'
        return 'fp16 overflow suspected: rerun with --no-amp (fp32 is the default)'
    return 'fp32 run produced non-finite loss/grad: check data/lr/grad-clip before rerun'


def reset_run_state(model, ctx):
    """Drop carried state after a non-finite step or a chunk switch."""
    ctx['state'] = None
    ctx['gs'] = None
    model.reset_streams()
    if getattr(model, 'memory_bank', None) is not None:
        model.memory_bank.reset()
    if getattr(model, 'explicit_reasoning', False):
        model.reset_reasoning()
    lc = getattr(model, 'logit_cache', None)
    if lc is not None:
        try:
            lc.cache.clear()
        except Exception:
            pass


def resolve_data_dir(arg):
    cands = []
    if arg:
        cands.append(arg)
    if os.environ.get('EVA_DATA_DIR'):
        cands.append(os.environ['EVA_DATA_DIR'])
    cands.append(r'/content/drive/MyDrive/eva_clm/data')
    cands.append(os.path.join(_REPO, 'wb'))
    for c in cands:
        if c and os.path.isdir(c) and _variant_files(c):
            return os.path.abspath(c)
    return os.path.abspath(cands[-1])


def resolve_out_dir(arg):
    if arg:
        return os.path.abspath(arg)
    if os.name != 'nt' and os.path.isdir('/content'):
        return '/content/t4_out'
    return os.path.join(_REPO, 'logs', 't4')


def _variant_files(data_dir):
    for pat in ('token_stream_*_eos.bin', 'token_stream_*_clean.bin',
                'token_stream_*.bin'):
        fs = sorted(glob.glob(os.path.join(data_dir, pat)))
        if fs:
            return fs
    return []


def _filter_file(path, vocab, cap):
    arr = np.memmap(path, dtype=np.uint16, mode='r')
    out = np.empty(cap, dtype=np.uint16)
    n = 0
    raw = 0
    raw_cap = min(len(arr), max(int(cap * 3) + 1, cap))
    step = 1 << 20
    while n < cap and raw < raw_cap:
        chunk = np.asarray(arr[raw:raw + step], dtype=np.uint16)
        raw += len(chunk)
        keep = chunk[chunk < vocab]
        take = min(len(keep), cap - n)
        out[n:n + take] = keep[:take]
        n += take
    return out[:n]


def load_mini_data(data_dir, vocab, train_count, holdout_name, cap):
    files = _variant_files(data_dir)
    if len(files) < 2:
        raise SystemExit(f'need >=2 token_stream files in {data_dir!r}, found {len(files)}')
    if holdout_name:
        hf = [f for f in files if os.path.basename(f) == holdout_name]
        if not hf:
            raise SystemExit(f'holdout {holdout_name!r} not found in {data_dir!r}')
        hf = hf[-1]
        train_files = [f for f in files if f != hf][:train_count]
    else:
        hf = files[-1]
        train_files = files[:-1][:train_count]
    if not train_files:
        raise SystemExit('no train files left after the hold-out split')
    train = [_filter_file(f, vocab, cap) for f in train_files]
    hold = _filter_file(hf, vocab, cap)
    if any(len(b) < 4 * 384 + 2 for b in train) or len(hold) < 4 * 384 + 2:
        raise SystemExit('filtered buffers too small for the chosen vocab; lower --vocab or --cap')
    return {
        'train': train,
        'hold': hold,
        'train_files': [os.path.basename(f) for f in train_files],
        'holdout_file': os.path.basename(hf),
    }


def build_prod_cfg(args):
    data_dir = resolve_data_dir(args.data_dir)
    out_dir = resolve_out_dir(args.out_dir)
    cfg = EVAConfig(
        D=2560, n_layers=24, bind_K=32,
        code_dim=64, codebook='twin_free', vocab=65536, mask_eos=False,
        mlp_groups=32, mlp_expand=4,
        batch_size=1, seq_len=384,
        lr=6e-4, max_steps=150_000,
        warmup_steps=300, log_interval=55, eval_interval=440,
        optimizer='eva_proj', scheduler='mirror',
        lr_boost_max=2.0, lr_improve_tol=0.002,
        per_layer_ls_lr=True, ls_ema_fast=0.99, ls_ema_slow=0.999,
        ls_mult_min=0.5, ls_mult_max=2.0, ls_mirror_mult_max=2.0,
        private_mem=True, expert_asymmetry=True, meta_trust=True,
        conv_kernel=48,
        gradient_checkpointing=True,
        head_mode='sigmoid_coded', embed_center=False, head_normalize=True,
        head_u_wall=1e-3, head_phantom_bits=32, head_phantom_noise=0.05,
        head_srl=True, head_srl_steps=2, head_srl_after=1045,
        head_phantom_slots=16,
        ucl_read_scale_floor=0.1, ucl_read_scale_floor_until=5000,
        head_phantom_max=64,
        stream_chunk_steps=1000,
        bind_twist_mode='trajectory_spiral', bind_traj_dims=3,
        hybrid_alpha_max=0.7, hybrid_alpha_min=0.3,
        w_pred_scale_init=3.0, bind_twist_gate=True,
        collective_layer=True, collective_layer_idx=None,
        collective_read_out=True, collective_uncert_theta=0.5,
        collective_uncert_kappa=3.0, collective_contra_thresh=-0.1,
        collective_contra_gain=6.0, collective_maturity_thresh=0.12,
        surprisal_weight=0.3, branch_balance_weight=0.1,
        variable_precision=True, precision_threshold=0.3,
        explicit_reasoning=True, reasoning_max_steps=8, reasoning_adaptive=True,
        use_amp=True,
        intent_bridge=True, bridge_glu=True, bridge_conn=0.1,
        maturation_enabled=True, pm_write_delay=0, intent_topdown=True,
        memory_bank=True, mem_l1_slots=3, mem_l2_slots=32, mem_min_write_mat=0.3,
        mem_bridge_dim=256, concept_birth_novelty_threshold=0.15,
        unified_concept_layer=True, unified_concept_S=8,
        logit_cache_enabled=True, logit_cache_max_entries=12,
        logit_cache_n_heads=8, logit_cache_scheduled_sampling=0.05,
        logit_cache_reset_on_resume=True,
        orth_weight=0.0, div_weight=10.0,
        aux_kill_switch=True, aux_kill_disable=True,
        balancer_align_every=8,
        readout_lr_mult=1.0,
        logit_cache_kv_dim=64,
        logit_cache_ms_spans='8,32,128,512,2048,8192',
        logit_cache_ms_max=16,
        head_phantom_thr_mode='noise', head_phantom_thr_k=2.0,
        head_phantom_thr_floor=0.002,
        head_u_clamp=8.0, phantom_lacuna_ema=0.9,
        contradiction_field=True,
        bounded_residual=True,
        data_dir=data_dir, save_dir=os.path.join(out_dir, 'ckpt'),
        log_dir=os.path.join(out_dir, 'logs'),
    )
    cfg.log_interval = 55
    cfg.eval_interval = 440
    cfg.eval_windows_budget = 64
    cfg.head_lacuna_ladder = tuple(
        float(s) for s in str(cfg.logit_cache_ms_spans).split(',') if s)
    return cfg


MINI_SHAPES = {
    'smoke': dict(D=64, n_layers=2, mlp_groups=8, vocab=256,
                  batch_size=4, seq_len=384),
    'ab': dict(D=512, n_layers=8, mlp_groups=8, vocab=8192,
               batch_size=4, seq_len=384),
}


def build_mini_cfg(scale, bounded, steps, eval_every):
    sh = MINI_SHAPES[scale]
    cfg = EVAConfig(
        D=sh['D'], n_layers=sh['n_layers'], mlp_groups=sh['mlp_groups'],
        vocab=sh['vocab'], batch_size=sh['batch_size'], seq_len=sh['seq_len'],
        code_dim=32, code_sparsity=6, codebook='legacy',
        lr=6e-4, warmup_steps=20, max_steps=steps,
        lambda_d_enabled=False,
        optimizer='eva_proj',
        gradient_checkpointing=True,
        memory_bank=True, mem_l1_slots=3, mem_l2_slots=32, mem_min_write_mat=0.1,
        matur_T0=0.0, matur_T_delay=0.0, matur_delta=1000.0,
        stream_chunk_steps=0,
        logit_cache_max_entries=8,
        balancer_align_every=8,
        head_srl=True, head_srl_steps=2,
        head_u_clamp=8.0, phantom_lacuna_ema=0.9, contradiction_field=True,
        bounded_residual=bool(bounded),
    )
    cfg.log_interval = 25
    cfg.eval_interval = eval_every
    return cfg


_BOUNDED_ONLY_KEYS = ('memory_bank.gate_W', 'memory_bank.gate_b')


def core_fp(model):
    h = hashlib.sha1()
    for k, v in sorted(model.state_dict().items()):
        if 'bounded' in k or k in _BOUNDED_ONLY_KEYS:
            continue
        if v.is_floating_point():
            h.update(k.encode())
            h.update(v.detach().to('cpu', torch.float32).numpy().tobytes())
    return h.hexdigest()[:12]


def detach_tree(o):
    if isinstance(o, torch.Tensor):
        return o.detach()
    if isinstance(o, (list, tuple)):
        return [detach_tree(x) for x in o]
    return o


def trust_snapshot(model):
    t = {}
    if getattr(model, 'bridge', None) is not None:
        try:
            t['bridge'] = float(model.bridge.readiness())
        except Exception:
            pass
    if getattr(model, 'maturation', None) is not None:
        try:
            rm = abs(float(model.maturation.gate.float().mean().detach()))
            t['intent'] = rm
            t['mem'] = rm
        except Exception:
            pass
    return t


def build_optimizer_for(model, cfg):
    return build_optimizer(model, cfg.lr, llrd_decay=1.0,
                           weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
                           optimizer=getattr(cfg, 'optimizer', 'adamw'))


def train_step(model, opt, clip, bal, scaler, ctx, x, y, step, use_amp, cfg,
               check_grad_finite=False):
    model.train()
    if getattr(model, 'explicit_reasoning', False):
        model.reasoning_enabled_step = step
    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        h = model.embed_tokens(x)
        out, state, gs, _ = model(h, ctx['state'], global_state=ctx['gs'],
                                  step=step, tokens=x)
        model.observe_output(model.lm_head(out))
        ce, aux = model.compute_losses(out, y, h_emb=h)
    ctx['state'] = detach_tree(state)
    ctx['gs'] = gs.detach() if isinstance(gs, torch.Tensor) else None
    if not bool(torch.isfinite(ce.detach())):
        opt.zero_grad(set_to_none=True)
        model.release_step_graph()
        return 'nonfinite_loss', float(ce.detach())
    gscale = scaler.get_scale() if use_amp else 1.0
    ce_s = ce * gscale
    aux_s = {k: (v * gscale if isinstance(v, torch.Tensor) else v)
             for k, v in aux.items()}
    try:
        bal.backward(ce_s, aux_s, model.parameters(), phase_model=model, step=step)
    except CheckpointError as e:
        print(f'  [ckpt-fallback] {str(e)[:100]} -> gc OFF, retry', flush=True)
        cfg.gradient_checkpointing = False
        for l in getattr(model, 'layers', []):
            if hasattr(l, '_ga_record'):
                l._ga_record = True
        opt.zero_grad(set_to_none=True)
        model.release_step_graph()
        if IS_CUDA:
            torch.cuda.empty_cache()
        return 'checkpoint_error', float(ce.detach())
    if use_amp:
        scaler.unscale_(opt)
    if check_grad_finite:
        bad_grads = nonfinite_gradient_names(model)
        if bad_grads:
            opt.zero_grad(set_to_none=True)
            model.release_step_graph()
            return 'nonfinite_grad', float(ce.detach())
    apply_tau_lr(model, getattr(model, 'tau_config', None), None)
    clip.clip(model.parameters())
    if hasattr(opt, 'set_trust'):
        opt.set_trust(trust_snapshot(model))
    if use_amp:
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < before
    else:
        opt.step()
        skipped = False
    opt.zero_grad(set_to_none=True)
    model.release_step_graph()
    return ('amp_skip' if skipped else 'ok'), float(ce.detach())


@torch.no_grad()
def evaluate_mini(model, hold, cfg, windows, step):
    model.eval()
    snap = None
    try:
        snap = model.snapshot_runtime_buffers()
    except Exception:
        pass
    lc = getattr(model, 'logit_cache', None)
    B, S = cfg.batch_size, cfg.seq_len
    need = B * S + 1
    off0 = max(len(hold) // 3, need + 1)
    ces = []
    for w in range(windows):
        off = off0 + w * B * S
        if off + need > len(hold):
            off = 0
        ch = np.asarray(hold[off:off + need])
        x = torch.from_numpy(ch[:-1].copy()).long().view(B, S).to(DEVICE)
        y = torch.from_numpy(ch[1:].copy()).long().view(B, S).to(DEVICE)
        if getattr(model, 'explicit_reasoning', False):
            model.reset_reasoning()
        if getattr(model, 'memory_bank', None) is not None:
            model.memory_bank.reset()
        model.reset_streams()
        if lc is not None:
            try:
                lc.cache.clear()
            except Exception:
                pass
        h = model.embed_tokens(x)
        out, _, _, _ = model(h, None, global_state=None, adaptive=False,
                             tokens=x, step=step)
        ce, _ = model.compute_losses(out, y, h_emb=h)
        ces.append(float(ce.detach()))
    if snap is not None:
        try:
            model.restore_runtime_buffers(snap)
        except Exception:
            pass
    model.train()
    return sum(ces) / max(len(ces), 1)


def run_arm(scale, bounded, args, data, deadline, use_amp):
    steps = args.steps
    eval_every = args.eval_every
    cfg = build_mini_cfg(scale, bounded, steps, eval_every)
    torch.manual_seed(args.seed)
    if IS_CUDA:
        torch.cuda.manual_seed_all(args.seed)
    model = EVAStack(cfg).to(DEVICE)
    params = model.param_count()
    fp = core_fp(model)
    print(f'[arm {"bounded" if bounded else "baseline"}] params={params:,} '
          f'init_fp={fp} bounded={bool(cfg.bounded_residual)}', flush=True)
    opt = build_optimizer_for(model, cfg)
    bal = LossBalancer(align=True, align_cap=10.0, eval_interval=eval_every,
                       align_every=int(getattr(cfg, 'balancer_align_every', 1) or 1),
                       safety_aux=('head_wall',))
    clip = GradientClipper(c=0.1)
    clip.attach(model)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    rng = np.random.default_rng(args.seed)
    bufs = data['train']
    need = cfg.batch_size * cfg.seq_len + 1
    bi = int(rng.integers(0, len(bufs)))
    offset = 0
    ctx = {'state': None, 'gs': None}
    ce_all, window_ce, history = [], [], []
    status_counts = {}
    t0 = time.time()
    step = 0
    retries = 0
    stopped_early = False
    chunks = 1
    failed_nan = False
    nan_status = None
    nan_step = None
    while step < steps:
        if time.time() > deadline:
            stopped_early = True
            break
        if offset + need > len(bufs[bi]):
            bi = int(rng.integers(0, len(bufs)))
            offset = 0
            chunks += 1
            reset_run_state(model, ctx)
        ch = np.asarray(bufs[bi][offset:offset + need])
        offset += cfg.batch_size * cfg.seq_len
        x = torch.from_numpy(ch[:-1].copy()).long().view(cfg.batch_size, cfg.seq_len).to(DEVICE)
        y = torch.from_numpy(ch[1:].copy()).long().view(cfg.batch_size, cfg.seq_len).to(DEVICE)
        status, ce = train_step(model, opt, clip, bal, scaler, ctx, x, y,
                                step, use_amp, cfg,
                                check_grad_finite=step < NAN_EARLY_STEPS)
        if status == 'checkpoint_error':
            retries += 1
            if retries > 2:
                status_counts['checkpoint_error'] = status_counts.get('checkpoint_error', 0) + 1
                break
            continue
        retries = 0
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in ('nonfinite_loss', 'nonfinite_grad') and step < NAN_EARLY_STEPS:
            failed_nan = True
            nan_status = status
            nan_step = step
            arm_name = 'bounded' if bounded else 'baseline'
            print(f'[arm {arm_name}] FAILED_NAN at step {step}/{steps}: {status} '
                  f'within the first {NAN_EARLY_STEPS} steps (no zero-substitution).',
                  flush=True)
            print(f'[arm {arm_name}] advice: {nan_advice(use_amp)}', flush=True)
            break
        step += 1
        if status in ('ok', 'amp_skip') and ce is not None:
            ce_all.append(ce)
            window_ce.append(ce)
        if step % eval_every == 0 or step == steps:
            val = evaluate_mini(model, data['hold'], cfg, args.windows, step)
            tel = training_telemetry(model)
            wall = time.time() - t0
            rec = {
                'step': step,
                'train_ce': (sum(window_ce) / len(window_ce)) if window_ce else None,
                'eval_ce': val,
                's_per_step': wall / max(step, 1),
                'tok_per_s': step * cfg.batch_size * cfg.seq_len / max(wall, 1e-9),
                'b_flow': tel.get('b_flow'),
                'b_flow_max': tel.get('b_flow_max'),
                'b_drift': tel.get('b_drift'),
            }
            history.append(rec)
            window_ce = []
            print(f'  [arm {"B" if bounded else "A"}] step={step}/{steps} '
                  f'train_ce={fmt_num(rec["train_ce"])} eval_ce={fmt_num(val)} '
                  f's/step={fmt_num(rec["s_per_step"], 2)} '
                  f'b_flow={fmt_num(rec["b_flow"], 3)} '
                  f'b_drift={fmt_num(rec["b_drift"], 4)}',
                  flush=True)
    wall = time.time() - t0
    del model, opt, bal, clip, scaler
    _gc.collect()
    if IS_CUDA:
        torch.cuda.empty_cache()
    ce_nums = [c for c in ce_all if c is not None]
    last_n = max(1, min(50, len(ce_nums) // 2)) if ce_nums else 0
    if failed_nan:
        run_status = 'failed_nan'
    elif stopped_early:
        run_status = 'timeout'
    else:
        run_status = 'ok'
    result = {
        'bounded': bool(bounded),
        'params': params,
        'init_fp': fp,
        'status': run_status,
        'failed_nan': failed_nan,
        'nan_status': nan_status,
        'nan_step': nan_step,
        'use_amp': bool(use_amp),
        'steps_done': step,
        'steps_target': steps,
        'stopped_early': stopped_early,
        'wall_s': round(wall, 1),
        's_per_step': wall / max(step, 1),
        'tok_per_s': step * cfg.batch_size * cfg.seq_len / max(wall, 1e-9),
        'chunks_seen': chunks,
        'status_counts': status_counts,
        'train_ce_first50': (sum(ce_nums[:last_n]) / last_n) if last_n else None,
        'train_ce_last50': (sum(ce_nums[-last_n:]) / last_n) if last_n else None,
        'train_ce_min': min(ce_nums) if ce_nums else None,
        'history': history,
        'cfg': {
            'D': cfg.D, 'n_layers': cfg.n_layers, 'mlp_groups': cfg.mlp_groups,
            'vocab': cfg.vocab, 'seq_len': cfg.seq_len,
            'batch_size': cfg.batch_size, 'lr': cfg.lr,
            'gradient_checkpointing': bool(cfg.gradient_checkpointing),
            'matur_T0': cfg.matur_T0, 'matur_T_delay': cfg.matur_T_delay,
            'matur_delta': cfg.matur_delta,
            'mem_min_write_mat': cfg.mem_min_write_mat,
        },
        'data': {'train_files': data['train_files'],
                 'holdout_file': data['holdout_file'],
                 'train_tokens': [int(len(b)) for b in data['train']],
                 'hold_tokens': int(len(data['hold']))},
    }
    return result


def summarize_ab(results, args, out_txt):
    arms = results['arms']
    b = arms['bounded']
    a = arms['baseline']
    def _fmt(v, nd=4):
        return fmt_num(v, nd=nd)
    lines = []
    lines.append('T4 bounded_residual A/B summary')
    lines.append(f'device={DEVICE} amp={results["use_amp"]} seed={args.seed} '
                 f'steps_target={args.steps} eval_every={args.eval_every} windows={args.windows}')
    lines.append(f'data: train={b["data"]["train_files"]} holdout={b["data"]["holdout_file"]} '
                 f'cap_per_file={args.cap}')
    lines.append(f'init_fp: bounded={b["init_fp"]} baseline={a["init_fp"]} '
                 f'match={b["init_fp"] == a["init_fp"]}')
    lines.append('')
    lines.append(f'{"metric":<24}{"bounded":>14}{"baseline":>14}')
    lines.append(f'{"params":<24}{b["params"]:>14,}{a["params"]:>14,}')
    lines.append(f'{"steps_done":<24}{b["steps_done"]:>14}{a["steps_done"]:>14}')
    lines.append(f'{"status":<24}{str(b.get("status", "ok")):>14}{str(a.get("status", "ok")):>14}')
    lines.append(f'{"stopped_early":<24}{str(b["stopped_early"]):>14}{str(a["stopped_early"]):>14}')
    lines.append(f'{"use_amp":<24}{str(b.get("use_amp", results.get("use_amp"))):>14}{str(a.get("use_amp", results.get("use_amp"))):>14}')
    lines.append(f'{"s_per_step":<24}{_fmt(b["s_per_step"], 2):>14}{_fmt(a["s_per_step"], 2):>14}')
    lines.append(f'{"tok_per_s":<24}{_fmt(b["tok_per_s"], 0):>14}{_fmt(a["tok_per_s"], 0):>14}')
    lines.append(f'{"train_ce_first50":<24}{_fmt(b["train_ce_first50"]):>14}{_fmt(a["train_ce_first50"]):>14}')
    lines.append(f'{"train_ce_last50":<24}{_fmt(b["train_ce_last50"]):>14}{_fmt(a["train_ce_last50"]):>14}')
    lines.append(f'{"train_ce_min":<24}{_fmt(b["train_ce_min"]):>14}{_fmt(a["train_ce_min"]):>14}')
    ev_b = [r['eval_ce'] for r in b['history'] if r.get('eval_ce') is not None]
    ev_a = [r['eval_ce'] for r in a['history'] if r.get('eval_ce') is not None]
    lines.append(f'{"eval_ce_first":<24}{_fmt(ev_b[0] if ev_b else None):>14}{_fmt(ev_a[0] if ev_a else None):>14}')
    lines.append(f'{"eval_ce_last":<24}{_fmt(ev_b[-1] if ev_b else None):>14}{_fmt(ev_a[-1] if ev_a else None):>14}')
    lines.append(f'{"eval_ce_min":<24}{_fmt(min(ev_b) if ev_b else None):>14}{_fmt(min(ev_a) if ev_a else None):>14}')
    bf = [r['b_flow'] for r in b['history'] if r['b_flow'] is not None]
    bd = [r['b_drift'] for r in b['history'] if r['b_drift'] is not None]
    lines.append(f'{"b_flow_mean/max":<24}{("%s/%s" % (_fmt(sum(bf)/len(bf), 3) if bf else "n/a", _fmt(max(bf), 3) if bf else "n/a")):>14}{"n/a (off)":>14}')
    lines.append(f'{"b_drift_mean/max":<24}{("%s/%s" % (_fmt(sum(bd)/len(bd), 4) if bd else "n/a", _fmt(max(bd), 4) if bd else "n/a")):>14}{"n/a (off)":>14}')
    lines.append('')
    lines.append('eval_ce curves (step: bounded / baseline)')
    by_step = {r['step']: r['eval_ce'] for r in b['history']}
    ay_step = {r['step']: r['eval_ce'] for r in a['history']}
    for s in sorted(set(by_step) | set(ay_step)):
        lines.append(f'  {s:>6}: {_fmt(by_step.get(s))} / {_fmt(ay_step.get(s))}')
    lines.append('')
    delta = None
    if ev_b and ev_a:
        delta = ev_a[-1] - ev_b[-1]
        lines.append(f'final eval delta (baseline - bounded) = {delta:+.4f} nat '
                     f'(positive = bounded better)')
        if delta > 0.01:
            verdict = f'bounded better on this D={b["cfg"]["D"]} run (single seed; check scale transfer)'
        elif delta < -0.01:
            verdict = f'baseline better on this D={b["cfg"]["D"]} run (bounded may hurt at this scale)'
        else:
            verdict = 'inconclusive (|delta| <= 0.01 nat, single seed, short run)'
        if b['stopped_early'] or a['stopped_early']:
            verdict += ' [STOPPED EARLY by --max-minutes]'
        lines.append(f'VERDICT: {verdict}')
    failed = [name for name, arm in (('bounded', b), ('baseline', a))
              if arm.get('failed_nan') or arm.get('status') == 'failed_nan']
    if failed:
        lines.append('')
        lines.append(f'EARLY-NaN STOP: {", ".join(failed)} stopped in the first '
                     f'{NAN_EARLY_STEPS} steps (non-finite loss/grad); '
                     f'metrics are NOT zero-substituted (n/a).')
        lines.append('advice: ' + nan_advice(results.get('use_amp', False)))
        if not (ev_b and ev_a):
            lines.append('VERDICT: no A/B verdict (an arm failed on early NaNs)')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return '\n'.join(lines), delta


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return path


def run_ab(args, use_amp, scale='ab'):
    data_dir = resolve_data_dir(args.data_dir)
    out_dir = resolve_out_dir(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    data = load_mini_data(data_dir, MINI_SHAPES[scale]['vocab'],
                          args.train_count, args.holdout, args.cap)
    print(f'[{scale}] data_dir={data_dir}')
    print(f'[{scale}] train={data["train_files"]} hold={data["holdout_file"]} '
          f'tokens={[int(len(b)) for b in data["train"]]}+{int(len(data["hold"]))}', flush=True)
    t_end = time.time() + args.max_minutes * 60
    results = {'mode': scale, 'seed': args.seed, 'use_amp': use_amp,
               'config': vars(args), 'data_dir': data_dir,
               'arms': {}}
    arms_left = 2
    for bounded in (True, False):
        key = 'bounded' if bounded else 'baseline'
        arm_deadline = time.time() + max(60.0, (t_end - time.time()) / arms_left)
        results['arms'][key] = run_arm(scale, bounded, args, data,
                                       arm_deadline, use_amp)
        arms_left -= 1
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_json = args.out or os.path.join(out_dir, f't4_{scale}_{ts}.json')
    out_txt = os.path.splitext(out_json)[0] + '.txt'
    text, delta = summarize_ab(results, args, out_txt)
    results['comparison'] = {
        'init_fp_match': results['arms']['bounded']['init_fp'] == results['arms']['baseline']['init_fp'],
        'final_eval_delta_baseline_minus_bounded': delta,
        'failed_nan_arms': [k for k, v in results['arms'].items() if v.get('failed_nan')],
    }
    save_json(out_json, results)
    print('\n' + text)
    print(f'\n[ab] JSON: {out_json}\n[ab] TXT:  {out_txt}')
    return 0


def run_dry_run(args, use_amp):
    out_dir = resolve_out_dir(args.out_dir)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_json = args.out or os.path.join(out_dir, f't4_dryrun_{ts}.json')
    result = {'mode': 'dry-run', 'ok': False, 'oom': False, 'oom_step': None,
              'steps_target': args.steps, 'steps_done': 0, 'use_amp': use_amp,
              'amp_fallback': False, 'failed_nan': False, 'device': DEVICE}
    if not IS_CUDA:
        result['error'] = 'cuda_unavailable'
        save_json(out_json, result)
        print(f'[dry-run] CUDA is not available on this host; nothing to measure. '
              f'Run it on the T4. JSON: {out_json}')
        return 2
    data_dir = resolve_data_dir(args.data_dir)
    files = _variant_files(data_dir)
    if not files:
        result['error'] = f'no token_stream files in {data_dir!r}'
        save_json(out_json, result)
        print(f'[dry-run] {result["error"]}')
        return 2
    data_file = args.holdout or os.path.basename(files[0])
    dpath = os.path.join(data_dir, data_file)
    if not os.path.exists(dpath):
        dpath = files[0]
    print(f'[dry-run] data={dpath}')
    cfg = build_prod_cfg(args)
    print(f'[dry-run] prod cfg: D={cfg.D} L={cfg.n_layers} seq={cfg.seq_len} '
          f'B={cfg.batch_size} gc={cfg.gradient_checkpointing} '
          f'bounded={cfg.bounded_residual} kv_dim={cfg.logit_cache_kv_dim} '
          f'ms={cfg.logit_cache_ms_spans}', flush=True)
    model = None
    try:
        try:
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
        except Exception:
            pass
        t_build = time.time()
        torch.manual_seed(1234)
        model = EVAStack(cfg)
        n_params = model.param_count()
        model = model.to(DEVICE)
        result['params'] = n_params
        result['build_s'] = round(time.time() - t_build, 1)
        print(f'[dry-run] built {n_params:,} params in {result["build_s"]:.1f}s '
              f'({torch.cuda.memory_allocated()/1e9:.2f} GB allocated)', flush=True)
        opt = build_optimizer_for(model, cfg)
        bal = LossBalancer(align=True, align_cap=10.0, eval_interval=440,
                           align_every=int(getattr(cfg, 'balancer_align_every', 1) or 1),
                           safety_aux=('head_wall',))
        clip = GradientClipper(c=0.1)
        clip.attach(model)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        arr = np.memmap(dpath, dtype=np.uint16, mode='r')
        B, S = cfg.batch_size, cfg.seq_len
        need = B * S + 1
        offset = 0
        ctx = {'state': None, 'gs': None}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times, ces, bflows, bdrifts = [], [], [], []
        n_skips = 0
        amp_fallback = False
        for step in range(args.steps):
            try:
                if offset + need > len(arr):
                    offset = 0
                    reset_run_state(model, ctx)
                ch = np.asarray(arr[offset:offset + need])
                offset += B * S
                x = torch.from_numpy(ch[:-1].copy()).long().view(B, S).to(DEVICE)
                y = torch.from_numpy(ch[1:].copy()).long().view(B, S).to(DEVICE)
                t = time.time()
                status, ce = train_step(model, opt, clip, bal, scaler, ctx, x, y,
                                        step, use_amp, cfg,
                                        check_grad_finite=step < NAN_EARLY_STEPS)
                torch.cuda.synchronize()
                if status in ('nonfinite_loss', 'nonfinite_grad'):
                    if use_amp and not amp_fallback and step < NAN_EARLY_STEPS:
                        amp_fallback = True
                        use_amp = False
                        scaler = torch.amp.GradScaler('cuda', enabled=False)
                        reset_run_state(model, ctx)
                        result['amp_fallback'] = True
                        result['amp_fallback_step'] = step
                        result['amp_fallback_reason'] = f'{status} at step {step}'
                        result['use_amp'] = False
                        print(f'[dry-run] AMP-overflow: {status} at step {step} -> '
                              f'auto-fallback to fp32 ({nan_advice(True, dry_run=True)}); '
                              f'JSON marks amp_fallback=True', flush=True)
                        continue
                    if step < NAN_EARLY_STEPS:
                        result['failed_nan'] = True
                        result['nan_status'] = status
                        result['nan_step'] = step
                        result['error'] = f'{status} at step {step} (early NaN stop)'
                        print(f'[dry-run] FAILED_NAN: {status} at step {step} '
                              f'({nan_advice(use_amp, dry_run=True)}); '
                              f'no zero-substitution, values stay n/a', flush=True)
                        break
                times.append(time.time() - t)
                if status in ('ok', 'amp_skip') and ce is not None:
                    ces.append(ce)
                if status == 'amp_skip':
                    n_skips += 1
                tel = training_telemetry(model)
                if tel.get('b_flow') is not None:
                    bflows.append(tel['b_flow'])
                if tel.get('b_drift') is not None:
                    bdrifts.append(tel['b_drift'])
                result['steps_done'] = step + 1
                print(f'  step={step} ce={fmt_num(ce)} dt={times[-1]:.2f}s '
                      f'peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB '
                      f'live={torch.cuda.memory_allocated()/1e9:.2f}GB '
                      f'status={status}', flush=True)
            except torch.cuda.OutOfMemoryError as e:
                result['oom'] = True
                result['oom_step'] = step
                result['error'] = f'OOM: {str(e)[:200]}'
                print(f'  [OOM] step={step}: {str(e)[:160]}', flush=True)
                break
        result['ok'] = (result['steps_done'] >= max(1, args.steps // 2)
                        and not result['oom'] and not result.get('failed_nan'))
        result['s_per_step'] = (sum(times) / len(times)) if times else None
        result['s_per_step_min'] = min(times) if times else None
        result['final_ce'] = ces[-1] if ces else None
        result['loss_finite'] = all(math.isfinite(c) for c in ces) and bool(ces)
        result['amp_skips'] = n_skips
        result['peak_vram_gb'] = torch.cuda.max_memory_allocated() / 1e9
        result['peak_reserved_gb'] = torch.cuda.max_memory_reserved() / 1e9
        result['live_vram_gb'] = torch.cuda.memory_allocated() / 1e9
        result['b_flow_mean'] = (sum(bflows) / len(bflows)) if bflows else None
        result['b_flow_max'] = max(bflows) if bflows else None
        result['b_drift_mean'] = (sum(bdrifts) / len(bdrifts)) if bdrifts else None
        result['b_drift_max'] = max(bdrifts) if bdrifts else None
        try:
            result['params_finite'] = bool(torch.stack(
                [torch.isfinite(p).all() for p in model.parameters()]).all())
        except Exception:
            result['params_finite'] = None
        if result.get('failed_nan'):
            t4_budget_note = (
                f'T4 verdict: FAILED (early NaN stop at step {result.get("nan_step")}, '
                f'values not zero-substituted); '
                f'{nan_advice(result.get("use_amp", False), dry_run=True)}')
        else:
            t4_budget_note = (
                'T4 verdict: ' + ('FITS' if result['ok'] else 'DOES NOT FIT / FAILED') +
                ('' if result['s_per_step'] is None else
                 f'; s/step={result["s_per_step"]:.1f}'))
        if result.get('amp_fallback'):
            t4_budget_note += (
                f' [AMP fp16 overflowed at step {result.get("amp_fallback_step")}; '
                f'auto-fallback to fp32, JSON amp_fallback=True; '
                f'rerun with --no-amp to skip the fp16 attempt]')
        result['note'] = t4_budget_note
        print(f'\n[dry-run] {t4_budget_note}')
        print(f'[dry-run] peak_alloc={fmt_num(result["peak_vram_gb"], 2)} GB '
              f'peak_reserved={fmt_num(result["peak_reserved_gb"], 2)} GB '
              f's/step={fmt_num(result["s_per_step"], 2)} '
              f'loss_finite={result["loss_finite"]} params_finite={result["params_finite"]} '
              f'b_flow={fmt_num(result["b_flow_mean"], 3)} '
              f'b_drift={fmt_num(result["b_drift_mean"], 4)}')
    except torch.cuda.OutOfMemoryError as e:
        result['oom'] = True
        result['error'] = f'OOM: {str(e)[:200]}'
        print(f'[dry-run] OOM: {str(e)[:200]}')
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {str(e)[:300]}'
        print(f'[dry-run] FAILED: {result["error"]}')
    save_json(out_json, result)
    print(f'[dry-run] JSON: {out_json}')
    return 0 if result.get('ok') else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='t4_bounded_ab.py',
        description='T4 runner: bounded_residual production smoke (--dry-run), '
                    'mid-scale A/B (--ab), CPU pipeline check (--smoke).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true',
                      help='prod config from notebook cell 4 on CUDA: peak VRAM, s/step, telemetry')
    mode.add_argument('--ab', action='store_true',
                      help='mid-scale bounded vs baseline A/B on real data')
    mode.add_argument('--smoke', action='store_true',
                      help='CPU end-to-end check: D=64, 2 layers, vocab=256, 20 steps')
    ap.add_argument('--steps', type=int, default=None, help='train steps (mode default)')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--data-dir', type=str, default=None,
                    help='dir with token_stream_*.bin (default: Colab drive, then <repo>/wb)')
    ap.add_argument('--out-dir', type=str, default=None,
                    help='output dir (default: /content/t4_out, then <repo>/logs/t4)')
    ap.add_argument('--out', type=str, default=None, help='exact JSON output path')
    ap.add_argument('--holdout', type=str, default=None,
                    help='hold-out file name (default: last file; dry-run: first file)')
    ap.add_argument('--train-count', type=int, default=3, help='number of train files')
    ap.add_argument('--cap', type=int, default=1_500_000,
                    help='max filtered tokens kept per file (ab/smoke)')
    ap.add_argument('--windows', type=int, default=2, help='eval windows (ab/smoke)')
    ap.add_argument('--eval-every', type=int, default=None,
                    help='eval cadence (default: ab 200, smoke 10)')
    ap.add_argument('--max-minutes', type=float, default=None,
                    help='wall-clock budget for the whole mode (default: dry 20, ab 95, smoke 5)')
    amp_grp = ap.add_mutually_exclusive_group()
    amp_grp.add_argument('--amp', action='store_true',
                         help='enable CUDA fp16 AMP for --ab/--smoke (default: off, fp32 first; '
                              '--dry-run keeps its fp16 attempt unless --no-amp)')
    amp_grp.add_argument('--no-amp', action='store_true',
                         help='force fp32 everywhere, including --dry-run (skip the fp16 attempt)')
    args = ap.parse_args(argv)

    if args.steps is None:
        args.steps = 8 if args.dry_run else (20 if args.smoke else 1200)
    if args.eval_every is None:
        args.eval_every = 10 if args.smoke else 200
    if args.max_minutes is None:
        args.max_minutes = 20.0 if args.dry_run else (5.0 if args.smoke else 95.0)
    if args.smoke:
        args.cap = min(args.cap, 400_000)
    if args.no_amp:
        use_amp = False
    elif args.amp:
        use_amp = bool(IS_CUDA)
    else:
        use_amp = bool(IS_CUDA and args.dry_run)
    amp_src = '--amp' if args.amp else ('--no-amp' if args.no_amp else
                                        ('dry-run attempt' if args.dry_run else 'default fp32'))

    print(f'[t4] device={DEVICE} torch={torch.__version__} '
          f'threads={torch.get_num_threads()} amp={use_amp} (amp_src={amp_src}) '
          f'mode={"dry-run" if args.dry_run else ("ab" if args.ab else "smoke")} '
          f'steps={args.steps}', flush=True)
    if args.dry_run:
        return run_dry_run(args, use_amp)
    return run_ab(args, use_amp, scale='smoke' if args.smoke else 'ab')


if __name__ == '__main__':
    sys.exit(main())
