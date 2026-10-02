"""Resume determinism test through the REAL scripts/train.py entry point.

Run A: 4 steps straight. Run B: 2 steps -> best.pt -> resume -> 2 more steps.
If resume carries the full state (M12), the step-4 checkpoints must be
bit-identical.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r'C:\EVA_CLM_OPT'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import train as T  # noqa: E402
from core import EVAConfig  # noqa: E402

T.args = argparse.Namespace(no_save_optimizer=False)

DATA = os.path.join(HERE, 'data')


def make_cfg(save_dir, max_steps):
    cfg = EVAConfig(
        data_dir=DATA, save_dir=save_dir, batch_size=2, seq_len=16,
        n_layers=2, D=64, vocab=256, mirror_k=32, bind_K=16,
        mlp_groups=4, mlp_expand=4, lr=3e-4, llrd=1.0,
        init_active_layers=2, stage_steps=15000, readiness_full=0.6,
        stage_mode='readiness', max_steps=max_steps, warmup_steps=1,
        log_interval=100, eval_interval=1, save_interval=5000,
        scheduler='mirror', gradient_checkpointing=False,
        logit_cache_enabled=False, explicit_reasoning=False,
        triad_reason=False, memory_bank=False, bridge_conn=0.0,
        intent_bridge=False, unified_concept_layer=False,
        private_mem=False, meta_trust=False, maturation_enabled=False,
        head_lacuna=False, head_temper=False, head_srl=False,
        variable_precision=False, cov_memory=False, inner_eye=False,
        meta_head=False, contradiction_field=False,
        noise_scale_min=0.0, noise_scale_max=0.0)
    cfg.warmup_steps = 1
    cfg.log_interval = 100
    cfg.eval_interval = 1
    return cfg


def install_snapshot_saver():
    orig = T._save_checkpoint_safely

    def saver(state, path):
        d = os.path.dirname(path)
        state = dict(state)
        m = getattr(T, '_last_model', None)
        if m is not None:
            state['all_bufs'] = {k: v.detach().clone()
                                 for k, v in m.named_buffers()}
        snap = os.path.join(d, f'snap_{int(state["step"]):04d}.pt')
        orig(state, snap)
        orig(dict(state), path)   # keep best.pt too (resume source)
    T._save_checkpoint_safely = saver


def stash_model():
    orig = T.evaluate

    def wrapped(model, streams, cfg, device, hold_n=None, step=None):
        T._last_model = model
        v = orig(model, streams, cfg, device, hold_n=hold_n, step=step)
        return -float(step or 0)   # strictly decreasing -> every eval saves
    T.evaluate = wrapped


def load_snap(save_dir, step):
    p = os.path.join(save_dir, f'snap_{step:04d}.pt')
    return torch.load(p, map_location='cpu', weights_only=False)


def compare(a, b, name):
    print(f'--- compare {name} ---')
    same = True
    for key in ('model',):
        sa, sb = a[key], b[key]
        ka, kb = set(sa), set(sb)
        if ka != kb:
            print(f'  {key}: key sets differ: only-A={sorted(ka - kb)[:5]} '
                  f'only-B={sorted(kb - ka)[:5]}')
            same = False
        diffs = []
        for k in sorted(ka & kb):
            if sa[k].shape != sb[k].shape:
                diffs.append((k, 'shape'))
            elif not torch.equal(sa[k], sb[k]):
                diffs.append((k, float((sa[k] - sb[k]).abs().max())))
        if diffs:
            same = False
            print(f'  {key}: {len(diffs)} tensors differ; worst:')
            for k, d in sorted(diffs, key=lambda x: -x[1] if isinstance(x[1], float) else 0)[:8]:
                print(f'    {k:55s} {d}')
    for key in ('optimizer', 'scheduler', 'balancer', 'depth_state',
                'stream_idx', 'offset'):
        va, vb = a.get(key), b.get(key)
        if key == 'optimizer':
            if va is None or vb is None:
                print(f'  {key}: None mismatch {va is None} {vb is None}')
                same = False
                continue
            sta, stb = va.get('state', {}), vb.get('state', {})
            if set(sta) != set(stb):
                print(f'  optimizer.state keys differ: {len(sta)} vs {len(stb)}')
                same = False
            for k in sorted(set(sta) & set(stb)):
                for sk in sta[k]:
                    x, y = sta[k][sk], stb[k][sk]
                    if isinstance(x, torch.Tensor) and not torch.equal(x, y):
                        print(f'  opt.state[{k}][{sk}] differs max='
                              f'{float((x - y).abs().max())}')
                        same = False
        else:
            if va != vb:
                print(f'  {key} differs: {va!r} vs {vb!r}')
                same = False
    print('  VERDICT:', 'BIT-IDENTICAL' if same else 'DIFFERS')
    return same


def force_save():
    """evaluate() wrapper: val loss always 'improves' so every canonical eval
    saves a snapshot (the real save condition is best_val_loss only)."""
    orig = T.evaluate

    def wrapped(model, streams, cfg, device, hold_n=None, step=None):
        v = orig(model, streams, cfg, device, hold_n=hold_n, step=step)
        return v - 0.01 * float(step or 0)
    T.evaluate = wrapped


if __name__ == '__main__':
    import shutil
    for d in ('runA', 'runB'):
        shutil.rmtree(os.path.join(HERE, d), ignore_errors=True)
    install_snapshot_saver()
    stash_model()
    torch.manual_seed(0)
    T.train(make_cfg(os.path.join(HERE, 'runA'), 8), resume_path=None)
    torch.manual_seed(0)
    T.train(make_cfg(os.path.join(HERE, 'runB'), 4), resume_path=None)
    torch.manual_seed(0)   # same seed on resume (test RNG restore)
    T.train(make_cfg(os.path.join(HERE, 'runB'), 8),
            resume_path=os.path.join(HERE, 'runB', 'best.pt'))
    print('runs done; use compare_pairs.py to pair by data offset')
