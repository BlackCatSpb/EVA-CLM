"""st0_stand.py -- Stage 0 calibrated mini-stand reproducing the EVA-CLM prod
pathology (bank-dominated residual stream) on real WAR data.

Repo is imported read-only; this file lives outside it. CLI:
    python st0_stand.py --calibrated --steps 0
    python st0_stand.py --calibrated --steps 150 --seed 0 --out st0_metrics.json

Metrics (JSON): flow norms per layer, memory-bank d/h per call, null-bind/mirror
dCE+rel, head/bind gradient ratio, bank-knockout CE, smoke CE trail.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(4)

REPO = r'C:\Users\black\OneDrive\Desktop\EVA CLM'
DEFAULT_DATA = os.path.join(REPO, 'wb', 'token_stream_WAR_eos.bin')
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.config import EVAConfig            # noqa: E402
from core.stack import EVAStack              # noqa: E402

# ---------------------------------------------------------------------------
# Stand constants
# ---------------------------------------------------------------------------
VOCAB = 1820
EMBED_SCALE = 30.0                 # calibration: embedding/flow scale (prod L0 h_in~72)
BANK_GAINS = [20000.0, 500.0]      # calibration: per-bank-call delta gain (L0 4000x, deep ~70x)
BIND_SCALE = 0.2                   # calibration: bind params scale (visibility / head-bind)
STREAM_CAP = 1e6                   # raised from 1e3; finite so training cannot run away
MATUR_T0 = 0.0                     # gates open from step 0 (mem_min_write_mat=0.1)
MATUR_T_DELAY = 0.0
MATUR_DELTA = 1000.0


def build_cfg(args) -> EVAConfig:
    return EVAConfig(
        D=256, n_layers=args.n_layers, mlp_groups=4,
        code_dim=16, code_sparsity=4, vocab=VOCAB,
        memory_bank=True, mem_min_write_mat=0.1,
        gradient_checkpointing=False,
        seq_len=args.seq, batch_size=1,
        stream_cap=STREAM_CAP,
        matur_T0=MATUR_T0, matur_T_delay=MATUR_T_DELAY, matur_delta=MATUR_DELTA,
        lr=6e-4, warmup_steps=10,
    )


def apply_calibration(model, cfg):
    """Scale the stand to the prod regime (all handles explicit)."""
    with torch.no_grad():
        last = model.memory_bank.fusion[-1]
        last.weight.normal_(0.0, 1.0 / math.sqrt(cfg.D))
        last.bias.normal_(0.0, 1.0 / math.sqrt(cfg.D))
        for layer in model.layers:
            if hasattr(layer, 'bind'):
                for p in layer.bind.parameters():
                    p.mul_(BIND_SCALE)
    model.reasoning_enabled_step = 10**9   # CANON: reasoning scale ~1


class BankGain:
    """Forward hook: out = h_in + gain[call_idx] * (out - h_in); logs d/h."""

    def __init__(self, model, gains):
        self.gains = list(gains)
        self.idx = 0
        self.stats = []
        self.hd = model.memory_bank.register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        g = self.gains[min(self.idx, len(self.gains) - 1)]
        h_in = inp[0]
        delta = out - h_in
        with torch.no_grad():
            hi = h_in.detach().norm(dim=-1).mean()
            dn = (g * delta).detach().norm(dim=-1).mean()
            hp = (h_in + g * delta).detach().norm(dim=-1).mean()
        self.stats.append({
            'call': self.idx, 'gain': g,
            'h_in': float(hi), 'h_post': float(hp), 'delta': float(dn),
            'ratio': float(dn / hi.clamp_min(1e-12)),
        })
        self.idx += 1
        return h_in + g * delta

    def remove(self):
        self.hd.remove()


def load_windows(path, seq, n_windows, vocab=VOCAB):
    arr = np.memmap(path, dtype=np.uint16, mode='r')
    w = np.asarray(arr[: max(4_000_000, seq * (n_windows + 1))], dtype=np.int64)
    w = w[w < vocab]
    if len(w) < seq * n_windows + 1:
        raise SystemExit(f'not enough tokens<{vocab}: {len(w)}')
    return [torch.from_numpy(w[i * seq: i * seq + seq + 1]).long().unsqueeze(0)
            for i in range(n_windows)]


def forward_ce(model, x, embed_scale, gains, step, reset=True, targets=None):
    if reset:
        model.memory_bank.reset()
    lc = getattr(model, 'logit_cache', None)
    if lc is not None and hasattr(lc, 'cache'):
        lc.cache.clear()
    bg = BankGain(model, gains)
    h = model.embed_tokens(x) * embed_scale
    out, _, _, _ = model(h, step=step, tokens=x)
    tgt = x if targets is None else targets
    ce = float(model.compute_loss(out, tgt))
    bg.remove()
    return out, ce, bg.stats


def measure(args, model, windows, calibrated):
    step = 200000
    gains = BANK_GAINS if calibrated else [1.0]
    embed_scale = EMBED_SCALE if calibrated else 1.0
    model.train()
    x_list = [w[:, :args.seq] for w in windows]
    y_list = [w[:, 1:args.seq + 1] for w in windows]

    h_out_norms = {}
    handles = []
    for i, layer in enumerate(model.layers):
        def mk(idx):
            def hook(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                h_out_norms[idx] = float(t.detach().norm(dim=-1).mean())
            return hook
        handles.append(layer.register_forward_hook(mk(i)))

    ce_x, ce_next, bank_calls = [], [], []
    for x, y in zip(x_list, y_list):
        with torch.no_grad():
            _, ce, stats = forward_ce(model, x, embed_scale, gains, step)
            _, cen, _ = forward_ce(model, x, embed_scale, gains, step, targets=y)
        ce_x.append(ce)
        ce_next.append(cen)
        bank_calls.append(stats)
    for hh in handles:
        hh.remove()

    # null tests (bind / mirror zeroed at module output, bank reset each pass)
    def null_ce(attr):
        def mk(_m, _i, out):
            if isinstance(out, tuple):
                return (torch.zeros_like(out[0]),) + tuple(out[1:])
            return torch.zeros_like(out)
        dc, rel = [], []
        for x in x_list:
            with torch.no_grad():
                o0, c0, _ = forward_ce(model, x, embed_scale, gains, step)
                n0 = o0.detach()
                hs = [getattr(l, attr).register_forward_hook(mk)
                      for l in model.layers if hasattr(l, attr)]
                o1, c1, _ = forward_ce(model, x, embed_scale, gains, step)
                n1 = o1.detach()
                for hh in hs:
                    hh.remove()
            dc.append(c1 - c0)
            rel.append(float((n1 - n0).norm() / n0.norm().clamp_min(1e-12)))
        return {'dCE': float(np.mean(dc)), 'rel': float(np.mean(rel))}

    null_bind = null_ce('bind')
    null_mirror = null_ce('mirror')

    # bank knockout (module bypassed; gains irrelevant) on x- and next-targets
    orig = model.memory_bank.forward
    model.memory_bank.forward = (lambda h, tokens, step=None, mat_gate=None,
                                 lacuna=None, write=True: h)
    ko_x, ko_next = [], []
    for x, y in zip(x_list, y_list):
        with torch.no_grad():
            _, cex, _ = forward_ce(model, x, embed_scale, gains, step)
            _, cen, _ = forward_ce(model, x, embed_scale, gains, step, targets=y)
        ko_x.append(cex)
        ko_next.append(cen)
    model.memory_bank.forward = orig

    # gradients (gain active, first window)
    model.zero_grad(set_to_none=True)
    model.memory_bank.reset()
    bg = BankGain(model, gains)
    x = x_list[0]
    h = model.embed_tokens(x) * embed_scale
    out, _, _, _ = model(h, step=step, tokens=x)
    ce_t = model.compute_loss(out, x)
    ce_t.backward()
    bg.remove()
    head_sq = bind_sq = mir_sq = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g2 = float(p.grad.norm()) ** 2
        if n.startswith('lm_head'):
            head_sq += g2
        elif '.bind.' in n:
            bind_sq += g2
        elif '.mirror.' in n:
            mir_sq += g2
    model.zero_grad(set_to_none=True)
    head, bind, mir = math.sqrt(head_sq), math.sqrt(bind_sq), math.sqrt(mir_sq)

    return {
        'calibrated': calibrated,
        'CE': float(np.mean(ce_x)), 'CE_windows': ce_x,
        'CE_next': float(np.mean(ce_next)), 'CE_next_windows': ce_next,
        'CE_knockout': float(np.mean(ko_x)),
        'knock_delta': float(np.mean(ko_x) - np.mean(ce_x)),
        'CE_next_knockout': float(np.mean(ko_next)),
        'knock_next_delta': float(np.mean(ko_next) - np.mean(ce_next)),
        'h_out_norms': h_out_norms,
        'bank_calls': bank_calls[0],
        'null_bind': null_bind, 'null_mirror': null_mirror,
        'grad': {'head': head, 'bind': bind, 'mirror': mir,
                 'head/bind': head / max(bind, 1e-30),
                 'head/mirror': head / max(mir, 1e-30)},
    }


def smoke_train(args, model, data, calibrated):
    from core.adaptation import LossBalancer, GradientClipper, build_optimizer
    cfg = model.cfg
    embed_scale = EMBED_SCALE if calibrated else 1.0
    gains = BANK_GAINS if calibrated else [1.0]
    opt = build_optimizer(model, cfg.lr, llrd_decay=cfg.llrd,
                          weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
                          optimizer=getattr(cfg, 'optimizer', 'adamw'))
    bal = LossBalancer(align=True, align_cap=10.0, eval_interval=1000,
                       align_every=1, safety_aux=('head_wall',))
    clip = GradientClipper(c=0.1)
    clip.attach(model)
    model.train()
    model.reasoning_enabled_step = 10**9
    rng = np.random.default_rng(args.seed)
    trail, n_bad, n_nan = [], 0, 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if step % 8 == 1:
            model.memory_bank.reset()
        off = int(rng.integers(0, max(1, len(data) - args.seq - 1)))
        x = torch.from_numpy(data[off:off + args.seq]).long().unsqueeze(0)
        y = torch.from_numpy(data[off + 1:off + args.seq + 1]).long().unsqueeze(0)
        bg = BankGain(model, gains)
        h = model.embed_tokens(x) * embed_scale
        out, _, _, _ = model(h, step=step, tokens=x)
        bg.remove()
        model.observe_output(model.lm_head(out))
        ce, aux = model.compute_losses(out, y, h_emb=h)
        if not torch.isfinite(ce):
            n_nan += 1
            opt.zero_grad(set_to_none=True)
            model.memory_bank.reset()
            continue
        bal.backward(ce, aux, model.parameters(), phase_model=model, step=step)
        if any(not torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None):
            n_bad += 1
        clip.clip(model.parameters())
        opt.step()
        opt.zero_grad(set_to_none=True)
        trail.append(float(ce.detach()))
        if step % 25 == 0:
            print(f'  step={step:4d} ce={np.mean(trail[-25:]):.4f} '
                  f'aux={len(aux)} badgrad={n_bad} nan={n_nan} '
                  f't={time.time()-t0:.0f}s', flush=True)
    return {'steps': args.steps, 'ce_trail': trail,
            'ce_first20': float(np.mean(trail[:20])) if trail else None,
            'ce_last20': float(np.mean(trail[-20:])) if trail else None,
            'nan_steps': n_nan, 'badgrad_steps': n_bad,
            'seconds': round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description='Stage-0 calibrated EVA-CLM mini stand')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=0, help='smoke-training steps (0 = measure only)')
    ap.add_argument('--arms', type=str, default='baseline,calibrated', help='stub arm list')
    ap.add_argument('--calibrated', action='store_true')
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--n-layers', type=int, default=2)
    ap.add_argument('--data', type=str, default=DEFAULT_DATA)
    ap.add_argument('--out', type=str, default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'st0_metrics.json'))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_cfg(args)
    model = EVAStack(cfg)
    if args.calibrated:
        apply_calibration(model, cfg)
    windows = load_windows(args.data, args.seq, args.windows)

    result = {
        'seed': args.seed, 'arms': args.arms, 'calibrated': bool(args.calibrated),
        'params_M': model.param_count() / 1e6,
        'cfg': {'D': cfg.D, 'n_layers': cfg.n_layers, 'mlp_groups': cfg.mlp_groups,
                'code_dim': cfg.code_dim, 'code_sparsity': cfg.code_sparsity,
                'vocab': cfg.vocab, 'mem_min_write_mat': cfg.mem_min_write_mat,
                'stream_cap': cfg.stream_cap, 'seq': args.seq, 'windows': args.windows},
        'calibration': ({'embed_scale': EMBED_SCALE, 'bank_gains': BANK_GAINS,
                         'bind_scale': BIND_SCALE, 'stream_cap': STREAM_CAP,
                         'matur_T0': MATUR_T0, 'matur_T_delay': MATUR_T_DELAY}
                        if args.calibrated else None),
        'measure': measure(args, model, windows, args.calibrated),
    }
    if args.steps > 0:
        arr = np.memmap(args.data, dtype=np.uint16, mode='r')
        w = np.asarray(arr[: max(4_000_000, args.seq * 500)], dtype=np.int64)
        data = w[w < VOCAB]
        result['smoke'] = smoke_train(args, model, data, args.calibrated)
        result['measure_after'] = measure(args, model, windows, args.calibrated)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ('seed', 'calibrated', 'measure', 'smoke',
                               'measure_after')}, ensure_ascii=False, indent=1))
    print(f'\nJSON: {args.out}')


if __name__ == '__main__':
    main()
