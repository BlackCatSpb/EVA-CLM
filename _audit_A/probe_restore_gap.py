# -*- coding: utf-8 -*-
"""Точный дифф: состояние ОДНОЙ модели до forward vs после restore."""
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\EVA_CLM_OPT')
from core.config import EVAConfig
from core.stack import EVAStack

cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                vocab=256, logit_cache_enabled=False,
                gradient_checkpointing=False, save_dir='.')
torch.manual_seed(0)
m = EVAStack(cfg).eval()


def snapshot_leaves(mod):
    out = {}
    for n, mm in mod.named_modules():
        for a, v in vars(mm).items():
            if a in ('_parameters', '_buffers', '_modules', 'training') or \
                    a.endswith('_hooks') or a in ('_non_persistent_buffers_set',):
                continue
            out[f'{n}.{a}' if n else a] = v
    for k, v in mod.named_buffers():
        out['BUF.' + k] = v
    for k, v in mod.named_parameters():
        out['PAR.' + k] = v
    return out


def cmp(la, lb, tag):
    mism = []

    def deep(x, y, path):
        if (x is None) != (y is None):
            mism.append((path, f'{type(x).__name__} != {type(y).__name__}'))
            return
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.shape != y.shape or not torch.equal(x, y):
                mism.append((path, f'tensor {float((x-y).abs().max()) if x.shape==y.shape else "shape"}'))
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for kk in set(x) | set(y):
                deep(x.get(kk), y.get(kk), f'{path}.{kk}')
            return
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            for i, (xx, yy) in enumerate(zip(x, y)):
                deep(xx, yy, f'{path}[{i}]')
            return
        if isinstance(x, (int, float, bool, str)) and x != y:
            mism.append((path, f'{x!r} != {y!r}'))
    for k in sorted(set(la) & set(lb)):
        deep(la[k], lb[k], k)
    print(f'--- {tag}: {len(mism)} mismatches')
    for k, why in mism[:20]:
        print(f'  {k:58s} {why}')


before = snapshot_leaves(m)
x = torch.randint(1, 256, (1, 16))
with torch.no_grad():
    snap = m.snapshot_runtime_buffers()
    m(m.embed_tokens(x), None, step=7, tokens=x)
    m.restore_runtime_buffers(snap)
after = snapshot_leaves(m)
cmp(before, after, 'before-forward vs after-restore')
