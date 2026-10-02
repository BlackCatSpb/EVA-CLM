# -*- coding: utf-8 -*-
"""Системный дифф A/B после snapshot+restore: какие атрибуты не покрыты."""
import os
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\EVA_CLM_OPT')
from core.config import EVAConfig
from core.stack import EVAStack


def _model(**kw):
    base = dict(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                vocab=256, logit_cache_enabled=False,
                gradient_checkpointing=False, save_dir='.')
    base.update(kw)
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**base))


def leaves(mod, prefix=''):
    out = {}
    for n, m in mod.named_modules():
        for a, v in vars(m).items():
            if a in ('_parameters', '_buffers', '_modules', 'training') or \
                    a.endswith('_hooks') or a in ('_non_persistent_buffers_set',):
                continue
            key = f'{n}.{a}' if n else a
            out[key] = v
    return out


torch.manual_seed(0)
a = _model().train()
x = torch.randint(1, 256, (1, 16))
with torch.no_grad():
    h = a.embed_tokens(x)
    a(h, None, step=5, tokens=x)
sd = {k: v.detach().clone() for k, v in a.state_dict().items()}
rt = a.snapshot_runtime_buffers()
rng = torch.get_rng_state()

torch.manual_seed(1)
b = _model().train()
b.load_state_dict(sd, strict=False)
b.restore_runtime_buffers(rt)
torch.set_rng_state(rng)

la, lb = leaves(a), leaves(b)
# параметры (исключены из leaves) — не мутирует ли forward их in-place?
for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
    if pa.shape != pb.shape or not torch.equal(pa, pb):
        print(f'PARAM DIFF {na}: {float((pa-pb).abs().max()):.3e}')
# конфиг (forward мог мутировать cfg)
for k in set(vars(a.cfg)) & set(vars(b.cfg)):
    va, vb = getattr(a.cfg, k), getattr(b.cfg, k)
    if isinstance(va, (int, float, bool, str)) and va != vb:
        print(f'CFG DIFF {k}: {va!r} != {vb!r}')
mism = []
for k in sorted(set(la) & set(lb)):
    va, vb = la[k], lb[k]
    if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
        if va.shape != vb.shape or not torch.equal(va, vb):
            mism.append((k, f'tensor diff={float((va - vb).abs().max()) if va.shape == vb.shape else "shape"}'))
    elif isinstance(va, (float, int, bool, str)) or va is None:
        if type(va) is not type(vb) or va != vb:
            mism.append((k, f'{va!r} != {vb!r}'))
    elif isinstance(va, (list, tuple, dict)):
        def _deep(x, y, path):
            if (x is None) != (y is None) or (
                    isinstance(x, torch.Tensor) != isinstance(y, torch.Tensor)
                    and not isinstance(x, (dict, list, tuple))
                    and not isinstance(y, (dict, list, tuple))):
                mism.append((path, f'type {type(x).__name__} != {type(y).__name__}'))
                return
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                if x.shape != y.shape or not torch.equal(x, y):
                    mism.append((path, f'tensor diff={float((x-y).abs().max())}'))
            elif isinstance(x, dict) and isinstance(y, dict):
                for kk in set(x) | set(y):
                    _deep(x.get(kk), y.get(kk), f'{path}.{kk}')
            elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
                for i2, (xx, yy) in enumerate(zip(x, y)):
                    _deep(xx, yy, f'{path}[{i2}]')
        _deep(va, vb, k)
for k, why in mism[:40]:
    print(f'{k:60s} {why}')
print('total mismatches:', len(mism))
