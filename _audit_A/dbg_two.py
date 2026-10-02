# -*- coding: utf-8 -*-
"""Дебаг двух упавших инвариантов: decode-самосогласованность и optimizer-restore."""
import importlib.util
import os
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\scripts')
from core.config import EVAConfig
from core.stack import EVAStack
from train import _restore_optimizer


def _model(**kw):
    base = dict(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                vocab=256, logit_cache_enabled=False,
                gradient_checkpointing=False, save_dir='.')
    base.update(kw)
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**base))


spec = importlib.util.spec_from_file_location(
    'gen_par', r'C:\EVA_CLM_OPT\scripts\generate.py')
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

print('=== decode self-consistency ===')
m = _model().eval()
torch.manual_seed(3)
ctx = torch.randint(1, 256, (1, 8))
nxt = torch.randint(1, 256, (1, 1))
with torch.no_grad():
    st = gs = it = rb = None
    ob, st, gs, rb = m(m.embed_tokens(ctx), None, step=7, tokens=ctx)
    ob2, st2, gs2, rb2, it2 = gen.decode_step(
        m, nxt, m.lm_head, st, gs, rb, it, step=7)
    ld = m.lm_head(ob2)[0, -1]
    o1, *_ = m(m.embed_tokens(nxt), st, global_state=gs, step=7,
               intent_state=it, tokens=nxt)
    lm = m.lm_head(o1)[0, -1]
print('decode vs manual diff:', float((ld - lm).abs().max()))

print()
print('=== optimizer restore ===')
x = torch.randint(1, 256, (1, 16))


def step(mm, opt):
    h = mm.embed_tokens(x)
    out, *_ = mm(h, None, step=5, tokens=x)
    ce, _ = mm.compute_losses(out, x, h_emb=h)
    opt.zero_grad(set_to_none=True)
    ce.backward()
    opt.step()
    return float(ce.detach())


torch.manual_seed(0)
a = _model().train()
oa = torch.optim.AdamW(a.parameters(), lr=1e-3, betas=(0.9, 0.95))
step(a, oa)
step(a, oa)
sd = {k: v.detach().clone() for k, v in a.state_dict().items()}
osd = oa.state_dict()
names = [n for n, _ in a.named_parameters()]
rt = a.snapshot_runtime_buffers()
rng = torch.get_rng_state()
ref = step(a, oa)
torch.set_rng_state(rng)

torch.manual_seed(1)
b = _model().train()
b.load_state_dict(sd, strict=False)
b.restore_runtime_buffers(rt)
ob = torch.optim.AdamW(b.parameters(), lr=1e-3, betas=(0.9, 0.95))
moved = _restore_optimizer(ob, b, osd, param_names=names)
print('restore moved:', moved)
torch.set_rng_state(rng)
ce_b = step(b, ob)
print('ce ref:', ref, 'ce b:', ce_b)
# параметры после шага
diffs = [(n, float((pa - pb).abs().max()))
         for (n, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters())
         if float((pa - pb).abs().max()) > 1e-6]
print('param diffs:', diffs[:5], 'count:', len(diffs))
# состояние оптимизатора
sa, sb = oa.state_dict(), ob.state_dict()
bad = 0
for i, (ga, gb) in enumerate(zip(sa['param_groups'], sb['param_groups'])):
    pass
for k in list(sa['state'])[:3]:
    va, vb = sa['state'][k], sb['state'].get(k, {})
    for f in ('exp_avg', 'exp_avg_sq', 'step'):
        if f in va:
            if f not in vb:
                bad += 1
            elif torch.is_tensor(va[f]):
                if not torch.equal(va[f], vb[f]):
                    bad += 1
                    print(f'  state[{k}].{f} differs')
            elif va[f] != vb[f]:
                bad += 1
                print(f'  state[{k}].{f}: {va[f]} vs {vb[f]}')
print('optimizer state mismatches:', bad)
