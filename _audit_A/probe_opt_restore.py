"""Optimizer state restore BY NAME roundtrip (scripts/train.py _restore_optimizer)."""
import argparse
import sys

import torch

sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\scripts')
import train as T  # noqa: E402
from harness import build, fwd  # noqa: E402

T.args = argparse.Namespace(no_save_optimizer=False)
BASE = dict(explicit_reasoning=False, triad_reason=False, bridge_conn=0.0,
            unified_concept_layer=False, private_mem=False, memory_bank=False,
            intent_bridge=False, logit_cache_enabled=False, variable_precision=False,
            head_lacuna=False, head_temper=False)


def make():
    torch.manual_seed(0)
    m, cfg = build(**BASE)
    cfg.noise_scale_min = 0.0
    cfg.noise_scale_max = 0.0
    from core.adaptation import build_optimizer
    opt = build_optimizer(m, cfg.lr, llrd_decay=cfg.llrd,
                          weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
                          optimizer=getattr(cfg, 'optimizer', 'adamw'),
                          readout_lr_mult=float(getattr(cfg, 'readout_lr_mult', 0.0) or 0.0))
    return m, cfg, opt


m1, cfg1, o1 = make()
x = torch.randint(1, cfg1.vocab, (2, 16))
y = torch.randint(1, cfg1.vocab, (2, 16))
m1.train()
state = gs = None
for step in range(2):
    o1.zero_grad()
    out, state, gs, _ = fwd(m1, cfg1, x, state, gs, step=step, adaptive=True)
    loss = m1.compute_loss(out, y)
    loss.backward()
    o1.step()
sd_opt = o1.state_dict()
names = T._opt_param_names(m1, o1)

m2, cfg2, o2 = make()
moved = T._restore_optimizer(o2, m2, sd_opt, param_names=names)
print('moved>0:', moved, '| slots old:', len(sd_opt['state']),
      'new:', len(o2.state_dict()['state']))
# compare by name
pos_old = {id(p): i for i, p in enumerate(
    p for g in o1.param_groups for p in g['params'])}
pos_new = {id(p): i for i, p in enumerate(
    p for g in o2.param_groups for p in g['params'])}
worst = 0.0
bad = 0
for n1, p1 in m1.named_parameters():
    n2, p2 = n1, dict(m2.named_parameters())[n1]
    i1, i2 = pos_old.get(id(p1)), pos_new.get(id(p2))
    if i1 is None or i2 is None:
        continue
    s1 = sd_opt['state'].get(str(i1))
    s2 = o2.state_dict()['state'].get(str(i2))
    if s1 is None and s2 is None:
        continue
    if (s1 is None) != (s2 is None):
        bad += 1
        print('  presence mismatch', n1)
        continue
    for k in s1:
        a, b = s1[k], s2[k]
        if isinstance(a, torch.Tensor) and not torch.equal(a, b):
            bad += 1
            worst = max(worst, float((a - b).abs().max()))
print('optimizer restore mismatches:', bad, 'worst', worst)
