"""Interleaved ablation benchmark: wall-min + process-CPU-time, neutral to drift."""
import sys, time, statistics, json
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
from common import build, make_batch, train_step

torch.set_num_threads(8)
OUT = r'C:\EVA_CLM_OPT\_audit_B\abl2_out.txt'
lines = []

VARIANTS = {
    'baseline': lambda cfg, model, bal: None,
    'no_reasoning': lambda cfg, model, bal: setattr(model, 'explicit_reasoning', False),
    'no_logit_cache': lambda cfg, model, bal: setattr(model, 'logit_cache', None),
    'no_bridge': lambda cfg, model, bal: setattr(model, 'bridge', None),
    'no_concept': lambda cfg, model, bal: setattr(model, 'concept_layer', None),
    'align_every=0': lambda cfg, model, bal: setattr(bal, 'align_every', 0),
    'gradalign=0.3': lambda cfg, model, bal: setattr(cfg, 'gradalign_weight', 0.3),
}


def make_variant(name, setup, grad_ckpt=False):
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=grad_ckpt)
    setup(cfg, model, bal)
    x, y = make_batch(cfg, batch=1, seq=384)
    state = gs = None
    for s in range(2):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
    return dict(cfg=cfg, model=model, opt=opt, sched=sched, bal=bal, clip=clip,
                x=x, y=y, state=state, gs=gs, wall=[], proc=[])


def step_once(v, step):
    t0w = time.perf_counter()
    t0p = time.process_time()
    v['state'], v['gs'], _, _ = train_step(v['cfg'], v['model'], v['opt'], v['sched'],
                                           v['bal'], v['clip'], v['x'], v['y'],
                                           step=step, state=v['state'], gs=v['gs'])
    v['wall'].append(time.perf_counter() - t0w)
    v['proc'].append(time.process_time() - t0p)


def run_set(variants, rounds=5, label=''):
    lines.append(f'=== {label} (interleaved, {rounds} rounds x 1 step each) ===')
    for r in range(rounds):
        for name, v in variants.items():
            step_once(v, 100 + r)
    for name, v in variants.items():
        w, p = v['wall'], v['proc']
        lines.append(f'{name:<22} wall: min={min(w):.4f} med={statistics.median(w):.4f} | '
                     f'cpu: min={min(p):.4f} med={statistics.median(p):.4f}')
    lines.append('')
    return variants


vs = {name: make_variant(name, setup) for name, setup in VARIANTS.items()}
run_set(vs, rounds=6, label='gc=False')

# gc=True vs baseline interleaved
vg = {name: make_variant(name, VARIANTS[name], grad_ckpt=True)
      for name in ('baseline', 'no_reasoning', 'align_every=0')}
run_set(vg, rounds=4, label='gc=True')

# pair compare same round for speedup ratios (min cpu)
lines.append('=== derived (min cpu time) ===')
base = min(vs['baseline']['proc'])
for name, v in vs.items():
    if name == 'baseline':
        continue
    m = min(v['proc'])
    lines.append(f'  {name:<22} cpu={m:.4f}s  speedup_vs_base={base/m:.2f}x')
gb = min(vg['baseline']['proc'])
for name, v in vg.items():
    if name == 'baseline':
        continue
    m = min(v['proc'])
    lines.append(f'  gc=True {name:<16} cpu={m:.4f}s  speedup_vs_gc_base={gb/m:.2f}x')
lines.append(f'  gc=True baseline min_cpu={gb:.4f}s vs gc=False baseline min_cpu={base:.4f}s '
             f'=> gc penalty {gb/base:.2f}x')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
