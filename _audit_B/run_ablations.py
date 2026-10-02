"""Module ablations + reasoning/head overhead + balancer cadence, median-of-N."""
import sys, time, statistics, json
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
from common import build, make_batch, train_step

torch.set_num_threads(8)
OUT = r'C:\EVA_CLM_OPT\_audit_B\abl_out.txt'
lines = []


def bench(label, setup=None, n=6, seq=384):
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=False)
    if setup:
        setup(cfg, model, bal)
    x, y = make_batch(cfg, batch=1, seq=seq)
    state = gs = None
    for s in range(2):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y,
                                     step=100 + i, state=state, gs=gs)
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    lines.append(f'{label:<48} median={med:.4f}s  min={min(ts):.4f}  max={max(ts):.4f}  all={[f"{t:.3f}" for t in ts]}')
    return med


def no_reasoning(cfg, model, bal):
    model.explicit_reasoning = False


def no_cache(cfg, model, bal):
    model.logit_cache = None


def no_bridge(cfg, model, bal):
    model.bridge = None


def no_concept(cfg, model, bal):
    model.concept_layer = None


def no_reasoning_memory(cfg, model, bal):
    model.explicit_reasoning = False


def align0(cfg, model, bal):
    bal.align_every = 0


def align_false(cfg, model, bal):
    bal.align = False


def gradalign(cfg, model, bal):
    cfg.gradalign_weight = 0.3


def gc_on(cfg, model, bal):
    pass  # built separately below


lines.append('=== module ablations (gc=False, batch=1 seq=384, median of 6) ===')
base = bench('baseline (all defaults)')
bench('explicit_reasoning=False', no_reasoning)
bench('logit_cache=None', no_cache)
bench('bridge=None', no_bridge)
bench('concept_layer=None', no_concept)
bench('balancer.align_every=0 (1 traversal)', align0)
bench('balancer.align=False (raw sum)', align_false)
bench('gradalign_weight=0.3 (4th traversal)', gradalign)

lines.append('')
lines.append('=== gradient checkpointing (fresh builds) ===')
cfg, model, opt, sched, bal, clip = build(grad_ckpt=True)
x, y = make_batch(cfg, batch=1, seq=384)
state = gs = None
for s in range(2):
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
ts = []
for i in range(4):
    t0 = time.perf_counter()
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=100 + i, state=state, gs=gs)
    ts.append(time.perf_counter() - t0)
lines.append(f'{"gradient_checkpointing=True":<48} median={statistics.median(ts):.4f}s all={[f"{t:.3f}" for t in ts]}')

# ---------- reasoning internals timing ----------
lines.append('')
lines.append('=== reasoning internals (gc=False, per full step) ===')
cfg, model, opt, sched, bal, clip = build(grad_ckpt=False)
x, y = make_batch(cfg, batch=1, seq=384)
state = gs = None
for s in range(2):
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
t_acc = {}
def timed(obj, name, label):
    orig = getattr(obj, name)
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            t_acc[label] = t_acc.get(label, 0.0) + (time.perf_counter() - t0)
            t_acc[label + '.n'] = t_acc.get(label + '.n', 0) + 1
    setattr(obj, name, w)
timed(model, '_knowledge_signal', 'knowledge_signal')
timed(model, '_last_conf', 'last_conf')
timed(model, '_adaptive_reasoning', 'adaptive_reasoning')
timed(model.reasoning_memory, 'forward', 'reasoning_memory')
timed(model.lm_head, 'forward', 'head.forward')
timed(model.embed, 'forward', 'embed.forward')
t0 = time.perf_counter()
state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=100, state=state, gs=gs)
dt = time.perf_counter() - t0
lines.append(f'full step = {dt:.4f}s; internal totals:')
for k in sorted(t_acc):
    if not k.endswith('.n'):
        lines.append(f'  {k:<26} {t_acc[k]:.4f}s  calls={t_acc.get(k+".n")}')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
