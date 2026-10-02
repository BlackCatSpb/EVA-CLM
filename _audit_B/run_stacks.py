"""Stack-attributed op profile: which Python location burns CPU."""
import sys
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
from common import build, make_batch, train_step
from torch.profiler import profile, ProfilerActivity

torch.set_num_threads(8)
OUT = r'C:\EVA_CLM_OPT\_audit_B\stacks_out.txt'

cfg, model, opt, sched, bal, clip = build(grad_ckpt=False)
x, y = make_batch(cfg, batch=1, seq=384)
state = gs = None
for s in range(2):
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)

with profile(activities=[ProfilerActivity.CPU], record_shapes=False,
             profile_memory=False, with_stack=True) as prof:
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=100, state=state, gs=gs)

lines = []
ka = prof.key_averages(group_by_stack_n=6)
rows = sorted(ka, key=lambda e: -e.self_cpu_time_total)
lines.append('=== key_averages(group_by_stack_n=6), top-60 by self_cpu_time ===')
for e in rows[:60]:
    st = getattr(e, 'stack', None)
    loc = ''
    if st:
        frames = [f'{f.name}:{f.lineno}' for f in st]
        loc = ' <- '.join(frames[:6])
    lines.append(f'{e.key[:44]:<44} self={e.self_cpu_time_total/1e6:8.4f}s n={e.count:<7} {loc[:220]}')

# raw events for index_put with stacks
lines.append('=== _index_put_impl_ raw events ===')
n = 0
for ev in prof.events():
    if ev.name == 'aten::_index_put_impl_':
        n += 1
        if n <= 8:
            st = getattr(ev, 'stack', None)
            frames = [f'{f.name}:{f.lineno}' for f in st] if st else []
            lines.append(f'#{n} dur={ev.cpu_time_total/1e6:.4f}s stack=' + ' <- '.join(frames[:8]))
lines.append(f'total index_put events: {n}')

# aggregate self time by immediate python caller (first frame above aten)
from collections import Counter
agg = Counter()
for e in prof.events():
    if e.name.startswith('aten::') or e.name.startswith('autograd::'):
        st = getattr(e, 'stack', None)
        key = 'NO_STACK'
        if st:
            py = [f for f in st if not f.name.startswith(('aten::', 'autograd::'))]
            if py:
                key = f'{py[0].name}:{py[0].lineno}'
        agg[key] += e.cpu_time_total
lines.append('=== self cpu time by nearest Python frame (aten+autograd events) ===')
for k, v in agg.most_common(40):
    lines.append(f'{k:<70} {v/1e6:8.4f}s')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines[:70]))
