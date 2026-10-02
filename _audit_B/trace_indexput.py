import sys, json
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
from common import build, make_batch, train_step
from torch.profiler import profile, ProfilerActivity

torch.set_num_threads(8)
cfg, model, opt, sched, bal, clip = build(grad_ckpt=False)
x, y = make_batch(cfg, batch=1, seq=384)
state = gs = None
for s in range(2):
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=100, state=state, gs=gs)
path = r'C:\EVA_CLM_OPT\_audit_B\trace.json'
prof.export_chrome_trace(path)
ev = json.load(open(path, encoding='utf-8'))['traceEvents']
idx = [i for i, e in enumerate(ev) if e.get('name') == 'aten::_index_put_impl_']
print('index_put events:', len(idx))
for i in idx[:6]:
    e = ev[i]
    print('--- event', i, 'dur_us', e.get('dur'), 'args', {k: v for k, v in e.get('args', {}).items() if k != 'stack'})
    st = e.get('args', {}).get('stack')
    if st:
        for f in st:
            print('   ', f)
    else:
        print('    no stack in event')
# python function events named forward with parent info around index puts
print('total events:', len(ev))
