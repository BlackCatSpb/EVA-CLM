import sys
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch, traceback
from common import build, make_batch, train_step

cfg, model, opt, sched, bal, clip = build(grad_ckpt=False)
x, y = make_batch(cfg, batch=1, seq=384)
state = gs = None
for s in range(2):
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)

hits = []

def cb(frame, event, arg):
    if event == 'c_call':
        nm = getattr(arg, '__qualname__', None) or getattr(arg, '__name__', str(arg))
        if 'setitem' in nm or 'index_put' in nm or 'scatter' in nm:
            st = traceback.extract_stack(frame)
            hits.append((nm, [f'{f.filename.split(chr(92))[-1]}:{f.lineno}' for f in st[-6:-1]]))
    return None

sys.setprofile(cb)
state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=100, state=state, gs=gs)
sys.setprofile(None)
print('hits:', len(hits))
for nm, st in hits:
    print(nm, ' <- '.join(st))
