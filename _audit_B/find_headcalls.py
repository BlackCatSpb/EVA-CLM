import sys, traceback
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
from common import build, make_batch, train_step
from core.embedding import SigmoidCodedHead

cfg, model, opt, sched, bal, clip = build(grad_ckpt=False)
x, y = make_batch(cfg, batch=1, seq=384)
state = gs = None
for s in range(2):
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)

orig = SigmoidCodedHead.forward
calls = []
def w(self, *a, **k):
    st = traceback.extract_stack()
    calls.append([f'{f.filename.split(chr(92))[-1]}:{f.lineno}' for f in st[-5:-1]])
    return orig(self, *a, **k)
SigmoidCodedHead.forward = w
state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=100, state=state, gs=gs)
SigmoidCodedHead.forward = orig
print('head.forward calls:', len(calls))
for c in calls:
    print('  ', ' <- '.join(c))
