"""Aten op-count scaling with depth/width: the dispatch-bound hypothesis."""
import sys, time
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
from common import build, make_batch, train_step
from torch.profiler import profile, ProfilerActivity

torch.set_num_threads(8)
lines = []
for (D, nl, V, K, S) in [(256, 4, 512, 16, 4), (256, 8, 512, 16, 4),
                         (512, 8, 512, 16, 4), (1024, 8, 4096, 32, 4)]:
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=False, D=D, n_layers=nl,
                                              vocab=V, code_dim=K, code_sparsity=S)
    x, y = make_batch(cfg, batch=1, seq=384)
    state = gs = None
    for s in range(2):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=100, state=state, gs=gs)
    ev = prof.events()
    aten = [e for e in ev if e.name.startswith('aten::')]
    autograd = [e for e in ev if e.name.startswith('autograd::')]
    lines.append(f'D={D} L={nl} V={V} params={model.param_count():,}: profiler events={len(ev)}, '
                 f'aten calls={len(aten)}, autograd events={len(autograd)}, distinct aten kinds={len(set(e.name for e in aten))}')
    del model
with open(r'C:\EVA_CLM_OPT\_audit_B\opcount_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
