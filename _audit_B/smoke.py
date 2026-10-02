import sys, time, json
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
from common import build, make_batch, train_step, build_cfg

torch.set_num_threads(torch.get_num_threads())
print('threads:', torch.get_num_threads())

for gc in (False, True):
    t0 = time.perf_counter()
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=gc)
    t_build = time.perf_counter() - t0
    nparam = model.param_count()
    print(f'gc={gc} build={t_build:.2f}s params={nparam:,} layers={len(model.layers)} '
          f'bridge={model.bridge is not None} cache={model.logit_cache is not None} '
          f'concept={model.concept_layer is not None} reason={model.explicit_reasoning}')
    x, y = make_batch(cfg, batch=1, seq=384)
    t0 = time.perf_counter()
    state, gs, ce, aux = train_step(cfg, model, opt, sched, bal, clip, x, y, step=0)
    dt = time.perf_counter() - t0
    print(f'  step0={dt:.3f}s ce={float(ce):.3f} aux_keys={sorted(aux.keys())}')
    t0 = time.perf_counter()
    state, gs, ce, aux = train_step(cfg, model, opt, sched, bal, clip, x, y, step=1, state=state, gs=gs)
    dt = time.perf_counter() - t0
    print(f'  step1={dt:.3f}s ce={float(ce):.3f}')
