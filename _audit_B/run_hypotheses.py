"""Structural hypotheses: pass counts, recompute, .item()/.cpu() syncs, allocations."""
import sys, time, json, gc as gc_mod, tracemalloc, io
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
import core.block as block_mod
import core.stack as stack_mod
from core.block import EVABlock
from core.logit_cache import LogitCacheAttention
from core.embedding import SigmoidCodedHead
from common import build, make_batch, train_step

torch.set_num_threads(8)
OUT = r'C:\EVA_CLM_OPT\_audit_B\hyp_out.txt'
lines = []


def section(s):
    lines.append('')
    lines.append('=' * 90)
    lines.append(s)
    lines.append('=' * 90)


# ---------------- H1: pass counts per training step ----------------
class Counters:
    def __init__(self):
        self.c = {}

    def wrap(self, obj, name, label):
        orig = getattr(obj, name)
        def fn(*a, **k):
            self.c[label] = self.c.get(label, 0) + 1
            return orig(*a, **k)
        setattr(obj, name, fn)


def h1(grad_ckpt, gradalign=0.0):
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=grad_ckpt)
    cfg.gradalign_weight = gradalign
    if gradalign > 0:
        # emulate notebook: gradalign term becomes BYPASS -> extra traversal
        pass
    x, y = make_batch(cfg, batch=1, seq=384)
    cnt = Counters()
    cnt.wrap(model, 'forward', 'EVAStack.forward')
    cnt.wrap(EVABlock, 'forward', 'EVABlock.forward')
    cnt.wrap(EVAStack := type(model), '_checkpointed_block', 'checkpointed_block(recompute)')
    cnt.wrap(block_mod, '_scan_chunks', '_scan_chunks')
    cnt.wrap(SigmoidCodedHead, 'forward', 'head.forward')
    cnt.wrap(model.embed, 'forward', 'embed.forward')
    cnt.wrap(model.tau_config, 'update', 'tau_config.update')
    cnt.wrap(model.bridge, 'probe_layer', 'bridge.probe_layer') if model.bridge else None
    n_ag = {'n': 0}
    orig_ag = torch.autograd.grad
    def ag(*a, **k):
        n_ag['n'] += 1
        return orig_ag(*a, **k)
    torch.autograd.grad = ag
    orig_bw = torch.Tensor.backward
    n_bw = {'n': 0}
    def bw(*a, **k):
        n_bw['n'] += 1
        return orig_bw(*a, **k)
    try:
        torch.Tensor.backward = bw
        can_bw = True
    except (TypeError, AttributeError):
        can_bw = False
    try:
        state = gs = None
        for s in range(2):
            state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y,
                                         step=s, state=state, gs=gs)
        cnt.c.clear()
        n_ag['n'] = 0
        n_bw['n'] = 0
        state, gs, ce, aux = train_step(cfg, model, opt, sched, bal, clip, x, y,
                                        step=100, state=state, gs=gs)
    finally:
        torch.autograd.grad = orig_ag
        if can_bw:
            torch.Tensor.backward = orig_bw
    lines.append(f'--- gc={grad_ckpt} gradalign_weight={gradalign} ---')
    for k, v in sorted(cnt.c.items()):
        lines.append(f'  {k:<38} {v}')
    lines.append(f'  torch.autograd.grad calls: {n_ag["n"]}')
    lines.append(f'  Tensor.backward calls: {n_bw["n"]}')
    lines.append(f'  TOTAL graph traversals: {n_ag["n"] + n_bw["n"]}')
    return cnt.c


# ---------------- H2: sync counts via sys.setprofile ----------------
def h2(grad_ckpt):
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=grad_ckpt)
    x, y = make_batch(cfg, batch=1, seq=384)
    state = gs = None
    for s in range(2):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
    counts = {}
    targets = {'item', 'cpu', 'tolist', 'numpy', 'bool', 'float', '__bool__', '__float__',
               'detach', 'clone', 'contiguous', 'to'}
    def cb(frame, event, arg):
        if event == 'c_call':
            nm = getattr(arg, '__name__', None) or getattr(arg, '__qualname__', str(arg))
            if nm in targets:
                counts[nm] = counts.get(nm, 0) + 1
        return None
    sys.setprofile(cb)
    state, gs, ce, aux = train_step(cfg, model, opt, sched, bal, clip, x, y, step=100, state=state, gs=gs)
    sys.setprofile(None)
    lines.append(f'--- gc={grad_ckpt} c_call counts in ONE step ---')
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f'  Tensor.{k:<12} {v}')
    return counts


# ---------------- H3: cheap vs align path timing ----------------
def h3():
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=False)
    x, y = make_batch(cfg, batch=1, seq=384)
    state = gs = None
    for s in range(2):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)

    def time_step(label, n=3):
        nonlocal state, gs
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y,
                                         step=200 + i, state=state, gs=gs)
            ts.append(time.perf_counter() - t0)
        lines.append(f'  {label:<42} median={sorted(ts)[len(ts)//2]:.4f}s  all={[f"{t:.3f}" for t in ts]}')
        return sorted(ts)[len(ts)//2]

    lines.append('--- align cadence effect (gc=False) ---')
    time_step('align_every=1 (full align, 3 traversals)')
    bal.align_every = 2
    time_step('align_every=2 (every 2nd aligns)')
    bal.align_every = 0
    time_step('align_every=0 (cheap only after seed)')
    bal.align_every = 1
    bal.align = False
    time_step('align=False (raw sum, 1 traversal)')
    bal.align = True


# ---------------- H4: tracemalloc ----------------
def h4():
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=False)
    x, y = make_batch(cfg, batch=1, seq=384)
    state = gs = None
    for s in range(2):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
    gc_mod.collect()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=100, state=state, gs=gs)
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    lines.append('--- tracemalloc top-15 allocation sites (Python allocations in one step) ---')
    diff = snap2.compare_to(snap1, 'lineno')
    for st in diff[:15]:
        lines.append(f'  {st}')
    total = sum(st.size_diff for st in diff)
    lines.append(f'  total Python alloc delta: {total/1e6:.2f} MB over {sum(st.count_diff for st in diff)} objects')


# ---------------- H5: forward-only cost scaling (D, layers) ----------------
def h5():
    from common import build_cfg
    import time as _t
    lines.append('--- forward-only time scaling (batch=1, seq=384) ---')
    for (D, nl, G, K) in [(256, 4, 8, 16), (256, 8, 8, 16), (512, 4, 8, 16), (512, 8, 8, 16)]:
        cfg, model, opt, sched, bal, clip = build(grad_ckpt=False, D=D, n_layers=nl,
                                                  mlp_groups=G, code_dim=K)
        x, y = make_batch(cfg, batch=1, seq=384)
        with torch.no_grad():
            for _ in range(1):
                h = model.embed_tokens(x)
                model(h, None, step=0, tokens=x)
            t0 = time.perf_counter()
            h = model.embed_tokens(x)
            out, st, gs2, _ = model(h, None, step=1, tokens=x)
            model.observe_output(model.lm_head(out))
            ce, aux = model.compute_losses(out, y, h_emb=h)
            dt = time.perf_counter() - t0
        lines.append(f'  D={D:<4} layers={nl} fwd(head+loss,no-grad)={dt:.4f}s params={model.param_count():,}')


if __name__ == '__main__':
    section('H1: full trunk passes per training step')
    h1(False)
    h1(True)
    h1(False, gradalign=0.3)
    h1(True, gradalign=0.3)
    section('H2: host-sync-ish c_call counts per step')
    h2(False)
    h2(True)
    section('H3: balancer cadence timing')
    h3()
    section('H4: tracemalloc')
    h4()
    section('H5: scaling')
    h5()
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
