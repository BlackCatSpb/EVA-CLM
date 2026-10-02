"""Section timing of (A) one forward pass and (B) one full training step."""
import sys, time, json, statistics
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
import core.block as block_mod
import core.stack as stack_mod
import core.bridge as bridge_mod
import core.training_control as tc_mod
from core.mlp import GroupedMLP
from core.mirror import GroupedCognitiveMirror
from core.bind import TrajectorySpiralBind, BottleneckBind, SpiralBind
from core.block import EVABlock, ExactSequenceMemory
from core.concept_layer import UnifiedConceptLayer
from core.logit_cache import LogitCacheAttention
from common import build, make_batch, train_step, Timer, fmt_table

torch.set_num_threads(8)


class Inst:
    def __init__(self):
        self.t = Timer()
        self.counts = {}
        self.per_layer = {}

    def wrap(self, obj, name, label):
        orig = getattr(obj, name)
        def fn(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                self.t.add(label, time.perf_counter() - t0)
                self.counts[label] = self.counts.get(label, 0) + 1
        setattr(obj, name, fn)

    def wrap_module(self, mod, label, per_layer=None):
        orig_fwd = mod.forward
        def fn(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig_fwd(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                self.t.add(label, dt)
                self.counts[label] = self.counts.get(label, 0) + 1
                if per_layer is not None:
                    self.per_layer.setdefault(per_layer, []).append(dt)
        mod.forward = fn


def install_forward_instrumentation(model, inst):
    inst.wrap_module(model.embed, 'embed', per_layer=None)
    inst.wrap_module(model.lm_head, 'head.forward')
    inst.wrap(model.lm_head, 'log_probs_for_target', 'head.log_probs_for_target')
    if model.bridge is not None:
        inst.wrap(model.bridge, 'inject_layer', 'bridge.inject_layer')
        inst.wrap(model.bridge, 'probe_layer', 'bridge.probe_layer')
        inst.wrap(model.bridge, 'record', 'bridge.record')
        inst.wrap(model.bridge, 'update_stream', 'bridge.update_stream')
    if model.concept_layer is not None:
        inst.wrap_module(model.concept_layer, 'concept')
    if model.logit_cache is not None:
        inst.wrap(model.logit_cache, 'augment', 'logit_cache.augment')
    if getattr(model, 'intent_probe', None) is not None:
        inst.wrap_module(model.intent_probe, 'intent_probe')
    if getattr(model, 'reasoning_memory', None) is not None:
        inst.wrap_module(model.reasoning_memory, 'reasoning')
    if getattr(model, 'memory_bank', None) is not None:
        inst.wrap_module(model.memory_bank, 'memory_bank')
    for i, layer in enumerate(model.layers):
        inst.wrap_module(layer, f'block.L{i}', per_layer=f'block.L{i}')
        inst.wrap_module(layer.mirror, f'mirror.L{i}')
        inst.wrap_module(layer.mlp, f'mlp.L{i}')
        inst.wrap_module(layer.bind, f'bind.L{i}')
        if getattr(layer, 'exact_memory', None) is not None:
            inst.wrap_module(layer.exact_memory, f'vpm.L{i}')
        if getattr(layer, 'precision_gate', None) is not None:
            inst.wrap_module(layer.precision_gate, f'precgate.L{i}')


def install_scan_instrumentation(inst):
    orig_scan = block_mod._scan_chunks
    def scan_fn(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_scan(*a, **k)
        finally:
            inst.t.add('scan._scan_chunks', time.perf_counter() - t0)
            inst.counts['scan._scan_chunks'] = inst.counts.get('scan._scan_chunks', 0) + 1
    block_mod._scan_chunks = scan_fn
    orig_comb = block_mod._combine_chunks
    def comb_fn(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_comb(*a, **k)
        finally:
            inst.t.add('scan._combine_chunks', time.perf_counter() - t0)
            inst.counts['scan._combine_chunks'] = inst.counts.get('scan._combine_chunks', 0) + 1
    block_mod._combine_chunks = comb_fn
    orig_chunk = block_mod._scan_chunk
    def chunk_fn(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_chunk(*a, **k)
        finally:
            inst.t.add('scan._scan_chunk', time.perf_counter() - t0)
            inst.counts['scan._scan_chunk'] = inst.counts.get('scan._scan_chunk', 0) + 1
    block_mod._scan_chunk = chunk_fn
    return orig_scan, orig_comb, orig_chunk


def uninstall_scan_instrumentation(saved):
    block_mod._scan_chunks, block_mod._combine_chunks, block_mod._scan_chunk = saved


def summarize(inst, title, topk=40):
    tot = sum(inst.t.t.values())
    lines = [f'=== {title} (sum of nested timers; overlaps counted once per call) ===',
             fmt_table(inst.t.t, total=tot)]
    lines.append('counts: ' + ', '.join(f'{k}={v}' for k, v in sorted(inst.counts.items())))
    return '\n'.join(lines), inst.t.t, inst.counts


def aggregate_components(raw):
    """Roll up per-layer component timers into global component buckets."""
    agg = {}
    for k, v in raw.items():
        if k.startswith('block.L'):
            agg['block.forward(total)'] = agg.get('block.forward(total)', 0) + v
        elif k.startswith('mirror.L'):
            agg['mirror'] = agg.get('mirror', 0) + v
        elif k.startswith('mlp.L'):
            agg['mlp'] = agg.get('mlp', 0) + v
        elif k.startswith('bind.L'):
            agg['bind'] = agg.get('bind', 0) + v
        elif k.startswith('vpm.L'):
            agg['vpm'] = agg.get('vpm', 0) + v
        elif k.startswith('precgate.L'):
            agg['precgate'] = agg.get('precgate', 0) + v
        else:
            agg[k] = agg.get(k, 0) + v
    return agg


def run(grad_ckpt):
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=grad_ckpt)
    x, y = make_batch(cfg, batch=1, seq=384)
    state = gs = None
    for s in range(3):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s,
                                     state=state, gs=gs)
    # ---------- Phase A: forward + head + losses only ----------
    inst = Inst()
    saved_scan = install_scan_instrumentation(inst)
    install_forward_instrumentation(model, inst)
    try:
        t0 = time.perf_counter()
        h = model.embed_tokens(x)
        out, state2, gs2, _ = model(h, state, global_state=gs, step=99, tokens=x)
        t_fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.observe_output(model.lm_head(out))
        t_obs = time.perf_counter() - t0
        t0 = time.perf_counter()
        ce, aux = model.compute_losses(out, y, h_emb=h)
        t_loss = time.perf_counter() - t0
    finally:
        uninstall_scan_instrumentation(saved_scan)
    aggA = aggregate_components(inst.t.t)
    totalA = t_fwd + t_obs + t_loss
    fwd_internal = sum(v for k, v in aggA.items() if k not in ('embed',))
    tableA = (f'[gc={grad_ckpt}] PHASE A forward-only: embed={t_fwd:.4f}s(whole model fwd incl embed) '
              f'observe_output={t_obs:.4f}s compute_losses={t_loss:.4f}s total={totalA:.4f}s\n'
              + fmt_table(aggA, total=t_fwd, title='forward internals (denominator = model.forward)'))
    # ---------- Phase B: full training step ----------
    inst2 = Inst()
    saved_scan2 = install_scan_instrumentation(inst2)
    # top-level timers
    tstep = {}
    orig_embed = model.embed_tokens
    orig_fwd = model.forward
    def timed(name, fn):
        def w(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                tstep[name] = tstep.get(name, 0.0) + (time.perf_counter() - t0)
        return w
    model.embed_tokens = timed('embed', orig_embed)
    model.forward = timed('model.forward', orig_fwd)
    orig_obs = model.observe_output
    model.observe_output = timed('observe_output', orig_obs)
    orig_cl = model.compute_losses
    model.compute_losses = timed('compute_losses', orig_cl)
    orig_bb = bal.backward
    bal.backward = timed('balancer.backward', orig_bb)
    orig_clip = clip.clip
    clip.clip = timed('clipper.clip', orig_clip)
    orig_step = opt.step
    opt.step = timed('optimizer.step', orig_step)
    orig_sstep = sched.step
    sched.step = timed('scheduler.step', orig_sstep)
    # count/attr autograd traversals
    autograd_calls = {'n': 0, 't': 0.0}
    orig_ag = torch.autograd.grad
    def ag(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_ag(*a, **k)
        finally:
            autograd_calls['n'] += 1
            autograd_calls['t'] += time.perf_counter() - t0
    torch.autograd.grad = ag
    orig_tb = torch.Tensor.backward
    def tb(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_tb(*a, **k)
        finally:
            autograd_calls['n'] += 1
            autograd_calls['t'] += time.perf_counter() - t0
    try:
        torch.Tensor.backward = tb
        _can_patch_tb = True
    except (TypeError, AttributeError):
        _can_patch_tb = False
    try:
        state, gs, ce, aux = train_step(cfg, model, opt, sched, bal, clip, x, y,
                                        step=100, state=state, gs=gs)
    finally:
        uninstall_scan_instrumentation(saved_scan2)
        torch.autograd.grad = orig_ag
        if _can_patch_tb:
            torch.Tensor.backward = orig_tb
    aggB = aggregate_components(inst2.t.t)
    totalB = sum(tstep.values())
    tableB = (f'[gc={grad_ckpt}] PHASE B full step top-level: total={totalB:.4f}s\n'
              + fmt_table(tstep, total=totalB, title='top-level'))
    tableB2 = fmt_table(aggB, total=sum(aggB.values()), title='nested component timers (includes recompute)')
    print(tableA)
    print(tableB)
    print(tableB2)
    print(f'autograd traversals this step: n={autograd_calls["n"]} t={autograd_calls["t"]:.4f}s '
          f'(backward+grad total)')
    print('counts:', {k: v for k, v in sorted(inst2.counts.items())})
    return dict(gc=grad_ckpt, phaseA=aggA, phaseB=aggB, toplevel=tstep,
                forward_total=t_fwd, observe=t_obs, losses=t_loss,
                autograd=autograd_calls, counts=inst2.counts,
                per_layer={k: v for k, v in inst2.per_layer.items()})


if __name__ == '__main__':
    res = {}
    for gc in (False, True):
        res[f'gc_{gc}'] = run(gc)
    with open(r'C:\EVA_CLM_OPT\_audit_B\sections.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('saved sections.json')
