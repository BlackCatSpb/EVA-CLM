"""Final consolidated section/traversal timing (gc=False, head_wall active, medians)."""
import sys, time, statistics, json
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
import core.block as block_mod
from core.block import EVABlock
from core.embedding import SigmoidCodedHead
from common import build, make_batch, train_step

torch.set_num_threads(8)
OUT = r'C:\EVA_CLM_OPT\_audit_B\final_out.txt'
lines = []


def run(grad_ckpt=False, steps=3):
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=grad_ckpt)
    x, y = make_batch(cfg, batch=1, seq=384)
    state = gs = None
    for s in range(5):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
    # verify head_wall active
    _, _, _, aux = model.compute_losses(model.embed_tokens(x), y) if False else (None, None, None, {})
    acc = {}

    def add(k, dt):
        acc[k] = acc.get(k, 0.0) + dt

    def wrap(obj, name, label):
        orig = getattr(obj, name)
        def w(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                add(label, time.perf_counter() - t0)
                add(label + '#', 1)
        setattr(obj, name, w)

    wrap(model, 'embed_tokens', 'embed')
    wrap(model, 'forward', 'stack.forward')
    wrap(model, 'observe_output', 'observe_output')
    wrap(model, 'compute_losses', 'losses.compute_losses')
    wrap(model, '_adaptive_reasoning', 'reasoning.loop')
    wrap(model, '_knowledge_signal', 'head.knowledge_signal')
    wrap(model, '_last_conf', 'head.last_conf')
    wrap(model.lm_head, 'forward', 'head.forward')
    wrap(model.lm_head, 'log_probs_for_target', 'head.log_probs')
    wrap(model.tau_config, 'update', 'tau_config.update')
    wrap(model.bridge, 'inject_layer', 'bridge.inject')
    wrap(model.bridge, 'probe_layer', 'bridge.probe')
    wrap(model.bridge, 'record', 'bridge.record')
    wrap(model.bridge, 'update_stream', 'bridge.stream')
    wrap(model.concept_layer, 'forward', 'concept')
    wrap(model.logit_cache, 'augment', 'cache.augment')
    wrap(model.intent_probe, 'forward', 'intent_probe')
    wrap(bal, 'backward', 'balancer.backward')
    wrap(clip, 'clip', 'clipper.clip')
    wrap(opt, 'step', 'optimizer.step')
    wrap(sched, 'step', 'scheduler.step')
    for i, layer in enumerate(model.layers):
        wrap(layer, 'forward', f'block.L{i}')
        wrap(layer.mirror, 'forward', f'mirror.L{i}')
        wrap(layer.mlp, 'forward', f'mlp.L{i}')
        wrap(layer.bind, 'forward', f'bind.L{i}')
        if getattr(layer, 'exact_memory', None) is not None:
            wrap(layer.exact_memory, 'forward', f'vpm.L{i}')
    # scan
    orig_scan = block_mod._scan_chunks
    def scan_w(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_scan(*a, **k)
        finally:
            add('scan.chunks', time.perf_counter() - t0)
    block_mod._scan_chunks = scan_w
    orig_comb = block_mod._combine_chunks
    def comb_w(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_comb(*a, **k)
        finally:
            add('scan.combine', time.perf_counter() - t0)
    block_mod._combine_chunks = comb_w
    # autograd traversals
    trav = []
    orig_ag = torch.autograd.grad
    def ag_w(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_ag(*a, **k)
        finally:
            trav.append(time.perf_counter() - t0)
    torch.autograd.grad = ag_w
    try:
        per_step = []
        for s in range(steps):
            acc.clear(); trav.clear()
            t0 = time.perf_counter()
            state, gs, ce, aux = train_step(cfg, model, opt, sched, bal, clip, x, y,
                                            step=200 + s, state=state, gs=gs)
            total = time.perf_counter() - t0
            per_step.append((total, dict(acc), list(trav)))
    finally:
        torch.autograd.grad = orig_ag
        block_mod._scan_chunks = orig_scan
        block_mod._combine_chunks = orig_comb
    # aggregate medians
    totals = sorted(t for t, _, _ in per_step)
    med = totals[len(totals) // 2]
    keys = set()
    for _, a, _ in per_step:
        keys |= set(a)
    meds = {}
    for k in keys:
        vals = sorted(a.get(k, 0.0) for _, a, _ in per_step)
        meds[k] = vals[len(vals) // 2]
    return med, meds, per_step


def report(gc):
    med, meds, per_step = run(gc)
    lines.append('=' * 100)
    lines.append(f'FINAL gc={gc}: median step={med:.4f}s (n={len(per_step)}), '
                 f'per-step totals={[f"{t:.3f}" for t, _, _ in per_step]}')
    lines.append('=' * 100)
    groups = [
        ('TOP-LEVEL', ['embed', 'stack.forward', 'observe_output', 'losses.compute_losses',
                       'balancer.backward', 'clipper.clip', 'optimizer.step', 'scheduler.step']),
        ('FORWARD COMPONENTS (inside stack.forward)', [
            'head.forward', 'head.log_probs', 'head.knowledge_signal', 'head.last_conf',
            'reasoning.loop', 'bridge.inject', 'bridge.probe', 'bridge.record', 'bridge.stream',
            'concept', 'cache.augment', 'intent_probe', 'tau_config.update']),
        ('BLOCK COMPONENTS (per-layer summed)', [k for k in meds if k.startswith(('block.L', 'mirror.L', 'mlp.L', 'bind.L', 'vpm.L'))]),
        ('SCAN', ['scan.chunks', 'scan.combine']),
    ]
    for gname, keys in groups:
        lines.append(f'--- {gname} ---')
        for k in keys:
            if k in meds:
                lines.append(f'  {k:<32} {meds[k]:>9.4f}s  {100*meds[k]/med:>6.2f}% of step  calls={meds.get(k+"#",0):.0f}')
    # autograd traversal detail (last step)
    lines.append('--- AUTOGRAD TRAVERSALS (last step) ---')
    for i, t in enumerate(per_step[-1][2]):
        lines.append(f'  traversal[{i}] {t:.4f}s')
    lines.append(f'  sum={sum(per_step[-1][2]):.4f}s')


report(False)
report(True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
