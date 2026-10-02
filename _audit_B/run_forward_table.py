"""Clean forward-only section table: medians over 5 passes, head_wall regime."""
import sys, time, statistics
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
import core.block as block_mod
from common import build, make_batch, train_step

torch.set_num_threads(8)
OUT = r'C:\EVA_CLM_OPT\_audit_B\forward_table.txt'


def measure(grad_ckpt=False, n=5):
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=grad_ckpt)
    x, y = make_batch(cfg, batch=1, seq=384)
    state = gs = None
    for s in range(4):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)

    per = []
    orig = {}
    def wrap(obj, name, label):
        orig[(id(obj), name)] = getattr(obj, name)
        o = getattr(obj, name)
        def w(*a, **k):
            t0 = time.perf_counter()
            try:
                return o(*a, **k)
            finally:
                cur[label] = cur.get(label, 0.0) + (time.perf_counter() - t0)
                cur[label + '.n'] = cur.get(label + '.n', 0) + 1
        setattr(obj, name, w)
    os_scan = block_mod._scan_chunks
    os_comb = block_mod._combine_chunks
    for it in range(n):
        cur = {}
        def scan_w(*a, **k):
            t0 = time.perf_counter()
            try:
                return os_scan(*a, **k)
            finally:
                cur['scan.chunks'] = cur.get('scan.chunks', 0.0) + (time.perf_counter() - t0)
        def comb_w(*a, **k):
            t0 = time.perf_counter()
            try:
                return os_comb(*a, **k)
            finally:
                cur['scan.combine'] = cur.get('scan.combine', 0.0) + (time.perf_counter() - t0)
        block_mod._scan_chunks = scan_w
        block_mod._combine_chunks = comb_w
        # install wrappers fresh each iteration
        wrapped = []
        def W(obj, name, label):
            o = getattr(obj, name)
            wrapped.append((obj, name, o))
            def w(*a, **k):
                t0 = time.perf_counter()
                try:
                    return o(*a, **k)
                finally:
                    cur[label] = cur.get(label, 0.0) + (time.perf_counter() - t0)
                    cur[label + '.n'] = cur.get(label + '.n', 0) + 1
            setattr(obj, name, w)
        W(model, 'embed_tokens', 'embed')
        W(model, 'forward', 'stack.forward')
        W(model, 'observe_output', 'observe_output')
        W(model, 'compute_losses', 'losses')
        W(model.lm_head, 'forward', 'head.forward')
        W(model.lm_head, 'log_probs_for_target', 'head.log_probs')
        W(model.tau_config, 'update', 'tau.update')
        if model.bridge is not None:
            W(model.bridge, 'inject_layer', 'bridge.inject')
            W(model.bridge, 'probe_layer', 'bridge.probe')
            W(model.bridge, 'record', 'bridge.record')
            W(model.bridge, 'update_stream', 'bridge.stream')
        if model.concept_layer is not None:
            W(model.concept_layer, 'forward', 'concept')
        if model.logit_cache is not None:
            W(model.logit_cache, 'augment', 'cache.augment')
        W(model.intent_probe, 'forward', 'intent_probe')
        if getattr(model, 'reasoning_memory', None) is not None:
            W(model.reasoning_memory, 'forward', 'reasoning_memory')
        for i, layer in enumerate(model.layers):
            W(layer, 'forward', f'block.L{i}')
            W(layer.mirror, 'forward', f'mirror.L{i}')
            W(layer.mlp, 'forward', f'mlp.L{i}')
            W(layer.bind, 'forward', f'bind.L{i}')
            if getattr(layer, 'exact_memory', None) is not None:
                W(layer.exact_memory, 'forward', f'vpm.L{i}')
        t0 = time.perf_counter()
        h = model.embed_tokens(x)
        out, state2, gs2, _ = model(h, state, global_state=gs, step=100 + it, tokens=x)
        model.observe_output(model.lm_head(out))
        ce, aux = model.compute_losses(out, y, h_emb=h)
        total = time.perf_counter() - t0
        per.append((total, dict(cur)))
        # restore
        for obj, name, o in wrapped:
            setattr(obj, name, o)
    block_mod._scan_chunks = os_scan
    block_mod._combine_chunks = os_comb
    return per


def report(grad_ckpt):
    per = measure(grad_ckpt)
    totals = sorted(t for t, _ in per)
    med = totals[len(totals) // 2]
    keys = set()
    for _, c in per:
        keys |= set(c)
    meds = {k: statistics.median([c.get(k, 0.0) for _, c in per]) for k in keys if not k.endswith('.n')}
    calls = {k: int(statistics.median([c.get(k + '.n', 0) for _, c in per])) for k in keys if not k.endswith('.n')}
    # aggregate block components
    agg = {}
    for k, v in meds.items():
        base = k.split('.L')[0] if '.L' in k else k
        if k.startswith(('block.L', 'mirror.L', 'mlp.L', 'bind.L', 'vpm.L')):
            agg[base] = agg.get(base, 0.0) + v
        else:
            agg[k] = v
    trunk = meds.get('stack.forward', 0.0)
    lines = [f'===== FORWARD-ONLY (gc={grad_ckpt}), batch=1 seq=384, median of {len(per)} passes =====',
             f'total embed+trunk+head+losses = {med:.4f}s   trunk(stack.forward) = {trunk:.4f}s   '
             f'head+losses+observe = {med - trunk - meds.get("embed",0):.4f}s']
    lines.append(f'{"section":<26} {"sec":>9} {"% trunk":>8} {"% total":>8} {"calls":>6}')
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        lines.append(f'{k:<26} {v:>9.4f} {100*v/max(trunk,1e-9):>7.2f}% {100*v/med:>7.2f}% {calls.get(k,0):>6}')
    lines.append('')
    lines.append('per-layer block times: ' + ', '.join(
        f'L{i}={meds.get(f"block.L{i}",0)*1000:.1f}ms' for i in range(len(meds) and 4)))
    return '\n'.join(lines), meds, med


if __name__ == '__main__':
    out = []
    for gc in (False, True):
        txt, _, _ = report(gc)
        out.append(txt)
        print(txt)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(out))
