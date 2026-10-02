"""Head share at production vocab; intent_probe share at larger D."""
import sys, time, statistics
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
from common import build, make_batch, train_step

torch.set_num_threads(8)
OUT = r'C:\EVA_CLM_OPT\_audit_B\vocab_head_out.txt'
lines = []


def wrap(obj, name, label, acc):
    orig = getattr(obj, name)
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            acc[label] = acc.get(label, 0.0) + (time.perf_counter() - t0)
            acc[label + '.n'] = acc.get(label + '.n', 0) + 1
    setattr(obj, name, w)


def bench(vocab, K, S, D=256, nl=4, tag=''):
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=False, vocab=vocab,
                                              code_dim=K, code_sparsity=S, D=D, n_layers=nl)
    x, y = make_batch(cfg, batch=1, seq=384)
    state = gs = None
    for s in range(3):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
    acc = {}
    wrap(model.lm_head, 'forward', 'head', acc)
    wrap(model, 'forward', 'stack.forward', acc)
    wrap(bal, 'backward', 'balancer.backward', acc)
    wrap(model, 'embed_tokens', 'embed', acc)
    if getattr(model, 'intent_probe', None) is not None:
        wrap(model.intent_probe, 'forward', 'intent_probe', acc)
    if getattr(model, 'reasoning_memory', None) is not None:
        wrap(model.reasoning_memory, 'forward', 'reasoning_memory', acc)
    ts = []
    for s in range(3):
        acc.clear()
        t0 = time.perf_counter()
        state, gs, ce, aux = train_step(cfg, model, opt, sched, bal, clip, x, y,
                                        step=100 + s, state=state, gs=gs)
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    lines.append(f'--- {tag} D={D} layers={nl} vocab={vocab} K={K} S={S} params={model.param_count():,} ---')
    lines.append(f'  step median={med:.3f}s all={[f"{t:.3f}" for t in ts]}')
    for k in sorted(acc):
        if not k.endswith('.n'):
            lines.append(f'  {k:<22} {acc[k]:.4f}s ({100*acc[k]/med:.2f}% of step) calls={acc.get(k+".n")}')
    return med


bench(512, 16, 4, tag='mini')
bench(65536, 32, 6, tag='production-vocab')
bench(4096, 32, 4, D=1024, nl=8, tag='production-like-D')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
