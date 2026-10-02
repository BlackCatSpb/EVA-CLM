"""Check when the head_wall safety term adds a full extra backward traversal; head V-scaling."""
import sys, time
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
from common import build, make_batch, train_step

torch.set_num_threads(8)
OUT = r'C:\EVA_CLM_OPT\_audit_B\headwall_out.txt'
lines = []

# --- 1) head_wall traversal trigger across steps ---
cfg, model, opt, sched, bal, clip = build(grad_ckpt=False)
x, y = make_batch(cfg, batch=1, seq=384)
state = gs = None
n_ag = {'n': 0}
orig_ag = torch.autograd.grad
def ag(*a, **k):
    n_ag['n'] += 1
    return orig_ag(*a, **k)
torch.autograd.grad = ag
lines.append('=== head_wall / traversal census over first 8 steps (gc=False, cfg.head_u_wall=%.1e) ==='
             % cfg.head_u_wall)
for s in range(8):
    n_ag['n'] = 0
    state, gs, ce, aux = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)
    u = getattr(model.lm_head, '_last_u', None)
    umax = float(u.abs().max()) if u is not None else float('nan')
    wall = float(aux['head_wall'].detach()) if 'head_wall' in aux else None
    lines.append(f'  step={s} traversals={n_ag["n"]} head_wall={"present" if wall is not None else "absent"} '
                 f'wall={wall} u_max={umax:.2f} aux_n={len(aux)}')
torch.autograd.grad = orig_ag

# --- 2) head cost vs vocab (D=256, L=384, code_dim=16) ---
lines.append('')
lines.append('=== head forward / log_probs cost vs vocab (D=256, L=384, code_dim=16, batch=1) ===')
for V, K, S in ((512, 16, 4), (4096, 32, 4), (16384, 32, 4), (65536, 32, 6)):
    cfg2, model2, _, _, _, _ = build(grad_ckpt=False, vocab=V, code_dim=K, code_sparsity=S)
    model2.train()
    h = torch.randn(1, 384, 256)
    tgt = torch.randint(0, V, (384,))
    with torch.no_grad():
        model2.lm_head(h)
        model2.lm_head.log_probs_for_target(h.reshape(-1, 256), tgt)
        ts_f, ts_l = [], []
        for _ in range(3):
            t0 = time.perf_counter(); model2.lm_head(h); ts_f.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); model2.lm_head.log_probs_for_target(h.reshape(-1, 256), tgt)
            ts_l.append(time.perf_counter() - t0)
    lines.append(f'  V={V:<6} head.forward={min(ts_f)*1000:.2f}ms  log_probs_for_target={min(ts_l)*1000:.2f}ms')

# --- 3) production-like single training step: D=1024, layers=8, vocab=4096 ---
lines.append('')
lines.append('=== production-like step (D=1024, n_layers=8, vocab=4096, seq=384, gc=False) ===')
t0 = time.perf_counter()
cfg3, model3, opt3, sched3, bal3, clip3 = build(grad_ckpt=False, D=1024, n_layers=8, vocab=4096,
                                                mlp_groups=8, code_dim=32)
lines.append(f'  build={time.perf_counter()-t0:.1f}s params={model3.param_count():,}')
x3, y3 = make_batch(cfg3, batch=1, seq=384)
state = gs = None
t0 = time.perf_counter()
state, gs, ce, aux = train_step(cfg3, model3, opt3, sched3, bal3, clip3, x3, y3, step=0, state=state, gs=gs)
lines.append(f'  step0 wall={time.perf_counter()-t0:.2f}s ce={float(ce):.3f}')
t0 = time.perf_counter()
state, gs, ce, aux = train_step(cfg3, model3, opt3, sched3, bal3, clip3, x3, y3, step=1, state=state, gs=gs)
lines.append(f'  step1 wall={time.perf_counter()-t0:.2f}s ce={float(ce):.3f}')
t0 = time.perf_counter()
with torch.no_grad():
    h3 = model3.embed_tokens(x3)
    out3, _, _, _ = model3(h3, None, step=2, tokens=x3)
    model3.observe_output(model3.lm_head(out3))
    ce3, aux3 = model3.compute_losses(out3, y3, h_emb=h3)
lines.append(f'  forward-only wall={time.perf_counter()-t0:.2f}s')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
