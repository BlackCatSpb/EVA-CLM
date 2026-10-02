"""MINIMAL resume-integrity repro (no scripts/train.py needed).

1) train m1 for 3 steps -> sd = state_dict(), snap = snapshot_runtime_buffers()
2) fresh m2 + load_state_dict(sd): the next forward DIFFERS from m1's
   (non-persistent runtime buffers are not in state_dict)
3) m2.restore_runtime_buffers(snap): the next forward MATCHES m1's.
"""
import torch
from harness import build, fwd

torch.manual_seed(0)
m1, cfg = build(explicit_reasoning=False, triad_reason=False, bridge_conn=0.0,
                unified_concept_layer=False, private_mem=False,
                memory_bank=False, intent_bridge=False, logit_cache_enabled=False,
                variable_precision=False, head_lacuna=False, head_temper=False)
cfg.noise_scale_min = 0.0
cfg.noise_scale_max = 0.0
opt = torch.optim.AdamW(m1.parameters(), lr=1e-3)
x = torch.randint(1, cfg.vocab, (2, 16))
y = torch.randint(1, cfg.vocab, (2, 16))

m1.train()
state = gs = None
for step in range(3):
    opt.zero_grad()
    out, state, gs, _ = fwd(m1, cfg, x, state, gs, step=step, adaptive=True)
    loss = m1.compute_loss(out, y)
    loss.backward()
    opt.step()
def detach_tree(o):
    if isinstance(o, torch.Tensor):
        return o.detach()
    if isinstance(o, (list, tuple)):
        return type(o)(detach_tree(t) for t in o)
    return o


state = detach_tree(state)
gs = gs.detach()

sd = {k: v.clone() for k, v in m1.state_dict().items()}
snap = m1.snapshot_runtime_buffers()

torch.manual_seed(0)
m2, cfg2 = build(explicit_reasoning=False, triad_reason=False, bridge_conn=0.0,
                 unified_concept_layer=False, private_mem=False,
                 memory_bank=False, intent_bridge=False, logit_cache_enabled=False,
                 variable_precision=False, head_lacuna=False, head_temper=False)
cfg2.noise_scale_min = 0.0
cfg2.noise_scale_max = 0.0
m2.load_state_dict(sd)
m2.train()

rng = torch.get_rng_state()
torch.set_rng_state(rng)
o1, _, _, _ = fwd(m1, cfg, x, state, gs, step=3, adaptive=True)
torch.set_rng_state(rng)
o2, _, _, _ = fwd(m2, cfg2, x, state, gs, step=3, adaptive=True)
d2 = float((o1 - o2).abs().max())

m2.restore_runtime_buffers(snap)

from probe_state_diff import collect, diff  # noqa: E402
rows = diff(collect(m1), collect(m2))
print('remaining unrecovered tensor attrs after restore_runtime_buffers:',
      len(rows))
for r in rows[:20]:
    print('   ', r)

torch.set_rng_state(rng)
o3, _, _, _ = fwd(m2, cfg2, x, state, gs, step=3, adaptive=True)
d3 = float((o1 - o3).abs().max())

print(f'plain load_state_dict : maxdiff={d2:.6e}  bit_equal={torch.equal(o1, o2)}')
print(f'+restore_runtime_bufs : maxdiff={d3:.6e}  bit_equal={torch.equal(o1, o3)}')
print('signal_norm_ema m1 vs fresh:',
      float((m1.layers[0].mirror._signal_norm_ema
             - m2.layers[0].mirror._signal_norm_ema).abs().max())
      if not torch.equal(m1.layers[0].mirror._signal_norm_ema,
                         m2.layers[0].mirror._signal_norm_ema) else 0.0)
