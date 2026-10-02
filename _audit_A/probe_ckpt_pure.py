"""Pure-CE gradient parity: gradient_checkpointing on vs off, identical init."""
import torch
from harness import build, fwd

BASE = dict(explicit_reasoning=False, triad_reason=False, bridge_conn=0.0,
            unified_concept_layer=False, private_mem=False, memory_bank=False,
            intent_bridge=False, logit_cache_enabled=False, variable_precision=False,
            head_lacuna=False, head_temper=False)


def run(gc, adaptive):
    torch.manual_seed(0)
    m, cfg = build(gradient_checkpointing=gc, **BASE)
    cfg.noise_scale_min = 0.0
    cfg.noise_scale_max = 0.0
    x = torch.randint(1, cfg.vocab, (2, 16))
    y = torch.randint(1, cfg.vocab, (2, 16))
    m.train()
    out, st, gs, rb = fwd(m, cfg, x, adaptive=adaptive, step=None)
    ce = m.compute_loss(out, y)
    ce.backward()
    return float(ce), {n: (p.grad.detach().clone() if p.grad is not None else None)
                       for n, p in m.named_parameters()}


for adaptive in (False, True):
    ce0, g0 = run(False, adaptive)
    ce1, g1 = run(True, adaptive)
    nmis = 0
    worst = 0.0
    worst_k = ''
    for k in g0:
        a, b = g0[k], g1[k]
        if a is None or b is None:
            nmis += (a is not b)
            continue
        d = float((a - b).abs().max())
        if d > 0:
            nmis += 1
            if d > worst:
                worst, worst_k = d, k
    print(f'adaptive={adaptive}: ce {ce0:.6f}/{ce1:.6f} equal={ce0 == ce1} '
          f'grad mismatches={nmis} worst={worst:.3e} ({worst_k})')
