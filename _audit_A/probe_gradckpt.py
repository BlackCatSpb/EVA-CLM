"""Gradient-checkpointing parity + LossBalancer backward under checkpointing."""
import copy

import torch
from harness import build, fwd

BASE = dict(explicit_reasoning=False, triad_reason=False, bridge_conn=0.0,
            unified_concept_layer=False, private_mem=False, memory_bank=False,
            intent_bridge=False, logit_cache_enabled=False, variable_precision=False,
            head_lacuna=False, head_temper=False)


def grads(m):
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in m.named_parameters()}


def run(gc, balancer_align=True):
    torch.manual_seed(0)
    m, cfg = build(gradient_checkpointing=gc, **BASE)
    cfg.noise_scale_min = 0.0
    cfg.noise_scale_max = 0.0
    x = torch.randint(1, cfg.vocab, (2, 16))
    y = torch.randint(1, cfg.vocab, (2, 16))
    m.train()
    out, st, gs, rb = fwd(m, cfg, x, adaptive=True, step=1)
    ce, aux = m.compute_losses(out, y)
    from core.training_control import LossBalancer
    b = LossBalancer(align=balancer_align)
    b.backward(ce, aux, m.parameters(), phase_model=m, step=1)
    return m, grads(m), float(ce)


if __name__ == '__main__':
    m0, g0, ce0 = run(False)
    m1, g1, ce1 = run(True)
    print(f'CE: no-ckpt={ce0:.6f} ckpt={ce1:.6f} equal={ce0 == ce1}')
    worst = 0.0
    nmis = 0
    for k in g0:
        a, b = g0[k], g1[k]
        if a is None or b is None:
            if a is not b:
                nmis += 1
                print('  None mismatch', k)
            continue
        d = float((a - b).abs().max())
        if d > 0:
            nmis += 1
            worst = max(worst, d)
            if nmis < 6:
                print(f'  grad diff {k}: {d:.3e}')
    print(f'grads: {nmis} mismatched, worst {worst:.3e}')

    # second call: LossBalancer with retain_graph semantics (kill measurement)
    torch.manual_seed(0)
    m2, cfg = build(gradient_checkpointing=True, **BASE)
    cfg.noise_scale_min = 0.0
    cfg.noise_scale_max = 0.0
    x = torch.randint(1, cfg.vocab, (2, 16))
    y = torch.randint(1, cfg.vocab, (2, 16))
    m2.train()
    out, st, gs, rb = fwd(m2, cfg, x, adaptive=True, step=1)
    ce, aux = m2.compute_losses(out, y)
    from core.training_control import LossBalancer
    b = LossBalancer(align=True)
    try:
        b.measure_kill(ce, aux, m2.parameters(), phase_model=m2)
        b.backward(ce, aux, m2.parameters(), phase_model=m2, step=1)
        print('measure_kill + backward: OK')
    except Exception as e:
        print('measure_kill + backward CRASH:', type(e).__name__, e)
