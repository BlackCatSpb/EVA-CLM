"""Proof that the checkpointed block forward is IMPURE: the recompute during
backward applies the in-forward EMA updates a second time."""
import torch
from harness import build, fwd

BASE = dict(explicit_reasoning=False, triad_reason=False, bridge_conn=0.0,
            unified_concept_layer=False, private_mem=False, memory_bank=False,
            intent_bridge=False, logit_cache_enabled=False, variable_precision=False,
            head_lacuna=False, head_temper=False)


def run(gc):
    torch.manual_seed(0)
    m, cfg = build(gradient_checkpointing=gc, **BASE)
    cfg.noise_scale_min = 0.0
    cfg.noise_scale_max = 0.0
    x = torch.randint(1, cfg.vocab, (2, 16))
    y = torch.randint(1, cfg.vocab, (2, 16))
    m.train()
    ema = m.layers[0].mirror._signal_norm_ema
    e0 = ema.detach().clone()
    out, st, gs, rb = fwd(m, cfg, x, adaptive=False, step=None)
    e1 = ema.detach().clone()
    ce = m.compute_loss(out, y)
    ce.backward()
    e2 = ema.detach().clone()
    print(f'gc={gc}: forward changed EMA by {float((e1-e0).abs().max()):.3e}; '
          f'backward changed EMA by {float((e2-e1).abs().max()):.3e}')


run(False)
run(True)
