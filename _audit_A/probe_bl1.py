"""Aux-loss finiteness at B*L == 1 (diversity std of one sample)."""
import torch
from harness import build, fwd

for (B, L) in ((1, 1), (1, 2), (2, 1), (2, 2)):
    torch.manual_seed(0)
    m, cfg = build(explicit_reasoning=False, triad_reason=False, bridge_conn=0.0,
                   unified_concept_layer=False, private_mem=False, memory_bank=False,
                   intent_bridge=False, logit_cache_enabled=False,
                   variable_precision=False, head_lacuna=False, head_temper=False)
    m.train()
    x = torch.randint(1, cfg.vocab, (B, L))
    y = torch.randint(1, cfg.vocab, (B, L))
    out, st, gs, rb = fwd(m, cfg, x, adaptive=True, step=1)
    ce, aux = m.compute_losses(out, y)
    bad = {k: float(v) for k, v in aux.items()
           if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()}
    print(f'B={B} L={L}: ce={float(ce):.4f} bad_aux={bad}')
