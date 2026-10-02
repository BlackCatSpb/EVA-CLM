import traceback

import torch
from harness import build, fwd

for mode in ('partitioned', 'cognitive_coded'):
    print('=' * 30, mode)
    torch.manual_seed(0)
    try:
        m, cfg = build(head_mode=mode, explicit_reasoning=False, triad_reason=False,
                       bridge_conn=0.0, unified_concept_layer=False, private_mem=False,
                       memory_bank=False, intent_bridge=False,
                       logit_cache_enabled=False, variable_precision=False,
                       head_lacuna=False, head_temper=False)
        x = torch.randint(1, cfg.vocab, (2, 8))
        m.train()
        fwd(m, cfg, x, adaptive=True, step=1)
        print('OK')
    except Exception:
        traceback.print_exc()

print('=' * 30, 'empty-B')
torch.manual_seed(0)
try:
    m, cfg = build(explicit_reasoning=False, triad_reason=False, bridge_conn=0.0,
                   unified_concept_layer=False, private_mem=False, memory_bank=False,
                   intent_bridge=False, logit_cache_enabled=False, variable_precision=False,
                   head_lacuna=False, head_temper=False)
    x = torch.randint(1, cfg.vocab, (0, 8))
    m.train()
    fwd(m, cfg, x, adaptive=True, step=1)
    print('OK')
except Exception:
    traceback.print_exc()
