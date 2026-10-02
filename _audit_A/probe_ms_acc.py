"""Multi-scale pool accumulators survive reset_cache()/document boundary."""
import torch
from harness import build, fwd

torch.manual_seed(0)
m, cfg = build(explicit_reasoning=False, triad_reason=False, bridge_conn=0.0,
               unified_concept_layer=False, private_mem=False, memory_bank=False,
               intent_bridge=False, variable_precision=False,
               head_lacuna=False, head_temper=False,
               logit_cache_enabled=True, logit_cache_scheduled_sampling=0.0,
               logit_cache_ms_spans='8,32')
m.train()
att = m.logit_cache.attention
x = torch.randint(1, cfg.vocab, (1, 12))
fwd(m, cfg, x, adaptive=False, step=1)
print('after doc A forward: _ms_acc_n =', att._ms_acc_n,
      '_ms_lens =', m.logit_cache.cache._ms_lens)
m.reset_cache()
print('after reset_cache(): _ms_acc_n =', att._ms_acc_n,
      '_ms_lens =', m.logit_cache.cache._ms_lens)
x2 = torch.randint(1, cfg.vocab, (1, 12))
fwd(m, cfg, x2, adaptive=False, step=2)
print('after doc B forward: _ms_acc_n =', att._ms_acc_n,
      'pools =', {k: len(v) for k, v in m.logit_cache.cache._kv_ms.items()})
