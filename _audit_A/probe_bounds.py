"""Boundary / NaN / dtype sweep: forward+backward across edge shapes and
module toggles. Prints only failures and non-finite results."""
import traceback

import torch
from harness import build, fwd

FAILS = []


def case(name, x=None, y=None, train=True, backward=True, **kw):
    try:
        torch.manual_seed(0)
        m, cfg = build(**kw)
        m.train(train)
        if x is None:
            x = torch.randint(1, cfg.vocab, (2, 8))
        if y is None:
            y = torch.randint(1, cfg.vocab, x.shape)
        out, st, gs, rb = fwd(m, cfg, x, adaptive=train, step=1 if train else None)
        loss = m.compute_loss(out, y)
        v = float(loss)
        finite = bool(torch.isfinite(out).all()) and (v == v)
        if backward and train:
            loss.backward()
            gnan = any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in m.parameters())
            if gnan:
                FAILS.append((name, 'non-finite GRAD'))
                print(f'FAIL {name}: non-finite grad')
        if not finite:
            FAILS.append((name, f'non-finite fwd loss={v}'))
            print(f'FAIL {name}: non-finite forward loss={v}')
    except Exception as e:
        FAILS.append((name, f'{type(e).__name__}: {e}'))
        print(f'CRASH {name}: {type(e).__name__}: {e}')
        traceback.print_exc(limit=4)


if __name__ == '__main__':
    base = dict(explicit_reasoning=False, triad_reason=False, bridge_conn=0.0,
                unified_concept_layer=False, private_mem=False, memory_bank=False,
                intent_bridge=False, logit_cache_enabled=False, variable_precision=False,
                head_lacuna=False, head_temper=False)

    case('B1L1-train', x=torch.randint(1, 256, (1, 1)), y=torch.randint(1, 256, (1, 1)), **base)
    case('B1L1-eval', x=torch.randint(1, 256, (1, 1)), y=torch.randint(1, 256, (1, 1)), train=False, **base)
    case('L1-sep', x=torch.full((2, 1), 2), y=torch.randint(1, 256, (2, 1)), **base)
    case('all-sep', x=torch.full((2, 8), 2), y=torch.full((2, 8), 2), **base)
    case('all-pad-targets', x=torch.randint(1, 256, (2, 8)),
         y=torch.zeros(2, 8, dtype=torch.long), **base)
    case('token0', x=torch.zeros(2, 4, dtype=torch.long),
         y=torch.zeros(2, 4, dtype=torch.long), **base)
    case('empty-B', x=torch.randint(1, 256, (0, 8)), y=torch.randint(1, 256, (0, 8)), **base)
    case('gradckpt', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         gradient_checkpointing=True, **base)
    case('intent_bridge', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         intent_bridge=True, **{k: v for k, v in base.items() if k != 'intent_bridge'})
    case('memory_bank', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         memory_bank=True, **{k: v for k, v in base.items() if k != 'memory_bank'})
    case('concept_layer', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         unified_concept_layer=True, **{k: v for k, v in base.items() if k != 'unified_concept_layer'})
    case('reasoning', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         explicit_reasoning=True, reasoning_adaptive=True,
         **{k: v for k, v in base.items() if k != 'explicit_reasoning'})
    case('reasoning-static', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         explicit_reasoning=True, reasoning_adaptive=False,
         **{k: v for k, v in base.items() if k != 'explicit_reasoning'})
    case('bridge', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         bridge_conn=0.3, **{k: v for k, v in base.items() if k != 'bridge_conn'})
    case('logit_cache', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         logit_cache_enabled=True, logit_cache_scheduled_sampling=0.0,
         **{k: v for k, v in base.items() if k != 'logit_cache_enabled'})
    case('logit_cache-ms', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         logit_cache_enabled=True, logit_cache_scheduled_sampling=0.0,
         logit_cache_ms_spans='8,32',
         **{k: v for k, v in base.items() if k != 'logit_cache_enabled'})
    case('vpm', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         variable_precision=True, **{k: v for k, v in base.items() if k != 'variable_precision'})
    case('softmax_free=False', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         softmax_free=False, **base)
    case('bind-spiral', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         bind_twist_mode='spiral', **base)
    case('head-partitioned', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         head_mode='partitioned', **base)
    case('head-cognitive', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         head_mode='cognitive_coded', **base)
    case('contradiction', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         contradiction_field=True, **base)
    case('sentence-emb-off', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         sent_boundary_emb=False, **base)
    case('surprisal', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         surprisal_weight=0.3, **base)
    case('mask_eos', x=torch.randint(1, 256, (2, 8)), y=torch.randint(1, 256, (2, 8)),
         mask_eos=True, **base)

    print()
    print(f'total failures: {len(FAILS)}')
    for f in FAILS:
        print('  ', f)
