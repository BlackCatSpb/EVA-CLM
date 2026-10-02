"""Probe: does snapshot/restore around an eval-like forward leave the training
state bit-exact? Functional test: out_base (no eval) vs out_after (eval +
restore) for the SAME second training step."""
import torch
from harness import build, fwd

torch.set_printoptions(precision=8)


def run_case(name, **kw):
    torch.manual_seed(0)
    m, cfg = build(**kw)
    # no forward noise: isolate the leak from RNG consumption
    cfg.noise_scale_min = 0.0
    cfg.noise_scale_max = 0.0
    torch.manual_seed(0)
    xA = torch.randint(1, cfg.vocab, (2, 16))
    xB = torch.randint(1, cfg.vocab, (2, 16))

    # ---- control: step1, step2 (no eval in between) ----
    m.train()
    out1, st, gs, rb = fwd(m, cfg, xA, adaptive=True, step=1)
    m.observe_output(m.lm_head(out1))
    out_base, st2, gs2, rb2 = fwd(m, cfg, xA, state=st, gs=gs, adaptive=True, step=2)

    # ---- experiment: same up to step1, then eval-like, restore, step2 ----
    torch.manual_seed(0)
    m2, cfg2 = build(**kw)
    cfg2.noise_scale_min = 0.0
    cfg2.noise_scale_max = 0.0
    m2.train()
    out1b, stb, gsb, rbb = fwd(m2, cfg2, xA, adaptive=True, step=1)
    m2.observe_output(m2.lm_head(out1b))
    snap = m2.snapshot_runtime_buffers()
    # eval-like: evaluate() semantics
    m2.eval()
    m2.reset_streams()
    if getattr(m2, 'memory_bank', None) is not None:
        m2.memory_bank.reset()
    if getattr(m2, 'explicit_reasoning', False):
        m2.reset_reasoning()
    fwd(m2, cfg2, xB, adaptive=False, step=None)
    m2.restore_runtime_buffers(snap)
    m2.train()
    out_after, _, _, _ = fwd(m2, cfg2, xA, state=stb, gs=gsb, adaptive=True, step=2)

    d = (out_base - out_after).abs().max().item()
    same = torch.equal(out_base, out_after)
    print(f'{name:34s} bit_equal={same} maxdiff={d:.3e}')
    return same, d


CASES = [
    ('baseline(all off)', dict(logit_cache_enabled=False, explicit_reasoning=False,
                               memory_bank=False, bridge_conn=0.0, intent_bridge=False,
                               unified_concept_layer=False, maturation_enabled=False,
                               triad_reason=False, inner_eye=False, meta_head=False,
                               head_lacuna=False, head_temper=False, head_srl=False,
                               variable_precision=False, private_mem=False)),
    ('intent_bridge', dict(intent_bridge=True, explicit_reasoning=False,
                           memory_bank=False, bridge_conn=0.0, triad_reason=False)),
    ('memory_bank', dict(memory_bank=True, explicit_reasoning=False, triad_reason=False)),
    ('bridge_conn', dict(bridge_conn=0.3, explicit_reasoning=False, triad_reason=False)),
    ('concept_layer', dict(unified_concept_layer=True, explicit_reasoning=False,
                           triad_reason=False)),
    ('reasoning', dict(explicit_reasoning=True, reasoning_adaptive=True,
                       triad_reason=False)),
    ('logit_cache', dict(logit_cache_enabled=True, explicit_reasoning=False,
                         triad_reason=False)),
    ('all_common', dict(intent_bridge=True, memory_bank=True, bridge_conn=0.3,
                        unified_concept_layer=True, explicit_reasoning=True,
                        triad_reason=False, logit_cache_enabled=True)),
]

if __name__ == '__main__':
    for name, kw in CASES:
        try:
            run_case(name, **kw)
        except Exception as e:
            import traceback
            print(f'{name:34s} CRASH {type(e).__name__}: {e}')
            traceback.print_exc(limit=3)
