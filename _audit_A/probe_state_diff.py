"""Probe: enumerate plain (non-buffer) tensor attributes that are NOT restored
by restore_runtime_buffers after an eval-like forward."""
import torch
from harness import build, fwd


def collect(m):
    out = {}
    for mod_name, mod in m.named_modules():
        for attr, val in vars(mod).items():
            if isinstance(val, torch.nn.Parameter):
                continue
            if isinstance(val, torch.Tensor):
                out[f'{mod_name}.{attr}'] = val.detach().clone()
            elif isinstance(val, (list, tuple)):
                for i, t in enumerate(val):
                    if isinstance(t, torch.Tensor):
                        out[f'{mod_name}.{attr}[{i}]'] = t.detach().clone()
    return out


def diff(a, b):
    keys = sorted(set(a) | set(b))
    rows = []
    for k in keys:
        if k not in a:
            rows.append((k, 'new', 'None', str(tuple(b[k].shape))))
        elif k not in b:
            rows.append((k, str(tuple(a[k].shape)), 'None', ''))
        else:
            if a[k].shape != b[k].shape:
                rows.append((k, str(tuple(a[k].shape)), str(tuple(b[k].shape)), 'shape'))
            elif not torch.equal(a[k], b[k]):
                d = (a[k] - b[k]).abs().max().item()
                rows.append((k, 'differs', '', f'maxdiff={d:.3e}'))
    return rows


def run(name, **kw):
    torch.manual_seed(0)
    m, cfg = build(**kw)
    cfg.noise_scale_min = 0.0
    cfg.noise_scale_max = 0.0
    xA = torch.randint(1, cfg.vocab, (2, 16))
    xB = torch.randint(1, cfg.vocab, (2, 16))
    m.train()
    out1, st, gs, rb = fwd(m, cfg, xA, adaptive=True, step=1)
    m.observe_output(m.lm_head(out1))
    before = collect(m)
    snap = m.snapshot_runtime_buffers()
    m.eval()
    m.reset_streams()
    if getattr(m, 'memory_bank', None) is not None:
        m.memory_bank.reset()
    if getattr(m, 'explicit_reasoning', False):
        m.reset_reasoning()
    fwd(m, cfg, xB, adaptive=False, step=None)
    m.restore_runtime_buffers(snap)
    m.train()
    after = collect(m)
    rows = diff(before, after)
    print(f'=== {name}: {len(rows)} unrecovered tensor attrs ===')
    for k, s1, s2, note in rows[:60]:
        print(f'  {k:60s} {s1:18s} {s2:18s} {note}')
    if len(rows) > 60:
        print(f'  ... and {len(rows) - 60} more')


CASES = [
    ('baseline', dict(explicit_reasoning=False, triad_reason=False)),
    ('intent_bridge', dict(intent_bridge=True, explicit_reasoning=False, triad_reason=False)),
    ('memory_bank', dict(memory_bank=True, explicit_reasoning=False, triad_reason=False)),
    ('bridge_conn', dict(bridge_conn=0.3, explicit_reasoning=False, triad_reason=False)),
    ('reasoning', dict(explicit_reasoning=True, reasoning_adaptive=True, triad_reason=False)),
    ('logit_cache', dict(logit_cache_enabled=True, explicit_reasoning=False, triad_reason=False)),
    ('all_common', dict(intent_bridge=True, memory_bank=True, bridge_conn=0.3,
                        unified_concept_layer=True, explicit_reasoning=True,
                        triad_reason=False, logit_cache_enabled=True)),
]

if __name__ == '__main__':
    for name, kw in CASES:
        try:
            run(name, **kw)
        except Exception as e:
            print(f'=== {name}: CRASH {type(e).__name__}: {e}')
