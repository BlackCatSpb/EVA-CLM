"""Targeted repros for the unrecovered attrs:
  A) lm_head._last_lacuna_rel leaks into the next train step (memory_bank read
     broadening uses `_lac`).
  B) stack._last_salience leaks into the next train step (intent bridge).
"""
import torch
from harness import build, fwd


def experiment(name, kw, make_live=None):
    torch.manual_seed(0)
    m, cfg = build(**kw)
    cfg.noise_scale_min = 0.0
    cfg.noise_scale_max = 0.0
    if make_live:
        make_live(m)
    xA = torch.randint(1, cfg.vocab, (2, 16))
    xB = torch.randint(1, cfg.vocab, (2, 16))

    def one(m, cfg, seed):
        torch.manual_seed(seed)
        m.train()
        out1, st, gs, rb = fwd(m, cfg, xA, adaptive=True, step=1)
        m.observe_output(m.lm_head(out1))
        snap = m.snapshot_runtime_buffers()
        # eval-like forward
        m.eval()
        m.reset_streams()
        if getattr(m, 'memory_bank', None) is not None:
            m.memory_bank.reset()
        fwd(m, cfg, xB, adaptive=False, step=None)
        # (evaluate() also clears the logit cache; not used in these cases)
        m.restore_runtime_buffers(snap)
        m.train()
        out2, _, _, _ = fwd(m, cfg, xA, state=st, gs=gs, adaptive=True, step=2)
        return out2, snap

    # control: no eval
    torch.manual_seed(0)
    m1, cfg1 = build(**kw)
    cfg1.noise_scale_min = 0.0
    cfg1.noise_scale_max = 0.0
    if make_live:
        make_live(m1)
    m1.train()
    o1, st1, gs1, _ = fwd(m1, cfg1, xA, adaptive=True, step=1)
    m1.observe_output(m1.lm_head(o1))
    base, _, _, _ = fwd(m1, cfg1, xA, state=st1, gs=gs1, adaptive=True, step=2)

    # with eval
    torch.manual_seed(0)
    m2, cfg2 = build(**kw)
    cfg2.noise_scale_min = 0.0
    cfg2.noise_scale_max = 0.0
    if make_live:
        make_live(m2)
    m2.train()
    o1b, st2, gs2, _ = fwd(m2, cfg2, xA, adaptive=True, step=1)
    m2.observe_output(m2.lm_head(o1b))
    snap = m2.snapshot_runtime_buffers()
    m2.eval()
    m2.reset_streams()
    if getattr(m2, 'memory_bank', None) is not None:
        m2.memory_bank.reset()
    fwd(m2, cfg2, xB, adaptive=False, step=None)
    m2.restore_runtime_buffers(snap)
    m2.train()
    after, _, _, _ = fwd(m2, cfg2, xA, state=st2, gs=gs2, adaptive=True, step=2)

    d = (base - after).abs().max().item()
    print(f'{name:40s} bit_equal={torch.equal(base, after)} maxdiff={d:.3e}')
    return base, after, m2


if __name__ == '__main__':
    # A) head lacuna leak with the memory bank consuming it
    experiment('memory_bank (lacuna leak)', dict(
        memory_bank=True, explicit_reasoning=False, triad_reason=False,
        head_lacuna=True, head_temper=True))

    # B) salience leak with a live intent path
    def live_intent(m):
        with torch.no_grad():
            m.bus_head_proj.weight.normal_(0, 0.1)
            m.intent_probe.weight.normal_(0, 0.1)
            m.intent_probe.bias.normal_(0, 0.1)
            for l in m.layers:
                if hasattr(l.mirror, 'w_intent'):
                    l.mirror.w_intent.normal_(0, 0.1)
    experiment('intent_bridge (salience leak)', dict(
        intent_bridge=True, explicit_reasoning=False, triad_reason=False),
        make_live=live_intent)
