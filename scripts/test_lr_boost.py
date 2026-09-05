"""Prototype: verify the LR upward-path fix in core/stack.py MirrorLRScheduler.

OLD code hard-capped ``min(1.0, mirror_mult)`` -> the adaptive multiplier could
ONLY damp LR, never boost. The fix removes the 1.0 ceiling (bounded by
``lr_boost_max``) and gates the boost on a validation downtrend, so LR can climb
back to "full speed" once the model stabilizes / the meta-core learns.

Two parts:
  A. Deterministic gating test (proves the new logic):
       - when val improving  -> mirror_mult>1 is ALLOWED through (boost)
       - when val NOT improving -> boost is BLOCKED (mult pinned to 1.0)
       - mult never exceeds lr_boost_max (self-bounded)
       - OLD (lr_boost_max=1.0) always caps at 1.0
  B. Integration smoke test on a real tiny model: finite, no crash, OLD caps<=1.

Run: py -3.12 scripts/test_lr_boost.py   (from repo root, PYTHONPATH=.)
"""
import math
import sys
import torch

from core.stack import WideBindStack, MirrorLRScheduler
from core.config import WideBindConfig


def tiny_cfg(boost_max=2.0):
    cfg = WideBindConfig(
        n_layers=6, D=96, bind_K=16, mlp_groups=6,
        mirror_k=8, mirror_k_staircase=False, vocab=256,
        code_dim=32, code_sparsity=6, head_mode='sigmoid_coded',
        intent_bridge=True, head_normalize=True,
    )
    cfg.lr_boost_max = boost_max
    return cfg


def build(boost_max=2.0):
    cfg = tiny_cfg(boost_max)
    model = WideBindStack(cfg).to('cpu')
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = MirrorLRScheduler(model, opt, base_lr=3e-3, warmup=0, cfg=cfg)
    sched.warmup = 0   # override cfg.warmup_steps so we exercise the stats branch
    sched._step = 100  # skip warmup/blend branch so mirror stats are used
    return model, opt, sched


class StatsMock:
    """Returns a scripted sequence of (var, mag, alpha, gate_var) tuples.
    Decreasing var simulates specialization -> var_ratio<1 -> boost wanted."""
    def __init__(self, seq):
        self.seq = seq
        self.i = 0
    def __call__(self):
        return self.seq[min(self.i, len(self.seq) - 1)]
    def advance(self):
        if self.i < len(self.seq) - 1:
            self.i += 1


def test_gating():
    results = {}

    # ---- NEW: upward path allowed when improving, blocked otherwise ----
    model, opt, sched = build(boost_max=2.0)
    # (var, mag, alpha, gate): start moderate, then specialize (var drops)
    stats = StatsMock([(0.010, 0.20, 0.30, 0.40),
                       (0.002, 0.20, 0.30, 0.40)])
    sched._mirror_stats = stats
    sched.report_val_loss(10.0)   # -> improving True (ema=10)
    sched.report_val_loss(9.0)    # -> improving True (9 < 10*1.002)
    sched.step()                  # mirror_mult computed from stats[0] (ratio~1)
    stats.advance()
    sched.step()                  # now var dropped -> mirror_mult > 1
    boost_mult = sched.last_mult
    results['new_boost_when_improving'] = boost_mult

    # now make val regress -> improving False, boost must be blocked
    sched.report_val_loss(11.0)   # 11 < 9.9*1.002? no -> improving False
    stats.advance() if stats.i < len(stats.seq) - 1 else None
    sched.step()                  # stats still low var -> mirror_mult>1 but gated
    capped_mult = sched.last_mult
    results['new_blocked_when_stalled'] = capped_mult

    # ---- OLD: hard cap at 1.0 even when improving ----
    model2, opt2, sched2 = build(boost_max=1.0)
    stats2 = StatsMock([(0.010, 0.20, 0.30, 0.40),
                        (0.002, 0.20, 0.30, 0.40)])
    sched2._mirror_stats = stats2
    sched2.report_val_loss(10.0)
    sched2.report_val_loss(9.0)
    sched2.step()
    stats2.advance()
    sched2.step()
    results['old_mult_when_improving'] = sched2.last_mult

    # ---- boost_max bound respected even with very strong specialization ----
    model3, opt3, sched3 = build(boost_max=1.5)
    strong = StatsMock([(0.010, 0.20, 0.30, 0.40),
                        (1e-5, 0.20, 0.01, 1e-5)])  # var/alpha/gate all crash -> mult maxed
    sched3._mirror_stats = strong
    sched3.report_val_loss(10.0)
    sched3.report_val_loss(9.0)
    sched3.step()
    strong.advance()
    sched3.step()
    results['new_bounded_by_boost_max'] = sched3.last_mult

    return results


def test_integration():
    """Real tiny model, short run: must stay finite; OLD caps at 1.0."""
    cfg = tiny_cfg(boost_max=1.0)
    model = WideBindStack(cfg).to('cpu')
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = MirrorLRScheduler(model, opt, base_lr=3e-3, warmup=10, cfg=cfg)
    g = torch.Generator().manual_seed(7)
    seq = torch.randint(2, cfg.vocab, (4096,), generator=g)
    max_mult = 0.0
    finite = True
    for step in range(120):
        i = torch.randint(0, len(seq) - 64, (1,)).item()
        x = seq[i:i + 64].unsqueeze(0)
        y = seq[i + 1:i + 65].unsqueeze(0)
        opt.zero_grad()
        h = model.embed_tokens(x)
        out, _, _, _ = model(h, step=step)
        model.observe_output(out)
        ce, _ = model.compute_losses(out[:, :-1], y[:, :-1])
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        max_mult = max(max_mult, sched.last_mult)
        if not math.isfinite(opt.param_groups[0]['lr']) or math.isnan(ce.item()):
            finite = False
        if step % 20 == 0:
            sched.report_val_loss(ce.item())
    return dict(max_mult=max_mult, finite=finite)


def main():
    g = test_gating()
    print('=== GATING (deterministic) ===')
    for k, v in g.items():
        print(f'  {k:28s}: {v:.6f}')

    integ = test_integration()
    print('\n=== INTEGRATION (real tiny model) ===')
    for k, v in integ.items():
        print(f'  {k:28s}: {v}')

    checks = [
        ('new boosts when improving', g['new_boost_when_improving'] > 1.001),
        ('new blocked when stalled', g['new_blocked_when_stalled'] <= 1.0001),
        ('old caps at 1.0',         g['old_mult_when_improving'] <= 1.0001),
        ('bounded by boost_max',    g['new_bounded_by_boost_max'] <= 1.5001),
        ('integration finite',      integ['finite'] is True),
        ('integration old caps<=1', integ['max_mult'] <= 1.0001),
    ]
    print('\n=== CHECKS ===')
    ok = True
    for name, passed in checks:
        print(f'  [{"PASS" if passed else "FAIL"}] {name}')
        ok = ok and passed
    print('\nRESULT:', 'OK' if ok else 'CHECK')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
