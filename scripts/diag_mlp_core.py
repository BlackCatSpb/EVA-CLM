"""
Diagnostic: test the "active MLP under untrained metacognitive core" hypothesis.

For a checkpoint (or two), it measures:
  1. MLP activation energy per layer   (the neocortex / executor)
  2. Distance of each param group from a fresh-init baseline
       - mlp      (GroupedMLP weights)
       - mirror   (GroupedCognitiveMirror: the metacognitive core)
       - bind/conv/other
  3. Core gate params that should be NEAR INIT if the controller is untrained:
       - mirror.mod_scale_mlp  (MLP gating, init = log(2.0))
       - mirror.w_sal          (salience gate, zero-init)
       - stack._tau_l_dev / _tau_intent_dev  (own tau, zero-init)
  4. global_state magnitude after a forward (the self-model the core builds)

Hypothesis signature (user, step 990+):
  MLP activation energy HIGH + MLP weights far from init, while mirror/core
  weights (mod_scale_mlp, w_sal, tau devs) stay NEAR INIT  ->  executor woke,
  controller has not learned to govern it yet  ->  bounded fluctuations.

Usage:
  python scripts/diag_mlp_core.py <ckptA.pt> [ckptB.pt] [--codebase PATH]

If only ckptA is given, every quantity is compared against a fresh-init model of
the checkpoint's own cfg. If ckptB is given, A is compared against B (B = the
"earlier / reference" state, e.g. best.pt@233 vs step_987.pt).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_ckpt(path, codebase=None):
    if codebase is not None:
        sys.path.insert(0, codebase)
    from core import model as _mod, config as _cfg
    ModelCls = getattr(_mod, 'WideBindStack', getattr(_mod, 'WideBandStack', None))
    if ModelCls is None:
        raise ImportError('Cannot find model class')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    model = ModelCls(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return cfg, model, ckpt


def build_fresh(cfg, codebase=None):
    if codebase is not None:
        sys.path.insert(0, codebase)
    from core import model as _mod
    ModelCls = getattr(_mod, 'WideBindStack', getattr(_mod, 'WideBandStack', None))
    return ModelCls(cfg).eval()


def mlp_energy(model, cfg, seq=64, batch=2):
    """Per-layer MLP output norm on random input (proxy for 'is MLP active')."""
    energies = []
    hooks = []

    def make(i):
        def hook(_m, _inp, out):
            try:
                t = out.detach().float()
                energies.append(t.norm(dim=-1).mean().item())
            except Exception:
                energies.append(float('nan'))
        return hook

    for i, layer in enumerate(model.layers):
        hooks.append(layer.mlp.register_forward_hook(make(i)))

    x = torch.randint(0, cfg.vocab, (batch, min(seq, cfg.seq_len)))
    h = model.embed_tokens(x)
    with torch.no_grad():
        _, _, gs, _ = model(h)
    for hh in hooks:
        hh.remove()

    gs_norm = float(gs.detach().float().norm().item()) if gs is not None else float('nan')
    return energies, gs_norm


def dist_from_init(model, init):
    """Per-group ||theta - theta_init|| / ||theta_init|| (how far each part trained)."""
    init_p = {n: p for n, p in init.named_parameters()}
    buckets = {'mlp': [], 'mirror': [], 'bind': [], 'conv': [], 'other': []}

    def bucket(name):
        if '.mlp.' in name:
            return 'mlp'
        if '.mirror.' in name:
            return 'mirror'
        if '.bind.' in name:
            return 'bind'
        if 'conv' in name:
            return 'conv'
        return 'other'

    for n, p in model.named_parameters():
        if n not in init_p:
            continue
        pi = init_p[n]
        d = (p.data - pi).norm().item()
        ni = pi.norm().item()
        buckets[bucket(n)].append((d, ni))

    out = {}
    for k, vals in buckets.items():
        if vals:
            dsum = sum(a for a, b in vals)
            nsum = sum(b for a, b in vals)
            out[k] = round(dsum / (nsum + 1e-12), 4)
    return out


def core_gate_state(model):
    """Core params that should be near init if the controller is untrained."""
    info = {}
    mods, sals = [], []
    for layer in model.layers:
        m = layer.mirror
        mods.append(torch.sigmoid(m.mod_scale_mlp).mean().item())   # init ~0.667
        if hasattr(m, 'w_sal'):
            sals.append(m.w_sal.mean().item())                       # init 0.0
    if mods:
        info['mod_scale_mlp(sigmoid)'] = round(sum(mods) / len(mods), 4)
    if sals:
        info['w_sal(mean)'] = round(sum(sals) / len(sals), 4)

    stack = model
    if hasattr(stack, '_tau_l_dev'):
        info['tau_l_dev(|max|)'] = round(stack._tau_l_dev.abs().max().item(), 4)   # init 0
        info['tau_intent_dev(|max|)'] = round(stack._tau_intent_dev.abs().max().item(), 4)
    return info


def report(path, cfg, model, init, ref_name, ref_init):
    e, gs = mlp_energy(model, cfg)
    d = dist_from_init(model, init)
    gates = core_gate_state(model)
    print(f"\n=== {path} (step {getattr(model, '_step', '?')}) ===")
    print(f"  MLP activation energy / layer: {[round(x, 3) for x in e]}")
    print(f"  global_state norm after fwd : {round(gs, 4)}")
    print(f"  Δ-from-{ref_name} norm ratio : {d}")
    print(f"  core gate (near-init = untrained): {gates}")
    return e, d, gates


def main():
    p = argparse.ArgumentParser(description='MLP-vs-core diagnostic')
    p.add_argument('ckptA', type=str)
    p.add_argument('ckptB', type=str, nargs='?', default=None,
                   help='reference checkpoint (e.g. best.pt@233). If omitted, A vs fresh init.')
    p.add_argument('--codebase', type=str, default=None)
    args = p.parse_args()

    cfgA, modelA, ckptA = load_ckpt(args.ckptA, args.codebase)
    modelA._step = ckptA.get('step', '?')
    if args.ckptB:
        cfgB, modelB, ckptB = load_ckpt(args.ckptB, args.codebase)
        modelB._step = ckptB.get('step', '?')
        init = build_fresh(cfgB, args.codebase)
        ref_name = f"init-of({os.path.basename(args.ckptB)})"
        eA, dA, gA = report(args.ckptA, cfgA, modelA, init, ref_name, init)
        eB, dB, gB = report(args.ckptB, cfgB, modelB, init, ref_name, init)
        mlpA = dA.get('mlp', 0.0); mirA = dA.get('mirror', 0.0)
        mlpB = dB.get('mlp', 0.0); mirB = dB.get('mirror', 0.0)
        print("\n--- comparison (A vs B) ---")
        print(f"  MLP energy  : A={[round(x,3) for x in eA]}  B={[round(x,3) for x in eB]}")
        print(f"  MLP trained : A={mlpA}  B={mlpB}")
        print(f"  CORE trained: A={mirA}  B={mirB}")
        if mirB > 0:
            print(f"  ratio MLP/CORE trained: A={round(mlpA/mirA,2) if mirA>0 else 'inf'}  "
                  f"B={round(mlpB/mirB,2) if mirB>0 else 'inf'}")
    else:
        init = build_fresh(cfgA, args.codebase)
        report(args.ckptA, cfgA, modelA, init, 'init', init)


if __name__ == '__main__':
    main()
