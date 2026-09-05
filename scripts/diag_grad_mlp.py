"""Diagnostic v2: gradients to MLP gate / w_sal / w_intent / tau with salience set & full loss.

Fixes v1: (1) call observe_output so _last_salience is populated (real training does this),
(2) backprop CE + sum(aux) to capture aux-driven grads (tau ladder etc.).
"""
import sys, os, math
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from core import WideBindConfig, WideBindStack
from torch.serialization import add_safe_globals

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/best.pt'
    B, S = 2, 64
    add_safe_globals([WideBindConfig])
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = ckpt['cfg']
    model = WideBindStack(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.train()
    dev = 'cpu'
    model.to(dev)

    mlp_out, sal_cache = {}, {}
    def hook(i):
        def f(mod, inp, out):
            mlp_out[i] = out.detach().norm(dim=-1).mean().item()
        return f
    idxs = [0, len(model.layers) // 2, len(model.layers) - 1]
    for i in idxs:
        model.layers[i].mlp.register_forward_hook(hook(i))

    x = torch.randint(0, cfg.vocab, (B, S), device=dev)

    # pass 1: get logits, populate _last_salience (as training does)
    model.zero_grad(set_to_none=True)
    h = model.embed_tokens(x)
    h_out, _, _, _ = model(h)
    logits = model.lm_head(h_out)
    model.observe_output(logits)
    sal = model._last_salience
    print(f'salience mean={sal.mean().item():.4f} (set after pass 1)')

    # pass 2: now salience is populated -> real gate paths active
    model.zero_grad(set_to_none=True)
    h = model.embed_tokens(x)
    h_out, _, _, _ = model(h)
    ls, aux = model.compute_losses(h_out[:, :-1], x[:, 1:])
    full = ls + sum(v for v in aux.values())
    full.backward()

    def gn(p):
        return p.grad.norm().item() if (p.grad is not None) else 0.0

    print(f'ckpt={os.path.basename(ckpt_path)} step={ckpt.get("step")} '
          f'CE={ls.item():.4f} full={full.item():.4f}')
    print('\n--- per-layer: usefulness | MLP out | grad(mod_scale_mlp) | grad(w_sal) | grad(w_intent) | grad(mlp.W_up) ---')
    for i in idxs:
        m = model.layers[i].mirror
        u = m._cached_usefulness.detach().mean().item() if hasattr(m, '_cached_usefulness') and m._cached_usefulness is not None else float('nan')
        ws = m.w_sal if hasattr(m, 'w_sal') else None
        wi = m.w_intent if hasattr(m, 'w_intent') else None
        wu = model.layers[i].mlp.W_up
        print(f'  L{i}: use={u:.4f} |mlp|={mlp_out.get(i, float("nan")):.3f} '
              f'g_mod={gn(m.mod_scale_mlp):.3e} g_wsal={gn(ws):.3e} g_wint={gn(wi):.3e} g_Wup={gn(wu):.3e}')

    print('\n--- global grad norms ---')
    for name in ['_tau_l_dev', '_tau_intent_dev']:
        p = getattr(model, name, None)
        if p is not None:
            print(f'  {name}: {gn(p):.3e}')
    hw = next(model.lm_head.parameters())
    print(f'  lm_head(first param): {gn(hw):.3e}')

    print('\n--- requires_grad ---')
    for nm in ['mod_scale_mlp', 'w_sal', 'w_intent', 'mod_scale_mem']:
        p = getattr(model.layers[0].mirror, nm, None)
        if p is not None:
            print(f'  {nm}: requires_grad={p.requires_grad}')

if __name__ == '__main__':
    main()
