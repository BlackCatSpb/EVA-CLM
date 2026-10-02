# -*- coding: utf-8 -*-
import sys, os, torch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import EVAStack
from core.migrate import migrate_state_dict

ck = torch.load(r'checkponts\latest.pt', map_location='cpu', weights_only=False)
cfg = ck['cfg']
m = EVAStack(cfg)
sd, _ = migrate_state_dict(ck['model'], m)
m.load_state_dict(sd, strict=False)
m.train()
torch.manual_seed(7)
x = torch.randint(1, cfg.vocab, (1, 64))

cap = {}
for idx in (0, 5, 11, 17, 23):
    def mk(i):
        def fn(mod, inp, out):
            cap[f'L{i}'] = out[0] if isinstance(out, tuple) else out
        return fn
    m.layers[idx].register_forward_hook(mk(idx))
_orig_aug = m.logit_cache.augment
def aug_spy(h, novelty=None):
    cap['pre_aug'] = h
    o = _orig_aug(h, novelty=novelty)
    cap['post_aug'] = o
    return o
m.logit_cache.augment = aug_spy

h = m.embed_tokens(x)
out, st, gs, _ = m(h, None, step=1045, tokens=x)
ce, aux = m.compute_losses(out, x, h_emb=h)
pts = {k: v for k, v in cap.items() if torch.is_tensor(v) and v.requires_grad}
names = list(pts)
g = torch.autograd.grad(ce, [pts[k] for k in names], retain_graph=True, allow_unused=True)
for k, gi in zip(names, g):
    print('  dce/d%-8s = %s' % (k, 'None' if gi is None else '%.3e' % float(gi.abs().max())))
