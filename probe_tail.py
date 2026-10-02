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
seen, cap = {}, {}

def hk(i):
    def fn(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        seen[i] = float(h.norm())
    return fn

hs = [l.register_forward_hook(hk(i)) for i, l in enumerate(m.layers)]
m.layers[23].register_forward_hook(lambda mod, inp, out: cap.__setitem__(
    'h23', (out[0] if isinstance(out, tuple) else out).detach()))
h = m.embed_tokens(x)
out, st, gs, _ = m(h, None, step=1045, tokens=x)
for i in hs:
    i.remove()
print('layer output norms:')
for i in (0, 1, 4, 8, 12, 16, 20, 22, 23):
    print('  L%-2d %10.4f' % (i, seen.get(i, float('nan'))))
print('model out norm: %.6f' % float(out.norm()))
h23 = cap['h23']
fn = m.final_norm_w * h23 * torch.rsqrt(h23.pow(2).mean(dim=-1, keepdim=True) + 1e-7)
print('L23 norm %.4f -> final_norm norm %.4f' % (float(h23.norm()), float(fn.norm())))
# and the embedding/stream before L0
h0in = h.detach()
print('embed norm %.4f' % float(h0in.norm()))
