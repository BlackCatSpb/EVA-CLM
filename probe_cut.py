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
h = m.embed_tokens(x)
h.retain_grad()
out, st, gs, _ = m(h, None, step=1045, tokens=x)
ce, aux = m.compute_losses(out, x, h_emb=h)

ro = m.lm_head.readout
g = torch.autograd.grad(ce, [out, ro, h], retain_graph=True, allow_unused=True)
print('dce/dout   :', None if g[0] is None else float(g[0].abs().max()))
print('dce/dreadout:', None if g[1] is None else float(g[1].abs().max()))
print('dce/dh_emb :', None if g[2] is None else float(g[2].abs().max()))
# the head's bit stats on this input
with torch.no_grad():
    h_g = out.reshape(1, 64, m.lm_head.K, -1)
    z = (h_g * ro.unsqueeze(0).unsqueeze(0)).sum(-1)
    T = torch.exp(m.lm_head.log_temp).clamp(0.1, 10.0)
    zt = z / T + m.lm_head.bit_bias
print('z abs mean %.4f max %.4f | zt mean %.4f max %.4f' % (
    float(z.abs().mean()), float(z.abs().max()), float(zt.abs().mean()), float(zt.abs().max())))
