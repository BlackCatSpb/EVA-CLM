# -*- coding: utf-8 -*-
"""Minimal check: does the CE graph reach the layers on latest.pt?"""
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
x = torch.randint(1, cfg.vocab, (1, 96))
y = torch.randint(1, cfg.vocab, (1, 96))
h = m.embed_tokens(x)
print('h.requires_grad =', h.requires_grad)
out, st, gs, _ = m(h, None, step=1045, tokens=x)
print('out.requires_grad =', out.requires_grad)
ce, aux = m.compute_losses(out, y, h_emb=h)
print('ce.requires_grad =', ce.requires_grad, 'ce=', float(ce))
m.zero_grad(set_to_none=True)
ce.backward()
n_layer = sum(1 for nm, p in m.named_parameters()
              if nm.startswith('layers.') and p.grad is not None)
n_head = sum(1 for nm, p in m.named_parameters()
             if nm.startswith('lm_head') and p.grad is not None)
n_emb = sum(1 for nm, p in m.named_parameters()
            if nm.startswith('embed') and p.grad is not None)
print(f'grads: layers={n_layer}  lm_head={n_head}  embed={n_emb}')
for nm, p in m.named_parameters():
    if p.grad is not None and nm.startswith('layers.0.'):
        print(f'  L0 example: {nm} grad_norm={float(p.grad.norm()):.3e}')
        break
