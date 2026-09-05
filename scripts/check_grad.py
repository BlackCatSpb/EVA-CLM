"""Check if gradients flow through hybrid_gate.log_tau."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from core import WideBindConfig, WideBindStack
add_safe_globals([WideBindConfig])

ckpt = torch.load('checkpoints/best 27.pt', map_location='cpu', weights_only=True)
cfg = ckpt['cfg']
model = WideBindStack(cfg)
model.load_state_dict(ckpt['model'], strict=False)
model.train()

dummy_tokens = torch.randint(0, cfg.vocab, (1, 32))
h_emb = model.embed(dummy_tokens)

out, state, gs, rb_rc = model(h_emb, tokens=dummy_tokens, adaptive=True)
print(f'h_out: {out.shape}')

# Targets = next token
targets = dummy_tokens[:, 1:]
ce_loss, aux_dict = model.compute_losses(out, targets, h_emb=h_emb)
print(f'CE: {ce_loss.item():.4f}')
print(f'aux_dict keys: {list(aux_dict.keys()) if aux_dict else "None"}')

total_loss = ce_loss
if aux_dict:
    for k, v in aux_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() == 0:
            total_loss = total_loss + v
            print(f'  aux {k}: {v.item():.4f}')

total_loss.backward()
print(f'\nTotal loss: {total_loss.item():.4f}')

print('\n' + '='*60)
print('hybrid_gate.log_tau grads after backward')
print('='*60)
for i in range(24):
    key = f'layers.{i}.mirror.hybrid_gate.log_tau'
    for name, p in model.named_parameters():
        if name == key:
            g = p.grad
            if i < 3 or i >= 22 or i == 12:
                if g is not None:
                    print(f'  L{i:2d}: grad={g.item():.8f}')
                else:
                    print(f'  L{i:2d}: grad=None (NO GRADIENT!)')
