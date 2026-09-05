"""Inspect memory bank + mirror experts from checkpoint."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from core import WideBindConfig
add_safe_globals([WideBindConfig])

ckpt = torch.load('checkpoints/best 27.pt', map_location='cpu', weights_only=True)
sd = ckpt['model']

print('='*60)
print('MEMORY BANK STATE')
print('='*60)

# L1
buf = sd.get('memory_bank.l1.buf')
age = sd.get('memory_bank.l1.buf_age')
if buf is not None:
    print(f'L1 buf: {list(buf.shape)} mean={buf.mean():.4f} std={buf.std():.4f}')
    print(f'L1 age: {age.tolist()}')
    n = F.normalize(buf, dim=-1)
    sim = (n @ n.T).fill_diagonal_(0)
    print(f'L1 inter-slot cosine: mean={sim.mean():.4f} max={sim.max():.4f}')

# L2
k = sd.get('memory_bank.l2.keys')
v = sd.get('memory_bank.l2.vals')
age = sd.get('memory_bank.l2.slot_age')
cons = sd.get('memory_bank.l2.slot_consumed')
if k is not None:
    print(f'\nL2 keys: {list(k.shape)} mean={k.mean():.4f} std={k.std():.4f}')
    print(f'L2 vals: {list(v.shape)} mean={v.mean():.4f} std={v.std():.4f}')
    print(f'L2 age: {age.tolist()}')
    print(f'L2 consumed: {cons.tolist()}')
    kn = F.normalize(k, dim=-1)
    sim = (kn @ kn.T).fill_diagonal_(0)
    print(f'L2 inter-key cosine: mean={sim.mean():.4f} max={sim.max():.4f}')

# L3
ck = sd.get('memory_bank.l3.concept_keys')
cv = sd.get('memory_bank.l3.concept_vals')
ca = sd.get('memory_bank.l3.concept_age')
cc = sd.get('memory_bank.l3.concept_count')
if ck is not None:
    print(f'\nL3 keys: {list(ck.shape)} mean={ck.mean():.4f} std={ck.std():.4f}')
    print(f'L3 vals: {list(cv.shape)} mean={cv.mean():.4f} std={cv.std():.4f}')
    print(f'L3 age: {ca.tolist()}')
    print(f'L3 count: {cc.tolist()}')
    ckn = F.normalize(ck, dim=-1)
    sim = (ckn @ ckn.T).fill_diagonal_(0)
    print(f'L3 inter-concept cosine: mean={sim.mean():.4f} max={sim.max():.4f}')
    kv_sim = F.cosine_similarity(ck, cv, dim=-1)
    print(f'L3 key-value cosine: {[f"{x:.4f}" for x in kv_sim.tolist()]}')

print('\n' + '='*60)
print('MIRROR EXPERTS — alpha_diag (expert specialization)')
print('='*60)
for i in range(24):
    key = f'layers.{i}.mirror.alpha_diag'
    if key in sd:
        a = sd[key]
        if i < 3 or i >= 22 or i == 12:
            print(f'  L{i:2d}: mean={a.mean():.4f} std={a.std():.4f} [{a.min():.4f}, {a.max():.4f}]')

print('\n' + '='*60)
print('MIRROR HYBRID GATE log_tau')
print('='*60)
for i in range(24):
    key = f'layers.{i}.mirror.hybrid_gate.log_tau'
    if key in sd:
        t = sd[key]
        if i < 3 or i >= 22 or i == 12:
            print(f'  L{i:2d}: log_tau={t.item():.6f} tau={torch.exp(t).item():.6f}')

print('\n' + '='*60)
print('MIRROR mod_scale_mem')
print('='*60)
for i in range(24):
    key = f'layers.{i}.mirror.mod_scale_mem'
    if key in sd:
        v = sd[key]
        if i < 3 or i >= 22 or i == 12:
            print(f'  L{i:2d}: mean={v.mean():.4f} std={v.std():.4f} [{v.min():.4f}, {v.max():.4f}]')
