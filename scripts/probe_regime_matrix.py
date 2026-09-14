# -*- coding: utf-8 -*-
"""Isolate the remaining train/eval split: matrix of regime flags on real weights."""
import sys, os, numpy as np, torch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import EVAStack
from core.migrate import migrate_state_dict

ck = torch.load(r'checkponts\best.pt', map_location='cpu', weights_only=False)
cfg = ck['cfg']
m = EVAStack(cfg)
sd, _ = migrate_state_dict(ck['model'], m)
m.load_state_dict(sd, strict=False)
print('noise cfg range:', getattr(cfg, 'noise_scale_min', None), getattr(cfg, 'noise_scale_max', None))

WB = r'..\WideBind\wb'
a = np.memmap(os.path.join(WB, 'token_stream_WAR_eos.bin'), dtype=np.uint16, mode='r')
o = int(len(a) * 0.25)
t = torch.from_numpy(a[o:o + int(cfg.seq_len) + 1].astype(np.int64)).unsqueeze(0)
x, y = t[:, :-1], t[:, 1:]

def run(tag, train_mode, step, adaptive, noise=None):
    if noise is not None:
        cfg.noise_scale_min = cfg.noise_scale_max = float(noise)
    m.train(train_mode)
    with torch.no_grad():
        h = m.embed_tokens(x)
        out, _, _, _ = m(h, None, step=step, tokens=x, adaptive=adaptive)
        ce, _ = m.compute_losses(out, y, h_emb=h)
    print(f'  {tag:58s} ce={float(ce):.3f}')

# M45: warm up so every streaming attr is a tensor BEFORE the snapshot
m.train()
with torch.no_grad():
    _h = m.embed_tokens(x)
    m(_h, None, step=2090, tokens=x, adaptive=True)
snap = m.snapshot_runtime_buffers()
run('A train  adaptive=T step=2090 noise=cfg',  True,  2090, True)
m.restore_runtime_buffers(snap)
run('B train  adaptive=T step=2090 noise=0',    True,  2090, True, 0.0)
m.restore_runtime_buffers(snap)
run('C eval   adaptive=F step=None  (notebook)', False, None, False)
m.restore_runtime_buffers(snap)
run('D eval   adaptive=T step=2090 (only self.training differs)', False, 2090, True)
m.restore_runtime_buffers(snap)
run('E train  adaptive=F step=None (only self.training differs)', True, None, False)
