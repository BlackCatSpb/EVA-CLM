# -*- coding: utf-8 -*-
"""M44 verification on a REAL checkpoint: train-regime vs notebook-eval-regime
CE on the same data, current code. Run: python probe_val_parity.py <ckpt.pt>"""
import sys, os, numpy as np, torch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import EVAStack
from core.migrate import migrate_state_dict

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'checkponts\best.pt'
WB = r'..\WideBind\wb'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
cfg = ck['cfg']
m = EVAStack(cfg)
sd, _ = migrate_state_dict(ck['model'], m)
miss, unexp = m.load_state_dict(sd, strict=False)
print(f'ckpt step={ck.get("step")} best_val={ck.get("best_val_loss")} '
      f'(missing={len(miss)} unexpected={len(unexp)})')

def batch(name, off_frac=0.25, seq=None):
    seq = seq or int(cfg.seq_len)
    a = np.memmap(os.path.join(WB, f'token_stream_{name}_eos.bin'), dtype=np.uint16, mode='r')
    o = int(len(a) * off_frac)
    t = torch.from_numpy(a[o:o + seq + 1].astype(np.int64)).unsqueeze(0)
    return t[:, :-1], t[:, 1:]

def run(tag, name, train_mode, step, adaptive):
    m.train(train_mode)
    x, y = batch(name)
    with torch.no_grad():
        h = m.embed_tokens(x)
        out, st, gs, _ = m(h, None, step=step, tokens=x, adaptive=adaptive)
        ce, _ = m.compute_losses(out, y, h_emb=h)
    print(f'  {tag:52s} ce={float(ce):.3f}')

print('PARITY probe (current code, real weights):')
for genre in ('WAR', 'THRILLER', 'TEACHER'):
    run(f'train-mode step=2090  [{genre}]', genre, True, 2090, True)
    run(f'eval-mode  step=None  [{genre}]', genre, False, None, False)
    print()
