"""What does a fresh model + load_state_dict MISS relative to the live training
model at checkpoint time? (non-persistent runtime buffers)"""
import sys

import torch

sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_A')

from core import EVAStack  # noqa: E402
from run_train import make_cfg, HERE  # noqa: E402
import os  # noqa: E402

ck = torch.load(os.path.join(HERE, 'runA', 'snap_0002.pt'),
                map_location='cpu', weights_only=False)
cfg = make_cfg(os.path.join(HERE, 'tmp'), 1)
torch.manual_seed(0)
m = EVAStack(cfg)
miss, unexp = m.load_state_dict(ck['model'], strict=False)
print('load_state_dict missing/unexpected:', len(miss), len(unexp))

live = ck['all_bufs']
now = {k: v.detach().clone() for k, v in m.named_buffers()}
rows = []
for k in sorted(set(live) | set(now)):
    if k not in live:
        rows.append((k, 'only-fresh', tuple(now[k].shape)))
    elif k not in now:
        rows.append((k, 'only-live', tuple(live[k].shape)))
    elif not torch.equal(live[k], now[k]):
        rows.append((k, 'differs', float((live[k] - now[k]).abs().max())))
print(f'{len(rows)} buffers not reproduced by construct+load_state_dict:')
for k, w, d in rows:
    print(f'  {k:70s} {w:11s} {d}')
