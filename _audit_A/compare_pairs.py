"""Pair checkpoints from runA and runB by DATA CURSOR (offset = update count)
and compare model/optimizer/scheduler state."""
import os
import re
import sys

import torch

sys.path.insert(0, r'C:\EVA_CLM_OPT')
from core import EVAConfig  # noqa: F401  (unpickling the ckpt's cfg)

HERE = os.path.dirname(os.path.abspath(__file__))


def snaps(d):
    out = {}
    for f in os.listdir(d):
        m = re.match(r'snap_(\d+)\.pt$', f)
        if m:
            ck = torch.load(os.path.join(d, f), map_location='cpu',
                            weights_only=False)
            out[int(ck['offset'])] = (f, ck)
    return out


def cmp_tensors(a, b, label):
    if a.shape != b.shape:
        return [(label, 'shape', tuple(a.shape), tuple(b.shape))]
    if torch.equal(a, b):
        return []
    return [(label, 'value', float((a - b).abs().max()), '')]


def main():
    A = snaps(os.path.join(HERE, 'runA'))
    B = snaps(os.path.join(HERE, 'runB'))
    common = sorted(set(A) & set(B))
    print('runA offsets:', sorted(A))
    print('runB offsets:', sorted(B))
    for off in common:
        fa, ca = A[off]
        fb, cb = B[off]
        print(f'=== offset {off}: runA {fa} (step {ca["step"]}) vs runB {fb} '
              f'(step {cb["step"]}) ===')
        diffs = []
        for k in sorted(set(ca['model']) | set(cb['model'])):
            if k not in ca['model'] or k not in cb['model']:
                diffs.append((k, 'missing', '', ''))
            else:
                diffs += cmp_tensors(ca['model'][k], cb['model'][k], k)
        print(f'  model diffs: {len(diffs)}')
        for d in diffs[:6]:
            print('   ', d)
        sa, sb = ca.get('optimizer') or {}, cb.get('optimizer') or {}
        da, db = sa.get('state', {}), sb.get('state', {})
        od = 0
        worst = 0.0
        for k in set(da) & set(db):
            for sk in da[k]:
                x, y = da[k][sk], db[k][sk]
                if isinstance(x, torch.Tensor) and not torch.equal(x, y):
                    od += 1
                    worst = max(worst, float((x - y).abs().max()))
        print(f'  optimizer state diffs: {od} (worst {worst:.3e})')
        print('  scheduler:', ca.get('scheduler') == cb.get('scheduler'))
        print('  balancer :', ca.get('balancer') == cb.get('balancer'))
        print('  stream_idx/offset:', ca.get('stream_idx'), cb.get('stream_idx'),
              ca.get('offset'), cb.get('offset'))
        print('  rng equal:', bool((ca['rng'] == cb['rng']).all()))


if __name__ == '__main__':
    main()
