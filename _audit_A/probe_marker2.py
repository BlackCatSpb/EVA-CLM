# -*- coding: utf-8 -*-
"""Маркер при ФОРСИРОВАННОМ recompute (backward нужен промежуточный тензор)."""
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from torch.utils.checkpoint import checkpoint as cp


def fn(x, mark=None):
    seen.append(mark)
    y = x.sin()          # промежуточный: backward y*y нужен y -> recompute
    return y * y


seen = []
x = torch.randn(4, requires_grad=True)
mk = object()
cp(fn, x, mark=mk, use_reentrant=False).sum().backward()
print('object-kwarg forced: calls=', len(seen),
      'same=', len(seen) == 2 and seen[0] is seen[1])

seen2 = []
mk_t = torch.empty(0)


def fn2(x, mark=None):
    seen2.append(mark)
    y = x.sin()
    return y * y


x2 = torch.randn(4, requires_grad=True)
cp(fn2, x2, mark=mk_t, use_reentrant=False).sum().backward()
print('tensor-kwarg forced: calls=', len(seen2),
      'same=', len(seen2) == 2 and seen2[0] is seen2[1])
