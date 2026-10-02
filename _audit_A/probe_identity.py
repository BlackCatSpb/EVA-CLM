# -*- coding: utf-8 -*-
"""Проверка: получает ли recompute ТОТ ЖЕ объект входного тензора (identity),
и на каком уровне (блок) он сохраняется."""
import sys

import torch

sys.path.insert(0, r'C:\EVA_CLM_OPT')
seen = []


def fn(x, y):
    seen.append(('call', id(x), id(y)))
    return x * y


x = torch.randn(4, requires_grad=True)
y = torch.randn(4, requires_grad=True)
from torch.utils.checkpoint import checkpoint as cp
out = cp(fn, x, y, use_reentrant=False)
out.sum().backward()
print('calls:', len(seen))
for tag, ix, iy in seen:
    print(f'  {tag} id(x)={ix} id(y)={iy} same_x={ix == id(x)} same_y={iy == id(y)}')
