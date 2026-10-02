# -*- coding: utf-8 -*-
"""Дебаг: сохраняется ли kwarg-объект/тензор в recompute (use_reentrant=False)."""
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from torch.utils.checkpoint import checkpoint as cp

# 1) object-kwarg
seen_obj = []


def fn_obj(x, mark=None):
    seen_obj.append(mark)
    return x * 2


x = torch.randn(4, requires_grad=True)
mk = object()
cp(fn_obj, x, mark=mk, use_reentrant=False).sum().backward()
print('object-kwarg: calls=', len(seen_obj),
      'same=', len(seen_obj) == 2 and seen_obj[0] is seen_obj[1])

# 2) tensor-kwarg
seen_t = []


def fn_t(x, mark=None):
    seen_t.append(mark)
    return x * 2


x2 = torch.randn(4, requires_grad=True)
mk2 = torch.empty(0)
cp(fn_t, x2, mark=mk2, use_reentrant=False).sum().backward()
print('tensor-kwarg: calls=', len(seen_t),
      'same=', len(seen_t) == 2 and seen_t[0] is seen_t[1])

# 3) object-positional
seen_p = []


def fn_p(x, mark):
    seen_p.append(mark)
    return x * 2


x3 = torch.randn(4, requires_grad=True)
mk3 = object()
cp(fn_p, x3, mk3, use_reentrant=False).sum().backward()
print('object-positional: calls=', len(seen_p),
      'same=', len(seen_p) == 2 and seen_p[0] is seen_p[1])
