# -*- coding: utf-8 -*-
"""Диагностика: почему L=1 decode != teacher-forced окно (maxdiff 3.47).
Изолируем: шаг, состояние, intent, salience, позиция."""
import importlib.util
import os
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\EVA_CLM_OPT')
from core.config import EVAConfig
from core.stack import EVAStack


def _model(**kw):
    base = dict(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                vocab=256, logit_cache_enabled=False,
                gradient_checkpointing=False, save_dir='.')
    base.update(kw)
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**base)).eval()


spec = importlib.util.spec_from_file_location(
    'gen_par', r'C:\EVA_CLM_OPT\scripts\generate.py')
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

m = _model()
torch.manual_seed(3)
ctx = torch.randint(1, 256, (1, 8))
nxt = torch.randint(1, 256, (1, 1))

with torch.no_grad():
    full = torch.cat([ctx, nxt], dim=1)
    oa, *_ = m(m.embed_tokens(full), None, step=7, tokens=full)
    la = m.lm_head(oa)[0, -1]

    # B1: тот же шаг, что у окна
    st = gs = it = rb = None
    ob, st, gs, rb = m(m.embed_tokens(ctx), None, step=7, tokens=ctx)
    ob2, st2, gs2, rb2, it2 = gen.decode_step(m, nxt, m.lm_head, st, gs, rb, it, step=7)
    lb = m.lm_head(ob2)[0, -1]
    print(f'same-step:  maxdiff={float((la-lb).abs().max()):.4e}')

    # B2: сравнить скрытое состояние последней позиции окна и декода
    ha = oa[0, -1]
    hb = ob2[0, 0]
    print(f'hidden:     maxdiff={float((ha-hb).abs().max()):.4e} '
          f'rel={float((ha-hb).abs().max()/ha.abs().max()):.4e}')

    # B3: полный forward на ctx (8 позиций) — состояние на выходе
    oc, *_ = m(m.embed_tokens(ctx), None, step=7, tokens=ctx)
    print(f'prefill-last vs decode hidden: '
          f'{float((oc[0, -1]-hb).abs().max()):.4e}')
    print(f'window-last  vs prefill-last:  '
          f'{float((ha-oc[0, -1]).abs().max()):.4e}')

    # B4: окно [ctx|nxt] и отдельный forward только nxt после prefill-состояния
    #      (тот же вход, что decode_step делает внутри)
    h1 = m.embed_tokens(nxt)
    o1, st3, gs3, rb3 = m(h1, st, global_state=gs, step=7,
                         intent_state=it,
                         reasoning_buffer=rb[0] if rb is not None else None,
                         reasoning_count=rb[1] if rb is not None else None,
                         tokens=nxt)
    l4 = m.lm_head(o1)[0, -1]
    print(f'manual L=1 vs decode:        {float((l4-lb).abs().max()):.4e}')
    print(f'manual L=1 vs window:        {float((l4-la).abs().max()):.4e}')
