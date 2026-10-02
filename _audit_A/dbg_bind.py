# -*- coding: utf-8 -*-
"""Дебаг: градиент bind-параметров последнего слоя при guard вкл/выкл."""
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\EVA_CLM_OPT')
from core.config import EVAConfig
from core.stack import EVAStack

SMALL = dict(n_layers=1, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)
cfg = EVAConfig(**{**SMALL, 'n_layers': 2, 'intent_bridge': True,
                   'memory_bank': True, 'bridge_conn': 0.1,
                   'unified_concept_layer': True, 'explicit_reasoning': True})
torch.manual_seed(0)
m = EVAStack(cfg).train()
for b in m.layers:
    with torch.no_grad():
        b.precision_gate.gate.bias.fill_(3.0)
opt = torch.optim.SGD(m.parameters(), lr=0.02)
state = None
for it in range(8):
    x = torch.randint(1, m.cfg.vocab, (1, 64))
    x[:, 31] = 2
    with torch.no_grad():
        h0 = m.embed_tokens(x)
        o0, _, _, _ = m(h0, state, step=20000 + it, tokens=x)
        m.observe_output(m.lm_head(o0))
    h = m.embed_tokens(x)
    out, state, gs, r = m(h, state, step=20000 + it, tokens=x)
    loss, aux = m.compute_losses(out, x, h_emb=h)
    total = loss + sum(v for v in aux.values() if isinstance(v, torch.Tensor))
    opt.zero_grad(set_to_none=True)
    total.backward()
    opt.step()
    m._reasoning_buffer, m._reasoning_count = r
    for n, p in m.named_parameters():
        if n.startswith('layers.1.w_d') or n.startswith('layers.1.b_d'):
            g = 0.0 if p.grad is None else float(p.grad.abs().sum())
            print(f'it={it} {n}: grad={g:.3e}')
    print('---')
