"""Shared mini-model harness for the correctness audit (scratch, read-only repo)."""
import os
import sys

import torch

REPO = r'C:\EVA_CLM_OPT'
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.config import EVAConfig  # noqa: E402
from core.stack import EVAStack  # noqa: E402


def mini_cfg(**kw):
    d = dict(D=64, n_layers=2, mlp_groups=4, code_dim=16, code_sparsity=4,
             vocab=256, logit_cache_enabled=False, gradient_checkpointing=False,
             save_dir=r'C:\EVA_CLM_OPT\_audit_A')
    d.update(kw)
    return EVAConfig(**d)


def build(**kw):
    torch.manual_seed(0)
    cfg = mini_cfg(**kw)
    m = EVAStack(cfg)
    m.eval()
    return m, cfg


def fwd(m, cfg, x, state=None, gs=None, step=None, adaptive=False):
    h = m.embed_tokens(x)
    out, state, gs, rb = m(h, state, global_state=gs, adaptive=adaptive,
                           tokens=x, step=step)
    return out, state, gs, rb


def ce(m, cfg, out, y, h_emb=None):
    return m.compute_loss(out, y, h_emb=h_emb)
