"""M31 lock: per-branch LN gives a true residual gradient highway — the
shallow half must receive CE gradient within ~2 orders of the deep half
(old block: L0/L23 = 1e-8 on production; straight-through attempt: inf)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack  # noqa: E402


def _layer_grad_norm(m, li):
    tot = 0.0
    for nm, p in m.named_parameters():
        if p.grad is not None and nm.startswith(f'layers.{li}.'):
            tot += float(p.grad.norm()) ** 2
    return tot ** 0.5


def test_shallow_half_receives_ce_gradient():
    cfg = EVAConfig(n_layers=6, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, seq_len=64, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=False, explicit_reasoning=False,
                    unified_concept_layer=False)
    torch.manual_seed(0)
    m = EVAStack(cfg).train()
    x = torch.randint(1, cfg.vocab, (1, 64))
    h = m.embed_tokens(x)
    out, st, gs, _ = m(h, None, step=5, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    ce.backward()
    g0, g5 = _layer_grad_norm(m, 0), _layer_grad_norm(m, 5)
    assert g0 > 0 and g5 > 0, 'a half of the depth received zero CE-grad'
    ratio = g0 / g5
    assert ratio > 1e-3, f'depth attenuation is back: L0/L5 = {ratio:.2e}'
    for nm, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f'grad overflow in {nm}'
