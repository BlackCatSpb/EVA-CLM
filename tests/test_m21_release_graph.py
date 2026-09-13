"""M21: release_step_graph drops cross-step graph pins, keeps values."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack   # noqa: E402


def test_release_keeps_values_drops_graphs():
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=False,
                    memory_bank=True, intent_bridge=True, explicit_reasoning=False,
                    vsa_decay_floor_k=2.0)
    torch.manual_seed(0)
    m = EVAStack(cfg).train()
    x = torch.randint(3, cfg.vocab, (1, 32)); x[:, ::11] = 2
    h = m.embed(x)
    out, st, gs, r = m(h, None, step=1, tokens=x)
    ce, aux = m.compute_losses(out, x, h_emb=h)
    ce.backward()
    held = 0
    vals0 = {}
    for l in m.layers:
        for an in m._GRAPH_ATTRS_BLOCK:
            v = getattr(l, an, None)
            if isinstance(v, torch.Tensor) and v.grad_fn is not None:
                held += 1
                vals0[(id(l), an)] = v.detach().clone()
    assert held > 0, 'expected graph-pinned attrs before release'
    n = m.release_step_graph()
    assert n >= held
    for l in m.layers:
        for an in m._GRAPH_ATTRS_BLOCK:
            v = getattr(l, an, None)
            if isinstance(v, torch.Tensor):
                assert v.grad_fn is None, f'{an} still pinned after release'
            key = (id(l), an)
            if key in vals0 and isinstance(v, torch.Tensor):
                assert torch.allclose(v, vals0[key]), f'{an} VALUE changed on release'
    assert torch.isfinite(m.layers[0]._cache_mlp_out.float()).all()
