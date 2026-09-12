"""B9 (audit 02a): embedding common-mode centering, cfg-gated (default off)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack   # noqa: E402


def _mk(center):
    cfg = EVAConfig(n_layers=1, D=256, mlp_groups=4, code_dim=64, code_sparsity=6,
                    codebook='twin_free', vocab=65536, head_mode='sigmoid_coded',
                    head_normalize=True, embed_center=center, save_dir='.',
                    logit_cache_enabled=False, memory_bank=False, intent_bridge=False)
    torch.manual_seed(0)
    return EVAStack(cfg).train()


def _stats(m):
    with torch.no_grad():
        toks = torch.randint(0, m.embed.codes.shape[0], (1, 2048))
        e = m.embed(toks)[0]
        ec = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cm = ec @ ec.T
        cm.fill_diagonal_(0)
        top1 = (m.lm_head(e).argmax(-1) == toks[0]).float().mean().item()
        return float(cm.mean()), top1


def test_embed_center_default_off_leaves_geometry_untouched():
    m = _mk(False)
    assert not m.embed.embed_center
    cos, top1 = _stats(m)
    assert cos > 0.9 and abs(top1 - 1.0) < 1e-6   # matches the 02a measured baseline


def test_embed_center_kills_common_mode_and_preserves_identity():
    m = _mk(True)
    assert m.embed.embed_center
    cos, top1 = _stats(m)
    assert cos < 0.05, f'centering should collapse common-mode, got cos={cos:.3f}'
    # identity path MUST survive (it is a constant shift absorbed by the head):
    assert abs(top1 - 1.0) < 1e-6, f'roundtrip broke under centering: top1={top1:.3f}'


def test_embed_center_state_dict_roundtrips_and_is_persist_free():
    m = _mk(True)
    sd = m.embed.state_dict()
    m2 = _mk(True)
    m2.embed.load_state_dict(sd)                   # buffers persistent=False must not block
    assert torch.allclose(m.embed._sig_mean, m2.embed._sig_mean)
