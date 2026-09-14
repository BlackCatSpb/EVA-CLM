"""M46 lock: train CE is surprisal-WEIGHTED (<= unweighted); EVAL CE is the
unweighted metric — the two are only comparable via ce_raw."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack  # noqa: E402


def _m():
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, seq_len=32, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=False, explicit_reasoning=False,
                    unified_concept_layer=False, surprisal_weight=0.3)
    torch.manual_seed(0)
    return EVAStack(cfg).train(), cfg


def test_ce_raw_is_the_unweighted_metric():
    m, cfg = _m()
    x = torch.randint(1, cfg.vocab, (1, cfg.seq_len))
    h = m.embed_tokens(x)
    out, _, _, _ = m(h, None, step=5, tokens=x)
    ce, _ = m.compute_losses(out, x, h_emb=h)
    raw = m._cached_losses['ce_raw']
    assert raw >= float(ce) - 1e-6, (raw, float(ce))   # weights are <= 1
    m.eval()
    out2, _, _, _ = m(h, None, step=None, tokens=x, adaptive=False)
    ce2, _ = m.compute_losses(out2, x, h_emb=h)
    assert abs(float(ce2) - m._cached_losses['ce_raw']) < 1e-6, 'eval must be unweighted'
