"""B11: locks for audit 02b follow-ups (d_mod centering + eval gate parity)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack                    # noqa: E402



def _cfg():
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=False, maturation_enabled=True)
    torch.manual_seed(0)
    return cfg


def test_eval_gate_matches_training_combined_gate():
    cfg = _cfg()
    m = EVAStack(cfg).train()
    m.maturation.readiness.data.fill_(0.5)          # fake competence
    m(m.embed(torch.randint(0, cfg.vocab, (1, 32))), None, step=1)
    # raw ramp at step 1 is ~0; the PUBLISHED gate must carry the readiness max
    assert float(m.maturation.gate.min()) > 0.4, \
        'combined gate not published for eval (F2B-04 regression)'
    g_train = m.maturation.gate.clone()
    m.eval()
    with torch.no_grad():
        out = m(m.embed(torch.randint(0, cfg.vocab, (1, 32))), None)
    assert torch.allclose(m.maturation.gate, g_train), 'eval mutated the gate'
    m.train()
