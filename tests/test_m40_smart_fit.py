"""M40 lock: SmartController is fitted to the coded head — single-source tau,
relative-to-EMA entropy adaptation (absolute Hn=H/ln(V) froze <0.4 under summed
Bernoulli log-odds, so top-k/temp never adapted), log_temp-normalized sampling."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack                      # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'scripts'))
from smart_controller import SmartController              # noqa: E402


def _model():
    cfg = EVAConfig(n_layers=3, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, seq_len=32, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=False, explicit_reasoning=False,
                    unified_concept_layer=False)
    torch.manual_seed(0)
    return EVAStack(cfg).eval(), cfg


def test_tau_comes_from_the_unified_field():
    m, cfg = _model()
    c = SmartController(m, cfg.vocab)
    want = [float(x) for x in m.tau_config.tau_l]
    assert len(c.tau_l_vec) == len(want)
    assert all(abs(a - b) <= 1e-4 * max(1.0, abs(b))
               for a, b in zip(c.tau_l_vec, want)), (c.tau_l_vec, want)


def test_relative_entropy_adapts_temp_and_topk():
    m, cfg = _model()
    c = SmartController(m, cfg.vocab)
    torch.manual_seed(1)
    g = torch.randn(cfg.vocab)
    sharp = g * 15.0                             # very low H (coded regime)
    wide = g * 4.0                               # flatter tail: H ~2-3x own base
    tempA, topkA = [], []
    for _ in range(14):
        t, tp, tk, rp, al = c.decide(sharp, {'trust_max': 0.9}, 0)
        tempA.append(t); topkA.append(tk)
    tempB, topkB = [], []
    for i in range(10):
        t, tp, tk, rp, al = c.decide(wide, {'trust_max': 0.9}, i + 1)
        tempB.append(t); topkB.append(tk)
    assert c._H_ema > 0
    # relative entropy MUST move the knobs under a 1.6x own-baseline jump;
    # the old absolute Hn≈0.2-0.3 never crossed any threshold -> constants.
    assert sum(tempB) / len(tempB) > sum(tempA) / len(tempA) + 1e-3, (
        'temperature did not rise with relative entropy')
    assert min(topkB) < max(topkA), 'top_k did not narrow with relative entropy'


def test_sample_truncates_top_k_and_normalizes_temp():
    m, cfg = _model()
    c = SmartController(m, cfg.vocab)
    logits = torch.zeros(cfg.vocab)
    logits[7] = 10.0
    logits[3] = 9.0
    logits[11] = 8.0
    picked = {c.sample(logits.clone(), 1.0, 1.0, 3, 0.0) for _ in range(120)}
    assert picked <= {7, 3, 11}, f'top_k=3 leaked outside the top: {picked}'
    ref = math.exp(min(c._temp_ref, 10.0))
    want = float(m.lm_head.log_temp.detach().mean().exp())
    assert abs(ref - want) < 1e-3
