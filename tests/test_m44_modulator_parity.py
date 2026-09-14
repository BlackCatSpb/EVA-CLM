"""M44 lock: the per-layer branch modulators must be computed from the SAME
carried mirror stats in train and eval (the 13.4-vs-5.4 val autopsy)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack  # noqa: E402


def _model():
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, seq_len=32, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=False, explicit_reasoning=False,
                    unified_concept_layer=False)
    # noise is a train-only regularizer; zero it so the parity test measures
    # ONLY the modulator path.
    cfg.noise_scale_min = cfg.noise_scale_max = 0.0
    torch.manual_seed(0)
    return EVAStack(cfg).train(), cfg


def test_modulators_parity_train_vs_eval():
    m, cfg = _model()
    x = torch.randint(1, cfg.vocab, (1, cfg.seq_len))
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    # warm the mirrors' carried stats with a few real training steps
    for _ in range(3):
        h = m.embed_tokens(x)
        out, st, gs, _ = m(h, None, step=5, tokens=x)
        ce, aux = m.compute_losses(out, x, h_emb=h)
        from core.training_control import LossBalancer
        opt.zero_grad(set_to_none=True)
        LossBalancer(align=True).backward(ce, aux, m.parameters(), phase_model=m)
        opt.step()

    def fwd(step, adaptive):
        h = m.embed_tokens(x)
        with torch.no_grad():
            out, _, _, _ = m(h, None, step=step, tokens=x, adaptive=adaptive)
            ce, _ = m.compute_losses(out, x, h_emb=h)
        return float(ce)

    m.train()
    ce_train = fwd(5, True)
    m.eval()
    ce_eval = fwd(None, False)   # notebook eval shape: step=None, adaptive=False
    rel = abs(ce_train - ce_eval) / max(ce_train, 1e-9)
    assert rel < 0.02, (
        f'M44 regression: eval runs different branch modulators '
        f'(train={ce_train:.4f} eval={ce_eval:.4f}, rel={rel:.3f})')
