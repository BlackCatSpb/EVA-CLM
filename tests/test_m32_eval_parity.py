"""M32 lock: the deliberation chain is document state — symmetric across
train/eval, isolated at document boundaries and by the runtime snapshot."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mini():
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, seq_len=64, save_dir='.', logit_cache_enabled=False,
                    memory_bank=False, intent_bridge=False, explicit_reasoning=True,
                    reasoning_max_steps=2, reasoning_adaptive=False,
                    unified_concept_layer=False)
    # NOTE: reasoning_adaptive=False is deliberate — the ADAPTIVE path (production)
    # never persists the chain attrs at all (M33 audit), so the document-chain
    # contract is exercised on the legacy adaptive=False path where it is live.
    torch.manual_seed(0)
    return EVAStack(cfg)


def _fwd(m, x, step):
    m.reasoning_enabled_step = 1500   # past the ramp (s=0.78) — production-like
    h = m.embed_tokens(x)
    with torch.no_grad():
        m(h, None, step=step, tokens=x)


def test_eval_forward_keeps_and_updates_the_chain():
    m = _mini()
    x = torch.randint(1, m.cfg.vocab, (1, 64))
    m.train()
    _fwd(m, x, 1)
    assert m._reasoning_buffer is not None, 'chain absent even in train (setup bug)'
    m.eval()
    _fwd(m, x, 2)
    assert m._reasoning_buffer is not None, \
        'M32 regression: eval erased the chain again (train/eval CE asymmetry)'


def test_snapshot_restore_protects_the_train_chain():
    m = _mini()
    x = torch.randint(1, m.cfg.vocab, (1, 64))
    m.train()
    _fwd(m, x, 1)
    pre = m._reasoning_buffer.detach().clone()
    snap = m.snapshot_runtime_buffers()
    m.eval()
    _fwd(m, x, 2)                    # val pass mutates the chain
    m.restore_runtime_buffers(snap)  # train document must be byte-identical
    assert torch.allclose(m._reasoning_buffer, pre, atol=1e-6)


def test_evaluators_reset_per_document_and_carry():
    t = open(os.path.join(ROOT, 'scripts', 'train.py'), encoding='utf-8', errors='replace').read()
    i = t.find('def evaluate')
    seg = t[i:i + 3200]
    assert 'reset_reasoning()' in seg and 'global_state=ogs' in seg
    nb = json.load(open(os.path.join(ROOT, 'notebooks', 'eva_colab.ipynb'), encoding='utf-8'))
    s10 = ''.join(''.join(c.get('source', [])) for c in nb['cells'] if 'TRAINING LOOP' in ''.join(c.get('source', [])))
    assert 'global_state=vgs' in s10
    assert s10.count('model.reset_reasoning()') >= 2  # per-document + boundary
