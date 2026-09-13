"""M34 lock: the logit-cache ring is τ-horizon bound and novelty-pruned.

Design contract this pins:
  * default horizon = cfg.tau_max (the top VSA scale — one dynamic range for
    the exact cache and the lossy ladder);
  * eviction = retention score novelty*exp(-age/tau), never the newest entry;
  * cache_horizon_tokens<0 = legacy blind FIFO (max_entries ring);
  * clear() wipes the bookkeeping lists too.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack                      # noqa: E402
from core.logit_cache import LogitCache                   # noqa: E402


def _mini(**kw):
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, seq_len=32, save_dir='.', logit_cache_enabled=True,
                    logit_cache_max_entries=64, memory_bank=False,
                    intent_bridge=False, explicit_reasoning=False,
                    unified_concept_layer=False, **kw)
    torch.manual_seed(0)
    return EVAStack(cfg), cfg


def test_novelty_prunes_not_age():
    c = LogitCache(50, 8, 64, n_scales=4, horizon_tokens=128)
    for i, n in enumerate([3.0, 1.0, 1.0, 1.0, 1.0, 5.0]):
        c.store(torch.zeros(1, 64, 8) + i, training=True, novelty=n)
    assert c._h_cache[-1][0, 0, 0].item() == 5.0, 'newest entry must never be evicted'
    assert sum(c._h_lens) <= 128
    assert 3.0 in c._h_scores, ('high-novelty history must outlive low-novelty '
                                f'neighbours; scores={c._h_scores}')
    c.clear()
    assert not c._h_cache and not c._h_lens and not c._h_scores


def test_legacy_fifo_when_horizon_zero():
    c = LogitCache(50, 8, 3, n_scales=4, horizon_tokens=0)
    for i in range(5):
        c.store(torch.zeros(1, 8, 8) + i, training=True)
    assert len(c._h_cache) == 3 and c._h_cache[-1][0, 0, 0].item() == 4.0
    assert c._h_cache[0][0, 0, 0].item() == 2.0   # pure age order


def test_stack_wires_auto_horizon_and_runs():
    m, cfg = _mini()
    assert m.logit_cache.cache.horizon_tokens == int(cfg.tau_max)
    model = m.train()
    x = torch.randint(1, cfg.vocab, (1, cfg.seq_len))
    state = None
    cache = model.logit_cache.cache
    for _ in range(12):
        h = model.embed_tokens(x)
        out, state, gs, _ = model(h, state, step=1, tokens=x)
        state = tuple(
            tuple(t.detach() if isinstance(t, torch.Tensor) else t for t in s)
            if isinstance(s, (tuple, list)) else s
            for s in state) if state else state
    span = sum(cache._h_lens)
    assert span <= int(cfg.tau_max) + max(cache._h_lens), f'horizon exceeded: {span}'
    assert len(cache._h_cache) == len(cache._h_lens) == len(cache._h_scores)


def test_m35_oom_ladder_retries_on_device_and_probes_back():
    # The 2026-09-13 live crash: the OOM retry re-read the batch and never
    # moved it to CUDA -> device mismatch in the head gather. The ladder must
    # re-read WITH device placement, be graded (recompute -> batch -> seq),
    # carry the probe-back clock, and keep the floor dump reachable.
    import json as _j
    nb = _j.load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'notebooks', 'eva_colab.ipynb'), encoding='utf-8'))
    s = ''.join(''.join(c.get('source', [])) for c in nb['cells']
                if 'TRAINING LOOP' in ''.join(c.get('source', [])))
    i = s.find('[OOM] retry')
    assert i > 0
    seg = s[max(0, i - 1500):i]
    assert 'x.to(device)' in seg and 'y.to(device)' in seg, 'retry must place the batch on device'
    assert '_laddered' in s and 'halving batch' in s
    assert '_oom_ckpt_step' in s and '[ckpt-probe]' in s
    assert 'halving window' in s
    j = s.find('top CUDA tensor holders')
    assert j > 0 and 'raise' in s[j:j + 400]
