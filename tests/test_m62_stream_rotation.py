"""M62 lock: the genre-rotation sampler + the bank-decay knob.

Why: the novelty machinery (lacuna gate, phantom bank, UCL births) fires on
DISTRIBUTION SHIFTS. The exhaustion-only sampler rotated once per ~1200 steps
(33M tokens) — ~6 shifts per 7.3k steps, so the chain had almost nothing to
test on, and the switch interval exceeded the bank's confirm-decay time
constant (measured: ph_confirmed 2 -> 0, ph_conf 0.24 -> 0.06 in one genre
block). M62 adds `stream_chunk_steps` (rotate every N steps, never into the
same genre) and exposes the bank's `decay` as `head_phantom_decay`.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core.phantom import PhantomBank  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _pick_stream_src():
    """Exec the sampler helper out of train.py (it guards side effects under
    __main__, but importing the whole script pulls the training stack)."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'scripts', 'train.py')
    src = open(p, encoding='utf-8').read()
    i0 = src.index('def _pick_stream')
    i1 = src.index('\n\n\ndef train(', i0)
    ns = {'torch': torch}
    exec(compile(src[i0:i1], 'pick', 'exec'), ns)
    return ns['_pick_stream']


def test_config_knobs_default_to_the_legacy_behaviour():
    cfg = EVAConfig(**SMALL)
    assert cfg.stream_chunk_steps == 0, 'rotation must be opt-in'
    # M64 (R1/R2 calibration): 0.99 per OBSERVE (~69 observes ~ 287 steps to
    # fade 0.5 -> 0.25 at the measured ~0.24 observes/step), not the old 0.999
    assert abs(cfg.head_phantom_decay - 0.99) < 1e-12, 'the M64 decay default'


def test_pick_stream_never_repeats_the_current_genre():
    pick = _pick_stream_src()
    rng = torch.Generator().manual_seed(42)
    n = 8
    for cur in range(n):
        got = {pick(cur, n, True, rng) for _ in range(400)}
        assert cur not in got, f'a same-genre "switch" was drawn (cur={cur})'
        assert got == set(range(n)) - {cur}, f'the pick skipped a genre: {got}'


def test_pick_stream_legacy_path_keeps_the_uniform_pick():
    pick = _pick_stream_src()
    rng = torch.Generator().manual_seed(42)
    got = {pick(3, 8, False, rng) for _ in range(800)}
    assert got == set(range(8)), 'the legacy pick must cover all streams'
    # deterministic for a given seed
    a = pick(3, 8, False, torch.Generator().manual_seed(7))
    b = pick(3, 8, False, torch.Generator().manual_seed(7))
    assert a == b


def test_pick_stream_single_stream_is_safe():
    pick = _pick_stream_src()
    rng = torch.Generator().manual_seed(0)
    assert pick(0, 1, True, rng) == 0
    assert pick(0, 0, True, rng) == 0


def test_bank_decay_knob_reaches_the_bank():
    torch.manual_seed(0)
    m = EVAStack(EVAConfig(**{**SMALL, 'head_phantom_decay': 0.9995}))
    assert abs(m.lm_head.phantom_bank.decay_rate - 0.9995) < 1e-12
    # the decay math itself: one call multiplies the confidences
    b = PhantomBank(n_slots=2, D=8, decay=0.5)
    b.confidence[0] = 0.8
    b.decay()
    assert abs(float(b.confidence[0]) - 0.4) < 1e-6
