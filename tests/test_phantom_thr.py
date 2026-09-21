"""T9.7b: the phantom observation threshold + the salience ladder.

The lacuna -> PhantomBank -> confirmed -> UCL -> grow_phantom_bits lifecycle was
dead at birth: the observation gate compared the relative salience against the
legacy constant 1.1, which sat ABOVE the whole signal range (p99 ~1.08), so
ph_obs stayed 0 forever. These tests pin the replacement:

  * `noise_threshold` — the self-calibrating bar (median + k*MAD of the recent
    per-call salience maxima, clamped). NOT a quantile: on a stationary stream
    the width collapses and equal values never fire; a genuine exceedance does.
  * `ladder_rung` — the gates read different rungs of the SAME ladder by tau
    (the ~128 rung for lacuna/temper, the longest rung for the observation).
  * the wiring: the head takes the ladder from the config (aligned with the
    cache ms-ladder up to 8192) and the observation actually fires on a spike.
"""
import torch

from core.config import EVAConfig
from core.embedding import SigmoidCodedHead, ladder_rung, noise_threshold

LADDER = (8.0, 32.0, 128.0, 512.0, 2048.0, 8192.0)


def test_ladder_rung_nearest_in_log_space():
    t = torch.tensor(LADDER)
    assert ladder_rung(t, 128.0) == 2
    assert ladder_rung(t, 100.0) == 2          # log-nearest to 128, not 512
    assert ladder_rung(t, 8192.0) == 5
    assert ladder_rung(t, 3000.0) == 4         # between 2048 and 8192 -> 2048


def test_ladder_rung_zero_selects_the_longest():
    t = torch.tensor(LADDER)
    assert ladder_rung(t, 0.0) == 5            # the base rate = the longest rung
    assert ladder_rung(t, -1.0) == 5
    assert ladder_rung(torch.zeros(0), 128.0) == 0


def test_noise_threshold_warmup_uses_fallback():
    ring = torch.full((64,), 1.01)
    assert noise_threshold(ring, 8, 2.0, 0.002, 1.0005, 1.10, 16, 1.1) == 1.1


def test_noise_threshold_calm_stream_never_fires_at_the_base():
    ring = torch.full((64,), 1.0)              # a perfectly calm stream AT the base
    thr = noise_threshold(ring, 64, 2.0, 0.002, 1.0005, 1.10, 16, 1.1)
    assert thr == 1.002                        # 1 + floor (the width is degenerate)
    assert not (1.0 > thr)                     # values at the base are NOT observed


def test_noise_threshold_spike_fires_typical_does_not():
    torch.manual_seed(0)
    quiet = 1.001 + 0.0005 * torch.randn(64)   # the maxima hover just above the base
    thr = noise_threshold(quiet, 64, 2.0, 0.002, 1.0005, 1.10, 16, 1.1)
    assert 1.0 < thr < 1.005                   # a modest bar over the base rate
    assert not (1.0005 > thr)                  # a base-level value is not observed
    assert 1.08 > thr                          # a real spike (p99 ~1.08) fires


def test_noise_threshold_scales_with_the_noise():
    torch.manual_seed(0)
    calm = 1.002 + 0.0002 * torch.randn(64)
    wild = 1.002 + 0.02 * torch.randn(64)
    thr_calm = noise_threshold(calm, 64, 2.0, 0.002, 1.0005, 1.10, 16, 1.1)
    thr_wild = noise_threshold(wild, 64, 2.0, 0.002, 1.0005, 1.10, 16, 1.1)
    assert thr_wild > thr_calm                 # the bar rises with the noise
    assert thr_wild <= 1.10                    # ... but is clamped


def test_noise_threshold_clamps():
    # a wild stream: the bar rises with the noise but is clamped at hi
    torch.manual_seed(0)
    wild = 1.0 + 0.2 * torch.randn(64)
    assert noise_threshold(wild, 64, 2.0, 0.002, 1.0005, 1.10, 16, 1.1) == 1.10
    # a calm constant stream at the base: the floor wins (1 + 0.002)
    assert noise_threshold(torch.full((64,), 1.0), 64, 2.0, 0.002,
                           1.0005, 1.10, 16, 1.1) == 1.002
    # an explicit low clamp binds when the floor is smaller than it
    assert noise_threshold(torch.full((64,), 1.0), 64, 2.0, 0.0001,
                           1.0005, 1.10, 16, 1.1) == 1.0005


def _mk_head():
    cfg = EVAConfig(D=32, vocab=64)
    cfg.head_lacuna_ladder = LADDER
    head = SigmoidCodedHead(cfg)
    head.train()
    head._pb_active = True
    head._tokens = None                        # per-position observations
    return head


def test_head_takes_the_cache_aligned_ladder():
    head = _mk_head()
    assert head._ladder_tau.tolist() == list(LADDER)
    assert head._gate_rung == 2                # the ~128 rung (lacuna/temper)
    assert head._base_rung == 5                # the 8192 rung (the base rate)
    assert head._thr_mode == 'noise'


def test_observation_fires_on_a_novelty_spike():
    torch.manual_seed(0)
    head = _mk_head()
    B, L, K, D = 1, 4, head.K, head.D
    u = torch.zeros(B, L, K)
    h_in = torch.randn(B, L, D)
    quiet = torch.randn(B, L, D) * 0.01
    for i in range(20):                        # fill the salience ring
        head._pb_step.fill_(head.phantom_every * (i + 1))
        head._phantom_mix(u, quiet, h_in)
    obs0 = head.phantom_bank.stats()['obs']
    head._pb_step.fill_(head.phantom_every * 30)
    head._phantom_mix(u, quiet * 30.0, h_in)   # the lacuna spike
    assert head.phantom_bank.stats()['obs'] > obs0
    assert head._meta_thr is not None
    assert 1.0005 <= float(head._meta_thr) <= 1.10
