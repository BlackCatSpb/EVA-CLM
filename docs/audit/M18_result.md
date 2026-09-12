# M18: code-profile cache + spiral age-addressing — final report

Date: fresh-run era, post B18. Data: real ANECDOTS stream (wb/), twin_free
K=64 S=6 codes, centered window bit-profiles (512 windows x 32 tokens).

## M18-A (SHIPPED, default OFF): `logit_cache_mode='profile'`

GPT-6's mechanism finding was correct: legacy inference read rebuilt
(B,M,V) + V@C on EVERY step and top-k64 discarded the logit-field tail.
Profile mode stores p = tanh(z/10)@C on write (K fp16); read is p->(K,V).
Locks: `tests/test_m18_profile_cache.py` (exactness, read path never
enters V-space, stack smoke). Byte parity at k=46 — the win is compute +
tail fidelity, NOT capacity. Not enabled on the money run (topk default):
revisit if inference read cost shows in the [ggeo]/step-time budget.

## M18-B (REJECTED as architecture): spiral phase addressing on real data

Hypothesis (mine + two external LLMs): golden/Fibonacci per-plane phase tags
turn the tape into an age-addressable superposition memory; codebook rank
doubles capacity (K=32 wall at N~64 predicted to move to N~128 at K=64).

Measured (purity = target-energy fraction of age-query readout):

N      none      random    fib(K=64)   fib(K=32, prev run)
 16    0.738     0.802     0.839       0.857
 32    0.611     0.683     0.710       0.705
 64    0.504     0.449     0.551       0.574
128    0.446     0.333     0.443       0.479
256    0.422     0.239     0.403       0.409
512    0.411     0.194     0.367       0.389

Verdict:
1. K=32 -> K=64 moved the wall NOT AT ALL (0.389 -> 0.367 = noise). The
   binding constraint is the INTRINSIC rank of centered window-mean profiles
   (~30 effective dims of variance at S=6 regardless of K) and real content
   correlation (mean cos 0.30 after centering), not the nominal codebook rank.
2. 'none' (pure content memory, no age axis) beats every phase scheme at
   large N. Age-addressed superposition pays purity it can never recover:
   content addressing + exact windows remain the right primitive.
3. Golden single-spiral ~= random (as before); fib multi-spiral best spiral —
   the earlier √N-interference theory was real but measures a secondary term;
   the primary term is content crosstalk in a rank-30 subspace.

Decision: spiral tape CLOSED. Architecture stays: exact logit-cache windows
(what the model already turns on via cache_gate) + VSA multi-scale state for
the compressed far past (now actually multi-scale after B18) + memory banks.
Phase tags may return ONLY as a write-time AGE FEATURE (one extra input dim
to k/v projections, learned, no addressing theory) — not scheduled.
