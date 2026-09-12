# 02A CODE GEOMETRY AUDIT

Agent #2a of the serial relay. Scope: **the code-geometry stack** — token id → code →
embedding, and the mirror image hidden → logits → token identity. Files under the
microscope: `core/vsa_utils.py` (`build_codes`, `twin_free_codes`,
`sparse_block_codes`), `core/embedding.py` (`PartitionedEmbedding`,
`SigmoidCodedHead`/`log_probs_for_target`, `_readout_rotated`, `code_sparsity`/
`code_dim` semantics), the cell-4 production knobs (`codebook='twin_free'`,
`code_dim=64`, `vocab=65536`, `head_mode='sigmoid_coded'`, `head_normalize=True`),
and `core/logit_cache.py::bit_profile` (same codebook object — coherence is this
audit's jurisdiction).

Mandatory input honored: `docs/audit/01_data_path.md` read in full (947 lines).
§2 (shapes/dtypes/RNG at the boundary) is taken as the contract, not re-audited;
the §5 handoff addressed to 2a (F-01/F-07/F-18, the `K=64` head and
`migrate_state_dict` questions) is answered point-by-point in §3. The tree is at
`c152f28` (B7) — two commits past the B6 baseline; where B7 touched my scope the
state is recorded honestly (fixed / needs-lock) rather than re-litigated.

Method notes.
- All anchors were read from the working tree; quoted lines are verbatim.
- Executed repros live in `%TEMP%\opencode\`: `repro2a_main.py` (results
  `repro2a_results.txt`), `repro2b_margin.py` (results `repro2b_results.txt`),
  `repro2c_ckpt.py` / `repro2d_ckpt2.py` (the on-disk `checkponts/best.pt`,
  results `repro2c_ckpt.txt` / `repro2d_ckpt2.txt`), `repro2a_hash.py`
  (cross-process codebook fingerprints). No repo file was modified;
  `git status --porcelain` clean at analysis time.
- Real corpus: `WideBind\wb\token_stream_{FANTASY,CHILDREN,DICTION}_eos.bin`
  windows (400k + 200k + 200k tokens, memmap) for the corpus-weighted statistics.
- Environment: torch 2.13.0+cpu, Python 3.14, Windows. Production geometry built
  directly (`D=2560, K=64, d=40, V=65536, twin_free`), the 24-layer trunk NOT
  executed (CPU); mini stacks (test-suite geometry, `D=512, K=16, V=1820`) used
  where a full forward was required.
- Confidence tags: **VERIFIED-REPRO** (executed), **STRONG-READ** (anchor-verified
  code path, deterministic consequence), **SPECULATIVE** (plausible, unproven).
- Suite baseline: `python -m pytest tests -q` → **287 passed in 54.15s** (run once,
  after all analysis; repo untouched). The mission brief quoted 285 — the B7 commit
  added three locks on top of agent 1's 284; the measured current number is 287.

Conventions: `vsa_utils.py:N` = `core/vsa_utils.py`. `embedding.py:N`,
`logit_cache.py:N`, `stack.py:N`, `training_control.py:N`, `config.py:N`,
`train.py:N` = `scripts/train.py`, `c7:N` / `c10:N` = notebook cell source lines
of cells 7 ("Data Streams") and 10 ("TRAINING LOOP").

---

## 1. Findings

### F2A-01 — The twin_free guarantee HOLDS at full production scale, exactly and verifiably; its margin is rare but thin | INFO (constitution fact) | VERIFIED-REPRO

The claim under audit (`vsa_utils.py:66-70`): constant-weight-(6) codes packed
greedily with max pairwise overlap ≤ S−2 = 4 ⇒ d_H ≥ 4; "measured knee T: 550 →
≥1200 at equal capacity".

Executed against the real `twin_free_codes(65536, K=64, S=6)` (build 1.74 s,
16.8 MB):

- Row weights: min=max=6 on all 65 536 rows (`all==6: True`).
- Row uniqueness: 65 536/65 536 distinct (int64-packed supports).
- **Exact certificate** (not sampling): every 5-subset of every code hashed to a
  packed key; 393 216 keys, **max multiplicity 1** ⇒ no two codes share 5+
  positions ⇒ gram off-diagonal max = **4 = S−2** ⇒ **d_min = 4** EXACTLY, for all
  2.147 G pairs.
- The bound is *attained*: 711 504 unordered pairs sit at overlap 4 (d_H = 4)
  — 0.033 % of pairs, mean 21.7 margin-1-bit rivals per code (min 3, p05 15, max 48).
- Sampled distance distribution (4 M pairs, incl. 53 self-draws): overlap hist
  [0:2 161 285, 1:1 465 037, 2:339 224, 3:33 074, 4:1 326, 5:0];
  **d_H mean 10.876, var 1.879, std 1.371**, min 4, max 12. Mean overlap
  0.5621 vs 0.5625 for unrestricted random codes — the packing removed the tail
  only, leaving first moments identical to random.
- Bit marginals are nearly perfect: p̄_k ∈ [0.09088, 0.09590], std 0.00103
  (ideal 6/64 = 0.09375) despite the greedy's randomness — the screening does not
  bias columns. `Cᵀ C`: full rank 64, eig ∈ [5340, 36 869], cond 6.90.
- The constraint is *doing work*: with `max_overlap=S−1=5` the same builder fits
  without objection (twins exist), and a random 20 k weight-6 pool contains 937
  pairs at overlap ≥5 (incl. full twins) — twin_free removes every one.
- The companion claim "the K=32 pool genuinely can't hold 65 k" verified:
  `twin_free_codes(65536, K=32, S=6)` packs only **20 518/65 536** then raises
  `ValueError` (11.9 s, loud).

Init-stage pair margins (the identity-path budget, `repro2b`): for all 711 504
d_H=4 pairs, in *both* query directions, score_true−score_rival in u-units at
cell-4 init: **min +0.2993, 1 % quantile +0.542, mean +0.768, correct-order
fraction 1.000000**. The identity path is alive by a certified positive margin on
every nearest neighbor at init — but the worst-case budget is ≈0.30 u-units, i.e.
the d_H=4 geometry gives 2 bits of Hamming slack and no more; noise-like erosion
of per-bit separation beyond ~80 % of the 0.377 init signal flips nearest
neighbors (see F2A-09 for the trained-state probe).

Suggested fix: none — this is a constitution fact.
Locks: **PARTIAL / MISDIRECTED** — `test_b2_regressions.py:148-160` verifies
overlap ≤ max only for a `(400, K=24, S=5, max_overlap=3)` pool and "determinism"
by calling `twin_free_codes` twice **in-process** — the second call hits
`_CODES_CACHE` (`vsa_utils.py:80-82`) and returns a clone, so the test locks the
cache, not the greedy. **The production tuple (65 536, 64, 6, seed 42) has no
fingerprint lock anywhere** (see F2A-02).

### F2A-02 — Codebooks are rebuild-only and un-fingerprinted: every trained geometry rides on an implicit promise that the greedy never changes | MEDIUM | VERIFIED-REPRO (positive case) + STRONG-READ (hazard) | NEEDS-LOCK

Target 1's sharp question: *codes persisted in checkpoint vs rebuilt — is the
rebuild bit-identical to the trained one?*

Actual code path, end to end:
- `build_codes(cfg)` (`vsa_utils.py:119-125`) dispatches on `cfg.codebook`;
  for `'twin_free'` it calls `twin_free_codes(cfg.vocab, K=cfg.code_dim,
  S=cfg.code_sparsity)` — **seed is not a cfg field**: the default `seed=42`
  (`vsa_utils.py:64`) is the entire identity contract. Same for `batch=2048` and
  the 4096-code screening chunk (`:92`), the intra-batch drop-both conflict rule
  (`:98-104`) and the empty-rounds abort (`:108-111`) — each one changes the
  greedy's RNG consumption order and therefore the whole assignment.
- Embedding and head register the codes as **`register_buffer('codes', codes,
  persistent=False)`** (`embedding.py:89/214/243/336`) → they are *not* in
  `state_dict()` (measured: `state_dict has embed.codes: False |
  lm_head.codes: False | cache codes_t: False`), and the best.pt envelope
  (keys measured: `step, model, optimizer, param_names, scheduler,
  best_val_loss, cfg, reasoning_enabled_step, recover_count, active_depth,
  detector, balancer, stream_idx, offset, rng, data_rng`) carries **no codebook
  hash, no codebook version** — `cfg` is the only re-materializer.
- Rebuild determinism was proven the hard way: sha256 of the (65 536×64) fp32
  tensor = `74f3f995f303b2ab5c33fc0076431a372951ecb797385654f5e4053098558645`,
  identical across **three fresh Python processes** and **OMP_NUM_THREADS ∈
  {1, 4, 8}**, and identical to an in-process rebuild after `_CODES_CACHE.clear()`
  (1.73 s). The determinism argument is airtight *inside* one (torch, code)
  version: binary-valued matmuls give exact integer overlaps (≤64 < 2²⁴, no
  fp32 ordering sensitivity), and the greedy never touches global RNG.
- Direct empirical hit on the legacy path: the only on-disk checkpoint
  (`checkponts/best.pt`, step 1045, **legacy K=32 geometry** — see F2A-09) *did*
  persist `embed.codes`/`lm_head.codes` (it predates `persistent=False`), and the
  saved matrix is **bit-identical (sha `98fed2eb…`) to the current
  `sparse_block_codes(65536,32,6)` rebuild**. Reconstruction = trained, for real,
  on a real artifact.

The hazard: reconstruction equals training **only while nobody edits the
defaults** — `seed`, `batch`, `max_overlap` derivation, screening order, chunk
size, or the torch CPU RNG/argsort implementation. Any such edit silently
re-materializes a DIFFERENT 65 536×64 assignment under the same cfg, while the
trained `embed_mix/basis/bit_bias/token_bias` still speak the old one — a total,
silent, error-free token-identity scramble at resume. Worse (concretely,
measured): on a current-code resume of an *old* checkpoint that DID persist
codes, `load_state_dict(strict=False)` puts `embed.codes` into the silently
discarded *unexpected* list — the checkpoint's own record of the codebook is
thrown away in favor of the rebuild, with no comparison, no warning.
(`load_state_dict` size-mismatches do raise even under `strict=False` — proven by
stack-trace when I loaded the 24-layer ckpt into a 2-layer model — but *content*
mismatch of same-shaped tensors is exactly what nobody checks.)

Suggested fix (NOT applied): store `codebook_sha256` (plus the build parameters)
in the envelope; on load, hash the rebuilt codes and raise on mismatch; add a
lock asserting sha256 of `twin_free_codes(65536,64,6)` == the recorded constant
and `sparse_block_codes(65536,32,6)` == `98fed2eb…`, with the determinism test
clearing `_CODES_CACHE` (and ideally one subprocess test) so it exercises the
greedy rather than the clone. → persistence doctrine to agent 5.

### F2A-03 — Resume silently re-initializes the ENTIRE identity path when the geometry knob changes: size-mismatch keys are filtered out and training continues with a fresh, untrained embed→head | HIGH (data-envelope integrity) | STRONG-READ

`train.py:272-280`:
```python
        # Filter size-mismatched keys (e.g. L2 slots changed 16->32)
        _model_sd = model.state_dict()
        _filtered = {}
        for k, v in sd.items():
            if k in _model_sd and _model_sd[k].shape != v.shape:
                print(f'  SKIP size-mismatch: {k} ckpt={list(v.shape)} model={list(_model_sd[k].shape)}')
            else:
                _filtered[k] = v
        missing, unexpected = model.load_state_dict(_filtered, strict=False)
```
The model was built earlier from the **CLI** cfg (`train.py:185/192`); the
checkpoint's own `cfg` is saved (`train.py:677`) but **never consulted on resume**
(grep `ckpt['cfg']` → zero hits in train.py; the notebook likewise loads
`ckpt['model']` into the cell-4-built model). Consequence: launching a K=64
cell-4 resume over a legacy K=32 checkpoint (or any vocab change: the pre-B7
default was 50 000 → `token_bias` shape differs) **skips exactly the tensors
`embed.embed_mix, embed.basis, lm_head.readout(=tied), lm_head.bit_bias,
lm_head.log_temp, lm_head.token_bias, lm_head._prop`** — i.e. 100 % of the token
identity machinery reverts to fresh init — while the trained trunk, optimizer
and cursor resume as if nothing happened. Every code of every token now maps
through a *different random* geometry; the head reads it with a *different*
basis. The trunk's learned stream is being fed an un-learned, re-randomized
code-mix; the CE will look "recovering" through the normal short transient and
the model's prior knowledge is scrambled behind three print lines.

Note the asymmetry that makes this easy to miss: *codes* themselves are
shape-stable (V×K changes shape with K but the buffer is non-persistent, so the
codes silently rebuild to the NEW cfg — never in sd), while the *trainable*
geometry is shape-broken — and the filter treats both as routine.

B2's own pathology report (embedding.py:98-103: "the identity path starts dead
and must be un-learned from noise") is exactly what this path reinstalls at
resume.

Suggested fix (NOT applied): maintain an explicit set of **critical-shape keys**
(`embed.*`, `lm_head.*`) whose size mismatch aborts the resume with a loud error
("geometry changed: K/vocab differ from checkpoint — refusing to continue"),
while only truly optional subsystems (L2 slots) may be skipped; additionally
compare `cfg` vs `ckpt['cfg']` code-geometry fields on load and abort on
mismatch. Lock: **NO LOCK** (no test executes the resume filter — agent 1 said
the same about F-11; reconfirmed here).

### F2A-04 — The "rope-in-nesting fix" is shipped UNWIRED: `_readout_rotated` has zero callers; `embed_rope=True` silently degrades the identity path (roundtrip 1.000 → 0.203) | MEDIUM | VERIFIED-REPRO

`embedding.py:180-193` defines `_readout_rotated(head, B, L, D)` with the docstring
claim "Rotating the readout with the SAME rope recovers ⟨Re, Rr⟩ = ⟨e, r⟩
exactly … the code identity path is then alive at initialization". Grep across
the whole repo (py + ipynb): **exactly one occurrence — the definition itself.**
No head ever calls it:

- `SigmoidCodedHead._gates` (`embedding.py:270-271`) reads `h_g · readout` with
  the readout unrotated; `self._embed_rope = rope` is stored (`:242`) and used
  nowhere. `PartitionedHead` (`:215`) and `CognitiveCodedHead` (`:337`) likewise
  store the rope and never apply it at readout.
- Executed probe: production-geometry head with `embed_rope=True`, batch of 256
  real-weighted ids as a length-256 sequence: **argmax decode top-1 = 0.2031**
  vs **1.0000** with the knob off (F2A-05 table). The B2 comment at
  `embedding.py:139-143` honestly reports the old 0.03 disaster and keeps the
  knob "for A/B" — but the compensation the docstring promises is absent, so any
  A/B through this knob measures a *broken* arm, not a rope-in-nesting design.
- The production default is safe: `config.py:64 embed_rope: bool = False`, cell 4
  never sets it → **not currently a live bug**; it is a live trap.

Suggested fix (NOT applied): either wire `_readout_rotated` into
`SigmoidCodedHead._gates` (and the other two heads) when `cfg.embed_rope` is on —
careful, `_gates` receives flattened (N,D) from `log_probs_for_target`, so
position must be carried in — or delete the helper + the knob. Minimum: make
`PartitionedEmbedding.__init__` warn that the paired head does not rotate its
readout. Lock: **NO LOCK** (roundtrip lock `test_b2_regressions.py:138-145` runs
only the mini legacy stack, knob off).

### F2A-05 — Embedding identity path at production geometry: id→code→mix→basis→D→head roundtrip top-1/top-5 = 1.0000/1.0000 (B2 claim VERIFIED); the embedding manifold is rank-64 of D=2560 by construction | INFO (constitution fact) | VERIFIED-REPRO

Executed at cell-4 geometry (`D=2560, K=64, d=40, twin_free(65536,64,6)`,
`embed_mix = I + 0.05·N(0,1)`@seed7, unit-row basis@seed5, tied readout,
`bit_bias = logit(p̄)` code-prior init):

| id set | n | top-1 | top-5 |
|---|---|---|---|
| corpus-weighted FANTASY/CHILDREN/DICTION (incl. forced ids 0,1,2,65532…65535) | 8 192 | **1.0000** | **1.0000** |
| uniform over V=65 536 | 8 192 | **1.0000** | **1.0000** |
| top-1000 most frequent real ids | 1 000 | **1.0000** | **1.0000** |
| test geometry (K=16, D=256, legacy) — the B2 claim's own turf | 1 820 (all) | **1.0000** | **1.0000** |

The claimed "roundtrip 0.000 → 1.000" (embedding.py:103) is **VERIFIED at
production geometry**, not just the QR test geometry. The reason the production
`K=64 > d=40` non-orthogonality (see F2A-06) does not hurt: the embedding places
`B_k` *inside segment k only* (`embedding.py:138`), and the tied readout dots
segment k against `r_k = B_k` — the segment frames are disjoint in D, so
`z_k = a_k·‖B_k‖² = a_k` exactly (isometry verified numerically: ‖segment‖ ==
a-row for probe ids). The pairwise margin certificate is F2A-01.

Structural fact for everyone downstream: **rank(embedding manifold) = 64 of
D = 2 560** (2 496 dims of the embedding output are identically zero); the
identity signal is a 64-coordinate object in σ(2Mc)-space. The trunk may of
course lift h out of that subspace (it does — conv/mirror write everywhere),
but anything that reads the *token identity channel* reads at most 64
coordinates. This is the ceiling any "code-space unification" proposal (F2A-12,
§3) inherits.

### F2A-06 — Common-mode verdict: 95.1 % of corpus-weighted embedding energy is one shared vector; centering is promising, not a mirage, and after centering the spectrum is flat | INFO (decision input) | VERIFIED-REPRO

Target 2, measured at production init geometry in the exact a-space (verified
isometric to the real forward):

- Activation function as-implemented: `a = σ(2·(cᵀM))` (`embedding.py:135`) —
  active bits mean **0.8800** (σ≈0.88 = σ(2)), inactive bits mean **0.5030**,
  global a-range [0.2281, 0.9549], per-bit signal 0.377. Inactive bits do NOT
  sit at 0 — they sit at 0.5, i.e. **58 of the 64 segment vectors contribute at
  ~half amplitude to every token**: the common mode is structural, not
  corpus-specific.
- Per-bit marginals of the codebook itself are near-uniform (p̄_k std 0.00103,
  F2A-01) ⇒ the DC component is (almost) the same vector for every token; one
  global subtraction suffices.
- **Mean pairwise cosine of embeddings, corpus-weighted: 0.9512** (median
  0.9502, spread of pair-cos 0.0095). Uniform-over-vocab identical (0.9503).
- **Common-mode share (‖μ‖²/E‖e‖²): 95.11 % corpus-weighted / 95.03 % uniform.**
  First (non-centered) principal direction energy share: **95.13 %**.
- After subtracting μ: mean pair-cos **0.0002**, cosine std 0.1888; centered
  spectrum PC1 share drops to **9.3 %**, i.e. the identity signal is spread over
  all 64 dims (centered covariance cond 1 447, worst direction ≈ 0.2 % of PC1 —
  still > 0 everywhere).
- Head-side accounting: the code-prior `bit_bias = logit(p̄)` init
  (`embedding.py:252-258`, the "_prop was dead" fix) already absorbs this DC in
  the *readout* logits at init (h=0 → model output is uniform to 4 decimals,
  see F2A-07), which is precisely why the roundtrip survives the 95 % DC. It
  does not absorb it for the *trunk*: conv/mirror/bridge/memory cosine geometries
  see embeddings at cos 0.9512 against each other from step 0.

Verdict for the pending 'centering' experiment: **promising, quantitatively** —
19.1× of embedding energy is a removable constant (rank-1, exactly known, no
information loss: after centering the pair-cos distribution is zero-mean, and
the identity channel is fully preserved in the residual: the active-bit bump is
0.377 wide vs post-centering noise ~0.02-0.06). A single learnable/fixed μ
subtraction at the embedding output (or per-segment mean removal inside the
block) has a large geometric payoff budget; the risk it should be tested against
is not signal loss (there is none) but interaction with the head's tied prior
(centering changes the h=0 calibration — the head must see the centered
features or keep its own copy of μ).

### F2A-07 — Head loss-mode geometry measured: uniform CE = ln V = 11.0904 exactly; factorized branch zero-information = K·ln2 = 44.36 and the veto ceiling sits UNDER it; code-prior init CE 9.05 (identity channel) | MEDIUM (the normalize=False leg) | VERIFIED-REPRO

Target 4, all executed:
- **normalize=True (production)**: head at `h = 0` emits per-position entropy
  **11.0904 nats = ln 65 536** to 4 decimals (max prob 2×10⁻⁵ ≈ 1.3/V) — the
  "uniform CE = ln V" reference of F-07 confirmed empirically, not just
  algebraically. Veto arithmetic at cell-4 config: hard ceiling
  `hard_veto_ceiling(65536)` = 2·ln V = **22.1808** (B7 single source,
  `training_control.py:404-414`); healthy cold start quoted by mission at ~15.5;
  garbage incidents at 34+. Band holds: 15.5 < 22.18 < 34 (1.43× above the cold
  line, 1.53× below garbage). At V=50 000 the ceiling is 21.66 — still > 15.5.
  `test_b7_regressions.py:23-30` locks exactly these inequalities. **B7 status:
  FIXED for the production branch** (both copies call the single source:
  `train.py:443`, c10:145-147).
- **normalize=False (the dormant bit-BCE leg, `embedding.py:318-322`)**:
  its zero-information value is K·ln2 = **44.361 > 22.181 = the ceiling**. Measured
  init CE on real ids at the code-prior init: **18.81** — i.e. a factorized-branch
  run passes the veto *only* thanks to the bit_bias prior init, and the veto
  fires on genuine information loss (drift toward uniform) at ~half the branch
  scale: the geometric margin between "prior-init healthy" (18.81) and "the
  ceiling" (22.18) is 3.4 nats. Any future head_normalize=False run (the knob is
  live in cfg, `config.py:61`) inherits a ceiling that is NOT the branch's
  uniform — the same *form* of geometry/loss mismatch B7 just eliminated, on the
  other leg. Suggested fix (NOT applied): `hard_veto_ceiling` should take the
  branch: `2·ln V` if `head.normalize else K·ln2 + slack`; or cfg-validate
  `head_normalize=True` whenever the watchdog is armed. Lock: **NOT LOCKED**
  (the B7 test asserts normalized-branch semantics only).
- Identity-channel CE (h = embedding(target), i.e. reconstruction through the
  roundtrip, `normalize=True`): **9.054** (median 9.058, p95 8.940) — ~2 nats
  below uniform; with top-1 decode already perfect, the residual is the softmax
  mass given to the 21.7 nearest twins.
- `code_sparsity`/`code_dim` consistency: no cfg validation exists (S9 probe:
  `__post_init__` codebook-related validation = NONE). Mismatch is impossible in
  the *live* wiring — both builders emit exact weight-S rows (measured) and the
  head's `self.S` is set-but-dead (`embedding.py:241`, no consumer in the
  sigmoid branch), `_bit_norm = cfg.code_sparsity` (`logit_cache.py:232`) divides
  by the true row weight — but `codebook='twin_free'` + `code_dim=32` is
  constructible and only explodes at model-build time, 11.9 s deep in the greedy,
  with a correct-but-slow ValueError. Cheap guard belongs in `__post_init__`
  (`comb(code_dim, code_sparsity) >= vocab` for twin_free, `code_dim ≤ D`,
  `D % code_dim == 0`).
- B7 residuals, cosmetic: the veto print still says "(uniform-bit NLL)"
  (train.py:445, c10:147 — the ceiling is 2·lnV, not bit-NLL since B7) and
  `scripts/scan_garbage.py:5` still derives 22.2 from "K·ln2 (32 бит)" —
  numerically coincidental under the new formula, forensically wrong.

### F2A-08 — bit_profile: cache codes ARE the head codes (same object), but the inference attention path applies tanh TWICE and the /10 temperature is calibrated for raw logits while production feeds log-probs — the profile floor drowns the signal 42:1 and the norm carries no confidence at init | MEDIUM | VERIFIED-REPRO (numbers) + STRONG-READ (reachability)

Coherence checks first (this is the mission's "same C object" question):
- `stack.py:202` passes `codes=self.lm_head.codes` into the cache;
  `logit_cache.py:227` registers `codes_t = codes.float().contiguous()` — on a
  contiguous float32 tensor both calls return *self*, so
  **`attention.codes_t is lm_head.codes` → True** (verified on the mini stack;
  construction path shared). Single rebuild risk, no divergence possible within
  a process. All three buffers non-persistent → all rebuilt from cfg on load
  (F2A-02 applies verbatim).
- `/10` vs actual input scale: the tensor that reaches `bit_profile` in
  production is the head's **log-prob field** (`head_normalize=True`): measured
  on a real-data mini pass — median −11.10, min −18.54, max −1.36, 59.6 % of
  vocab entries at |z| > 10, fraction in the linear regime |z| < 3: **0.000**.
  So `tanh(z/10)` is not "linearizing" anything: the bulk of the vocabulary is
  pinned at −0.80…−0.99 (soft sign), and the only linear zone is the hot-token
  tip. That is *defensible as a top-mass detector* but it is the opposite of
  what "tanh(z/10) keeps small logits proportional" reads like.
- **Double tanh (real, executed)**: `LogitAttention.forward` inference branch
  pre-compresses `cached = tanh(cached/10)` (`logit_cache.py:329`) and then calls
  `bit_profile` (`:331`, `:280`) which applies `tanh(·/10)` **again** →
  effective map `tanh(tanh(z/10)/10)`, slope at z≈−9: **0.00484** vs the single
  pass 0.075 — a ×20 attenuation and near-total linearization (double-profile is
  cos 1.00000 to the single one, so direction survives, scale does not —
  LayerNorm downstream hides this). Meanwhile the *second* consumer,
  `logit_to_hidden(bit_profile(cached_logits))` at `:442`, gets the **single**
  tanh on the same stored data. Two code-space views of one cache, one 20×
  different in slope — a genuine coherence break inside one module.
- **Does the norm carry confidence?** Not at init: single-tanh profile per-position
  norm **349.23 ± 5.11 (CV 1.5 %)**, per-bit floor ≈ −87.1 (mini numbers;
  production extrapolation ≈ (V/K)·E[tanh(z/10)] ≈ 1024·(−0.8) ≈ −820), against
  per-position signal std 0.207 → **floor/signal = 42:1**. The profile norm is
  dominated by Σ_v tanh(z_v/10)·(bit count of column k) ≈ constant; only the
  `k_norm` LayerNorm's mean-subtraction (which the attention K-side does apply,
  `:332`; the `logit_to_hidden` residual at `:442` does NOT) recovers a usable
  discriminative part. Confidence *would* show up as the hot-tip deviation
  (a peaked position moves few cells from −11 to −1, +0.7·1 u each) — measurable
  after centering, invisible before.
- Reachability (STRONG-READ, see F2A-10): none of the above fires in the current
  training loop; it is a latent inference-path property.

Suggested fix (NOT applied): single-source the compression — one
`profile_field(logits)` used by both consumers; either remove the `:329` pre-tanh
or make `bit_profile` accept pre-squashed input; re-pick the temperature for the
*actual* input distribution (log-probs at −11) e.g. `tanh((z−z̄)/σ_z)`, or center
first (connects to F2A-06 — the same common-mode disease: the cache profile has
a 42:1 DC floor exactly like the embedding has a 20:1 one). Locks:
`test_b1_regressions.py:148` compares two profiles' order — it locks nothing
about double-tanh or the floor.

### F2A-09 — The ONLY on-disk checkpoint (best.pt, step 1045) is legacy-K32 geometry and carries a pre-B2 state; trained-geometry probes still ran on it: identity survives 1045 steps, but basis-row norms already drifted to 0.51–0.64 — the margin budget is eroding toward the F2A-01 floor | INFO + WARNING | VERIFIED-REPRO (probe) — with an artifact caveat

`checkponts/best.pt` (2.36 GB) envelope: step **1045**, best_val_loss **11.594**
(= lnV + 0.50 — one full eval after training start), cfg = **`codebook='legacy',
code_dim=32`, vocab=65 536, D=2 560, 24 layers, head_normalize=True,
embed_rope=False** — i.e. this is NOT the cell-4 twin_free config; no twin_free
production checkpoint exists on this host. **While this audit was running the
file was removed to a Chrome re-download** (`checkponts/Неподтверждено 89058.crdownload`)
— all probe results below were captured from the file before it vanished; they
describe the step-1045 legacy snapshot and are re-runnable only after the
re-download completes.

Probes run on the saved tensors (fresh current-code geometry objects + `embed.*`
/ `lm_head.*` load, then 8 192 corpus-weighted FANTASY ids):
- embed→head roundtrip: **top-1 = top-5 = 1.0000** (identity path still alive
  at 1045 steps of Adam).
- `saved codes == current rebuild`: bit-identical (F2A-02's positive case).
- Moved from init (the erosion numbers, for the margin budget of F2A-01):
  `embed_mix` max|Δ| 1.319 (mix is genuinely learning), `bit_bias` max|Δ| 0.010
  (prior pinned), `log_temp`: exp ∈ [1.000, 1.000] (unchanged at 3 decimals —
  1045 steps × λ⁻²-adjacent role LR, plausible), **basis row norms: 1.000 →
  [0.5146, 0.6423]** — squared-norm spread 0.26–0.41, i.e. per-bit z-scale
  already down ~2.4–3.8× and *unequally* by 1.56×. With tied readout both drift
  together (z_k = a_k‖B_k‖² — the ORDER survives), but every segment's margin
  shrinks with its ‖B_k‖² — a 3.8× margin compression after 1045 steps on the
  0.30-min pair margins extrapolates to a thin floor at full training. Not a
  defect *per se* (CE gradient protects what matters), but the geometry should
  be *monitored*: `‖B_k‖² min/max spread` and the d_H=4 pair margin are exactly
  the channels to log. **The 15.5 cold-start reference of the mission is NOT
  reproducible from repo logs** (only `logs/mini_smf.log`: step 0 CE 9.279 for a
  16-layer D=512 synthetic run); the mini 2-layer real-data cold start measured
  here is 9.119 (lnV 7.507) — different geometry, treat as illustrative.
- EOS/vocab edge on trained weights: id 2 (EOS) and id 65 535 self-decode rank 0,
  log-prob ≈ −9.3 — the model can and will emit them (see F2A-11).

### F2A-10 — The inference-mode cache path (bit_profile + k_proj_l + logit_to_hidden) is UNREACHABLE in production wiring: `augment(h)` never carries logits, so R1's scheduled sampling and the whole train/inference representation alignment are dead code in the live loop | MEDIUM | VERIFIED-REPRO (probe) + STRONG-READ

- Training integration is `stack.py:713-714`: `h = self.logit_cache.augment(h)`;
  `augment` (`logit_cache.py:448-458`) calls `forward(h, logits=None,
  training=True)`. The scheduled-sampling gate (`:416-419`) requires
  **`logits is not None`** — it is None by construction. Probe: ratio forced to
  1.0, 20 `augment` calls → `store(training=True)=20, store(training=False)=0`.
  The 5 % "align train/inference representations (R1)" doctrine (config.py:315)
  therefore aligns nothing: the model *trains* with h-cache only.
- Inference-mode store/attend exists only via `process_with_cache`
  (`stack.py:946-975`), whose callers are `scripts/{test_eva_cache,
  test_unified_cache,bench_cache_gen}.py` — **not** `scripts/generate.py`,
  **not** `core/live_inference.py` (grep: zero hits). The compressed-logits
  "inference" half of the dual-mode design is exercised only by bench scripts.
- Consequence chain: agent 1's F-21 RNG census row "logit_cache scheduled
  sampling `torch.rand(1)` — CPU default, covered" describes a consumer that
  **never executes** (correcting their coverage table — the CPU-generator row
  becomes vacuous, not wrong-risky). The k_proj_l/v_proj_l parameters exist in
  every best.pt envelope and receive **no gradient in the training loop** —
  they train only if someone routes generation through `process_with_cache`.
  Optimizer groups for dead parameters → agent 2b.
- Suggested fix (NOT applied): compute the head logits once per step in the
  loop (they are computed for salience anyway, `train.py:436` / c10:89 — currently
  AFTER `model.forward`, and `augment` runs *inside* forward before the head —
  a chicken-and-egg ordering the design should own explicitly: either feed the
  *previous* step's detached logit field into the cache (matches M8 salience
  doctrine of 1-step delay), or pass logits through forward. Then R1 becomes
  live. Lock: NO LOCK (nothing asserts the ss branch fires).

### F2A-11 — EOS / vocab-edge rows: no special codes, EOS=2 is a fully-trained 5.7-6.1 % of stream, head covers id 65 535 in argmax space; the real gap is the REASONING-token collision at vocab=65 536 | LOW/INFO | VERIFIED-REPRO + STRONG-READ

Answering F-18 + the handoff question "can the model EVER predict them / does
head vocab==65536 cover 65535":
- `build_codes(cfg)` accepts V=65 536 under both builders: legacy
  `sparse_block_codes(65536,32,6)` fine (colw 11 957–12 505, balanced,
  row-weight 6), twin_free verified at (65 536, 64, 6) (F2A-01). Rows 0, 1, 2,
  65 532…65 535 are ordinary weight-6 codes (supports printed in
  `repro2a_results.txt:60-66`) — **no reserved geometry anywhere in the id→code
  map**; the codebook cannot express "this row is a control token".
- Training targets: `mask = targets != 0` only (`losses.py:36`, `mask_eos=False`
  production) ⇒ id 2 is trained (6.07 % of the 800 k-token probe windows —
  agent 1's 5.7 % confirmed locally), ids ≥ 2 including 65 535 are legal labels;
  roundtrip decode at init ranks them correctly (forced into the id sets);
  `token_bias` zeros(cfg.vocab) (`embedding.py:260`) indexes 65 535; the trained
  best.pt probe self-decodes 2 and 65 535 at rank 0 (F2A-09). Verdict: the head
  CAN emit every real id, EOS included.
- The actual edge hazard, quantified from my side: with vocab = 65 536 the
  reserved reasoning ids (THINK…END ≥ 65 536, embedding.py:121-131) **clamp onto
  real corpus row 65 535** (which occurs 708×/327 M in FANTASY, 6×/400 k in my
  window — rare but real). If reasoning tokens are ever wired into generation
  streams (agent 3's scope), every control token reads/writes as that one junk
  symbol — the M9 warning fires once, then silence. Either reserve head rows by
  expanding vocab above the reasoning range (uint16 corpus + 4 control rows
  needs vocab ≥ 65 540 and a data-side remap — WideBind pipeline change) or keep
  control tokens *outside* the token-id channel entirely (a separate bus). Not
  my decision; flagged to agents 3/5.

### F2A-12 — compression.py re-materializes the codebook with the WRONG builder for twin_free models (latent; inert today only because codes are non-persistent), plus the cfg knob surface has no code-geometry validation | LOW (latent) | STRONG-READ

`core/compression.py:24` strips `embed.codes`/`lm_head.codes` from compressed
state ("their regeneration to decompress_sd" — `:275`), but
`decompress_sd` hard-imports and calls `sparse_block_codes(cfg.vocab, K, S)`
(`:226, :232`) and injects the result at `sd['embed.codes']`, `sd['lm_head.codes']`
(`:255-256`). For a `codebook='twin_free'` model this re-injects the LEGACY
codebook. Currently inert: (a) current models never *have* those keys in
state_dict (non-persistent → REMOVABLE suffixes strip nothing; `compress_sd` of a
twin_free model simply lacks the keys), and (b) the decompressed sd is loaded
with `strict=False` in `scripts/generate.py:281/297`, where `embed.codes` is an
ignored unexpected key. But the intent of the function is precisely "recreate
the deterministic buffers", and it recreates them with the wrong dispatcher —
the day codes become persistent (F2A-02's obvious fix!) or a load path assigns
buffers from sd directly, the compressor swaps twin_free → legacy *silently and
without shape error* (same K whenever K matches — the shape is identical for
equal cfg!). The one-line fix: `build_codes(cfg)` instead of
`sparse_block_codes(...)`.
Knob validation (config): none of `codebook/code_dim/code_sparsity/vocab/D`
mutual constraints exist in `__post_init__` (measured "NONE"); failure modes
range from late (K=32 twin_free: 11.9 s greedy then correct ValueError) to
late-and-expensive (D%K asserts live in three head classes — embedding.py:92,
:218, :245 — fine but after a full codebook build). Lock: NO LOCK.

---

## 2. Geometry facts table (the constitution baseline)

All numbers measured on the CURRENT tree (c152f28), torch 2.13.0+cpu, real
corpus windows for the weighted rows. Repro ids in parentheses.

| # | Quantity | Value | Conditions |
|---|---|---|---|
| 1 | twin_free build time / size | 1.74 s, 65536×64 fp32 = 16.8 MB | (65536,K=64,S=6), 8 threads |
| 2 | row weights | all == 6 | F2A-01 |
| 3 | unique rows | 65536 / 65536 | packed-support hash |
| 4 | gram off-diag MAX (exact certificate) | **4** = S−2 | 5-subset keys, 393216/393216 distinct |
| 5 | d_min (Hamming) | **4**, attained | 711 504 exact d_H=4 pairs (0.033 % of 2.15 G) |
| 6 | d_H mean / variance / std | 10.876 / 1.879 / 1.371 | 4 M sampled pairs; max 12 |
| 7 | mean pairwise overlap | 0.5621 (random baseline 0.5625) | packing moves the tail only |
| 8 | d_H=4 rivals per code | mean 21.7, p05 15, max 48 | exact neighbor list |
| 9 | init pair margin, worst d_H=4 pair (both directions) | **min +0.2993**, p1 +0.542, mean +0.768 u-units; 100.0000 % correct order | F2A-01/repro2b |
| 10 | bit marginals p̄_k | 0.09088…0.09590, std 0.00103 (ideal 0.09375) | near-perfect balance |
| 11 | CᵀC spectrum | rank 64, eig 5340…36 869, **cond 6.90** | code-space conditioning |
| 12 | codebook fingerprint (twin_free 65536,64,6) | sha256 `74f3f995f303b2ab5c33fc0076431a372951ecb797385654f5e4053098558645` | identical across 3 processes, 1/4/8 threads, cache-clear rebuild |
| 13 | legacy fingerprint (65536,32,6) | sha256 `98fed2eb6dcea14e5c9560bc2a2f64b4…` | == the codes *persisted* in best.pt (F2A-02) |
| 14 | embedding activation a=σ(2Mc) | active 0.8800±0.0260, inactive 0.5030±0.0607, per-bit signal 0.377, range [0.228,0.955] | production init |
| 15 | embedding rank | 64 of D=2560 (2496 dims identically zero) | isometry to a-space verified |
| 16 | mean pairwise cos(e,e′) corpus-weighted | **0.9512** (median 0.9502, σ 0.0095) | 300 k pairs |
| 17 | common-mode energy share | **0.9511** corpus-weighted / 0.9503 uniform; top-PC 0.9513 | the centering-experiment number |
| 18 | after centering | mean cos 0.0002 (σ 0.189); centered PC1 share 0.0932; centered cond 1447 | signal fully preserved |
| 19 | roundtrip embed→head top-1/top-5 | **1.0000 / 1.0000** on 8192 corpus-weighted + 8192 uniform + 1000 hot real ids (incl. 0,1,2,65532-5) | production geometry, init |
| 20 | roundtrip, embed_rope=True | **top1 0.2031** (fix unwired) | F2A-04 |
| 21 | uniform CE, normalize=True, h=0 | **11.0904 nats = ln V** (max prob 2e-5) | F2A-07 |
| 22 | init identity-channel CE (h=embed(target)) | 9.054 (median 9.058) | normalize=True |
| 23 | factorized branch (normalize=False): uniform / init-CE | **44.361 (=K·ln2)** / 18.810 | vs ceiling 22.181 — F2A-07 |
| 24 | hard-veto ceiling V=65536 / V=50000 | 22.1808 / 21.6563; band 15.5 < 22.18 < 34 ✓ | B7, locked |
| 25 | init log-prob field (mini, real data) | min −18.54, median −11.10, max −1.36; frac \|z\|<3: **0.000**; frac \|z\|>10: 0.596 | bit_profile input reality |
| 26 | bit_profile single-tanh | per-pos norm 349.23±5.11 (CV 1.5 %); per-bit floor −87.1 (≈ −820 production extrapolation); **floor/signal 42:1** | mini; cache codes_t `is` head codes = **True** |
| 27 | bit_profile double-tanh (attention inference path) | slope 0.00484 vs single 0.075 vs intended 0.1 (**20.6× off**); cos to single = 1.00000 | F2A-08 |
| 28 | profile-norm confidence at init | ≈ none (1.5 % CV, DC-dominated) | F2A-08 |
| 29 | scheduled sampling firing | **0/20** stores(training=False) at ratio 1.0 (logits≡None via augment) | F2A-10 |
| 30 | twin_free @K=32 | fits 20 518/65536 then ValueError (11.9 s) | docstring claim verified |
| 31 | random weight-6 20 k pool | 937 pairs ≥ overlap-5 (incl. twins) | the filter's job, unfiltered |
| 32 | best.pt (legacy, step 1045) roundtrip / basis norms | top-1 1.0000; ‖B_k‖: 1.000 → **0.5146…0.6423** (squared 0.26–0.41, spread 1.56×); bit_bias Δ≤0.010; log_temp ≡1.000; token_bias ≲0.005 | trained-state probe, F2A-09 |
| 33 | codes in state_dict (current code) | embed.codes NO, lm_head.codes NO, codes_t NO; **lm_head._prop YES (persistent)**, embed_mix/token_bias YES | F2A-02 |
| 34 | mini cold-start CE (2-layer D512 V1820, real ids) | 9.119 vs lnV 7.507 (+1.61 trunk-noise nats) | illustrative only |
| 35 | suite baseline | **287 passed, 54.15 s** (tree clean) | post-analysis |

---

## 3. Reactions to audit 01 + B7 status check

**Handoff → 2a, item 1 (F-01, the vocab clip).** B7 fixed it in the way I would
have asked for: silent clip → loud `ValueError` in BOTH twins
(`train.py:59-64` ≡ c7:14-17, verified in current source), `--vocab` default
65 536 (`train.py:761`), config default 65 536 (`config.py:16`), locked by
`test_b7_regressions.py:33-47` (repo-relative import this time — the F-19
machine-path lock was NOT among the B7 fixes). **Geometry-side residuals:**
(i) the notebook call sites still never pass `vocab` (c10:658/691/959 — the
guard is inert there, harmless only because uint16 < 65 536 makes it
mathematically unreachable), so the twins again differ in *behavior* despite
identical signatures; (ii) `codebook='legacy'` remains train.py's default
(`config.py:65`) while cell 4 says `twin_free` — the "same stream, different
head topology" divergence F-01 flagged is alive on the codebook axis. Not
re-auditing data-path scope; recorded because it changes WHICH codes the ids
become — my jurisdiction. Needs-lock: a twin-parity test that asserts
identical (signature AND call-site vocab policy) behavior, extended to the
codebook default.

**Handoff → 2a, item 2 (F-07, K is the veto ceiling).** B7: **FIXED and verified
live** — single source `hard_veto_ceiling(vocab)=2·lnV`
(`training_control.py:404-414`), both loops consume it (`train.py:443`,
c10:145), `model.lm_head.K` is no longer read by any ceiling code (grep).
My deeper measurement (F2A-07): the normalize=True branch's uniform is exactly
lnV=11.0904 (measured), the ceiling 22.1808 = 2.000×uniform sits at 1.43× the
cold-start quote and 0.65× the garbage class — the historical calibration is
reproduced *and* geometry-free. **Residual for the same family**: the
normalize=False leg's uniform is 44.36 > 22.18 — the ceiling now sits BELOW the
zero-information point of the factorized branch it originally described; a
future `head_normalize=False` experiment inherits a ceiling calibrated for the
other branch. Also stale: the veto's own print label "(uniform-bit NLL)" and
`scan_garbage.py:5`'s K·ln2 derivation (both numerically coincidental now).
Needs-lock: branch-aware ceiling + updated labels.

**Handoff → 2a, item 3 (F-18, head emits 65 535?).** Confirmed YES from every
angle available to me: row 65 535 is an ordinary twin_free weight-6 code
(support [3,10,21,44,46,53]); `build_codes` accepts V=65 536 under both
builders; token_bias covers it; argmax decode of a forced 65 535 at init is
correct (it is inside the 1.0000 sets); trained best.pt self-decodes it at rank
0 with logp −9.33. EOS=2 is trained (6.07 % of probe windows, mask only id 0,
`losses.py:36`) and equally emittable. One forward-looking correction to my
side of the ledger: with vocab==65 536==|uint16| there is NO id room for the
THINK…END block the M9 comment references — they will clamp onto *real* id
65 535 when generation starts emitting them. That is the true remaining edge of
F-18 (F2A-11), a decision for agents 3/5 + the WideBind pipeline.

**Handoff → 2a, item 4 (`migrate_state_dict` W_out+K growth × K=64 heads — does F-11's restore assume right?).** Answered concretely: `migrate.py` touches
only `*.bind.W_out` / `bind_coh_gate` / `freq_scale` + memory-bank/old-head key
retirement; **no head code/geometry tensor migrates at all**. The head K=32→64
question never reaches migration because train.py:272-280 pre-filters size
mismatches — which is precisely the problem: see F2A-03 (HIGH). The
`_restore_optimizer` W_out padding branch (train.py:110-124) pads `exp_avg` with
zeros / `exp_avg_sq` with 1.0 — sane for bind rows, irrelevant for the head
(no head key ever size-changes through migration: its shape is cfg-fresh by
construction). The positional-load bug F-11 stands; the geometry-adjacent part
is handed to agent 2b.

**Correction to audit 01 §2.2 / F-21 from my side:** the consumer
"logit_cache scheduled sampling `torch.rand(1)` (logit_cache.py:417-418)" is
**unreachable in the production training wiring** (F2A-10 probe: 0/20 fires at
ratio 1.0) — the CPU-default-generator row of their RNG table is vacuous. Does
not weaken F-21's main (CUDA noise) point; it removes one covered-by-luck entry
from it.

**Where B7 touched my scope but did not finish (status summary):**
F-01 fixed (2 residual drifts noted), F-07 fixed (1 leg uncovered, labels
stale), F-18 facts confirmed (1 future collision exposed). F-02/F-03 (L2
buffers/eval-writes) — outside geometry; the B7 lock (`test_b7_regressions.py:50-74`)
reads correct from here.

---

## 4. Not verified + handoffs

Not verified here (scope or host):
1. **Trained twin_free state.** No twin_free checkpoint exists on the host
   (best.pt is legacy-K32 step 1045, and it is currently mid re-download —
   see F2A-09 caveat). Therefore: post-training margin erosion, bit_bias/
   log_temp drift, and the per-code recall cliff (the B2 docstring's T≥1200
   claim) are verified only at init and at 1045 legacy steps.
2. Full production-trunk (24×D2560) forward on CPU — mini stacks used;
   cross-segment behavior of the *trunk* (not embed/head) unmeasured.
3. CUDA-side codebook rebuild (the greedy runs on CPU memmap-side tensors both
   before and after `.to(device)`? — codes are built on CPU and only cloned;
   no device RNG involved; argued deterministic, not executed on CUDA).
4. Recall-knee reproduction (needs the retrieval harness from audit A;
   the overlap facts it depends on are locked here in F2A-01).
5. `analyze.py` / `generate.py` consumers of head outputs — not read for
   geometry assumptions line-by-line.

Handoffs:
- **→ 2b (optimizer/param groups):** (a) F2A-09's trained-state drift census —
  bit_bias/log_temp ≈ frozen at 1045 steps vs basis norms collapsing toward
  AdamW decay: check the role-LR for `embed.*/lm_head.readout`
  (`adaptation.py:188-190` gives them λ⁻² ≈ 1/9 of base — deliberate?) and
  whether `log_temp`/`bit_bias` (role default 1.0) should move faster;
  (b) `k_proj_l/v_proj_l/logit_to_hidden` receive zero gradient in the loop
  (F2A-10) but ride in every checkpoint and optimizer state — momentum on dead
  params; (c) token_bias is exactly at its (zero) init at step 1045 — confirm
  it is in the optimizer's param list (I could not finish the
  `param_names` probe because the file vanished mid-run).
- **→ 3 (losses/aux):** (a) the factorized-branch prior-init CE is 18.81 and
  uniform 44.36 — any aux on that branch must state which scale it normalizes
  against; (b) `compute_salience` (stack.py:928-938) sigmoid()s **log-probs**
  (median −11 → sigmoid ≈ 1.7e-5 before normalization): the 0.01–0.5 "log-prob
  norms" claim in its comment matches init, but the channel is a ln-scale proxy
  of confidence — worth a geometry review when the centering experiment lands;
  (c) h_emb two-ended read contract confirmed alive at production geometry
  (F2A-05) — losses.py:19-47 is consistent with the head branches measured here.
- **→ 4 (control/watchdog):** F2A-07 in full: band verified for the production
  branch; the normalize=False leg sits under the ceiling; stale "(uniform-bit
  NLL)" label + `scan_garbage.py:5` derivation; suggested
  branch-aware `hard_veto_ceiling(vocab, normalize, K)`; the margin facts
  (min d_H=4 pair margin 0.30 → basis-norm compression ×0.26 at 1045 steps) are
  the numbers a geometry-aware soft veto should watch instead of raw CE if
  anyone wants early identity-loss detection.
- **→ 5 (persistence/eval doctrine):** F2A-02 (codebook fingerprint in the
  envelope + rebuild-sha assert on load; make the determinism test clear
  `_CODES_CACHE`), F2A-03 (critical-shape-key abort for embed.*/lm_head.*
  resume), the best.pt/cfg-not-reconsulted-on-resume fact, and the state of
  checkponts/ (2.4 GB legacy step-1045 checkpoint replaced by an unfinished
  `.crdownload` during this audit — forensic value: it IS the B2-era geometry
  witness; re-lock its sha after re-download).

---

### Appendix A — repro manifest (all green, this audit)

- `repro2a_main.py` → `repro2a_results.txt`: S1 exact certificates
  (5-subset/4-subset hashing), S4 determinism, S5 production geometry + corpus
  common-mode, S6 roundtrips (4 sets), S7 CE band, S8 bit_profile probes +
  ss-deadness, S9 edge ids/builders.
- `repro2a_hash.py`: sha256 of twin_free(65536,64,6) in 3 fresh processes,
  OMP threads 1/4/8 — all `74f3f995…58645`.
- `repro2b_margin.py` → `repro2b_results.txt`: reconstructs init u from the
  exact source-verified init recipe and certifies pair margins over ALL
  711 504 d_H=4 pairs in both query directions (recovered by exact 4-subset
  pairing — the recovered count matches the histogram-derived 711 504).
- `repro2c_ckpt.py` / `repro2d_ckpt2.py` → their txt: best.pt envelope census,
  legacy codes == rebuild, trained roundtrip, geometry drift census.
- Suite: `python -m pytest tests -q` → **287 passed in 54.15s** (mission brief
  said 285; measured current = 287 on clean tree).

### Appendix B — the one-paragraph geometry model (for the non-quantitative reader)

Every token is a 6-of-64 bit pattern; no two patterns share more than 4 bits,
and only ~22 codes sit at that worst distance, giving every token a certified
positive logit margin over its closest rivals at init. The pattern becomes a
64-coordinate activation vector where "off" is 0.5 (not 0) and "on" is 0.88 —
which makes all 65 536 embeddings 95 % the *same* vector (the centering
opportunity) and the identity channel a 64-dimensional object inside a
2 560-dimensional space. The head reads the same 64 coordinates through the
tied basis (unit rows make this exact), debiases with the code prior, and at
init decodes token identity with zero errors and worst-pair margin 0.30 — an
eroding margin (the trained basis rows already lost 60–74 % of their squared
norm in 1 045 steps). The cache shares the identical codebook object — but only
its inference path touches it, and that path currently (1) never runs in the
real loop, (2) double-compresses the evidence, and (3) normalizes a log-prob
field with a raw-logit temperature, leaving the per-bit profile 42 parts floor
to 1 part signal. Fix the run-wiring before the math; fix the math (centering)
before unifying this space with memory and intent.
