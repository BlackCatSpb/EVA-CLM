# 01 DATA PATH AUDIT

Agent #1 of the serial relay. Scope: **uint16 token files → (x,y) tensors at the
model**, plus the **batch-VETO machine**, the **document-boundary rotation**, and
**eval isolation**, in BOTH copies of the loop (`scripts/train.py` and
`notebooks/eva_colab.ipynb`, cells 7/8/9/10 — the notebook is a *separate* copy
by explicit doctrine, `docs/AGENT_BRIEF.md:35`).

Method notes.
- All anchors below were read from the working tree; quoted lines are verbatim.
- Provable claims were exercised against scratch repros in
  `%TEMP%\opencode\repro1.py` (imports the classes *by extracting the exact
  class source from train.py and from the notebook JSON* — nothing patched).
- Real corpus facts from full-file numpy scans of
  `WideBind\wb\token_stream_{ACTS,ECONOMY,DOCUMENT,CHILDREN,FANTASY}_eos.bin`
  (bincount over all uint16 ids, 2.2 GB streamed) and the tokenizer JSON.
- Baseline suite: `python -m pytest tests -q` → **284 passed in 114.33s**
  (run once, after all analysis; the repo was not modified).
- Confidence tags: **VERIFIED-REPRO** (executed), **STRONG-READ** (anchor-verified
  code path, deterministic consequence), **SPECULATIVE** (plausible, unproven).

Conventions used: `train.py:N` = `scripts/train.py` line N.
`c7:N` / `c9:N` / `c10:N` = notebook cell source line N of cells index 7
("7. Data Streams" — the TokenStream twin), index 9 ("8b. Adaptation module"),
index 10 ("9. TRAINING LOOP" — the loop the mission calls "cell 10").
`core/x.py:N` = that module.

---

## 1. Findings

### F-01 — TokenStream twin drift: the notebook has NO vocab-clip; train.py's clip default (vocab=50000) silently corrupts 1.3–7.8 % of the REAL corpus | HIGH | VERIFIED-REPRO

The two `TokenStream` copies are *not* semantically identical — they have
different **signatures**, hence different tensors for the same bytes.

`scripts/train.py:48-65`:
```python
    def get_batch(self, seq_len, batch_size, offset, vocab=50000):
        needed = batch_size * seq_len + 1
        wrapped = offset + needed > self.len
        if wrapped:
            offset = 0
        chunk = self.data[offset:offset + needed]
        if vocab is not None:
            # uint16-файлы могут содержать токены ≥ vocab → device-side assert
            # в codes[tokens] (index out of bounds); клипим до безопасности.
            chunk = np.clip(chunk, 0, vocab - 1)
        x = torch.from_numpy(chunk[:batch_size * seq_len].reshape(batch_size, seq_len).copy())
        y = torch.from_numpy(chunk[1:batch_size * seq_len + 1].reshape(batch_size, seq_len).copy())
        ...
        return x.long(), y.long(), offset + batch_size * seq_len, wrapped
```

`notebooks/eva_colab.ipynb` cell 7 lines 6-18 (the twin):
```python
    def get_batch(self, seq_len, batch_size, offset):
        needed = batch_size * seq_len + 1
        wrapped = offset + needed > self.len
        if wrapped:
            offset = 0
        chunk = self.data[offset:offset + needed]
        x = torch.from_numpy(chunk[:batch_size * seq_len].reshape(batch_size, seq_len).copy())
        y = torch.from_numpy(chunk[1:batch_size * seq_len + 1].reshape(batch_size, seq_len).copy())
        ...
        return x.long(), y.long(), offset + batch_size * seq_len, wrapped
```
Diff: **the entire `vocab` parameter and `np.clip` guard exist only in
train.py**. Every other line of the two classes is identical (verified by
executing both copies against the same synthetic streams — repro R1/R3).

Mechanism of harm, with real-data evidence:
- The live corpus is full uint16 vocabulary. Full-file scans:
  `FANTASY`: **25,471,330 / 326,955,957 tokens ≥ 50000 (7.79 %)** across
  **14,043 distinct high ids**, including **708 occurrences of id 65535**
  (max uint16); `CHILDREN` 1.71 M/22.1 M (7.7 %); `DOCUMENT` 463 k (2.9 %);
  `ECONOMY` 84 k (2.4 %); `ACTS` 9,957/740 k (1.3 %). All five files have
  `max_id ≥ 65524`; four files actually hit **65535**.
- `scripts/train.py:754` CLI default is `--vocab 50000` (and
  `core/config.py:16` `vocab: int = 50000`). Any train.py run that does not
  explicitly pass `--vocab 65536` therefore **folds every one of those 14k real
  ids onto the single codebook row 49999** (`np.clip`, train.py:57). Repro R3:
  input ids `[1,2,49999,50000,60000,65535,...]` → train.py batch
  `[[1,2,49999,49999]]` vs notebook batch `[[1,2,49999,50000]]`.
- The production notebook config sets `vocab=65536` (cell 4 line 11) where the
  missing clip is *harmless* (uint16 < 65536 always) — so this is not "the
  notebook is broken", it is: **the two data paths give different tensors for
  identical bytes, and the copy that carries the guard carries the dangerous
  default**. train.py additionally defaults `codebook='legacy'`
  (`core/config.py:65`, K=code_dim=32, `sparse_block_codes(50000,32,6)`) while
  the notebook uses `codebook='twin_free', code_dim=64, vocab=65536`
  (cell 4 lines 9-11) — so the "same" TokenStream feeds *different head
  topologies* in the two entry points.
- The clip is also **silent**: no counter, no warning (contrast
  `core/embedding.py:120-131` which added a loud one-time M9 warning for the
  identical event downstream).

Suggested fix (NOT applied): make the two classes byte-identical (single source
of truth, e.g. move `TokenStream` into `core/data.py` and have cell 7 exec it —
the AGENT_BRIEF already says every train.py edit must be hand-ported); change
the train.py default to `vocab=65536` / assert `cfg.vocab >= 65536` for
`uint16` streams; make the clip count clipped tokens and warn once like M9.

Locks: `tests/test_product_invariants.py:659-686` (M8.3) execs the **train.py
copy only**, via a **hard-coded absolute machine path**
`C:\Users\black\OneDrive\Desktop\EVA CLM\scripts\train.py` (line 665) — it
locks the 4-tuple contract, not the clip, and dies on any other checkout.
The notebook twin: **NO LOCK** (see also F-19).

---

### F-02 — Document rotation calls `memory_bank.reset()`, which zeroes the TRAINABLE PARAMETERS `L2.keys`/`L2.vals` in place | HIGH | VERIFIED-REPRO (parameter status) + STRONG-READ (call path)

Rotation (both copies): `train.py:411-412`
```python
                if getattr(model, 'memory_bank', None) is not None:
                    model.memory_bank.reset()  # reset streaming banks at document boundary
```
(c10:73-74 identical). `core/memory_bank.py:461-464` → `L2Bank.reset()`:
```python
    def reset(self) -> None:
        self.keys.data.zero_()
        self.vals.data.zero_()
```
but `keys`/`vals` are **not buffers** (`core/memory_bank.py:202-203`):
```python
        self.keys = nn.Parameter(torch.randn(n_slots, bridge_dim) * 0.02)
        self.vals = nn.Parameter(torch.randn(n_slots, bridge_dim) * 0.02)
```
and they are live differentiable tensors (`read()`: `k = self.keys`,
`v = self.val_norm(self.vals)` — memory_bank.py:291-292), i.e. they sit in
`named_parameters()`, in optimizer groups, and in `state_dict()`/best.pt
(`smoke_notebook.py:95` even asserts param_names coverage).

Consequence chain, every document boundary (~every `len/(batch·seq)` ≈ 639k
steps→ no: with FANTASY 327 M tokens and 1025-token steps, ≈ 319k steps — i.e.
*per genre switch*, realistically every few thousand steps under seq
curriculum):
1. optimizer Adam moments (`exp_avg`, `exp_avg_sq`) keep pointing at values
   that were just zeroed under them;
2. the mid-document half-life of L2 content is a *hard wipe*, not a decay;
3. if best.pt is written shortly after a rotation, it stores zeroed
   slots — resume inherits that.

Also inconsistent granularity: `L2Bank._n_overwrites`/`_n_consumed` are plain
Python ints (memory_bank.py:226-227,320-221) — reset() clears them (320-321)
but they are NOT in checkpoints, while `_write_idx` IS a persistent buffer
(225). The "bank" is simultaneously a buffer, a parameter and an int.

Suggested fix (NOT applied): split storage from parameters — L2 content slots
should be `register_buffer(persistent=False)`; keep learned projections only.
Or rotate a *shadow* bank. Requires a decision from agents 2b (optimizer) and
3 (memory) — handoff §5.

Locks: **NO LOCK**. `tests/` never instantiate `StreamingMemoryBank.reset()`.

---

### F-03 — EVAL isolation hole: validation forwards write hold-out sentences into `L2.keys/vals` (parameters); `snapshot_runtime_buffers()` cannot see parameters | HIGH | VERIFIED-REPRO

The eval contract (both loops) is `snapshot → eval → restore`:
train.py:710 `_rt_snap = model.snapshot_runtime_buffers()` / :740
`model.restore_runtime_buffers(_rt_snap)`; c10:353 / c10:425. The docstring
`core/stack.py:1019` states the premise:
```
        Parameters are not touched by eval (no optimizer step) → buffers only.
```
**False.** Eval calls the exact forward `model(h, None, adaptive=False,
tokens=vx)` (train.py:733, c10:383). With `step=None`, `core/stack.py:379-383`
takes `mat_gate = self.maturation.gate` (the last TRAIN gate — a buffer);
`stack.py:522-525` then enters the memory bank whenever
`mat_gate[i] >= _min_write_maturation` (0.3, `core/config.py:303`) — **not
gated on `self.training`** — and `StreamingMemoryBank.forward` writes at every
SEP (memory_bank.py:404 `is_sep = (tokens == 2)`; real corpus: id 2 occurs
5.7 % of tokens) via `L2Bank.write` → `self.keys.data[slot] = new_key`
(266-267, parameters).

Repro R12 (mini EVAStack, D=512, 2 layers, memory_bank=True, `m.eval()`,
gate forced mature, exactly the eval call):
```
l2.keys in named_buffers(): False
eval forward wrote L2: write_idx 0->2, keys mutated: True
after snapshot restore: write_idx back to 0 (buffer) but keys still mutated: True
```
So hold-out document content is consolidated into TRAINING parameters, then
saved into best.pt (`model.state_dict()` includes keys/vals), and the
`_write_idx` buffer rolls BACK under mutated content → the slot bookkeeping is
desynchronized (bank claims fewer writes than it contains; novelty/age buffers
`slot_age` etc. are restored but keys/vals are not — *torn state*).
Note also: per-layer memory-bank call means ONE sentence is written
`n_layers` times (R12: one SEP, two layers → write_idx +2).

Suggested fix (NOT applied): make `StreamingMemoryBank.forward` a no-write
when `not self.training` (reads only), or move keys/vals to buffers, or
snapshot/restore `named_parameters()` in `snapshot_runtime_buffers`.

Locks: **NO LOCK** for the parameter path. `tests/test_b1_regressions.py:79-90`
locks `_last_bus` restore; `tests/test_product_invariants.py:629-640` locks
buffer restore. Neither covers L2. → handoff agent 5 (eval doctrine).

---

### F-04 — Rotation does NOT reset the whole stream-bound state; the biggest miss, `_intent_stream`, actively defeats the notebook's own `intent_state=None` reset | HIGH | STRONG-READ (item-by-item anchors)

Rotation resets (train.py:407-416, c10:66-76): local `state`, `gs`,
(+`intent_state` in the notebook), `bridge.bridge_stream.zero_()`,
`memory_bank.reset()`, `logit_cache.cache.clear()`, `reset_reasoning()`.

What a "new document" leak means and what leaks — every mutable piece of
stream-bound state written during forward, with rotation coverage:

| # | state | written at | buffer? | rotation reset? | eval snapshot? |
|---|-------|-----------|---------|-----------------|----------------|
| 1 | `model._intent_stream` (per-layer carry, (1,1,G,k)×n_layers) | stack.py:638 | no (plain attr) | **NO** | YES (`__attrs__`, stack.py:1023) |
| 2 | `model._last_salience` (B,L,1) | stack.py:944 via loop `observe_output` train.py:429 / c10:89 | no | **NO** | not written in eval (loop-level) |
| 3 | `model._last_bus` | stack.py:639 | no | **NO** | YES (1023) |
| 4 | `model._gs_velocity` | stack.py:629 (step≥5000) | no | **NO** (local `gs=None` only) | skipped (step None) |
| 5 | `mirror._private_mem` (G,k EMA "collective memory") + `_pm_step` | mirror.py:609,630-631 | buffers (`_private_mem` persistent=True! 257-258) | **NO** | YES |
| 6 | `mirror._delta_var/_gate_ema/_concept_sim_ema/_behavior_div_ema/_div_run/_div_run_rec/_trust_matrix/_prev_trust_matrix/_meta_private_mem/_hp_grad/_grad_norm_ema/_ig_norm_ema/_ctr_norm_ema/_ls_var_run/_signal_norm_ema/_last_magnitude/_last_gates/_last_h_pool/_residual_var_ema/_pm_coh/_prev_grad_norm` | mirror.py:414-940 | buffers 194-300 | **NO** | YES |
| 7 | `bridge._tgt_mean`, `bridge_loss_init/ema`, `inj_ratio` | bridge.py:168,203,239-240 | buffers (persistent `_tgt_mean` 84) | **NO** | YES |
| 8 | `block._mlp_now_ema/_mlp_base_ema/_mlp_cnt` + `_mlp_ratio` float | block.py:596-604 | buffers 269-271 | **NO** | YES |
| 9 | `block._cache_mlp_mod/_cache_mlp_out/_cache_conv_out/_cache_bind_out/_cache_mirror_out` (grad-carrying) | block.py:367,502,533-534,577 | no | no (overwritten next forward) | overwritten next train fwd |
| 10 | UCL `concept_keys/vals/age/count/confidence/_mature/_resvar_ema/_resvar_var/_step/_n_*` | concept_layer.py:222-273,377-389 | buffers 76-105 | **NO** | YES |
| 11 | `layers[*].b_i/b_d` — **Parameters** rewritten in-place each train step by AdaptiveController | stack.py:299-306 (`fill_` / `data.lerp_`) | no (params) | NO (not doc state; flagged for agent 2b) | not written (adaptive=False) |
| 12 | `layers[*]._pi_v` power-iteration vector | losses.py:144-147 | no | NO | re-seeded only on shape change |
| 13 | spiral/traj bind `_step_count` etc. | bind.py:397,602-603 | plain/buffers | **NO** | YES (buffers) |
| 14 | `logit_cache` ring (`_h_cache,_logit_cache,_kv_h,_position`) | logit_cache.py:67-72,91-103 | no | YES — `clear()` covers all four (173-178) ✓ | cleared before+after ✓ |
| 15 | `reasoning_buffer/count` | stack.py:664-666 | no | YES via `reset_reasoning()` (900-903) ✓ | guarded off (234,664) ✓ |
| 16 | `model._phase_ratio_ema/_std` | train.py:194-195,559-562 | plain lists | no (not doc state) | n/a |

Why #1 is severe: `core/stack.py:333-336`
```python
            if isinstance(self._intent_stream, list) and len(self._intent_stream) == n_layers:
                intent_streams = [_to_kmax(s) for s in self._intent_stream]
            elif isinstance(intent_state, list) and len(intent_state) == n_layers:
```
The internal attribute has **priority over the caller's argument**. The
notebook's rotation sets local `intent_state = None` (c10:67) — and it is
**ignored**, because `_intent_stream` is still the previous document's
n_layers list written at :638. The "document boundary" for the intent bus
exists in neither copy. train.py is in the same boat (never passes
`intent_state` at all, train.py:425). #2/#3: the last batch of the OLD
document supplies salience and bus to the FIRST batch of the NEW document
(salience applies only on shape match, stack.py:429-431 — constant B=1,L=512
in the notebook always matches; the train.py curriculum accidentally breaks
the match when seq_len changes). #5: `_private_mem` and `_delta_var`,
`_concept_sim_ema`, `_behavior_div_ema`, `_gate_ema`, `_residual_var_ema` are
`persistent=True` buffers — cross-document expert memory is *checkpointed*
and *never rotated*.

**This table is the handoff contract requested by the mission; §5 assigns
ownership.**

Suggested fix (NOT applied): a `model.reset_document_state()` on EVAStack that
owns the whole list (bridge_stream, memory banks, cache.clear, reasoning,
`_intent_stream=None`, `_last_salience=None`, `_last_bus=None`,
`_gs_velocity=None`) — both loops call it; add a test that diffs the reset set
against a grep of forward-side attributes.

Locks: **NO LOCK**. `smoke_notebook.py` hand-writes a *paraphrase* of the loop
without rotation at all (lines 61-90).

---

### F-05 — train.py resume guard: hold-out cursor reset is immediately clobbered (dead store) | MEDIUM | VERIFIED-REPRO (logic)

`train.py:354-357`:
```python
    stream_idx = resumed_stream_idx   # continue the data cursor (audit M12)
    if stream_idx >= max(len(streams) - _hold_n, 1):   # pre-M13 cursor (M13)
        stream_idx, offset = 0, 0
    offset = resumed_offset
```
The guard (M13) zeroes `offset`, then line 357 **unconditionally restores the
stale `resumed_offset`**. Repro R7:
```
train.py : stream_idx=0 offset=123456   (guard defeated)
notebook : stream_idx=0 offset=0        (c10:6-12 assigns offset BEFORE the guard — correct order)
```
Consequence: a pre-M13 checkpoint (cursor pointing into what is now the
3-file hold-out) resumes at `stream_idx=0` **with the hold-out file's byte
offset** — usually mid-something on a different, still-trained file (no
contamination), or > len → the rotation guard at :402 resets it on step 0
(then `offset == 0` fires rotation anyway). So it is self-healing *eventually*,
but violates the guard's intent (fresh document) and is a 1-line order bug.

Fix (NOT applied): move line 357 above the `if`.
Locks: **NO LOCK**.

---

### F-06 — train.py `evaluate()` returns 0.0 for an empty/tiny eval pool → "best" checkpoint saved with fake perfect val | MEDIUM | VERIFIED-REPRO

`train.py:699-742` ends `return total_loss / max(total_steps, 1)` — if every
hold-out file is shorter than `batch_size*seq_len+1` the loop `continue`s
(:723-724) and the function returns **0.0** (no `_val_ok` concept). The caller
(:667 `if val_loss < best_val_loss:`) then treats 0.0 as a record improvement
and writes best.pt (:670-688) claiming `best_val_loss=0.0`.
Repro R8 (train.py's own `evaluate` + stub model + 50-token hold-out file):
```
train.evaluate(tiny hold-out) = 0.0  0.0 < inf -> True
```
The notebook got this right: c10:369/388 `_val_ok` flag →
`EVAL ... NO HOLD-OUT DATA` and NO depth/scheduler/watchdog/arm_ce updates,
no save (c10:418-421). Divergence + real hazard for small `--data-dir` smoke
runs and for CI fixtures.

Fix (NOT applied): mirror `_val_ok`; raise/return NaN when `total_steps == 0`.
Locks: **NO LOCK**.

---

### F-07 — Hard veto ceiling `K·ln2` doubled itself when B2 moved to twin_free (K=64) while the comments still compute 22.2; under `head_normalize=True` the CE being gated is softmax-over-vocab NLL, so "uniform-bit NLL" no longer describes it | MEDIUM | VERIFIED-REPRO

Both copies, `train.py:436-441` ≡ `c10:145-151`:
```python
            _ce_uni = float(getattr(model.lm_head, 'K', cfg.bind_K)) * 0.6931471805599453
            if ce_val > _ce_uni:
```
- `model.lm_head.K = build_codes(cfg).shape[1]` (embedding.py:213/240/330) =
  `cfg.code_dim`. **train.py defaults** (legacy, code_dim=32) → K=32 →
  ceiling 22.18 nats (what the M14 comments and `scripts/scan_garbage.py:5`
  still quote: "K·ln2 (32*0.693 = 22.2)"). **Notebook cell 4**
  (`codebook='twin_free', code_dim=64, vocab=65536`) → K=64 → ceiling
  **44.36 nats** — the repro built the real codebook:
  `twin_free_codes(65536,K=64,S=6).shape == (65536, 64)`.
- The quantity gated, however, is `ce_loss` from
  `SigmoidCodedHead.log_probs_for_target` with `head_normalize=True`
  (embedding.py:315-317 + cell 4 line 42): a **gathered log-softmax over
  V=65536**, whose *uniform* value is `ln 65536 = 11.09` nats, not K·ln2.
  K·ln2 is the ceiling of the per-bit Bernoulli branch (`normalize=False`,
  embedding.py:318-322). The "uniform-bit NLL = zero-information ceiling"
  rationale holds only for the factorized loss; with normalization the
  threshold is an arbitrary multiple of uniform (2× uniform legacy, 4× uniform
  twin_free).
- Live-incident calibration breaks: the M14 comment cites the 2026-09 garbage
  region "CE drifted 5.8→34→67"; the hard veto was specced to catch ≥22.2.
  Under the current cell-4 config, a CE-34 region **passes** the hard veto
  (34 < 44.36) — only the soft veto (F-08) still catches it. A cfg change
  silently halved the hard veto's coverage radius with zero code change and
  zero comment update.
- The `cfg.bind_K` fallback (`bind_K` = binding width, 32/64) is *almost* dead
  (all three head classes define `.K`) but if it ever fires it is the WRONG
  quantity — a head-K vs bind-K conflation waiting for the day a head lacks
  `.K`.

Fix (NOT applied): express the ceiling from the actual loss model:
`cfg.vocab`-based uniform + slack (`ln V + 3·σ_batch`), or derive
`min(K,·)·ln2` per head branch and assert the head branch matches the formula;
update `scan_garbage.py` docstring. Locks: **NO LOCK** (no veto test exists —
grep of tests/ for veto: zero hits).

---

### F-08 — Soft veto numeric surface: false-veto windows after `arm_ce`, a degenerate zero-baseline accept-all→veto-all inversion, and NaN slipping through BOTH vetoes; NaN endgame differs per copy | MEDIUM | STRONG-READ + numeric repro

`train.py:468-476` ≡ `c10:184-193`:
```python
            _fc = watchdog._stats.get('ce')
            if _fc:
                _soft_thr = _fc[0] * (1.0 + 4.0 * watchdog.rel_margin)
                if ce_val > _soft_thr:
```
- `_fc[0]` is the **fast EMA** (a=0.99, half-life ≈ 69 —
  `core/training_control.py:195-197`); `rel_margin=0.15` (:200) → threshold
  `1.6 × fast_EMA`. Repro R10 with the real `FailureDetector`: after 100 steps
  at CE=11 → threshold 17.600, a CE=18 batch is vetoed.
- `arm_ce()` (training_control.py:319-328, called at train.py:665 / c10:394)
  **pops** the ce stats; the very next `check()` re-bootstraps
  `fast = seed_value` (:248-252). If the seed happens to be an easy batch
  (low CE), the 1.6× band sits under the healthy mean → false vetoes for up
  to ~1 fast half-life. Self-limiting because `check()` observes the batch
  BEFORE the veto-skip (stats keep moving toward the true level even while
  steps are skipped) — but each vetoed step also costs an LR step (F-10).
- Right after a *document rotation* into an unseen genre, first-batch CE
  spikes (banks emptied, cache cleared) while the EMA still carries the old
  genre's low level → first batches of the hardest genres get preferentially
  vetoed: a systematic, data-selection bias in the training stream itself.
- Degenerate baseline: 100 steps at CE=0 (near-deterministic file) →
  `_fc=[0.0,...]` is truthy, threshold 0.0, `any positive CE vetoes: True`
  (R10) — total veto-lockout of a stream region is possible only if CE was
  exactly 0 (SPECULATIVE reachability, mechanism STRONG).
- NaN semantics (float): `NaN > x` is False everywhere →
  a NaN batch **passes both vetoes** (R10: `NaN > thr = False`).
  `check()` forces an alarm on non-finite CE (training_control.py:363-369,
  stats stay clean — `_observe` is skipped), so `_rb=True` suppresses the
  soft-veto branch. Endgame diverges:
  train.py:506-507 `raise RuntimeError` → **crash**;
  notebook c10:218-223/259-261 `math.isfinite(disp_loss)` → **silently skip
  the step, keep training** (and consume one alarm strike per
  D6+ two-strike, c10:205-215). Same doctrine ("vetoes keep exploding
  batches out of the weights"), opposite failure policy.
- `inf > ceiling` is True → +inf hard-vetoes; −inf → passes both
  (a −inf CE is unlearnable nonsense that would then poison backward;
  SPECULATIVE).

Fix (NOT applied): gate the soft veto on `n >= _min_samples` of the ce stats
(`_fc[2]`), on `watchdog.ce_armed`, and on `math.isfinite(ce_val)`; align the
NaN policies of the two loops; make `evaluate()` non-vetoable regions impossible
(see F-06). Locks: **NO LOCK** (arm_ce *re-bootstrap* is locked at
test_product_invariants.py:857-870 — but not the veto that consumes it).

---

### F-09 — M12 resume drift: the notebook re-bootstraps the CE baseline after resume; train.py carries the old session's CE baseline through | MEDIUM | STRONG-READ

`c9 (cell 8b):96-99`:
```python
watchdog.load_state_dict(_resume_detector_sd)
# A resumed session is a regime change (fresh CUDA allocator, possibly
# fresh LR path): CE baseline from the old session must re-bootstrap.
watchdog._stats.pop('ce', None)
```
`train.py:331-336`:
```python
        # M12: full-state resume — watchdog/balancer baselines ride in best.pt
        watchdog.load_state_dict(ckpt.get('detector'))
```
— no pop. The stated rationale (regime change) is a behavior change; under it
the post-resume soft veto restarts from the first new batch (F-08 window)
where train.py keeps the historical baseline. Neither is wrong a priori — they
are **different policies on the same M12 decision**, i.e. exactly the
notebook/core divergence class AGENT_BRIEF:34-35 warns about.

Locks: `test_product_invariants.py:836-841` locks save/load of stats — not the
pop policy. **NO LOCK** for the divergence.

---

### F-10 — A veto skips `scheduler.step()`/`optimizer`/`depth.update()` but NOT `watchdog._cur_step`, LR-warmup accounting, or the loop's `step` — the internal clocks decouple | MEDIUM | STRONG-READ

Hard veto: `train.py:437-441` / `c10:146-151` — `continue` before
`depth.update` (443/152), before `check()` (465/177) and before
`scheduler.step()` (611/250). Soft veto: `train.py:468-476` / `c10:184-193` —
`continue` after `check()` but still before `scheduler.step()`. Meanwhile
`for step in range(...)` keeps advancing and `watchdog.check(ce_val, step)`
(465/177) receives the loop step, so `eval_block = self._cur_step < self.warmup`
(training_control.py:268) advances past warmup while the *LR* scheduler's own
counter (advanced once per non-vetoed step) lags — during a sustained veto
storm the run enters "post-warmup" watchdog semantics (B5 PH re-arm,
:333-342) with LR still mid-warmup. The M14b rationale ("drags the EMA toward
the spike") is honored, but the clock coupling was clearly never considered.
Also `tokens_seen` bookkeeping diverges between copies: counted **after** the
vetoes in train.py (:586) — vetoed tokens never appear in tok/s — vs **before**
in c10 (:132). Cosmetic, but it is the same two-copies-one-name divergence.

Fix (NOT applied): advance the scheduler by the loop-step delta (or drive
everything from a single `non_vetoed_step` counter documented in the contract).
Locks: **NO LOCK**.

---

### F-11 — train.py's by-name optimizer restore (`_restore_optimizer`) is DEAD CODE; the resume path does the positional load the docstring itself calls broken | HIGH (data-envelope integrity) | STRONG-READ

`train.py:73-82` docstring:
```python
def _restore_optimizer(optimizer, model, ckpt_opt):
    """Restore AdamW state BY PARAMETER NAME.

    Positional load shifts state onto wrong params whenever the parameter
    list changed (freq_scale/bind_coh_gate/W_out+K added): index i in the old
    checkpoint no longer refers to the same parameter. ...
```
`grep _restore_optimizer scripts/train.py` → **only the definition (line 73)**.
The actual resume is `train.py:306-312`:
```python
        optimizer = _make_opt(cfg.lr)
        if 'optimizer' in ckpt and ckpt['optimizer'] is not None and not args.no_save_optimizer:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
```
— a **positional** `load_state_dict` of a saved state. The notebook *does*
wire its own by-name version (c8:122-165, called at c9:44) — and the two
implementations have themselves drifted (W_out+K growth: train.py:110-124
pads and sets `exp_avg_sq=1.0`; c8:150-151 silently truncates `v[:p.shape[0]]`).
Additionally `train.py:307/673` reference `args` — a `__main__`-only global:
calling `train(cfg, resume_path=...)` as a library raises `NameError` the
first time a checkpoint contains an optimizer. Also `train.py:328` prints
`'Optimizer/scheduler rebuilt FRESH (no momentum restore)'` right after a
successful momentum restore — actively misleading forensic line.

Fix (NOT applied): call `_restore_optimizer(optimizer, model, ckpt['optimizer'])`
in the resume branch (mirroring c9), unify the two implementations, remove the
`args` reach-into from `train()`.
Locks: **NO LOCK** (no test executes train.py's resume path).

---

### F-12 — batch>1: `y`'s last column of row *i* is `x`'s first token of row *i+1* (flattened-stream shift) | LOW | VERIFIED-REPRO

Repro R1, file `[0..999]`, `get_batch(4, 2, 7, 65536)`:
```
x= [[7,8,9,10],[11,12,13,14]] y= [[8,9,10,11],[12,13,14,15]]
VERIFIED: y == x shifted by exactly one (inside rows); y[0,-1] == x[1,0];
offset out = 7 + 2*4 = 15 (contiguous, gap-free, overlap-free)
```
So the off-by-one is *correct in stream coordinates* (every target is the true
next token, both copies byte-identical in arithmetic — R1) but the row-1
boundary column trains the model to predict a token whose context it never
saw (standard nanoGPT flattening, 1/seq_len of the CE mass; ≤ 2 % at L=64 …
0.2 % at L=512). The notebook (batch=1, c5:2) never hits it; train.py default
batch=2 does. No document-boundary awareness inside a batch: a 512-token
window can straddle two source documents only via wrap (impossible in-loop,
see F-13) — rows of a batch>1 read ONE contiguous slice (rows i and i+1 are
adjacent windows of the same file). Documented consequence for agent 5
(val metrics: train.py's val_ppl is inflated by ~1/seq per row boundary).
Locks: none needed; noting for the contract.

---

### F-13 — `get_batch`'s internal EOF rewind (re-read token 0 of the SAME stream without any state reset) is unreachable from the two loops but live for any other caller; both loops ignore the returned flag | LOW | VERIFIED-REPRO + STRONG-READ

Repro R5: `get_batch(4, 2, 995)` → `wrapped=True`, batch from offset 0, new
offset 8 — i.e. *same stream, no rotation, no state reset, cursor teleports to
8*. In both loops the rotation block guarantees `offset + need <= len` before
every call (train.py:401-418 / c10:62-77: identical `_need` formula) — so the
wrap is dead *there*; `evaluate()` explicitly breaks on `wrapped`
(train.py:729-730, c10:379-380) — also fine. But `train.py:418` binds the flag
to `_wrapped` and never reads it (c10:77 same), and `scripts/generate.py`,
`analyze.py` and future callers can trip it. Defense-in-depth: the flag exists
because of M8; assert `not _wrapped` instead of discarding it.
Lock: `test_product_invariants.py:659-686` (train.py copy only, R-contract).

---

### F-14 — train.py CLI flags that do not exist as behavior: `--head`, `--amp-obj`, `--no-amp-pred` (and `--save-interval`) | LOW | STRONG-READ

`train.py:776-786` defines `--head {partitioned,codec}`, `--amp-obj`,
`--no-amp-pred`, `--traj-*`. `grep args.head args.amp_obj args.no_amp_pred` →
**zero uses**; the `EVAConfig(...)` build (793-824) never passes `head_mode`
→ `core/config.py:60` default `"sigmoid_coded"` (stack.py:31-40) regardless of
the flag. A user selecting `--head partitioned` silently trains/loads a
sigmoid_coded head (then `migrate_state_dict`/`load_state_dict` strict=False
absorbs the shape surprise, train.py:266-273). `--save-interval` is stored in
cfg but `Periodic step_*.pt checkpoints DISABLED` (:690) — dead knob, matching
M12 single-file doctrine but undocumented in `--help`.
Fix (NOT applied): wire or remove. Locks: NO LOCK.

---

### F-15 — train.py's entire log line is nested under `device == 'cuda'` — silent loop on CPU | LOW | STRONG-READ (indentation evidence)

`train.py:625` opens `if step % cfg.log_interval == 0:` computing `lc`,
`aux_str`, `gate_str`, `mod_scl` (626-641); then :644
```python
            if device == 'cuda' and step % max(cfg.log_interval, 1) == 0:
```
owns the actual `print(...)` at :653 (same block, 16-space indent — the
memgov governor lines :645-652 are inside it). On CPU: zero log output. Not a
correctness bug on the GPU production path; flagged because CPU smoke runs
look hung. Locks: NO LOCK.

---

### F-16 — Notebook OOM-retry permanently skips ~half a batch of corpus AND mutates the checkpointed `cfg` | MEDIUM | STRONG-READ

`c10:108-113`:
```python
                if cfg.seq_len > 64 and batch_size == 1:
                    cfg.seq_len //= 2
                    x, y, offset, _w = streams[stream_idx].get_batch(cfg.seq_len, batch_size, offset)
```
At this point `offset` already advanced by the *original* `batch*seq_len`
(get_batch succeeded at :77 — the OOM happened in the forward). The retry
reads a shorter window from the *new* offset: the region
`[old_offset + b·S/2, old_offset + b·S)` is consumed into discarded tensors
— a silent, repeating data hole (every OOM), with the *previous* attempt's
x,y (valid tokens, already paid) thrown away. Worse: `cfg.seq_len` is the
same object pickled into best.pt (`'cfg': cfg`, c10:404; train.py:677), so a
transient OOM shrink propagates into every resume until the M15 regrow
(c10:255-258) probes it back — and `orig_seq_len` is *not* in the envelope,
so a resumed session caps regrow at the already-shrunk value forever
(`orig_seq_len = cfg.seq_len` captured at c9:111 *after* cell 4 — fine within
one session, wrong across resume). train.py has no OOM handler at all (OOM →
crash → Ctrl+C path saves nothing, :691-694).
Fix (NOT applied): on retry re-read from the *pre-attempt* offset
(`offset_pre` saved before get_batch); persist `orig_seq_len`.
Locks: NO LOCK.

---

### F-17 — Eval start-region divergence: train.py reads hold-outs from `len//2`, notebook from `len//4`; train.py's comment claims "3/4-region" | MEDIUM (metric comparability) | STRONG-READ

train.py:725: `offset = max(stream.len // 2, cfg.batch_size * cfg.seq_len + 1)`
with comment :719-720 "each read from its 3/4-region".
c10:374: `voff = max(eval_stream.len // 4, batch_size * cfg.seq_len + 1)`
(comment c10:356-359 "read from its 3/4-region" — `len//4` start = the last
75 % ✓ matches the comment for the notebook only).
Consequences: (i) val_loss numerics from the two entry points are not
comparable across copies/sessions and neither is "the 3/4 region" as
train.py's comment claims; (ii) window geometry is otherwise identical:
contiguous stride `batch·seq`, per-file budget `min(100//hold_n, len//(bs·seq))`
(train.py:726-727 ≡ c10:375-376), `wrapped → break` (no re-read), fresh
`state=None` per batch, `adaptive=False` (train.py:733 ≡ c10:383). Within an
eval, **no overlap between successive batches** (stride = window length; the
+1 lookahead token is consumed as the next row's first input). Train/eval doc
disjointness holds *by cursor construction* (pool bound `max(len(streams) -
_hold_n, 1)`, train.py:405 / c10:64 — hold-out files never sampled in the
training rotation) except the documented single-file case (_hold_n=1 ⇒
pool=streams[:len-1] when ≥2 files; with exactly 1 file `max(0,1)=1` → pool =
the hold-out file itself — M8 "single-stream semantics", honored verbatim in
both copies).
With the real 39-file corpus, `_hold_n=3` (`train.py:172` ≡ `c7:37`) and the
sorted() hold-out is alphabetically fixed: **TEACHER, THRILLER, WAR** — three
war/profession genres as the entire proxy for "the language"; val trends
encode that domain choice. (Handoff agent 5; quantifies external claim (b).)
Locks: NO LOCK (no test runs `evaluate()` with more than the R8-style stub).

---

### F-18 — EOS/vocab edge in the REAL data: EOS=2 trained (mask_eos=False), ids 0/1 never occur, id 65535 occurs mid-stream, and only train.py damps state on EOS | INFO (facts) + LOW (divergence) | VERIFIED-REPRO (scans)

Full-file scans (see transcript, all five sampled files):
- `min_id = 2` everywhere; ids **0 (`<|pad|>`) and 1 (`<|bos|>`) never occur**
  in the *_eos.bin streams (tokenizer JSON: 0=pad,1=bos,2=eos,3=unk;
  `vocab size: 50000` in BOTH shipped tokenizer files — the real .bin streams
  use ids beyond that tokenizer, i.e. the repo's tokenizer.json is NOT the
  generator of the high range; a 65536-vocab convention lives in the *files*,
  `russian_tokenizer/tokenizer_v65536.json` is byte-identical to
  `tokenizer.json`, sha f571fa52…).
- **id 2 = 5.7 % of all tokens** (FANTASY 18.5 M / 327 M) — sentence
  boundaries, consistent with every in-code assumption:
  `train.py:618` `if (y[:, -1] == 2).any():` ×0.1 state damping
  (**train.py only** — no equivalent in c10; divergence),
  `memory_bank.py:404` `is_sep = (tokens == 2)` (model-level, both copies),
  `mask_eos=False` (`core/config.py:27`, cell 4:12) — the decision to *train*
  EOS is honored in `losses.py:36-38` (mask only id 0 — which never occurs,
  so the mask is a no-op on real data).
- **id 65535 = max uint16 occurs 708× in FANTASY, 21× in CHILDREN, 1× in
  DOCUMENT — mid-file** (median gap 207k tokens in FANTASY), i.e. it is an
  in-stream symbol, not an EOF marker. With `vocab=65536` (notebook) it is a
  legal row: `twin_free_codes(65536,...)` → (65536,64), row 65535 exists with
  weight 6 (repro R9) → **the head CAN represent 65535**; token_bias also
  covers it (embedding.py:260 `torch.zeros(cfg.vocab)`). With any
  `vocab ≤ 65535` config, 65535 is clipped (train.py:57) or M9-warned+clamped
  (embedding.py:124-132) onto the last row.
- Assumptions `tokens < vocab`: enforced in exactly one copy (F-01) + the
  embedding guard; nothing asserts `cfg.vocab == 65536` ↔ uint16 alignment
  anywhere. The `Audit M9` warning (embedding.py:121-131) mentions reserved
  reasoning tokens THINK..END ≥ 65536 — with vocab=65536 and uint16 input
  those can never appear from TokenStream, only from generation.

Fix (NOT applied): pin `--vocab 65536` default in train.py; add the train.py
EOS-damping equivalent to the notebook or drop it (decision); record the
id-65535 sentinel meaning upstream (WideBind tokenizer pipeline).
Locks: NO LOCK on conventions.

---

### F-19 — The only TokenStream test locks the train.py copy through a hard-coded absolute machine path; the notebook loop is locked by nothing | LOW | STRONG-READ

`tests/test_product_invariants.py:664-665`:
```python
    spec = importlib.util.spec_from_file_location(
        '_train_mod', r'C:\Users\black\OneDrive\Desktop\EVA CLM\scripts\train.py')
```
On any other checkout the test still "passes" only because the extraction
falls back to reading the file at *that* path — it silently audits the
developer's disk, not the repo. `scripts/smoke_notebook.py` claims "faithful"
notebook-path coverage (line 1) but hand-copies a rotation-less, veto-less,
TokenStream-less loop (61-90) — it cannot catch ANY of F-01…F-10 on the
notebook side. Given AGENT_BRIEF:35's doctrine ("правки scripts/train.py на
Colab не применяются; меняй саму ячейку"), the twin-drift class is exactly
where the project bleeds, and it is the least locked surface.
Fix (NOT applied): repo-relative path; add a twin-parity test that execs BOTH
classes and asserts equality of (a) returned tensors on the same bytes for
`vocab=None/65536` and (b) signature parity; CI-run smoke_notebook against the
notebook JSON. Locks: partially self-referential.

---

### F-20 — train.py's AMP block wraps only `embed_tokens` — `use_amp=True` changes one op, not the model | LOW (latent) | STRONG-READ

`train.py:423-425`:
```python
            with autocast('cuda', enabled=use_amp):
                h = model.embed_tokens(x)
            out, state, gs, _ = model(h, state, global_state=gs, step=step, tokens=x)
```
The trunk forward is OUTSIDE the context; the GradScaler still scales the
losses (:527-530) and `scaler.unscale_` runs (:593) — with `use_amp=False`
everywhere today (config `core/config.py:58`, cell 4:64, AGENT_BRIEF:34 "fp32
обязателен") this is dead code, but anyone toggling the flag gets *silently
different* semantics than the notebook's bf16-everything policy
(c10:86 wraps the whole forward; c5:12-14 bf16-only). Divergence-by-construction
in the same "two copies" family. Fix (NOT applied): move the two lines into
the context or drop AMP from train.py. Locks: NO LOCK.

---

### F-21 — M12 RNG envelope covers CPU-default + doc-shuffle generators only; the per-step stochasticity lives in the CUDA generator → "full restart state" is false on the production (GPU) host | MEDIUM | STRONG-READ (consumer census) + VERIFIED-REPRO (CPU determinism holds; CUDA unavailable on this host)

Envelope (identical both copies): save `train.py:685` ≡ `c10:412`
`'rng': torch.get_rng_state(), 'data_rng': rng.get_state()`; load
`train.py:335-336,345-348` ≡ `c10:17-20`. Consumers found in the live path:

| consumer | generator | covered? |
|---|---|---|
| doc rotation pick `torch.randint(..., generator=rng)` train.py:405 / c10:64 | dedicated CPU `rng` | YES (`data_rng`) — repro R11: state round-trip reproduces the draw sequence |
| `logit_cache` scheduled sampling `torch.rand(1)` logit_cache.py:417-418 (5 %/step, `training` only) | **CPU default** (no device arg) | YES (`rng`) — repro R11: get/set round-trip is exact |
| VSA write noise `torch.randn_like(i_gate)` block.py:401-403 — `noise_scale>0` (AdaptiveController range 0.001…0.05, config.py:188-189) **and training** → every step × every layer | **device default** (CUDA on Colab) | **NO** — `torch.get_rng_state()` is documented "the random number generator state" of the **CPU** generator; CUDA generators (`torch.cuda.get_rng_state`) are never saved |
| stable-rank aux power-iteration seed `torch.randn(..., device=bind_W.device)` losses.py:146 (once per layer per session — `_pi_v` not checkpointed) | device default | NO |
| traj-manifold `randperm` bind.py:615 (only with `traj_manifold=True`, off) | device | NO (moot) |
| CPython `random`, numpy RNG | — | none in this path (grep census; memmap reads are deterministic) |

So **resume determinism holds exactly on CPU** (both repros green) and
*cannot* hold bit-exactly on the T4/A100 the loop actually runs on — the
M12 comment "one checkpoint carries the FULL restart state" (train.py:680)
overstates. Fix (NOT applied): `torch.cuda.get_rng_state_all()` into the
envelope + restore guarded by device_count, or route the per-step noise
through a dedicated saveable generator. Locks: `test_product_invariants.py:836-841`
(watchdog stats only).

---

## 2. State/data contract this block produces / consumes

(§2 is the normative handoff for agents 2a, 2b, 3, 4, 5 — everything a later
agent assumes about the data path is written here, pedantically.)

### 2.1 Tensor flow, dtypes, devices

```
uint16 .bin ──np.memmap(mode='r')── TokenStream.data (numpy uint16, read-only,
                                     lazy page-cache; .len = token count)
  get_batch(seq_len, batch, offset, [vocab]):
     chunk  = data[o : o + b·s + 1]              # numpy uint16 view → np.clip copy
     x      = chunk[:b·s].reshape(b, s)          # uint16, .copy() materialized
     y      = chunk[1:b·s+1].reshape(b, s)       # = x flattened-shifted-by-1 (R1)
     return x.long(), y.long()                   # int64, CPU
  loop: x, y = x.to(device), y.to(device)        # train.py:420 / c10:82
  h = model.embed_tokens(x)                      # float32 (B,L,D) — AMP: only here in train.py (F-20)
  out = model(h, state, gs, step, tokens=x)      # float32 (B,L,D) + side-effects §2.5
  ce   = compute_losses(out, y, h_emb=h)         # scalar float32; targets flattened row-major consistently
```
- `y[i,t] = stream[o + i·s + t + 1]` — cross-row spillover at
  `t = s-1` (F-12). `mask = targets != 0` in losses (losses.py:36) — vacuous on
  real data (no id 0 occurs; F-18).
- Model expectations: int64 indices `< cfg.vocab` — guaranteed by clip
  (train.py only), by embedding's clamp+warn (embedding.py:119-132, both),
  and by the uint16-vs-65536 arithmetic in the notebook path ONLY.
- dtype of `ce_val` = python float (`ce_loss.item()`) — veto comparisons are
  IEEE floats (NaN semantics F-08).

### 2.2 RNG census → see F-21 table (generators, consumers, coverage).

### 2.3 Cursor / offset semantics

- State: `stream_idx ∈ [0, max(N−hold,1))`, `offset ∈ [0, len]` — a byte
  position (token index) **in the current stream**; `offset==0` is overloaded:
  "fresh start" AND "rotate now".
- Per step: `_need = batch·seq_len + 1`; if `offset == 0 or offset + _need >
  len(streams[stream_idx])` → ROTATION EVENT (§2.4), pick
  `stream_idx ← Uniform[0, max(N−hold,1))` from the dedicated CPU generator
  `rng` (seed 42, state in envelope). Then exactly one `get_batch`; returned
  `offset = old + batch·seq_len`; **the +1 lookahead is never skipped** (next
  batch starts where x's coverage ended: y consumed it).
- In-loop `wrapped` from get_batch is mathematically impossible after
  rotation (§F-13) — the guard and get_batch use the same `_need` formula;
  kept as contract only for evaluate() (break) and external callers.
- Checkpoint boundary: best.pt/KeyboardInterrupt save `('stream_idx','offset')`
  mid-document; resume re-enters at that byte and the rotation fires only on
  exhaustion. Pre-M13 guards: train.py:355 (broken by F-05), c10:9-12 (works).
- Vetoed batch: cursor ALREADY advanced (get_batch precedes veto) → bad batch
  is *skipped*, never retried (R13: worst-case full-veto pass is O(len/token)
  then rotation; **no infinite re-veto of the same region on any document,
  including the last training file** — rotation replaces the cursor, doesn't
  rewind it).
- Hold-out files (`streams[-3:]` on 39-file corpus: TEACHER/THRILLER/WAR) are
  never rotation candidates; `evaluate()` owns them read-only from their
  midpoints (§F-17).

### 2.4 Document-boundary event schema (the rotation block)

Fired when (§2.3): resets `state`, `gs` (+ `intent_state`, notebook),
`bridge.bridge_stream.zero_()`, `memory_bank.reset()` (**F-02 — zeroes
params**), `logit_cache.cache.clear()` (complete: logit_cache.py:173-178),
`model.reset_reasoning()`. Does NOT reset — the exhaustive leak list of F-04:
`_intent_stream` (and it overrides the loop's own reset), `_last_salience`,
`_last_bus`, `_gs_velocity`, all mirror EMAs incl. the *persistent*
`_private_mem`/`_delta_var`/`_gate_ema`/`_concept_sim_ema`/… , bridge
readiness (`_tgt_mean`, `bridge_loss_*`, `inj_ratio`), block observers
(`_mlp_*_ema`, `_mlp_ratio`, `_mlp_cnt`), UCL concepts/counters, bind
`_step_count`, `_pi_v`, `layer.b_i/b_d`, `_phase_ratio_*`, scheduler/optimizer
counters. Contract for downstream agents: **treat the document boundary as
"stream buffers cleared" only for the four objects named in the block;
everything else in F-04's table flows across "documents".**

### 2.5 forward-side side effects (what a forward mutates; rotation ✗/✓, eval snap ✗/✓)

Fully enumerated in the F-04 table (rows 1-16) — that table IS the contract
requested in mission point 3. Additional eval-mutability notes: eval passes
`step=None, adaptive=False` → skips AdaptiveController b_i/b_d writes
(stack.py:271), momentum velocity (:348), scheduled sampling
(logit_cache.py:417), reasoning-attr write (stack.py:234,664), triad re-pass
(:678-681 needs step≠None) — all correctly isolated — **except** the memory
bank L2 parameter writes (F-03) and mirror's tied `W_out.copy_(W_projᵀ)`
(idempotent re-derivation, harmless; mirror.py:384).

### 2.6 best.pt envelope (both copies; save sites train.py:670-686,
c10:399-413, KeyboardInterrupt c10:441-453)

```
step, model, optimizer(+param_names), scheduler, best_val_loss, cfg,
reasoning_enabled_step, recover_count, active_depth, detector (watchdog stats:
recover_count/ce_armed/cooldown/viol/stats/last_viol_name), balancer,
stream_idx, offset, rng (CPU default gen), data_rng (doc-shuffle gen)
```
Gaps: CUDA generator (F-21); notebook's KeyboardInterrupt save drops
`reasoning_enabled_step` (compare c10:441-453 vs c10:399-413 — the interrupt
save is missing the key → reasoning ramp resets to 0 on a Ctrl+C resume;
train.py never saves on Ctrl+C at all, :691-694); `_last_salience/_intent_
stream/_pi_v/orig_seq_len` not in envelope (F-16, F-04).

### 2.7 Eval call contract

`evaluate`: `model.eval()`, logit_cache cleared before AND after, fresh
`reasoning` before (train.py:712-715 ≡ c10:350-355), snapshot/restore around
(F-03 exception), per-stream `state=None` per batch, `adaptive=False`, budget
`min(100//hold_n, len//(bs·seq))` batches from midpoint, `wrapped→break`,
`model.train()` at exit (train.py:741 ≡ c10:426). Side effect on control
objects: `watchdog.arm_ce()` + `scheduler.report_val_loss` + `depth.update`
only on a *successful* eval (notebook gates on `_val_ok`, train.py does not —
F-06).

---

## 3. Prior decisions in code comments touching this path — honored?

| decision | where declared | honored? |
|---|---|---|
| **M8** rotation reachable before read; state reset at boundaries; `wrapped` 4-tuple; val uses hold-out; salience from head logits | train.py:60-64, 370-372, 394-418, 426-429; c7:14-17, c10:56-76, 89 | **Partially.** Rotation reachable ✓ (both); the four resets ✓, but "document state" was under-specified: `_intent_stream`/`_last_salience`/`_last_bus` flow across documents (F-04) and the notebook's `intent_state=None` is provably overridden (stack.py:333). Salience-from-logits ✓ train.py:429/c10:89. Hold-out ✓ except the documented 1-file degenerate case. |
| **M8** `stream.len` attribute crash fix in evaluate | train.py:703-709 comment | ✓ used at :723,725. |
| **M9** loud once-warning for out-of-vocab ids | embedding.py:120-131 | ✓ present; but train.py's np.clip (F-01) fires BEFORE the warning can — ids ≥50000 never reach the embedding under default CLI → the M9 sentinel is blind to the 50000-clip corruption. |
| **M12** envelope = full restart state ("cursor/RNG/val-history all ride in best.pt", cell 4 comment :18; train.py:680) | both | **Half-true**: keys exist & are restored in both copies ✓ (train.py:335-338,345-348; c10:17-20, c8:172-179) — CPU-determinism reproducible (R11) but CUDA noise not covered (F-21); CE-baseline policy diverged (F-09); train.py positional optimizer load vs the by-name fix it ships (F-11); KeyboardInterrupt save misses `reasoning_enabled_step` (c10:441-453). |
| **M13** 3-file file-level hold-out, budget preserved | train.py:170-172; c7:32-41 | ✓ both; alphabetically-fixed hold-out domain noted (F-17); guard bug F-05 (train.py), correct in c10. |
| **M14** hard veto = "uniform-bit NLL" ceiling | train.py:433-441; c10:136-151 | ✓ code present in both, but the ceiling's numerical identity broke with B2 (F-07): comment 22.2 vs live 44.36; CE-34 incident class now passes. |
| **M14b** soft veto `> fast·(1+4·rel_margin)` | train.py:466-476; c10:178-193 | ✓ both, formula-identical; NaN slips (F-08), unarm windows (F-08), clock decoupling (F-10). |
| **D6 / D6+** sensor-only, two-strike stop, no auto-rollback | training_control.py:131-140; train.py:477-495; c10:194-215, c9:83-91 | ✓ honored (train.py `sys.exit(2)`, notebook `_alarm_stop`+break). Stale comment train.py:444 ("rollback + fresh Adam + LR rewind") contradicts the code it annotates — forensic hazard. |
| **B3** best.pt only advances on an improving *isolated* eval | c10:199-200 comment, :395; train.py:667 | ✓ gated on `val < best` in both; but isolation is breached (F-03) and train.py can "improve" on fake 0.0 val (F-06). |
| **B4/B5** single-source margins/floors; PH re-arm after warmup freeze | training_control.py:187-204, 262-268, 333-341 | ✓ both loops construct `FailureDetector(model, k_sigma=3.0, warmup=cfg.warmup_steps)` identically (train.py:222/327; c9:91). |
| **decision #3 / R6** cache boundaries | train.py:413-414, 302-304, 712-713, 738-739; c10:71-72, 350-352, 422-424 | ✓ cleared at rotation, eval both ends, resume (train.py; notebook cache starts empty post-restart — equivalent). `clear()` complete (logit_cache.py:173-178). |
| **M16** memory governor hysteresis 0.85/0.70 | train.py:642-652; c10:278-291 | ✓ mirrored (train.py variant is inside the cuda-gated print block, F-15). |
| **M5** gradalign in-core | train.py:497-503; c10:93-96 | ✓ both declare it; live flag diverged: c10:26 hard-sets `cfg.gradalign_weight = 0.3`, train.py leaves cfg (0.0 default) → **gradalign is ON in Colab, OFF via train.py** with identical code comments. |

---

## 4. Cross-check of the external LLM's claims

**(a) "vocab/eos edge unexamined" — PARTIALLY REFUTED, direction confirmed.**
The repo *has* examined pieces: the M9 out-of-vocab warning
(embedding.py:120-131), the uint16≥vocab clip rationale (train.py:54-56),
`mask_eos` explicitly documented and decided against (config.py:26-27), EOS=2
assumed consistently in three places, twin_free's 65 536 sizing comment
(vsa_utils.py:64-73). **But** the edge was never quantified against the real
corpus: no artifact shows anyone counted ids ≥ 50000 or id 65535 in the .bin
files (`scan_garbage.py` looks for *entropy*, not the id ceiling). The 7.8 %
≥50000 fraction, the in-stream id 65535, the dead `--vocab 50000` default and
its collision with the notebook's 65536 are new (F-01/F-18). Claim's spirit
stands.

**(b) "train/eval contamination unquantified" — CONFIRMED, and worse than asked.**
Cursor-level file disjointness is real (§2.3, pool bound train.py:405/c10:64 +
midpoint reads) and the 1-file degenerate case is documented. But *state*-level
contamination was quantified here for the first time: validation documents are
written into the training parameters `L2.keys/vals` by the eval forward
(F-03, VERIFIED-REPRO with exact tensor bookkeeping), the eval `write_idx`
counter is rolled back under the leak (torn bank), and both loops read eval
regions that disagree (len//2 vs len//4, F-17). Also the hold-out IS the
alphabetical tail (TEACHER/THRILLER/WAR) — a domain prior baked into every
val_loss number ever printed (F-17).

**(c) "cursor persistence works (M12)" — MOSTLY CONFIRMED, with three
exceptions.** The keys exist, are saved in both copies and both resume paths,
mid-document continuation is real, and CPU-side determinism is reproducible
(R11: doc-shuffle sequence and CPU-default draws bit-identical after
set_state). Exceptions: (i) on CUDA (the only production platform) per-step
stochasticity consumers are outside the envelope (F-21); (ii) train.py's
holdout-cursor guard is defeated by a dead store (F-05); (iii) the *optimizer*
companion of "persistence" in train.py is positional-load, with the by-name
fix shipped but never called (F-11) — momentum "persisted" into possibly-wrong
parameters after any architecture edit; and the notebook's Ctrl+C save drops
`reasoning_enabled_step` (§2.6). Verdict: persistence *of the cursor* works;
"full restart state" overstates.

---

## 5. Not verified + explicit handoffs

Not verified here (out of scope or infeasible on this host):
1. Anything requiring CUDA (TF32/bf16 paths, `cuda.get_rng_state_all`, OOM
   handler c10:98-130, memgov pulses).
2. Real training-run forensics (logs/, checkponts/, best.pt contents) — the
   envelope was checked at code level only.
3. The WideBind *writer* pipeline (how ids ≥50000/65535 were produced — the
   repo ships an inconsistent 50 000-token tokenizer JSON next to 65 536-wide
   streams).
4. Whether F-02's parameter zeroing is *intended* memory semantics or an
   oversight (needs the architect's word — flagged both ways).
5. Repro of the CE≈34 pass-through (F-07) on the live checkpoint — needs the
   actual best.pt + corpus region.

Handoffs — addressed:
- **→ Agent 2a (head/codes)**: F-01/F-07/F-18. `SigmoidCodedHead.K` is the
  veto ceiling; decide the loss-mode semantics the veto should assume
  (normalize=True ⇒ ln V scale, K·ln2 stale). token_bias (vocab=65536) and
  codebook row 65535 exist — confirm head can *emit* 65535 (argmax space) and
  that `migrate_state_dict`'s `W_out +K` growth interacts with `K=64` heads the
  way F-11's restore assumes.
- **→ Agent 2b (optimizer/param groups)**: F-02 (Adam moments under
  in-place-zeroed L2 keys/vals; momentum desync at every rotation),
  F-11 (positional restore active; two drifted `_restore_optimizer`
  implementations; train.py's `param_names` writer (:674) vs notebook's
  (c10:402) — same list, different construction),
  F-10 (veto-skipped `scheduler.step()` vs watchdog `_cur_step` divergence),
  stack.py:299-306 (AdaptiveController writes `b_i/b_d` *Parameters* every
  train step outside the optimizer).
- **→ Agent 3 (memory/banks/reasoning)**: the F-04 leak table is yours.
  Priorities: `_intent_stream` priority override (stack.py:333) making the
  loop-level intent reset a no-op; `_private_mem` persistent=True =
  cross-document AND cross-restart expert memory (rotation cannot reach it);
  `_reasoning_buffer` is *never written back* in the adaptive-gate config
  (`_adaptive_reasoning` discards buf/count; stack.py:659 vs 664-666) →
  cross-step chain-of-thought is dead in the notebook config — decide whether
  that is the "false reasoning persistence" M-something intended;
  F-03 (L2 eval writes) needs a write/no-write policy for `not self.training`.
- **→ Agent 4 (losses/watchdog)**: F-07/F-08/F-09/F-21 +
  `compute_losses` side effects on the RNG (`_pi_v` seed, losses.py:146) and
  on aux_dict contents the veto formulas read (`watchdog._stats`).
- **→ Agent 5 (eval/metrics doctrine)**: F-03, F-06, F-17 (val-region
  divergence, fake 0.0 val, alphabetical hold-out domain trio,
  val_loss incomparability between the two loops), F-12 (row-boundary CE
  inflation is batch=2-only → train.py's val_ppl has a systematic term the
  notebook's doesn't), §2.7.

---

### Appendix A — repro manifest (all green; script at `%TEMP%\opencode\repro1.py`)

R1/R2 shift-exactness + stride; R3 clip divergence table; R4 reshape crash
(both copies, `cannot reshape array of size 6 into shape (2,4)`); R5 wrap
rewind; R6 rotation-guard sufficiency (False ⇒ pool-length assumption);
R7 dead-store; R8 evaluate()→0.0; R9 twin_free (65536,64) + ceilings
22.18/44.36 + row-65535 weight-6; R10 FailureDetector thresholds, NaN/inf/0
semantics; R11 RNG round-trips; R12 eval-writes-params (write_idx 0→2→0,
keys mutated→still mutated); R13 all-veto liveness (249 steps/pass);
R14 corpus scan facts. Suite: 284 passed unchanged.
