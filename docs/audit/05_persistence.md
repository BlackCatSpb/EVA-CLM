# 05 PERSISTENCE AUDIT

Agent 5 (final in relay) — persistence & resume integrity of the `best.pt` envelope.
Tree audited: `ee3a7dc` (B14 committed), baseline suite **314 passed** before and after
(this artifact adds zero repo changes). Both copies read end-to-end: `scripts/train.py`
(887 lines) + `notebooks/eva_colab.ipynb` cells 4/8/9/10, plus every state-owning module
(`adaptation`, `lr_scheduler`, `training_control`, `stack`, `mirror`, `block`, `bind`,
`memory_bank`, `maturation`, `concept_layer`, `bridge`, `embedding`, `logit_cache`,
`adaptive_gate`, `tau_config`, `eva_optim`, `migrate`).

Method: full AST census of every `register_buffer(...)` (112 registrations, 39 of them
`persistent=False`), every save site, every `state_dict/load_state_dict`; then CPU probes
running the REAL code paths (`importlib` on `train.py`, `exec` of the real cell-8 source).
Every finding is tagged **[VERIFIED-REPRO]** (probe output below), **[STRONG-READ]**
(anchor-level certainty, no probe) or **[SPECULATIVE]**.

**Mission 1 headline answer — "is EVERY mutable runtime state covered by save/resume?"**
No. The envelope is complete for *parameters and persistent buffers* and for the data
cursor/RNG, but six state classes live outside it (streaming loop state, one-step attrs,
deferred alpha pending, CUDA RNG, session policy counters, live cfg edits), and in the
NOTEBOOK copy — the one that actually runs Colab — the envelope is *contaminated*:
best.pt is written **inside** the eval snapshot window, baking 22 eval-mutated persistent
buffers (including hold-out-derived UCL concepts and bridge stream content) into the
artifact the D6+ doctrine calls "the last CLEAN save". And in BOTH copies the resume
entry point is broken: train.py crashes unconditionally at train.py:284; the notebook's
cell 8 raises NameError on a fresh-session resume. The M12 claim "one checkpoint carries
the FULL restart state" (train.py:726) is still false after B6–B14.

---

## 1. Findings

### CRITICAL — train.py's run/save/resume paths cannot complete (five independent crashes, all outside any test's reach)

`rg` census: **zero tests import or call `scripts/train.py:train()`** (only
`EVAStack.train()` mode toggles — 314 green tests never touch the loop that the audit
mission is about). Every finding below was found by *executing* the real loop on CPU
mini-data (probes P3/P5, §3).

**F5-01 [VERIFIED-REPRO] train.py resume crashes on EVERY attempt — the str-unpack
contract break.** `train.py:284`
`missing, unexpected = verify_identity_resume(model, ckpt, _skipped)` unpacks a **string**
into two names. `training_control.py:461` `return fp_now` (e.g. `'1024x16-76cb265c…'`,
40+ chars) — `ValueError: too many values to unpack (expected 2)` fires on every resume
with an existing checkpoint, *after* the model load but *before* any optimizer/scheduler/
detector restore. The notebook (c8:96) calls it bare — correct. Consequence: the entire
train.py resume block (train.py:285–353) — including the B12 by-name optimizer wire
(train.py:323), B14 `depth.put_state` (train.py:347), the B13 detector normalize —
**has never executed in this file**. B8's own test (`tests/test_b8_regressions.py:38`)
asserts the str return, so the mismatch is locked in the wrong direction: nobody
re-read the *call site*. Fix: `missing, unexpected = model.load_state_dict(...)` is
already line 283; line 284 must drop the assignment and keep only the (fatal) call.

**F5-02 [VERIFIED-REPRO] train.py resume *and* save reference the `__main__`-only global
`args`.** `train.py:318` (`not args.no_save_optimizer`) and `train.py:719–720` read the
argparse namespace defined only under `if __name__ == '__main__'` (train.py:800/846).
`hasattr(train_module,'args')` → **False** when the module is imported; programmatic
`train(cfg, resume_path=…)` raises `NameError: args`. The function is not self-contained —
it silently depends on its own CLI frame. (This is also why no test could have found
F5-01 "easily": calling train() from a test hits F5-02 first on both paths.)

**F5-03 [VERIFIED-REPRO] train.py's momentum restore is a no-op: param_names is saved
TOP-LEVEL but looked up INSIDE the optimizer dict.** The writer stores
`'param_names': _opt_param_names(...)` as an envelope key (train.py:720); the by-name
restorer reads `ckpt_opt.get('param_names')` — i.e. inside `ckpt['optimizer']`
(train.py:91) — which `torch.optim.Optimizer.state_dict()` never contains. Live log from
the battery (`out_resume10.txt`): `WARNING: checkpoint has no param_names — optimizer
state NOT restored (fresh Adam)` → **train.py resume always runs on fresh Adam moments**,
the exact B12/F3-08 incident class ("positional restore shifts state" was fixed into
"restore nothing"). Then the code prints *both* `'Optimizer state restored BY NAME
(momentum preserved)'` (train.py:324 — unconditional, fires whenever `_restore_optimizer`
returns at all) *and* `'Optimizer/scheduler rebuilt FRESH (no momentum restore)'`
(train.py:342). The notebook reader checks `ckpt.get('param_names')` first (c8:127) — so
**the same envelope restores momentum in Colab and never in train.py**. B12's commit
message claims the fix "verified the call site replaced" — the argument plumbing is
still broken.

**F5-04 [VERIFIED-REPRO] train.py dies with TypeError immediately after its FIRST
best.pt save.** `train.py:734` `generate_report(save_path)`; the imported
`analyze.save_html_report` requires `(ckpt, cfg, model, wake, live, head, …)`
(analyze.py:1522 — 6 positional). The try/except at train.py:25–29 only guards the
*import*, not the call. Battery: first improving eval wrote best.pt, then
`TypeError: save_html_report() missing 5 required positional arguments` — exit 1. A
train.py run can save exactly once, then crashes. (The notebook has no report call.)

**F5-05 [VERIFIED-REPRO] train.py's default configuration crashes at step 0 before any
training.** Log block train.py:674–677: `rg = getattr(model,'reasoning_gate',None)`;
`gates = getattr(model,'_reasoning_gates',None)`; `if gates:` — `_reasoning_gates` is a
**tensor buffer** (stack.py:71, `zeros(reasoning_max_steps=8)`, persistent=False), so the
truthiness test raises `RuntimeError: Boolean value of Tensor with more than one value is
ambiguous`. `explicit_reasoning` defaults **True** (config.py:41 "canonical Colab stack")
and the log block runs at `step % log_interval == 0` → step 0 → crash. This is the shape
of the F-04-era "buffer-ification" hazard (B7/B10 moved plain attrs into buffers; the
loop wasn't swept). `gates is not None and gates.numel()` is the one-line fix.
Battery note: probes had to run with `explicit_reasoning=False` to exercise anything.

**F5-06 [VERIFIED-REPRO] The notebook writes best.pt INSIDE the eval snapshot window →
every eval-improving save bakes eval-mutated state into the envelope (train.py does not
— copies diverge).** Cell 10 order: `_rt_snap = snapshot_runtime_buffers()` (c10:377) →
eval forwards → `depth.update/report_val_loss/arm_ce` (c10:416–418) → **`torch.save(...)`
(c10:423–437)** → only then `restore_runtime_buffers(_rt_snap)` (c10:449). Probe P2 ran
the real eval pattern (model.eval(), adaptive=False, 6 batches) after 12 warm training
steps and diffed `state_dict()` at the three moments: **22 persistent state_dict keys are
dirty at the save instant**:

```
_bus_rms, tau_config._gate_tau_cache (×4 owners), layers.*.bind._step_count,
layers.*.mirror.hybrid_gate._last_indep_mean / _last_relative_entropy / _last_gate_std,
bridge.bridge_stream,
concept_layer.concept_keys / concept_vals / concept_age / concept_count / _step /
_n_updates / _n_skipped, maturation._gate_tau_cache
```

Two of these are not counters but **content**: `concept_layer.concept_keys/vals` are
updated with the hold-out documents (UCL was never given the B7 `self.training` gate that
saved L2Bank — the F-03 species survives in the concept store), and `bridge.bridge_stream`
(persistent, c10:70 zeroes it at rotation) carries **val-document text state** into
best.pt. Consequences: (a) every notebook "clean" resume starts with the concept layer
and bridge having partially memorized TEACHER/THRILLER/WAR — the very files every
subsequent val_loss is measured on, weakening the B3 val-gated "best" selection itself;
(b) hybrid-gate statistics (`_last_*` buffers) that tune `log_tau` arrive eval-biased;
(c) counters inflate (+ ~100 per save — `bind._step_count` is incremented under eval:
bind.py:400–401 gates on `_ggeo_freeze`, NOT on `self.training` — while mirror's
`_fwd_count` is correctly training-gated (mirror.py:754–755)). The **live** side is clean:
probe found *zero* snapshot blind spots (buffer census post-restore all restored, incl.
`__attrs__` `_last_bus`/`_intent_stream` value-exact) — the hole is purely save-vs-restore
*ordering* in the notebook, plus the same trap for Ctrl+C landing inside the eval block
(c10:461 handler saves whatever is live then — dirty by the same mechanism, and 01's
known `reasoning_enabled_step` drop, F5-11). train.py: `evaluate()` restores at :792 and
the save happens at :716 after it returns — the train.py envelope is clean. Fix: move the
c10 save after `restore_runtime_buffers` (one line of reordering), and/or gate UCL +
`bridge_stream` + `bind._step_count` writes on `self.training`.

### HIGH

**F5-07 [VERIFIED-REPRO] Streaming loop state is not in the envelope — a mid-document
resume is a cold document start.** `state` (per-layer VSA stream), `gs`, `intent_state`,
and the one-step attrs `model._last_salience`, `model._intent_stream`, `model._last_bus`,
`model._last_intent_state` are loop locals/plain attrs — never saved (01 §2.4/F-16, now
quantified from the persistence side). Battery (P3, identical seeds/cfg/weights,
2-layer mini, save at step 10 mid-document, fresh process, resume, 10 steps):
cursor-aligned **max |ΔCE| = 0.367 nat**; mid-warmup save (step 5): **0.460 nat**.
Attribution: seeding the orphaned `_phase_ratio_ema/_std` from the baseline into the
resume changed nothing (0.3666396 vs 0.3666382 — not the driver at this horizon);
the controlled replay (probe (c): same weights, same cursor, same RNG, the ONLY
difference = `state/gs` flushed at step 10) reproduces the scale directly:
**first-step |ΔCE| = 0.444, second 0.062, then data-tracking**. The saved cursor says
"continue inside this document at byte 698k" while every stream buffer says "new
document" — an *inconsistent* state, not merely a lossy one (rotation resets buffers at
offset 0; resume never rotates yet has rotated-memory). Options are in §5 (save the state,
or resume-at-boundary doctrine).

**F5-08 [VERIFIED-REPRO] The B6 ggeo diagnostic window omits mirror modules, so B13's
F4-01 fix is defeated every `log_interval` steps.** The freeze selector at
train.py:556–558 / c10:231–233 covers only `hasattr(_m,'_step_count')`; mirror modules
are matched by `_alpha_pending` (hasattr — mirror.py:381 sets it at ctor so it's always
true), which the *backward* window DOES include (train.py:573–575, c10:247–249 — the B13
fix). With `gradient_checkpointing=True` (production), each `autograd.grad` pass inside
`balancer.grad_geometry` recomputes the block and re-executes mirror's `if self.training
and override < 0.1 and not _ggeo_freeze:` region (mirror.py:473–511): it **flushes
`_alpha_pending` (lerp 0.01 + novelty push onto `alpha_diag.data`) and advances
`_residual_var_ema` in place** (mirror.py:483–489) as a side effect of a pure diagnostic.
Probe P4, real code, 2-layer model, ckpt ON, exact flag patterns: as-shipped —
`Δalpha_diag during ggeo = 2.553e-01`, `Δ_residual_var_ema = 3.340e-01`;
with `_alpha_pending` added to the ggeo selector — **both exactly 0.000e+00**
(during the *backward* window both are 0.000 either way — B13 holds there). At
`log_interval=55` (production notebook) the alpha self-regulation law runs at ~2× rate
on 1 in 55 steps (+1.8% duty — small, but it is precisely the undocumented
control-law-rate change B13 exists to prevent, re-introduced through the diagnostic
door); ggeo also re-pends the same target, so the *real* application semantics on those
steps are flush@diagnostic + flush@next-forward. One-line fix, both copies; needs a lock
(count lerp applications across a grad_geometry window).

**F5-09 [VERIFIED-REPRO] "Reopen cognitive gate" fires on EVERY resume in both copies
and clobbers two restored tensors.** `cfg.mlp_gate_b_init` **defaults to 0.25**
(config.py:255) — neither cell 4 nor the train.py CLI ever sets it to 0 — so the
`if getattr(cfg,'mlp_gate_b_init',0.0) > 0` blocks (c8:100–105, train.py:294–301) run on
every resume: `layer.mlp.mlp_gate_b.data.fill_(0.25)` (a trained **Parameter**) and
`layer.mirror.hybrid_gate.log_tau.data.fill_(log(tau_val)=0)` (a **persistent buffer**
that the gate self-tunes, adaptive_gate.py:103/105). Both values arrive from best.pt
trained, then get pinned back to their init within the same cell — a silent per-restart
lobotomy of exactly the MLP-gate wake-up state this hack was invented to fix once.
Observed live: `reopened cognitive gate: mlp_gate_b -> 0.25, hybrid_gate tau -> 1.000`
printed during the P6 real-cell-8 exec against a same-arch envelope. The block predates
B8's envelope discipline; it should fire only when the ckpt *lacks* those keys (legacy),
or behind an explicit flag.

**F5-10 [STRONG-READ] The notebook Ctrl+C save overwrites best.pt with an UNGATED
state — "best = last CLEAN" (B3/D6+) is true only between evals.** c10:461–479:
KeyboardInterrupt writes the full payload to **best.pt itself** with no `val < best`
check and no `reasoning_enabled_step` key (01 §2.6 still true post-B14 — compare c10:429
vs c10:465–477). Scenario that breaks the doctrine: watchdog ALARM:WARN (strike 1/2)
→ training continues on divergent weights → user Ctrl+C's → best.pt = divergent state
with `best_val_loss` still holding the old (better) number, so the NEXT session's
improvement bar is measured against a best that its own envelope no longer represents.
train.py's answer (train.py:737–740: "Ctrl+C — keeping last best.pt, no checkpoint")
preserves the doctrine but **throws away up to `eval_interval−1` steps** and makes the
`interrupt_step_*.pt` glob in its own auto-resume (train.py:257) dead — nothing in the
repo ever writes that filename (grep: exactly one hit, the glob). Recommended shape
(§5): Ctrl+C writes `interrupt_step_{step}.pt` in *both* copies; best.pt stays val-gated.

**F5-11 [VERIFIED-REPRO] Save-step replay: resume repeats one trained step and drifts
all step-coupled counters by +1.** Both loops do `start_step = ckpt['step']`
(train.py:343, c8:169) and `for step in range(start_step, …)` (train.py:380, c10:50),
but the save happens *after* that step's optimizer+scheduler+reasoning increment
(save site is inside the eval block at the end of the step body). Battery: the replayed
"step 10" consumed the batch that baseline trained at step 11 (cursor 65→130 in the
tap), giving naive-alignment ΔCE 0.549 (= replay artifact, not RNG): one extra
`optimizer.step()` + `scheduler._step` + `reasoning_enabled_step` per restart, and a
permanent −1 offset between loop step number and data position after each death.
At 500 restarts/run the LR clock alone is +500 vs the curriculum/`_step`-gated
schedules. Cheap fix: save `'step': step + 1` or resume with `start_step+1`.
Related [STRONG-READ]: there is **no final save** anywhere (periodic writes disabled —
train.py:736 comment; notebook comment c10:455–459 promises a "step-0 seed written in
setup" — grep: no such write exists in either copy): a death before the *first*
improving eval loses **everything** (up to step 1045 in production), and the last
non-improving window of a long run is never checkpointed at all.

### MEDIUM

**F5-12 [VERIFIED-REPRO] Deferred alpha pending is lost at every save/resume.**
`_alpha_pending` (mirror.py:381/481–511) is a plain attr; probe P2 confirms it is
**non-None at the save instant** (the last training forward's target, survives the
eval because the write region requires `self.training`). Envelope has no key for it
(X3: `'…_alpha_pending' not in ckpt['model']`), `__attrs__` doesn't include it, and the
first forward of the new session finds None → one `lerp_(target,0.01) + novelty push`
is silently dropped per restart. The *target* tensor must be CPU-copied at save or the
queue must be flushed before `state_dict()` (flush changes alpha_diag — do it BEFORE
the save line, both copies; it is exactly one application the old session already
"decided").

**F5-13 [VERIFIED-REPRO + STRONG-READ] Policy/session counters orphaned; the balancer
key is a lie.** `alarm_strikes` (train.py:231 / c10:23) is never saved → D6+ forgets
"already warned" across a death: each session gets a fresh free WARN (04's orphan row —
re-confirmed post-B14, envelope census has no key). `balancer` envelope census:
`{'ema_ce': None, 'ema_A': None, 'ema_aux': {}, 'align': True}` — the align path never
updates them (03 F3-25), so the key ships 4 nulls forever; drop or populate. `model.
_phase_ratio_ema/_std` (train.py:202–203, orphaned per 04) are reset each boot —
measured NOT to contribute at 10 steps (F5-07 attribution), they matter only for
long-horizon mir_s hysteresis; acceptable but should be named in the table below.
`tokens_seen`/`t0` are cosmetic. `clipper._p_scale` is rebuilt by `attach()` (fine).

**F5-14 [VERIFIED-REPRO] Cross-version tolerance of B6–B14 envelope changes — mostly
good, three soft failures.** Legacy battery (P5): hand-corrupted envelope —
4-wide detector stats, `code_fp`/`depth_state`/`reasoning_enabled_step` removed,
`param_names` removed, `orig_lrs` truncated, 3 buffer keys deleted, 1 junk key added —
resumed through the real train.py path (with F5-01/02 shims) **without a crash** and
completed its steps. Item by item:
(a) *4-wide stats FIXED (B13)* — `load_state_dict` pads `f+3 → [:7]`
(training_control.py:237–239); probe: 4→7 ✓, **but 1/2/3-wide → 4/5/6-wide**, still
short of the 7-tuple unpack at training_control.py:258 (no observed legacy at ≤3, edge
only). (b) *position-based optimizer FIXED (B12)* — by-name path exists but is dead in
train.py per F5-03; in the notebook it works. Legacy without `param_names` → documented
"fresh Adam" warning in both. (c) *`_usefulness_temp` persistent False→True (F4-04)* —
new saves carry it (census confirms); a pre-B13 envelope lacks it → strict=False filter
keeps **ctor 2.0** (mirror.py:380) — probe: value 2.0 after legacy load, no crash; i.e.
the F4-04 "stuck at 2.0 ≠ post-warmup 0.51" symptom *persists for legacy envelopes*
(scheduler only writes it during warmup/blend, lr_scheduler.py:191–200 — never again).
Same benign-init logic for `_residual_var_ema` (probe: 0.1 ctor). (d) *`orig_lrs`
wrong count* → rejected, fresh snapshot kept (lr_scheduler.py:300–306; P5 run with
1-elem list and 17 groups: clean); **same-count reordering still silently accepted**
(F4-09 family open — and notebook↔train.py group counts differ by construction,
llrd 1.0 vs 0.9, cross-copy resume always trips the reject branch → uses the NEW
construction's lrs, which is the safe direction). (e) *NEW→OLD downcast*: no forward
compat layer exists; old readers ignore unknown keys (strict=False) but e.g. a pre-B14
cell 9 has no `depth.put_state` and pre-B8 cell 8 no `verify_identity_resume` — state
officially **'none' supported**; only the B8 *aborts* protect the boundary.

**F5-15 [VERIFIED-REPRO] migrate.py eats three CURRENT keys on every resume.**
`migrate_state_dict` (core/migrate.py:95–100) unconditionally deletes
`logit_cache.attention.{k_proj_l,v_proj_l}.weight` and `logit_cache.logit_to_hidden.weight`
— written for the *old V×D* shapes, but the current architecture re-registers the same
names as K×D params, so the probe (X1, fresh same-arch envelope): `changed-count: 3`,
exactly those three keys DROPPED → resume silently re-inits them from ctor (their saved
values discarded) and prints the misleading "MIGRATED 3 keys (W_out +K, …)" banner
(observed in every battery run, including the notebook cell-8 exec). Today harmless —
02a F2A-10 proved these three receive **zero gradient** in the training loop — so the
envelope carries dead weight that the migration then deletes, a self-cancelling pair;
the moment the logit-cache inference half is wired (02b Appendix B: "the cache shares
the identical codebook object — but only its inference half touches it, and that path
currently never runs"), this rule becomes a silent trained-state eraser. Shape-gate the
drop (V×D only) and drop the dead trio from the save instead.

**F5-16 [STRONG-READ] Watchdog CE baseline: the two copies resume it differently.**
train.py:346 loads the detector state verbatim (CE baseline survives); notebook c9:103
pops `ce` post-load ("a resumed session is a regime change" — deliberate). With
B13's `arm_ce` now first-val-only (training_control.py:335–339), the train.py variant
keeps a warm CE watch across restarts while the notebook always re-bootstraps ~100
disarmed samples — one envelope, two sensor policies (the same species 04 F4-02
killed inside arm_ce). Pick one (the notebook's pop is the safer default given
allocator/regime changes) and make both call sites identical.

**F5-17 [VERIFIED-REPRO] FORCE_FRESH hygiene: one stale symbol survives the switch; no
global seed anywhere.** Exec of the REAL cell-8 source (P6): pass 1 (FORCE_FRESH=False)
defines `_resume_depth_state` (c8:176); flip FORCE_FRESH=True and re-run cell 8 — every
other `_resume_*` is reset at the top of the cell (c8:11–14) but **`_resume_depth_state`
is assigned only inside the resume branch** and survives in the kernel globals
(identity-confirmed); cell 9's guard `'_resume_depth_state' in dir()` (c9:100) is an
*existence* test → a "fresh start" feeds the PREVIOUS session's depth dict into the new
DepthController: probe shows `put_state({'active':12,'val_ema':1.70,…,'last_depth_step':9120})`
on a ctor-fresh controller — wrong depth AND a warm plateau integrator on a step-0 model
(on the 2-layer probe model `active` clamps to 2 so only the scalars visibly move; in
production `active` is the damage). Fix: `else: _resume_depth_state = None` at c8:185–187
(or gate c9:100 on `not starting_fresh`). Also: neither copy ever calls
`torch.manual_seed` (grep) — fresh boots are NOT reproducible-by-construction; only the
envelope path is. `verify_identity_resume`'s abort message instructs "write a migration
in `scripts/migrate.py`" — **that file does not exist** (Test-Path False; the migration
layer is `core/migrate.py`). train.py's own `hasattr(model,'_loss_lr_factor')` block
(train.py:302–307) is dead code (the factor lives on the scheduler; grep stack.py: 0 hits).

**F5-18 [VERIFIED-REPRO] CUDA RNG still outside the envelope (01 F-21 open).**
train.py's envelope census keys (18): no CUDA generator; resume restores the global CPU
gen (`torch.get_rng_state` train.py:731/361) and the doc-shuffle gen (`data_rng`,
RNG-stream R11-exact), so per-step VSA write-noise `randn_like` (block.py:401–403, device
default) and every other CUDA-side draw diverge on the only platform where training
actually runs. CPU battery confirms the CPU-side half of resume is otherwise bit-tight:
with F5-07's state flushed, divergence is exactly replay+streaming-cold, no RNG noise
floor. [VERIFIED-REPRO: envelope key list probe]

**F5-19 [STRONG-READ] cfg rides in the envelope but is never consulted on resume; the
OOM governor state is session-only.** Both copies build cfg from scratch (cell 4 / CLI)
and ignore `ckpt['cfg']` — 02a's flagged fact, consequences here: the M16 memgov/OOM
seq_len shrink (c10:108–113, 279–282) and `orig_seq_len` (c9:115) are not persisted, so
a machine that died OOM-shrunk at seq=128 restarts at cell-4's 512 and re-triggers the
OOM cascade; `cfg.gradalign_weight` is hard-set in c10:26 but left at 0 in train.py
(01 M5 divergence) — the saved cfg would have *detected* this mismatch if anything
compared it. Minimum: `assert`-log `ckpt['cfg']` vs live cfg on resume (D/K/layers/vocab/
code_dim/bind_K are anyway policed by shapes + code_fp; seq/governor fields are not).

**F5-20 [STRONG-READ] GradScaler state not persisted.** train.py:241 builds a fresh
`GradScaler(enabled=use_amp)` every boot (notebook cell 5 similar); with AMP on (B10
path), every resume restarts the loss-scale search (transient step-skips). Include
`scaler.state_dict()` in the envelope when `use_amp` — one key.

### LOW

**F5-21 [VERIFIED-REPRO] Save-site mechanics are sound where they exist.**
train.py's `_save_checkpoint_safely` (train.py:19–24): torch.save → `path+'.tmp'`, then
`shutil.move` onto the target — a torn tmp never shadows best.pt; on Windows the rename
fallback copy2+remove is slower but correct; on Drive/FUSE (Colab) rename is copy-based,
and an interrupted *second* save leaves a stale `.tmp` that nothing reads (globs match
`best.pt`/`step_*/interrupt_step_*` only) — no cleanup, cosmetic. The notebook inlines
the identical tmp+move twice (c10:421–438, 463–478) — equivalent. Cell-8 corruption
fallback (c8:44–46) catches `RuntimeError` — a truncated best.pt raises exactly that
(P6b probe: `RuntimeError` on a 1/3-truncated file) → numbered-fallback path OK.

**F5-22 [STRONG-READ] Contradiction/forensic hazards at save sites.** train.py:459
stale comment "CE explosion -> rollback + fresh Adam + LR rewind" survives D6 (01 §3
flagged :444; the file has since shifted — same lies, new line numbers); train.py:342
print contradicts :324 print within the same resume; the c10:455–459 "step-0 seed"
policy comment describes a write that doesn't exist (F5-11).

**F5-23 [VERIFIED-REPRO] Inert-but-persistent counters ride every envelope.**
`bind._step_count` (persistent, incremented under eval, zero readers post-B3 — P2 census
shows it among the dirty-22), `mirror._fwd_count` (training-gated, read only through
`_usefulness_temp=0` which never happens after warmup blend), `_pm_step` (legacy path).
02b's proposal — demote to `persistent=False` at the next state_dict format touch —
stands; today they are schema-width hazards (F4-03 class) with zero function.

**F5-24 [STRONG-READ] No envelope schema/version key.** 04's handoff asked for a width/
schema tag on detector stats: normalization shipped (F5-14a), but the envelope itself is
still untagged (18 bare keys, §3 census) — every future tolerance check has to infer
version from key presence. One `'schema': N` line fixes this for the next fresh run.

---

## 2. Envelope completeness table (the canonical one)

Save sites enumerated (mission 1): **train.py has exactly ONE** — eval-improve
(train.py:716–732, `_save_checkpoint_safely`); periodic step_*.pt disabled (train.py:736);
KeyboardInterrupt saves NOTHING (train.py:737–740). **Notebook has TWO** — eval-improve
(c10:423–438) and KeyboardInterrupt (c10:465–478, the 17-key subset missing
`reasoning_enabled_step`). Keys written by the eval-improve site (probe census, 18):

```
step, model, code_fp, optimizer, param_names, scheduler, best_val_loss, cfg,
reasoning_enabled_step, active_depth, recover_count, detector, depth_state,
balancer, stream_idx, offset, rng, data_rng
```

| # | Mutable state touched by a step | in save? | restored correctly? | shape-safe across geometry (B8 guard)? | verdict / gap |
|---|---|---|---|---|---|
| 1 | model Parameters (all: trunk, mirror, `alpha_diag`, `b_i/b_d`, τ-ladder `log_scale`, `log_skip_alpha`, `mlp_gate_b`, `hybrid_gate` params, W_out, reasoning_gate, memory-bank projections, `log_tau`…) | Y (`model`) | Y — migrate→shape-filter→strict=False→`verify_identity_resume` | YES: size-mismatch skipped, identity-path mismatch FATAL (B8; train.py path unreachable = F5-01) | OK (but see rows 3, 33; `mlp_gate_b`/`log_tau` re-clobbered = F5-09) |
| 2 | persistent buffers — census: 73 register sites w/o `persistent=False`, incl. NEW B7 `memory_bank.{l1,l2}.keys/vals/_write_idx/buf/buf_age/slot_*` (memory_bank.py:119–229), B13 `mirror._usefulness_temp` (mirror.py:380), `_residual_var_ema`, `_gate_ema`, `_private_mem`, `_pm_step`, `_delta_var`, `_fwd_count`, `_concept_sim_ema`, `_behavior_div_ema`, UCL `concept_*`/`_step`/`_n_*` (concept_layer.py:76–105), `bridge.bridge_stream`, `_tgt_mean`, maturation 6 (maturation.py:79–84), `block._tau_s`, τ-config caches, `bind._step_count` | Y | Y for train.py; **Y-but-eval-contaminated for notebook** (F5-06: 22 keys dirty at save instant) | YES (same filter; filter skips mismatches silently — L2 slot change re-inits slots, acceptable) | GAP F5-06 (ordering), F5-23 (inert counters), F5-14c (legacy lacks `_usefulness_temp`) |
| 3 | **NON-persistent buffers (39 sites)** — `embed.codes`(×6 rebuilds, code_fp-guarded), `_mix_scale`/`_sig_mean`(centering EMAs, B9), block `_mlp_{now,base}_ema/_cnt`, mirror `_signal/_ig/_ctr/_grad_norm_ema`, `_ls_var_run`, `_div_run(_rec)`, `_pm_coh`, `_hp_grad`, `_last_magnitude/_gates/_h_pool`, `_cached_*_buf`, `_pos_id_buf`, `_alpha_override`, `_trust/_prev_trust/_meta_private_mem`, `_damp_tau`, `bridge_loss_ema/_init/inj_ratio`, `shifts`, `_circ_*_idx`, beam/traj buffers, `codes_t`, `_reasoning_gates`, `_expl_ema`, `_cached_birth_gate` | N (by definition) | ctor defaults; `_pm_step` legacy special-case c8:109–118 | n/a | **OK only while every one is derived or re-bootstrapping.** Watch-list: `bridge_loss_ema/inj_ratio` (readiness — feeds eva trust + bridge gate → cold re-learn each restart); `block._mlp_*` (re-warm by design, reset_cache also scrubs them train.py:314/stack.py:998); `_alpha_override`=0 == post-warmup rest ✓ and warmup-mid resumes get it rewritten by scheduler per `_step` ✓ |
| 4 | optimizer `state` (exp_avg/exp_avg_sq/steps; EVAAdamW per-param extras) | Y | **train.py: NEVER (F5-03 key-location bug → fresh Adam every resume)**; notebook: Y by-name; legacy w/o names → fresh Adam (documented) | Y: W_out+K handled two ways — train.py PADS (zeros / exp_avg_sq=1), notebook TRUNCATES (c8:153 `v[:p.shape[0]]` yields old-shaped tensor → `optimizer.load_state_dict` size-error escapes its try (raise happens at :46 call in c9, outside any try) — notebook dialect broken for growing W_out | GAP F5-03 [train] + dialect split [03's two-dialects handoff — confirmed] |
| 5 | optimizer group meta (`lr` positional min-len train.py:144–146; EVA role/trust/cap/layer_idxs fresh) | Y(partial) | lr: restored then OVERWRITTEN by `scheduler.step()` every group each step (lr_scheduler.py:262–264) → moot; meta: fresh ctor = authoritative (positional dict.update NOT done — `_restore_optimizer` only copies 'lr' into fresh groups) | len-guarded both | OK (by accident) |
| 6 | EVAAdamW `_trust` dict (eva_optim.py:181/212) | N | lost; refilled by `set_trust` before step — **train.py yes (643–651), notebook NO** (c10 never calls it — F2B-07 open, 03's) | n/a | DIVERGE axis already owned by 03 |
| 7 | scheduler: `_step, _last_log, _tau_{var,mag,1malpha,gate_var}, _orig_lrs, _best_val_loss, _loss_lr_factor, _val_ema, _val_improving` (+ `_ls_fast/_slow` only if enabled-and-bootstrapped) | Y (probe key census: `step,last_log,type,tau_×4,orig_lrs,best_val_loss,loss_lr_factor,val_ema,val_improving`) | Y; mid-warmup resume continues temp/override schedules exactly (driven by `_step`, lr_scheduler.py:178–200); `_ls_*` = None if saved pre-blend-end → re-bootstrap (B14 length guard :90) | orig_lrs: count-guarded (F5-14d); reorder-acceptance open (F4-09) | OK + note: `_lr_damp_steps` NOT saved (INERT per 04, reset by hasattr guard B12 ✓); `_ls_mult` derived ✓ |
| 8 | watchdog: `recover_count, ce_armed, cooldown, viol, stats(7-wide), last_viol_name`; `_cur_step`/`_warm_frozen` NOT saved | Y (Y for the first six) | Y; `_cur_step` re-derived from `scheduler._step` per check (B14 train.py:484/c10:178) ✓; `_warm_frozen=True` default correct + release zeroes PH (04 T1) ✓ | 4-wide→7 padding ✓ (F5-14a; ≤3-wide edge unhandled) | OK except CE-pop asymmetry F5-16 |
| 9 | depth: `active, _val_ema, _val_var, _prev_val, _last_depth_step` + per-param `requires_grad` mask | Y (B14: `active_depth` + `depth_state`) | Y — `put_state→set_depth` rewrites the mask (adaptation.py:132–140); train.py:334–340 legacy heuristic fallback for pre-B14 ckpts | n/a | OK; notebook fresh-path stale leak **F5-17** |
| 10 | balancer `ema_ce/ema_A/ema_aux/align` | Y | Y | n/a | **4 nulls — align mode never writes them (F5-13)**; `last_cos` NOT saved (orphan, no reader) |
| 11 | two-strike policy: `alarm_strikes` | **N** | default 0 each session | n/a | ORPHAN (D6+ per-session reset) |
| 12 | data cursor `stream_idx, offset` (+ stale-holdout guards train.py:370–371 / c10:9–12) | Y | Y | n/a | OK |
| 13 | RNG: global CPU (`rng`), doc-shuffle generator (`data_rng`) | Y | Y (`.cpu()` guards map_location) | n/a | OK — but CUDA generators ABSENT (F5-18, 01 F-21) |
| 14 | GradScaler | **N** | fresh scale each boot | n/a | F5-20 (AMP only) |
| 15 | loop streaming state: `state` (per-layer VSA), `gs`, `intent_state` | **N** | None at resume → cold doc mid-document | n/a | **F5-07 — the single largest measured divergence source (0.44 nat first step)** |
| 16 | one-step attrs: `model._last_salience`, `_intent_stream`, `_last_bus`, `_last_intent_state` | **N** (snapshot `__attrs__` covers the last two FOR EVAL ONLY — stack.py:1037, value-restored verified by probe) | None at resume | n/a | F5-07 cluster; `_intent_stream` also survives document rotation (01 F-04) — dual gap |
| 17 | `mirror._alpha_pending` (deferred control write + novelty push) | **N** | None → one application lost | n/a | F5-12 |
| 18 | mirror plain caches: `_cached_hp`, `_cached_pred_error_norm`, `_traj_state`, `_pi_v`, layer `_pi_v` | N | None at boot; **eval no longer writes them (B14 F4-11 re-verified — probe `p_attrs`: `plain attrs written by EVAL: ['_last_bus','_intent_stream']` only)** | n/a | OK (streaming-only per B10; one-step semantics) |
| 19 | maturation: `gate/readiness/pen_ema/pen_init/tau_norm/_lf` (all persistent) + published B11 combined gate | Y | Y; time-part recomputed from restored `step` each forward | n/a | OK — mission's "ema counters, readiness, _t?" — no hidden `_t`; step-derived by contract |
| 20 | memory banks L1/L2 content + `_write_idx` (incl. B7 keys/vals) | Y | Y; **training-only writes hold: eval produced ZERO bank dirt in the P2 census** | slot-count changes → filtered re-init | OK |
| 21 | UCL concepts (content + counters) | Y | train.py Y-clean; **notebook Y-eval-contaminated** (F5-06 dirty list) | n/a | GAP F5-06 |
| 22 | reasoning: `reasoning_enabled_step` | Y (eval save) / **N (notebook KB save)** | Y / resets 0 on Ctrl+C-resume (01 §2.6 still true); +1 replay drift per F5-11 | n/a | GAP |
| 23 | `_reasoning_buffer/_count` (plain, stack.py:907–910) | N | None | n/a | moot (chain dead in adaptive config — 03/02b F2B-08) |
| 24 | codebook identity: `embed.codes` rebuild (non-persistent) | Y indirectly: `code_fp` (B8) | `verify_identity_resume` fatal-compare (train.py:284 unreachable / c8:96) | the guard IS the geometry check | OK post-B8 (for the notebook) |
| 25 | step counter `step` itself | Y | +1 replay bias (F5-11) | n/a | DIVERGE small-but-systematic |
| 26 | `cfg` | Y | **ignored on resume (both)** — live construction wins; OOM-shrunk seq_len / memgov `gradient_checkpointing` flip / gradalign flag all session-only | shape fields policed by F5-01/B8 | F5-19 |
| 27 | `model._phase_ratio_ema/_std` (train.py phase scaler) | N | zero/one reset each boot | n/a | ORPHAN (04 row; measured non-driver ≤10 steps, F5-07 attribution) |
| 28 | `logit_cache.cache` dict (window entries + scheduled-sampling state) | N | empty on resume — train.py explicit `reset_cache()` (train.py:313–315), notebook fresh dict | n/a | OK (decision #3: cache is document-scoped; but see F5-15 re the three cache *params*) |
| 29 | `stream` memmaps (file list order) | N (paths implicit) | sorted(glob) same on same host; corpus change silently shifts `stream_idx` semantics | n/a | SPECULATIVE risk: cursor is meaningful only w.r.t. identical file set |
| 30 | watchdog `_min_samples`, margins/floors, clipper c, scheduler warmup/blend constants | N | from code/ctor | n/a | OK by doctrine (code is the schema; cfg row 26 covers the rest) |

Score: **5 classes entirely outside the envelope** (15, 16, 17, 11, 14 — plus CUDA RNG in
13), **2 eval/ordering contaminations in the production copy** (2, 21), **1 dead restore**
(4/train.py), **2 systematic +1 drifts** (22, 25 via F5-11).

---

## 3. Resume-divergence battery results

All CPU, fp32, 2-layer/D=128 mini (features ON: banks, UCL, cache, bridge, spiral,
private-mem, maturation forced-warm `matur_T0=2`), 4×20 000-token synthetic streams
(`token_stream_G*_eos.bin`), scripted eval values (real evaluate() monkey-patched so the
save site fires deterministically), `FailureDetector.check` tapped for the (step, LR-clock,
CE) trace, `torch.manual_seed(1234)` in-process (train.py itself seeds nothing). The three
findings-level shims (F5-01, F5-02, F5-04/F5-05) were applied for the battery and logged.

| run | save point | steps after | result |
|---|---|---|---|
| A `base` | (saves at 5 & 10 & 15) | 20 steps, uninterrupted | CE 9.29→7.29 monotone-ish; reference trace |
| B `stop10` | best.pt @ step 10 (post-warmup-blend, armed detector, bank `_write_idx`>0, `depth_state` written, `_alpha_pending` non-None) | 11 steps | envelope + state census captured (§2, F5-11) |
| C `resume10` (fresh process, fresh boot, resume, run to 20) | ← B's best.pt | 10 steps | **cursor-aligned max \|ΔCE\| = 0.3666**; replay step (step 10 re-run at step-11's cursor) alone: 8.457 vs baseline's 7.908 (**0.549**, structural — F5-11, not noise) |
| C′ `resume10p` — identical + baseline `_phase_ratio_ema/_std` seeded into first step | | | max \|ΔCE\| = **0.3666 (unchanged to 1e-5)** → phase-EMA orphan NOT the driver |
| D `stop5`→`resume5` (save MID-WARMUP: `_alpha_override`>0, `_ls_mult` None, temp schedule mid-ramp) | best.pt @ 5 | 15 steps | cursor-aligned max \|ΔCE\| = **0.4603** over 9 events; schedules continue correctly from restored `_step` (no LR-path divergence detected in the tap) |
| E `legacy` (P5 battery: 4-wide stats, no code_fp/depth_state/reasoning key, no param_names, orig_lrs truncated, 3 buffer keys stripped, junk added) | resume | 5 steps | **no crash**; `[B8] legacy checkpoint without code_fp — relying on shape checks only`; `WARNING: no param_names → fresh Adam`; `_usefulness_temp` = ctor **2.0**, `_residual_var_ema` = 0.1 (F5-14c); scheduler restored step=15 (orig_lrs rejected) |

**Attribution of the 0.367** (probe (c), `p_attr.py`): identical 12-step replay of the
same loop with the SAME weights/cursor/RNG stream, the single injected difference =
`state/gs` flushed at step 10: **|ΔCE| first step 0.4439, second 0.0620** — same order as
the battery → **F5-07 (streaming state) + F5-11 (replay) explain the whole divergence**;
the remaining candidates (pending alpha, watchdog/balancer restores, cursor, RNG,
schedules) contribute below that noise floor on this window, i.e. the envelope machinery
B12–B14 built works — *inside the notebook's restore semantics*; train.py's is unreachable
(F5-01..05) and its momentum restore is a no-op (F5-03).

**ggeo control (P4, ckpt=True):** as-shipped `Δalpha_diag = 2.55e-1`,
`Δ_residual_var_ema = 3.34e-1` per diagnostic window; fixed flag selector: `0.0 / 0.0`
exactly — F5-08 confirmed and its fix confirmed sufficient.

**Notebook cell exec probes (P6/P6b):** cell-8 source run against a real envelope →
pass-1 NameError `verify_identity_resume` (F5-02) before symbols are supplied, then a
clean resume whose prints captured F5-09 (gate clobber banner), F5-15 (MIGRATED 3);
FORCE_FRESH re-run kept `_resume_depth_state` alive (identity check) → F5-17;
truncated torch.load → RuntimeError → cell-8 fallback adequate (F5-21).

**Suite after all probes: `python -m pytest tests -q` → 314 passed in 58.59 s** (repo
untouched; zero temp artifacts inside the tree).

---

## 4. Handoff answers from agents 1-4 (addressed)

**← 01 §2 contract + §5 (→ agent 5 items F-03, F-06, F-17, F-12, §2.7):**
- §2.6 envelope inventory re-derived post-B7–B14: `code_fp` and `depth_state` added
  (train.py:718/728, c10:424/433); `param_names` still top-level in both (c10:426);
  KeyboardInterrupt still drops `reasoning_enabled_step` (F5-10); still no CUDA RNG
  (F5-18); `_last_salience/_intent_stream/_pi_v/orig_seq_len` still outside (F5-07/19).
- **F-03 (eval writes into training memory)** — L2 bank fixed by B7: probe confirms zero
  eval dirt on bank buffers. **But the same species is alive in UCL `concept_keys/vals`
  and `bridge.bridge_stream`, and the notebook's save-before-restore (F5-06) ships it
  into best.pt.** This is the concrete answer to "does the saved state include
  eval-contaminated buffers?": YES for the notebook copy (22 keys), NO for train.py.
- **F-06** closed (B12 NaN + report-gate; battery eval-NaN branch not re-triggered — the
  NaN path skips `report_val_loss/arm_ce` but `val_loss < best` at train.py:713 is
  NaN-safe, verified by reading; the notebook's `_val_ok` gate equivalent).
- **F-17/F-12** — persistence angle: val-region geometry depends on `cfg.seq_len` which
  F5-19 shows is session-only → after an OOM-shrink death, val batches (and the
  row-boundary CE term) change scale silently across resume. Doctrine-level incomparability
  confirmed but owned by 01.
- **§2.7** honored by both copies; `arm_ce` cadence is now first-val-only (B13) —
  re-verified at the call sites.
- **M12 "full restart state"** — final answer after the exhaustive table: false; see the
  five entirely-outside classes in §2 (rows 11, 14, 15, 16, 17 + CUDA RNG row 13).

**← 02a §4 (→ agent 5: fingerprint-in-envelope, critical-shape abort, cfg-not-
consulted, checkponts/ forensics):**
- `code_fp` shipped both copies, verified on load (fatal mismatch + legacy warn path) —
  the mechanism works in the notebook; in train.py the load path can't be reached
  (F5-01). `_CODES_CACHE` clearing inside the determinism test: out of envelope scope,
  not done.
- Critical-shape abort: implemented (`verify_identity_resume`), **but its train.py call
  site is the crash** (F5-01) — B8 hardened the function and simultaneously broke the
  caller.
- cfg-not-reconsulted: **confirmed and extended** (F5-19: OOM governor + gradalign flag
  are the concrete casualties).
- `checkponts/` on-disk best.pt: only `anomaly_track.json` present on this host — the
  2.4 GB legacy witness never re-landed; on-disk legacy resume remains
  [NOT-VERIFIABLE-HERE]; the synthetic legacy battery (P5) substitutes for it.

**← 02b §5 (→ agent 5: snapshot set additions, rotation order, by-name optimizer,
persistent=False candidates, B10 re-measures):**
- `block._traj_state`, `mirror._cached_hp`, `_cached_pred_error_norm`, `_pi_v`: probe
  `p_attrs` shows **eval writes none of them** (post-B14) → no snapshot-extension needed
  for eval-isolation; `__attrs__` already carries `_last_bus/_intent_stream`
  value-exactly (stack.py:1037–1043 verified). For DEATH-persistence the honest candidates
  are `_alpha_pending` (row 17) and the loop state trio (row 15) — not the caches
  (streaming-only per B10).
- Rotation order: `reset_cache()` clears `_traj_state` — verified irrelevant for training
  (windowed path never reads it; B10 `set_stream_mode` contract).
- Optimizer by-name in train.py: **"still unwired" is now stronger — wired but dead**
  (F5-03, key-location).
- `_step_count/_pm_step/_fwd_count` → `persistent=False` candidates: confirmed (F5-23),
  plus the F4-03-style width hazard they create; ship with the B15 format touch.
- B10/AMP re-measures: not this host (CPU only) — flagged for the A100 restart list.

**← 03 §5 (→ agent 5: F3-08 + two `_restore_optimizer` dialects, F3-07/F3-27 scheduler
schema, F3-12/F3-22 snapshot set, F3-25 balancer nulls):**
- F3-08: by-name wired at train.py:323 but **never fires** (F5-03); notebook by-name fires.
  Dialect split now asymmetric in *correctness*: train.py pads W_out+K (right), notebook
  truncates → raises out of its try (F5-04 row #4 in table §2) — merge to one module-level
  implementation (recommend `core/`).
- F3-07 `_lr_damp_steps`: legacy hasattr guard verified live (P5, no AttributeError).
- F3-27 `orig_lrs`: positional + count-only; mismatch → keep fresh (safe direction;
  notebook↔train.py group-count split guarantees rejection cross-copy); same-count
  reorder acceptance open (F4-09), no lock — needs a test that reorders two equal-count
  differently-scaled groups and asserts rejection or by-name remap.
- F3-12 `_step_count` inflation: **bind counter fixed in both windows; the MIRROR control
  writes are NOT fixed in the ggeo window** — that's F5-08, the live remnant of exactly
  this family. `_pi_v` outside snapshot: confirmed, reseeded once per session, benign.
- F3-25: confirmed verbatim from the envelope (balancer = 4 nulls, F5-13).

**← 04 §5 (explicit handoffs to agent 5):**
- **F4-03 schema tag**: stats normalization shipped (4→7, verified) — but the *envelope*
  still has no `type/schema` key (detector has none; scheduler carries `'type':
  'MirrorLRScheduler'` only). → F5-24, one-line B15 item.
- **F4-06 depth doctrine**: `depth_state` in envelope + `put_state` in BOTH copies
  (train.py:347, c9:100) — round-trip verified on real saves. The remaining leak is the
  notebook FORCE_FRESH staleness (F5-17), which re-opens the exact "instant unlock" B14
  closed, but across a *fresh start* instead of a resume.
- **F4-04/F4-11/F4-13 orphan-table items**: `_usefulness_temp` now persistent (new saves
  carry; legacy still mis-inits 2.0 — F5-14c); `_alpha_override` persistent=False is the
  right call (resting 0, scheduler-owned — table row 3); `_cached_hp/
  _cached_pred_error_norm` eval-write hole **closed and probe-verified** (p_attrs);
  `model.reset_document_state()`: not shipped; from the persistence side the needed
  set is *smaller* than 01 proposed for eval (B14 already covers it) but *larger* for
  death — see §5 item 2 for the exact list to add to the envelope instead.
- **"snapshot doctrine incl. __attrs__ extension candidates"**: candidates are (a)
  `_alpha_pending` (add to `__attrs__` AND the envelope, or flush-before-save —
  flush-before-save is semantically cleaner: the old session already decided),
  (b) nothing else eval-relevant remains (probe-proven), (c) for resume, extend the
  envelope with loop state (F5-07), not the snapshot.
- F4-05/F4-10 closure: no persistence residue found — the B14 clocks agree inside the
  envelope (`_step` restored; `check` keys off it in both copies).
- "best = last CLEAN" doctrine audit: holds for the two eval-improve saves, **breaks for
  the notebook Ctrl+C overwrite** (F5-10) and is *notional* before the first improving
  eval (no seed, F5-11). Should the divergent state EVER become best.pt? No — and today
  the only door through which it can is exactly that ungated handler.

**Not-verified-here list (agent 5's own):** CUDA/bf16 numerics of resume (F5-18 makes
bit-exactness impossible there anyway), Drive-FUSE rename latency under real 2.4 GB
envelopes, real 24-layer state-sizes for the §5 state-in-envelope recommendation, the
on-disk legacy best.pt (absent).

---

## 5. Recommendations for the restart (what B15 must ship WITH the first fresh run)

Ordered by what has to be *simultaneous* with the run start (envelope schema decisions)
vs. what can follow:

1. **Revive the runner first** (blocking, otherwise none of this is exercised): the
   five train.py killers — `verify_identity_resume` call (F5-01), `args` leak (F5-02),
   `_restore_optimizer` top-level `param_names` lookup (F5-03), `generate_report` args
   (F5-04, or drop the report call into a try — it must never kill a save), `if gates:`
   (F5-05). Then add **`tests/test_b15_persistence.py`**: a mini-corpus, ~8-step,
   real-`train()` save + fresh-call resume + trace-continuity assertion. Zero tests
   executing train.py is the systemic reason five fatal bugs sat under a green 314.
2. **Envelope v2 (fresh run — no legacy burden), add keys:**
   - `'schema': 1` (F5-24) + `'code_fp'` (already), `'cuda_rng'` =
     `torch.cuda.get_rng_state_all()` guarded by `cuda.is_available()` (F5-18);
   - `'stream_state'`: detached `state`, `gs`, `intent_state` (the `_detach_state`
     helper already exists), + `'onestep'`: `_last_salience/_intent_stream/_last_bus`,
     + `'alpha_pending'` — OR adopt the *document-boundary doctrine*: only save at
     rotation (`offset==0`) and delete this row; pick one deliberately, the current mix
     (mid-doc cursor + cold state) is the worst of both (F5-07; the 0.44 nat probe number
     is the price at D=128 — production 24-layer state is larger but the CE effect is
     not scale-free: it is the same one-document-context loss; measure once on A100);
   - `'alarm_strikes'`, `'scaler'` (F5-20), `'seq_len_live'` + `orig_seq_len` (F5-19),
     and `'reasoning_enabled_step'` in the *interrupt* save (F5-10);
   - `'step': step + 1` at the save sites to kill the replay +1 (F5-11) — pair with the
     `start_step` reads in both copies.
3. **Move the notebook save after `restore_runtime_buffers`** (c10:438 → after :449) —
   one reorder that closes F5-06 for the *future* envelope; also gate UCL +
   `bridge_stream` + `bind._step_count` writes on `self.training` so eval stops touching
   them at all (defense in depth + honest val).
4. **ggeo selector fix** `hasattr(_m,'_step_count') or hasattr(_m,'_alpha_pending')` in
   train.py:556 / c10:231 (F5-08 — probe-proven zero-delta) + a lock that counts
   `_residual_var_ema` writes across a `grad_geometry` call.
5. **Force-fresh hygiene** (F5-17): `else: _resume_depth_state = None` in cell 8, and
   cell 9's guard becomes `… if (not starting_fresh) else None`.
6. **Make the resume-clobbers opt-in** (F5-09): gate the "reopen cognitive gate" block on
   `('layers.0.mlp.mlp_gate_b' not in ckpt['model'])` (true legacy repair) instead of
   the always-on config default; same for any future resume-time `.data` overrides —
   a resume hook must *prove* the key is missing before overwriting it.
7. **One optimizer restorer, in core/**, with the pad-dialect (F5-03/§2-row-4),
   `param_names` accepted at both locations; notebook cell 9's bare call wrapped so a
   restore failure is loud, not positional.
8. **migrate.py shape-gate** the logit-cache drops (`shape[1]==D && shape[0]==vocab` only)
   (F5-15); fix the `scripts/migrate.py` pointer in the B8 abort message (F5-17).
9. **Interrupt file unification** (F5-10/F5-11): both copies write
   `interrupt_step_{step}.pt` (full v2 payload) on Ctrl+C; train.py's already-coding
   auto-resume glob stops being a tombstone; best.pt stays val-gated; add a real step-0
   seed write at setup (the c10 comment already pretends it exists) so a pre-first-eval
   death costs one session, not the model's whole life.
10. **Legacy shim for old envelopes** (F5-14c): on missing `_usefulness_temp` with
    restored `_step > warmup+50`, fill 0.5 (blend-end value) instead of leaving ctor 2.0;
    widen the detector pad to `while len < 7` (F5-14a edge).
11. Housekeeping with the same format touch: `persistent=False` on
    `bind._step_count/mirror._fwd_count/_pm_step` (F5-23), drop the redundant top-level
    `recover_count`, and prune the balancer key or populate it (F5-13).

Probe manifest (temp-only, outside the tree):
`%TEMP%\opencode\p5\probe_eval_and_logic.py` (P1/P2), `p3_run.py` + `P3A/B/C/D/E` modes
(battery + envelope census + legacy), `p_attr.py` (attribution), `p4p6.py` (ggeo,
notebook-cell exec, corruption), `p_attrs.py` (eval plain-attr census), `nb_cell{4,7,8,9,10}.py`
(cell extraction). Repo diff: this document only.
