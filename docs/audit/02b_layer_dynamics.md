# 02B LAYER DYNAMICS AUDIT

Agent **2b** of the serial relay, on the CURRENT tree (B6–B9 shipped: grad_geometry,
vocab-loud, L2-buffers/ceiling, identity-resume guard B8, embed_center B9 opt-in).
Scope: **everything between embedding-in and logits-out** — the per-layer block, the
VSA multi-scale memory state and its cross-window transfer, the bind/trajectory
spiral, the Cognitive Mirror, the grouped MLP, the collective concept layer (UCL),
the reasoning loop, the τ-ladder as consumed per layer, the bridge gates, and the
optimizer/grad-plane that moves all of them.  `embedding.py`/`vsa_utils` codebook
geometry are 2a's turf and not re-audited.

Mandatory inputs honored:
- `docs/audit/01_data_path.md` §2 (data contract) — taken as normative: `h` float32
  (B,L,D), `out = model(h, state, gs, step, tokens)`, y flattened-shifted-by-1,
  rotation semantics of §2.4, side-effect table F-04.  Every handoff addressed to
  2b is answered in §3, point by point.
- `docs/audit/02a_code_geometry.md` read in full; its §4 handoffs to 2b answered in
  §3; the §2 geometry-facts table is the shared numeric constitution (rank-64
  identity channel, `K=64`, pair-cos 0.9512 at embedding — my per-layer numbers are
  quoted against that baseline).

Method.
- All anchors read from the working tree; quoted lines verbatim (`file:N`).
- Executed repros live in `%TEMP%\opencode\`: `repro2b_A.py` … `repro2b_G.py`
  (mini EVAStack D=256, n_layers=2, bind_K=16, vocab=1820, seq 64, batch 1, seeds
  fixed 42; real token ids streamed from
  `WideBind\wb\token_stream_CHILDREN_eos.bin` (mod 1820) for the data-passing
  probes).  Production-scale ladder/τ/gate numbers are ANALYTIC from the exact
  code formulas (they are closed-form python floats), marked as such.
- Confidence tags: **VERIFIED-REPRO** (executed here), **STRONG-READ**
  (anchor-verified, deterministic consequence), **SPECULATIVE**.
- Suite at audit time: `python -m pytest tests -q` → **294 passed in 56.38s**
  (clean tree, run after all analysis; repo not modified by this agent).

Mini-ladder reality check (used throughout): with `dev=0` the B3-normalized cumsum
gives `tau_norm_i = i/n` (NOT i/(n−1)) — even layer 0 sits at τ = τ_min·(τ_max/
τ_min)^{1/n} ≈ 9.2, never 8; the 2-layer mini is τ=(64, 512), τ_mid=181, so mini
"shallow/deep" ratios are 0.354/2.828 (not 0.125/8).  Production (24 layers):
`τ_l = 9.17·1.166^l`, ladder 9.2→512, `τ_mid = √(τ_0·τ_23) ≈ 68.5`, layer ratios
`τ_l/τ_mid ∈ [0.134, 7.47]`.

---

## 1. Findings

### F2B-01 — The trajectory-spiral's cross-position arm is dead in steady training AND permanently killed by the first eval (B2's fix is shadowed by its own cache; `_traj_state` is a plain attribute that escapes the eval snapshot) | **HIGH** | **VERIFIED-REPRO**

Three mechanisms compound, all in `core/bind.py` + `core/block.py`:

**(a) Steady training runs on a frozen, detached window-1 cache.**
`block.py:369-380`:
```python
        if isinstance(self.bind, TrajectorySpiralBind):
            if traj_state is None and self.training:
                traj_state = getattr(self, '_traj_state', None)
            ...
            bind_out, new_traj, coherence = self.bind(h, traj_state)
            if traj_state is None:
                self._traj_state = new_traj.detach()
                traj_state_out = None
```
B2's in-graph shift (`bind.py:403-412`) fires **only when `traj_state is None`**.
In the training loop the state tuple carries `traj_state_out = None` (the cache
branch, line 378), so from window 2 onward the cache is read at line 371, the
in-graph path is skipped, and the cache is *never rewritten* (`traj_state`
non-None → line 376 not taken).  The trajectory channels `d=1,2` for the whole
rest of the document are the **first window's content, detached, frozen**:
measured `‖cached traj channels‖ = [4.0, 5.66]` (real forward) and
`|out(cached) − out(zeros)| = 0.137` — the content is non-zero, so it does feed
the spiral, but it is stale copy from window 1, not `h_{t−d}` of *this* window.
Consequences, both measured:
- within-window cross-position Jacobian `∂out[t]/∂h[t−1]`:
  **fresh window-1: 2.94** (t−2: 1.72, t: 26.7) → **steady windows: 0.0 exactly**.
  The README's `∂out_t/∂h_{t−d}` phase-rotation gradient path exists during
  exactly one window per document-stream start.
- `w_v_re[:, d≥1]` grads exist only on that first window.

**(b) Any eval forward overwrites the cache with zeros — permanently.**
In eval, `traj_state is None` (state fresh per batch, `adaptive=False` path),
`self.training=False` kills both cache-read (line 370) and in-graph shift
(`bind.py:403`), so `bind.py:413-414` feeds **zeros**; then `block.py:376-377`
executes unconditionally (`if traj_state is None:` — no training guard):
`self._traj_state = new_traj.detach()` where `new_traj = traj[:,1:] = 0`s.
Measured: after **one** eval call, `_traj_state.all() == 0` → True, and the next
TRAIN window's d≥1 channel norms = `[0.0, 0.0]`; the spiral degenerates to the
memoryless single-position product — **exactly the B2 pathology** ("grad
w_v_re[:,d≥1] ≡ 0") reinstalled, and this time for the remainder of the run,
because the zeros-cache also blocks any future in-graph re-derivation.
`eval_interval = 1045` → the trajectory arm is alive for steps 0..~1044 only,
modulo document rotations.

**(c) `_traj_state` is outside every isolation mechanism.**
`snapshot_runtime_buffers` (`stack.py:1021-1030`) snapshots named buffers +
`__attrs__ = ('_last_bus','_intent_stream')`.  `_traj_state` is a plain
`EVABlock` attribute → invisible to the eval snapshot/restore, invisible to the
document rotation (train.py:411-416 resets neither), cleared only by
`reset_cache()` (stack.py:1007) — which is called **only on resume**
(train.py:313-314), never at rotation or eval.  So: rotation hands the *previous
document's* trajectory (or the post-eval zeros) to the next document; eval hands
zeros to training.  Audit 01's F-04 row 13 ("eval YES (buffers)") is **wrong for
the one trajectory piece that matters** — `_step_count` is a buffer (and inert),
`_traj_state` is not (and is on the forward path).

Fix candidates (NOT applied): guard `block.py:376` with `self.training`; re-derive
the in-graph channels whenever `traj_state` came from the cache (i.e. make the
cache a fallback for STREAMING, not for windowed training); or pass the blended
`traj_state_out` through the state tuple in training too.  Needs a decision —
this silently deletes the depth of the binding mechanism the README sells.

---

### F2B-02 — The VSA multi-scale state decays with a **0.5-per-token content/pen floor**: at the operating point (pen ≈ 8, b_d ≈ 2..5) ALL FOUR scales of ALL layers carry ≈ 0 across one window; the τ-ladder is a first-window-only artifact (the "43x claim" heritage, now ×17–×1013 measured) | **HIGH** | **VERIFIED-REPRO** (mini, real corpus ids) + STRONG-READ (production extrapolation)

Exact transfer math (`block.py:388-471`): what is carried is `mem_state`/
`mu_state` (B, S·D), updated as a linear scan `M_t = decay_t·M_{t−1} + input_t`
via `_scan_chunk`/`_combine_chunks` (fp64 internally, exact — verified isolated:
`Δcombined_last == D^64·Δs0` to 5 digits).  Per-token per-channel decay
(`block.py:413-415`):
```python
        decay = (d_s_vec * d_mod_vec).clamp(min=0.01, max=1.0)
```
with `d_s = exp(−1/τ_s)` (τ_s = `_base_vsa`·τ_l/τ_mid, `_base_vsa ≈
[8.0, 28.9, 100.6, 350.4]`) and `d_mod = σ(h·w_d + b_d)` FURTHER multiplied by
`pen_decay_factor = 1 − (σ(pen + w_d_pen) − σ(w_d_pen))` (`block.py:21-28, 407-410`).

Operating-point census (real CHILDREN ids, stale-pen windows, init weights):
- `pen = ‖(hp − pred_k)/‖hp‖‖₂ over (G,k)` — a norm over G·k dims, NOT a mean:
  measured **mean 8.1 per position** (and the same quantity is *the design
  target* of the thresholds — see F2B-06).  At pen=8.1, `w_d_pen=0`:
  `pen_decay_factor → 1 − (σ(8.1) − σ(0)) = 0.5002` — the M3 "centered" form is
  **born at its own asymptote**: its floor is 0.5 and it is pinned there for any
  pen ≳ 3 (pen ≥ 64 gives e^{−64}, i.e. the clamp floor).
- At init `b_d = 2.0/5.0` → `σ ≈ 0.881/0.993`; combined per-token decay
  ≈ 0.49 (L0) / 0.49–0.50 (L1).  The τ-ladder survives only as a ~2e-3 relative
  per-scale difference.

Measured impulse (linear response of the carried state across one 64-token
window, buffer+cache-neutralized, eps = 1 % of the state slice): **window decay
r = 0.0 (underflow) for all 8 layer×scale combinations**, with the capture-based
per-channel stats:

| layer · scale | τ_nom (tok) | mean-channel window decay | τ_eff (tok) | speed-up vs nominal |
|---|---|---|---|---|
| L0 s0 | 2.8 | 2.2e-21 | 1.3 | ×2.1 |
| L0 s1 | 11.3 | 1.8e-20 | 1.4 | ×8.0 |
| L0 s2 | 45.3 | 3.1e-20 | 1.4 | ×31.8 |
| L0 s3 | 181.0 | 3.5e-20 | 1.4 | ×126.7 |
| L1 s0 | 22.6 | 2.2e-21 | 1.3 | ×16.8 |
| L1 s1 | 90.5 | 1.8e-20 | 1.4 | ×64.3 |
| L1 s2 | 362.1 | 3.1e-20 | 1.4 | ×254.1 |
| L1 s3 | **1448.2** | 3.5e-20 | 1.4 | **×1013.5** |

Nominal half-life of the slowest scale = 15.7 windows; measured **0.02 windows
(~1 token)**.  The carried state's share of the readout at the first positions of
the next window is 0.58 → 0.28 (t=4) → 0.004 (t=31) — i.e. even the part that
"is read" is the last-token residue of the previous window, and at production
`seq_len=512` the entire carried term is `0.49^512 ≈ 1e-154` = **zero in fp32**:
**no scale reaches earlier-window context across a full production window at
init.**  The 4×-decades "multi-scale VSA" is, in steady streaming, four copies
of a ~1-token EMA.  (This is the quantitative heir of the "43× claim" audit
lineage — measured distance from nominal here is 17× to 1000×+.)

Self-healing paths that exist in code, with their real budgets:
- `w_d_pen` (per-expert, init 0, optimizer role 'scalar' = full lr): driving it
  to ≈ −pen makes `σ(pen+w)−σ(w) → 0` and restores the ladder (∂ recovered at
  w=−12: factor 0.982).  Gradient measured alive (0.05 max) — but only from
  window 2 on (pen must exist; F2B-08).  Whether 150k steps × Adam(3e-4) walk
  there is *by learning* or not is data-dependent — SPECULATIVE.
- The AdaptiveController lerps `b_d` toward 7.7–8.3 (F2B-10), removing the σ(d_mod)
  penalty — but it does nothing about the pen factor: **even after the lerp, the
  0.5 asymptote alone caps every τ_eff at 2× the per-token step**, i.e. slowest
  production scale τ_eff ≤ 2 tokens ⇒ window(512)-carry ≤ e^{−256} ≈ 0.
- `clamp(min=0.01)` never binds at init; it would bind for pen ≳ 4.6 channels if
  `w_d_pen` went positive.

Companion effect — the write side is *inflated* by the same scalar:
`igate_logit += gamma_surprisal·pen` (`block.py:394-395`, γ_init=0.25 at
τ-mid ⇒ +2.0 at pen=8 → i_gate = softplus(−2.5+2.0+coh-boost)≈0.15–0.25 vs
0.08 designed), and `train.py:629-631` multiplies the state tuple by 0.1 on every
EOS-final batch (~1/64 of windows at B=1·L=64; more rarely in production).
The "surprisal-gated write" and the "prediction-error-aware decay" were designed
for a normalized error in [0,1]; they receive a ~√(G·k)-scaled norm instead —
same unit bug family as the mirror thresholds (F2B-06).

Classification: the decay table is VERIFIED-REPRO at D=256 mini with real ids;
the production statement (carry=0 across a 512-token window) is STRONG-READ: it
is the same closed-form at larger L, and the legacy 24-layer evidence
(`checkponts/best.pt`) cannot re-check it (file mid re-download during this
audit, per 2a's caveat).

---

### F2B-03 — Mirror streaming caches (`_cached_hp`, `_cached_pred_error_norm`) are forward-path inputs AND are mutated by eval; the snapshot contract does not cover them → held-out data measurably steers the next training window (B3-family hole beyond F-03) | **MEDIUM** | **VERIFIED-REPRO**

`block.py:407-434` reads `pen = mirror._cached_pred_error_norm` (fallback when no
state) and `hp_cached = self.mirror._cached_hp` (per-expert write modulation), and
the M10 decision made the eval path write both too (`mirror.py:503-506`,
`_cached_pred_error_norm` also at 521 in training).  They are plain attributes:
`snapshot_runtime_buffers` (stack.py:1021) restores buffers + `_last_bus`/
`_intent_stream` only.  Measured: train-window-without-eval vs
train-window-after-one-eval-despite-snapshot/restore:
**max|Δout| = 0.169, rel-norm 0.84 %** — i.e. the hold-out batch's K-space state
sets the next training step's write modulation and (via pen) the decay/writes
amplitude (see F2B-02: pen is the *dominant* term).  Also, two *consecutive
identical eval* forwards differ (max|Δ| = 0.0061, rel 0.6–0.8 %): "same weights,
same input ⇒ same output" fails across the batch boundary — not a counter bug,
a genuine one-step-stale streaming input.  Within training this staleness is
intended (streaming doctrine); across the eval/train boundary it is the same
contamination class as 01 F-03, one notch subtler, and it is NOT in the
F-04 table's snapshot column for the two rows that matter (rows 5–6 cover
`_private_mem`/EMAs — buffers — not `_cached_hp`/`pen`).
Fix candidate: add `('_cached_hp','_cached_pred_error_norm','_traj_state',
'_pi_v')` to the `__attrs__` snapshot list (shape-tolerant), or re-disable the
eval write of `_cached_*` while keeping the M10 "identical paths" by using the
*previous training window's* caches during eval instead.

---

### F2B-04 — Eval reuses the RAW maturation ramp while training uses `max(ramp, readiness)`: the same weights face a systematically *un-woken* trunk in validation (bridge injection, UCL gate, memory-bank gate all ~50–200× weaker in eval at early steps) | **MEDIUM** | STRONG-READ + computed schedule (numbers below)

`stack.py:378-393`:
```python
            if step is None:
                mat_gate = self.maturation.gate            # eval: buffer
            else:
                mat_gate = self.maturation.step_gate(step, self._tau_l_dev.detach())
                mat_gate = torch.maximum(mat_gate, self.maturation.readiness.detach().clone())
```
`step_gate` writes the *pre-max* ramp into the `gate` buffer (`maturation.py:150`),
so the eval-time `self.maturation.gate` **drops the readiness floor**.  Measured/
computed: readiness = σ((sat−0.3)/0.2), sat ≈ 0 at init → **0.182** for every
layer from step ~300; the raw ramp at t=495 (24-layer, dev=0, τ_norm_i=i/24) is
mean 4.2e-4, per-layer [9.2e-7 … 2.8e-3].  Consequences (all fed by
`mat_gate[i]`):
- bridge injection scale (`stack.py:485-488`) — train ×0.18, eval ×3e-4;
- private-memory write scale (`mirror.py:604-607`) — (train only, moot);
- UCL `mat_gate` arg (`stack.py:565`) — writes skipped in eval *by the gate* even
  where training would (readiness-floor ≥0.1) enter `_maybe_write`; (eval writes
  separately disabled by `allow_write`? — UCL is called with `allow_write=True`
  from stack:572 unconditionally — 01 F-03's per-layer-bank gate now has
  `self.training` in the bank (memory_bank.py:413), but the **UCL still writes in
  eval** once its internal `mat`/gates allow — its buffers are snapshot-covered,
  so the damage is restore-able, unlike F-03's era.)
- the *logged* `layer_gate_*` aux channels (losses.py:379, 393-398) read
  `stack.maturation.gate[l]` (raw) — see §2 for the 0.0008 reconciliation.
The M8-era comment "eval: reuse the LAST TRAIN gate" (stack.py:380-381) is
therefore **not** what happens: eval reuses the last *ramp*, not the last *gate*.

---

### F2B-05 — Dead knobs & dead routing on the layer-dynamics surface: `bind_twist_gate` (True in cell 4!) is not implemented in the shipped bind mode; `w_pred_scale_init`, all `collective_*` fields, `stack.param_groups` (whole λ-group policy incl. `tau_dev_lr_mult`, `gate_lr_mult`), `cache_grad_norms()` fallback, `collective_stats`/`projector_signals` — verified unreferenced/unwired | **MEDIUM (config-integrity)** | **VERIFIED-REPRO** (hasattr/grep/optimizer-census)

- `bind_twist_gate=True` (config.py:360, cell 4 sets it): the gate
  (`w_gate_proj`, bind.py:131-135) exists **only inside `BottleneckBind`**;
  `TrajectorySpiralBind` (the default `bind_twist_mode='trajectory_spiral'`)
  has no gate at all — measured `hasattr(bind,'gated') == False` on the mini
  stack.  The live per-bind gate is instead the per-expert `w_bind_gate`
  (block.py:219, 526: `sigmoid(0)=0.5` at init) + coherence write-boost
  `1 + bind_coh_gate(0.5)·mean(coh≈0.44) → ≈1.22×`.
- `w_pred_scale_init=3.0` (config.py:72, cell 4): **zero consumers repo-wide**
  (grep).  The live `pred_scale_mod` comes from `AdaptiveController.
  pred_scale_mod` (delta_var-centered, clamp [0.1,3]).
- `collective_layer / collective_read_out / collective_contra_thresh=-0.1 /
  collective_contra_gain=6.0 / collective_*`: no consumer in `core/` — the
  collective is `UnifiedConceptLayer` (own learnable `log_tau_contra`,
  `uncert_kappa`), and `block.collective = None` (block.py:295).  The
  `stack.collective_stats()` and `projector_signals` readers always return
  None (they iterate per-layer `l.collective`).
- `EVAStack.param_groups()` (stack.py:1145-1295) — includes the ONLY consumer
  of `cfg.tau_dev_lr_mult` (0.2) and legacy `gate_lr_mult` — has **no caller in
  train.py or the notebook** (both build via `core.adaptation.
  build_optimizer`).  Measured: `_tau_l_dev` lands in a group with **lr = 3.0e-4
  (= cfg.lr, role 1.0)**, i.e. the declared "conservative system-lever" runs 5×
  hotter by train.py's own intent and 5× hotter than `param_groups` claims.
  Worse, `_role_lr_mult('_tau_l_dev')` hits no rule (only `'tau_config.'` does,
  but named_parameters dedups the shared tensor under the *alias* name
  `_tau_l_dev` — the `tau_config._tau_dev` name literally does not appear in
  `named_parameters()`, measured).
- `mirror.cache_grad_norms()` (mirror.py:932-940): no-arg fallback copies the
  never-written `_hp_grad` buffer (zeros) into `_prev_grad_norm`.  Verified: it
  zeroes the live hook-captured value (1.04e-2 → 0).  No current caller in
  core/scripts — but it is a public, docstring-advertised API ("Устанавливается
  извне через cache_grad_norms") — one `git grep` from a generation driver
  future into a silent gate-deadening.  (The live channel is the `hp`
  register-hook at mirror.py:412-416, which works.)
- `stack.collective_stats` reads retired attrs (`N_s`, `U_s`, `_births_allowed`)
  that would AttributeError if `collective` were ever non-None — dead on both
  ends.

---

### F2B-06 — The mirror's own operating-point units: `pen ≈ 6–10` where thresholds were specced for O(1) — `u_gate` (UCL uncertainty) and the `pred_scale_mod`/`dvar_mod` gates are born saturated; `_pi_v`, `_usefulness_temp`, `_damp_tau` cross-checks | LOW-MED | VERIFIED-REPRO + STRONG-READ

- UCL `u_gate = σ(κ(pen − e^{log_tau_uncert}))`, `log_tau_uncert` init →
  threshold 2.72 (concept_layer.py:349-354): measured fire-rate (>0.5) =
  **1.000** with mean 0.99999 on real data at init — the "ask the archive only
  when uncertain" gate opens fully because pen is a norm, not a rate (same root
  cause as F2B-02).  c_gate (contra, threshold 0 on cos) fires 0.51 — that one is
  genuinely sign-based.  Mirror-side `contra = σ(disagreement − 1.0)`
  (mirror.py:536) — measured mean 0.5001, fire 0.510 (chance-level at init, as
  designed for random experts).
- `mirror._usefulness_temp` (buffer, init 2.0, persistent=False) is rewritten by
  `MirrorLRScheduler` to `max(temp, 0.1)` (lr_scheduler.py:188) — **never ≤ 0**,
  so mirror.py:736-738 always overrides the `prog`-based temperature.  The
  train-vs-eval asymmetry that B3's doctrine warns about — `n_eff = _fwd_count`
  (train) vs `n_eff = step or 0` (eval) at mirror.py:722-729 — is therefore
  currently **shadowed**: if the M7 wiring were ever flagged off (`_usefulness_temp
  → 0`), eval forwards with `step=None` would run temp=3.0 against training's
  converged ~0.3: re-add a guard (use the *last train* count in eval, or drop the
  counter branch entirely — it is now dead code with a live landmine).
- `_ar_mode` damping (mirror.py:461-464, "opt-in by generation drivers"): not an
  attribute of a fresh mirror (measured `hasattr == False`) → getattr default
  False, fine; the doc-rotation/reasoning loop paths never set it — generation
  scripts must; not verified here (out of scope, agent 5).

---

### F2B-07 — Two copies, three grad-planes: train.py and the notebook apply *different* per-layer gradient multipliers, different trust signals, and a different `llrd_decay` to the same core | **MEDIUM** | STRONG-READ (census) + VERIFIED (group lr values)

- **τ-LLRD application**: notebook c10:247 `apply_tau_lr(model, tau_config,
  ls_mults)` scales every layer param's grad by `ls_mult × (τ_l/τ_ref)^−γ`
  (training_control.py:98-124); train.py:583-596 inlines **only `ls_mults`**
  (×phase-scaling for mirror), never the τ-lr term → deep-layer grads differ by
  up to ~3.4×/5× between the two arms at identical weights.  `apply_tau_lr` is
  also applied **after** `clipper.clip()` in the notebook (c10:243 vs 247), while
  train.py scales before clip (:583 vs :605) — the AGC ratio invariant `‖g‖ ≤
  c·‖θ‖` is enforced pre-scale in one copy and post-scale… actually post-clip
  scaling in the notebook breaks the c-interpretation (the clipped-and-then-multiplied
  grad no longer satisfies `‖g‖ ≤ c‖θ‖` by the applied factor).  Same order-bug
  shape as audit M-series, new instance.
- **Mirror phase scaling**: train.py:556-578 multiplies mirror grads by a
  self-referential `mir_s ∈ [0.2,2]` (fast ratio vs EMA); the notebook has no
  phase block.  `mean_mirror_scale`/`mr` log fields exist only because of it.
- **Optimizer `llrd_decay`**: train.py `_make_opt` uses `cfg.llrd = 0.9`;
  notebook cell 9 line 33 builds `build_optimizer(..., llrd_decay=1.0, ...)` —
  two different depth-LR laws on top of the identical λ-role map (measured group
  lrs: e.g. `b_d` at 8.87e-5 = 3e-4·λ⁻² in both, since layer 0; at layer 23 they
  diverge 0.9²³× = 0.086× vs 1.0×).
- **EVAAdamW trust**: train.py:611-619 feeds `set_trust({bridge, intent, mem})`
  (maturation-based, floor tscale ≥0.5); the notebook never calls `set_trust`
  → trust defaults to 1.0 (eva_optim.py:235) on the production (A2 `eva_proj`)
  arm.  Same optimizer class, different branch-suppression policy; in Colab the
  immature-branch damping is *off*.
- **gradalign**: `cfg.gradalign_weight = 0.3` hard-set in cell 10:26 vs train.py
  default 0.0 (already 01-M5; restated because it toggles the F2B §1.4 hook
  machinery entirely between arms).

---

### F2B-08 — Grad-path liveness census for the layer-dynamics modules (what trains when): spiral trajectory, reasoning chain, UCL write path, intent-bridge U8 knobs | **MEDIUM** | **VERIFIED-REPRO**

- Reasoning loop (`explicit_reasoning=True`, `reasoning_adaptive=True`):
  `_adaptive_reasoning` returns **only `h_acc`** (stack.py:775-890) while the
  caller re-binds `reasoning_buffer, reasoning_count = ...` only in the
  NON-adaptive branch (660-663); the attribute write-back at 664-666 therefore
  stores the ORIGINAL `None`s — measured: after a training step
  `model._reasoning_buffer` is `NoneType`.  Agent 3's "the chain-of-thought is
  dead in the notebook config" is **confirmed** (it is `stack.py:659` discarding
  `buf/count`, not a doc-rotation issue; `reset_reasoning` coverage is moot).
  What *does* live: step-0 candidate path — `step_encoder` grad 5.8e-3,
  `output_proj`/`step_query/key/value` grads **exactly 0.0** (the empty-buffer
  blend `output = current_step·empty + …·(1−empty)` (reasoning.py:81-82) kills
  everything downstream of the empty attention), `reasoning_gate.proj.weight`
  grad 9.2e-5 (STE on step-1 gate works), gates 2..K masked to exactly 0 by
  `run.float()` (prev_open ≈ 0) — documented sequential gating, working.
  `K = max(1, round(8·τ_norm.mean())) = 6` at the mini mean; scale ramp
  1e-3 → 0.632 at t=1000 and the ramp IS synced per-step in **both** copies
  (train.py:378-379 is inside the loop — re-read, correcting my initial read).
- The mission's B7-collision hypothesis — *"reasoning output routed through
  embed → THINK..END ids ≥ vocab get clamped → module silently dead"* — is
  **REFUTED with proof**: `ReasoningMemory`/`ReasoningGate` consume only `h`,
  `know` and `r` (reasoning.py:37-97, stack.py:775-890); no token ids enter the
  module and no reasoning output is re-embedded anywhere in `core/` (the only
  id-channel is the head, which emits ids < vocab; embedding.py:144's own
  comment: "reserved reasoning tokens are NOT wired").  The module is *alive but
  shallow* (one deliberation step, no memory), not *dead by clamping*.
- UCL write path: `write_q_proj/write_v_proj` grads **NONE** at early steps —
  correct for now (writes gated by `mat_gate≥0.1` AND internal `_mature≥0.3/0.1`,
  measured `_mature=0.0083` at init, births 0 / 6 real-data steps,
  `skipped=384`); the M6 functional-write grad path is intact by code reading
  (`index_copy` off the buffer, source-live) but **cannot fire until maturity
  drifts up** — and when it fires under bf16 it will crash (F2B-§4.1).  Read
  path IS on the gradient path now: `q_proj`/`out_proj`/`read_scale` grads
  measured non-zero (3.7e-5 / 3.8e-4 / 5.4e-5) — no detach kills it (the
  amplitude cap at concept_layer.py:371-373 uses `.detach()` norms only — safe).
  One detach worth noting: the per-expert `gate` fed into `_maybe_write` is
  `mirror._cached_gate` — a `.detach()`ed tensor (mirror.py:920) — so design
  principle #4's "expert gate weights the shared representation" learns the
  gate **not** through UCL (only hp does); SPECULATIVE intent, flagged.
- Intent bridge (production default `intent_bridge=True`): `w_intent` gets grad
  9.0e-2 despite zero-init (via `hp−ik`); `_w_alpha_expert` (U8 per-expert carry
  deviation, stack.py:448-453) has grad **0.0** at init — its live path is
  `ig = einsum(hp−ik, w_intent)`, and w_intent=0 ⇒ ∂L/∂α = 0: liveness is
  chained behind w_intent growth (delayed, undocumented).  The U8 formula itself
  is correctly `α = (1−1/τ_l)·(1+(2σ(w)−1)(2·τ_norm−1))`, clamped [0, 0.999],
  τ≥2 (verified at stack.py:446-452; τ_norm<0.5 inverts the deviation sign —
  by design "centered").
- spiral: `w_v_re[:, d]` grads [0.012, 0.012, 0.009] **only on the fresh
  window-1 forward**; per F2B-01 they vanish after window 2 and after the first
  eval.

---

### F2B-09 — B6–B9 items *inside this scope*: honestly fixed | INFO (verification)

- **grad_geometry (B6)** — `LossBalancer.grad_geometry` (training_control.py:
  560-612) is `autograd.grad`-only, touches no `.grad`, and (measured) does not
  perturb the gradalign hook: nothing backprops through `h_mlp` in it (diversity
  reads `_cached_group_out`, upstream of the hooked tensor).  Sound.
- **L2 buffers (B7)** — `memory_bank.py:206-207` now `register_buffer('keys'…
  'vals')`, and `L2Bank.forward` write gated by `self.training`
  (memory_bank.py:413): **both 01 F-02 (Adam-momentum-under-wipe) and F-03
  (eval-writes-parameters, torn bank) are fixed** — keys/vals are now inside
  `snapshot_runtime_buffers` and never written in eval.  Residual: `_n_overwrites
  /_n_consumed` remain plain ints (not restored), and `mem_tau_reg` still targets
  `log_tau` Parameters only.  A surviving instance of the *pattern*: the
  resume-path `reset_skip_alpha` knob (train.py:285-289) zeroes a Parameter
  in-place while Adam carries momentum toward it — verified-adjacent ghost
  below (§2).
- **identity-resume guard (B8)** — `verify_identity_resume` + codebook sha
  fingerprint are wired in BOTH loops (train.py:284 / c8:96; c10:400/442) —
  F2A-03 closed.  Minor: train.py:284 overwrites `missing, unexpected` with the
  fingerprint return (cosmetic).
- **embed_center (B9)** — opt-in knob, `config.py:65` default False, wired in
  `embedding.py:114-123, 151` (centered by the sampled code-prior sigmoid mean;
  the comment honestly admits code-centering left cos 0.935 and centers the
  *dense* activations instead).  Layer-dynamics consequence when switched ON:
  trunk-wide pair-cos drops from 0.9512 → ~0 (2a's F2A-06 numbers), so every
  cosine/entropy channel in the mirror, UCL sims and memory bank changes meaning;
  the `bit_bias` prior absorbs the head-side DC.  Fresh-run-only, as the
  comment says.  Untouched here (geometry = 2a).
- **B3 lesson (train≡eval)**: `_hybrid_alpha` is τ-static (bind.py:363-377) —
  verified: repeated eval forwards are free of counter-dependence; the residual
  eval-to-eval non-determinism measured (0.84 rel) is the *stale-streaming-input*
  channel (F2B-03), not a step counter; `_step_count` still increments under
  eval/no-grad (`bind.py:397`, measured 33→36) but feeds nothing — checkpoint
  bloat + confusion tax only.

---

### F2B-10 — AdaptiveController: b_i/b_d are dual-driven (Adam + hard lerp), and the controller's equilibrium *shrinks deep-layer writes to 0.001* because it assumes the un-crushed ladder | MEDIUM | VERIFIED-REPRO (targets) + STRONG-READ

`stack.py:298-306` (01 handoff #4): per train-step, outside the optimizer:
```python
                    if smooth >= 1.0:
                        layer.b_i.fill_(b_i_val)
                        layer.b_d.fill_(b_d_val)
                    else:
                        layer.b_d.data.lerp_(b_d_t, 1.0 - smooth)
                        layer.b_i.data.lerp_(b_i_t, 1.0 - smooth)
```
`vsa_b_d_smooth=0.999` → 0.1 %/step (τ≈1000 steps) pull toward the controller
target computed from `expl = min(1, |mirror|/0.25)`.  Measured at init
(real-data forward): `expl ≈ 0.43–0.53` (|mirror|≈0.12) → b_i targets
**−4.84 (L0) / −6.77 (L1)** (i_gate equilibrium 0.0079 / **0.0011**) and b_d
targets 7.67 / 8.28 (σ→0.9995).  Both `b_i` and `b_d` are ordinary Parameters
with grads (b_d grad norm measured 0.198) and Adam state (lr λ⁻²=0.296×), so the
CE-gradient and the controller fight over the same coordinates; worse, the
controller's write-rate law `i_gate = c/τ_l` (adaptive_controller.py:101-128,
B3-fixed to solve softplus⁻¹ exactly) is derived from the *nominal* τ_l — under
the F2B-02 crushed ladder (τ_eff ≈ 1.4) the equilibrium memory norm
‖M‖≈i_gate·‖h‖·τ_eff is **~50–100× below** the design invariant
"‖M_l‖ = const across layers".  Eval: `adaptive=False` skips the writes ✓; but
the *values* are the trained ones, no eval effect.  `fill_` (no_grad) on leaf
Parameters is autograd-legal; no version-counter hazard (the graph is rebuilt
per window).

---

### F2B-11 — MLP/experts & gradalign: bookkeeping verified, one latent race, one definition note | LOW | VERIFIED-REPRO + STRONG-READ

- Routing-balance signal source: `balance_loss` consumes
  `mirror._cached_gate_usage = expert_gate.mean(dim=(0,1))` — **forward gates,
  not gradient magnitudes** ✓ (B2 doctrine).  Same for `gate_l1`
  (`expert_gate.mean()`, mirror.py:912), `delta_var` (var of the forward `delta`,
  mirror.py:713 — under `no_grad`, train-only), `diversity` (correlation of
  forward group-out norms), `mlp_ratio` (forward ‖h_mlp‖ EMAs).  The remaining
  grad-magnitude-as-signal channel is **by design**: `grad_mod` (mirror.py:709
  + `grad_mod_input`, 42-52) and the `hp` register-hook (412-416) feeding
  `_prev_grad_norm/_grad_norm_ema` into the expert gate logits — scale-free via
  EMA division (M4), never exported to the watchdog (`_mets` has no grad channel,
  train.py:459-475).  `cache_grad_norms()` zero-fallback: F2B-05.
- gradalign hook (block.py:578-591): target `‖∂CE/∂mlp_out‖` per expert.  With
  `LossBalancer.backward(phase_model=model)` the freeze (training_control.py:
  654-656 / 709-711) works — **measured**: post-full-backward targets match the
  pure-CE reference to ≤ 0.4 % (e.g. L0 `[.01575,.01397,.05139,.03452]` vs
  `[.01575,.01396,.05119,.03454]`).  Latent race: the `not aux_tensors` early
  return (training_control.py:644-652) runs `sum(bypass).backward()` **before**
  `_ga_record=False` is set at 654 — if the aligned-aux dict is ever empty while
  gradalign is on, the bypass gradient is recorded as the CE target.  In the
  real loops `pred` is always a tensor in train mode, so unreachable — SPECULATIVE
  reachability, STRONG race logic.  Also the target is one-step-stale by design
  (hook fires during the *current* CE backward while `compute_losses` for the
  *next* step reads it) — documented, fine.
- asymmetry init census (mini): α ladder 0.85→0.99 per expert (exact),
  `W_proj` rows orthonormal (‖Gram−I‖ = 0.0000), log_scale ladder
  [−2.797, 0.018] = ln(τ_g min/max)-geometric (M4-fixed, bounded), and
  per-expert within-ladder std 0.0 + noise (log_scale_init_std 0.05).
- `mlp_ratio` metric: fast/slow EMA ratio of the global scalar ‖h_mlp‖ with
  cold-start rebase (block.py:593-604).  Measured over 12 real-data steps:
  0.993→0.931 (L0/L1) — drifts *below* 1 while ‖h_mlp‖ decays slowly (ratio
  semantics are symmetric, not runaway-only); runaway detection (B4) is
  `max_layers(ratio)` vs slow·(1+1.0) AND floor 2.0 — a halving does not fire,
  a doubling must exceed BOTH → definition consistent with the docstring.
- 48-tap conv interplay with state: causal explicit pad fixed (M11), carry
  (B,D,47) dtype = h.dtype — **cross-context dtype carry landmine** §4.2.

---

## 2. Measured dynamics table (the constitution numbers for §4/§5 agents)

Mini = D=256, L=2 layers, bind_K=16, seq=64, batch=1, real CHILDREN ids,
weights at init unless stated; Prod = 24-layer D=2560 cell-4 ladder computed
from the exact code formulas (closed form).  Repro ids in parentheses.

| # | Quantity | Value | Where / conditions |
|---|---|---|---|
| 1 | τ ladder as consumed (mini) | τ_l=[64,512], τ_mid=181; VSA scales L0 [2.8, 11.3, 45.3, 181], L1 [22.6, 90.5, 362, 1448] tok | repro2b_A/B; `_base_vsa=[8.0,28.9,100.6,350.4]` |
| 2 | prod layer ratios τ_l/τ_mid | [0.134 … 7.47] (τ_norm_i=i/24, never 0) | F2A-consistent, analytic |
| 3 | operating pen (mirror pred-error norm over G·k) | **8.1** mean (real ids, steady) | repro2b_E |
| 4 | pen_decay_factor at that pen | **0.5002** (asymptote) | repro2b_E |
| 5 | per-token decay, init, steady windows | ≈0.49 all scales (mean-channel) | repro2b_D |
| 6 | **window-carry r (64 tok), all 8 layer×scale** | **0.0 (fp32 underflow); isolated math 2e-21…3.5e-20** | repro2b_C/D |
| 7 | speedup vs nominal τ | ×2.1 … **×1013** (L1 s3) | repro2b_D |
| 8 | half-life (windows): nominal vs actual | L1 s3: 15.7 → **0.022** | repro2b_D |
| 9 | carried-state share of readout (positions) | t=0: 0.58, t=4: 0.28, t≥31: ≤0.004 | repro2b_C |
| 10 | production seq=512 cross-window carry at init | ≤ 0.5^512·1 ≈ **1e-154 → 0.0** | STRONG-READ |
| 11 | i_gate init | mean softplus(−2.5 + content) ≈ 0.12; window-2 boost γ·pen ≈ +2.0 → ~0.2 | repro2b_C/D |
| 12 | twist pair-rotation isometry | max abs rel dnorm = **1.19e-7** (real forward) | repro2b_A |
| 13 | bind gain ‖out‖/‖h‖ | 0.451; coherence mean 0.443, max 0.996 | repro2b_A |
| 14 | hybrid α (τ-static, B3) | L0 0.500, L1 0.300; eval-repeat determinism unaffected by it | repro2b_A |
| 15 | `_step_count` | increments under eval/no-grad (33→36), **feeds nothing** | repro2b_A |
| 16 | cross-position Jacobian ∂bind_out[t]/∂h[t−1] | fresh W1 **2.94**; steady cache **0.0**; eval **0.0** | repro2b_B |
| 17 | `_traj_state` after 1 eval | all-zero, overwrites live training cache (plain attr, NOT snapshotted) | repro2b_B |
| 18 | eval→train leak despite snapshot/restore | max abs 0.169, **rel 8.4e-3** | repro2b_B |
| 19 | repeated identical eval forwards | max abs Δ 0.0061 / rel ~1e-3…0.84 (first pair) | repro2b_A/B |
| 20 | gradalign vs CE-only reference | ≤ 4e-4 rel — **freeze works** | repro2b_E |
| 21 | cautious ÷√ρ energy ratio (toy, correlated sign noise) | 1.000 / 1.019 / 1.048 / 1.096 for ρ≈1/.95/.88/.75; amp bounded ×2 (floor 0.5·max(.5,1−2/√n)) | repro2b_E; eva_optim.py:288-298 |
| 22 | LossBalancer bound | per-param ‖aux‖≤‖CE‖ literally (sign-mask + clamp), bypass same | training_control.py:685-708 |
| 23 | mirror expert gates (real data, init) | gate_l1 0.44→0.47 over 12 steps; contra mean 0.5001 fire 0.510 | repro2b_G/E |
| 24 | UCL u_gate fire-rate | **1.000** (born-open, pen 8.1 ≫ e^1 = 2.72 thr) | repro2b_E |
| 25 | UCL maturity / writes | _mature 0.0083; births 0, updates 0, skipped 384/6 steps | repro2b_E |
| 26 | private_mem write at step 0 | ON with scale σ((0.182−0.3)·10)=0.233; _private_mem ‖·‖ grows; read path alive | repro2b_E |
| 27 | mlp_ratio (12 real-data steps) | L0 0.993→0.953, L1 0.989→0.931 (healthy band; watchdog floor 2.0, margin 1.0) | repro2b_G |
| 28 | layer_gate at step 495 (prod ladder) | raw ramp per-layer [9.2e-7 … 2.8e-3], mean **4.2e-4** → live log 0.0008 is mid-band (≈layer 21 or small positive dev): **schedule math confirmed**, log value = ramp NOT max-with-readiness | repro2b_G |
| 29 | readiness floor (train effective gate) | σ((0−0.3)/0.2)=0.182 → train=0.18 vs eval/log=ramp (≤2.8e-3) → **F2B-04 gap** | repro2b_G |
| 30 | global_ready (bridge-control) time | shallowest crosses 0.1 at step ≈ **9250** (dev=0) | repro2b_F |
| 31 | reasoning: module liveness | step0 path alive (step_encoder grad 5.8e-3, gate-1 STE 9.2e-5); step_query/key/value/output_proj grad **0.0**; gates [1.0,0,…]; K=6; **buffer never persists** (NoneType after step) → chain dead, module ≠ dead-by-clamp (no ids involved) | repro2b_G/E |
| 32 | intent bridge grads | w_intent 9.0e-2 (alive despite zero-init); `_w_alpha_expert` **0.0** at init (chained) | repro2b_G |
| 33 | optimizer census | `token_bias` in optimizer, grad 0.131, role lr 1.0× (not λ⁻²); `log_temp` grad 2.0 role 1.0; `k_proj_l/v_proj_l`: in groups, **grad None, NO state** (no momentum on dead params); `_vsa_tau_log` excluded by name ✓; `_tau_l_dev` → role 1.0 → **lr 3e-4** (τ_dev_lr_mult=0.2 dead: `param_groups` has no callers) | repro2b_E/fixups |
| 34 | L2 keys/vals | now persistent **buffers**; writes skipped in eval (`_can_write = self.training…`) → 01 F-02/F-03 (parameter part) **FIXED** | memory_bank.py:206-213,413 |
| 35 | Adam-ghost toy (the F-02 mechanism, kept for posterity) | after `.data.zero_()` + 20 steps w/ ~0 grads, ‖p‖ re-grows 0→0.029 from stale momentum (lr 1e-3) | repro2b fixup |
| 36 | b_i/b_d controller targets vs grads (init) | b_i → −4.84/−6.77, b_d → 7.67/8.28 (τ-lerp τ~1000 steps) while b_d.grad norm 0.198 + Adam state exist → dual drive | repro2b_G/C |
| 37 | expert-asymmetry init | α 0.85→0.99 exact; W rows orth ‖G−I‖max 0.0; log_scale [−2.80, 0.02] | repro2b_G |
| 38 | suite | **294 passed in 56.4 s** (clean tree) | post-analysis |

---

## 3. Reactions: 01 §5 handoffs to 2b, 02a §4 handoffs, B6–B9 honesty check

**01 → 2b, F-02 (Adam moments under in-place-zeroed L2 keys/vals).** Superseded by
B7: `memory_bank.py:206-207` moved keys/vals to `register_buffer` → they left
`named_parameters()`, optimizer groups and any momentum-desync; the rotation wipe
is now a pure buffer op. **Honest status: FIXED** for the momentum part; the
*semantics* part of F-02 ("mid-document hard wipe is not decay") stands as a
design decision for agent 3.  Residual bookkeeping inconsistency also shrunk:
`_n_overwrites/_n_consumed` are still plain ints (not restored by snapshot) but
no longer mixed with parameters.  I kept the ghost mechanism on record (§2.35)
because train.py:285-289 `reset_skip_alpha` reproduces the *pattern* (zeroing a
Parameter under live Adam state) on the resume path.

**01 → 2b, F-10 (veto-skipped scheduler vs watchdog clock).** For my scope the
consequence is: `MirrorLRScheduler.step()` (→ `ls_mults` → grad multipliers, and
→ `_usefulness_temp` writes) is gated on `scheduler.step()`, which vetoes skip
(train.py:622) — so during a veto storm the per-layer LS-modulation and the
usefulness temperature **freeze while maturation ramps (loop `step`) advance**:
the two controllers that shape layer dynamics desynchronize from the τ-field's
own step input.  Also `watchdog._cur_step` drives warmup, not the grad plane —
the direction of the skew matters less than that **three clocks** (loop step,
scheduler step, watchdog step) now gate different layer-dynamics inputs.  Agent 4
owns the resolution.

**01 → 2b, F-11 (positional optimizer restore; dead by-name impl).** STILL
UNFIXED in train.py: `_restore_optimizer` is defined (train.py:81-149, name-based,
with the W_out+K padding branch) and **never called** — the live resume is again
`optimizer.load_state_dict(ckpt['optimizer'])` (train.py:320) after
`_make_opt` rebuild.  The notebook wires its own by-name path (c8:127+).  In
layer-dynamics terms: any insertion/reordering in `named_parameters()` (e.g. the
B-era addition of `_w_alpha_expert`, `bridge_glu_net`, or a new mirror knob)
silently re-labels Adam momenta slot-wise — `exp_avg` of `layers.7.b_d` can be
restored into `layers.7.b_i` or a `w_q_dyn`.  The `param_names` list is written by
both loops but with *different constructions* (train.py:685 via `_opt_param_names`
vs c10:402 inline id-map comprehension) — same list, semantically equal today.
Needs-lock (agent 5 territory): resume must restore by name or refuse.

**01 → 2b, stack.py:299-306 (AdaptiveController writes b_i/b_d Parameters every
train step outside the optimizer).** Answered fully in **F2B-10**: yes, dual
driver (Adam + 0.1 %/step lerp), and the controller's own equilibrium is derived
from the *nominal* ladder, so under F2B-02 it deepens the write-rate inversion
(deep i_gate → 0.0011) instead of stabilizing it.  Also note `vsa_b_d_max=12`
means the controller can eventually push `σ→0.999994` (decay-mult τ≈180k) — if
`w_d_pen` recovers the ladder, b_d and pen modulation must be *co-read* by the
optimizer; two knobs, one quantity (per-token decay).

**01 §5 "cross-check cell10 snapshot list" (which state I must check).** Checked
each row of F-04's table that touches my scope: rows 5–8 are buffers → snapshot
✓; row 11 → F2B-10; row 13 ("spiral/traj `_step_count` … eval YES (buffers)") is
**half-wrong**: `_step_count` is a buffer and inert, but the forward-relevant
piece is `block._traj_state`, a **plain attribute** written by eval — §1.17/
F2B-01; row 9's `_cache_*` "overwritten next fwd" understates the mirror pair
(`_cached_hp`, `_cached_pred_error_norm` are *read before overwrite* as stale
inputs, and eval mutates them → F2B-03, measured leak 0.84 %).  These two are my
contribution to the future `reset_document_state()` list (agent 5 owns it).

**02a → 2b (a): role LR for `embed.*/lm_head.readout`, and whether
`log_temp/bit_bias` should move faster.** Census (`adaptation.py:184-201` + live
groups): `embed.*`, `lm_head.readout*`, `lm_head.proj*` → λ⁻² ≈ 0.296× (deliberate,
matches the F2A-09 "basis norms drift" story).  But `lm_head.log_temp`,
`lm_head.bit_bias`, `lm_head.token_bias` match NO role → **1.0×** (full lr, wd 0,
no decay): they are *not* LR-starved.  Their apparent freeze at step 1045 is then
about gradient geometry, not LR: measured at mini, `log_temp.grad ≈ 2.0` and
`token_bias.grad ≈ 0.13` — nonzero and *large* — so Adam must be oscillating
around the calibrated fixed point (log_temp enters every bit-logit; its optimum is
sharp).  Recommendation: leave LRs as-is; if anything, AGC c=0.1 ratio on
`log_temp` (‖θ‖≈small scalar) may throttle it — agent 4's AGC c review.  (b)
`k_proj_l/v_proj_l/logit_to_hidden`: they ARE in the param groups, but with
grad None Adam creates **no state** (verified: absent from `opt.state` after a
step) → no momentum, no decay, no drift; the cost is envelope bytes, not dynamics
— 2a's "momentum on dead params" concern is **refuted** for the default AdamW arm
(EVAAdamW path also skips `p.grad is None`, eva_optim.py:240-241).  (c)
`token_bias` in the optimizer list: **confirmed present**, role-scalar, grad
nonzero — its zero-value at 1045 is an optimum, not exclusion.

**B6–B9 honesty check (my scope).** B6 grad_geometry: verified clean (§1.4 note,
F2B-11).  B7 buffers/ceiling: L2 fix confirmed in-tree (F2B-09) — and honest
caveat: 01's F-03 repro (write_idx 0→2, keys mutated) no longer reproduces on
this tree.  B8 identity-resume: wired, both loops (F2B-09).  B9 embed_center:
opt-in, wired, default-off — no layer-dynamics measurement applies until a fresh
run flips it (handoff: whoever flips it should re-measure F2B-02: centering
changes `h` norms → pen, i_gate, decay all move).  Where prior audit findings
were *not* fixed but look named: F-04 intent-stream priority override (still
stack.py:333), `_last_intent_state` (c10:133) reads an attribute **nothing ever
sets** — the "fixed" reset is decorative; M5 gradalign divergence (cell10 hard-set
0.3) still alive (F2B-07).

---

## 4. bf16 landmine inventory (for B10; the A100 restart is autocast-everything: cell 10 wraps the full forward + losses)

Policy recap: notebook `_AMP_DTYPE = bfloat16`, fp16 "disabled by design"
(c5:12-14); train.py's AMP wraps only `embed_tokens` with fp16-era code +
GradScaler (01 F-20) — the two arms' AMP semantics diverge before any op does.
Micro-tests were run under **CPU bf16 autocast** (cast list is narrower than
CUDA's, but *dtype-plumbing* failures — the ones that crash — are shared).
Static CUDA-list reasoning is marked accordingly.

1. **HARD FAIL — `core/concept_layer.py:230-231, 259-261 (and any future
   masked put): `index_copy` with fp32 self ← bf16 source.** Under CUDA autocast,
   `write_q_proj` (nn.Linear) emits bf16 → `q_n/new_key/new_val` bf16, while
   `self.concept_keys/concept_vals` are fp32 buffers.  Reproduced exact error
   (`index_copy_(): self and source expected to have the same dtype … Float …
   BFloat16`).  Fires only when the UCL write path opens (internal `_mature` ≥
   0.1/0.3 + `mat_gate ≥ 0.1` — train-effective gate floor 0.182 from ~step 300
   makes this *early-possible*, data-dependent on conf/birth thresholds): this is
   the strongest candidate for the A100 bf16 crash the user reported.
   Fix (B10): `.float()` the two index_copy sources (concept_layer.py:216-217,
   230-231, 259-261) — the read path then promotes cleanly; or wrap the UCL call
   in `torch.autocast(enabled=False)` like the mirror.
2. **HARD FAIL (cross-context) — `core/block.py:357 `torch.cat([conv_state,
   h_perm])` requires identical dtypes**, but `conv_state` dtype is *whatever the
   producing context was* (line 355 `dtype=h.dtype`), while the loop carries it
   across windows.  Mixed autocast boundaries — e.g. train.py's embed-only AMP
   (F-20) toggled per call, generation under autocast after a fp32 eval, or the
   planned B10 partial rollback — crash at the first carried window.  Same class:
   `stack.py:351-354` `_gs_velocity` `.to(device)` (dtype not normalized) and
   the fp32 mem-state vs bf16 trunk state asymmetry (currently benign because
   every consumer `.float()`s it — `block.py:445-448` — fragile, assert it).
3. **HARD FAIL (any bool-mask write) — `Tensor.index_put_((bool_mask,),
   bf16_src)` into fp32 storage** raises "Index put requires the source and
   destination dtypes match" (reproduced in micro-test).  No current forward-path
   hit found (memory_bank's L2 write uses integer-slot assignment which *does*
   cast legally — verified), but it is the failure mode for any B10-era patch that
   writes autocast outputs into the fp32 banks/EMAs by mask.
4. **fp32 anchors to preserve (they are load-bearing for bf16):**
   `block.py:442-448` (VSA scan forced fp32/fp64, exact carry verified in
   isolation), `block.py:492, 542, 563` autocast-disable for mirror/VPM/spectral
   (mirror's exp/log/sigmoid ladder and the DCT would under/overflow otherwise),
   `_scan_chunk` fp64 reciprocals (A1/A2).  Mirror is a **non-public anchor**: the
   block calls `self.mirror(..., h.float())` — any caller that calls
   `GroupedCognitiveMirror.forward` directly (tests, future drivers) silently
   loses it; make the anchor internal (a `torch.autocast(enabled=False)`
   decorator on `GroupedCognitiveMirror.forward` itself).
5. **Precision (not crash) under bf16:**
   (a) `losses.py:144-155` power iteration: autocast casts `bind_W @ v` to bf16
   ⇒ σ̂ quantized to ~0.4 % — stable-rank penalty still meaningful, but reseed
   + 4-step PI at bf16 is noisier; pin `with autocast(enabled=False)` around the
   PI block.  (b) `logit_cache.bit_profile` sums of ±tanh over V=65536 in bf16:
   column sums reach ~10²·√V ≫ bf16 8-bit mantissa — accumulation granularity
   ~2–4 units against per-position signal 0.2 (2a's 42:1 floor/signal) → the
   cache profile loses *all* discriminative content earlier than the fp32 path
   already does; promote to fp32 (`codes_t` is fp32; einsum casts it down — see
   2a F2A-08 for the coherence half).  (c) `mirror` anchors keep signal-normal EMA
   math fp32 ✓.  (d) `AGC` (`adaptation.py:437-457`) norms are fp32 (params fp32)
   ✓; `p.grad.mul_` in-place on fp32 grads ✓.
6. **torch solve-family / removed-op scan (core/ + scripts/):** zero hits for
   `torch.solve|symeig|ger|lstsq|svd|cholesky|matrix_rank|histc`; the only
   `linalg` call is `torch.linalg.qr` at `embedding.py:181`, build-time fp32 →
   **no CUDA-bf16 linalg gap**.  `.type(...)` casts: none.  Hardcoded
   `torch.float32` in reductions: `bind._manifold_read`/`_zeck_weight` build
   fp32 tensors then multiply into h-dtype (`bind.py:689-690` — `.to(h.dtype)`
   missing on `decay`? it is `dtype=hp.dtype` ✓ fine) — checked, benign.
7. **bf16 + RNG:** `torch.randn_like(i_gate)` (block.py:401-403) is CUDA-gen
   seeded (01 F-21) — unchanged by AMP, but B10 will change numerics anyway;
   `randperm` in manifold (off).  No determinism claims survive bf16; do not add
   new ones.
8. **GradScaler leftovers:** train.py still builds a fp16-era scaler path
   (`scaler.step`, `unscale_` at 603-609) while the notebook sets
   `scaler = None`; if B10 enables `_USE_AMP` in the notebook, train.py's
   `use_amp` branch (fp16 semantics!) becomes the odd one — retire it (01 F-20).

---

## 5. Not verified + handoffs

Not verified here:
1. CUDA-side bf16 execution (host is CPU; §4 entries are micro-test + cast-list
   reasoning, with the index_copy error reproduced under CPU bf16).
2. Any *trained* twin_free production state (best.pt is the legacy K=32 step-1045
   artifact and was mid re-download — same caveat 2a recorded; the pen/decay
   operating-point numbers are at INIT + real ids, and 1045-step drift of
   `w_d_pen`, `b_d`, `gamma_surprisal` was not measurable on it — the ladder
   verdict F2B-02 should be **re-measured on the first real checkpoint after
   B10**).
3. The generation/streaming path (`live_inference.py`, `generate.py`,
   `process_with_cache`) — the `traj_state`-via-state-tuple lerp branch
   (`block.py:379-380`) executes only there; not exercised.
4. 43×-heritage provenance archaeology (historical claims), only the current-code
   measurement.
5. multi-seed: all dynamics numbers are seed-42/7/1/3/5 single runs (closed-form
   dominated; spread across seeds for the table numbers is below the orders of
   magnitude at stake).

**→ Agent 3 (memory/banks/reasoning/UCL):**
- F2B-02's pen unit bug is *the mirror's* quantity (`pred_error_norm`,
  mirror.py:491 — norm over (G,k), magnitude √(G·k)·RMS) consumed by
  block-decay, block-writes, UCL u_gate (F2B-06) and AdaptiveController diff —
  pick ONE convention (per-position RMS) at the producer; it re-tunes the ladder,
  the birth gate and the write amplitude simultaneously.
- UCL bf16 index_copy (F2B-§4.1) lives in your file; the fix changes
  concept_layer buffers' dtype policy → coordinate with the L2-buffer doctrine
  (B7) so the "storage-as-buffer" pattern doesn't re-grow dtype landmines.
- Reasoning cross-step chain: confirm the F2B-08 return-value read (stack.py:659
  discards `(buf,count)`); the fix is a 3-line change on the boundary you own
  (return them like the non-adaptive branch does) — and then decide whether eval
  should see them (it must not: `_reasoning_attr` already isolates).
- mirror `_usefulness_temp`-permanently->0.1 masks the train/eval counter branch
  (F2B-06): if you ever honor `_usefulness_temp=0`, fix `n_eff` for eval first.

**→ Agent 4 (losses/control/watchdog):**
- F2B-04 (eval uses raw ramp, train uses max-with-readiness) — the watchdog's
  "arm at first val" and the eval metrics both sit on this gap; the same
  `mat_gate` divergence decides whether UCL/bank channels are *measurable* in eval.
- F2B-07: apply_tau_lr (post-clip in the notebook!) vs train.py inline ls-only +
  phase-scaling; llrd 0.9-vs-1.0; set_trust present in only one arm; gradalign
  weight divergence.  These are layer-dynamics *numerically*, not cosmetically.
- `layer_gate_*` aux channels log the raw ramp (§2.28): label them
  `matur_ramp_*` or log the effective gate — and reconcile with the F-10 clock
  (three clocks gate ls_mults/_usefulness_temp/maturation).
- Watchdog channels are all forward-magnitude (B2 doctrine ✓, §2.23/§2.27);
  `grad_mod` remains the single documented grad-magnitude input, EMA-relative,
  gate-internal only.

**→ Agent 5 (persistence/eval doctrine):**
- Add to the snapshot/reset set: `block._traj_state`, `mirror._cached_hp`,
  `mirror._cached_pred_error_norm` (measured forward-relevant, plain attrs —
  F2B-01/F2B-03), plus `_pi_v` (01 row 12) — or land `model.reset_document_state()`
  as 01 proposed, listing all six.
- Rotation order: `reset_cache()` clears `_traj_state` — the rotation block should
  call it (or the trajectory cache must be keyed by document epoch).
- optimizer by-name restore (F-11) still unwired in train.py; with 24 layers ×
  λ/LLRD role groups the blast radius of slot-shift is exactly the layer-dynamics
  knobs (b_d momenta landing on scale_w, etc.).
- `_step_count`/`_pm_step`/`_fwd_count` are persistent counters riding in
  best.pt for no forward-path reason today (inert) — candidates for
  `persistent=False` on the next state_dict format touch.
- When B10 flips `_USE_AMP`: run the §4.1 micro-test as a regression lock
  (`index_copy` dtype), and re-measure F2B-02 (bf16 `h` slightly changes σ(d_mod)
  via rounding only — the conclusion is dtype-independent, but the table numbers
  will move).

**Suite:** `python -m pytest tests -q` → **294 passed in 56.38 s** (tree clean at
analysis start and end; this audit added no repo changes beyond this document).

---

### Appendix A — repro manifest (all in `%TEMP%\opencode\`, seeds fixed)

- `repro2b_A.py`: ladder census; first (flawed, feedback-contaminated) impulse
  probe; twist isometry 1.19e-7; `_step_count` under eval; eval determinism;
  trajectory cache lifecycle (frozen W1 content; post-eval zeroing; d≥1 channel
  norms 0.0); hybrid-α per layer; bind gain/coherence.
- `repro2b_B.py`: linear-impulse methodology iterations (the chaos was *itself*
  a finding: stale `_cached_hp`/pen cross-call coupling → F2B-03); cross-chunk
  readout share; steady-cache Jacobian = 0.0; eval zero-trajectory; bind gate
  census; b_i/b_d targets; gradalign post-backward magnitudes.
- `repro2b_C.py`: buffer-neutralized impulse (r=0 all 8 scale×layer — the ladder
  kill); carried-share profile; mlp_ratio/prev_grad_norm/delta_var/gate_l1
  single-step census; `cache_grad_norms()` self-destruct demo.
- `repro2b_D.py`: per-channel decay capture (real CHILDREN ids) → the half-life /
  speed-up table (§2.5-8); i_gate operating stats.
- `repro2b_E.py`: pen operating point (8.1) & factor (0.5002); UCL grads/births/
  skips/u-gate/contra/private-mem; gradalign race battery (aux-full backward vs
  CE-only reference); cautious ÷√ρ toy; role-lr + optimizer census; L2-param
  existence check (fixed-status probe).
- `repro2b_F.py`: CPU bf16 full fwd+bwd smoke (passes; narrow cast list);
  dtype micro-tests (`index_copy` FAIL, slice-assign OK, `index_put_` FAIL);
  maturation ramp at 495/1045/2090/9000, global_ready ≈ 9250; two-copies
  apply_tau_lr census.
- `repro2b_G.py`: asymmetry init census; U8/W-grad liveness (window-1 vs
  window-2); 12-step real-data training loop (`_mlp_ratio`, gate_l1,
  mod_scale_mlp σ=0.667 pinned); layer_gate band at 495; reasoning liveness
  battery; eval `_cached_pred_k` absence.
- Suite re-run post-analysis: 294 passed.

### Appendix B — the one-paragraph dynamics model (for the non-quantitative reader)

Between embedding and logits there are three memory systems and the audit finds
two of them quietly disabled at the operating point: the VSA's four-timescale
reservoir, which the math says should carry a paragraph, in practice carries one
token — because the "surprise" signal feeding it is measured in units of
√(experts×dims) and pins every timescale to its own speed-limit (0.5 decay per
token), four identical fast leaks wearing a ladder's name; the trajectory spiral,
whose cross-position gradient survives exactly one window per document and is
then frozen, and after the first validation pass overwritten with zeros —
permanently, through a cache the eval-isolation machinery cannot see; and the
code-space logit cache, which (2a) never runs its inference half at all.  What
*does* run is the mirror ensemble on one-step-stale K-space, whose hold-out
contamination is now a measurable 0.8 % forward perturbation, and whose gates the
optimizer damps with forward magnitudes honestly — while the two training copies
(disabled eval gates vs readiness-floored train gates, λ-groups vs 0.9^l LLRD,
trust on/off, τ-LLRD after vs never) are three different models wearing one
checkpoint format.  The fixes are small and known: normalize pen at the source,
guard the trajectory cache with `self.training`, `.float()` two tensors for
bf16, and let one clock, one gate and one optimizer builder own the layer stack.
