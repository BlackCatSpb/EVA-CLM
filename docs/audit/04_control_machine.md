# 04 CONTROL MACHINE AUDIT

Agent #4 of the serial relay. Scope — **everything that decides WHEN/HOW the model
changes without being the loss itself**: `FailureDetector` (B4/B5 calibration, the
two-strike D6+ policy, cooldown, warmup freeze), `MemoryGovernor` (the M16 85/70
gradient-checkpointing hysteresis toggle), the M14 hard + M14b soft vetoes,
`LossBalancer` internal state (ema_ce / align history / last_cos), `LRController` +
`MirrorLRScheduler` (warmup, boost, plateau damping, readiness write-back, every
state attr and its save/load), `DepthController` (active depth vs maturation — two
independent schedules), `AdaptiveController` (the b_i/b_d lerp law and its c/τ
target vs the post-B10 pen reality), `aux_anneal`, the per-layer `ls_mult`
fast/slow-EMA machinery and its bounds, and the readiness pipeline
(pred_err → readiness EMA → global_ready → arm_ce → soft-veto). All claims are
against the **current tree** (HEAD 4aa0753, B6–B12 shipped; mission baseline
305 passed — re-verified here, §5).

Method.
- Every class in scope read line-by-line from the working tree:
  `core/training_control.py` (663 l), `core/lr_scheduler.py` (282 l),
  `core/adaptation.py` (402 l), `core/adaptive_controller.py` (194 l),
  `core/maturation.py` (149 l), `core/stack.py` forward/glue, `core/block.py`,
  `core/mirror.py`, `core/bind.py`, `core/losses.py`, `core/config.py`; the two loop
  copies `scripts/train.py` (866 l) and `notebooks/eva_colab.ipynb` cells 4/8/9/10
  (dumped to `%TEMP%\opencode\audit4\nb_cells.txt`).
- Executed probes live in `%TEMP%\opencode\audit4\`:
  `p1_detector.py` (alarm×eval×resume, arm_ce, non-finite, two-strike),
  `p4_lr.py` (adversarial MirrorLRScheduler/LRController sim on real classes),
  `p2_ckpt.py` (governor×checkpointing grad/CE equivalence), `p5_f306.py`
  (τ-ladder invariant + apply_tau_lr/AGC), `p6_depth.py` (DepthController
  persistence), `p3_veto.py` (hook/veto/arm_ce idempotency), `p7_leak.py`
  (eval→train plain-attr leak), `p8_temp.py` (`_usefulness_temp` resume
  discontinuity), `p9_data.py` (checkpoint `.data` double-apply + legacy detector
  pickle + ggeo recompute census). No repo file modified.
- Confidence tags: **VERIFIED-REPRO** (executed here), **STRONG-READ**
  (anchor-verified, deterministic), **SPECULATIVE**.
- Baseline: `python -m pytest tests -q` → **305 passed in 58.94 s** (clean tree).
- Conventions: `train.py:N` / `tc.py:N`(training_control) / `lr.py:N`
  (lr_scheduler) / `ad.py:N` (adaptation) / `ac.py:N` (adaptive_controller)
  / `mat.py:N` (maturation) / `stack.py:N` / `mirror.py:N` / `block.py:N`
  / `bind.py:N` / `losses.py:N` / `c10:N` notebook training-loop line.

State graph first (§2 table is its serialization). Every control-plane object is
mutated by one or more of seven event classes: **TRAIN step** (forward→losses→
balancer.backward→clip→opt.step→scheduler.step), **VETO** (hard/soft — skip
backward/scheduler), **ALARM** (watchdog.check→True), **EVAL** (snapshot→N forwards
step=None→restore→arm_ce/report_val_loss), **RESUME** (rebuild objects→load_state_dict),
**GOVERNOR** (cfg.gradient_checkpointing flip at log cadence), **ROTATION**
(document-boundary resets). Orphans = attributes mutated by TRAIN but absent from
best.pt, the eval snapshot, and the rotation reset simultaneously.

---

## 1. Findings (F4-…)

### F4-01 — HIGH — VERIFIED-REPRO — Gradient-checkpointing does NOT change the optimizer's gradient (CE bit-identical, ‖Δg‖/‖g‖=8.2e-4) but it DOES double-fire every in-forward `.data` controller write; the M16 governor silently changes the mirror's alpha self-regulation rate

The mission's §4 central question, measured cleanly (fresh model, snapshot/restore
of all params+buffers around each arm, identical RNG seed — `p2_ckpt.py`, `p9_data.py`):

- **Forward:** `dCE = 0.000e+00` for ckpt OFF vs ON (10.205717087 both). Checkpoint
  recompute replays RNG exactly, so the stochastic write-noise
  (`torch.randn_like(i_gate)`, block.py:420 — measured **3 calls / OFF forward**) is
  reproduced, not re-rolled. The CE the loss/veto/watchdog read is bit-equal.
- **Backward (what `optimizer.step()` consumes):** global relative gradient
  deviation **‖g_off − g_on‖ / ‖g_off‖ = 8.15e-4**; max per-param ABS deviation
  1.8e-2 on `layers.1.mirror.W_proj` (whose grad is O(10²), so ≈ 1e-4 relative).
  This is float-reassociation noise from the second forward, not a different
  update direction. **Verdict on the naive question: checkpointing does NOT
  silently change learning at the weight-update level** — a train step ON vs OFF
  applies the same AdamW/EVAAdamW delta to float noise.
- **The catch — controller side effects re-execute.** The block forward mutates
  state via **`.data` writes that bypass autograd's recompute-restore**:
  `mirror.py:480 self.alpha_diag.data.lerp_(alpha_target, 0.01)` and
  `mirror.py:489 self.alpha_diag.data.add_(novelty_push)` (the `alpha_novelty`
  anti-collapse push). Measured (`p9a`, 5 steps, `lr=0` so only the `.data` path
  moves alpha_diag): **OFF |Δ| = 3.98e-3, ON |Δ| = 7.97e-3 → ratio 2.00** — the
  lerp+push fire **once per recompute, not once per step**. With
  `_alpha_override = 0` (post-warmup default) the lerp branch is live every training
  step (`override < 0.1` gate, mirror.py:473), so in a real post-warmup run with
  `gradient_checkpointing=True` (the **config default**, config.py:380, and cell 4
  keeps it on) `alpha_diag` self-regulates at **double the nominal rate**.
- Recompute census (`p9c`, ckpt ON, 2-layer mini): one `balancer.backward` re-runs
  the block forward **4×** (CE autograd.grad + aux + bypass + the real backward all
  traverse the checkpoint boundary); one `grad_geometry` logged step re-runs it
  **10×** (CE + 8 terms). So on log steps the `.data` writes stack ~6–10 deep.
- **Governor coupling.** M16 flips `cfg.gradient_checkpointing` True/False at
  85 %/70 % reserved-VRAM (train.py:669-674, c10:881-886). Because the flag is read
  at forward time (stack.py:534 `self.cfg.gradient_checkpointing and self.training`),
  the toggle is immediate — and every flip **changes alpha_diag's effective learning
  rate** (and the b_i/b_d-free mirror self-tuning generally) with zero log line beyond
  `[memgov] VRAM … -> checkpointing ON/OFF`. Two arms identical except VRAM history
  diverge in the mirror gate policy. This is the same dual-drive family as F3-13
  (alpha: lerp + push + aux loss + Adam), now with a **fifth** driver: the
  checkpointing governor re-executes the lerp.

Severity: HIGH — it is a control-plane actuator (the governor) that silently changes
another control law (mirror alpha adaptation) as a side effect of a VRAM decision,
and it is default-on in the production cell. Not caught by any B12 test (B12's
`_ggeo_freeze` only guards `_step_count`, bind.py:400, which is inert; it does not
and cannot guard `.data` lerps that legitimately must run once). Fix candidates
(NOT applied): run the alpha lerp/push **outside** the checkpointed region
(hoist to stack after the block returns, driven by a cached `alpha_target`), or
guard the `.data` write with `torch.utils.checkpoint` "is-recomputing" detection,
or make the lerp idempotent (set, not accumulate). Locks: `test_b12_regressions.py
:62 test_ggeo_freeze_blocks_step_count_side_effects` locks `_step_count` only.

### F4-02 — HIGH — STRONG-READ + numeric repro — `arm_ce()` re-anchors the soft-veto CE baseline at every eval; combined with the M14b veto cadence this puts a `1.6×seed` soft ceiling in play that a single easy post-eval batch can lower for ~340 steps

`arm_ce()` (tc.py:319-328) sets `ce_armed=True` **and pops `_stats['ce']`/`_viol['ce']`**.
It is called once per successful eval (train.py:690, c10:989), i.e. **not just the
first eval**. Consequences:

- After every eval the CE fast-EMA restarts from the *next single batch's* value
  (`_observe` seeds `[v,v,1,v,0,0,0]` on a missing key, tc.py:248-251).
- The M14b soft veto threshold is `_fc[0]*(1+4·rel_margin) = fast_ema*1.6`
  (train.py:485, c10:774). Measured (`p1 T3`): healthy session at CE≈12 has
  soft-thr 19.2; after `arm_ce()` + **one** easy 9.0 batch the threshold drops to
  14.4, and a normal 15.5 batch is now vetoed (`veto@15.5? True`). It takes
  **~339 fast-EMA steps** (half-life 69, but reaching 11.9 from seed 9.0 measured
  339) for the band to reopen — so a single low-CE batch right after eval transiently
  tightens the escalation gate. This is 01 F-08's "self-healing false-veto window",
  but the driver is arm_ce cadence, not the first-eval arm.
- `arm_ce` is **idempotent** as a state op (p3c: three arms in a row all leave
  `ce_armed=True`, `'ce' not in _stats`) — no compounding. The hazard is purely the
  per-eval re-anchor.
- **The CE channel is dead until the first eval**: `p1 T1/T5` needed an explicit
  `arm_ce()` before any sustained CE divergence would alarm (the `name=='ce' and
  not ce_armed` guard, tc.py:313, silently warms CE forever pre-arm). train.py in
  the M8 era "never armed CE" (01 §3 D6 row); it arms now (train.py:690) — verified
  present in both copies.

Fix (NOT applied): arm once (guard the pop with `if not self.ce_armed`), or
re-anchor the *slow* baseline too rather than dropping to a single-batch seed. Locks:
`test_product_invariants.py:857-870` locks re-bootstrap exists; none lock the veto
that consumes it (01 already noted). **→ still open, agent 4 owns the policy.**

### F4-03 — MEDIUM — VERIFIED-REPRO — `FailureDetector.load_state_dict` crashes on a legacy (pre-B4) 4-wide stats pickle; the one on-disk `best.pt` is pre-B2 era and its detector stats are the wrong width

`_observe` unpacks `fast, prev, n, slow, dvar, ph, phmin = s` (tc.py:253) and later
`s[0],s[1],s[2],s[3],s[4]=…` (tc.py:260), assuming a **7-element** list. A checkpoint
whose `detector['stats']['ce']` is the old 4-element `[ema,var,prev,n]` layout (any
best.pt saved before the B4 slow-EMA / B5 Page-Hinkley fields) raises
`IndexError: list assignment index out of range` on the **first post-resume armed
check** (`p9b`). This is the exact F3-07 pattern (attribute/schema older than the
pkl) that B12 patched for `_lr_damp_steps` but **not** for the detector — the
detector has **no version tag** in its payload (state_dict keys: recover_count,
ce_armed, cooldown, viol, stats, last_viol_name only — no `type`/`schema`).
`load_state_dict` (tc.py:227-236) copies whatever list widths it finds.

Real-world reach: 02a/02b both reference the single on-disk
`checkponts/best.pt` (step 1045, **legacy K32, pre-B2**) as the resume witness;
resuming it (the documented FORCE_FRESH=False path) loads its `detector` verbatim.
If that pickle predates B4 the very first eval+check crashes the run. Fix (NOT
applied): width-normalize in `load_state_dict` (`s = (s+[0.0]*7)[:7]`) or gate on a
`sd.get('stats_layout', …)`. Needs-lock. **→ persistence/legacy schema to agent 5.**

### F4-04 — MEDIUM — STRONG-READ + probe — `_usefulness_temp` (persistent=False) is written by the scheduler **only during warmup/blend**, so a resume that starts past warmup leaves the mirror's usefulness softmax at the constructor default 2.0 instead of the ~0.51 the continuous run converged to

`MirrorLRScheduler.step()` sets `layer.mirror._usefulness_temp` **only** in the
`self._step < warmup_end + blend_steps` branch (lr.py:193); the post-warmup else
branch (lr.py:194-241) never rewrites it. `_usefulness_temp` is
`register_buffer(..., persistent=False)` (mirror.py:378), so it is **absent from
`model.state_dict()`** and therefore from `best.pt`. Measured (`p8`): continuous
run to step 200 → temp 0.509 (frozen thereafter); fresh model + loaded scheduler
`_step=200` → temp **2.000** (constructor default, never touched because step() is
already past warmup). Consumer: mirror.py:743 `if float(_t_ov)>0.0: temp=_t_ov`
→ the 2.0 **always** overrides the intrinsic schedule (`clamp(3·e^{−2},0.3,3)=0.406`
at large n). Net: **any post-warmup resume runs the usefulness→gate selector at
temp 2.0 (~5× softer than the 0.41 the pre-warmup→blend path produced), changing
expert selection distribution after every restart.** Same for `_alpha_override`
(persistent=False; stays 0 post-warmup which happens to be the correct resting
value, so alpha is benign — only the temperature diverges). The buffer is covered by
the eval snapshot (it IS a named buffer) so eval isolation is fine; the defect is
purely the **cross-session** path. Fix (NOT applied): re-assert the blend-end temp
in the else branch, or make the buffer persistent, or store it in the scheduler
state_dict. **→ orphan-state (buffer not in best.pt) → agent 5.**

### F4-05 — MEDIUM — STRONG-READ — Three independent clocks (loop `step`, `scheduler._step`, watchdog/DepthController counters) advance on disjoint event sets; vetoes and empty-evals decouple warmup, maturation, LR-adapt and depth-unlock

- **Veto skips `scheduler.step()` but not the loop step** (01 F-10 unchanged): hard
  veto `continue`s before lr/scheduler (train.py:455, c10:738), soft veto likewise
  (train.py:490, c10:780). So during a sustained veto storm `scheduler._step`
  **freezes** (LR stays at whatever warmup fraction it reached) while the loop
  `step` (→ `maturation.step_gate(step)` mat.py, watchdog `_cur_step`) keeps
  advancing: the trunk's **maturation ramp and the alarm's warmup-freeze release
  (B5) advance while the LR never finishes warming**. The two can only reconverge
  when vetoes stop.
- **Watchdog `_cur_step` = loop step** (tc.py:332), so `eval_block = _cur_step <
  warmup` (tc.py:268) and the B5 `_warm_frozen` release (tc.py:333-341) key off the
  **loop** step, not the LR step. A run stuck vetoing through step≥warmup releases
  the freeze and re-arms PH with `self._viol={}` reset while LR is still ~linear
  mid-warmup (legitimate monotone drift the freeze exists to ignore) — the B5
  protection is scoped to the wrong clock. (`p1 T1/T2` show the release itself is
  clean: ph zeroed, 0/200 first-sample fires — the residual risk is only the
  clock mismatch.)
- **DepthController warmup is its OWN constant** (`warmup_steps` ctor default 2000,
  ad.py:100, called with the *loop* step at train.py:457/c10:739) and its
  plateau-eval path only fires on `val_loss is not None`. Empty-pool eval:
  train.py now guards on `math.isfinite(val_loss)` (B12, train.py:682) so it does
  NOT call `depth.update(step, NaN)`; the notebook passes `depth.update(step, None)`
  on no-data (c10:1016) which is also inert (early return ad.py:126). Both safe —
  **but** a hard veto also skips `depth.update`? No: depth.update(step) is at
  train.py:457 / c10:739, i.e. **after the hard-veto continue**, so depth plateau
  detection is skipped on vetoed steps too (it only accumulates on learned steps).
  Minor but another clock asymmetry.
- **`scheduler.set_step()` fallback** (train.py:332, c10:537, used when a ckpt has
  no scheduler state — pre-fix checkpoints) sets `_step` to the loop step but
  `_ls_*`/`_tau_*` baselines to None → they re-bootstrap against the resumed model.
  03 F3-26 verified this is ramp-continuous and benign; re-confirmed (var_mult
  defaults 1/1).

Fix (NOT applied): drive watchdog/depth/LR-adapt from a single
`non_vetoed_learning_step` counter, or advance `scheduler.step()` on veto too (LR
clock = loop clock) and document which is canonical. **This is agent 4's call per
02b §5 ("Agent 4 owns the resolution").** Locks: none.

### F4-06 — MEDIUM — STRONG-READ + probe — `DepthController` plateau integrator is NOT in best.pt; only `active` survives, so a resume resets `_last_depth_step=-1e9` and the next plateau eval unlocks with **no eval-interval spacing**

`DepthController` (ad.py:87-150) keeps `_val_ema/_val_var/_prev_val/
_last_depth_step`. **None are serialized** — the envelope carries only
`active_depth` (train.py:704, c10:1002). On resume a fresh controller is built
(train.py:334, c9:549) and `set_depth(_saved_depth)` restores `active` only
(ad.py:118-121). Measured (`p6`): after resume `_last_depth_step=-1000000000`, so
the guard `(step - self._last_depth_step >= self.eval_interval)` (ad.py:143) is
trivially true on the **first** post-resume plateau, unlocking `init_k+inc` without
the intended one-unlock-per-eval_interval rate limit. In a frequent stop/resume
Colab session this can walk depth up faster than plateau cadence intends (bounded
by max_depth, so not catastrophic). The plateau **slope/variance** state is also
lost (a genuine restart of the diminishing-returns estimator). Two-schedule note:
depth progression (parameter unfreeze, plateau-driven, ad.py) and maturation ramp
(wake-up gate, time-driven, mat.py) are fully independent schedules — depth is
`val_loss`-gated, maturation is `step`-gated, and neither reads the other; the only
coupling is that `set_active_depth` (ad.py:76-84) zeroes `requires_grad` while
maturation still ramps frozen layers' gates. Fix (NOT applied): persist the four
integrator scalars in the envelope (agent 5); needs-lock.

### F4-07 — MEDIUM — VERIFIED-REPRO — `MirrorLRScheduler._update_ls_mult` divides by the fast-EMA with no `max(r,ε)` floor → ZeroDivisionError on an exactly-frozen `log_scale` layer; and a size-1 `log_scale` yields NaN variance silently pinned to the 2.0 boost bound

`r = self._ls_fast[i] / max(self._ls_slow[i], 1e-10)` guards the **denominator**
but `1.0 / r` (lr.py:104) is unguarded. Measured (`p4 S6`): a genuine bootstrap with
`var(log_scale)==0.0` exactly (constant per-expert scale — reachable once a layer's
`log_scale` collapses, e.g. `div`/`ls_reg` driving all experts equal, or a frozen
uniform-init layer) → first real `_update_ls_mult` after warmup raises
**ZeroDivisionError** and escapes both training loops (only KeyboardInterrupt is
caught). This is the same "one-line fix, dead run" class as F3-07. Additionally
`torch.var()` of a size-1 tensor is **NaN** (p4: `var() of size-1 tensor: nan`) and
`max(lo, min(hi, 1.0/nan))` silently evaluates to `lo/hi`-boundary (min with nan
returns the non-nan operand) → a degenerate layer is pinned to the 2.0 **boost**
rather than 1.0. Reachability of the NaN leg is low (log_scale is (G,k), G,k>1 in
production); the exact-zero leg is the real hazard. Fix (NOT applied):
`1.0/max(r, 1e-6)` and a `nan_to_num` on vals. Needs-lock: NONE (no ls test
exercises a constant-variance layer).

### F4-08 — MEDIUM — VERIFIED-REPRO — `apply_tau_lr` applied **after** AGC (notebook order) multiplies the per-param gradient that AGC just bounded, breaking `‖g‖ ≤ c_eff‖θ‖` by up to 7.7× on the shallow layer at the ls bound; `ls_mult` is applied to the grad plane only, never to `param_groups['lr']`

`GradientClipper.clip` (ad.py:437-457) enforces per-param `‖g‖ ≤ c·(τ_ref/τ_l)^γ·‖θ‖`
(c=0.1, attach-built τ map ad.py:416-431). In the notebook the order is
clip-then-`apply_tau_lr` (c10:838 → :842); in train.py there is NO `apply_tau_lr`
call at all (F3-03/D-04 confirmed). Measured on a real `GradientClipper`+`apply_tau_lr`
(`p5`), starting from grads pinned at each layer's c_eff:

| τ_l | post-CLIP ‖g‖/‖θ‖ (=c_eff) | ls·lr_mult applied | post-apply ratio | ×own AGC bound | ×base c=0.1 |
|---|---|---|---|---|---|
| 8   | 0.386 | 2.0 · 3.86 | **2.986** | **7.73×** | 29.9× |
| 64  | 0.100 | 1.0 · 1.00 | 0.100   | 1.00×     | 1.0×  |
| 512 | 0.026 | 0.5 · 0.026| 0.0033  | 0.13×     | 0.03× |

So the AGC guarantee (and with it the "23 % of tensors are decided by AGC not the
aux bound" ledger line 03 F3-19) holds **pre**-`apply_tau_lr` only in the notebook;
post-scale the shallow block is governed by `ls·lr_mult`, not by c. Note this is a
**grad-plane** effect (a uniform per-layer positive scale preserves the aux:CE
*direction* ratio, so it is not a balancer-honesty break) — it is an LR-magnitude
break (the per-layer effective step size is multiplied by ls·lr_mult with
F3-09's caveat that Adam mostly cancels a *constant* per-step scale; here ls is
trend-varying so it does bite transiently). The mission's specific question —
"does mult feed BOTH optimizer lr and apply_tau_lr consistently?" — answer: **`ls_mult`
never touches `param_groups['lr']` in either copy** (only the single global
`mult` does, lr.py:255-257); it is a pure grad multiplier, and the two copies feed
it differently (train.py inline base×ls / mirror×clamp(mir·ls)/mir pre-clip
train.py:594-607; notebook `apply_tau_lr(ls)` post-clip c10:842). **Not consistent.**

### F4-09 — MEDIUM — VERIFIED-REPRO — the `LRController`/`MirrorLRScheduler` boost path has a dead `lr_boost_max=0` footgun, `orig_lrs` positional restore still accepts a reordered same-count list, and `_loss_lr_factor` "warm-restart" is per-EVAL not per-step (03 F3-17b re-confirmed and sharpened)

Adversarial sim through the **real classes** with a fake model (p4):
- **Boost bounded**: `last_mult ≤ lr_boost_max+ε` held (max observed 1.001); no
  ratchet-to-ceiling (self-limiting via the EMA baselines, as the comment claims).
- **Damp floor reachable & one-way-ish**: single +50 % spike → factor 0.5; steady
  improvement past `best·0.98` → full restore (p4 S2: 0.5→1.0 in 4 evals). From the
  0.05 floor, plateau-zone recovery is `+1/lr_warm_restart_tau` **per
  `report_val_loss`** (lr.py:160-162) = **190 evals** to reach 0.95 (≈ 199 000
  steps at eval_interval 1045) unless a genuine `val<best·0.98` fires — confirms
  03 F3-17(b): the "reaches 0.95 in ~600 steps" comment is false; it is ~600
  *evals*.
- **Interlock**: damp and boost compose multiplicatively (`mult = m·_loss_lr_factor`,
  lr.py:239) with `m` clamped to ≥0.2 and factor to ≥0.05 → floor `0.2·0.05=0.01×`
  base; both lift independently. No deadlock found.
- **Resume continuity**: `state_dict`→`load_state_dict` mid-damp reproduced the LR
  sequence **exactly** (p4 S4, `all(|a-b|<1e-12)`), **except** `_lr_damp_steps` is
  NOT in the payload (still a dead counter — never read, F3-20) and `_ls_fast/slow`
  only persist when `_ls_enabled`. B12's F3-07 fix (`hasattr` guard lr.py:140) is
  VERIFIED working: a legacy scheduler pickle missing the attr no longer crashes the
  first plateau eval (p4 S4 "plateau eval right after resume: OK"). **B5/B12 status:
  the damp-resume crash IS fixed; the `_lr_damp_steps` orphan remains (dead).**
- **`orig_lrs` positional restore** (lr.py:293-299) accepts any same-LENGTH list —
  reordered group LR's load silently (p4 S7; my groups were equal-valued so the demo
  under-shows, but the code path is length-only). Same class as F3-08 for the
  optimizer; agent-5 persistence.
- **`lr_boost_max=0` footgun**: config.py:171 advertises "(0=disable boost)" but
  `m = min(m, boost_max)` (lr.py:235) with boost_max=0 → **every group LR = 0**
  (p4 S3b). "Disable boost" must be `boost_max=1.0`; 0 is a run-killer. Not the
  live value (2.0) but a landmine in the help text.

### F4-10 — MEDIUM — VERIFIED-REPRO — the `LossBalancer` `backward()` **bypass-only** early-return path is UNBOUNDED (51× in repro) and leaves the gradalign CE-target poisoned by the bypass gradient (03 F3-04b half, still open)

When `aux_dict` contains **only** `BYPASS_AUX` terms, `backward()` runs
`sum(bypass.values()).backward()` raw at tc.py:644-646 / 657-658 **before** the
sign-mask/clamp block, and before `_ga_record` is toggled (it is set False at
:663 only on the aligned path). Measured (p3b): a `gradalign`-only aux scaled 100×
lands at **51× ‖g_CE‖** on the parameter — the "aux bounded by ‖g_CE‖ by
construction" claim (docstring tc.py:17, class header) is **false on that path**.
Unreachable with today's producers (pred/gate_l1/diversity always emit tensors, so
`aux_tensors` is non-empty whenever gradalign exists) → latent, one config edit from
live. The mission's specific hook-staleness question — "does the MIRROR/GRADALIGN
hook leave `_ga_record` stale (last backward's grad targets carried into next step)?"
— answer (p3a): a **veto** skips `balancer.backward` entirely, so `_ga_record` stays
True and the block's `_gradalign_tgt` from the **last learned** backward is carried
into the next step unchanged (one-step-stale is by-design; veto storms make it
N-step-stale). The bypass-only path is the *unbound addition*, the *separate*
half of F3-04. Fix (NOT applied): clamp bypass on the early-return too (03's
"one-line" — NOT taken in B12). Needs-lock.

### F4-11 — MEDIUM — VERIFIED-REPRO — eval→train leak via **plain-attribute** mirror caches survives the snapshot/restore contract; the block treats them as forward inputs (02b F2B-03, still unfixed, now with the concrete control-plane consequence)

`snapshot_runtime_buffers` restores named buffers + `_last_bus`/`_intent_stream`
only (stack.py:1035-1043). But `mirror._cached_pred_error_norm` and
`_cached_hp` are **plain attributes** written even in eval (mirror.py:511-512) and
read as forward inputs by the block: `pen = mirror._cached_pred_error_norm`
(block.py:345-346) drives the VSA **write gate** `igate_logit +=
gamma_surprisal·pen` (block.py:405-406), the **decay penalty** `pen_decay_factor`
(block.py:427) and (via `hp_cached`) the per-expert **write modulation**
(block.py:441-449). Measured (`p7`, 2-layer mini): a hold-out eval batch sets
`_cached_pred_error_norm` to mean **0.359**, and the **next training forward's
output shifts by rel-norm 0.39** despite full snapshot/restore. The eval batch's
prediction-error level therefore directly steers the first post-eval training
window's memory write rate and decay. (The rel-0.39 is inflated by the fresh-model
cold start — with `_cached_hp=None` initially, activating the write-modulation path
at all is the jump; in a steady run the delta is the 02b-measured ~0.8 % but it is
**data-selection bias**, not zero.) Fix candidates unchanged from 02b: add
`_cached_hp/_cached_pred_error_norm/_pi_v` to the snapshot `__attrs__`, or don't
write them in eval. **→ agent 5 (snapshot doctrine) + open control finding.**

### F4-12 — LOW/INFO — VERIFIED-REPRO — watchdog two-strike + cooldown are consistent post-resume; B5 freeze release has NO first-sample roulette on the resume path

`p1 T1` (the mission's attack surface #1): alarm fires at step 1352, `recover_count=1,
cooldown=50, viol={}, ce_armed=True, last_viol_name='ce'` saved. Resume loads a
fresh detector (train.py:341 + load_state_dict :346): `_warm_frozen=True,
_cur_step=0, cooldown=50, viol={}, ce_armed=True`. First post-resume `check(step=1400)`
releases the warmup freeze, **zeroes every channel's ph/ph_min** (tc.py:340-341) and
returns False; across **200 seeded first-sample variants zero instant-fires**. So the
B5 re-arm is safe on resume (the loaded `cooldown` correctly suppresses triggers for
the remaining 50 checks; `_warm_frozen` reset is the right default). Two-strike
(p1 T5): strikes at 1602→1655, **gap 53 ≥ cooldown 50** (min_consecutive=3),
recover_count=2. `arm_ce` during cooldown (c10:989 + tc) does **not** clear the
cooldown (p5 tail) — only pops `ce` stats. NaN/+inf/−inf CE all force an alarm via
the non-finite bypass (tc.py:363) but **−inf still passes the soft veto** (`−inf > thr`
is False) — the stats are left clean (observe skipped) so it cannot poison the EMA
(p1 T4). The alarm-vs-veto interaction is internally consistent; the D6+ stop on the
loop side is train.py:509 `sys.exit(2)` vs notebook `_alarm_stop`+break (c10:801) —
same authority, different exit mechanics (01 F-08 family, still divergent).
- **Blind spot (p1 T5 tail)**: a *constant* runaway level that is also the
  `_observe` **seed** (i.e. divergence that begins at/just after an
  arm-ce/re-bootstrap boundary and never moves) **never alarms** — the relative rule
  needs `value > slow·(1+margin)` and a self-seeded slow equals the value. The
  hard-veto (2·lnV=22.18) catches gross garbage; a CE plateau at ~20 that starts flat
  post-eval is invisible to the sensor until it drifts. Information-theoretically
  acceptable (a truly flat 20 may be healthy), but worth logging as a sensor
  limitation.

### F4-13 — LOW/INFO — ADAPTIVE controller is state-pure but its equilibrium is mis-normalized vs the crushed ladder (F2B-10/F3-06 answered; B12 did NOT touch, confirmed)

`AdaptiveController` (ac.py) is **entirely `@staticmethod`** — no instance state,
no save/load, nothing to persist (good: no orphan states there). Its outputs are
recomputed from mirror buffers each forward and land in the b_i/b_d **Parameters**
via the stack lerp (stack.py:290-306). Mission question — *the c/τ target vs the
post-B10 pen reality, quantify the F3-06 invariant miss* — done (`p5`):

- The b_i law solves `i_gate = c/τ_nom` EXACTLY (B3-fix, ac.py:121-127,
  `c=0.166·32=5.312`), i.e. it assumes a memory **lifetime = τ_nominal**.
- Post-B10 the per-token pen arm contributes `f=exp(−0.0916)=0.9125` independent of
  `d_s` (03 F3-06). Combined `decay = exp(−1/τ_nom)·0.9125` → **τ_eff**: L0 4.6
  (nom 8), L11 9.1 (nom 54), L23 **10.6** (nom **431**). So every nominal τ beyond
  ~11 tokens is unreachable; the ladder is crushed at the slow end (8× milder than
  pre-B10's ×1013 but same mechanism).
- The design invariant `‖M_l‖=i_gate·‖h‖·τ_l=const` becomes `i_gate·‖h‖·τ_eff`:
  the ratio actual/design at init is **0.58 / 0.17 / 0.025** for L0/L11/L23 — the
  slowest layer carries **~1/135** of its intended memory budget. The c/τ write law
  therefore **over-writes shallow and under-serves deep** under the pen arm.
- **The b_i/b_d target VALUES themselves are NOT mis-scaled by B10** (targets are
  pen-blind: `b_d=2+3·lf .. b_d_max` and `b_i=softplus⁻¹(c/τ)` — 03 F3-06 answer
  "do not re-tune b_d" re-confirmed). The miss is that the *lives they encode* no
  longer equal the *lives the pen delivers*. B12's documented decision ("bounded
  redesign = B12/B13 ladder package", commit 9f544ee / block.py:411-418) is the
  correct home; **B12 did NOT take it** (commit message: "NOT taken: F3-03 … +
  F3-06 … queued for the B13 ladder package") — **verified still open, and correctly
  deferred.**

---

## 2. Orphan-state inventory

Legend — **train**: mutated on a learned step; **eval**: mutated during an eval
forward; **best.pt**: rides in the checkpoint; **snap**: covered by
`snapshot_runtime_buffers`; **rot**: cleared at document rotation; **verdict**:
OK / ORPHAN (train-mutated, in none of best.pt/snap/rot) / DIVERGE
(save/load asymmetry) / INERT (mutated, feeds nothing).

| attr | owner | train? | eval? | best.pt | snap | rot | verdict |
|---|---|---|---|---|---|---|---|
| `_cur_step` | FailureDetector | Y | skip-check | **N** | n/a | n/a | DIVERGE: set from loop step each check; not saved → resets 0 on resume, but B5 release re-derives correctly (p1 T1). OK. |
| `_warm_frozen` | FailureDetector | Y (releases) | — | **N** | n/a | n/a | OK: default True is correct resume init; release path zeroes ph (p1 T1). |
| `_stats` (7-wide) | FailureDetector | Y | Y (obs) | **Y** | n/a | n/a | OK for 7-wide; **F4-03 crash on ≤6-wide legacy pkl.** |
| `_viol`,`_cooldown`,`ce_armed`,`recover_count` | FailureDetector | Y | arm pops ce | **Y** | n/a | n/a | OK; consistent post-load. |
| `_last_viol_name` | FailureDetector | Y | — | Y | — | — | OK (debug). |
| `alarm_strikes` | loop-local | Y | — | **N** | — | — | **ORPHAN**: not in envelope → resume forgets "warn once"; a resumed run gets a fresh free warn. (D6+ re-arms strikes=0.) |
| `ema_ce/ema_A/ema_aux/align` | LossBalancer | bal only (dead) | — | Y | — | — | INERT: align-mode never updates them (F3-25); `align` restored but ctor value already True → no-op. |
| `last_cos` | LossBalancer | Y | — | **N** | — | — | ORPHAN (diagnostic; no reader) — 03 F3-25. |
| `BYPASS` early-return grads | LossBalancer | Y | — | n/a | n/a | n/a | **F4-10** unbound 51× on bypass-only path. |
| `_step` | MirrorLRScheduler | Y | — | **Y** (`step`) | — | — | OK; veto does NOT advance it (F4-05 clock skew). |
| `_tau_var/_mag/_1malpha/_gate_var` | MirrorLRScheduler | Y | — | Y | — | — | OK; re-bootstrap on set_step fallback (benign). |
| `_loss_lr_factor`,`_best_val_loss`,`_val_ema`,`_val_improving` | MirrorLRScheduler | report_val | eval report | Y | — | — | OK; continuity verified exact (F4-09). |
| `_lr_damp_steps` | MirrorLRScheduler | Y | — | **N** | — | — | INERT (never read) + legacy-pkl guarded by B12 hasattr (F3-07 fixed). |
| `_orig_lrs` | MirrorLRScheduler | read | — | Y (len-guarded) | — | — | **DIVERGE**: positional, reordered same-count accepted (F4-09). → agent 5. |
| `_ls_fast/_ls_slow` | MirrorLRScheduler | Y (if ls on) | — | cond (only if `_ls_enabled`) | — | — | OK; but **n-grow resume → IndexError** (p4 S6d) and exact-0-var → **ZeroDivisionError** (F4-07). |
| `_ls_mult` | MirrorLRScheduler | Y | — | N (derived) | — | — | OK; train.py reads it via `getattr(scheduler,'_ls_mult')` → None during warmup → ls block skipped (train.py:594). |
| `_alpha_override` (buf) | mirror | set by sched | snapshot | **N (persistent=False)** | Y | Y | OK post-warmup (resting 0 = ctor). |
| `_usefulness_temp` (buf) | mirror | set by sched (warmup only) | snapshot | **N** | Y | Y→2.0 | **F4-04**: post-warmup resume stuck 2.0 ≠ continuous 0.51. |
| `_cached_hp` (attr) | mirror | Y | **Y (writes)** | N | **N** | N | **ORPHAN / F4-11** forward input, leaks val→train. |
| `_cached_pred_error_norm` (attr) | mirror | Y | **Y (writes)** | N | **N** | N | **ORPHAN / F4-11** pen input to block gates. |
| `_cached_pred_k/_gate/_usefulness/…` (attrs) | mirror | Y | Y | N | N | N | ORPHAN (streaming inputs; read only in-adaptive → train-only paths mostly). |
| `_pred_loss_term` (attr) | mirror | Y | cleared | N | N | N | OK (graph handle; reset_cache nulls, mirror.py:510). |
| `_traj_state` (attr) | block | B10: train+stream only | **no longer (B10)** | N | N | N | B10 closed the eval-write hole (bind.py:403-414, block.py:378-385); still not in snapshot but eval no longer writes it → F2B-01 fixed. |
| `_pi_v` (attr) | layer | Y (loss) | **Y (loss)** | N | N | N | ORPHAN (F3-22): search seed, shape-keyed; reset_cache doesn't touch it. Benign numerically, snapshot-inconsistent. |
| `alpha_diag` (param) | mirror | Y (.data lerp + Adam) | — | **Y** (param) | n/a | n/a | **F4-01**: `.data` write re-fired per recompute under ckpt; also Adam + aux + lerp triple-drive (F3-13). |
| `b_i`,`b_d` (params) | block | Y (.data lerp stack) + Adam | — | Y | n/a | n/a | dual-drive F2B-10; writes NOT gated on active depth → frozen layers still lerped; ckpt recompute does NOT re-fire (the lerp is in `stack.forward`, outside `_checkpointed_block`) → F4-01 does not apply here. |
| `active`,`_val_ema`,`_val_var`,`_prev_val`,`_last_depth_step` | DepthController | Y (eval-gated) | — | only `active` | — | — | **F4-06**: 4 integrator scalars ORPHANED → spacing guard bypassed on resume. |
| `cfg.gradient_checkpointing` | cfg (governor) | Y (governor flip) | — | **Y** (cfg pickle) | — | — | the ONLY governor memory; flip is a control actuator → triggers F4-01 side effect. |
| `model._phase_ratio_ema/std` | stack (loop-owned) | Y (phase scaler) | — | **N** | — | — | ORPHAN: train.py:202-203/583-584 EMA of mirror/base grad ratio; never persisted → re-bootstrap each session; feeds only train.py's `mir_s` (notebook has no phase block → D-08). |
| `_mlp_cnt/_mlp_now_ema/_mlp_base_ema` (buf) | block | Y | snapshot (non-persistent) | N (persistent=False) | Y | Y | OK for eval; **F4-01**: recompute re-runs `.add_`/`.mul_` (in-place) → NOT double (buffer writes are restored by non-reentrant ckpt, verified `_mlp_cnt` single at 1.0 forward-only, p2). |
| `_mlp_ratio` (float attr) | block | Y | snapshot-blind | N | N | N | ORPHAN float, watchdog input via `_mets['mlp_ratio']` — recomputed each fwd, fine. |
| maturation `gate/readiness/pen_ema/pen_init/tau_norm` (buf) | MaturationController | Y (step!=None) | eval reuses (B11) | Y | Y | Y(bridge) | OK; eval reads published combined gate (stack.py:399-400, B11). `global_ready` is `gate>thr` derived, not stored. |
| `scheduler._step` vs loop `step` vs watchdog `_cur_step` | (system) | — | — | — | — | — | **F4-05 three-clock skew.** |

---

## 3. Handoff answers (every item addressed to agent 4) + B6–B12 status in scope

**From 01 §5 → agent 4 (losses/watchdog): F-07 / F-08 / F-09 / F-21 + `compute_losses`
RNG & aux_dict side effects read by the veto.**
- **F-07 (hard veto ceiling K·ln2)**: **CLOSED by B7**, re-verified live — both
  copies call `hard_veto_ceiling(cfg.vocab)=2·ln V` (train.py:450, c10:732; the
  `model.lm_head.K` read is gone). At V=65536 → **22.1808**, and 02a measured the
  normalize=True uniform is exactly ln V=11.09 so 2·lnV=2.000× uniform; band
  15.5<22.18<34 holds. 02a's residual (normalize=False leg has uniform 44.36 > the
  22.18 ceiling — a future `head_normalize=False` run inherits a ceiling calibrated
  for the other branch) is a **watchdog-policy gap I own**: recommend
  `hard_veto_ceiling(vocab, normalize, K)` branch-aware; stale "(uniform-bit NLL)"
  print labels (train.py:452, c10:734) still say the pre-B7 name.
- **F-08 (soft-veto numeric surface)**: re-derived above — **F4-02** quantifies the
  `arm_ce`-per-eval re-anchor (1.6×seed band, ~339-step reopening) which is the
  concrete mechanism behind F-08's "false-veto window"; NaN/±inf bypassed-into-alarm
  semantics re-verified (F4-12); the two copies' NaN **endgame** still diverges
  (train.py `raise` :520 vs notebook skip-step c10:854) — D-19, unchanged.
- **F-09 (resume CE-baseline policy divergence)**: STILL OPEN and it is a **control
  decision, not a loss one** — notebook pops `ce` after resume (c9:573) to force
  re-bootstrap; train.py does not (train.py:346). Given F4-02 (arm_ce re-anchors
  anyway at the first post-resume eval), the train.py carry is actually the more
  stable choice *if* eval is reachable; recommend standardizing on "carry + let the
  first eval re-bootstrap once" and deleting the notebook pop, OR arming once
  (guard the pop). **My call: single-arm.**
- **F-21 (RNG envelope)** and the `compute_losses` RNG side effect: `_pi_v` reseed
  (`losses.py:146` `torch.randn(...)`, device default gen) fires once/layer/session;
  it is a plain attr (F4-11 orphan table) → not in envelope, not snapshotted. Under
  the checkpointing finding **F4-01** the noise RNG *is* correctly replayed (dCE=0),
  so the write-noise is not an RNG-envelope problem; the `_pi_v` power-iteration seed
  remains an eval-mutated plain attr (fold into `reset_document_state()`, agent 5).

**From 02a §4 → 4 (control/watchdog):** F2A-07 fully — the band for the production
(normalize=True) branch is geometry-correct (2·lnV sits 1.43× above cold-start, 0.65×
below the garbage class); the normalize=False leg (uniform 44.36) is the one branch
the fixed ceiling mis-covers. 02a's suggestion that a *geometry-aware* soft veto watch
the d_H=4 pair margin (0.30 → compressed ×0.26 at 1045 steps) rather than raw CE is
sound and is the natural extension once the ladder (F4-13) is redesigned — noted for
B13, not actionable on the current sensor (it consumes forward magnitudes only, B2
doctrine).

**From 02b §5 → 4:** (a) F2B-04 eval-vs-train maturation gate — **resolved by B11**
(combined `max(ramp,readiness)` published to `maturation.gate`, stack.py:399-400, so
step-None eval reuses the train gate); the watchdog "arm at first val" and the eval
metrics now see the same trunk. `layer_gate_*` aux channels still log the published
combined gate (losses.py:369 `_mat = stack.maturation.gate[l]`) → they read the
COMBINED value post-B11 (02b §2.28's "raw ramp" caveat is now stale — the ramp-only
log no longer applies; recommend relabel or re-confirm on next run). (b) F2B-07
apply_tau_lr post-clip + per_layer_ls_lr + trust/gradalign divergences → answered in
**F4-08** (order breaks AGC, quantified) and it is a control-plane law, agent 4
ack. (c) "three clocks gate ls_mults/_usefulness_temp/maturation — Agent 4 owns the
resolution" → **F4-05**; resolution proposed. (d) watchdog channels are
forward-magnitude only (B2 ✓) — confirmed, except the sensor-const blind spot
F4-12 tail.

**From 03 §5 → 4:** (a) **F3-01 train.py fake-0.0 LR death** — FIXED by B12
(train.py:682 `math.isfinite(val_loss)` gate + `evaluate` returns NaN on empty pool
train.py:776) — verified in tree; notebook `_val_ok` gate was already correct. (b)
**F3-17(c) boost-gate laxity** — re-confirmed (p4 S1: `_val_improving` true in 25/40
plateau evals, i.e. boost gate "open ~2/3 of plateau time"; the 0.002 tolerance is
one-sided-lenient exactly as 03 said). (c) **F3-17(b) damp/restore units** — the
shared `_lr_damp_steps` crash surface is FIXED (B12 hasattr, p4 S4) but the counter
is still a dead orphan (never read, never persisted) → recommend deleting it or
making the restore-rate per-step with an explicit step count. (d) **FailureDetector
docstring still advertises "loop rebuilds FRESH Adam and rewinds the LR controller";
`LRController.rewind()` has ZERO callers** — confirmed (grep: def ad.py:357 + the
docstring mentions only, no call site in either loop); the `optimizer` setter
re-binding (ad.py:320-325) that "MUST propagate into the wrapped scheduler" is also
never exercised (nothing re-binds a fresh optimizer post-D6). Both are **dead
recovery code** contradicting the D6 sensor-only contract — recommend delete
`rewind`+setter-rebinding or gate them behind the manual-recovery script. (e)
`apply_tau_lr` docstring (tc.py:99-106) "scaled … after AGC — the same mechanism the
loop already used" **documents F4-08 as intent**; when the order is fixed, fix the
text too. (f) trust sources (bridge.readiness, maturation mean) are control-plane
quantities fed to the optimizer (train.py:625-630 `set_trust`); not persisted —
coordinate with agent 5 if trust ever lands in ckpt (notebook never calls set_trust,
D-06, so its trust arm is inert=1.0 — a control divergence I flag but is optimizer-scope).

**B6–B12 status checks in my scope:**
- **B5 freeze/re-arm under resume+double-eval**: VERIFIED CLEAN (p1 T1: 0/200
  first-sample fires, ph zeroed on release, cooldown carried). **B5 holds.**
- **B12 damp fix under legacy pickle**: VERIFIED FIXED for `_lr_damp_steps`
  (p4 S4) — but **the same legacy-pickle class is NOT fixed for the detector stats
  (F4-03, 7-wide unpack) or the scheduler `_ls_fast` length (F4-07/p4 S6d)** —
  B12 closed one schema-drift site, three siblings remain.
- **B12 `_ggeo_freeze` survival under governor toggle**: VERIFIED (p2d: freeze flag
  is a plain module attr, orthogonal to `cfg.gradient_checkpointing`, survives an
  OFF→ON flip; and it correctly stops the `_step_count` double-tick on the ggeo path
  — `_step_count WITH freeze = 0`, NO freeze = 2 after one ckpt fwd+bwd). **B12
  works as scoped** — its limit (F4-01) is that it guards only the *inert counter*,
  not the load-bearing `.data` lerps.
- **B11 gate parity**: re-confirmed alive (stack.py:379-400); eval `global_ready`
  reads the published combined gate (`p3d` code-anchored).
- **B10 pen RMS / spiral**: spiral `traj_state` train==eval in-graph both modes
  (bind.py:407-414), eval no longer zeroes the cache — F2B-01 closed; pen now
  per-dim RMS → the `_cached_pred_error_norm` leak (F4-11) is now *correctly scaled*
  (mean 0.359 not 8.1) but still **crosses the eval/train boundary**.

---

## 4. Governor × checkpointing learning-equivalence measurement (mission §4)

**Design.** Two independent EVAStack builds at identical seed (build determinism
verified max|Δw|=0.0 across two fresh same-seed stacks), same input batch, `torch.
manual_seed` reset before each forward so the `randn_like(i_gate)` noise stream
replays; gradient_checkpointing toggled via `cfg` (read at forward time). All plain
attrs (`_cached_hp`, `_traj_state`, `_gradalign_tgt`, `_intent_stream`, …) and all
named buffers null/restored to the identical pre-step state before each arm so the
ONLY difference is the checkpoint flag. `p2_ckpt.py`/`p9_data.py`.

**Result — the optimizer does not see a difference:**

| quantity | ckpt OFF | ckpt ON | deviation |
|---|---|---|---|
| CE (forward) | 10.205717087 | 10.205717087 | **0.000e+00 (bit-exact)** |
| global ‖g‖ | 153.2667 | 153.2690 | 1.5e-3 abs |
| ‖g_off−g_on‖ | — | — | 1.25e-1 |
| **relative grad deviation** | — | — | **8.15e-4** |
| max per-param abs grad dev | — | — | 1.8e-2 @ `mirror.W_proj` (≈1e-4 rel) |

So the **weight update is equivalent to float re-association noise** — turning the
governor on/off does **not** change what Adam applies on a given step. The "if big,
checkpointing silently changes LEARNING" test the mission poses: it is **small** at the
gradient level (8e-4), forward bit-identical.

**But LEARNING ≠ one gradient.** The second-order effect (**F4-01**) is that
checkpointing re-executes the block forward during backward, and the mirror's
**`.data` self-regulation runs once per recompute**:
`alpha_diag.data.lerp_ + data.add_` measured **2.00× per step** (OFF 3.98e-3 → ON
7.97e-3 |Δ| over 5 steps). Recompute census: `balancer.backward`=4 block-forwards,
`grad_geometry` logged step=**10** block-forwards (2 layers). Consequences:
- alpha_diag (expert decay ladder) drifts toward its target ~2× faster under ckpt →
  different specialization trajectory from an otherwise-identical non-ckpt run.
- The M16 governor flipping the flag at a VRAM pulse **changes the effective rate of
  a control law mid-run**, with only a one-line log — no other part of the system
  knows the adaptation constant just moved.
- Buffer-`.add_`-style state (`_mlp_cnt`, mirror EMAs) is *not* double-applied (the
  non-reentrant restore reverts named-buffer in-place writes: `_mlp_cnt` stayed 1.0
  on a forward-only ckpt run, p2) — so the exposure is specifically the **Parameter
  `.data` writes and Python-attr rebinds that escape the autograd buffer-restore**,
  i.e. exactly the class F3-12 predicted for `_step_count` (which B12 froze) and
  which B12 did **not** freeze for `alpha_diag`.

**Trajectory state under recompute:** B10 made training in-graph (bind.py:407-414)
and the carry cache streaming-only (`_stream_mode`), so `_traj_state` is not read in
windowed training → the recompute cannot corrupt it (the F2B-01 hole is closed
independent of the governor). Dropout: the coded head/trunk use RMSNorm-style
scaling, no `nn.Dropout` on the forward path was found in the block — the only
per-step stochasticity is the i_gate write-noise, which RNG-restore handles (dCE=0).

**Verdict (§4 answer):** checkpointing is learning-equivalent at the **single-step
gradient** level (8e-4, forward exact) but **NOT state-equivalent at the
multi-step controller level** — it doubles the mirror `.data` adaptation cadence,
and the governor's toggle is therefore a *de facto* (undocumented, unlogged as such)
change to a control-law rate. Highest-leverage fix: move the `alpha_diag` lerp/push
out of the checkpointed region (it belongs after the block, on cached
`alpha_target`, once per step).

---

## 5. Not verified / handoff to agent 5 (persistence)

**Not verified here (scope or host):**
1. **CUDA-side** behaviour — all probes are CPU fp32; the bf16/autocast governor
   interaction (the A100 `_USE_AMP` path, notebook wraps the whole fwd in autocast)
   may change the F4-01 RNG-replay guarantee if autocast caching interacts with
   checkpoint recompute; the `.data`-double-apply conclusion is dtype-independent
   (it is a Python/control-flow effect) but the 8.15e-4 grad noise will move under
   bf16.
2. **Production ladder magnitudes** (D=2560, 24 layers): F4-08/F4-13 are the exact
   closed-form code formulas evaluated at real τ/ls, but the actual `ls_mult`
   reached and the real `alpha_diag` drift rate are unmeasured on a trained model
   (no twin_free ckpt on host; legacy best.pt K32).
3. **DepthController + maturation interaction on a trained trunk** (F4-06) — the
   spacing-guard-bypass is proven on the class, not in a live multi-thousand-step
   resume.
4. The **legacy detector stats width** of the on-disk `checkponts/best.pt` (F4-03) —
   file mid-`.crdownload` during this relay (02a caveat); re-test the resume path
   against it once it lands.

**Handoffs → agent 5 (persistence doctrine):**
- **F4-03**: detector `stats` lists need a width/schema version tag (or tolerant
  load); the envelope `detector` has no `type`/`schema` key (contrast `scheduler`
  which does). Legacy best.pt → 7-wide unpack crash.
- **F4-06**: DepthController `_val_ema/_val_var/_prev_val/_last_depth_step` are
  train-mutated and in **nothing** — persist alongside `active_depth` or accept the
  documented spacing-bypass.
- **F4-04 / F4-11 / F4-13 (orphan table)**: the snapshot/`best.pt`/rotation triple
  misses `{mirror._cached_hp, _cached_pred_error_norm, layer._pi_v,
  block._traj_state}` (eval writes two of them → cross-boundary leak) and
  `scheduler`-owned non-persistent buffers `{_usefulness_temp, _alpha_override}`.
  Land `model.reset_document_state()` (01's proposal) AND a scheduler-buffer save
  (or make `_usefulness_temp` persistent) before the next schema touch.
- **F4-09 / F3-08 family**: `orig_lrs` positional len-only restore (scheduler) and
  the group-hyperparams positional restore (EVAAdamW trust/cap/role) both need
  by-name re-mapping like the optimizer; B12 wired `_restore_optimizer` by name
  (train.py:323 — the F3-08 positional-restore is now FIXED, verified the call site
  replaced) but the **scheduler** orig_lrs path was not touched. Needs-lock: no test
  executes either resume filter with a reordered/shrunk group list.
- **F4-02**: single-arm `ce_armed` (or persist `_last_armed_step`); currently
  `arm_ce` fires every eval — decide whether the re-anchor is desired and if so
  document it as a cadence-coupled control actuator.

**Suite:** `python -m pytest tests -q` → **305 passed in 58.94 s** (tree clean at
analysis start and end; this artifact adds zero repo changes beyond this document).
