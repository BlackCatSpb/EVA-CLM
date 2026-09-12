# 03 LOSSES & OPTIMIZATION AUDIT

Agent 3 · serial relay · tree B6–B11 (HEAD 0397325) · audit of CURRENT code · baseline
**299 passed** (verified clean at start and end of this audit; this artifact adds zero code
changes).

Scope: `compute_losses` (core/losses.py + EVAStack glue) → every aux term →
`LossBalancer` (align path, sign-mask, per-param bound, BYPASS_AUX, grad_geometry) →
post-balancer grad surgery (phase scaling, ls_mults, apply_tau_lr) → AGC `GradientClipper`
→ `build_optimizer`/LLRD/roles → `EVAAdamW` + `eva_proj` (AdamP) arm math/state →
`MirrorLRScheduler`/`LRController` (warmup, mirror-adaptive mult, boost, plateau damping,
state/resume) → and the FULL notebook-vs-train.py drift enumeration for everything between
loss and `optimizer.step()`.

Method: line-by-line read of core/losses.py, core/training_control.py, core/adaptation.py,
core/lr_scheduler.py, core/eva_optim.py, core/adaptation.py, scripts/train.py (all 837
lines), notebooks/eva_colab.ipynb (full cell dump), plus CPU mini-model probes in
`%TEMP%\opencode\audit3\` (probeA_losses, probeB_balancer, probeB2_recompute, probeC_optim,
probeD_pen, probeE_adam, probeF_misc, probeF2_resume, probeG_bounds, probeH_fakeval,
probeI_roles). Classifications: **VERIFIED-REPRO** (executed), **STRONG-READ** (code walk
+ partial execution), **SPECULATIVE**.

Prior batches honored in-scope: B6 grad_geometry (F3-11, F3-12 — present, pure, but two
defects), B7 veto ceiling + L2 buffers (single-source `hard_veto_ceiling(cfg.vocab)` called
in BOTH loops, VERIFIED-REPRO — drift row closed; bank log_taus are Parameters and
`mem_tau_reg` correctly targets only them — probe D3), B8 identity-resume guard (wired both
loops; cosmetic overwrite persists — F3-27), B9 embed_center (OFF; no loss-path consumer to
update), B10 pen-RMS + spiral (operating point re-measured against every consumer — §3.2),
B11 gate parity (eval-side loss math re-verified: A3 — eval CE now sees the same
`maturation.gate` values the train path published at save; no eval/train CE-parity defect
remains in the loss itself beyond the by-design surprisal asymmetry, F3-18).

---

## 1. Findings

Severity order. IDs F3-01… stable for the constitution.

### F3-01 — HIGH — VERIFIED-REPRO — train.py's fake-0.0 empty eval **permanently kills the LR controller** (01 F-06's second, worse half)

train.py's `evaluate()` returns `total_loss / max(total_steps, 1)` → **0.0** when the eval
pool is empty/short (01 F-06 documented the fake-best-save). The same 0.0 then flows to
`scheduler.report_val_loss(val_loss)` (train.py:674–677, ungated) and `MirrorLRScheduler`
anchors `_best_val_loss = 0.0` on first report (core/lr_scheduler.py:131–136). Thereafter
**every real val loss is a "regression"** (`val > best·(1+0.05) = 0`) → `_loss_lr_factor`
halves per eval to the 0.05 floor, and the full-restore branch
(`val < best·0.98 = 0`) is arithmetically unreachable (lr_scheduler.py:139–146).
Probe (probeH): feed `0.0`, then four real evals → factor `1.0 → 0.5 → 0.25 → 0.125 →
0.0625`, `_val_improving` irrelevant. The notebook's `_val_ok` gate (c10:388) skips
report+arm+save when no hold-out data ran — the two copies' LR fates diverge **hard** on a
short corpus. Fix: same `_val_ok` gate train-side, or return `float('nan')` and guard.

### F3-02 — HIGH — VERIFIED-REPRO — the two copies run DIFFERENT schedule cadences: train.py's CLI `--warmup/--log-interval/--eval-interval/--save-interval/--max-steps` are SILENTLY CLOBBERED by `__post_init__`

`EVAConfig.__post_init__` re-derives those fields from `LambdaConfig` (core/config.py:404,
395–432) AFTER the constructor — probe (C5): `EVAConfig(warmup_steps=500,
log_interval=100, eval_interval=1000)` → effective **warmup=101, log=55, eval=233,
save=987** (λ₃-derived: F₁₀·λ=101, F₁₃=233…). The notebook KNOWS this and re-pins
**after** init: `cfg.warmup_steps=1200; cfg.log_interval=55; cfg.eval_interval=1045`
(c4:93–95, comment: "Пересчитывает … и ЗАТИРАЕТ аргументы"). train.py never re-pins (its
CLI values are passed INTO the constructor → destroyed). Mission item 4 ("confirm both
copies read the SAME values"): **they do not** — notebook 1200/1045/55 vs train.py
101/233/55. Consequences, all live in my scope: LR warmup length (11.9×), AGC τ-map
rebuild cadence (4.5×), FailureDetector warmup-freeze release (B5 policy — the two-strike
stop and PH re-arm fire at different steps), DepthController plateau, `report_val_loss`
arrival cadence (damp/boost/restore clocks), `eval_interval`-derived EMA decays
(LossBalancer `_ema_decay`, DepthController `a`), and the 500k-vs-λ eval budget. This is
the single largest non-semantic "same-code-different-run" driver between the arms.
Fix: pin post-init in train.py exactly like c4 (single source: a `apply_colab_pins(cfg)`
helper both copies call, or make `__post_init__` respect explicitly-provided fields).

### F3-03 — HIGH — STRONG-READ (all sub-diffs VERIFIED by static+probe) — F2B-07's "three grad-planes" is now **four planes**, and the arms apply different LAWS between balancer and step

Between `balancer.backward` and `optimizer.step`, the two copies differ on every axis:

| plane | train.py | notebook |
|---|---|---|
| mirror phase multiplier `mir_s∈[0.2,2]` (EMA-of-EMA sigmoid on mirror/base grad ratio) | :556–580, applied to `mirror_parameters` | **absent** |
| scheduler `ls_mult` (per-layer var(log_scale) trend) | :583–596 (base ×ls; mirror ×clamp(mir_s·ls)/mir_s) — **but `per_layer_ls_lr` is hard-wired `False`** at :828, so the block is dead even though the CLI flag exists (:785) | active (c4:27 `per_layer_ls_lr=True`) |
| τ-LLRD `lr_mult=(τ_l/τ_ref)^−γ` | **never applied** (train.py has no `apply_tau_lr` call) | `apply_tau_lr` **AFTER clip** (c10:247 vs :243) |
| MLP depth-boost param hooks `exp(0.10·l)` (up to ×9 at L24) | `model.apply_mlp_depth_gradient_boost()` :206 → stack.py:1110–1136 (`register_hook` on every MLP param; **optimizer+resume safe, balancer- invisible**) | **never called** (grep 0) |

The hook plane (mlp-boost) is the sneakiest: it multiplies gradients INSIDE
`balancer.backward`'s autograd passes, so LossBalancer's per-param bound is applied to
already-boosted grads (bound preserved per-param, ratio CE:aux preserved — AGC then clips
the boosted total), while the notebook's post-clip `apply_tau_lr` multiplies AFTER AGC and
**breaks** `‖g‖≤c_eff‖θ‖` by up to `ls·lr_mult` (probe G table: at production ladder
extremes the shallow layer gets ×(2·(8/64)^−0.65=3.65)≈**7.3× post-AGC**; deep ×(0.5·0.27)
≈ 0.13). The optimizer law split compounds it: `_make_opt` uses `llrd_decay=cfg.llrd=0.9`
(train.py:216) vs `llrd_decay=1.0` (c9:33) — measured group lrs (probeI, 4-layer mini):
`layers.3.b_d`: **6.46e-5** (train.py) vs **8.87e-5** (notebook, then grad-scaled ×0.26
post-clip ≈ net 2.3e-5); `layers.3.conv.weight` 2.19e-4 vs 3e-4×(ls·lr_mult)∈[0.07,0.6].
Also the arms differ on the optimizer itself: train.py CLI defaults to `adamw`, notebook to
`eva_proj` (c4:22) → cautious/trust/caps/AdamP/dual-LRD all differ (§3 of the drift table).
None of this is new dataflow — 02b flagged four of these axes (F2B-07) — the increment here
is (a) the mlp-boost plane, (b) quantified post-clip violation, (c) `per_layer_ls_lr=False`
making train.py's own ls code dead on arrival while the notebook's is live.

### F3-04 — HIGH — VERIFIED-REPRO — LossBalancer bypass bound is 3× looser than documented; the bypass-only early-return is UNBOUNDED

(a) Bypass (`gradalign`) is bounded by `p.grad.norm()` *after* the aligned-aux addition
(training_control.py:697–708) → per-param worst case ‖aux_total‖ ≤ ‖g_CE‖ (align) +
‖g_CE+align‖ (bypass) ≤ **3·‖g_CE(param)‖**, not the advertised 1×. Toy probe (B3/E3-B):
100× bypass → measured addition **2.24×/1.78×** of ‖g_CE‖ (bound holds at 3× — probeG:
4 real-model params sit in the (2,3] band, zero above 3). (b) **Latent hole**: when
`aux_dict` contains ONLY bypass terms, `backward()` early-returns at :644–652 and runs
`sum(bypass.values()).backward()` raw — measured **101.4×** ‖g_CE‖ (E3 CASE A), with
`_ga_record` still True (set False only at :654–656) → the gradalign CE-target is
poisoned by the bypass gradient (F2B-11's race, now with the unbound-addition half).
Unreachable with current producers (pred/gate_l1 always tensors in train mode), one
cfg-edit away. Fix both with one line: run the same clamp in the early-return path.

### F3-05 — HIGH — VERIFIED-REPRO — the UCL uncertainty gate flipped from born-open to **born-closed** under B10's pen rescale: thresholds were NOT recalibrated (direct answer to the 02b §5 handoff)

`pen = ((raw_pred_error/hp_norm)²).mean(-2,-1)^½` (mirror.py:497, B10) now measures
**mean 0.353, p50 0.343, p95 0.574, max 0.675** at init (probe D1, mini, real-shaped
ids). UCL `u_gate = σ(κ·(pen − e^{log_tau_uncert}))` with `log_tau_uncert` init **1.0 →
threshold 2.718** (concept_layer.py:108,349–354): measured fire-rate(>0.5) = **0.0000**
(mean 0.0009) vs pre-B10's "1.000 born-open" (F2B-06). `out = read·u_gate·c_gate·scale`
(concept_layer.py:367) ⇒ the concept-readout channel contributes ≈0 at the operating point;
it can re-open only if `log_tau_uncert` (Parameter, role scalar lr 3e-4 ✓ in-group —
probeI) learns down ~2 log-units. Same unit-mismatch family as F2B-06/F2B-02, opposite
failure direction, same root cause: threshold specced on a scale nobody normalizes to.
The other pen consumers are now CORRECT: block decay `pen_decay_factor` mean 0.913
(min 0.838) instead of the 0.5002 asymptote; igate boost `γ_surprisal·pen` = +0.088 vs
+2.0 (design band 0.08–0.25 ✓; probe D1). **Action: recenter `log_tau_uncert` init to
ln(0.35)≈−1.05 (or make it a running quantile of pen) — a one-line, fresh-run change.**

### F3-06 — HIGH — VERIFIED-REPRO — post-B10, the τ-ladder is *still* ≈crushed at the slow end (pen arm caps τ_eff ≈ 11 tokens), and the AdaptiveController lerp law is unaware of the pen arm at all (F2B-10/F2B-02 companion answer)

Per-token leak from the pen arm alone at init: `−ln f` mean **0.0916** (probe D1) ⇒ every
nominal τ beyond ~11 tokens is decayed 8%+/token regardless of `d_s`/`b_d`: production
slowest scale τ_nom≈1448 behaves as τ_eff≈**10.7**. B10 turned a 0.5-pin (τ_eff≈1.4) into
a 0.91-pin — same mechanism, 8× milder, and now *learnable* (∂σ′/∂w_d_pen ≈ 0.245 at
pen≈0.35 vs ≈0.0003 pinned → `w_d_pen` (role scalar, lr 3e-4 ✓ probeI) actually has
gradient to offset). The controller targets are **unit-independent of pen**:
`b_d = b_d_max − expl·(b_d_max − (2+3·l/L))`, `expl = min(1, |mirror|/0.25)`
(adaptive_controller.py:91–99) — so the 7.7–8.3 targets from F2B-10 are NOT mis-scaled by
B10 and must NOT be re-tuned for pen reasons (mission question answered: **no**), but the
write-rate law they solve (`i_gate = c/τ_nom`, softplus⁻¹ exact, B3-fixed) presumes the
memory lifetime equals τ_nom — with the pen arm it doesn't: equilibrium ‖M‖≈i_gate·‖h‖·τ_eff
is **~100× under target** on the slowest scale at init (c=5.31/τ → i_gate=0.011 at τ=512;
τ_eff≈11 → ‖M‖ = i_gate‖h‖·11). B11's "bounded redesign = B12" decision is confirmed as
the correct home for this; do not "fix" it by re-tuning b_d.

### F3-07 — HIGH — VERIFIED-REPRO — `MirrorLRScheduler.load_state_dict` resume crash: `AttributeError: '_lr_damp_steps'` on the first plateau-zone eval after any damped resume (BOTH copies)

`_lr_damp_steps` is created only in `report_val_loss`'s first-call init
(lr_scheduler.py:136) and touched at :152 (`+= 1`) when `factor<1.0` in the plateau zone,
but is **neither in `state_dict()` nor restored** (:259–303). Repro (probeC C6): scheduler
damped to 0.5 → state_dict → fresh scheduler.load_state_dict → `report_val_loss(val in
plateau band)` → AttributeError, which escapes both training loops (only
KeyboardInterrupt is caught). Real-world reach: best.pt saved while damped (common during
plateau training), resumed, first eval lands within [0.98·best, 1.05·best] (also common) →
**dead run**. One-line fix (`self._lr_damp_steps = sd.get('lr_damp_steps', 0)` in
load + persist it in state_dict).

### F3-08 — HIGH — STRONG-READ — train.py's optimizer restore is still POSITIONAL; the by-name restorer it ships is dead code (01 F-11 still open, my scope per mission #3)

train.py defines `_restore_optimizer` (:81–149, the by-name remapper) and never calls it —
the resume path :317–323 rebuilds then `optimizer.load_state_dict(ckpt['optimizer'])`
(torch: state keys are POSITIONAL group-index ints; my probeC-C3 confirms keys `0…431` and
that load validates only group-count/param-count, not param identity: any role-map or
arch edit that preserves counts silently lands `exp_avg` on foreign parameters — exactly
the blast radius 02b predicted for 24×λ-groups; the W_out+K partial-restore branch at
:112–133 exists ONLY in the uncalled function). The notebook wired its own by-name
restorer (c8:125–168, called at c9:44–47 — the two copies' `_restore_optimizer` bodies
also differ: train.py pads exp_avg_sq with 1.0 for grown W_out rows; the notebook
slice-truncates, different resume math). Additionally :318's guard references the
**global `args`** (`args.no_save_optimizer`) — a NameError hazard for any caller of
`train()` outside `__main__`, and the `print` at :339 ("Optimizer/scheduler rebuilt FRESH
(no momentum restore)") contradicts the restore that just succeeded at :321. This is the
single highest-risk persistence item in the loss→update chain.

---

### MEDIUM

### F3-09 — MEDIUM — VERIFIED-REPRO (probe E1) — per-layer **gradient pre-multiplication is near-invariant under Adam**: the phase/ls/τ-lr machinery is mostly placebo *as an LR control*

Doubling a param's gradient every step and comparing against doubling its pg['lr'], under
AdamW(0.9,0.95)+wd: relative param divergence **0.18% at t=20, 0.26% at t=50, 0.47% at
t=200** (probe E1: `m̂/√v̂` cancels the constant; only decoupled-wd's ratio and eps shift
slightly). Implications: (i) train.py's `ls_mult`/`mir_s` grad scalers barely change the
update; (ii) the notebook's post-clip `apply_tau_lr` DOES change things — but chiefly by
re-scaling grads **against the AGC bound** (the F3-03 violation), an unintended control
surface; (iii) the only *real* per-layer LR distribution in the codebase is
`build_optimizer`'s role×depth group lrs + the global scheduler multiplier — which differ
between the arms anyway (llrd 0.9 vs 1.0 + τ-lr). Recommendation: if per-layer LR is
wanted, it belongs in `param_groups['lr']` (or a per-group scheduler), not in
`p.grad.mul_`.

### F3-10 — MEDIUM — VERIFIED-REPRO (probeB) — LossBalancer semantic the docstring doesn't state: aux can NEVER oppose CE on any coordinate, and is ZERO on any parameter CE doesn't touch

The sign-mask keeps exactly the coordinates where `gce·gau>0` (:694). Measured (B3 real
path): **0.000 fraction of final-grad coordinates oppose CE** — every aux that "fights"
CE is silently amputated per-coordinate. And B1 disjoint-support probe: aux with 1e6-mass
purely on coordinates where `gce==0` contributes **exactly 0** (mask requires product>0;
a zero-CE coord is never "agreeing"). Consequences: regularizers on dead-CE params
(F2A-10's k_proj_l family if any aux ever targets them) cannot act through the balancer at
all; "which terms fight CE" is structurally unanswerable post-mask (B6's [ggeo] reports the
pre-mask cos, good, but the ledger must remember it). The bound itself holds literally
per-parameter (probe B1 case 1: clamp ratio measured 1.000000119) — so the mission's
"can an adversarial aux push a single parameter far?" answer is **no** beyond
2·(align)+2·(bypass) worst-case norms, F3-04's looseness notwithstanding. Zero-duty-cycle
regime (B2 sim): per-coordinate agreement survives down to ρ≈−1
(ρ=0 → 48.9% coords/69.2% norm; ρ=−0.9 → 13.9%; only exact-anti → 0.000).

### F3-11 — MEDIUM — VERIFIED-REPRO (probeD4) — B6 grad_geometry truncates to the first **8 terms alphabetically** — `gradalign, intent_tau, ls_reg, mem_tau_reg, nuc, orth, pred, reinforce, signal_ent, tau_dev_reg, w_m2v` never print

`grad_geometry(..., max_terms=8)` (:560) iterates `sorted(aux_dict)` and `break`s at 8
(:587–588). A production-shaped mini emits 17–18 tensor terms (probe A2/D8); the sorted
first-8 = `[alpha_novelty, balance, branch, bridge_conn, decorr, div, diversity,
gate_l1]`. So every `[ggeo]` line the run produces is blind to the governance term
(gradalign), the τ-ladder terms, pred, and signal_ent — the exact terms audits keep asking
about. The doc comment ("one extra backward pass per term — call sites gate it to
log_interval") is honest about cost; the cap itself is undocumented. B6 status verdict:
**implemented and pure (no .grad/graph mutation, VERIFIED), but the 8-term cap silently
defeats its purpose.** One-arg fix: `max_terms=None` or alphabetical rotation.

### F3-12 — MEDIUM — VERIFIED-REPRO (probeB0/B2/A6) — gradient-checkpointing × (ggeo + multi-pass balancer.backward): forward side-effects re-execute 8–12× per logged step; one persistent buffer is NOT restored

Default `gradient_checkpointing=True` (config.py:380) + cell 4 keeps it on. Per logged
step the loop triggers: ggeo (up to 8 `autograd.grad`) + balancer (3 grads) — each
checkpointed segment re-runs the block forward (probeB2: block_fwd ×12 ggeo +×4
backward on a 2-layer toy). Non-reentrant checkpointing **reverts in-place buffer
mutations** (measured: `_mlp_cnt` frozen at 1.0, `_signal_norm_ema`/`_mlp_ratio` —
watchdog inputs — safe ✓), but `bind._step_count` advances via `self._step_count += 1`
(attribute re-bind, bind.py:397) and the rebind **survives recomputation**: measured
1→7→9 in one logged step. It's a `persistent=True` buffer → the ~8×/logged-step
inflation rides into best.pt (feeds nothing — 02b row 15 — but it IS checkpoint bloat +
confusion tax, and it proves the restore mechanism has a hole class: *any* forward
side-effect done as attribute-rebind rather than `.add_` re-applies per recompute).
Also reproduced (probe A6): if anything mutates plain-attribute forward caches BETWEEN a
step's forward and its backward (an interleaved eval/OOM-retry forward, a resume-shape
probe), the checkpoint recompute mismatches and PyTorch raises `CheckpointError
(489 vs 485 saved tensors)` — the production loop order (forward→losses→backward, eval
only after step) is safe **today**, and `_checkpointed_block` explicitly threads
`_cached_pen`/`_cached_hp` (stack.py:1087–1105, B10-aware ✓). Status hazard, not bug.

### F3-13 — MEDIUM — STRONG-READ — `alpha_diag` is triple-driven (F2B-10 pattern on a new coordinate): controller `.data` lerp + novelty `.data` push + aux loss + Adam

mirror.py:470–490: every forward in train mode — gated on the scheduler's
`_alpha_override < 0.1`, i.e. ACTIVE ONLY AFTER WARMUP (lr_scheduler writes the override
down to 0 by `warmup+blend`; note this makes the controller's writes cadence-coupled to the
warmup schedule, whose value differs between the copies, F3-02) — does
`alpha_diag.data.lerp_(alpha_target, 0.01)` (:480, target = sigmoid(2.2−ln relative
residual-var)) and (when `alpha_novelty_weight>0`)
`alpha_diag.data.add_(adapted_w·2·centered/G)` with `adapted_w = w·max(1, 0.1/(std+0.01))`
— the boost makes the push **anti-vanishing at collapse** (std→0 ⇒ push→×10·w/G), then
`.clamp_(0.01,0.99)`. Meanwhile the `alpha_novelty` aux term (losses.py:320–331, same cfg
weight gating both!) also pushes −var via Adam, and Adam's moments on alpha_diag are
blind to the .data writes — the exact "Adam + hard lerp fight" of F2B-10, plus a hidden
magic constant (`0.1/(std+0.01)`, `0.01` lerp rate) that is NOT in cfg (B4 single-source).
(`log_skip_alpha` checked — it is a plain Parameter with NO hard write in forward, mirror
uses it read-only at :799; the dual-drive there does NOT exist. Same family:
`_residual_var_ema` is a buffer — snapshot-covered ✓.)

### F3-14 — MEDIUM — VERIFIED (probeC C1/C2 + grep) — dead-knob census in optimizer/LR space (F2B-05 items re-verified; plus two new)

Consumers-none, values-live-in-cfg (and some in cell 4):
`tau_dev_lr_mult=0.2` — only read by `stack.param_groups` (:1202) whose only… *zero*
callers (both loops build via `build_optimizer`) → **the τ-ladder shaper `_tau_l_dev`
trains at FULL 3e-4** (probeI: group lr 3.0000e-04 in BOTH arms; measured in C1/C2
`named_parameters` dedup keeps the `_tau_l_dev` alias so `_role_lr_mult` misses every
rule — the 02b claim CONFIRMED and still true).
`gate_lr_mult=5.0` — same dead builder only. `vsa_b_lr_mult=0.1` — same (b_d's actual
0.296× comes from `_role_lr_mult`'s `_VSA_PARTS`, a DIFFERENT constant than the cfg
field). `log_scale_l2_weight=0.01` — zero consumers (ls_reg's threshold is the baked
`2.3` in losses.py:284). `mlp_mod_scale_reopen` — zero consumers (train.py:294–301
reopens via `mlp_gate_b_init`/`hybrid_gate.log_tau` instead).
`accum_steps` — consumed by NOTHING but checkpoint_inspector's display; the mission's
"grad_accum (accum_steps>1 paths tested?)" answer: **accumulation is not implemented in
either loop** (single micro-batch per step; nothing to test — and the veto/anneal/clock
maths in F3-10/F3-16/F-10 would all need revisiting if it ever is).
`EVAConfig.optimizer` — train.py CLI has NO flag to set it (argparse at :756+ omits
it) → train.py can never run the eva arms without editing source (notebook: c4:22).

### F3-15 — MEDIUM — VERIFIED-REPRO (probeA2/G) — `nuc` is a zero-gradient fixture at init; `branch` is the aux that currently owns the summed-grad direction

Two ledger extremes. (i) `nuc` (B2 stable-rank): fresh orthogonal `bind.W_proj` ⇒ power
iteration returns σ̂ ≈ σmax and SR/rank_ub clamps to the ceiling ⇒ penalty 0, and the
clamp gradient path is **0 at the ceiling** (probe A2: ratio 1.7e-8, top param grads
exactly 0.0) — it can only ever fire *after* collapse, by which point it's the damage
report not the guard. Fine as an aux-tripwire, misleading as a "regularizer"; keep in
ledger labeled. (ii) `branch` ratio **1.08×‖g_CE‖** at init, cos −0.008 vs CE (A2) — it
dominates `aux_total`, so on branch-heavy steps the per-param sign-mask is effectively a
branch-vs-CE negotiation; the other 16 terms mostly live below 2e-2 ratios. The
train.py-only anneal (F3-16 drift row) exists exactly because of this; the notebook runs
branch at full strength from step 1 — with branch 13.8 raw value vs CE ~10 (mini),
that's a materially different early-training attractor between the copies.

### F3-16 — MEDIUM — VERIFIED (static + probeG) — the align pipeline's AMP loss-scaling, anneal and ggeo inputs are sound in train.py but form a THIRD aux-weighting surface the notebook lacks

train.py:532–541: `aux_anneal` (branch,diversity ×step/5000, magic const, cfg field
ABSENT — `getattr(cfg,'aux_anneal_tau',5000)`), then `gscale=scaler.get_scale()`
(=1.0 with the default no-AMP; GradScaler(enabled=False).get_scale() verified 1.0, probe
E5) multiplies ce+aux BEFORE `balancer.backward`. Consequences: (i) with `use_amp=True`
(fp16, the F-20 legacy path), `balancer` bounds ratios on SCALED grads — `p.grad` carries
×gscale, `scaler.unscale_(optimizer)` before clip (:604) undoes it — ordering is CORRECT
(scaler step/unscale sequence verified by read; fp16 semantics themselves are the F-20
problem); (ii) the anneal re-scales two terms AFTER `compute_losses` returns → `last_cos`
and ggeo reflect the annealed sum (consistent) — but the notebook has NO anneal and NO
gscale: the two loops feed the SAME `balancer.backward` different-magnitude aux dicts at
every early step (×0→1/5000 ramp vs ×1 flat). Single-source the anneal (both or neither).

### F3-17 — MEDIUM — VERIFIED-REPRO (probeF5/F6) — scheduler state-integrity trio: positional `orig_lrs` restore, undocumented damp/restore units, boost-gate laxity

(a) `orig_lrs` restored whenever `len` matches (lr_scheduler.py:288–294) — probeF5: a
same-count reordered list loads silently → `pg0 lr 1.47e-4 vs correct 2.25e-4` (wrong
per-group LR, no warning). Group COUNT equality is a weak guard across the two arms'
32-vs-16 groupings — fine cross-arm (count differs → refuse ✓), unsafe against
intra-arm reshaping (role-map edits). (b) `_loss_lr_factor` restore rate
`+1/200 per report_val_loss` — comments say "per step → reaches 0.95 in ~600 steps"
(:153–157); it's **per EVAL**, so from the 0.05 floor it takes ~190 evals ≈ 200k steps at
eval_interval=1045 — the "warm-restart" is effectively a one-way ratchet down over a
150k-step run unless a `val<0.98·best` event fires (which a damped, barely-improving run
by definition rarely produces). (c) boost gate: `improving = val < val_ema·(1+0.002)` —
tolerance is one-sided-LENIENT (val 0.2% WORSE than EMA still counts as downtrend):
probeC-C8 plateau-noise sim → improving in **26/40** evals; probeD7 (falling-var model,
genuine `mirror_mult>1`) → boost actually engaged, max mult **1.068** sustained windows.
No ratchet-to-2.0 observed (mult recomputes each step, self-limits via EMA), but
"boost only while val is on a genuine downtrend" overstates the gate's selectivity: it is
"only while val is NOT on a >0.2% up-move", open ~2/3 of plateau time.

### F3-18 — MEDIUM — VERIFIED (probeA3/E6) — train≠eval CE by construction (surprisal + noise), aux-in-eval is computed-then-discarded in both copies — acceptable, but the ledger must say so, and gradalign/pred ARE the only train-mode-exclusive terms

Probe A3 (same batch, coded head): train-mode CE 5.12 (sw=0.3, `stack.training` gates the
weighting, losses.py:41) vs eval-mode CE 10.58 for the *identical draw* — 0.01 parity at
sw=0, big offset at 0.3. The val metric the LR controller and best-save are gated on is
the **unweighted** CE — a defensible doctrine ("val = plain NLL") but currently
undocumented; the objective-vs-metric gap is *systematic* in the notebook (which trains
with sw=0.3) and *zero* in train.py (sw=0). Per-term eval census (A3): only `pred`
disappears (mirror eval-guard :510) + `gradalign` (training guard :264) + surprisal;
every other aux tensor is RECOMPUTED at eval (no_grad) and discarded — cost, not
correctness, but it's also how `stack._cached_losses`/lbg side-effects get eval-mixed if
a future consumer reads them after eval. The `lbg_*` floats and `mb_scale`/`layer_gate_*`
keys in `aux_dict`/`_cached_losses` are LOG-ONLY — never enter backward (verified A2
DEAD-typed) — mission line items "lbg_tau, mb_scale" formally discharged as
diagnostics-not-losses.

### F3-19 — MEDIUM — VERIFIED-REPRO (probeG/D1/D3) — LossBalancer + AGC end-to-end with real cfg values on a real mini: the aux bound is a rounding error NEXT TO AGC; the "who bounds what" constitution line should be drawn honestly

Window-2 mini (3 layers, gradalign live): ‖g_CE‖=65.8; raw summed-aux block 0.170·‖g_CE‖;
post-balancer effective aux 0.170 (duty≈mask-only), ‖g_final‖=1.024·‖g_CE‖,
`last_cos=−0.166` (old global gate would have zeroed everything — F2B/B2's fix earns its
keep here); post-AGC ‖g‖=0.987·‖g_CE‖, 49/210 params clipped (probeG B4). So in this
regime the AGC ratio `c_eff·‖θ‖` (not the aux bound) decides the update for ~23% of
tensors, and the balancer's contribution to the clipped result is ≤3%. Ordering question
answered: AGC never *undoes* the aux bound (uniform positive per-param scaling preserves
the aux:CE direction ratio) — the notebook's post-clip τ-lr (F3-03) is what breaks the
AGC invariant, not vice versa. τ-AGC `c_eff` table verified in-run: (τ=32,128,512)→
c·(64/τ)^0.65 = (0.157, 0.064, 0.026), γ source = `tau_config.llrd_gamma`
(=cfg.tau_llrd_gamma=0.65, wired at stack.py:43 ✓), floors `max(τ,1e-6)` +
`‖θ‖<eps(1e-3)→skip` present and non-binding on the mini.

---

### LOW / INFO

### F3-20 — LOW — VERIFIED — ledger hygiene: dead args, dead producer branches, stale comments
`compute_losses(..., pred_weight=)` — body never references it (1 occurrence = signature,
probe A4) — same for the `h_emb` docstring-vs-reality (F3-21); `pred_w` producer
(:426–433) can never fire (`hasattr(lm_head,'pred_w')=False` on SigmoidCodedHead, A4);
`orth` producer gate `getattr(cfg,'orth_weight',1e-4)` (:166) uses a 1e-4 fallback while
the real default is 0.0 — harmless but a split-brain default; config comments that LIE:
`div_weight … (bypasses spectral alignment)`, `gate_repulse_weight … (bypasses
spectral)`, `alpha_novelty … (heuristic, no spectral)` — BYPASS_AUX=('gradalign',) only
(A5: verified); `losses.py:435–440` "Raw auxiliary losses — NO per-loss magic weights"
— true except the two in-house multipliers (`nuc ×cfg.nuclear_weight` baked at :456,
`gradalign ×cfg.gradalign_weight` at :278 — the latter is the term's REAL strength and
its only cfg-single-source; the former is the double-weighting-bug pattern the comment
claims extinct, in mild form); `_adamp_project` has 17 unreachable lines after `return u`
(eva_optim.py:118–134 — the old global variant; forensic hazard for the next reader);
`_lr_damp_steps` is incremented but never read (dead counter, cf F3-07); train.py:284
cosmetically overwrites `missing/unexpected` with the fingerprint return (F2B-09's note,
still present); `LossBalancer(align_cap=10.0)` still passed by BOTH loops into the
documented-ignored field (:497) — remove the call-site arg in cleanup.

### F3-21 — LOW — VERIFIED — `h_emb` never reaches the coded-head loss call; the two-ended read contract (02a §5c) is honored in the HEAD, but losses.py's `log_probs_for_target(h.reshape(...))` passes the TRUNK h and drops `h_emb` entirely

losses.py:27–28 uses `h` (post-trunk) for `log_probs_for_target`; `h_emb` is accepted at
:10 and never read (A4: grep count). train.py:441/745 and notebook c10:90/384 all pass
`h_emb=h` in good faith. 02a verified the identity roundtrip at the head level — this is
the loss-side wiring gap, not a gradient bug. Either consume `h_emb` in the coded branch
(the B2 two-ended design intent) or delete the parameter chain.

### F3-22 — LOW — VERIFIED — `_pi_v` is a plain per-layer attribute (nuc power iteration, losses.py:144–147): not snapshot-covered, mutated at TRAIN and EVAL loss time, and shape-keyed only by `bind_W.shape[1]`
Same leak class F2B-01/03 documented for the mirror caches. Effect today: eval forwards
re-rotate the iteration vector (harmless-ish; it's a search seed), and `reset_cache()`
does not clear it. Fold into the F-04 snapshot doctrine list with `_cached_*`.

### F3-23 — INFO — VERIFIED — optimizer per-arm coverage is COMPLETE except by-design exclusions: nothing is silently unoptimized
probeC-I: both arms cover 432/436 named_parameters; missing = exactly the four
`layers.*._vsa_tau_log` (documented M7 stack-override exclusion; 24 in production) —
they are live Parameters in forward fallback paths (block.py:401 `tau_s is None`) yet
never trained; in the stack flow they receive no grad (fallback branch unused) → no
grads, no state, but they DO occupy named_parameters and WOULD be silently unoptimized
if a future caller used the fallback. `requires_grad=False` filter in the eva branch
(:250) is order-safe in both loops (optimizer built before DepthController freeze —
train.py:220→224, notebook c9:43→77). `token_bias` IS in the optimizer (lr 3e-4,
role scalar; probeI) — closes 02a handoff (c); `embed.*/lm_head.readout`-class λ⁻²
confirmed: `embed.basis` 8.87e-5 + wd 0.01, `embed_mix` 8.87e-5, while `log_temp`,
`bit_bias`, `token_bias`, `_signal_log_weights`, `w_d_pen`, `gamma_surprisal`,
`log_tau_uncert` all sit at FULL 3e-4 (02a handoff (a) measurement complete: the
"deliberate?" question now has a number — basis decays 3.4× slower AND still decays
under wd 0.01 every step — consistent with F2A-09's observed norm erosion).

### F3-24 — INFO — VERIFIED — EVAAdamW update-rule line-audit vs paper: core math correct
step(): `m.lerp_(g,1−b1)` ✓; `v.mul_(b2).addcmul_(g,g,1−b2)` ✓; `bc1=1−b1^t`,
`bc2=1−b2^t`, `denom=(√v/√bc2)+eps`, applied `p.add_(u, alpha=−lr/bc1)` ✓ (the
bc1-scaling placement matches torch's `denom`/`bias_correction` algebra — and mode='adamw'
is the bit-parity branch, matching torch's ULP notes in-code); decoupled wd
`p.mul_(1−lr·wd)` BEFORE apply ✓ dim≥2 only ✓. τ-aware pieces: `slow_ema` OFF by
default (measured-group beta_slow_g=0.0051/slow_mix_g=0.041 present but gated by
`group["slow_ema"]`), `trust`: `0.5+0.5·clamp(t,0,1)` default 1.0 ⇒ **no-op unless
set_trust called** (train.py:619 calls; notebook doesn't → F2B-07 drift, honest effect =
immature-branch damping off in Colab), `update_cap` for τ-role = dev_max/(Δt·lr)=0.25 at
build lr — **computed once from build-time lr, then scheduler anneals group lr → cap
stale-quantum** (0.25 with lr=3e-4 is already far above |u|≈1·, so it never binds at the
top; it binds only for the deep τ-groups whose lr shrank… minor), `cautious` = mask + ÷√ρ̄
with the B2 null-band floor (matches F2B table row 21's measured energy ratios ✓), AdamP
= B2 per-row |cos|≥2/√fan_in rewrite, norm-preserving ✓ (deviation from paper: projects
anti-radial too — removes |cos| either sign; paper only suppresses radial-growth side;
documented in-code, defensible, flag for the constitution text). `state_dict` complete:
all EVA extras (role/cap/trust/layer_idxs) ride in groups; `st['step']` is a 0-dim float
tensor — round-trips. One fragility: because group *hyperparams* are restored positionally
with state (torch behavior), the same F3-08 count-guardless mapping can swap `trust_key`
between groups on an in-arm reshaping — one more reason for the by-name fix.

### F3-25 — INFO — VERIFIED — `LossBalancer.loss()`/`balance`-mode is production-dead; `align_cap` accepted-but-ignored (both loops still pass `align_cap=10.0`); `state_dict()` persists four always-None EMAs into best.pt
D5: no caller of `balancer.loss(...)` in either loop (only tests/test_product_invariants
calls it). `_update_balance` is invoked ONLY from `loss()` (:548) → in the production
align path `ema_ce/ema_A/ema_aux` never move → the envelope `balancer` key
(01 §2.6) carries `{None,None,{},True}` forever; `last_cos` has no reader (diagnostic
orphan). Verdict: either delete the balance-mode surface or wire `_update_balance` into
`backward()` if the resume-audit trail was supposed to include it (agent 5 note).

### F3-26 — INFO — VERIFIED — scheduler mid-ramp resume is CONTINUOUS (02a F2A-03's open question, tested): state_dict round-trip reproduces the LR sequence bit-exact through the blend segment
probeF2: save at step 900 (warmup 1200), new scheduler `load_state_dict`, both run 400 →
lr sequences identical (`_step` + warmup formula pure; τ-EMAs + `_ls_fast/slow` restored —
guarded on `_ls_enabled`); `_alpha_override`/`_usefulness_temp` buffers are *rewritten*
every step during the ramp (not read-only state) so the resume mid-ramp cannot leave them
stale; the `set_step()` fallback (no saved state — train.py:329/c9:65) is also ramp-
continuous (only EMA baselines re-bootstrap, benign: var_mult defaults 1/1=1). Parity of
the two copies here is good.

### F3-27 — INFO — VERIFIED — cosmetic: `verify_identity_resume(...)` return overwrites `missing, unexpected` (train.py:284); `print` at :339 lies about restore; `evaluate()` returns 0.0 (F3-01). All previously flagged by 02b; unchanged.

---

## 2. AUX-TERM LEDGER — canonical table

Probe basis: production-shaped mini (3–4 layers, D=256, vocab 1820, memory_bank+bridge+
maturation+UCL on), TRAIN window-1 + window-2 + eval; ratios = ‖g_term‖/‖g_CE‖ from
per-term `autograd.grad(allow_unused)` (A2/G); "balancer" = align (per-coord sign mask +
per-param ‖aux‖≤‖g_CE(param)‖) unless marked **bypass**. Sign = sanity of the
gradient direction vs the term's stated job. `train≡eval` = what the eval path does with
the term.

| term | weight source | grad live/dead (measured) | sign | train≡eval? | balancer | notes |
|---|---|---|---|---|---|---|
| **CE** (coded head, masked) | primary; `surprisal_weight` 0.3 nb / 0.0 py | ‖g‖=65.8 (ref) | ✓ NLL | train-mode adds surprisal reweight (×w̄≤1 renorm by mask — biases train CE DOWN vs eval metric; F3-18) | (the yardstick) | mask `t≠0` vacuous on real corpus (01 F-18); `h_emb` dead (F3-21); NOTE: the head's `bus_bias = bus_head_proj(stack._last_bus)` channel (losses.py:22–26) means the intent-bus head trains THROUGH CE — one of only two head-side aux-adjacent paths; `bus_head_proj` is also zero-init role 'zero_init' (wd-off) |
| **bridge_conn** | `bridge_conn=0.1` is a **model-build flag, not a loss weight** (raw emitted) | LIVE 1.75e-3, 5 params (probes only) | ✓ InfoNCE CE≥0 | recomputed at eval, discarded; cone-EMA guarded by `is_grad_enabled` ✓ | align | B2 design VERIFIED: center+detach target (bridge.py:200–204), in-batch negatives w/ duplicate-token mask (:219–226), learnable temp init λ⁻² clamp[0.05,2] (:205); +0.1·cross-layer-diversity magic baked inside (:236) |
| **pred** | none (raw) | LIVE 2.72e-2, 183 params | ✓ MSE≥0 | **absent at eval** (mirror :510 term=None) | align | M5 fix intact (live pred side, detached hp target :504) |
| **reinforce** | cfg `reinforce_weight=0.001` is a **no-op** (not read) | LIVE 6.3e-4, 210 | ✓ MSE(usefulness, gate.detach()) — only usefulness-predictor learns | recomputed+discarded | align | gate→detached ⇒ "align self-assessment to applied gate" |
| **alpha_novelty** | cfg `alpha_novelty_weight=0.05`: gates term AND drives hard `.data` push (mirror :481–489) | LIVE 1.42e-4, 3 | ✓ −var ⇒ spread | caches eval-live ⇒ recomputed | align | **dual-drive conflict F3-13**; config comment "no spectral" FALSE |
| **balance** | none | LIVE 2.6e-3, 217 | ✓ HHI−1/n, clamped ≥0 | recomputed | align | usage = forward gates ✓ B2 doctrine |
| **branch** | cfg 0.1 nb / 0.0 py ⇒ **term missing in train.py default**; train.py ALSO anneals ×step/5000 (magic) | **LIVE 1.08**, 239 — largest aux at init | ✓ logvar-pairwise² | caches from last TRAIN forward (eval: stale-but-unused) | align (+anneal in py only) | F3-15/F3-16; caches written with grad (block :367/521/552–553) |
| **diversity** (corr→I) | none; per-layer τ-authority weight `1−e^{−τ/τmin}` baked at :124 (principled); annealed py-only | LIVE 1.71e-2, 263 | ✓ | recomputed | align | scale-invariant (B2 covariance→correlation fix VERIFIED live) |
| **decorr** | none (always emitted) | LIVE 1.6e-3, 205 | ✓ cos² ≥0 | recomputed | align | reads live weighted signals (mirror :687–702) |
| **div** | cfg `div_weight=10` = gate only (10 never multiplies) | LIVE 4.4e-5, 3 | ✓ −var ⇒ spread | recomputed | align | config says "bypasses spectral" — FALSE (BYPASS_AUX=('gradalign',)) |
| **gate_l1** | none (always) | LIVE 5.0e-3, 217 | ✓ mean-gate ≥0 ⇒ sparsify | recomputed | align | also a watchdog channel (train+nb ✓ same) |
| **gate_repulse** | cfg 0.3 = gate only | LIVE 8.3e-4, 217 | ✓ −H(usage) uniformizes | recomputed | align | "bypasses spectral" FALSE |
| **signal_ent** | none (always) | LIVE 4.1e-3, 3 | ✓ sign = −H ⇒ maximize entropy (M5 fixed) | recomputed | align | consumes τ-scheduled `θ/τ_signal` (B2's "actual gate" consistency ✓) |
| **gradalign** | **cfg `gradalign_weight` REAL, applied in losses (:278)**; 0.3 hard-set nb (c10:26) vs 0.0 py ⇒ term nb-only | LIVE from window 2 (one-step-stale hook target ✓ by design) | ✓ MSE(rel dists) | **train-only** ✓ (:264) | **bypass** (BYPASS_AUX) | bound ≤‖p.grad‖ per param ⇒ ≤3× total (F3-04); CE-target freeze race in bypass-only path (latent) |
| **lbg_tau** | — | **DEAD — float** in `_cached_losses`/`aux_dict` (A2) | n/a | log-only | (none) | diagnostic, not a loss — constitution row to kill the naming |
| **lbg_diversity** | none | LIVE when `global_ready` (graph into LBG gate weights via fresh-leaf diag clone, losses :373–374/400–407 — target detached w.r.t. trunk ✓) | ✓ max−H ≥0 | requires `_layer_diagnostics` (train-forward populates) | align | post-B11 gate parity makes the nb eval/log path consistent |
| **intent_tau** | cfg `intent_tau_hierarchy_weight` 0.01 = gate only | LIVE 1.4e-3, **exactly 1 param: `_tau_l_dev`** | ✓ (actual−target)² | recomputed | align | M5 fix holds (actual side live); target τ detached ✓ — regularizes the ladder whose LR is itself mis-set (F3-14) |
| **mb_scale** | — | DEAD — float diagnostic (:423) | n/a | log-only | (none) | |
| **w_m2v** | cfg 0.01 = gate only | LIVE 4.6e-4, 3 (`w_mem2v`) | ✓ | recomputed | align | M5 fix holds |
| **signal… surprisal** | `surprisal_weight` — CE-internal, **train-only** | (in CE) | weights hard tokens w∈(0,1) | **train≠eval by design** (F3-18) | (in CE) | notebook-only magnitude (0.3 vs 0.0) |
| **nuc** | `nuclear_weight=1e-5` **pre-multiplied** in losses (:456) — the one remaining double-weighting pattern | ratio **1.7e-8, grads 0.0** — clamped at SR ceiling at init ⇒ inert tripwire | ✓ | recomputed | align | PI vector `_pi_v` plain attr (F3-22); bf16 PI precision unpinned (02b §4.5a carried) |
| **orth** | cfg gate (0.0 in cell 4 AND py default) | absent (off) | — | off | align-if-on | default parity ✓ both copies |
| **ls_reg** | **no cfg** — hinge threshold `2.3` magic (:284); cfg `log_scale_l2_weight` dead | absent at init (ls<2.3 ⇒ 0); LIVE under inflation | ✓ upper-hinge | recomputed | align | F3-14/F3-20 |
| **mem_tau_reg** | magic ×0.01 prior, ×0.1 inversion (:493–500) | LIVE on l1/l2 `log_tau` Parameters only (banks' keys/vals are buffers post-B7 ✓) | ✓ L2-to-prior + inversion penalty | recomputed | align | B7-conformant |
| **tau_dev_reg** | magic ×0.01 (:519) | DEAD at init (grad=0 at dev=0 — expected of dev²), LIVE once dev moves | ✓ | recomputed | align | "prevent one-sided collapse" — magnitude is a config-invisible constant |
| **pred_w** | — | **unreachable** — SigmoidCodedHead has no `pred_w` (A4) | — | — | — | dead code, delete with n_pred_w |
| **layer_gate_mean/std/min/max, lbg_global_ready, mb_*_overwrites, mb_l3_*** | — | DEAD floats (diagnostics) | n/a | log-only | (none) | `_cached_losses` merge feeds log lines, never grads |

Effective totals at the operating point (probeG, window 2, mini): summed raw aux
0.17·‖g_CE‖ (cos +0.058; global last_cos −0.166 on the summed-vs-CE geometry), post-
balancer 1.024·‖g_CE‖ total, post-AGC 0.987·‖g_CE‖. Sign-mask zero-duty measured only at
exact anti-correlation (ρ=−1 → 0.000; ρ=0 → 0.489 duty/0.692 norm retained).

---

## 3. Reactions to 01 / 02a / 02b handoffs + B6–B11 status in scope

### 3.1 01_data_path §2 (the contract) — consumed as given; two amendments
- §2.1 line "`ce = compute_losses(out, y, h_emb=h)` — targets flattened row-major" holds;
  **amend**: `h_emb` is not read (F3-21), `pred_weight` not read (F3-20), the coded branch's
  eval/train CE are different scalars when surprisal is on (F3-18), and the train.py
  evaluate() 0.0 sentinel (F3-01) poisons the LR controller, not just the best-save gate.
- §2.6 envelope: for my scope `optimizer(+param_names)` (F3-08 positionality),
  `scheduler` (F3-07 crash, F3-26 continuity ✓, F3-27(a) positional `orig_lrs`),
  `balancer` (F3-25 inert), `code_fp`/identity guard (B8 ✓ re-verified wired both).
  One new key-side fact: EVAAdamW group HYPERPARAMS (role/cap/trust) ride positionally
  in `param_groups` and are restored by torch on top of freshly-built ones (F3-24).
- §2.7 eval contract: verified from the loss side — eval calls `compute_losses` in both
  copies (train.py via `compute_loss` wrapper discarding aux; notebook unpacks and
  discards), so aux **producers** (bridge cones, lbg diag reset, `_pi_v`) do run at eval;
  the `is_grad_enabled`/`training` guards (bridge :202, mirror :510, losses :264/41)
  cover every *state*-mutating side effect I could find except `_pi_v` (F3-22) and
  `_step_count`-class rebinds under recomputation (F3-12).

### 3.2 02a §4→3 handoffs
- (a) factorized-branch scale claims for aux: none of my terms normalize against branch
  CE — `nuc`'s stable-rank and `orth`'s gram are on `bind.W_proj`, unit-clean; noted.
- (b) `compute_salience` sigmoid-of-log-probs (median −11 ⇒ ~1.7e-5, 02a): it feeds the
  **gate** (`sal·w_sal`), never a loss; its normalization-by-mean keeps salience≈1 despite
  scale — geometry review stays with B9 centering experiment as 02a said; no loss-side
  action.
- (c) h_emb two-ended contract: consistent with head branches (they DO accept
  `log_probs_for_target(h)` — the head's two inputs are trunk-h and codes), but the
  *embedding-side* argument `h_emb` is dropped by losses.py (F3-21). 02a's "losses.py:19–47
  is consistent" holds for what the code does, not for what the signature advertises.

### 3.3 02b §5→3 handoffs — every named item answered
- **"pen-unit fix changed surprise magnitudes feeding igate/UCL thresholds — re-check
  every threshold calibrated on the OLD inflated pen scale"**: done, F3-05/F3-06.
  Verdicts: igate ✓ back in band (+0.088); block-decay ✓ learnable (0.913 factor);
  **UCL u_gate ✗ now born-CLOSED** (fire 0.0000; threshold 2.718 vs pen max 0.675) —
  one-constant fix proposed; `conf=σ(−pen)` (:193) now ∈[0.34,0.5] — sensible again;
  mirror `contra`/`pred_scale_mod` unaffected (cos/EMA-based).
- **F2B-06 thresholds**: same audit — u_gate (above); `_usefulness_temp` M7 wiring +
  `max(temp,0.1)` floor verified still never ≤0 (lr_scheduler.py:188) so the eval/counter
  branch stays shadowed ✓ their read; `_pi_v`, `_damp_tau` cross-checks: `_pi_v` IS
  loss-reachable at eval (F3-22, new).
- **F2B-10 b_i/b_d dual-drive + AdaptiveController lerp targets 7.7–8.3 — mis-scaled
  post-B10?**: **No** — targets are pen-blind (`expl`/τ laws, adaptive_controller.py:91–128
  re-read line-by-line); the *write-rate invariant* they encode still assumes τ_eff=τ_nom
  which the pen arm (and σ(b_d) rest-floor) break — quantified F3-06 (τ_eff≈11 vs 1448;
  ~100× invariant miss at slow end, init); B12 redesign is the right home; ALSO found one
  NEW dual-drive coordinate of the same family on the mirror (`alpha_diag`: lerp + novelty
  push + aux loss) — F3-13.
- **F2B-05 `tau_dev_lr_mult` 'zero callers' claim**: confirmed true at B11 (F3-14) —
  `_tau_l_dev` measures lr 3e-4 role-1.0 in BOTH arms (probeC C1), and the only consumer
  (`stack.param_groups` → the 'tau_dev' bucket) has no callers. Extended the same audit to
  `gate_lr_mult`, `vsa_b_lr_mult`, `log_scale_l2_weight`, `mlp_mod_scale_reopen`,
  `accum_steps` — all dead; `per_layer_ls_lr` CLI flag dead-wired in train.py.
- **F2B-07's four divergences**: all four re-confirmed live (τ-LLRD post-clip,
  phase-block single-copy, llrd 0.9-vs-1.0, set_trust single-copy, gradalign 0.3-vs-0.0)
  and are rows D-04/D-05/D-08/D-09/D-10 in §4 below, plus two they didn't have
  (mlp-depth-boost hooks, per_layer_ls_lr dead flag, aux_anneal single-copy).
- **F2B-04/B11 parity**: the loss path is now consistent (A3: eval CE parity at
  sw=0 within noise; the gate the eval sees is the published combined gate — no
  eval-side loss defect found).

### 3.4 B6–B11 in-scope status (honesty ledger)
B6 grad_geometry — **shipped but crippled by the 8-term alphabetical cap** (F3-11); the
purity claim holds (probeB0: autograd.grad only, `.grad` untouched — and the recompute
side-effect caveat F3-12 belongs to it). B7 — veto ceiling single-source ✓ (both loops,
probe grep), L2 buffers ✓ (ledger `mem_tau_reg` note). B8 — guard wired ✓ (cosmetic
F3-27). B9 — off; ledger unaffected (when on, `signal_ent`/`div`/cosine channels move, as
02b said). B10 — pen fix landed; **its downstream threshold recalibration did not** (F3-
05/06 — this is my headline contribution). B11 — gate parity honored at the loss boundary
✓; the d_mod-centering revert rationale (dead log_tau grads) is consistent with my ledger
(τ-shapers are exactly the thin-gradient terms: intent_tau 1 param, w_m2v 3, div 3 —
fragile by construction).

---

## 4. Notebook vs train.py — FULL drift table (axis | train.py line | notebook line | severity)

H = high (changes optimized math), M = medium (changes numbers/cadence/state),
L = low/cosmetic/latent. Notebook lines = cell:index in the code cells (c4/c5/c7/c8/c9/c10
per the dump).

| # | axis | train.py | notebook | sev |
|---|---|---|---|---|
| D-01 | CE surprisal weighting | cfg default **0.0** (CLI never sets) | **0.3** (c4:58) | **H** — different objective (F3-18) |
| D-02 | gradalign | 0.0 (absent term) | `cfg.gradalign_weight=0.3` runtime (c10:26) | **H** — governance channel only in Colab (M5, still) |
| D-03 | optimizer arm | `getattr(cfg,'optimizer','adamw')` — CLI exposes none (:218) | `eva_proj` (c4:22) | **H** — cautious/AdamP/trust/role-wd all differ |
| D-04 | τ-LLRD (`lr_mult`) in update | **never applied** | applied **after clip** (c10:247; :243 clip first) | **H** — AGC invariant broken by ≤7.3× (F3-03) |
| D-05 | llrd_decay | 0.9 (cfg.llrd, :216) | 1.0 (c9:33) | **H** — different per-depth law (row: b_d L3 6.46e-5 vs 8.87e-5, probeI) |
| D-06 | set_trust (branch damping) | called (:611–619, floor 0.5 via tscale) | never → trust=1.0 | **H** — immature-branch damping OFF in Colab (F2B-07) |
| D-07 | mlp depth-boost param hooks | `apply_mlp_depth_gradient_boost()` :206 | never | **H** — extra grad plane |
| D-08 | mirror phase scaling `mir_s` | :556–580 | absent | **H** (train-only plane; near-invariant under Adam, F3-09) |
| D-09 | `per_layer_ls_lr` | **False hard-wired** (:828) — CLI flag :785 dead | True (c4:27) | M — train.py's ls code is dead, notebook's live |
| D-10 | aux_anneal (branch,diversity) | present (:532–537), τ=5000 magic | absent | M — early-step aux magnitudes differ (×0.1 vs ×1.0 at step 500) |
| D-11 | effective warmup/eval/log | λ-derived **101/233/55** post-clobber (C5) | **1200/55/1045** (c4:93–95) | **H** (F3-02) — cadence of everything LR/AGC/watchdog/depth |
| D-12 | seq_len policy | multi-scale curriculum 64→512 (:386–403; magic 32k/96k consts) | fixed 512 + OOM halving (c4:15, c10:108–113) | **H** — window length changes CE scale AND every EMA/ggeo stat |
| D-13 | batch | CLI default 2 | 1 (c5:2) | M (bridge in-batch pool N, aux means) |
| D-14 | AMP semantics | fp16-era: embed-only autocast + GradScaler(:240/434/538–541/603–609); loss×get_scale | bf16 full forward+loss autocast, `scaler=None` (c5, c10:86) | **H** (F-20/02b§4.8; when nb flips `_USE_AMP` the fp32-anchor gaps in the LOSS open: PI block unpinned) |
| D-15 | TokenStream vocab check | loud ValueError with `cfg.vocab` (:429) | loud check exists but **call sites never pass vocab** (c10:77/378) | M — B7 protection dead-by-arg in notebook (F-01 family) |
| D-16 | optimizer resume | positional `load_state_dict`; by-name restorer **dead** (:81 def, :320 call) | by-name `_restore_optimizer` wired (c8:125–168, c9:44) | **H** (F3-08) |
| D-17 | scheduler restore | :324–330 | c9:60–66 | M — same crash landmines both (F3-07), same `orig_lrs` len-guard (F3-27a) |
| D-18 | watchdog CE baseline on resume | **keeps** old-session CE stats (loads detector whole) | **pops 'ce'** → re-bootstrap (c9:101) | M (01 F-09 still open) |
| D-19 | NaN CE endgame | `raise RuntimeError` (:517) | skip-step (disp_loss non-finite, c10:219/259) | M (01 F-08 family; aux NaN unguarded in BOTH — poison hazard F3-16/E6) |
| D-20 | KeyboardInterrupt save | none (keeps last best.pt, :702–705) | saves envelope **minus `reasoning_enabled_step`** (c10:441–453) | M (01 §2.6) |
| D-21 | EOS state damp | ×0.1 on EOS-final (:629–633) | absent | M (01 F-18; drifts streaming state semantics between docs) |
| D-22 | `model.observe_output` placement | outside autocast (:440) | inside (c10:89) | L — salience bf16-vs-fp32 (detached both; scale-only) |
| D-23 | intent_state threading | not passed (:436) | passed (c10:88) | L — overridden by `_intent_stream` anyway (01 F-04) |
| D-24 | eval aux cost | CE via `compute_loss` (aux discarded internally) | `compute_losses`, aux discarded | L (same result; notebook pays aux build) |
| D-25 | eval start region | `len//2` (:736, comment "3/4") | `len//4` (c10:374) | M (01 F-17; val histories not comparable) |
| D-26 | depth resume fallback | `min(8+step//15000·4)` when no active_depth (:337) | `min(8+start//15000·4)` (c9:75–76) | L — same formula, notebook reads ckpt first |
| D-27 | log line | nested under `device=='cuda'` (F-15, :655) | always prints (c10:339) | L (silent CPU loop in train.py; aux_str content differs: py `_cached_losses` raw merge vs nb aux+cached merge with >1e-6 filter) |
| D-28 | memory governor | inside cuda block (:655–663) | c10:283–291 | L parity otherwise ✓ |
| D-29 | `_make_opt` cfg side-effect | none | `cfg.lr = float(lr)` (c9:30) | L (mutates pickled cfg identically-valued) |
| D-30 | align_cap=10.0 call | :227 | c9:83 | L (ignored field both) |
| D-31 | balancer construction | eval_interval=cfg.eval_interval | same | ✓ parity (value differs via D-11 only) |
| D-32 | ggeo inputs | post-anneal `ce_s/aux_s` | raw | L (ratios scale-invariant; anneal window differs) |

---

## 5. Not verified / handoffs

Not verified here:
1. Any CUDA-side behavior (bf16 loss numerics, TF32, GradScaler fp16 end-to-end) — CPU host.
2. Production-scale magnitudes (D=2560, 24 layers): all aux ratios/operating points are mini
   (ratios that depend on D through norms — e.g. `nuc`'s rank_ub=2560 clamp, `div`'s
   intra_weight=√(d/G) — should be re-measured on the A100 run via the [ggeo] line after
   F3-11's truncation fix).
3. The legacy best.pt (K32 step-1045) — trained-state drift of `w_d_pen`, `log_tau_uncert`,
   `gamma_surprisal` was NOT measurable on it (same caveat 02b left); F3-05/06's
   "self-healing is now real" claim is an init+learning-rate argument, not a trained-state
   observation. **Re-measure D1/D2 on the first real B10+ checkpoint.**
4. The generation/streaming path's loss-side effects (`process_with_cache` never computes
   losses; fine) — untouched.
5. Whether `stack.param_groups()`'s τ_dev/gate buckets *should* be revived vs deleted —
   a doctrine call (I only prove they're dead).

Handoffs to agent 4 (control plane):
- F3-01 (train.py fake-0.0 LR death) + F3-17(c) boost-gate laxity + F3-17(b) damp/restore
  units — the watchdog/plateau policy now shares the `_lr_damp_steps` crash surface.
- FailureDetector docstring still advertises "on trigger the loop rebuilds a FRESH Adam and
  rewinds the LR controller" (adaptation.py:44–46 + :311–315 "re-binding after rollback") —
  **D6 removed rollback; `LRController.rewind()` has ZERO callers** (grep: only its def and
  docstrings; `optimizer` setter's `_orig_lrs` re-snapshot machinery is therefore also
  never exercised). Either delete `rewind`+setter-rebinding or document them as recovery
  for the manual path; the stale comment is the same species as train.py:444/339.
- `apply_tau_lr`'s docstring "scaled … after AGC — the same mechanism the loop already
  used" (:99–106) *documents* the F3-03 order bug as intent — when fixing, fix the text too.
- F2B-07's trust-signal sources (bridge.readiness, maturation mean) are control-plane
  quantities read by the optimizer — coordinate with agent 5 on persistence if trust ever
  lands in ckpt.

Handoffs to agent 5 (persistence):
- F3-08 positional optimizer restore (train.py) + param_names saved-but-unused + the two
  `_restore_optimizer` dialects (pad vs truncate for W_out+K).
- F3-07 `_lr_damp_steps` crash + F3-27 `orig_lrs` count-guardless positional restore;
  scheduler-state schema note for the envelope audit.
- F3-12 `_step_count` 8× inflation lands in best.pt; `_pi_v` (F3-22) +
  `mirror._cached_hp/_cached_pred_error_norm` remain outside the snapshot set (02b's
  `model.reset_document_state()` proposal stands; from the loss side I additionally need
  `_pi_v` in it or `persistent=False`'d out of relevance).
- F3-25: the `balancer` envelope key currently preserves four `None`s — either populate or
  drop.

**Baseline suite at artifact completion: 299 passed (57.5–60s, CPU, clean tree before and
after; this audit modified zero repo files other than this document).**
