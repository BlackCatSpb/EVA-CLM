# REVISION_NOTES_block2 — закрытие очереди EVA-CLM перед A100 (блок 2)

Дата: 2026-10-10. Репозиторий: `C:\Users\black\OneDrive\Desktop\EVA CLM`.
Режим: изменения в рабочем дереве, **без коммитов**. Полный прогон:
**725 passed** (было 721; +4 новых теста), 2 pre-existing warning, 119 с.
torch 2.13.0+cpu, Python 3.14, `torch.set_num_threads(4)`.

Задача: корневой фикс recompute-примеси при `gradient_checkpointing=True`
(gc): checkpointed-регион обязан быть ЧИСТЫМ — recompute не меняет состояние.

---

## 1. ПРОБЛЕМА (до фикса, замер на мутационно-богатом мини-конфиге)

Конфиг: `n_layers=2, D=128, mlp_groups=4`, `memory_bank=True`,
`intent_bridge=True`, `unified_concept_layer=True`, `private_mem=True`,
`meta_trust=True`, `variable_precision=True`, `inner_eye=True`,
`explicit_reasoning=False`, B=2×L=16, 2 модели с одним сидом, без
optimizer.step (forward+backward, state переносится). Скрипт:
`%TEMP%\opencode\probe_block2.py` (+`_detail`, `_lockstep`, `_state`,
`_when`, `_multipass`, `_extra`).

| замер | 1 шаг | 4 шага |
|---|---|---|
| grad maxabs gc on/off | **5.914e-3** | **6.849e-3** |
| grad relL2 | 2.922e-4 | 6.596e-4 |
| dCE | 0.0 | 4.96e-4 |
| буферы расходящиеся | **51/149** | **70/149** (`_mlp_cnt` ровно ×2, +4 за 4 шага) |

Локализация (lockstep, шаг 0 бит-точен; расхождение начиналось на шаге 1 и
только в RECOMPUTE, first pass бит-идентичен):
1) `_mlp_cnt` double-count — recompute повторно исполнял observer;
2) `_pred_k`/`alpha_eff` drift 9.1e-4 — pending-flush `alpha_diag.data.lerp_`
   исполняется ПОСЛЕ чтения `pred_k` в том же forward; recompute видел
   post-flush значение (это же и корень F4-01 «alpha self-regulation ×2»);
3) `inner_eye.expert_bias` rel 1.0e-2 — общий `_o_rms_ema` (shared-модуль
   эволюционирует по слоям внутри forward, recompute идёт в обратном порядке);
4) при нескольких autograd-проходах за шаг (паттерн балансера) —
   `_prev_grad_norm`, обновляемый backward-хуком, читался recompute'ом
   следующего прохода (drift 2.2e-3).

---

## 2. ИНВЕНТАРИЗАЦИЯ МУТАЦИЙ checkpointed-региона

Регион: `EVAStack._checkpointed_block` → `EVABlock.forward` и весь вложенный
путь (`bind`/`mirror`/`mlp`/`inner_eye`). Стек-уровневые обновления
(`tau_config.update`, `b_i/b_d` lerp, bridge/bank/UCL/head) ЛЕЖАТ ВНЕ
checkpoint'а и recompute'ом не повторяются (подтверждено аудитом 04 §F4-01).

### Группа A. Накопительные мутации (EMA/счётчики; пропускаются при recompute)
`core/mirror.py`:
- update-then-read (запись пропущена; чтение видит S1 = прочитанное первым
  проходом): `_signal_norm_ema[i]` (:820), `_residual_var_ema` (:584),
  `_grad_norm_ema` (:864), `_delta_var` (:855; S0-читатель, см. B),
  `_fwd_count` (:887), `_ig_norm_ema` (:1027), `_ctr_norm_ema` (:1063),
  `_ls_var_run` (:1133), `_concept_sim_ema` (:711), `_behavior_div_ema`/
  `_div_run`/`_div_run_rec` (:715-724), `_trust_matrix`/`_prev_trust_matrix`
  (:719-721), `_meta_private_mem` (:726), `_pm_step` (:771), `_gate_ema`
  (:1148; S0-читатель, см. B);
- controller-запись: flush `alpha_diag.data` + rebind `_alpha_pending`
  (:600-611) — на recompute пропущены целиком (F4-01: ровно одно применение
  за шаг);
- `_private_mem` write (:790) и `_pm_coh` fill (:956) — пропущены (S0/S1, B).

`core/block.py`:
- `_pen_ema` (:744), `_cos_fs_ema` (:842) — update-then-read, запись пропущена;
- observer MLP: `_mlp_now_ema`/`_mlp_base_ema`/`_mlp_cnt`/`_mlp_cnt_py`
  (:1004-1018) — блок пропущен целиком;
- `_scan_floor_bound` (:694), `_traj_state` (:669/:674), `_chi_time` (:846) —
  sticky/carry/диагностика, пропущены.

`core/bind.py`:
- `TrajectorySpiralBind._step_count` (:422-427) — `_recomp` добавлен к
  `_ggeo_freeze`;
- `TrajectoryManifoldBind._push_transitions` (:621-649): `trans_buf`,
  `_trans_idx`, `_total`, `beam_centers/counts/age` — dry-run: те же
  граф-опы, тот же `randperm` при срабатывании пересборки (checkpoint
  восстановил RNG), НИ ОДНОЙ записи.

`core/inner_eye.py` (shared):
- `_o_rms_ema` (:45-60) — при recompute запись пропущена, чтение (post-value
  первого прохода) считается ЛОКАЛЬНО; зеркало на входе recompute ставит
  per-layer pre-call S_i, после вызова возвращает S_final (`core/mirror.py`
  :1084-1098/:1122-1124).

### Группа B. S0-читатели (чтение ДО записи; протокол «restore S0 / put-back S1»)
`core/mirror.py:_EARLY_READ_BUFS` (:430):
`_gate_ema` (читается InnerEye-фичами, пишется EMA), `_delta_var` (читается
`pred_scale_mod`/feats, пишется EMA), `_pm_coh` (читается write-gate,
пишется fill), `_private_mem` (читается help_k/учителем, пишется write),
`alpha_diag` (читается `pred_k`, пишется flush'ем — Parameter, `.data`),
`_prev_grad_norm` (читается `grad_mod`, пишется backward-хуком — put-back не
нужен, хук перепишет в конце своего backward'а).
На входе recompute: `_rc_pre[name]←S1`, затем `name←S0` (снимок первого
прохода `_fwd_snap0`); на месте записи S1 возвращается (кроме
`_prev_grad_norm`).

### Группа C. Идемпотентные set-кэши/граф-кэши
- **пропущены** (сохраняется значение первого прохода): `block._cache_conv_out/
  _cache_mlp_mod/_cache_bind_out/_cache_mirror_out/_cache_mlp_out/_cov_y_norm/
  _cov_h_norm/_precision_mean`, `mirror._pred_loss_term*`, `_cached_decorr`,
  `_cached_gate_l1/_cached_gate_usage/_cached_usefulness/_cached_gate`,
  `_cached_ig_eff_t`, `_last_mlp_mod`, `_last_magnitude/_last_gates/
  _last_h_pool`, `mlp._cached_group_out`;
- **оставлены** (пересчитываются бит-в-бит; это перенос состояния между
  шагами, а не накопление): `mirror._cached_hp/_cached_pred_k/
  _cached_pred_error_norm` + `_cached_*_buf` copies, `_pred_loss_term` —
  пропуск этих write'ов ломает перенос `pen` на следующий шаг (проверено:
  multi-step расходится) и нарушает строгий контракт non-reentrant
  checkpoint (число saved-tensors, см. §3).

`*` `_pred_loss_term`: граф-оп `F.mse_loss` обязан исполняться в recompute —
checkpoint сверяет число saved-tensors форварда/recompute (пропуск давал
`CheckpointError: 639 vs 637`); значение идемпотентно.

### Группа D. Плумбинг детектора (намеренно оставлен)
`block._ckpt_mark_seen`, `block._fwd_py_snap`, `mirror._fwd_snap0/
_fwd_py_snap/_rc_pre`, `_nan_at/_nan_logged`, `_gradalign_tgt` и хуки
(`h_mlp.register_hook`, hp-hook) — инструментация/детекция, не состояние
обучения.

---

## 3. МЕХАНИЗМ ФИКСА

- `_rc` (identity-маркер checkpoint'а, M65-opt2) остаётся детектором;
  `EVABlock.forward:513-526` протягивает `_sub._recomp=_rc` в
  `bind`/`mirror`/`mlp`; mirror — в shared `inner_eye`.
- Update-then-read: мутация пропускается, чтение видит S1.
- Read-before-write: S0→S1 протокол (группа B).
- Пропуск мутаций НЕ меняет граф: все пропущенные записи — no_grad/буферы/
  attr-rebind; единственное исключение (граф-оп `_pred_loss_term`) оставлено
  исполняться.
- Ошибки, которые пришлось учесть отдельно (честная хроника):
  1. non-reentrant checkpoint строго сверяет saved-tensors → нельзя
     пропускать граф-опы, даже «диагностические» (`_pred_loss_term`, FFT-опы
     манифолда) — исправлено dry-run'ом манифолда и возвратом mse;
  2. shared-модуль `InnerEye` требует per-layer S_i, а не общего S0;
  3. flush `alpha_diag` — Parameter-запись ВНУТРИ региона (F4-01): включён в
     S0/S1-протокол, после фикса rate ровно ×1;
  4. multi-pass балансера: `_prev_grad_norm` добавлен в S0-протокол.

---

## 4. КОНТРАКТ (замеры до/после, тот же мини-конфиг, один сид)

| метрика | ДО (HEAD) | ПОСЛЕ |
|---|---|---|
| 1 шаг: grad maxabs | 5.914e-3 | **0.000e+00** |
| 1 шаг: grad relL2 | 2.922e-4 | **0.000e+00** |
| 1 шаг: буферы diff | 51/149 | **0/149** |
| 4 шага: grad maxabs | 6.849e-3 | **0.000e+00** |
| 4 шага: grad relL2 | 6.596e-4 | **0.000e+00** |
| 4 шага: dCE | 4.96e-4 | **0.000e+00** |
| 4 шага: буферы diff | 70/149 | **0/149** |
| multi-pass (3×autograd.grad+backward): maxabs | 2.184e-3 | **0.000e+00** |
| изолированный конфиг audit-A5 | 6.9e-6 (остаток) | **0.000e+00** |

Доп. конфиги после фикса (4 шага): `traj_manifold=True` 0.0/0 буферов,
`cov_memory=True` 0.0/0, `n_layers=1` 0.0/0, `explicit_reasoning=True`
0.0/0. CE первого прохода бит-идентичен (dCE=0 на всех замерах).

OFF-контракт: gc=False после фикса vs pristine HEAD (git worktree, тот же
сид): выходы обоих шагов, eval-выход, все 209 градиентов и все 149 буферов
**бит-идентичны** — чистый путь не затронут. Bounded-путь: `test_bounded_
residual.py` (4 теста, вкл. gc=True) зелёный.

Eval-изоляция/снимок: `tests/test_snapshot_coverage.py` (21 атрибут,
restore/бит-точность) зелёный; новый
`test_m66_eval_snapshot_roundtrip_under_gc` — snapshot→eval-forward→restore
бит-точен под gc.

---

## 5. ТЕСТЫ

`tests/test_m66_recompute_purity.py` (+4):
1. `test_m66_checkpointed_gradients_bit_match_single_step` — одна модель,
   два backward (пересборка, gc вкл/выкл), maxabs < 1e-6 (замер 0.0;
   pre-fix 5.9e-3) + невакуумность (градиенты ненулевые);
2. `test_m66_recompute_does_not_advance_state_multi_step` — 4 шага: градиенты
   < 1e-6, ВСЕ буферы бит-равны gc=False, счётчики `_mlp_cnt`/`_step_count`/
   `_fwd_count` == 4 (double-count-замок), `alpha_diag` бит-равен (flush ×1);
3. `test_m66_multi_autograd_pass_bit_parity` — 3×`autograd.grad`+backward за
   шаг: градиенты+все буферы бит-равны (замок `_prev_grad_norm`);
4. `test_m66_eval_snapshot_roundtrip_under_gc` — снимок/restore под gc.

Обновлён устаревший комментарий `test_audit_agents_fixes.py::
test_checkpoint_gradients_match_uncheckpointed` (остаток «в очереди» —
закрыт; порог консервативный, жёсткий бит-лок — в m66).

**Саботаж** (`%TEMP%\opencode\sabotage_block2.py`; temp-копия `core/`+тестов,
порча → pytest нового файла обязан КРАСНЕТЬ): **4/4 RED** —
1) `_rc=False` в block.py → 3 failed; 2) `alpha_diag` убран из
`_EARLY_READ_BUFS` → 1 failed; 3) `_rc=False` в mirror.py → 3 failed;
4) `_prev_grad_norm` убран из `_EARLY_READ_BUFS` → 1 failed.
Непорченная копия — 4 passed.

Полный прогон: `python -m pytest tests -q` → **725 passed**, 0 failed,
2 pre-existing warning (requires_grad-scalar в audit_agents_invariants;
torch.tensor-tensor в t9_meta_ladder), 118.8 c. Арифметика: 721 + 4 = 725.

---

## 6. ФАЙЛЫ / ДОКСТРИНГИ

- `core/block.py` — preamble `_rc` (:501-526), guard'ы группы A/C;
- `core/mirror.py` — `_EARLY_READ_BUFS` (:430), preamble S0/S1 (:452-472),
  guard'ы всей группы A/C, inner_eye-протокол (:1084-1124);
- `core/bind.py` — `_step_count` `_recomp`, manifold dry-run;
- `core/inner_eye.py` — локальный post-value `_o_rms_ema`;
- `core/mlp.py` — `_cached_group_out` guard;
- `core/stack.py:1703-1731` — докстринг `_checkpointed_block` (роль маркера и
  идемпотентных кэшей);
- `core/config.py:632-649` — честные числа после фикса (0.0; до — 5.9e-3/
  6.8e-3, 70/149 буферов), ссылка на замок.

Научная часть (формулы, loss-математика) не менялась; forward-значения
gc=False бит-идентичны HEAD, first-pass значения gc=True бит-идентичны
gc=False.

---

## ОСТАТОЧНЫЕ ОГРАНИЧЕНИЯ (честно)

1. Замеры — CPU fp32. На A100/бf16 recompute может дать не бит-, а
   fp-расхождение (другие ядра редукции); семантика «recompute не двигает
   состояние» от железа не зависит, но паритет там не измерен.
2. Группа C «оставленных» кэшей (`_cached_hp/_cached_pred_k/
   _cached_pred_error_norm/_pred_loss_term`) на recompute пересчитывается, а
   не пропускается: это переносимые между шагами значения, их пропуск ломал
   multi-step паритет и saved-tensor-контракт checkpoint. Значения
   бит-идентичны (побочно: они не пишутся повторно только по значению).
3. `_prev_grad_norm` при recompute без backward'а (чисто диагностический
   forward) остаётся S0-восстановленным — эквивалент gc=False-поведения без
   backward; в training-пути хук всегда перепишет.
4. NaN-ветки (`_nan_ret`) в recompute могут оставить S0 у группы B до конца
   прохода — деградированный режим уже сломанного обучения (guard'ы не
   трогались).
5. Финальные 2 warning — pre-existing, не связаны с блоком.
6. Не коммитил (по требованию): все изменения в рабочем дереве; в репозиторий
   добавлен только `tests/test_m66_recompute_purity.py`; саботаж-скрипт и
   probe'ы — в `%TEMP%\opencode` (в репо не добавлены).
