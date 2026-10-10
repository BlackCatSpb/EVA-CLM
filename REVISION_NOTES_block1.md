# REVISION_NOTES_block1 — закрытие очереди EVA-CLM перед A100 (блок 1)

Дата: 2026-10-10. Репозиторий: `C:\Users\black\OneDrive\Desktop\EVA CLM`.
Режим: изменения в рабочем дереве, **без коммитов**. Полный прогон: **721 passed**
(было 704; +17 новых тестов), 2 pre-existing warning, 138 c, без OOM (RAM в рамках
конвенции репо ~1 ГБ; torch CPU 2.13.0, Python 3.14).

---

## 1. SOURCE-ЛОКИ → поведенческие (11 файлов по отчёту аудита)

### 1a. `tests/_srclock.py` переписан (был мёртв — ни один тест не импортировал)
- `strip_comments` (`tests/_srclock.py:68`): комментарии удаляются через `tokenize`
  (спаны COMMENT заменяются пробелами) — `#` внутри строковых литералов больше НЕ
  вырезается; `norm` (:99-105) поверх этого сжимает пробелы. Старый `re.sub(r'#.*','')`
  удалён. Самопроверка: `x = 'a#b'` сохраняется, хвостовой комментарий режется.
- `calls` (:292-312): AST-проход, учитывает алиасы `b = f; b(...)`
  (транзитивно, плюс `from x import f as g`, см. `_collect_aliases` :185) и НЕ
  считает вызовы в мёртвых ветках: константно-ложные `if`/`while`/`IfExp`
  (`const_bool`) и код после безусловного `return`/`raise` в том же блоке.
- Точечные AST-проверки для локов: `has_call`/`has_call_in` (:321/:335,
  аргументы сравниваются по `ast.unparse` — эквивалентность узла, а не имя),
  `assigns`, `has_augassign` (:410), `has_if` (:419), `has_compare` (:434),
  `has_str` (:448), `str_values` (:457), `has_dict_entry` (:463).
- Остаточные ограничения задокументированы в докстринге модуля (строки 1-31):
  не отслеживаются `self.f = f`, `partial`, `getattr`, параметры-функции; алиас,
  переопределённый другим значением, даёт переучёт; мёртвыми считаются только
  константные ветки и код после безусловного return/raise в блоке (без
  dead-store/динамической недостижимости).

### 1b. Миграция локов (файл:строка; саботаж-проверка в 1c)
| Файл:строка | Что мигрировано | Стало |
|---|---|---|
| `tests/test_b12_regressions.py:84` | train.py wiring | `has_call('_restore_optimizer', args=[...])` (точный узел), `assigns('cfg.warmup_steps','args.warmup')`, `has_call('float',["'nan'"])` |
| `tests/test_b12_regressions.py:97` | порог u_gate | `assigns('self.log_tau_uncert','nn.Parameter(torch.tensor(-0.6931))')` |
| `tests/test_m32_eval_parity.py:62` | evaluate/цикл | `find_def('evaluate')` + `has_call_in` на reset_reasoning/global_state=ogs; ячейка — `has_call`/`calls` |
| `tests/test_m58_chain_locks.py:136` | train.py статик | `has_augassign(cfg.seq_len, floordiv)` (нет), `has_name('_gate_missing')`, `has_call('mul_', args=['ls_m'])` (нет); raw-маркеры watchdog сохранены (AST слабее) |
| `tests/test_m58_chain_locks.py:151` | FailureDetector | `find_def`/`has_name` отсутствуют |
| `tests/test_m58_chain_locks.py:160` | notebook resume | вызовы/присваивания/`has_if`/`has_dict_entry` вместо подстрок |
| `tests/test_m42_rolling_ckpt.py:27` | flush/энит | `has_compare(step%495,'==','0')`, `has_call(reset_cache/empty_cache)`, `has_str('val_history.jsonl')`, `len(call_sites('codebook_fingerprint'))==1` |
| `tests/test_m42_rolling_ckpt.py:40` | resume best | строковые ЛИТЕРАЛЫ (`has_str`): комментарий про latest.pt легален, живой литерал — нет |
| `tests/test_m24_never_stop.py:42-53` | interventions | обязательные вызовы через `has_call` (balancer.backward/clipper.clip/apply_tau_lr/scheduler.step/release_step_graph); FORBIDDEN-маркеры — raw (см. ограничение) |
| `tests/test_m39_math_audit_followup.py:29` | мёртвый хвост | последний стейтмент `_adamp_project` обязан быть `return u`; `has_name('uf')` ложно |
| `tests/test_m39_math_audit_followup.py:42` | --llrd | `has_call('add_argument', args=["'--llrd'"], kwargs={'default':'1.0','type':'float'})` + help-строка через `str_values` |
| `tests/test_m62_stream_rotation.py:27` | `_pick_stream` | функция компилируется из AST-узла FunctionDef (убран срез по `src.index`) |
| `tests/test_t9_cov_block.py:132` | resume-порядок | AST-порядок: decl до resume-блока; никакого `state=None/gs=None` с lineno > строки `_tstate(ckpt['stream_state'])` на любой глубине resume-блока |
| `tests/test_product_invariants.py:574` | TokenStream | класс компилируется из AST-узла ClassDef |
| `tests/test_b8_regressions.py:51` | двойной tanh | AST `LogitAttention.forward`: нет `torch.tanh(...)` с аргументом, содержащим `cached` |
| `tests/test_b10_spiral_and_ladder.py:63` | index_copy dtype | все `index_copy(0,...)`: 3-й аргумент обязан содержать `.to(`; проверено 4 вызова (старый текст ловил 3 — многострочный пропускал) |
| `tests/test_t9_cov_memory.py` | — | source-локов нет (всё поведенческое), миграция не требовалась |

README-проверки в m39 и FORBIDDEN-маркеры в m24 оставлены raw-поиском: README —
не Python; маркеры-строки/комментарии AST не видит вовсе (AST-версия была бы
СЛАБЕЕ исходной). Это осознанное «усиление вместо миграции».

### 1c. Саботаж (temp-копия репо, порча → тест обязан КРАСНЕТЬ)
Скрипт `%TEMP%\opencode\sabotage_block1.py` (в репо не добавлен): копия
tests/core/scripts/notebooks в temp, порча по одной, `pytest <node>`, откат.
**Результат: 17/17 мигрированных локов поймали порчу (RED), 0 escaped:**
1. b12 wiring: model→model2 в `_restore_optimizer` — RED.
2. b12 порог: `-0.6931`→`-0.5` — RED.
3. m32: `global_state=ogs`→`None` — RED.
4. m58: `cfg.seq_len //= 2` в хвост — RED.
5. m58: `class FailureDetector` — RED.
6. m58: удалён `depth.put_state(...)` из ячейки 9 — RED.
7. m42: `step % 495`→`494` в ячейке 10 — RED.
8. m42: литерал `'latest.pt'` в train.py — RED.
9. m24: `balancer.backward`→`backward2` в ячейке — RED.
10. m24: `sys.exit(2)` в train.py — RED.
11. m39: мёртвый `uf = u.reshape(-1)` — RED.
12. m39: `--llrd default=0.5` — RED.
13. m62: `_pick_stream` возвращает текущий жанр — RED.
14. t9_cov: `state = None` после restore в resume-блоке — RED.
15. product_invariants: get_batch не отдаёт offset после wrap — RED.
16. b8: `torch.tanh(cached_dummy)` в forward — RED.
17. b10: `.to(keys.dtype)` убран — RED.
Дополнительно саботированы новые гарды (задачи 2-3): combined-бонд отключён,
NaN-гард aux отключён, `+1e-8` в nb возвращён, last_scale не сбрасывается,
`_cached_gate_l1` выпал из снимка, `include_caches=False` не фильтрует —
**итого 23/23 RED.**

---

## 2. БАЛАНСЕР (`core/training_control.py`)

### (a) NaN/Inf-гард
- Счётчик `n_nonfinite` (:323); помощники `_finite_flags` (:393) и
  `_drop_nonfinite` (:401) — 0-dim bool-флаги пачкой (один sync), non-finite
  вклад параметра → `None` (вклад 0). Применено: cheap-путь (non-finite aux
  ЗНАЧЕНИЕ не входит в total, :626), align-путь aux (:730), bypass (:791),
  `_add_safety` (:816, стал instance-методом). Телеметрия выведена:
  `scripts/train.py:701` (`bal_nf=`), ячейка 10 ноутбука (та же строка).
- Тесты `tests/test_balancer_guards.py`: inf/NaN aux (:33,:52) — `p.grad` =
  только CE (2.0, не inf); cheap inf-значение (:62) — `b.grad = 2 + s·0.2`;
  bypass inf (:77), safety inf (:87) — конечны; счётчик ≥1; нормальный путь —
  `n_nonfinite == 0` и числа прежние (:97).
- Саботаж: гард aux отключён → тест RED.

### (b) Bypass-бонд: combined per-param ≤ 2·‖CE‖
- Реализация: после добавления bypass общий не-CE вклад `Δ = g_final − g_CE`
  рескейлится к норме ≤ ‖g_CE‖ (:799-813); при Δ ≤ ‖CE‖ множитель ровно 1.0.
  До фикса адверсариальная сумма достигала 4·‖CE‖ (aux ≤ ‖CE‖ поверх CE, затем
  bypass ≤ ‖p.grad‖ поверх обоих). Докстринг класса обновлён (:192-210):
  sign-mask-инъекция (не PCGrad-проекция) + combined-бонд.
- Тесты: адверсариал aux=bypass=100×CE → `b.grad ≤ 2·2 + 1e-5` и `> 2.001`
  (:108); under-bound не рескейлится (2.2 точно, :125); независимые нормы
  на двух параметрах (:150). Саботаж (`_sc = 1.0`) → RED (было 8.0 при 2·CE=4).

### (c) `+1e-8` в `nb` убран
- `core/training_control.py:753-765`: `nb` — сырой `sqrt(den_b)`; при `nb==0`
  или `na==0` cos = 0.0 без деления; порог `nb >= scale_min_ratio·na` считается
  по сырому nb, как и раньше.
- Тесты: gau≈1e-12 → cos > 0.999 (старая формула давала ~1e-4, :136);
  порог не сломан: nb/na≈1e-12 и 0.049 → scale_ema None, ровно 0.05 → сеется
  с cap scale_max (:150). Саботаж (возврат старой cos-формулы) → RED.

### (d) last_scale + докстринг
- `self.last_scale = None` до невалидного/отсутствующего замера (:750), scale
  ставится только при валидном gate. Докстринг LossBalancer описывает
  cos-масштаб как implementation detail, фактическая реализация — per-param
  sign-mask.
- Тесты: невалидный замер после валидного → `last_scale is None`, EMA не
  тронута (:181); нулевой aux-градиент → cos=0.0, без деления, без сева (:194).
  Саботаж (`self.last_scale = None` → `pass`) → RED.

---

## 3. СНИМОК: инвентарь write-before-read (~20 атрибутов)

`core/stack.py`:
- `snapshot_runtime_buffers(include_caches=True)` (:1431): головные атрибуты
  добавлены в :1508-1515 (`_ext_phantom_dirs`, `_last_ph_sat`, `_tokens`,
  `_kp_active_py`, `_ell_ema_ready`, `_pb_active`, `_temper_active`,
  `ucl._mature_py`); `lcache.cache._position` (:1549); per-layer (:1552-1578):
  `blk._cache_mlp_out/_cache_mlp_mod/_fwd_py_snap/_tau_norm`,
  `mir._cached_gate_usage/_cached_gate_l1/_cached_decorr/_last_mlp_mod/
  _tau_signal_used/_pred_loss_term/_fwd_py_snap`, `mlp._cached_group_out`.
  Клонирование — `_snap_value` с detach (:1657; снимок не пинит граф шага).
- `restore_runtime_buffers` знает префиксы `mlp.` и `ucl.` (:1596-1609),
  `lcache.cache._position` (:1629).
- `include_caches=False` (:1580): forward-граф-кэши (10 имён, `_CACHE_ATTRS`
  :1460) не кладутся в ПЕРСИСТЕНТНЫЙ снимок — их клонирование в прод-конфиге
  стоило бы ~0.5 ГБ на best.pt; они write-before-read и пересоздаются первым
  forward после резюма. Обновились вызовы: `scripts/train.py:763` (envelope
  best.pt), `core/birth_ledger.py:117`, `core/regulator_ledger.py:222`; ячейка
  10 ноутбука (save-вызов). Eval-изоляция (`evaluate`, `_rt_snap`) — default
  True, покрытие полное.
- Итог инвентаря: 21 позиция (перечислены в `tests/test_snapshot_coverage.py:22-37`).
  Прочие кандидаты state_diff_full.txt (`_last_salience`, `bridge._preds`,
  `lm_head._tokens` и др.) уже были закрыты ранее/закрыты сейчас.

Тесты `tests/test_snapshot_coverage.py`:
- покрытие инвентаря default-снимком и материализация живым forward'ом (:78);
- `include_caches=False` исключает ровно граф-кэши (:90);
- после restore значения бит-равны снимку, `data_ptr` отличается (свежий клон),
  мутация модели не портит снимок, повторный restore работает (:103);
- eval-изоляция: все 21 атрибута бит-точно восстановлены после eval-forward,
  выход после restore в ≥4 раза ближе к train-эталону, чем val-ветка
  (`d_after < 0.25·d_eval`, :162; абсолютный хаос ~0.1 — известный
  pre-existing fp/аллокаторный эффект, зафиксирован в журнале).
- Саботаж: `_cached_gate_l1` выпал из снимка → RED; `include_caches=False` не
  фильтрует → RED.

---

## 4. ПОЛНЫЙ ПРОГОН

- `python -m pytest tests -q` → **721 passed**, 0 failed, 2 warnings
  (pre-existing: requires_grad-scalar в audit_agents_invariants;
  torch.tensor-tensor в t9_meta_ladder), 137.9 c.
- Арифметика: 704 (базовые из журнала «Предзапусковые пункты 1-4 закрыты») +
  13 (`test_balancer_guards.py`) + 4 (`test_snapshot_coverage.py`) = 721.
- Научная часть (формулы обучения, loss-математика) не менялась: правки —
  гарды, докстринги, снимок, тесты.

## ОСТАТОЧНЫЕ ОГРАНИЧЕНИЯ (честно)
1. `_srclock.calls` — базовый AST-проход (ограничения перечислены в п.1a);
   полная замена оставшихся текстовых локов на поведенческие не входила в блок.
2. Точный список «20 непокрытых» в docs не сохранён явно: восстановлен по
   `_audit_A/state_diff_full.txt` («=== baseline: 20 unrecovered tensor attrs ===»)
   и списку задачи; покрыто 21 имя (сверх — `_pred_loss_term`, `_tokens`,
   lazy-флаги, `_fwd_py_snap` оба, `_tau_norm`, `_position`).
3. Logit-кольцо (`_kv_h`/`_kv_sent`/…) в снимок НЕ входит (кроме `_position`) —
   действующий контракт `evaluate()` чистит его явно; тест это воспроизводит.
4. Выход после forward+restore расходится ~0.1 (train-режим, малая D):
   pre-existing fp/хаос-эффект; контракт проверяется на уровне состояния
   (бит-точно) + относительного сравнения выхода.
5. Не коммитил (по требованию): все изменения в рабочем дереве.
