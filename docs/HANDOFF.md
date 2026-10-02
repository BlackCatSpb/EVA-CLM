# HANDOFF — эстафета архитектора (прочитать ПЕРВЫМ делом)

> Ты — архитектор и глава разработки нового типа ИИ. **CLM — cognitive learning
> model.** Мандат оператора: **никаких компромиссов**. «У нас нет не критических
> багов». Строгость к багам относится и к тебе. Политика «выглядит нормально» =
> провал. Либо архитектура доказывает полноценность, либо ты находишь ровно то,
> что её ломает. Это реальный проект, а не упражнение.

## 0. Где мы и как всё устроено

- **Песочница (рабочая копия):** `C:\EVA_CLM_OPT` — копия `C:\Users\black\OneDrive\Desktop\EVA CLM`,
  вынесена ЗА пределы OneDrive, **git remote отсоединён** (кампания не задевает живой прогон).
  Все коммиты — локальные, в ветке master.
- **Живой прогон** — на Colab (ноутбук `notebooks/eva_colab.ipynb`), синк через
  `gdrive:eva_clm/` (rclone; exe: `%LOCALAPPDATA%\Programs\rclone\rclone.exe`).
  Логи/история: `gdrive:eva_clm/logs/train_live.log`, `.../checkpoints/val_history.jsonl`.
- **Среда:** Windows, PowerShell 5.1 — НЕ используй heredoc (<<) и &&; пиши
  `.py`-скрипты в `C:\Users\black\AppData\Local\Temp\opencode\` и запускай
  `python file.py`. Python 3.14, torch **CPU-only** локально, pytest работает.
  Большие чекпоинты (15.9 ГБ) не трогать; мини-модели: `EVAConfig(D=128,
  n_layers=2, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=256,
  logit_cache_enabled=False, gradient_checkpointing=False, save_dir='.')`.
- **Ритуал:** любая правка → `python -m pytest tests -q` (полный набор, ~2 мин) →
  коммит в песочнице. Ожидаемое состояние: всё зелёное (~675 passed + 2 xfailed).
- **Скретч/доказательства агентов:** `_audit_A/`, `_audit_B/`, `_audit_C/`,
  `_audit_D/` (репро, замеры, отчёты — не удалять, это доказательная база).

## 1. НЕЗАВЕРШЁННАЯ РАБОТА (сделать в первую очередь!)

### 1.1 Проверить и закоммитить незакоммиченное (сделано ПОСЛЕ коммита 9023d14)
Изменены/созданы, НЕ проверены полным прогоном (прогон был прерван):
- `tests/test_audit_agents_invariants.py` — НОВЫЙ файл, 9 тестов: детерминизм
  свежих моделей; eval-изоляция end-to-end; by-name restore оптимизатора
  (состояние моментов бит-точно + CE в допуске 1e-4); state_dict roundtrip;
  decode_step самосогласованность (atol 1e-3); характеристика раскола
  window-vs-chunked; emphasis читает data-часть; ring ровно max_entries;
  τ-пути ценза не все None.
- `tests/test_tau_improvements.py` — `test_deterministic_forward` переписан
  строго (свежие модели бит-идентичны).
- `tests/test_b14_control_locks.py`, `tests/test_product_invariants.py`,
  `tests/test_m64_telemetry.py`, `tests/test_t9_cache_gate.py` (+import math),
  `tests/test_t9_phantom_upgrade.py` — тавтологии заменены строгими проверками.
- `core/stack.py` — расширен `snapshot_runtime_buffers`/`restore_runtime_buffers`:
  добавлены `bridge._preds` (одношаговый кэш — читается forward'ом!),
  `_cached_concept_dendrogram`, `_sal_q`, `_meta_thr`, `_meta_levels`,
  `_spike_n`, `_spike_stats`, `_last_p`; restore-маршрут знает префикс `bridge.`;
  списки клонируются поэлементно в снимке и свежо на каждый restore.
**Действие:** `python -m pytest tests -q`; если зелено — закоммитить с описанием
«ревизия тестов: тавтологии→строгие, +9 инвариантов, снимок полный».

### 1.2 Найденное свойство динамики (задокументировано в тестах, решить)
Forward — **хаотическое отображение**: состояние восстанавливается БИТ-точно
(доказано `_audit_A/probe_restore_gap.py`: 6 benign-расхождений, 0 тензорных),
но после forward+restore выход расходится на ~0.026 из-за fp-сдвига аллокации
(один поток — то же число; свежие модели с одним сидом — бит-идентичны).
Тесты используют: состояние — бит-точно, выходы — допуски (CE 1e-4, decode 1e-3).
**Очередь:** решить, нужен ли layout-stable/deterministic режим, или принять
как свойство. НЕ маскировать: в тестах стоят комментарии с замерами.

## 2. КРИТИЧНЫЙ ПРЕДСУЩЕСТВУЮЩИЙ БАГ (корневой фикс — главная задача)

**В чистом пути `gradient_checkpointing=False` bind-параметры последнего слоя
`layers.N.w_d/b_d/w_d_pen` МЁРТВЫ** (нулевой градиент). Они влияют только на
detached состояние; в gc=True градиенты им даёт ИМЕННО recompute-артефакт.
Т.е. «обучаемость bind» — иллюзия чистого пути (тест
`test_bind_pen_dead_in_clean_path` помечен xfail strict=True и ждёт фикса).
**Как чинить:** `core/bind.py:329-355` (определения), `core/block.py:667-690`
(применение decay `d_mod = sigmoid(h_v*w_d+b_d)/sigmoid(b_d)`), найти, где
градиент обрывается (detach/no_grad в применении decay или в state-апдейте),
провести путь CE→decay; затем вынести state-апдейты из checkpointed-региона
(чистый recompute без restore-костылей) и вернуть gc=False как дефолт.
Проверка: xfail-тест должен стать PASS, `test_no_unregistered_dead_parameters`
зелёным при обоих gc.

## 3. ОЧЕРЕДЬ ПРОИЗВОДИТЕЛЬНОСТИ (замеры агента B, `_audit_B/`)

1. **LossBalancer = 79.6% шага** (3 полных обхода графа: CE/aux/safety).
   Safety-обход исчезает после B1-фикса (микро-стена не эмитится, порог 1e-6) —
   проверить фактически на живом логе (bal_a/bal_b счётчики).
   Векторизовать сборку градиентов (2221 параметр × 3 dot + 2 norm = 0.168s)
   через `torch._foreach_norm/_foreach_mul_`.
2. **FFT вместо gather+einsum в bind** (`core/bind.py:463-466, 386-388`):
   37.0 → 0.74 ms/слой (50×), 18.9MB/слой; `index`+`_index_put` = 17.5% шага.
   Нужен fallback и проверка ошибки (err 3.8e-6 на кросс-корреляции).
3. **Gradient checkpointing**: +19% forward, 1.66× шага. После п.2 раздела 2 —
   рассмотреть дефолт False (точные градиенты + скорость).
4. Прочее из отчёта B: 5 лишних head-вызовов (V=65536 → 33.6ms каждый),
   intent-probe через среднее, двойной embed target, `.item()`-хвосты.

## 4. МЕТОДЫ (агент C, `_audit_C/audit_C_report.md`)

Вердикты: почти всё SUSPECT/DECORATION — **ни одной off-аблации на CE** для
mirror/maturation/bridge/UCL/phantom/cache/bind/reasoning/LossBalancer.
3 дешёвых эксперимента (сделать и оформить HTML-отчётом):
1. Атрибуция ветвей на замороженном чекпоинте (eval-only): full / cache off /
   VSA-mem off (scale_w=0) / bind off / mirror off → CE и Δctx.
2. Голова на frozen h (`scripts/probe_frozen_head.py` уже есть): current /
   emphasis_gain=0+log_temp=0 / head_read_full=True → CE и argmax==bias.
3. Скелет-аблация на мини (2-3k шагов): full vs (`unified_concept_layer=False`,
   `head_lacuna=False`, bridge injection off, `explicit_reasoning=False`,
   aux kill) → CE-трейл.
Топ-5 противоречий идеологии (в отчёте C): VSA vs verbatim KV-кэш 512; коды vs
хедж к приору (BIAS-DECOMP 100%, модель хуже униграммы); τ-лестница vs мёртвые
контракты (mat_delay/gate_tau без потребителей); метакогниция vs инертность
(mod_eff 0.0005 при гейте 0.395); открытый словарь vs чурн (births 10531/
retired 2273/median_verdict −38943) и ph_cos_p50=0.98.

## 5. ТЕСТЫ (агент D, мутационное тестирование)

Сделано: 3 фейковых assert'а убраны; PAD/all-PAD/restore-None/τ-сигнал/
blacklist закрыты; conftest автосид; `_srclock.py` (нормализация + AST) для
новых локов. **Осталось:** 11 source-локов → поведенческие (список в отчёте D,
файлы: test_b12/test_m32/test_m58/test_m42/test_m24/test_m39/test_m62/
test_t9_cov/test_product_invariants/test_b8/test_b10); STE-примитивы
(`StraightThroughRound/Quantize.backward` недостижим из-за `.to(uint8)` —
задокументировать/убрать); unseeded random в test_math_audit/test_model
(conftest сидит torch, но не random/numpy-global).

## 6. ЗАВЕРШЁННЫЕ КАМПАНИИ (контекст, не переделывать)

- **M65-OPT** (коммиты 19cb965…339cd19): 8 батчей — скрытые сбои, ~160
  host-sync/forward, пересчёты, мёртвый код + 15 unused config-полей,
  дубликаты (decode_step, _signal_weights), тесты (conftest/projector/
  tau_compression/losses), проводка `mlp_mod_scale_reopen` (дефолт 0.0),
  UCL pre-birth рычаг `ucl_birth_confidence` (дефолт 0.6), доки/ноутбук.
- **M65-OPT-2** (84d866b, 9023d14): аудит 4 агентов. Исправлено: A1 `_st`
  unbound (head_mode≠sigmoid); A2 резюм терял 27 non-persistent буферов
  (maxdiff 9.7e-2→бит-точно; снимок в конверте чекпоинта train.py+ноутбук);
  A3 off-by-one резюма (+1); A4 утечка salience/lacuna через eval; A5
  recompute-примесь (маркер-детектор вызова checkpoint + узкий restore EMA
  зеркала; 0.324→0.000, 5.8e-3→6.9e-6); A6 `--head` игнорировался; A7 B*L=1
  NaN; A8 пустой батч; A9 ms-аккумуляторы через clear(); B1 микро-стена
  покупала 3-й backward (22-25% шага) — порог 1e-6.
- **Доказанные факты:** eval-изоляция (snapshot/restore) работает бит-точно;
  by-name restore оптимизатора — 225 слотов, 0 skipped; decode_step самосогласен;
  forward НЕ chunk-инвариантен (окно 9 ≠ 8+1, замер 3.47) — кадровый раскол
  train-окна vs генерации L=1, тест-характеризация стоит, корневой фикс — в
  очереди (per-token каденция апдейтов).

## 7. ЖИВОЙ ПРОГОН (наблюдение)

Best: **7.8559@18920**, далее 19360 (7.8504 — лучший!), лог 14520→19360+;
Δctx последние eval +0.006…+0.044; bias-only растёт 7.873→7.894 (приор
деградирует); спайк-налог (2.6/шаг) — M65-кламп включён в ноутбуке;
мета: `_LAST_DCTX`/`dctx_own`/`dctx_mix` (предвыборка униграммы на старте);
инфра: FORCE_FRESH=False, резюм с Drive-best, eval_windows_budget=64.
Ноутбук при переносе правок из песочницы требует git pull + перезапуск ядра.

## 8. ПЕРВЫЕ ШАГИ ПОСЛЕ ПРОЧТЕНИЯ

1. `cd C:\EVA_CLM_OPT; git status --short` — увидеть незакоммиченное (см. 1.1).
2. `python -m pytest tests -q` — убедиться, что зелено (~675 passed + 2 xfailed).
3. Закоммитить незакоммиченное (ревизия тестов + снимок) — если зелено.
4. Взять задачу из раздела 2 (корневой фикс bind-градиента) — это приоритет №1.
5. Дальше по очереди 3 → 4 → 5, каждый шаг: полный прогон + коммит + регистр
   в `docs/ARCHITECTURE_JOURNAL.md`.


## 9. ПЛАН СЛЕДУЮЩЕГО ЗАПУСКА (критический путь)

Порядок:
1. ПЕРЕНЕСТИ кампанию M65-OPT из песочницы в главный репо (ноутбук запускается
   из C:\\Users\\black\\OneDrive\\Desktop\\EVA CLM). Песочница remote отсоединён:
   git -C C:\\EVA_CLM_OPT format-patch 0702466..HEAD -o C:\\Users\\black\\AppData\\Local\\Temp\\opencode\\m65patches
   затем в главном репо git am (там НЕзакоммичен birth_ledger + свои коммиты —
   разрешить конфликты; либо применить только нужные коммиты). Проверить, что
   в главном репо есть: _soft_floor (core/block.py), _hrr_conv (core/bind.py),
   A1-A9/B1 (core/*, scripts/train.py), снимок runtime (core/stack.py), тесты.
   Прогнать в главном репо python -m pytest tests -q (ожидание: 680 passed).
2. НОУТБУК (cell4): gradient_checkpointing=False — после корневого фикса
   чистый путь КОРРЕКТЕН и ~1.66-2.2x быстрее (замеры A/B); B=1, память
   проходит (комментарий самого ноутбука: no-ckpt at B=1 fine, 60-70 tok/s);
   риск — пики eval: eval_windows_budget=64 уже стоит, OOM-recovery (seq shrink)
   в cell9 есть. При OOM на старте — вернуть True (потеря 1.66x, не блокер).
3. use_amp=True — включать ПОСЛЕ первого eval (bf16-крах починен в B10b,
   но доктрина ноутбука: fp32-чистый первый eval). ~2x.
4. bind-телеметрия: убедиться, что в лог идут градиентные нормы bind по слоям
   (falsifier фикса: глубокие слои теперь ДОЛЖНЫ сдвинуться систематически).
5. После запуска: следить за Δctx (KPI) и val; первый eval ~440 шагов.


## 10. РАУНДЫ 3-5 АДВЕРСАРИАЛЬНЫХ АУДИТОВ (состояние: 686 passed, оба репо синхронны)

Кампания перенесена в main (23 патча, коммиты до 5a7395d), затем 3 раунда
аудитов с фиксами (main: df082ed, 1106d2b, 8c75a7f; песочница: e98d149,
6e2947c, ca000d0):
- Round 3 (7 находок): soft-floor ОТКЛОНЁН (делал decay>1: cum_decay 1.234/
  чанк, 28.9x/512 токенов при tau=4111) -> _FloorSTE (forward=hard clamp
  бит-в-бит, backward identity); bind_traj_dims=1 краш; снимок (tuple/dict
  типы, клоны, IndexError); тавтологичные тесты m20/m25 восстановлены.
- Round 4 (3): list-шаринг при restore -> клон; _gs_velocity=None краш на
  step>=5000 -> guard; тест-замок data_ptr. Саботаж 6/7, 1 пробел закрыт.
- Round 5 (3): краш head_lacuna=False+memory_bank=True+temper -> getattr-гард
  + регресс-тест; честные числа gc-примеси (single 1.17e-3, 4 шага 2.68e-2 —
  НЕ 6.9e-6, это цена OOM-fallback'а); коррекция атрибуции round-4 (причина
  не graph-кэши, а control-write b_i/b_d при adaptive=True; при adaptive=False
  повтор после restore бит-точен).
Готовность запуска: main 686 passed, gc=False, FFT-bind, STE, память
подтверждена (22.9/25.1 GB, запас 13.5-14.9 GB до 40GB), ноутбук обновлён.
Осталось (честно, НЕ закрыто): (1) корневой фикс recompute-примеси (вынос
state-апдейтов из checkpointed-региона) — gc=True остаётся fallback'ом с
примесью ~1e-3/шаг; (2) 19 непокрытых снимком write-before-read атрибутов
(контрпримеров нет, но контракт неполон); (3) eval-пик памяти с накоплением
кэша не измерен (оценка 7-10 GB); (4) очередь: LossBalancer (_foreach_* ~9%),
_last_conf x7 (~6-14%), 11 source-локов, эксперименты агента C.
