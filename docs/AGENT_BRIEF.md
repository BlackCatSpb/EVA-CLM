**Кому:** внешний агент (помощь по обучению WideBind)
**Тема:** что сделано для успеха текущего прогона — gradient-reactive loss (`gradalign`) + fp32

**1. Контекст**
WideBind — нестандартная LLM-архитектура с метакогнитивным ядром (mirror/self-consistency, intent bridge, collective memory, VSA-лестница, explicit reasoning, variable-precision memory). Обучаем с нуля на Colab T4 в **fp32 (`use_amp=False`)** — AMP ранее ломал фазу «кризиса согласования» и расходил модель.

**2. Проблема, которую решили**
После выхода на хорошую валидность (val≈10.6) обнаружили, что гейты модуляции `mod_scale_mlp`, `w_sal`, `w_intent` «заморожены» на init: `sigmoid(mod_scale_mlp)=0.667` (= init). Модель при этом била рекорды, то есть заморозка была benign, но динамическая модуляция MLP/саленс не работала вообще.

**3. Диагностика (корень причины)**
Гейты **НЕ отключены структурно**: `requires_grad=True`, в optimizer, градиент доходит до весов MLP. Причина — они сидят в (почти) плоских областях лосса и лишены целевого сигнала: модель выучила всё нужное через τ-гейты / `intent_w` / веса MLP, оставив модуляцию на разумных init-дефолтах. То есть «градиента нет, потому что лосс от модуляции почти не зависит».

**4. Решение: gradient-reactive governance loss (`gradalign`)**
Идея: дать гейтам лосс «от экспертов, реагирующий на градиент с обучающих данных».
- В `core/block.py` (forward) кэшируется сырой MLP-выход `_cache_mlp_out` и пер-экспертный гейт `_cache_mlp_mod` `(B,L,G)`.
- В train-цикле после CE: `g_target = ‖∂CE/∂mlp_out‖` по экспертам (per-group norm, **DETACH — без 2-го порядка**), нормируется на max в слое.
- `L_gradalign = Σ MSE(mlp_mod_norm, g_target_norm)` добавляется как **bypass-aux** (прямо в градиенты, минуя spectral-alignment), давая прямой градиент в `mod_scale_mlp` и `usefulness_predictor`.
- Включается параметром `cfg.gradalign_weight` (0 = OFF, по умолчанию). Есть в `scripts/train.py` и в Colab-cell.

**5. Валидация**
- Mini-прототип (`WideBind Mini`, commit `f0aa2f5`): синтетика (order-2 Markov). Baseline `ga_w=0` → `mod_mlp=0.667` (точно заморожен); `ga_w=0.3` → `0.667→0.650` за 385 шагов, потеря стабильна. Гипотеза подтверждена.
- Main (commit `6383883`): те же кэши + поле `gradalign_weight`; smoke-тест — ненулевой градиент в `mod_scale_mlp`.
- Colab (live): `gradalign_weight=0.3` поверх `best.pt` (step 2796).

**6. Результат текущего прогона (step 2805→3520)**
- `mod_mlp`: **0.667 → 0.658** (монотонный дрейф вниз — гейт жив).
- `val_loss`: **10.4984 @ step 3495 — новый абсолютный рекорд** (старый 10.6043).
- `ce` устойчиво ниже случайного (11.09): ~10.1–10.5.
- `intent_w` растёт 1.46 → 1.585 (intent-bridge задействуется сильнее).
- `gradalign`-член стабилен (5.2→3.5), остальные aux без сюрпризов.
Вывод: gradient-reactive loss разморозил гейт **и** сопровождается улучшением валидности до рекорда.

**7. Что знать, если будешь править обучение**
- **fp32 обязателен** (`use_amp=False`).
- `notebooks/colab.ipynb` имеет **СОБСТВЕННЫЙ inline training-loop (cell 9)** — он НЕ вызывает `scripts/train.py`. Правки `scripts/train.py` на Colab не применяются; меняй саму ячейку. В cell 9 обязательна строка `batch_size = getattr(cfg, 'batch_size', 1)` (иначе `NameError`).
- В логах `loss` может быть **отрицательным** — это НОРМА для нестандартной композитной цели (куча aux-термов). Доверяй `ce` и `val_loss`, не `loss`.
- `cos_sim(diversity, CE)` в `analyze.py` даёт взорванные числа (~1e10) — игнорируй.
- `best.pt` — **живой** best-чекпоинт, перезаписывается при улучшении val (сейчас = step 3495).
- **OOM-риск:** `scripts/diag_gate_gradalign.py` (сравнение `mlp_mod` с целью gradalign) грузит 2 чекпоинта × 2.2GB + графы и вызвал OOM на ноуте; делать memory-safe (batch=1, no графы сразу).
- Журнал эксперимента: `docs/LIVE_TRAINING_LOG.md` (CRLF).

**8. Ключевые коммиты**
Mini `f0aa2f5` (прототип) · main `6383883` (порт) · журнал `7b7078d`/`8666422`/`9aecb38`/`d6647d1`/`5944160`/`dc7bbeb`.
