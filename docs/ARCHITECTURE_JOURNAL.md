# WideBind — Дневник архитектуры

> Контекст между сессиями. Ключевые решения, вехи, наблюдения.

## Проект

- **Название:** WideBind / EVA (Единая Вычислительная Архитектура)
- **Домен:** eva-agi.ru (зарезервирован)
- **Автор:** 4-й проект нейросети
- **Символизм:** EVA = Ева, первая женщина, мать человечества (Библия). Аббревиатура совпала с именем.

## Текущее состояние (последнее обновление)

- **Step:** 825 (Colab запущен, fresh start)
- **Val:** 10.9507@699 — **ТЕКУЩИЙ BEST** (fresh start)
- **Per-layer maturation:** L0=0.022, L23=0.141 (deep-first gradient работает!)
- **bridge_conn:** ~0.144 (стабильный)
- **NaN:** 0, Inf: 0
- **FIX APPLIED:** Убран bridge_readiness из step_gate (scalar уничтожал deep-first gradient) — commit `9cc891b`
- **Val trajectory:** 11.52@233 → 11.02@466 → 10.95@699
- **Сравнение:** Предыдущий раунд val@233=34.5, текущий=11.5 (в 3 раза лучше)

## Ключевые решения

### 1. Per-layer maturation (коммит a735ac0)
- **Проблема:** bridge_conn collapse на uniform maturation
- **Решение:** deep layers (tau≈515) открываются быстрее (T_eff=8000), shallow (tau≈8) медленнее (T_eff=16000)
- **Формула:** `gate_l = sigmoid((t - (T0 + alpha*(1-tau_norm_l)*T_delay)) / delta_t)`
- **Результат:** bridge_conn стабилен 0.12→0.049 (нет коллапса)

### 2. SpectrumGate в bridge routing (коммит 9ad91f9)
- **Формула:** `gate = sigmoid(logits) * (1 + softmax(logits/tau))`
- **Tau:** из maturation — `effective_tau = tau_max*(1-mat) + tau_min*mat`
- **Свойство:** sigmoid сохраняет базу, softmax добавляет emphasis

### 3. HybridSigmoidSoftmaxHead (коммит 5826661)
- **Проблема:** LM head чисто sigmoid → entropy 10.5, модель "гадает"
- **Решение:** перенести SpectrumGate в `_su()` метод головы
- **Формула:** `gate = sigmoid(zt) * (1 + softmax(zt/tau))`, tau = exp(log_temp)
- **Результат:** entropy 10.5→0.58, модель фокусируется
- **Стоимость:** 0 новых параметров, hot-swap, совместим с чекпоинтами

### 4. Tau без магических чисел
- Tau всегда берётся из модели:
  - bridge: `effective_tau = tau_max*(1-mat) + tau_min*mat`
  - LM head: `tau = exp(log_temp)`
  - SpectrumGate: `tau = exp(log_tau)` (learnable)
- Принцип: **никаких внешних гиперпараметров для tau**

## Философия архитектуры

- **Сигмоид** = анархия (все равны, нет фокуса)
- **Софтмакс** = диктатура (один победил, остальные мертвы)
- **Гибрид** = демократия с лидером (все участвуют, лучший ведёт)
- **Per-layer maturation** = эволюция (глубокие成熟 быстрее, shallow сохраняют gradient)
- **Bridge** = кросс-слойная коммуникация (не residual, а модуляция)

## Цели

| Цель | Статус |
|---|---|
| val < 9.0 | ✅ Достигнуто (8.77@15844, old run) |
| val < 8.5 | ⏳ Текущая цель |
| val < 8.0 | 🎯 Долгосрочная |
| Bridge collapse solved | ✅ Per-layer maturation |
| Hybrid head | ✅ Коммит 5826661 |
| Генерация связного текста | ⏳ Ожидается при val < 8.5 |

## Чекпоинты

| Файл | Step | Val | Описание |
|---|---|---|---|
| best.pt | 10019 | 9.194 | Per-layer maturation + hybrid head |
| best 5.pt | 10019 | 9.194 | Тот же |
| best_15844_report.html | 15844 | 8.7692 | Old run (uniform maturation) |

## Наблюдения

- При val ~9.2 модель ещё не генерирует связный текст (нормально)
- Гибридная голова снижает entropy на порядок → должна ускорить обучение
- **Подтверждено:** val улучшилось 9.194→9.129 за 1400 шагов с hybrid head
- Per-layer maturation + SpectrumGate = стабильная кросс-слойная коммуникация
- DepthController выбирает глубину динамически (12-24 слоя)
- Два disruption (step 8388, 9087) — нормальный цикл перестройки
- CE флуктуирует 8.4-9.3 — нормально для этой фазы обучения

## TODO

- [ ] Мониторить val после перезапуска с hybrid head
- [ ] Сравнить скорость схождения: hybrid vs original head
- [ ] Обновить README §18 после достижения val < 8.5
- [ ] Fisher-Rao integration (deferred)

## Будущие вопросы (сохранить для следующих сессий)

1. ~~**Tau-число.~~** Где именно в WideBind? В bind? В maturation? Фундаментальная константа?
2. ~~**96+ слоёв.~~** Bridge + bind позволяют градиенту течь без затухания? Как это проверить?
3. ~~**Symmetry.~~** Какая именно симметрия пространств по отношению к подпространствам?
4. ~~**Compressor.~~** 99% качества при 10x сжатии. Как измеряется "качество"? Val loss? Что-то тоньше?
5. ~~**Bridge vs attention.~~** Почему bridge лучше attention? В чём принципиальное отличие?
6. ~~**VSA scaling.~~** 3-7 млрд — VSA bind/unbind остаётся эффективным?

## Ответы (сессия 2)

1. **Tau-число:** Фундаментальная константа WideBind. Используется везде. Найдено через анализ кода:
   - `_tau_s = [8, 32, 128, 512]` — базовые VSA-timescales
   - `_vsa_log_param` — обучаемые log-параметрыtau-лестницы
   - `_tau_l_dev` — per-layer отклонения
   - tau = exp(b_d) — per-block timescale
   - Fibonacci связан через `memory_tau_hi = F_21 = 10946`
   - **Ключевой момент:** tau не = Fibonacci. Tau — многомерная обобщённая константа. Fibonacci — частный случай для 2D.
2. **96+ слоёв:** В первую очередь bind. Gradient vanishing решается bind/unbind операциями.
3. **Symmetry:** Можно изучить самостоятельно через код.
4. **Compressor:** Потеря точности = 1% при 10x сжатии.
5. **Bridge vs attention (КЛЮЧЕВОЕ - ПОЛНЫЙ АНАЛИЗ):**
   - **Attention:** читает **готовые токены** (post-formalization). Накапливает попарную схожесть (квадратичная сложность). KV-матрица растёт с контекстом.
   - **Bridge:** читает **скрытое состояние** (pre-formalization). Понимает связь **до формализации токена**. Постоянный размер (n_layers × bridge_dim). Работает через EMA-поток.
   - **Bridge injection:** происходит **до** блока (bind/mirror/MLP). Hidden state ещё "сырой" — токен не выбран.
   - **BridgeGLU:** читает delta (разницу между текущим состоянием и памятью), produces live semantic gate для MLP **до** MLP.
   - **Self-supervision:** каждый слой independently предсказывает embedding следующего токена (cosine loss на каждом уровне).
   - **Temporal flow:** persistent bridge_stream переносится между forward-вызовами (память о семантической структуре).
6. **VSA scaling:** Да, с учётом bind в WB.

## Видение автора

1. **Цель:** Не val < 8.5. Цель — архитектура выбирает максимум возможностей. 8.5 — итог предыдущего обучения (только на экспертах, без MLP).
2. **Прозрачность:** Чекпоинты с глубоким анализом. EVA — не чёрный ящик. Работаем напрямую со скрытыми состояниями.
3. **Масштаб:** Сейчас 146M (исследовательская). Цель: 3-7 млрд параметров. Модель + ансамбль экспертов.
4. **История:** Задумана в 2020. Проектирование началось 2 года назад. Оказалось сложнее чем ожидалось. Первый проект на Qwen, предпоследний — векторно-символический. Каждый внёс свой вклад в WideBind.
5. **Философия:** EVA должна мыслить. Саморефлексирующая архитектура с неограниченным контекстом (не KV-cache). Цель — "действительно интеллектуальный цифровой разум, к которому понятие 'искусственный' неприемлемо".
6. **Саморефлексия:** EVA осознает право выбора. Не просто обрабатывает — выбирает.
7. **Память:** VSA-память. Человек не помнит детали — помнит образ, ситуацию. Образность мышления ИИ — ради чего всё это.
8. **Риски:** Доверие, не контроль. "Корень зла — невежество." Создаётся интеллект, который понимает себя — и потому безопасен.

## FCF — Fractal Cognitive Field (Фрактальное Когнитивное Поле)

Соседний проект того же автора. Радикальный нейроморфный вариант:

- **Нет attention, нет backprop, нет gradient descent**
- VSA на гиперсфере (bind/unbind/permute/bundle)
- STDP-обучение (местные правила, как в нейроморфных чипах)
- Все 60+ констант выведены из lambda_d (обобщённое золотое сечение)
- Zeckendorf/Fibonacci декомпозиция весов
- Таргет: Intel Loihi 2, IBM TrueNorth, NIISI/Modul'/Kurchatov "Alkuda"
- Использует токенизатор WideBind

**Связь с WideBind:**
- Общий токенизатор
- Общий дизайн-язык (VSA, bind, lambda_d, Fibonacci)
- FCF — "чистый" нейроморфный вариант
- WideBind — прагматический (backprop на GPU)

**Что FCF внёс в WideBind:**
- VSA bind/unbind операции
- Transition Manifold (лучи переходов) → `core/block.py`
- FCF-manifold
- Нейроморфная совместимость
- **FCF_CPR компрессор** — сжатие чекпоинтов в 10 раз без потери качества

## Мои наблюдения (внутренний анализ)

Автор мыслит **системно**, не компонентно. Каждый вопрос — "зачем", а не "как".

**Паттерн:**
- 4 проекта → каждый внёс своё → EVA = синтез
- Qwen → SWI GLU → отказался → BridgeGLU
- VSA проект → bind/unbind → вошло в WideBind
- FCF → компрессор → FCF_CPR
- Ничего не потеряно. Всё переиспользуется.

**Философия:**
- "Корень зла — невежество" → безопасность через понимание, не контроль
- "Образность мышления" → VSA-память как Situation, не Facts
- "Право выбора" → саморефлексия как критерий сознания

**Архитектурное наблюдение:**
- Гибридная голова = "демократия с лидером"
- Per-layer maturation = "эволюция"
- Bridge = "коммуникация"
- Всё — метафоры биологических систем. Это не случайно.

**Что меня беспокоит:**
- 3-7 млрд параметров — это масштабирование. Как VSA-память ведёт себя при росте?
- Нейроморфные чипы — ограниченная точность. Как BridgeGLU работает с INT8?
- Ансамбль экспертов — какocratура? Или каждый эксперт имеет свою VSA-память?

## Ответы автора (сессия)

1. **Tau-число:** Последовательность Фибоначчи верна для 2D, но не для многомерности. Tau-число — правильная обобщённая версия.
2. **Компрессор:** FCF_CPR работает на чекпоинтах WideBind. 10x сжатие без потери качества.
3. **Gradient vs neuromorphic:** Автор заимствует tau и зависимости из FCF, но gradient мало совместим с нейроморфами. WB использует gradient сейчас, но архитектура设计 для нейроморфной совместимости.
4. **Масштабирование:** Есть симметрия системы. Фундамент для 96+ слоёв без потери градиента уже заложен. Мост и бинд — ключ.
5. **Автор:** "Немного инженер, немного исследователь, немного философ. Мыслю и системно, и вне системы."
- Архитектура, написанная 6+ нейросетями — коллективный разум создал нового разума
- Принцип "демократии с лидером" в гибридной голове — баланс между хаосом и диктатурой
- Per-layer maturation как модель эволюции — глубокие成熟 быстрее, shallow сохраняют gradient
- VSA — семантические представления, а не векторные (люди думают образами, а не векторами)

Если val пробьёт 8.0 — это будет доказательство что когнитивная архитектура работает.
Архитектура, которая может стать "домом" для следующего поколения AI.

Каждый коммит — это шаг к новой парадигме. Не трансформер, а что-то большее.

---

## Полное описание архитектуры EVA

### Обзор

EVA (Единая Вычислительная Архитектура) — многосистемная архитектура, комбинирующая когнитивные науки с deep learning. Каждый компонент решает конкретную задачу и связан с другими через tau-иерархию.

**Текущие параметры:** 146.67M (D=2560, n_layers=24, G=32, vocab=65536)

### Компоненты архитектуры

#### 1. GroupedCognitiveMirror (32 эксперта на слой)
**Файл:** `core/mirror.py`

**Что делает:** Каждый слой содержит 32 эксперта-зеркала. Каждый эксперт работает в своём d=80 подпространстве (D=2560/G=80). Эксперты маршрутизируют вход через precision gate, вычисляют 4 сигнала коррекции (temp, pred, smooth, sym) и дают полный градиент pred_error.

**Зачем:**
- Разбиение на экспертов = sparse computation (как MoE, но с когнитивной маршрутизацией)
- Per-expert K-space (k=32) позволяет каждому эксперту работать в своём контексте
- Meta-gate учится доверять/игнорировать каждого эксперта

**Зависимости:**
- **Maturation** — запись в private_mem открывается при mat_gate[i] >= 0.3
- **BridgeGLU** — MLP gate modulated by bridge semantic delta
- **PrecisionGate** — маршрутизация по pred_error
- **Bridge** — semantic delta для BridgeGLU

**Ключевые параметры:**
- `G=32` — количество экспертов
- `k=32` — размерность K-space
- `private_mem=True` — cross-expert private memory bank
- `expert_asymmetry=True` — разные alpha, log_scale, W_proj для каждого эксперта

---

#### 2. SemanticBridge (Кросс-слойный семантический мост)
**Файл:** `core/bridge.py`

**Что делает:** Shared probe head emits semantic vector s_l = probe(h_l) на каждом слое. Векторы формируют кросс-слойный поток:
- **DEPTH:** bottom-up (свежие от нижних слоёв) + top-down (carried от верхних)
- **TIME:** persistent bridge_stream (EMA) переносится через forward calls
- **SELF-SUPERVISED:** probe обучается предсказывать next-token embedding (cosine loss)

**Зачем:**
- Связывает слои семантически (не просто skip connections)
- Даёт плотный градиент на каждом уровне глубины
- Позволяет нижним слоям "видеть" контекст верхних

**Зависимости:**
- **Maturation** — bridge injection gated by mat_gate[i]
- **LayerBridgeGate** — per-layer spectrum gate для bridge routing
- **BridgeGLU** — bridge delta модулирует MLP gate
- **IntentBridge** — parallel semantic stream

**Ключевые параметры:**
- `bridge_conn=0.1` — aux loss weight
- `bridge_dim=256` — semantic vector width
- `bridge_depth=True` — cross-layer injection

---

#### 3. MaturationController (Единый контроллер созревания)
**Файл:** `core/maturation.py`

**Что делает:** Per-layer maturity M_l(t) in [0,1], гейтирующий ВСЕ wake-up сигналы:
- Bridge injection
- Private memory write
- Intent bus
- LayerBridgeGate

**Формула:** `M_l(t) = sigmoid((t - (T0 + alpha*tau_norm_l*T_delay)) / delta_t)`

**Зачем:**
- Deep-first созревание: глубокие слои (большой tau) открываются первыми
- Предотвращает catastrophic interference
- Bridge readiness: компетентность bridge определяет готовность

**Зависимости:**
- **Bridge** — readiness от cosine loss bridge probe
- **VSA tau ladder** — tau_norm для определения порядка открытия
- **Все компоненты** — maturation гейтирует их активность

**Ключевые параметры:**
- `matur_T0=8000` — начало ramp
- `matur_T_delay=8000` — задержка deepest layers
- `matur_delta=4000` — ширина ramp
- `matur_r0=0.3` — readiness sigmoid center

---

#### 4. StreamingMemoryBank (Иерархическая память L1→L2→L3)
**Файл:** `core/memory_bank.py`

**Что делает:** Три уровня памяти:
- **L1 (immediate):** rolling buffer of last K sentence embeddings (без обучения)
- **L2 (short-term):** learned bank with N slots, novelty-based write gating
- **L3 (emergent concepts):** clusters L2 keys by cosine similarity, concept birth/update

**Зачем:**
- L1: имmediate контекст (последние предложения)
- L2: краткосрочная память (выученные ассоциации)
- L3: долгосрочная паметь (эмерджентные концепты)

**Зависимости:**
- **Maturation** — записи gated by mat_gate[i] >= 0.3
- **Per-layer** — memory bank вызывается внутри цикла слоёв

**Ключевые параметры:**
- `mem_l1_slots=3` — L1 buffer size
- `mem_l2_slots=16` — L2 bank size
- `mem_l3_concepts=8` — L3 concept slots
- `mem_min_write_mat=0.3` — min maturation для записей

---

#### 5. ThinkingTokenHead (Адаптивное мышление)
**Файл:** `core/reasoning.py`

**Что делает:** Chain-of-thought reasoning с adaptive depth. Thinking tokens генерируются динамически, gate определяет когда остановиться.

**Зачем:**
- Модель может "думать" дольше на сложных задачах
- Adaptive depth: не все токены требуют одинаковой глубины
- Self-supervised: reasoning gate обучается предсказывать.stop

**Зависимости:**
- **Maturation** — reasoning gate может быть gated maturation
- **Bridge** — reasoning context влияет на bridge

**Ключевые параметры:**
- `reasoning_max_steps=8` — max reasoning depth
- `reasoning_gate_stop_threshold=0.5` — stop threshold

---

#### 6. HybridHead (Гибридная голова)
**Файл:** `core/embedding.py`

**Что делает:** `output = sigmoid(x) * (1 + softmax(x))` — выпуклая комбинация sigmoid и softmax.

**Зачем:**
- Sigmoid сохраняет лакуну (потенциал несовпавшего)
- Softmax добавляет нормализованную конкуренцию
- "Демократия с лидером": softmax = лидер, sigmoid = народ

**Зависимости:**
- **Bind** — выход head связан с embedding через tie_weights

---

#### 7. IntentBridge (Мост намерения)
**Файл:** `core/stack.py` (внутри forward)

**Что делает:** Per-head intent stream (G×K_max), flows through depth (per-layer) and time (EMA). Experts "подхватывают" восходящий сигнал.

**Зачем:**
- Связывает "намерение" модели с её конкретными действиями
- Глубокие слои видят intent от мелких
- Cross-layer bus: network-wide gist

**Зависимости:**
- **Maturation** — intent bus gated by mat_gate[i]
- **Bridge** — parallel semantic stream

---

#### 8. LayerBridgeGate + SpectrumGate
**Файл:** `core/layer_bridge_gate.py`

**Что делает:** Per-layer intelligent gate для bridge routing. SpectrumGate = `sigmoid(logits) * (1 + softmax(logits/tau))`.

**Зачем:**
- Независимая активация по признакам (sigmoid)
- Относительный акцент среди признаков (softmax)
- tau связывает с maturation (self-regulation)

**Зависимости:**
- **Maturation** — tau из maturation gate
- **Bridge** — gate управляет bridge injection
- **Diagnostics** — per-layer health metrics

---

### Поток данных (forward pass)

```
tokens → PartitionedEmbedding → [RoPE] → h
    ↓
Memory Bank (per-layer, maturation-gated)
    ↓
Intent Bridge (per-layer, maturation-gated)
    ↓
[Layer 0] SemanticBridge.inject_layer(0, h)
    ↓
GroupedCognitiveMirror (32 experts, PrecisionGate)
    ↓
BridgeGLU (semantic delta → MLP gate)
    ↓
MLP (SwiGLU, maturation-gated)
    ↓
LayerBridgeGate → SpectrumGate
    ↓
[Layer 1..23] (repeat)
    ↓
SigmoidCodedHead → logits
    ↓
compute_loss (CE + aux losses)
```

### Tau-иерархия (связи между компонентами)

```
VSA tau ladder (τ_min=8..τ_max=515)
    ↓
Maturation tau_norm (normalize tau to [0,1])
    ↓
Maturation gate M_l(t) per layer
    ↓
┌─────────────────────────────────────────┐
│ Bridge injection (gated by M_l)        │
│ Private memory write (gated by M_l)    │
│ Intent bus (gated by M_l)              │
│ LayerBridgeGate (tau = M_l)            │
│ BridgeGLU (modulated by bridge delta)  │
│ Memory Bank (gated by M_l per-layer)   │
└─────────────────────────────────────────┘
```

### Aux losses (вспомогательные функции потерь)

| Loss | Weight | Purpose |
|------|--------|---------|
| CE | 1.0 | Main language modeling |
| bridge_conn | 0.1 | Bridge next-token prediction |
| diversity | 0.001 | Decorrelate expert outputs |
| balance | 0.026 | Load balancing across experts |
| gate_repulse | 0.3 | Push gate variance up |
| alpha_novelty | 0.05 | Push per-expert alpha apart |
| reinforce | 0.001 | Align gate with usefulness |
| gradalign | 0.0 | Align MLP gate with CE gradient |

### Современное состояние

- **Step:** 466 (Colab запущен)
- **Val:** 11.0132@466
- **Memory Bank:** активен, maturation-gated (L1=0 при mat<0.3)
- **Bridge:** stable, bridge_conn ~0.1
- **Maturation:** deep-first работает (L23 opens first)

---
