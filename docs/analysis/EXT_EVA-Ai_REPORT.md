# EXT_EVA-Ai_REPORT — полный разбор проекта `C:\Users\black\OneDrive\Desktop\EVA-Ai`

**Дата разбора:** 2026-09-19
**Метод:** чтение кода (не README), инвентаризация, трассировка вызовов, разбор формул, поиск мёртвых ветвей и расхождений контрактов. Логи запуска приложены как доказательная база.
**Контекст:** EVA-Ai — предшественник EVA-CLM; из него в EVA-CLM перенесён лайфцикл фантомов (confirm 0.75 / archive 0.25). Цель отчёта — извлечь всё ценное (в первую очередь корреляционные/связывающие методы) и зафиксировать грабли.

---

## 1. Паспорт проекта

EVA-Ai — это **два проекта в одной папке**, которые надо разделять:

| Слой | Что это | Где | Состояние |
|---|---|---|---|
| **A. Приложение EVA-Ai** | Модульный «когнитивный ассистент»: CoreBrain + EventBus + память FractalGraphV2 + этика/противоречия + веб-поиск + Flask-GUI + FCPipeline на OpenVINO | `eva_ai/` (518 `.py`, 143 796 строк) | Запускалось (есть логи 2026-07-28…08-01), но **нейросетевая часть (FCPipeline) в рантайме ни разу не поднялась** |
| **B. Исследовательская ветка λ_d / MemBind** | «LLM без attention»: bind-адаптация, ковариационная память, спектральный оператор на корнях Фибоначчи; 23 эксперимента | `experiments/` (4 722 строки), `tests/` (750 строк) | Перенесена копипастой из проекта **FCP** (`C:\...\FCP\ld_model`, `train_phase2.py`, `russian_chunks.npy` 3.4 ГБ, `token_stream.bin` 47.6 ГБ). В EVA-Ai **не запускается**: нет `ld_model`, нет данных, нет чекпоинтов |

Дополнительно в корне лежат:
- `Analysis.md` (345 строк) — предыдущий статический аудит (частично устарел: например, дубль `_init_mode_controller` уже исправлен — в `core/brain_components.py` остался один метод, строка 50).
- `EVA_log.txt` — 27 381 строка, 3.8 МБ рабочего лога.
- `session_backup.json` — **490 МБ** сериализованного состояния (мусор в корне).
- `libinpoc_txt/main-russian/**` — корпус русской литературы (тысячи `.txt`), учебные данные.
- `models/ruadapt_qwen3_4b_openvino_ModelB/` — GGUF 2.5 ГБ + OpenVINO int4 (2.2 ГБ) + KV-диск-кэш 36×32 МБ и **отдельная исследовательская ветка анализа регионов Qwen** (`region_causal*.py`, отчёты JSON, `np.corrcoef`).
- Git-репозитория нет (`fatal: not a git repository`).

**Ключевой факт по логу** (`EVA_log.txt`):
- `FCPipeline: cannot import name 'FCPPipelineV15'` (строка 171);
- `[FCP] Model path does not exist!` → `[FAIL] fcp_pipeline не создан` (551–552, 2421–2422, 3814–3815, …);
- `[FCP API] Failed to initialize: Exception from src\inference\src\cpp\core.cpp:85` (OpenVINO GenAI, 1117 и далее ~10 раз);
- `'CoreBrain' object has no attribute 'qwen_model_manager'` при чате через WebGUI (4302, 24447, 25261, 25318);
- `memory_percent: 98.2–100.0%` CRITICAL десятки раз;
- `'FCPipeline' object has no attribute 'unload_models'`, `'ResponseGenerator' object has no attribute 'unified_bridge'` (4376–4379 и далее).
Итог: работал CoreBrain + GUI + шина событий + куратор графа, а генеративная часть отсутствовала.

---

## 2. Инвентаризация (без мусора)

```
EVA-Ai/
├── Analysis.md, analysis_session.md, opencode_session_EVA_Ai.md   # прошлый аудит
├── EVA_log.txt (3.8MB), session_backup.json (490MB)               # рабочие артефакты
├── brain_config.json                                              # вся конфигурация (185 строк)
├── libinpoc_txt/main-russian/**                                   # корпус (не код)
├── eva_ai/                        # 518 .py / 143 796 строк
│   ├── core/ (108 файлов)         # CoreBrain, миксины, FCPipeline, event bus
│   ├── memory/ (51)               # MemoryCore, FGv2, кэши, PIE, TCM
│   ├── knowledge/ (19)            # KG, ConceptMiner (фантомы!), GraphCurator
│   ├── fcp_core/ (22)             # KCA, SRG, CrossAttention, TrainableGate, LoRA
│   ├── fcp_gnn/ (5)               # GNN-энкодер, hybrid integration
│   ├── mlearning/ (~30)           # FractalModelManager, FractalTransformer
│   ├── reasoning/ (13+5)          # SelfReasoningEngine, fractal_ml
│   ├── contradiction/ (15), ethics/ (16), learning/ (~30)
│   ├── websearch/ (9), gui/web_gui/ (25), neuromorphic/ (7), ues/, fcp_ues/
├── tests/ (14 файлов / 750 строк)      # диагностика λ_d, НЕ pytest
├── experiments/ (23 файла / 4 722 строки)  # λ_d / MemBind исследования
└── models/ruadapt_qwen3_4b_openvino_ModelB/  # модель + 20 скриптов анализа
```

Тесты — диагностические скрипты без `assert` и без pytest; **все** импортируют `ld_model`, которого в EVA-Ai нет (проверено `glob **/ld_model/**` → пусто), поэтому в этом проекте они неисполнимы.

---

## 3. Архитектура приложения (слой A)

### 3.1 Ядро
- `core/core_brain.py` — `CoreBrain` собирается из ~12 миксинов (components/query/init/state/…). Точка входа `eva_ai/run.py`; конфиг `core/brain_config.py` читает `brain_config.json` (падает с `FileNotFoundError`, если нет — строка 4 лога).
- Две шины событий: `EventSystem` (legacy) + `EventBus` (глобальный синглтон) + мост (`event_bus_bridge.py`). В логе реально работает `EventBus` («EventBus worker loop запущен»), legacy — балласт.
- `process_query()` — 4 стратегии: FG-only → Qwen → GGUF/FCPipeline → шаблонный ответ (`brain_query.py`). Двойной вызов `_extract_key_concepts()` (Analysis.md #8).

### 3.2 FCPipeline (генерация на OpenVINO)
`core/fcp_pipeline.py` (3 225 строк): `KVCachedPipeline`, `PatchedLLMPipeline`, `FCPipeline`, фабрика `create_fcp_pipeline` (3223). Внутри инициализируются: `KCADetector` (748), `GraphStateInjector` (803/1086), `CrossAttentionFusion` (938–950, hidden_dim=2560, graph_dim=384), `TrainableGate` (952–964, input_dim=2560, 3 источника), `ExpertSystem` (966), `ThinkingController` (985), `ToolOrchestrator` (994), `ReasoningChainManager` (1020), `HybridLayerProcessor` (1117).
Инъекция знаний в генерации — `fcp_pipeline.py:2301–2419`:
- CrossAttention между скрытым состоянием и эмбеддингами подграфа;
- KCA-детектор лакун/противоречий (с **фиктивными** gate_weights `0.5+0.3·sin(iπ/18)`, строки 2345–2347);
- `TrainableGate.forward([val_proxy, ca_output, kca_correction_vec])` (2398);
- вес инъекции `kca_weight = 0.07`, для int4 — `0.2`, int8 — `0.12` (2411–2417).

### 3.3 Память
- `memory/memory_core.py` — SQLite-нейроны (`active_memory`, `long_term_memory`), поля working/semantic/episodic; SQL f-string для имени таблицы (147) — известная проблема.
- `memory/fractal_graph_v2/` — **основной граф знаний**: `storage.py` (82 КБ, SQLite + lazy/full режимы, HNSW-индекс, кластеризация), `__init__.py` (74 КБ, `FractalMemoryGraph`: векторизация, группы, self-dialogue, retrieve), `graph_indexer.py` (HNSW cosine), `hierarchy_index.py` (L0→L3 навигация), `semantic_context_cache.py` (FAISS), `embeddings.py` (multilingual-e5-base, 768d), `optimizations.py` (HNSW/NLI/инкрементальная кластеризация/PathCache).
- `memory/pie_integration/` — L1/L2 граф, `routing_engine.py` (правила генерации по домену), `activation_profiler.py` (fingerprint = среднее эмбеддингов запроса и ответа).
- `memory/semantic_cache.py` — семантический кэш ответов (порог cosine 0.87), `temporal_context.py` — TCM со смешанным скором.

### 3.4 Знания/обучение/прочее
- `knowledge/concept_miner.py` — **ConceptMiner/ACI v3.1**: детекция семантических лакун в кластерах FGv2, генерация гипотез, валидация (NLI/ontology/ethics/web), лайфцикл фантомов (см. §5.10).
- `knowledge/graph_curator.py` — фоновая «курация» графа каждые 5 минут: garbage cleanup, promote/demote уровней, консолидация; адаптивный интервал 0.5–2× от нагрузки/размера.
- `contradiction/` — 15 файлов: `contradiction_miner.py` (косинус ≥0.75 + NLI ≥0.65), `detect_semantic.py` (TF-IDF/Jaccard), `core_detection.py` (Jaccard слов), `detect_temporal.py` и др.
- `neuromorphic/sim_core.py` — спайковый симулятор; корреляции `np.corrcoef` между спайк-трейнами (395–409).
- `mlearning/` — FractalModelManager, FractalTransformer (attention с аддитивным «фрактальным» bias), FractalTrainer (HF-подобный цикл), текстовое качество.
- `gui/web_gui/` — Flask-сервер и API; именно сюда приходили запросы чата, падавшие на `qwen_model_manager`.

---

## 4. Исследовательская ветка λ_d / MemBind (слой B)

Это, судя по коду, главный «интеллектуальный экспорт» EVA-Ai/FCP. Общая идея: **языковая модель без attention и без softmax-гейтов**, где роль связывания играют:
1. билинейный bind `u⊙v` (VSA/FCF-подобный),
2. ковариационная память `M = d·M + i·kᵀk` с параллельным сканом,
3. фиксированный спектральный оператор `V·diag(λ)·Vᵀ` на корнях Фибоначчи.

Компоненты (эволюция):
`SigmoidGate` (baseline) → `BindGate` (test_bind_gate.py) → `CovGate` (sequential, test_cov_gate.py) → `PScan` (parallel scan, test_pscan_gate.py) → `MemBind` (multi-head + feedback, test_membind.py) → `ld_model/core.py` (в FCP, «superseded»-пометки в шапках файлов).

Конфигурация по умолчанию во всех экспериментах: `D=896`, `VOCAB=50000`, `N_MODES=4`, `N_LAYERS=12`, `B=4`, `accum=4`, `seq=64`, `lr=1e-3`, `warmup=250`, `steps=5000`, `clip=1.0`, AdamW wd=0.01, weight tying, данные `russian_chunks.npy` (20 000 чанков).

### 4.1 Зафиксированные результаты (из кода)
`experiments/analyze_architecture.py:217–222` — сравнение на 5000 шагов:

| Архитектура | PPL | tok/s | Качество | Скорость |
|---|---|---|---|---|
| CovGate (sequential) | 438 | 260 | 100% | 1× |
| BindGate (no memory) | 793 | 1175 | 55% | 4.5× |
| PScan (parallel cov) | 688 | 1212 | 64% | 4.7× |
| **MemBind** | **466** | **962** | **94%** | **3.7×** |

`experiments/test_dct_sliding.py:6–11` (docstring, чекпоинт phase2_step30000.pt, 2.6M токенов, 89M параметров): DCT loss 14.6; slide 14.6; **both 14.3** (baseline 14.9), nan=0/inf=0, «генерация связанная».
`experiments/analyze_context.py:173–191`: ковариационное состояние = **96 КБ** (24 слоя × 4 головы × 256 float) вместо KV-кэша; на L=128K MemBind ≈3.45M MAC/ток vs Transformer ≈229M MAC/ток (**≈66×**), VRAM не зависит от контекста.
`experiments/analyze_scaling.py:283–293`: MemBind ~2× дешевле трансформера по FLOPs/ток при том же D; ёмкость памяти считается как DoF суммы τ rank-1 обновлений (τ≈8–100).
`tests/test_learnable_v.py`, `test_v_delta_divergence.py` — проверки Cayley-ортогональности `R` (ratio ‖RᵀRv‖/‖v‖ = 1.0) и расхождения `cos(R_i, R_{i+1})` между слоями.
`experiments/experiment_importance.py:184–189`: «gate spread (decisiveness) — zero-cost salience signal; high-spread токены сходятся медленнее → adaptive depth».

---

## 5. Корреляционные методы (главный раздел)

Ниже — все найденные методы связывания/сопоставления/корреляции с формулами и точными ссылками. Статус: **[D]** — работает/доказан в рамках проекта, **[B]** — перенесён/используется в EVA-CLM, **[X]** — мёртв/сломан/недостижим.

### 5.1 VSA-bind: билинейное связывание `u ⊙ v` вместо гейта
**Где:** `experiments/test_bind_gate.py:146–195` (`BindGateBlock`), `test_membind.py:144–146, 200–205`, `test_cov_gate.py:147–150, 170–173`, `test_pscan_gate.py:151–154, 172–175`.
**Формула:**
```
h_conv = CausalConv1d(h);  h_norm = RMSNorm(h + h_conv)
u = h_norm · W_u        (D×r)
v = h_norm · W_v        (D×r)
bound = u ⊙ v           (r)            # FCF/VSA bind, element-wise multiply
h_adapt = h_norm + (bound · W_out)     # W_out инициализирован нулём → тождество на старте
Δ = V · diag(λ) · Vᵀ · h_adapt
h_out = h + Δ
```
**Назначение:** заменить sigmoid/softmax-гейт (`α·λ`) на content-dependent pre-transform без внимания; гипотеза из шапки файла: «Gating (α·λ) можно заменить на content-dependent pre-transformation через низкоранговый bind (u*v)». Параметров всего `3·D·r` (~0.06·D² на слой). **[D]** — проверен, выигрывает у SigmoidGate (см. `test_bind_gate.py:410–419` — печатает ratio Bind/Sigmoid PPL), но проигрывает ковариационным вариантам по PPL (793 vs 438–466).

### 5.2 Ковариационная память (mLSTM/GLA-подобная)
**Где:** `test_cov_gate.py:118–213` (sequential), `test_pscan_gate.py:122–202` + `parallel_prefix_scan:64–96` (parallel), `test_membind.py:112–213` (multi-head + feedback).
**Формулы (CovGate/PScan):**
```
K = h_norm·W_k;  Q = h_norm·W_q                     (r)
i = exp(h_norm·W_i + b_i)                            # input gate ∈ (0,∞), может усиливать
d = sigmoid(h_norm·W_decay + b_decay)                # forget gate ∈ (0,1), b=1.0 → d≈0.73
M[t] = d[t]·M[t-1] + i[t]·(k[t]ᵀ k[t])               # self-covariance, r×r
h_mem[t] = (q[t]·M[t])·W_read                        # W_read zero-init
h_total = h_adapt + h_mem
```
Важные решения из комментариев кода:
- «self-covariance (kᵀk) more stable than cross (kᵀq)» (`test_cov_gate.py:192`);
- параллельный скан — ассоциативная свёртка `combine((A1,B1),(A2,B2)) = (A1·A2, A2·B1+B2)` по Хиллису–Стилу, O(log L), **autograd-safe** (без in-place, через `torch.cat`, `test_pscan_gate.py:64–96`);
- в MemBind — **multi-head** (H=4, r=8/16): у каждой головы свои `W_k,W_q,W_i,W_decay,W_read`, свои скорости забывания → мультимасштабная память; формула обновления та же (einsum, `parallel_prefix_scan`), чтение `mem_h = q_h @ M_h @ W_read_h`.
**MemBind feedback (уникальная часть):**
```
v_enh  = v + W_mem2v · Σ_h mem_h      # память модулирует bind-сигнал
h_adapt = h_norm + (u ⊙ v_enh)·W_out
```
`test_membind.py:200–205`. Это прямая петля «прошлое → трансформация текущего токена».
**Ёмкость:** `analyze_scaling.py:97–110` — при d=0.88 (τ≈8) эффективный ранг ≈4 из 16, DoF/голова ≈ (4·(32−4))/2=56 → 1792 бит/голову (fp32); при d=0.99 (τ=100) ранг 16, 240 DoF → 7680 бит. Суммарно 4 головы ≈ 7–30 Кбит на слой. **[D]** — лучший PPL среди вариантов; **[B-потенциал]** — прямо ложится на «память-банки» EVA-CLM.

### 5.3 Спектральный оператор `V·diag(λ)·Vᵀ` (замена attention-контекста фиксированной рекуррентностью)
**Где:** все эксперименты; теория — `experiments/analyze_math.py`, `analyze_architecture.py:206–213`, `explore_fib_spectra.py`, `test_dct_sliding.py`.
**Формула:** `h_spec = V · diag(λ_1..λ_K) · Vᵀ · h`, где λ_K разбиты на K блоков размером `D/K`.
- λ — **корни k-шаговых чисел Фибоначчи**: решения `x^k = x^{k-1}+…+1`; λ₂=1.618 (φ), λ₃=1.839, λ₄=1.927, λ₅=1.966, λₖ→2. Ищутся бисекцией (`fibonacci_roots`, 54–65).
- `V` — случайная ортогональная (32 отражения Хаусхолдера, 69–75) **или DCT** (`test_dct_sliding.py:19–25`); с DCT оператор становится буквальным фильтр-банком (DCT→масштаб полос→IDCT, `analyze_math.py:164–176`).
- Устойчивость: λ<2; при L=128 λ^L от 1.6¹²⁸≈1e26 (усиление) до 1.966¹²⁸≈1e37 — «expansion», поэтому важна нормировка/клип.
- **λ-sliding по слоям**: `scale = 0.5 + layer_idx/(n_layers−1)` (`test_dct_sliding.py:27–30`) — нижние слои «толще/грубее», верхние «тоньше» (аналогия с пластиной Hladi).
- **λ-tied heads** и **learnable δ** предложены в `analyze_math.py:178–256` (δ аддитивно опасно: (λ+0.01)^128 даёт ×2; безопаснее экспоненциальный множитель `λ^(t(1+δ))`).
**[D]** — DCT+slide дали лучший loss и «связную» генерацию на чекпоинте.

### 5.4 KCA — Knowledge Conscious Attention (замена attention на граф)
**Где:** `eva_ai/fcp_core/__init__.py:102–253` (`KnowledgeConsciousAttention`), `fcp_core/kca_detector.py` (эвристический вариант), `fcp_core/graph_injection.py`.
**Формулы (полная версия):**
```
K_k = H·W_Kk;  V_k = H·W_Vk                 # H = эмбеддинги узлов подграфа (заморожены на цикл)
Q_k = X_prev·W_Qk
A = softmax( Q_k K_kᵀ / √d )                # attention токен→узел
E_lacuna = (1 − A)·V_k                      # «лакуна»: чего модель НЕ видит
H_norm = H/‖H‖;  node_sims = H_norm H_normᵀ # корреляция узлов между собой
для каждого токена i: u,v = top-2 узла по A[i]
  если node_sims[u,v] < contradiction_sim_threshold:  E_contra[i] = H[u] − H[v]
E_corr = λ_l·E_lacuna + λ_c·E_contra
E_corr ← E_corr · ρ^t                        # адаптивное затухание (damping)
gamma = sigmoid([X_prev ; E_corr]·W_g + b_g) # гейт инъекции (concat 2d→d)
X_new = X_prev + gamma ⊙ E_corr
```
Сходимость — `ConvergenceController` (45–99): (1) насыщение гейта `g < gate_threshold` два шага; (2) **детектор осцилляции по косинусу последовательных дельт**: `cos_sim = ⟨Δ_t, Δ_{t-1}⟩/(‖Δ_t‖‖Δ_{t-1}‖) < osc_threshold` → усреднение последних 3 состояний; (3) лимит циклов.
**Назначение:** итеративно «дотягивать» скрытые состояния к графу знаний, компенсируя лакуны (низкое внимание) и противоречия (низкая корреляция топ-узлов). **[D]** — реализовано и встроено в FCPipeline; **[B-потенциал]** — oscillation-detector и «(1−A)» как сигнал незнания.

### 5.5 CrossAttentionFusion — прямой attention-заменитель (модель ↔ граф)
**Где:** `eva_ai/fcp_core/cross_attention.py:8–114`.
**Формула:** классический multi-head: `Q=X·W_q` (модель), `K=G·W_k`, `V=G·W_v` (граф), `scores = QKᵀ/√d_head`, стабильный softmax (вычитание максимума), `out = softmax·V·W_o`. Xavier-инициализация, `update_weights()` для обучения.
**Интеграция:** `fcp_pipeline.py:938–950` (hidden_dim=2560, graph_dim=384) и вызов на `val_proxy` (2301–2335).
**Проблема [X]:** эмбеддинги FGv2 — 768d, а `W_k` инициализирован под 384 → `np.dot` падает; исключение ловится, `ca_output = val_proxy` (2332–2333). Фактически cross-attention **никогда не срабатывает** в проде.

### 5.6 SRG — Semantic Relevance Gate (маршрутизация по косинусу и энтропии)
**Где:** `eva_ai/fcp_core/__init__.py:256–292`; порог `srg_cosine_threshold = 0.85` (`fcp_core/config.py:69`).
**Формулы:**
```
sim = ⟨query_vec, response_vec⟩ / (‖q‖‖r‖)
H = −Σ p log₂ p,  p = softmax(logits)
is_coherent = sim ≥ 0.85;  is_confident = H ≤ srg_entropy_threshold
(direct | reasoning | variational)
```
**Назначение:** решение «отвечать сразу / запускать рассуждение / варьировать» на основе корреляции запрос-ответ + уверенности. Удалён метод `evaluate_from_probs` с комментарием-могильником: «mathematically incorrect (treated probabilities as embeddings)» (294) — полезное предупреждение.

### 5.7 Инъекция графа в KV (масштабирование Key / коррекция Value)
**Где:** `eva_ai/fcp_core/graph_injection.py:157–345`.
**Формулы:**
```
key_scaled = key · (1 + s_key·(gate_weight − 0.5))      # s=0.10
value_corr = value + s_val·gate_weight·proj(graph_vector)  # s=0.15
```
плюс KCA-коррекции по слоям и «activation gate»: `avg_gate ≥ 0.85` → early exit. `GraphStateInjector.compute_graph_vector` (88–155) ожидает 768-мерные эмбеддинги, паддит/обрезает, затем GNN → graph_vector 256 и gates 36.
**Проблема [X]:** KCADetector получает `graph_vec[:256]`, а `_create_value_correction` использует первые 128 dims как head_dim; gate_weights в проде — синтетический синус (fcp_pipeline.py:2345–2347). Всё обёрнуто в try/except → тихо не работает.

### 5.8 Similarity-инфраструктура (по всему приложению)
| Метод | Формула/порог | Файл:строка | Назначение | Статус |
|---|---|---|---|---|
| `SimilarityEngine` | cosine·0.5 + euclidean·0.3 + jaccard·0.2; `find_most_similar` | `memory/fractal_cache/similarity_engine.py:22–137` | универсальное сравнение эмбеддингов | [D] |
| `SemanticCache` | cosine > **0.87** → ответ из кэша; LRU 200 | `memory/semantic_cache.py:28–110` | кэш ответов | [D] |
| `EmbeddingsManager` | e5-base 768d, L2-норм., `compute_similarity` = dot(norm) | `memory/fractal_graph_v2/embeddings.py:184–248` | семантический поиск узлов | [D] |
| `GraphIndexer` | HNSW `space='cosine'`, `top_k·2`, `min_similarity` | `memory/fractal_graph_v2/graph_indexer.py:211–309` | ANN по узлам поверх SQLite | [D] + [X] fallback (см. §9) |
| FGv2 `semantic_search` | HNSW → иерархия → полный перебор; пороги 0.5/0.4 | `storage.py:864–1028` | поиск знаний | [D] |
| `HierarchicalIndex` | навигация L0→L3, cosine по 20-мерным превью | `hierarchy_index.py:133–217` | O(log n) доступ | [X] (обрезанные эмбеддинги) |
| `SemanticContextCache` | FAISS IVF inner product, min 0.3, LRU + smart_evict | `semantic_context_cache.py:86–310` | кэш сырых контекстов | [X] (см. §9) |
| `ActivationProfiler` | fingerprint = mean(emb(query), emb(response)) → 768 | `pie_integration/activation_profiler.py:261–342` | домен запроса, routing | [D] с hash-fallback |
| `RoutingEngine` | domain по profiler: confidence>0.5; embedding sim>0.7 | `pie_integration/routing_engine.py:161–231` | параметры генерации по домену | [D] |
| `TemporalContext` | score = 0.5·cos + 0.25·time_decay + 0.15·relevance + bonus + 0.05·recency | `memory/temporal_context.py:220–324` | память с временем | [D] |
| `FractalAddress` | cosine по нормированным 384d + иерархический путь L0..L3 (branching 16) | `reasoning/fractal_address.py:37–77` | адресация хранения | [D] |
| GNN trainer antonym | `score = max(0, −cos)·boost(2.0)`; target `score>0.3`; BCE | `fcp_core/online_trainer.py:752–786` | обучение противоречиям | [D] (если тренер запускается) |
| GNN trainer gap | `gap = (1 − var/var_max)·sparsity`; rare-boost; BCE | `online_trainer.py:788–830` | детекция лакун | [D] |
| Cluster centroid pull | `emb ← normalize(emb + α·(centroid − emb))` | `online_trainer.py:861–911` | борьба с разреженностью групп | [D] |
| TCM triplet | `loss = max(0, d(a,p) − d(a,n) + 0.5)`, обновление эмбеддингов in-place | `memory/temporal_context.py:330–439` | контрастивное «дообучение» | [X] (эвристика, не сохраняется) |
| Neuromorphic corr | `np.corrcoef(spike_train_i, spike_train_j)`, `mean(|corr|)` | `neuromorphic/sim_core.py:395–409` | сила синапсов | [D] |
| Qwen region analysis | `np.corrcoef(dAB, add)`, cos-сравнение сдвигов | `models/.../region_causal8.py:65,87` и др. | исследование слоёв Qwen | [D] (отдельная ветка) |

### 5.9 Корреляции в детекции противоречий
- `contradiction/contradiction_miner.py`: пары `sim(u,v) ≥ τ_sim=0.75` **и** `contra(u,v) ≥ τ_contra=0.65` (NLI BART-mnli, `_compute_contradiction:467+`), `min_confidence=0.4`, кэш косинусов (`_compute_similarity:442–465`), исключение пар с «исключающими» отношениями. O(n²) по узлам.
- `contradiction/detect_semantic.py:15–61`: семантическая дивергенция `1 − TF-IDF-cosine` (fallback Jaccard) и лексическая `1 − Jaccard`.
- `contradiction/core_detection.py:824–853`: Jaccard по словам; пороги избыточности 0.95/0.8.
- KCA (5.4) — косинус между узлами подграфа для поиска конфликта.

### 5.10 Лайфцикл фантомов (ConceptMiner) — то, что уехало в EVA-CLM
**Где:** `eva_ai/knowledge/concept_miner.py`; конфиг — `brain_config.json:157–173`.
**Класс `PhantomCandidate` (36–72):** `id, cluster_id, centroid[], nodes[], variance, semantic_gap, title/definition/rationale, status, confidence, confirmations, rejections, parent_group_id, validation_nli/ontology/ethics, web_verification, phantom_type, coherence_score, ambiguity_score`.
**Конфиг по умолчанию (90–108):** `base_threshold=0.30`, `dedup_radius=0.15`, `max_candidates_per_cycle=3`, `cycles_before_stable=5`, **`confidence_threshold_confirm=0.75`**, **`confidence_threshold_archive=0.25`**, `variance_k=2.0`.
**Статусы (28–33):** `provisional → confirmed → stable`, тупик `archived`.
**Правила перехода (`_update_lifecycle:1035–1061`):**
```
provisional:  web_verification.verified → confidence += 0.25
              confidence ≥ 0.75 → confirmed + _integrate_candidate()
confirmed:    confirmations ≥ 5 → stable
любой ≠ archived: confidence < 0.25 → archived
```
Плюс архивация по результатам валидации: NLI-противоречие/ontology/ethics (772–793).
**Классификация фантомов (`classify_phantom_entity:1091–1118`):**
```
variance > 0.5  и coherence < 0.3        → ambiguous
nodes > 10      и variance < 0.2         → emerging
coherence > 0.7 и nodes < 5              → abstract
"time"/"old" в заголовке                  → temporal
иначе                                     → regional
```
**Разрешение неоднозначности (`resolve_phantom_ambiguity:1120–1139`):** центроид → `fg.retrieve_similar(centroid, top_k=3)`, если `similarity > 0.7` → merge. **`retrieve_similar` и `merge_nodes` в FGv2 отсутствуют** (grep по всему `eva_ai` не находит определений) → ветка недостижима (hasattr=False), т.е. **[X]** — мёртвый код, хотя сами константы (0.7) уехали в EVA-CLM как `merge=0.7`.
**Метрики (142–153, 1063–1077):** `phantom_detection_rate`, `hypothesis_confirmation_ratio`, `validation_rejection_rate`, `candidates_archived/confirmed/rejected`, аудит-лог `phantom_audit_log.json` (каждое отклонение с причиной/дисперсией/уверенностью).
**Связь с EVA-CLM:** `EVA CLM/core/phantom.py:1–13` прямо пишет: «the EVA-Ai lacuna lifecycle, transplanted from the knowledge graph to hidden states… confirm 0.75 / archive 0.25 — the ConceptMiner constants», и переносит также `merge=0.7` (dedup) и `conf_init=0.5`. **[B] подтверждён кодом.**

### 5.11 Корреляции в лоссах (сводка)
- **Кросс-энтропия** — основная в экспериментах λ_d и `mlearning/fractal_trainer.py` (через HF-модель).
- **BCE по косинусным целям** — `online_trainer.py` (антонимы + лакуны).
- **Triplet margin** — `temporal_context.py:369–432` (эвристический, без autograd).
- **Косинус как метрика/сигнал**: KCA (узлы), SRG (запрос-ответ), contradiction (0.75), semantic cache (0.87), GNN antonym, cluster pull, EVA-CLM phantom (`sim = En @ Dnᵀ`).
- **Pearson/Spearman** — только в анализе: `experiments/experiment_importance.py:91–102` (связь salience-сигналов гейта с частотой токена).
- **np.corrcoef** — нейроморфный симулятор.

---

## 6. Обучение и оптимизация

### 6.1 λ_d / MemBind
- Loss: `F.cross_entropy`; NaN/Inf-шаги пропускаются (sanity-check + `continue`).
- AdamW(lr=1e-3, wd=0.01); warmup 250 шагов + косинус до нуля; grad clip 1.0; аккумуляция 4 (эфф. батч 16); seq 64/128.
- Weight tying embedding↔lm_head; **инициализация эмбеддингов `U(−1/√D, 1/√D)`** — с комментарием-фиксом: «Init N(0,1) даёт logits в ±900, убивает softmax» (`test_fresh_start.py:36–38`).
- Параллельный скан без in-place — autograd-safe (иначе `backward` ломается).
- Сравнение архитектур — см. таблицу §4.1.
- Cayley-параметризация `V` (learnable V, `V_rank=16`) с проверкой ортогональности `RᵀR≈I` и градиентов (`tests/test_learnable_v.py`).
- Экстраполяция контекста: модель без позиционных эмбеддингов, поэтому L>128 «by design» (`experiment_extrapolation.py`).

### 6.2 Приложение
- FCPipeline: online-обучение GNN/LoRA (`fcp_core/online_trainer.py`), ShadowLoRA с атомарной сменой и **авто-rollback при деградации >0.1** (`shadow_lora.py:38–120`), Ada-LoRA.
- Куратор графа: forced-run каждые 300 с; promote/demote/consolidate/cleanup; адаптивный интервал (0.5–2×) по CPU/RAM и числу узлов (`graph_curator.py:265–326`).
- Контрадикции: цикл по расписанию (3600 с), max 5 кандидатов/цикл, `dry_run=false`, priority coefficients α=0.4/β=0.3/γ=0.3 (`brain_config.json:143–155`).
- ConceptMiner: `dry_run=true` в конфиге приложения (!), max 3 кандидата/цикл, web-валидация включена (`brain_config.json:157–173`).
- Ресурсы: ResourceManager переводит систему в CRITICAL при memory >98–99%; в логе видно, что это происходило постоянно.

---

## 7. Что доказано, что мертво

### 7.1 Доказано (по коду/логам/результатам)
1. **MemBind как лучший компромисс**: PPL 466 vs BindGate 793 / PScan 688 при 3.7× скорости (analyze_architecture.py).
2. **Bind (u⊙v) работоспособен как замена sigmoid-гейта** (test_bind_gate.py; W_out=0 init, стабильное обучение).
3. **Параллельный скан ковариационной памяти** работает и обучается end-to-end (PScan/MemBind, 5000 шагов без NaN).
4. **DCT-базис и λ-sliding** дают лучший loss и связную генерацию на чекпоинте (docstring test_dct_sliding.py).
5. **Ковариационное состояние 96 КБ и независимость VRAM от длины контекста** — расчётно (analyze_context.py), подтверждается архитектурой (нет KV-кэша).
6. **Лайфцикл фантомов 0.75/0.25** реально существует в коде и реально перенесён в EVA-CLM (код обеих сторон).
7. **Приложение работало** как сервис: логи 4 дня, WebGUI отвечает 200, куратор графа работает, EventBus живой.
8. **Cayley-ортогональность и градиенты** проверены (test_learnable_v: ratio=1.0).

### 7.2 Мертво / недоделано (с доказательствами)
1. **FCPipeline в рантайме не поднимался** — EVA_log.txt: 171, 551–552, 850, 1117, 2421, 3814, 4994, 7011, 8766 и др.
2. **Эксперименты/тесты неисполнимы в EVA-Ai** — импорт `ld_model` (нет), данные `russian_chunks.npy`/чекпоинты (в FCP).
3. **Phantom resolution мёртв** — `retrieve_similar`/`merge_nodes` не реализованы (concept_miner.py:1128–1156).
4. **TrainableGate несовместим по размерностям** — `W1` под `2560·3`, а источники `2560 + 2560 + 256`; исключение ловится в fcp_pipeline.py:2404–2406 → тихий fallback.
5. **CrossAttentionFusion несовместим** — graph_dim=384 vs реальные 768 (fcp_core/cross_attention.py:15 vs embeddings.py).
6. **Фиктивные эмбеддинги**: `_get_query_embedding` = `hash(text)`+`np.random` (fcp_gnn/hybrid_integration.py:470–483); `GraphEncoderRuntime` — случайные веса (graph_encoder.py:222–226); hash-fallback в `FractalEmbedder:53–63` и `ActivationProfiler:321–342`.
7. **`MemoryNeuron.get_similarity` не существует**, но вызывается (memory_working.py:200; ltm_retrieval.py:44,88) → AttributeError в retrieve при текстовом запросе и nlp_model.
8. **`GraphIndexer.search` fallback**: превращает эмбеддинг в строку и ищет «слова» в content (graph_indexer.py:252–256); `conn.close()` внутри цикла по строкам (290).
9. **`SemanticContextCache`**: FAISS IVF обучается на `np.random.randn(1000,…)` (99); в `add` — `train(np.array([embedding]))` (194–195); FAISS-ветка считает inner product **ненормированных** эмбеддингов, numpy-ветка — косинус (297–299) → несогласованная метрика.
10. **`HierarchicalIndex`**: в кэш кладутся только первые 20 dims эмбеддинга (115, 230), затем они нормируются как полноценные векторы и по ним идёт cosine → навигация бессмысленна.
11. **`embeddings.encode_single`**: при недоступной модели `encode()` возвращает `None`, а `encode_single` делает `len(None)` → TypeError (155–162), хотя комментарий обещает None.
12. **TCM triplet** обновляет копии эмбеддингов в буфере без сохранения (413–423), `avg_loss` считается, но модель не обучается.
13. **`session_backup.json` 490 МБ** и `EVA_log.txt` 3.8 МБ в корне; git отсутствует.
14. **Дублирование EventSystem/EventBus**, deprecated-инициализаторы, 4 источника `model_path` — из Analysis.md, частично актуально.

---

## 8. Что переносимо в EVA-CLM и что нельзя

### 8.1 Переносимо (приоритет по ценности)
1. **Лайфцикл фантомов — расширить уже перенесённое.** EVA-CLM взял ядро (0.75/0.25, conf_init=0.5, merge=0.7). Из EVA-Ai стоит добрать: `cycles_before_stable=5`; правило `web/внешнее подтверждение → +0.25`; классификатор `ambiguous/emerging/abstract/temporal/regional` (пороги variance/coherence/node_count — concept_miner.py:1091–1118); аудит-лог отклонений с причинами (183–194); «high priority» = `conf > 0.7 и тип emerging/ambiguous` (1186). В EVA-CLM это ляжет на `confirmed_directions()` и консолидацию (M55).
2. **Ковариационная память как субстрат корреляций**: `M[t] = d·M[t-1] + i·kᵀk`, чтение `q M W_read`, exp-input-gate, multi-head с разными decay, обратная связь `v_enh = v + W_mem2v·mem`. В EVA-CLM (память-банки, τ-поле) это даёт «второй порядок» корреляций без attention и с состоянием O(r²) на голову. Переносить формулу + `parallel_prefix_scan` (autograd-safe).
3. **Bind `u⊙v` с zero-init W_out** — минимальная (3·D·r) и проверенная замена гейтам; идеально соответствует тезису EVA-CLM «σ×softmax гибрид», как «σ-free» ветка.
4. **Спектральный оператор с DCT/λ-sliding** — нуль параметров, фильтр-банк по полосам; λ<2; полезно для τ-поля как фиксированная частотная структура.
5. **Oscillation-detector KCA** (`cos(Δ_t, Δ_{t-1}) < θ` → усреднение состояний) — дешёвый контроль итеративных циклов генерации/рассуждения.
6. **SRG-роутинг** (`cos ≥ 0.85` + энтропия → direct/reasoning/variational) — маленький, детерминированный контроллер режима.
7. **Antonym-repel loss** `max(0,−cos)·boost` + **cluster centroid pull** — приёмы против коллапса/разреженности в embedding-пространстве; легко переносятся в любой embedding-тренинг.
8. **Паттерн HNSW + иерархическая навигация** (с исправлением 20-мерного бага) для больших банков памяти.

### 8.2 Не переносить
- **CoreBrain-миксины, двойной EventBus, GUI, websearch, ethics/contradiction as-is** — связанность через `brain=brain`, синглтоны, 100% RAM, отсутствие тестов.
- **FCP numpy-injection стек** (CrossAttention/TrainableGate/GraphStateInjector/KCADetector в том виде): несовместимые размерности, фиктивные гейты и эмбеддинги, глушение ошибок `except: fallback`.
- **Ручной backprop `TrainableGate.update`** — хрупкий; в EVA-CLM использовать torch autograd.
- **FGv2 storage целиком** (SQLite + lazy/full + обрезанные эмбеддинги) — только идеи, не код.
- **Случайные/hash-эмбеддинги как fallback** — они молча ломают семантику; в EVA-Ai уже есть правильный комментарий («Случайные векторы ломают семантический поиск», embeddings.py:150–153) — придерживаться его везде.

---

## 9. Грабли и риски (сводный список)

1. **`except Exception: fallback` как норма** (fcp_pipeline 2331–2333, 2380–2382, 2404–2406, graph_injection 311–330) — контрактные ошибки не видны, компоненты «зелёные», но не работают.
2. **Отсутствие единого контракта размерностей**: 384/768/256/128/2560/20 — встречаются все; ни одна граница не валидируется.
3. **API-методы, вызываемые через `hasattr`** (`retrieve_similar`, `merge_nodes`) — ветки кода, которые никогда не исполняются, но выглядят рабочими.
4. **Метрики схожести несогласованы** (cosine vs inner product; нормированные vs нет; усечённые vs полные) — пороги 0.7/0.75/0.87/0.5 означают разное в разных модулях.
5. **Копипаста без зависимостей**: 23 эксперимента + 14 тестов ссылаются на `ld_model`/данные другого проекта; docstring-результаты невоспроизводимы здесь.
6. **Ресурсный ад**: лог полон memory 98–100%, `unload_models` отсутствует, модель перезагружается циклами; `session_backup.json` 490 МБ.
7. **SQL f-string** в `memory_core.py:147`; Jaccard-«similarity» в core_detection вместо эмбеддингов; O(n²) в contradiction_miner.
8. **`_update_lifecycle`**: provisional-кандидаты, не достигшие 0.75 и не получившие явного rejection, **никогда не архивируются** по confidence (первая ветка не проверяет archive) — возможен вечный «provisional» мусор.
9. **KCADetector индексация**: `_generate_corrections` перебирает `enumerate(lacuna_layers + contradiction_layers)` и берёт `gate_weights[i]` по позиции в объединённом списке (244–266) — веса не соответствуют слоям; `_detect_contradictions` индексирует `graph_vector[i]` по позиции слоя (202–207).
10. **λ-чувствительность**: аддитивный δ к λ при L=128 меняет выход в разы (analyze_math.py:240–248); устойчивость только при λ<2.
11. **FAISS-кэш обучается на случайных векторах**, а затем на одном векторе — IVF-центроиды мусорные; эвикция из FAISS не удаляет вектор из индекса (индекс растёт, `index_to_id` — список, `pop` сдвигает индексы → рассинхрон).
12. **Отсутствие тестов/CI/git** — любая правка не проверяется; «тесты» не запускаются.
13. **Смешение языков/кодировок** в логах (cp1251-байты в UTF-8 логе) — читаемость логов низкая.
14. **Риск ложной уверенности**: все ключевые числа (PPL, tok/s, 96 КБ) получены в ветке FCP, а не в EVA-Ai; при переносе в EVA-CLM их нельзя цитировать как «результаты EVA-Ai» без пометки.

---

## 10. Приложение: карта ключевых файлов

| Область | Файл | Что смотреть |
|---|---|---|
| Ядро | `eva_ai/core/core_brain.py`, `brain_query.py`, `fcp_pipeline.py` | сборка, стратегии, FCP |
| KCA/SRG | `eva_ai/fcp_core/__init__.py` | `KnowledgeConsciousAttention` 102–253, `ConvergenceController` 45–99, `SemanticRelevanceGate` 256–292 |
| KCA-детектор | `eva_ai/fcp_core/kca_detector.py` | лакуны/противоречия/коррекции |
| Инъекция | `eva_ai/fcp_core/graph_injection.py` | key/value коррекции, activation gate |
| Cross-attn/гейт | `eva_ai/fcp_core/cross_attention.py`, `trainable_gate.py` | numpy MHA, softmax-гейт |
| GNN | `eva_ai/fcp_gnn/graph_encoder.py`, `hybrid_integration.py` | SAGEConv/HNSW, hybrid processor |
| Память | `eva_ai/memory/fractal_graph_v2/{storage,__init__,graph_indexer,hierarchy_index,semantic_context_cache,embeddings,optimizations}.py` | FGv2, поиск, кластеризация |
| Similarity | `eva_ai/memory/fractal_cache/similarity_engine.py`, `memory/semantic_cache.py` | cosine/euclid/jaccard, кэш 0.87 |
| PIE | `eva_ai/memory/pie_integration/{routing_engine,activation_profiler,fractal_graph_l1_l2}.py` | домен/routing |
| Фантомы | `eva_ai/knowledge/concept_miner.py`, `brain_config.json:157–173` | лайфцикл 0.75/0.25 |
| Противоречия | `eva_ai/contradiction/contradiction_miner.py`, `detect_semantic.py`, `core_detection.py` | 0.75+0.65, Jaccard/TF-IDF |
| Онлайн-обучение | `eva_ai/fcp_core/online_trainer.py` | antonym/gap losses, centroid pull |
| TCM | `eva_ai/memory/temporal_context.py` | score-смесь, triplet |
| λ_d | `experiments/{test_bind_gate,test_cov_gate,test_pscan_gate,test_membind,analyze_math,analyze_architecture,analyze_scaling,analyze_context,experiment_importance,test_dct_sliding,explore_fib_spectra,distill_zeckendorf}.py` | bind/cov/scan/spectrum |
| Тесты λ_d | `tests/test_learnable_v.py`, `test_v_delta_divergence.py`, `test_adaptive_*.py` | Cayley, adaptive gain |
| Логи | `EVA_log.txt` | падения FCP, RAM 100% |

### Итоговые метрики проекта
- `eva_ai/`: **518 .py / 143 796 строк**; `experiments/`: 23 / 4 722; `tests/`: 14 / 750; скрипты модели: 20 / 2 143.
- Рабочие артефакты: лог 27 381 строка / 3.8 МБ; `session_backup.json` 490 МБ; корпус литературы — гигабайты.
- Настоящих unit-тестов нет; pytest отсутствует.
- Статус: приложение — «работающий каркас без генератора»; исследовательская ветка λ_d/MemBind — «перенесённая, невоспроизводимая здесь, но содержательно самая ценная часть».

---

## 11. Краткие выводы для EVA-CLM

1. **Фантомы**: EVA-CLM уже использует ядро лайфцикла из EVA-Ai; стоит добрать `cycles_before_stable=5`, правило внешнего подтверждения (+0.25), классификатор типов фантомов и аудит-лог отклонений.
2. **Корреляции второго порядка**: ковариационная память (`d·M + i·kᵀk`, multi-head, feedback) — самый сильный неочевидный актив EVA-Ai; прямо применима к банкам/τ-полю.
3. **Главный урок по граблям**: в EVA-Ai всё держится на неявных контрактах (размерности, нормализация, наличие методов), а ошибки глушатся `except: fallback`. В EVA-CLM эти контракты нужно валидировать явно (assert/check на границах модулей), иначе получится «зелёная» система, которая ничего не делает — как FCPipeline в EVA-Ai.
