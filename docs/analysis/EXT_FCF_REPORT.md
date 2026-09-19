# EXT-FCF — Внешний разбор проекта FCF / EVA (Fractal Cognitive Field)

*Разбор по коду, не по README. Все ссылки — `файл:строка` в `C:\Users\black\OneDrive\Desktop\FCF`.*

| Параметр | Значение |
|---|---|
| HEAD git | `21d1061` «docs: rewrite README in WideBind style…», 2026-08-17 00:48 +0300 |
| Размер | 212 файлов, 5.1 GB (из них ~3.4 GB корпуса, 984 MB чекпойнт, ~0.7 MB Python-кода) |
| Python | 97 `.py`, ключевые 22 файла = 12 228 строк (concept_space 2969, stdp_trainer 1822, crystal_generator 1334, fcf_config 1153) |
| Тесты | `py -3.12 -m pytest tests -q` → **390 passed, 1 failed, 7 skipped, 32.5 s** (failed — `TestQNV14::test_dirty_cids_syncs_cpu`, в изоляции проходит → order-dependent flaky) |
| Обучение | 270 000 / 1 000 000 строк `corpus_1m.txt`, 41 539 812 пар (`checkpoints/meta.json`), state.pkl 984 MB, последняя строка лога: batch=106 s, ETA 57 h |
| Статус | Экспериментальный research-проект, не production. Активное обучение остановлено на 270K (продолжение — вручную `train.bat --resume`) |

---

## 0. Что это за проект

**FCF = Fractal Cognitive Field** (в коде/доках также называется **EVA**; `eva/agi_protocol.py:1-17` объявляет 4 аксиомы EVA). Расшифровка «Fractal Cognitive Field» — из `ARCHITECTURE.md:1` («Фрактальное Когнитивное Поле»); README: `# FCF / EVA — Fractal Cognitive Field`.

Цель: языковая модель **без трансформеров, без attention, без backprop и без градиентного спуска**. Память — обучаемое векторное поле на гиперсфере (VSA-суперпозиция), обучение — локальные правила (STDP, негативная выборка, контрастив, морфемная гармонизация), все числовые константы якобы выведены из λ_d (обобщённое золотое сечение). Русский BPE, концепт = токен словаря.

Это **прямой предшественник/родственник EVA-CLM**: токенизатор — `tokenizer.json` «WideBind v65536» через адаптер `SPCompatTokenizer` (`eva/symbolic/sp_compat.py`), а идеи FCF уже частично перенесены в EVA-CLM (`core/bind.py: TrajectorySpiralBind`, `_hybrid_bind`, `_rebuild_beams`, `_manifold_read`; `core/vsa_utils.py: zeckendorf_codes`).

Хронология: V1–V22+ (в репозитории осталось 5 аудиторских отчётов V22 от 2026-06-23), затем переход на WideBind-токенизатор и полный прогон STDP+CollocationMatrix (16–17.08.2026, журнал `docs/TRAINING_JOURNAL.md`). Аббревиатура в доках неоднократно трактуется как «нейро-символическая модель на саморганизующейся гиперсфере» (`ARCHITECTURE.md:3`).

---

## 1. Инвентаризация

### 1.1 Дерево (без мусора/бинарников)

```
FCF/
├── AGENTS.md, README.md, ARCHITECTURE.md, requirements.txt
├── train_full.py · inference.py · eval_metrics.py
├── train.bat · train_fast.bat · run_train.bat · train.ps1
├── eva/
│   ├── agi_protocol.py              # манифест 4 аксиом (dead: никем не импортируется)
│   ├── morph.py                     # pymorphy3 + rule-based морфемный разбор (440 строк)
│   └── symbolic/
│       ├── concept_space.py         # ЯДРО: ConceptSpace, FractalField, EntityField, Harmonizer, HDC
│       ├── stdp_trainer.py          # ЯДРО: все правила обучения
│       ├── crystal_generator.py     # ЯДРО: beam-генерация, RRF, GPU-тензоры
│       ├── fcf_config.py            # FCFConfig + FormulaCoefficients + 15 ParamDef
│       ├── branch_network.py        # CollocationMatrix (λ_d+PMI)
│       ├── syntax_lattice.py        # PPMI n-граммы + AMI-прунинг
│       ├── transition_manifold.py   # лучи VSA-переходов
│       ├── vsa_attention.py         # VSA-attention (Zeckendorf + signed spiral_bundle)
│       ├── hdtransformer_layer.py   # VSA-native «трансформер» (attention без LSH)
│       ├── semantic_piece.py        # MorphSTDP, CharEnvelope
│       ├── multi_level_encoder.py   # char→word→sent→cluster (λ_d-траектории)
│       ├── alphabet_basis.py        # детерминированные векторы букв/слов/морфем
│       ├── morph_vocab.py           # Zeckendorf-пути лемм/форм (Natasha)
│       ├── sp_compat.py             # HF BPE под API SentencePiece
│       ├── morpheme_tokenizer.py    # 3-уровневое сито (эксперимент)
│       ├── fibonacci_utils.py       # λ_d, Zeckendorf, TemporalZeckendorf, spiral_bundle, φ-разбор
│       ├── fractal_encoding.py      # Zeckendorf-пути + LCP-близость
│       ├── hormonal_system.py       # DA/5HT/NA/ACh модуляция
│       ├── parameter_optimizer.py   # Param cascade + PlateauDetector (в пайплайне НЕ используется)
│       ├── adaptive_controller.py   # доли подпространств, пороги ёмкости
│       ├── adaptive_error_tracker.py# EMA ошибок по концептам (используется)
│       ├── dimension_coordinator.py # VRAM-оценщик (используется частично)
│       ├── lsh_index.py             # LSHIndex/EntityFieldIndex (dead: никем не импортируется)
│       ├── checkpoint_manager.py    # AtomicCheckpointManager (dead)
│       ├── federated.py             # федеративная агрегация (dead)
│       ├── rng_registry.py, seed_registry.py
│       └── experimental/            # VSAGrid, VSAConvLayer, VSACNN, ResidueEncoder, vsa_utils
├── model/                           # HF-совместимый слой (устаревший, см. §6.3)
├── api/                             # FastAPI (тонкая обёртка над model/)
├── scripts/                         # BPE/морфология/диагностика/визуализация (17 скриптов)
├── tests/                           # test_stdp.py (153 KB!), test_spiral_primitives.py, test_sp_compat.py, test_morpheme_tokenizer.py
├── reports/                         # 5 аудитов V22 от 2026-06-23
├── docs/                            # TRAINING_JOURNAL, FIBONACCI_SEQUENCES, LANGUAGE_LAMBDA, MATHEMATICAL_FOUNDATIONS, FUTURE_DIRECTIONS, INTEGRATION_PLAN
├── data/antonyms.json               # 22 пары антонимов
├── real_data/                       # корпуса, tokenizer.json, bpe_morph.model, morph_vocab.json, morpheme_65k.json
└── checkpoints/                     # state.pkl (984 MB) + meta.json + corpus_lines.txt
```

### 1.2 Артефакты и данные

| Файл | Размер | Назначение |
|---|---|---|
| `real_data/tokenizer.json` | 8.0 MB | **основной** токенизатор — HF ByteLevel BPE (WideBind v65536) |
| `real_data/bpe_morph.model` | 5.8 MB | legacy SP BPE 65K–256K, морфемно размеченный корпус |
| `real_data/morpheme_65k.json` | 3.4 MB | 3-уровневый MorphemeTokenizer (эксперимент) |
| `real_data/corpus_1m.txt` | 163 MB | текущий обучающий корпус (1M строк) |
| `real_data/full_corpus_ru_clean.txt` | 1.52 GB | 9.35M строк (полный) |
| `real_data/full_corpus_ru_morph.txt` | 1.68 GB | то же с морфемным маркером `\u037E` |
| `real_data/val_corpus.txt` | 76 MB | валидация (PPL/acc) |
| `real_data/wiki_download/wikipedia_rudataset.parquet` | 597 MB | исходный дамп Википедии |
| `real_data/morph_vocab.json` | 71.6 MB | MorphVocab (Natasha, ~4.8 ч парсинга) |
| `checkpoints/state.pkl` | 984 MB | pickle {cs, lattice, concept_error, hormones} |
| `checkpoints/meta.json` | — | `{"lines": 270000, "pairs": 41539812}` |

### 1.3 Git-история (последние 20)

Последний блок коммитов — 16–17.08.2026: `9290210` (CollocationMatrix + MultiLevelEncoder + λ_d-rebuild), `960bd79` (SPCompat/ByteLevel BPE), `eb585e4` (3-уровневый MorphemeTokenizer), `e8e5e60` (signed phi digits, spiral_bundle, signed attention), `3510a90`/`d09face`/`3000806` (журнал обучения 230K/210K/…), `21d1061` (README в стиле WideBind). Рабочее дерево чистое.

---

## 2. Архитектура

### 2.1 Поток данных (обучение, `train_full.py:171-246`)

```
corpus line → SPCompatTokenizer.encode → ids
   ↓ batch=100 строк
STDPTrainer.train_batch (crystal_generator.py:1110 → stdp_trainer.py:180)
   ├─ _build_pairs: окно 4, skip-2, PMI-гейт, field-overlap, θ-гейты, антонимы   (stdp_trainer.py:676)
   ├─ _gpu_stdp_apply: scatter_add-ядро + momentum + destab + beam-pull          (stdp_trainer.py:1137)
   ├─ _gpu_poststdp_fused:
   │     ├─ _negative_sampling_gpu (sim>0.1)                                     (stdp_trainer.py:1360)
   │     └─ _contrastive_objective_gpu (top-2000 cosine, cooc/field-маски)      (stdp_trainer.py:1481)
   ├─ HDTransformerLayer.train_step (если use_hd_transformer=True)               (stdp_trainer.py:278-304)
   ├─ _centroid_pull_batch + _cluster_centroid_pull                              (stdp_trainer.py:1627/1685)
   ├─ lattice.update(ids) + _update_hdc_ngrams                                   (stdp_trainer.py:309-314)
   ├─ colloc.observe(2, cid, cid) — BPE-пары                                    (stdp_trainer.py:316-325)
   ├─ MultiLevelEncoder: char-траектория → colloc.observe(3, sent_vec, sent_vec) (stdp_trainer.py:327-367)
   └─ _harmonize_batch (dirty words): char↔word↔sent↔para binds, Harmonizer,
        MorphSTDP, morph-manifold, EntityField→STDP feedback                     (stdp_trainer.py:383)
   ↓ каждые 10K строк: pickle-чекпойнт + тестовая генерация                      (train_full.py:138-160)
```

Генерация (`crystal_generator.py:580-737`): beam width 5 → `_branch` (6+ сигналов через RRF) → softmax+top-p → EOS по пунктуации. `_branch` (`crystal_generator.py:818-1032`) собирает кандидатов из: BMSSP-BFS по PPMI-графу (`_graph_search`, :741), n-грамм решётки, HDC-памяти, секторного/полного векторного поиска, CollocationMatrix, TransitionManifold-beam, prior; затем VSA-attention-реранкинг, homeostatic boost, intent-бонус, анти-повтор/n-gram-блок, field-mask фильтр.

### 2.2 Представление концепта

- `FractalField` (`concept_space.py:229`): латентный код `z ∈ R^2048`, разбитый на 3 подпространства с пропорцией `λ²:λ:1` (`z_c` content ≈ 1170, `z_a` ≈ 723, `z_m` ≈ 155 при φ; фактические `l_c/l_a/l_m` — `adaptive_controller.py:55-63`).
- Вектор: `v = normalize(z · B)`, `B ∈ R^{2048×768}` — ортонормированный базис (QR от seed-RNG, `concept_space.py:263-266`).
- **В реальном прогоне `dim=256`, `latent_dim=2048`** — `train_full.py:102` (`ConceptSpace(vocab_size=V, dim=256)`); README/ARCHITECTURE описывают 768D. Это важный дрейф документации (§6.3).
- Инициализация: `z_c` — разреженный шум ~3% активных, `z_a` — малый шум, `z_m` — почти ноль (`concept_space.py:379-412`); rare-токены (freq<3) переинициализируются случайными единичными векторами (`reinit_rare`, :1857).
- L1 soft-threshold по `z_c`: `strength = λ_cid · max(0, 1 − 2·CE)`, цель — плотность 8%, `λ_cid` адаптируется (`:319-345`, `adjust_l1_lambdas:506`).
- Динамическая ёмкость: `grow_capacity` (плотность >15%, +50% латента, QR-добор базиса, :537), `prune_capacity` (>98% мёртвых измерений, :607).
- Поля: `field_bits[cid] = packbits(sign(z · W_proj))`, `W_proj ∈ R^{2048×512}` — обучаемая гиперплоскостная LSH; хеббово обновление + QR + collapse-guard (`:428-504`). Секторный индекс 3 уровней [4,10,20] бит с фокальным поиском (`:782-824`).

### 2.3 Внешние зависимости

`torch 2.5.1+cu121` (CUDA доступна), `numpy 2.4.3`, sentencepiece, tokenizers (HF), pymorphy3/natasha (морфология), scipy (свёртки), sklearn (mutual_info в dead-ветке), transformers (HF-обёртка). `torch.compile` включается на Volta+/≥3GB (`stdp_trainer.py:1813-1822`).

---

## 3. Корреляционные методы (ядро разбора)

Классификация: **A** — VSA-связывание (bind/unbind), **B** — статистические корреляции (PMI/PPMI/co-occurrence), **C** — cosine-механизмы (сходство/поиск/ранжирование), **D** — корреляционные градиенты обучения, **E** — слияние сигналов.

### A. VSA-связывание

#### A1. FFT-HRR bind / unbind — `concept_space.py:38-48`
```python
_hrr_bind(a,b)   = irfft(rfft(a) * rfft(b), n=len(a))        # circular convolution
_hrr_unbind(c,b) = irfft(rfft(c) * conj(rfft(b)), n=len(c))  # circular correlation
```
Назначение: обратимое связывание (`unbind(bind(a,b),b) ≈ a`, SNR ~ √D). Используется как база для всех остальных bind. Комментарий в шапке файла (:33-36) прямо фиксирует мотивацию.

#### A2. Hybrid bind/unbind + α-куррикулум — `concept_space.py:77-99`
```python
bind:   combined = α·hrr + (1−α)·(a*b);  normalize
unbind: combined = α·hrr_corr + (1−α)·(c*b);  normalize
```
`α` берётся из `_alpha_from_curriculum()` (:63-75): `α_min + (α_max−α_min)·exp(−decay_rate·t)`, `α_max=0.618`, `α_min=0.382`, `decay_rate=0.5` (`fcf_config.py:377-379`, rebuild :502-504). Смысл: на старте HRR-heavy (обратимость), к концу — выразительность element-wise. Куррикулум задаётся глобально `_set_alpha_curriculum(epoch,total)` (:57). **Грабля:** глобальные переменные; в inference/генерации α фиксирован на `hybrid_bind_alpha` (V22 NEURO, баг #6). Этот же hybrid-bind уже портирован в EVA-CLM (`core/bind.py:383`).

#### A3. Masked hybrid bind (поселективное связывание) — `concept_space.py:128-135`
```python
result = a.copy(); result[mask > threshold] = hybrid_bind(a,b)[mask > threshold]
```
«Attention на уровне измерений» без learned-проекций. В пайплайне не используется, покрыт тестом (`tests/test_stdp.py:1918`).

#### A4. Zeckendorf-weighted bind — `concept_space.py:101-123`
Вес `w∈[0,7]` раскладывается `Zeckendorf(w)` (напр. 7=5+2), каждая часть → `bind(vec, normalize(vec·(part/F_max)))`, результаты bundle-суммируются. Вес — структура, не скаляр. Используется в attention-механизмах.

#### A5. Signed φ-разбор и `spiral_bundle` — `fibonacci_utils.py:317-375`
```python
signed_phi_digits(x, K) → {j: digit∈{−1,0,1}}, rem   # x = Σ digit_j·φ^{−j}
spiral_bundle(vecs, weights) = Σ a_i·r_i / Σ max(a_i, 0)
```
Антизнание (`a_i<0`) не разбавляет знаменатель; при отсутствии положительных весов → ровно нулевой вектор («нет знания» структурно). Покрыто `tests/test_spiral_primitives.py` (telescopic_zero, norm growth, suppression).

#### A6. HDC n-граммы — `concept_space.py:847-952`
```python
hdc_bind = hybrid_bind; hdc_unbind = hybrid_unbind
hdc_permute(v,n) = roll(v,n); hdc_fib_permute = roll(v, Fib(t) mod dim)
hdc_bundle(v,accum,lr) = accum·(1−lr) + v·lr
hdc_ngram_repr(w1..wn) = ρ^{n−1}(w1) ⊛ … ⊛ wn          # bind(permute(wi, n−1−i), …)
hdc_update_ngram: running average next_code; LFU-эвикция при переполнении (max = 17711)
hdc_predict: query = mem_repr/||·||; score = cos(query, code)
```
Назначение: резервный n-граммный предсказатель. **Грабля (подтверждено кодом):** `_update_hdc_ngrams` пишет ключи в естественном порядке (`stdp_trainer.py:92`: `prefix = tuple(ngram[:-1])`), а `_branch` запрашивает `ctx_cids = list(reversed(cids[-2:]))` (`crystal_generator.py:849`) — ключи не совпадают, память фактически write-only; работает только fallback-ветка «context-as-probe» (`hdc_predict`, :922-936). Плюс fallback — полный перебор всех кодов O(V·L) на каждом шаге генерации.

#### A7. EntityField: кросс-уровневые bind/query — `concept_space.py:1081-1296`
```python
V(entity) += normalize(V(entity) + lr·bind(V(context), role))   # bind(), :1210
query(entity) = unbind(V(entity), role)                          # :1240
```
Роли CHAR/MORPH/WORD/SENT/PARA — квазиортогональные (QR от seed, :1110-1114). `_proj` (JL-проекция 768↔2048) для синхронизации word_store. Хранилище — dict `{(etype,id): fp16}`, cap 50K, LRU-кэш пар bind, TTL-cleanup (`cleanup:1255`), `decay(0.999)` (:1291). Смысл: «информация свёрнута в поле», unbind восстанавливает суперпозицию контекстов. Это концептуальный аналог VSA-памяти EVA-CLM.

#### A8. Harmonizer compose/decompose/harmonize — `concept_space.py:1304-1546`
```python
compose_word(parts, ctx): word = bundle_i(bind(morph_i, role_i))
                          + контекстная модуляция ROOT: root += 0.3·unbind(sent_vec, WORD_POS)
decompose_word(word):     morph_i = unbind(word, role_i)
harmonize(word):          pred_up = compose_word(morphs, sent_vec)
                          pred_dn = unbind(sent_vec, WORD_POS)
                          error   = 0.5·(pred_up − actual) + 0.3·(pred_dn − actual)
                          итерации до n_iter, разрыв при расхождении >1.5×
```
Роли ROOT/PREFIX/SUFFIX/ENDING/WORD_POS/WORD_ROLE. Назначение: связать морфемы и слова, top-down контекст. Интеграция — `_harmonize_batch` (см. §3.E4), с drift-skip cos>0.95.

### B. Статистические корреляции

#### B1. PMI внутри STDP-пар — `crystal_generator.py:1036-1098`
```python
PMI = ln( P(next|prev) / P(next) ),  P(next|prev)=count_pair/count_prev, P(next)=count_next/total
pmi_w = clamp(pmi/2 + 0.2, min=pmi_gate_min, max=2.0)
gate:  skip если pmi_w_raw ≤ min(pmi_gate_min·strength, pmi_gate_min·max(0.25, 1−0.75·CE_target))
```
Расстояние 1 — биграммы, 2 — skip-2. Высокий PMI = специфичная коллокация → больший LR. Инлайн-версия на GPU — `stdp_trainer.py:726-754` (та же формула).

#### B2. PMI-нормировка CollocationMatrix — `branch_network.py:40-55`
```python
pmi = log2(p_st/(p_s·p_t));  max_pmi = log2(total/min(count_s,count_t))
return min(1, pmi/max_pmi)          # нормировано в [0,1]
```
Клампинг в [0,1] делает PMI сопоставимым с λ-prior. `total==1 → 1.0`.

#### B3. PPMI-прунинг n-грамм + AMI — `syntax_lattice.py:196-252`
```python
PPMI(c|prefix) = max(0, log2 P(c|prefix)/P(c))
adjusted = raw_pmi − α/√cnt   (AMI-коррекция, α=0.5)
удаляются transitions с adjusted < 0.5 и cnt < 2 (только n≥3)
```
Мотивация в комментарии (:204-205): редкие переходы систематически завышены, 1/√cnt ≈ ожидаемый PMI при независимости. Экономия памяти ~50-60%.

#### B4. PPMI-кеш связей — `syntax_lattice.py:474-522`
```python
ppmi = max(log2(pair_p/marg_p), 0); кеш симметричен (a,b)/(b,a)
connections_of(cid, use_ppmi=True) — ранжирование соседей по PPMI
```
Используется в `_semantic_bootstrap`, `_graph_search`, hard-negative-фильтре.

#### B5. PMI-граф + BMSSP-BFS — `crystal_generator.py:741-814`
Рёбра: `w = max(min, 1 − min(ppmi/cap,1)·strength)` (высокий PPMI = короткое ребро). Мульти-источниковый BFS с бюджетом `B=2.0`, depth≤5, topk=8; финальный скор — RRF: `(n_src/total_src)/(B+dist)`.

### C. Cosine-механизмы

#### C1. λ_d-prior CollocationMatrix — `branch_network.py:135-200`
```python
cos = dot(s,t)/(||s||·||t||); dist = 1−cos
prior = λ^(−α·dist·capacity);  damping = λ^(−decay_st)
colloc = (1−β)·prior·damping + β·PMI,  β = (λ−1)/λ
```
4 уровня: d=1..4, `λ_d` = 2.0, φ, 1.839, 1.928; ёмкости `F^(d)_{d+8}`; `α=(λ−1)/F^(d)_{d+4}` (`_level_lam/_level_capacity/LevelConfig.build`, :28-73). **Уровень 2 — ключи по cid (STDP-safe)**, уровни 1/3/4 — хеш векторов (`_key`: FNV-1a от первых 256 dims, :353-361). `generate()` — пропорциональный выбор с top-k и `weights^(1/T)` (без softmax, :288-336). `hormonal_learn` (:261-284) — подкрепление/затухание переходов.

#### C2. Векторный поиск: top-k, секторы, LSH — `concept_space.py:2686-2725, 782-824`
- `topk_similar_concepts`: `sims = mat @ v_norm`, argpartition, сэмплирование 2000 при необходимости.
- `search_in_sector(query, depth)`: кандидаты из `_sector_index[depth][prefix]`, cosine внутри сектора — O(|sector|) вместо O(V).
- `focal_refine`: прогрессивное углубление 0→1→2 при нехватке кандидатов.
- `LSHIndex` (`lsh_index.py:7-79`): 4 таблицы × 8 бит случайных гиперплоскостей, union кандидатов, cosine-досчёт — **dead** (никем не импортируется).
- `field_overlap` (`concept_space.py:834-843`): popcount(AND битовых полей) — Hamming-корреляция LSH-подписей.

#### C3. VSAAttention (cosine → signed Zeckendorf → bind) — `vsa_attention.py:58-172`
```python
sim = dot(q,k)/||k||;  w = clamp(round(max_weight·sim), −7, +7)          # signed
parts = Zeckendorf(|w|);  value_part = bundle(bind(value, weight_hv(part)))
agg = spiral_bundle(contribs, signed_weights)                             # Σmax-нормировка
multi-head: bind(agg_h, head_role_h)
```
`weight_hv` — seed-RNG квазиортогональный вектор (исправление V21: раньше линейный scale). Позиции — Fibonacci shift. Интегрирован в `_branch` как реранкер (`crystal_generator.py:912-931`), `n_heads=1, use_fib_pos=False`. Покрыт тестами signed attention.

#### C4. HDTransformerLayer — `hdtransformer_layer.py:43-183`
```python
sims = [cos(q,k) for k in kv]; top_k отбор
adaptive quantile: z = (sim−mean)/std → clip ±2 → [0,7]
tree = Zeckendorf(w); weighted = bind(value, weight_hv(part)); sum → normalize
residual: out = normalize(q + aggregated);  FFN: _fractal_convolution(out, (3,5,7))
```
Заявлен как «LSH-attention без QK^T», но `self._lsh = None` (:39) и `_lsh_attention` делает **полный O(N²)** перебор. В пайплайне включён (`use_hd_transformer=True`, `stdp_trainer.py:278-304`): после негативной выборки каждый выход слоя тянет вектор концепта на 10%.

### D. Корреляционные градиенты обучения (не backprop)

#### D1. STDP-ядро: корреляция пары — `stdp_trainer.py:1056-1132`
```python
y = clamp(vg·vc, min=0.05)                       # корреляция контекст-цель
pair_delta = vc·elr − vg·(y·elr)                 # риманов tangent к vc (аналог Hebb)
lr = fw · exp(−dist/2) · pmi_w · field_w
     · (base + ACh·scale)·(base + DA·scale) · cluster_potential
θ_fast/slow: TemporalZeckendorf либо exp(−d/τ)
scatter_add по unique_gen; CE-EMA: ce ← decay·ce + (1−decay)·mean(1−y)
антоним: pair_delta *= −2.0
```
Плюс `_apply_vector_update` с `max_shift=0.5` и синхронизацией латентного кода (`concept_space.py:2420`), momentum (`_mom_t`, bf16), градиентный шум, destab-шум (случайные концепты вместо PPMI-соседей).

#### D2. Негативная выборка — `stdp_trainer.py:1360-1419`
```python
mask = sim(gv, noise) > 0.1;  neg_lr = avg_elr·ratio·0.3·(1 + 2·CE·field_gate)
grad = Σ_noise − Σ_sim·gv;   v_new = normalize(gv − grad·neg_lr)
```
Отталкивание от случайных концептов, взвешенное ошибкой (чем хуже знает — тем сильнее). CPU-версия :1325-1358 (тот же порог sim>0.1).

#### D3. Контрастив с hard-негативами — `stdp_trainer.py:1481-1620` (GPU), :1432-1479 (CPU)
```python
sim = g_vecs @ all_vecs.T                       # top-2000
valid_hn = ~self & ~cooc & (0.05 < sim < cos_upper)
cos_upper = 0.3 если overlap>0 (same field), 0.999 если overlap==0 (cross-field)
grad_hn = mean(cos·v_neg) − v_local;  v += grad_hn·contr_lr
contr_lr = avg_elr·0.3·(1 + 2·CE·field_gate)
cross-field reg: reg_val>0.2 & overlap==0 → repulsion с фактором 0.05
```
Ключевая идея: «похожие, но не встречающиеся вместе» разводятся; межполевые — агрессивнее, внутриполевые — мягче. Cooc-маски строятся scatter-ом из STDP-пар (:1516-1526); field-overlaps — чанкованный popcount (:1529-1540).

#### D4. Центроидные притяжения — `stdp_trainer.py:1627-1734`
```python
# предложение:
centroid = mean(vecs);  cn = normalize(centroid)
pull = cn − (v·cn)·v;   v += pull·lr·0.3          # риманов tangent, :1648
# кластер (по _cluster_map, pull_strength=0.05): то же по членам кластера
```
Регуляризатор уровня предложения/кластера. `_repel_centroid` (`concept_space.py:2379-2416`) — глобальное отталкивание от общего центроида тем же tangent-градиентом `sim·v − cn`, сила `|sim|·strength`.

#### D5. Латеральное торможение — `stdp_trainer.py:1017-1048, 1285-1319`
```python
sim = gv @ gv.T;  mask = sim > 2·inh_threshold;  diag=False
inhibit = Σ_j mask_ij·sim_ij·v_j − (Σ_j mask_ij·sim²_ij)·v_i    # векторно
v += normalize(inhibit)·inh_strength·base_lr
```
Риманов градиент (касательный к сфере), предотвращает коллапс. Fractal-версия: `sim·v_other − sim²·v` (`concept_space.py:2596-2597`). Collapse Guard из README (авто-усиление при cos_mean>0.08) **в коде не найден** — есть только `fluctuate_fractal` с масштабированием амплитуды по `current_cos` (`:2361-2369`).

#### D6. Semantic bootstrap — `stdp_trainer.py:851-926`
```python
pos_mean = normalize(mean(top-PPMI соседей));  neg_mean = normalize(mean(random))
pull = pos_mean − (v·pos_mean)·v;  push = (v − neg_mean·(v·neg_mean))·0.5
grad = clip_norm((pull+push)·lr, 0.3);  v += grad
```
Контрастив «притянись к PPMI-соседям, оттолкнись от несвязанных». В текущем `_train` не вызывается (метод доступен).

#### D7. Destabilization (Ланжевен) — `stdp_trainer.py:1182-1200`
С вероятностью `p = clamp(ce·0.5·destab_scale, ≤0.5)` концепт смещается к случайному концепту (вместо PPMI-соседа): `noise = v_rand − (v·v_rand)·v`, `acc = acc·(1−mix) + mix·noise·elr`. Выход из локальных аттракторов. Дополнительно `fluctuate_fractal` (:2361) — глобальный дрейф `c ← c·decay + N(0,amp)`.

#### D8. Антоним-репел — `stdp_trainer.py:807-844, 1094-1100`
Словарь `data/antonyms.json` (22 пары) + хардкод-fallback (:43-59); перезагрузка каждые 100 батчей; пары-антонимы инвертируют STDP-градиент ×(−2).

#### D9. EntityField→STDP feedback — `stdp_trainer.py:600-629`
```python
char_query = query('w', cid);  char_query_768 = proj.T @ char_query
error = char_query_768 − (v·char_query_768)·v
v += clip(error, ±0.1)·0.001·max(0.1, 1−2·CE)
```
Обратная связь символьного уровня в концепт-пространство (мягкая, с CE-затуханием).

### E. Слияние сигналов

#### E1. RRF (Reciprocal Rank Fusion) — `crystal_generator.py:878-910`
| Сигнал | Вес | Формула вклада |
|---|---|---|
| graph (BMSSP) | 0.420 | `rrf_graph · score` |
| syntax (n-граммы) | 0.259 | `rrf_syntax/(K+rank)` |
| hdc | 0.160 | `rrf_hdc·score/(K+1)` |
| vector (cosine/сектор) | 0.099 | `rrf_vector·sim/(K+1)` |
| colloc | 0.382 (α) | `rrf_colloc_alpha·colloc(2,prev,cid)` |
| beam (manifold) | `branch_conf_scale`=0.5 | `0.5·beam_sim/(K+1)` |
| prior | 0.061 | `rrf_prior/(K+1)·(1−min(freq/987,1))` |

Веса — нормированные λ-степени: `rrf = λ^k/Σλ^j`, k = 2,1,0,−1,−2 (`fcf_config.py:333-339`, rebuild :511-521). Далее: VSA-attention реранк (:912-931), homeostatic boost (`× (1+0.309·z)`, :933-936), intent-бонус `sim·(1−sim)` (:938-948), анти-повтор `exp(−penalty·count)` и блок n-грамм (:950-966), field-mask фильтр + бонус за overlap (:968-988), softmax + top-p + энтропия (:993-1032).

#### E2. RRF в graph search — `crystal_generator.py:806-814`: `(n_src/total_src)/(B+dist)`.

#### E3. Homeostatic boost — `concept_space.py:2654-2671`: z-score usage, клип ±0.3; usage EMA (α=0.1) + decay 0.98 (`:2619-2627`).

#### E4. Hormonal modulation — `hormonal_system.py`
DA/5HT/NA/ACh (0..1): DA — reward (match/mismatch/novelty/mastery/boredom), ACh — пластичность/новизна, 5HT — риск, NA — неопределённость. Выходы: `modulate_temperature`, `modulate_beam_width`; LR в STDP умножается на `(0.5+ACh·scale)·(0.5+DA·scale)` (`stdp_trainer.py:1080`). Тонические базлайны из λ: DA 0.618, ACh 0.382, NA 0.236, 5HT 0.146 (`fcf_config.py:341+`). **Дрейф:** при `use_fib_generalized=True` (default) код использует `tonic_decay = 1/λ = 0.618` (`hormonal_system.py:54-58`), а README обещает `1−λ^{−6}=0.944`.

#### E5. TemporalZeckendorf θ-гейт — `fibonacci_utils.py:284-314`
```python
zlen = len(zeckendorf(distance));  θ_base = 1/(1+zlen)
fast/slow — линейный rolloff после 5/10
```
Заменяет `exp(−d/τ)`; медленные пары дублируются (fast+slow). В GPU-ядре — та же схема (`stdp_trainer.py:1083-1088`), но с `dist` из меты.

---

## 4. Сводная таблица корреляционных методов

| # | Метод | Формула (ядро) | Файл:строка | В пайплайне | Статус |
|---|---|---|---|---|---|
| A1 | FFT-HRR bind/unbind | свёртка/корреляция | concept_space.py:38-48 | да (база) | ✅ |
| A2 | Hybrid bind + α-куррикулум | α·hrr+(1−α)·ew | concept_space.py:77-99 | да | ✅ (α заморожен вне train) |
| A3 | Masked bind | bind только при mask>T | concept_space.py:128-135 | нет | ⚠️ только тест |
| A4 | Zeckendorf-weighted bind | Σ bind(vec, vec·p/F) | concept_space.py:101-123 | через attention | ✅ |
| A5 | spiral_bundle (signed) | Σa·r/Σmax(a,0) | fibonacci_utils.py:358-375 | да (attention) | ✅ |
| A6 | HDC n-граммы | ρ-перестановки + bind + bundle | concept_space.py:847-952 | да | ⚠️ ключи reversed (write-only) |
| A7 | EntityField bind/query | V += bind(ctx, role); query=unbind | concept_space.py:1210-1252 | да | ✅ (cap 50K, fp16) |
| A8 | Harmonizer compose/decompose | bundle(bind(morph,role)); unbind | concept_space.py:1364-1546 | да | ✅ (drift-skip) |
| B1 | PMI→LR | ln(P(n|p)/P(n)) → clamp | crystal_generator.py:1036-1084 | да | ✅ |
| B2 | Нормированный PMI colloc | min(1, pmi/max_pmi) | branch_network.py:40-55 | да | ✅ |
| B3 | PPMI+AMI-прунинг | max(0,log2 P/P) − α/√cnt | syntax_lattice.py:196-252 | да (build) | ✅ |
| B4 | PPMI-кеш связей | симметричный кеш | syntax_lattice.py:474-522 | да | ✅ |
| B5 | BMSSP-BFS + RRF | w=1−ppmi/cap; (n_src/tot)/(B+d) | crystal_generator.py:741-814 | да | ✅ |
| C1 | CollocationMatrix | (1−β)λ-prior+β·PMI | branch_network.py:135-200 | да | ✅ L2; ⚠️ L1/L3/L4 хеш-ключи |
| C2 | Cosine top-k/сектор/LSH | cos + O(1) сектор | concept_space.py:2686,782; lsh_index.py | topk/сектор да | LSHIndex dead |
| C3 | VSAAttention | cosine→Zeckendorf→bind | vsa_attention.py:58-172 | да (реранк) | ✅ |
| C4 | HDTransformerLayer | cosine→quantile→Zeckendorf→bundle | hdtransformer_layer.py:43-183 | да | ⚠️ LSH не подключён, O(N²) |
| D1 | STDP pair kernel | vc·elr − vg·(y·elr) | stdp_trainer.py:1056-1132 | да | ✅ |
| D2 | Негативная выборка | Σn − Σsim·v при sim>0.1 | stdp_trainer.py:1360-1419 | да | ✅ |
| D3 | Контрастив hard-neg | mean(cos·neg)−v, cooc/field | stdp_trainer.py:1481-1620 | да | ✅ |
| D4 | Центроидные pulls | cn − sim·v | stdp_trainer.py:1627-1734 | да | ✅ |
| D5 | Латеральное торможение | sim·v_j − sim²·v_i | stdp_trainer.py:1285-1319 | да | ✅ |
| D6 | Semantic bootstrap | tangent pull/push | stdp_trainer.py:851-926 | нет в _train | ⚠️ вызывается вручную/тестами |
| D7 | Destab + fluctuate | шум к случайному концепту | stdp_trainer.py:1182; concept_space.py:2361 | да (destab_scale>0) | ✅ |
| D8 | Антоним-репел | ×(−2) | stdp_trainer.py:1094-1100 | да | ✅ (22 пары) |
| D9 | EntityField→STDP | tangent от unbind(word) | stdp_trainer.py:600-629 | да | ✅ |
| E1 | RRF-слияние | Σ w_i·rank-функция | crystal_generator.py:878-910 | да | ✅ |
| E2 | Homeostatic boost | z(usage), клип ±0.3 | concept_space.py:2654-2671 | да | ✅ |
| E3 | Гормоны | DA/ACh/NA/5HT → lr/T/beam | hormonal_system.py | да | ✅ |
| E4 | TemporalZeckendorf θ | 1/(1+len(Z(d))) | fibonacci_utils.py:284-314 | да | ✅ |

---

## 5. Обучение и оптимизация

### 5.1 Что заменяет лосс

Глобального лосса нет. Суммарный эффект батча — аддитивные обновления векторов (scatter_add), каждое из которых локально. «Ошибка» концепта — `CE = EMA(1 − y)`, `y = cos(v_gen, v_ctx)` (`stdp_trainer.py:1114-1123`); она модулирует: L1 (сильнее при низкой CE), neg_lr, contr_lr, destab-вероятность, pull_strength (EntityField-feedback), кластерный потенциал. Порядок применения правил фиксирован (`_train`, :261-367): STDP → negative → contrastive → HDTransformer → centroid pulls → lattice/HDC/colloc/ML → harmonize.

### 5.2 Балансировка

- **Гейты LR:** freq_weight `1/(1+ln(max(fa,fb))·scale)`, dist_weight `exp(−d/2)`, PMI, field_weight `min(1+ln(overlap+1)·scale, cap)`, θ fast/slow, гормоны, cluster_potential (minesweeper-инверсия: высокая CE → буст до 1.2, низкая → 0.8, по README).
- **Римановы касательные** во всех pull/push — не «сдувают» вектор с гиперсферы, нормализация после каждого шага.
- **max_shift=0.5**, `max_grad_norm` на группу, momentum μ=0.9 (`_mom_t` bf16), gradient noise (опция).
- **Дубль пары:** каждая STDP-пара добавляется дважды (fast+slow) при `slow_lr>1e-6` (:784-797, :835-844).

### 5.3 Расписания и адаптив

- Расписания — Fibonacci/λ (`fcf_config.py`): eval 1597, decay 4181, checkpoint 4181, warmup 987, cosine T0 6765, beam buffer 10946.
- `FormulaCoefficients.rebuild(lam, vocab, d)` пересчитывает ~92 поля из λ (fcf_config.py:479-570); `use_fib_generalized=True` по умолчанию.
- **ParameterOptimizer (15 ParamDef с триггерами `mean_cos`, `delta`, `ng_new`, `vacc1_stuck`, `*_plateau`) в реальном пайплайне не инстанцируется** — `train_full.py` его не создаёт, `CrystalGenerator` не имеет `opt`; используется только в тестах (`tests/test_stdp.py:276`). README §11 и ARCHITECTURE §11 описывают его как активный — **дрейф документации**.
- `AdaptiveArchitectureController` (пороги роста/прунинга, доли подпространств) — используется внутри `FractalField`.
- `AdaptiveErrorTracker` (EMA ошибок, LRU) — используется как `gen.concept_error`.
- `PlateauDetector` — часть parameter_optimizer.py, вне пайплайна.

---

## 6. Что доказано, что мертво, какие грабли

### 6.1 Доказано (факты из репозитория/прогонов)

1. **Тесты:** 390 passed / 1 failed / 7 skipped за 32.5 s (проверено на HEAD). Покрыты VSA-примитивы, spiral-примитивы (signed φ, telescopic zero, suppression), EntityField, Harmonizer, STDP-интеграция, GPU-пути, SPCompat (15 тестов), MorphemeTokenizer (15), FibonacciUtils. Единственный сбой — order-dependent (`test_dirty_cids_syncs_cpu` проходит изолированно).
2. **Обучение реально шло** 270K строк / 41.5M пар / ~9.5 ч последнего прогона; чекпойнты растут 909→984 MB; коллокации L2 4.85M→8.74M, L3 2563→4521 (монотонно, `docs/TRAINING_JOURNAL.md`, `train.log`).
3. **Пространство не коллапсировало:** `_train_status.json`: `cos_mean=0.00045`, `cos_std=0.0392` (очень низкая средняя корреляция — хорошо).
4. **Качественный сдвиг генерации** (журнал): 160K — цепочки коллокаций; 180K — предложения с предлогами/падежами; 250–270K — длинные связные структуры («получил место в россии», «с запада и стал род птиц семейства»); устойчивый мусор — цифровые/латинские байтовые токены WB-словаря.
5. **VSA-композиция морфем** проверялась на e5 (`scripts/test_vsa_composition.py`) — но скрипт ссылается на отсутствующий `full_corpus_ru.txt`, артефактов прогона в репо нет.
6. **PPL/acc-артефактов нет**: `eval_metrics.py` есть, `eval_*.json` в `real_data/` отсутствуют; `_evaluate` (`stdp_trainer.py:1740-1810`) считает «PPL» как `exp(mean(max_cos − ln(len(ctx))))` — это **не** языковая перплексия, сравнивать с LM нельзя. `vec_perplexity` = `exp(mean(−cos))` — вообще не вероятность.

### 6.2 Мертво/недоделано (по коду)

| Компонент | Факт | Последствие |
|---|---|---|
| `HDTransformerLayer._lsh` | `self._lsh = None` (hdtransformer_layer.py:39), `_lsh_attention` — полный перебор | O(N²), имя «LSH» ложно; включён в обучении |
| `GpuChunkManager.load_batch` | вызывается только в собственном docstring (crystal_generator.py:1182) | секторный paging не работает, только dirty-трекинг |
| `ZeckendorfQuantizer` | только в тестах (test_stdp.py:468+, test_spiral_primitives.py:333) | в пайплайне не участвует |
| `LSHIndex`/`EntityFieldIndex` | никем не импортируются | dead |
| `AtomicCheckpointManager` | никем не импортируется (пишут pickle вручную) | dead |
| `federated.py` | никем не импортируется | dead |
| `agi_protocol.py` | никем не импортируется | манифест |
| `ParameterOptimizer`/`PlateauDetector` | только тесты | автотюнинг не работает |
| `_hybrid_bind_masked`, `_analogy`, `VSACNN/VSAConvLayer`, `ResidueEncoder` | только тесты/`experimental` | не в пайплайне |
| `build_zeckendorf_fields`/`init_fields` | не вызываются из `train_full.py` | legacy-путь полей |
| `qwen_knowledge.py` | удалён; остался только `seed_from_qwen` (.npy) и путь `qwen_knowledge.npz` | Qwen-дистилляция не активна (ни `.npz`, ни `qwen_concept_vectors.npy` нет) |
| `MorphSTDP.char_vecs` | заполняется собственными seed-векторами (`_ensure_char_vec`, semantic_piece.py:68-75), не из `CharEnvelope.vecs` | два несогласованных символьных пространства |
| HF-слой `model/` | `_SPTokenizer` (sentencepiece) + старые пути `concept_space.json`; основной пайплайн — `tokenizer.json` + `state.pkl` | HF/API-обёртка несовместима с текущими чекпойнтами |
| `docs/FUTURE_DIRECTIONS.md` | 8 из 9 идей — «Статус: Идея» | план, не факт |
| AGENTS.md | ссылается на удалённые `qwen_knowledge.py`, `viz_tsne.py`, `precompute_qwen_knowledge.ipynb`, «294 passed» | устаревший журнал |

### 6.3 Грабли (самое важное)

1. **Двойное определение перехода (CPU vs GPU).** `_cpu_stdp_apply` и morph-манифолд пушат `T = hybrid_unbind(v_next, v_prev)` (`transition_manifold.py:136-146`), а GPU-путь пушит риманов tangent `T = vg − cos·vc` (`stdp_trainer.py:1166-1169`). Один и тот же `TransitionManifold` получает два разных типа векторов; `_branch` для beam-score использует `_vsa_transition` (unbind). Манифолд систематически «загрязнён».
2. **HDC-память write-only.** Ключи пишутся `(w1,w2)`, запрашиваются `(w2,w1)` (`stdp_trainer.py:92` vs `crystal_generator.py:849`). Накопление 17 711 записей × 2048 fp32 ≈ 145 MB без единого попадания; работает только fallback context-as-probe с полным перебором O(V).
3. **CollocationMatrix L1/L3/L4 на хеш-ключах от изменяющихся векторов.** `_key` берёт `round(v[:256]·1e3)`; векторы меняются каждый батч → старые записи не переиспользуются, память растёт (L3 = 4521 записей), а hit-rate близок к нулю. STDP-safe только уровень 2 (cid).
4. **Деградация скорости.** В логе batch 17.8–19 s на 230–260K, затем **106.4 s на 270K** (rate 4 l/s, ETA 57 h). Причина в коде не устранена; кандидаты — рост `colloc`/`lattice.ngrams`/HDC-памяти и ежебатчевый `_graph_cache.clear()`/полные проходы.
5. **Pickle-чекпойнты 984 MB.** Хранение всего `cs`+`lattice`; пришлось добавлять `__getstate__` для lock/лямбд (журнал 16.08); не портируемо, не совместимо между версиями.
6. **Дрейф документации против кода:** README/ARCHITECTURE — 768D/2048, реальный прогон — `dim=256`; «391 тест» — фактически 390+1; «hdc_memory_max=50000» — в коде 17 711 (F₂₂); «Collapse Guard cos_mean>0.08» — в коде нет; «tonic_decay=0.944» — в коде 0.618; ParameterOptimizer описан как активный — мёртв.
7. **Нестандартные метрики.** `perplexity` и `vec_perplexity` из `_evaluate` — эвристики, не PPL; `cos_mean` в `_train_status.json` считается по случайной подвыборке (2000) в eval_metrics, но статус-файл пишется другим кодом.
8. **Двойной учёт векторного сигнала в RRF:** `rrf_vector` добавляется и в общем цикле (:891), и в VSA-attention-блоке (:931) — один и тот же вес дважды.
9. **MorphSTDP → Harmonizer:** 768D→2048D теперь проецируется через `basis.T` (`stdp_trainer.py:656-665`) — исправлено, но `harm.morphemes` и `morph_stdp.morphemes` живут раздельно; прямое обновление `harm.morphemes` без dirty-трекинга (:580-598).
10. **Flaky-тест** `test_dirty_cids_syncs_cpu` — order-dependent (глобальное состояние GPU/CPU).
11. **Абсолютные пути** `C:\Users\black\...` в `train_full.py:13-15`; скрипты ссылаются на несуществующие `full_corpus_ru.txt`, `concept_space_*k.json` и т.п.
12. **SeedRegistry** бросает `RuntimeError` при повторной регистрации имени с другим master-seed (`seed_registry.py:25-33`) — при нескольких реестрах/тестах возможны ложные падения.
13. **`_gpu_stdp_core` мутирует `gen._ce_t`** (side effect в «чистой» функции, названной torch.compile-friendly) — ломает идемпотентность/compile-контракт.
14. **`hdc_predict` вызывается на каждом шаге генерации** и в худшем случае сканирует все коды (65 536 × 2048) — скрытая O(V) в луче.

---

## 7. Переносимость в EVA-CLM

EVA-CLM (`core/`) уже унаследовал: `_hybrid_bind`/`_hrr_bind` (`core/bind.py:379-389`), TrajectorySpiralBind с буфером переходов и перестройкой лучей (`bind.py:592-703`), zeckendorf_codes (`core/vsa_utils.py:19`), Fibonacci shifts. Поэтому ниже — только то, чего в EVA-CLM **нет** или что реализовано иначе.

### 7.1 Брать (топ-3 приоритета)

1. **CollocationMatrix как независимый статистический сигнал** (`branch_network.py`).
   - Что: cid-ключевая разреженная матрица `(1−β)·λ-prior + β·PMI` по 4 уровням, без softmax; L2 (token→token) полностью STDP/gradient-safe, т.к. ключи — id.
   - Куда в EVA-CLM: рядом с `logit_cache`/`memory_bank` как источник для финального гейта или для дешёвой CPU-валидации кандидатов; можно подавать как prior в σ×softmax-голову (аналог prior-слагаемого) или в диагностику.
   - Оговорка: брать только L2-логику и формулы `prior = λ^(−α·dist·cap)`, `β=(λ−1)/λ`; уровни на хеш-ключах не переносить (грабля #3).
2. **Римановы tangent-регуляризаторы без новых параметров** (`stdp_trainer.py:1627-1734`, `1285-1319`, `concept_space.py:2379-2416`).
   - Что: `pull = cn − (v·cn)v` (центроид), `inhibit = Σ sim·v_j − Σ sim²·v` (латеральное торможение), `repel = sim·v − cn` (глобальный центроид). Все — касательные к сфере, не требуют learned-весов, работают как auxiliary-loss/EMA-правило.
   - Куда: `core/losses.py`/`adaptation.py` как anti-collapse регуляризатор для кодов/направлений концептов и банков памяти; особенно полезно там, где эмбеддинги живут на сфере/нормируются.
3. **Hard-negative mining с cooc- и field-масками** (`stdp_trainer.py:1481-1620`).
   - Что: `top-2000 cosine`, исключение self и co-occurrence, порог `0.05<sim<0.3` внутри поля и `sim<0.999` между полями, отдельный cross-field repulsion; сила ∝ ошибке (`1+2·CE`).
   - Куда: если в EVA-CLM есть/будет contrastive-компонента для концептов/фантомов — это готовый рецепт; при чистом CE можно применить как mining для «лакун» (uncertainty) и фантом-банка.

Дополнительно (второй эшелон): homeostatic boost по usage (z-score, клип ±0.3) — балансировка редких концептов; антоним-репел ×(−2) — дешёвый и точный; `field_bits` (обучаемая бинарная гиперплоскостная LSH + 3-уровневый секторный индекс + focal_refine) — O(1)-маршрутизация кандидатов, ложится на concept_layer/logit_cache; MorphSTDP/CharEnvelope — индукция морфем для OOV/фантомов.

### 7.2 Брать с осторожностью

- **RRF-слияние сигналов** (`crystal_generator.py:878-910`): сама схема полезна (ранговое слияние устойчиво к разным шкалам), но веса λ-степеней — подгонка под FCF-набор сигналов; в EVA-CLM сигналы другие (memory reads, phantom confirmations, logit-cache) — пересчитать веса, не копировать 0.420/0.259/0.160.
- **TemporalZeckendorf θ**: интересная «лестница без τ», но EVA-CLM строит всю дисциплину на τ-поле; смешивать две метрики времени рискованно.
- **VSAAttention/HDTransformer** как реранкеры: идея (cosine → дискретный вес → bind вместо scale) ценна, реализация — O(N²), RNG-векторы на каждый вызов, α-куррикулум через глобалы. Брать только формулу `spiral_bundle(Σmax)` и signed-веса; LSH нужно реализовать (в FCF он не реализован).
- **EntityField cross-level bind**: концептуально совпадает с VSA-памятью EVA-CLM, но реализация — dict fp16 с pickle, cap 50K; для EVA-CLM переписать на тензоры/слоты банков.

### 7.3 Не брать

- **Парадигму «без backprop» целиком**: EVA-CLM обучается градиентами (σ×softmax CE); локальные STDP-правила не совместимы с CE-контуром без превращения в auxiliary loss, а тогда теряется весь смысл FCF.
- **FractalField (3 подпространства + per-concept L1 + динамическая ёмкость)**: в EVA-CLM своё кодовое пространство (`core/embedding.py`, `twin_free_codes`, sparse_block_codes); дублирование представлений даст конфликт шкал и чекпойнтов.
- **HDC n-граммную память** (A6): баг ключей + O(V) fallback + 145 MB; банки L1/L2 в EVA-CLM функционально сильнее.
- **Pickle-чекпойнты и dict-хранилища** (`state.pkl`, EntityField): не tensor-friendly, не масштабируются, ломают миграции.
- **GpuChunkManager, ZeckendorfQuantizer, LSHIndex, AtomicCheckpointManager** — unfinished/isolated в FCF; тащить незавершённое нет смысла.
- **Глобальный beam+softmax/top-p генератор**: распределение в EVA-CLM принадлежит голове; FCF-генератор — отдельный мир с собственными prior/MMI/anti-rep.
- **Культ λ_d как самоцель**: «0 эмпирических констант» приводит к формулам вида `tonic_decay=1/λ` вопреки собственному README; в EVA-CLM уже есть дисциплина τ — не импортировать нумерологию.

### 7.4 Риски/грабли при переносе

1. Не копировать «сигнал + тот же сигнал ещё раз» (двойной `rrf_vector`, грабля #8) — при слиянии проверять уникальность вкладов.
2. Любая память на hash-ключе от обучаемого вектора (L3) деградирует молча — ключ должен быть стабильным (id/слот).
3. Все cosine-пороги FCF (0.1, 0.3, 0.8, 0.95) калиброваны под 256D/768D-сферу FCF; в EVA-CLM другая размерность/масштаб — перекалибровать на своих распределениях (`cos_mean=0.00045` в FCF против типичных значений EVA-CLM).
4. Римановы касательные требуют нормировки после каждого шага и клампа `max_shift`; без них — расходимость.
5. Не тащить `_evaluate`-метрики как KPI: это эвристики, а не PPL.
6. При переносе кода помнить, что `α-куррикулум` — глобальное состояние (не потокобезопасно, не сериализуется в чекпойнт).
7. В FCF нет единого валидационного артефакта (eval JSON), поэтому «улучшения» методов в FCF не доказаны численно — в EVA-CLM любой перенос должен сразу получать A/B на своём валидационном контуре.

---

## 8. Приложения

### 8.1 λ_d и ключевые константы

| d | λ_d | Пример использования |
|---|---|---|
| 2 | 1.618034 (φ) | subspace l_c:l_a:l_m, RRF-веса, PMI/prior, α_max=0.618 |
| 3 | 1.839287 | CollocationMatrix уровень d=3 |
| 4 | 1.927562 | CollocationMatrix уровень d=4 |
| ∞ | 2.0 | уровень d=1 (`_level_lam`) |

Буферы/окна: beam buffer 10946 (F₂₁), HDC memory 17711 (F₂₂), morph manifold 2000, checkpoint 4181 (F₁₉), eval 1597 (F₁₇), hormonal windows 55/987 (F₁₀/F₁₆).

### 8.2 Ключевые конфиги (`fcf_config.py`)

`dim=768` (default) / фактически 256, `latent_dim=2048`, `max_n=3`, `ppmi_prune_threshold=0.5`, `ami_alpha=0.5`, `use_ami_correction=True`, `fractal_l1_lambda=0.001`, `l1_target_density=8%`, `entity_field_max_entities=50000`, `harm_drift_cos=0.95`, `fractal_hdc_memory_max=17711`, `beam_cos_threshold=0.8`, `beam_pull_strength=0.01`, `graph_search_B=2.0/depth=5/topk=8`, флаги `use_morph_stdp/use_vsa_attention/use_hd_transformer/use_temporal_zeckendorf/use_fib_generalized=True`.

### 8.3 Команды

```bash
# тесты
py -3.12 -m pytest tests -q

# обучение (resume)
py -3.12 train_full.py --resume --corpus real_data\corpus_1m.txt --learned-fields ^
  --field-bits 512 --neg-samples 3 --context-window 4 --pmi-gate 0.0 --gen-every 10000

# инференс/диагностика
py -3.12 inference.py --prompt "мир" | --neighbours "мир" | --retrieve "запрос" | --eval
py -3.12 eval_metrics.py latest
```

### 8.4 Топ-3 корреляционных метода FCF (для быстрого доступа)

1. **Hybrid VSA bind/unbind (FFT-HRR ⊕ element-wise, α-куррикулум)** + `spiral_bundle` со знаковыми весами — `concept_space.py:38-166`, `fibonacci_utils.py:317-375`.
2. **STDP pair kernel `vc·elr − vg·(y·elr)`** с мультипликативными гейтами (freq/dist/PMI/field/θ/гормоны) и scatter_add — `stdp_trainer.py:1056-1132`.
3. **Contrastive hard-negative mining с cooc/field-масками** (cosine top-2000, cross-field repulsion) — `stdp_trainer.py:1481-1620`.

### 8.5 Топ-3 переносимых идеи

1. CollocationMatrix L2 (cid-ключевой λ-prior+PMI) как независимый prior-сигнал.
2. Римановы tangent-регуляризаторы (centroid pull / lateral inhibition / global repel) без новых параметров.
3. Hard-negative mining с cooc-масками и field-overlap гейтингом (+ homeostatic boost и антоним-репел ×(−2) как дешёвые добавки).

### 8.6 Топ-3 грабли

1. Два несовместимых определения перехода (tangent на GPU vs unbind на CPU) в одном TransitionManifold; HDC-память write-only из-за reversed-ключей.
2. Hash-ключи от обучаемых векторов (CollocationMatrix L1/L3/L4) — тихая потеря hit-rate и рост памяти; плюс деградация batch 18 s → 106 s к 270K.
3. Дрейф документации (768D vs 256D, «391 тест» vs 390+1, ParameterOptimizer «активен» — мёртв, Collapse Guard отсутствует) и нестандартные PPL-метрики без валидационных артефактов.

---

*Отчёт составлен по фактическому коду HEAD `21d1061` (2026-08-17), проверочный прогон тестов выполнен 2026-09-19 (py 3.12.8, torch 2.5.1+cu121).*
