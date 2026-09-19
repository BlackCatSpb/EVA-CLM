# EXT_FCP_REPORT — Внешний разбор проекта FCP (MemBind)

**Дата разбора:** 2026-09-19
**Источник:** `C:\Users\black\OneDrive\Desktop\FCP`
**Метод:** чтение кода (не README), git-история, логи/чекпоинты, инвентаризация артефактов.
**Адресат:** EVA-CLM (`core/`), как проект-потомок.

---

## 0. Резюме для нетерпеливых

| Вопрос | Ответ |
|---|---|
| Что это | `FCP` — папка-наследник двух проектов: (1) **MemBind** — рекуррентная LM без attention (ковариационная память + билинейный bind + спектральный оператор), (2) вложенный legacy-проект **EVA-Ai** с модулем **FCP = Fractal Cognitive Processor**. Активная кодовая база — `ld_model/` + `train_large.py` (MemBind, июль 2026). |
| Статус | Исследовательский прототип. Последний коммит `d8120d3` — 2026-07-06; рабочее дерево грязное (fib_v не закоммичен). Крупный прогон (11.9B токенов, 39 жанров) **не завершён**. Большинство скриптов ссылаются на **отсутствующие чекпоинты**. |
| Объём | 2383 файла, **98.1 GB** (47.6 GB `token_stream.bin` = 11.9B токенов, 3.4 GB `russian_chunks.npy`, 3.3 GB чекпоинтов). Код корня: 129 файлов, ~9.7 MB. |
| Код ядра | `ld_model/core.py` 952 строки, `ld_model/readout.py` 252, `train_large.py` 567, `train_phase2.py` 438, `generate.py` 152. |
| Главное для EVA-CLM | Матричная ковариационная память `M[t]=d·M[t-1]+i·k⊗k` с параллельным сканом, first moment µ, random features, cognitive mirror (std по головам), ZeckendorfReadout (дерево битов), факторизация `V_shared + W_code`. |

---

## 1. Идентификация и происхождение

### 1.1. Что такое «FCP»

В корне — проект **MemBind** (README.md:1: «MemBind: Weight-Factorized Covariance Memory LM»). Имя папки `FCP` — легаси от вложенного `eva_ai/`:

- `eva_ai/core/fcp_pipeline.py:1` — «FCPPipelineV15 — интеграция FCP Pipeline в EVA-Ai»;
- `_archive/EVA_DOCUMENTATION.md:207` — «### 12. FCP (**Fractal Cognitive Processor**) - 65% → 85%»;
- `eva_ai/fcp_core/__init__.py` — «FCP Core Components - KCA, SRG, ConvergenceController, Types» (Knowledge Conscious Attention, Semantic Relevance Gate, FractalGraphV2).

То есть **FCP = Fractal Cognitive Processor** — модуль старого EVA-Ai (Qwen+LoRA+GNN+граф, OpenVINO), а не название LM-ядра. Родословная активного ядра: `EVA-Ai (eva_ai)` → `λ_d / LDStack` → `MemBind` (git: `c6051c2` «Initial: MemBind architecture» 2026-07-04 → `d8120d3` 2026-07-06). Косвенное свидетельство дальнейшей линии — чекпоинт проекта `WideBind` внутри папки (см. §6.4, риск 8).

### 1.2. Git-история (8 коммитов, 4–6 июля 2026)

```
d8120d3 2026-07-06 Optimizations: factorized training, AMP support, ZeckendorfReadout, seq_len=128
5065577 2026-07-05 prepare_corpus: cache token_counts.npy, resume support
680af7d 2026-07-05 Fix Pass 2 OOM: use memmap instead of np.empty for 11.9B tokens
5fc0171 2026-07-05 Multi-timescale heads + Random Features + First Moment + Cognitive Mirror
229d1dc 2026-07-05 Large-scale training prep: b_decay=4.0 (tau=55), b_i=1.0, prepare_corpus, train_large
31d7a08 2026-07-05 MemBind 30K steps: reorganization + DCT/λ-sliding prep
5d191db 2026-07-04 Fibonacci spectrum + blocks: fib_seq spectrum type with variable block sizes
c6051c2 2026-07-04 Initial: MemBind architecture — multi-head covariance memory + bind feedback
```

Незакоммиченные изменения: `ld_model/core.py` (+52), `train_large.py` (+52), `generate.py`, `README/docs`, `train_action.bat`; `docs/ARCHITECTURE.md` — untracked. Именно в этом diff — `fib_v` (Fibonacci V_emb) и интеграция ZeckendorfReadout в стек/трейнер (сам `readout.py` закоммичен в `d8120d3`), т.е. последняя волна работы не зафиксирована.

### 1.3. Инвентаризация (без мусора)

```
FCP/
├── ld_model/core.py            # MemBindBlock, MemBindStack, LDBlock, LDStack, scan, факторизация
├── ld_model/readout.py         # ZeckendorfReadout (дерево)
├── train_large.py              # per-genre обучение (factorized/zeckendorf/fib_v)
├── train_phase2.py             # обучение 89M (LDStack/MemBindStack, cosine)
├── generate.py                 # инференс (auto-detect чекпоинта)
├── colab_train.py, debug_eval.py
├── prepare_corpus.py, build_index.py, prepare_russian_data.py, download_libinpoc.py
├── token_stream*.bin (47.6+2.2+1.1+3.5+1.8 GB), russian_chunks.npy (3.4 GB)
├── russian_tokenizer/tokenizer.json   # BPE 50K
├── checkpoints/ (ACTION_step0.pt, interrupt_ACTION_step0.pt), checkpoints_micro/ (WideBind!)
├── docs/ ARCHITECTURE.md, LAMBDA_ARCHITECTURE.md, TRAINING_LOG.md, REPORT_30K.md, ROADMAP.md, SUMMARY.md, LONGTERM_MEMORY.md
├── experiments/ (28 скриптов: bind/cov/pscan/membind/mirror/multiscale/wide/…)
├── logs/ training_phase2*.log, training_fibseq.log, training_log.txt, eva_ai.log
├── outputs/ html-отчёты, _cube_test.txt, _fib_factor_test.txt, _gen_results.txt (bind/cov output пустые)
├── notebooks/ (lambda_colab.zip, colab_phase2_ru.ipynb)
├── tests/ (10 скриптов: smoke_phase1, diag_*, test_adaptive_*, test_learnable_v*, …)
├── scripts/ (inject_qwen_embeddings, qwen_knowledge_loader, verify_db)
├── _archive/ (старый EVA-Ai: PDF, ConceptNet, STUBS_REPORT.md, LAMBDA_*_DOCUMENTATION.md, run_eva.bat…)
└── eva_ai/ (вложенный legacy EVA-Ai: ~2000 файлов, FCP pipeline, Qwen, LoRA, GNN, веб-сервер)
```

---

## 2. Архитектура MemBind (по коду)

### 2.1. Поток данных (factorized, основной режим)

```
ids (B,L)
 ├─ embed:  F.embedding(ids, E_code) @ V_shared.T            train_large.py:223
 │          E_code = Zeckendorf-коды (V,K) frozen            core.py:70-85
 │          или embed_fib: bits → repeat_interleave → V_emb  train_large.py:197-202
 ├─ MemBindStack × n_layers                                  core.py:822-888
 │    ├─ MemBindBlock (см. 2.2)
 │    └─ MLP (factorized): rms_norm→V_shared→SiLU→up_code→down_code→V_shared.T  core.py:880-883
 ├─ final rms_norm
 └─ lm_head: (h @ V_shared) @ E_code.T + bias                train_large.py:227
             или lm_head_fib: h@V_emb → per-bit sum → E_code.T  train_large.py:204-213
             или ZeckendorfReadout (log-probs)              readout.py:91-125
```

Состояние между шагами (инкрементальный инференс): `(cov_state (B,H,r,r), mu_state (B,H,r), conv_state (B,D,k-1))` — `core.py:624-631, 702-704`.

### 2.2. `MemBindBlock` — ядро (`ld_model/core.py:453-789`)

Два варианта: `factorized=True` (`_forward_factorized`, 617-704) и стандартный (`_forward_standard`, 706-789). Отличия — только пространство вычислений (K-мерное vs D-мерное) и спектральный оператор (по-координатно vs по блокам).

Конструктор (factorized-ветка, 486-555):
- `V_shared` (D×K) — общий обучаемый базис, передаётся из `MemBindStack` (`core.py:833-836`, init = первые K столбцов DCT-II);
- bind: `W_u_code, W_v_code (K,bind_r)`, `W_out_code (bind_r,K)` — 503-505;
- память: `W_k_rf/W_q_rf (H,p,r)` при `cov_rf`, иначе `W_k_code/W_q_code (H,K,r)` — 509-518;
- гейты: `W_i_code (H,K)`, `b_i=1.0`, `W_decay_code (H,K)`, `b_decay` из τ — 520-529;
- чтение: `W_read_code (H,r,K)`, `W_mem2v_code (K,bind_r)` — 530-531;
- µ: `W_k_mu`, `q_mu (H,r,1)`, `W_mu_mem_code (H,K)` — 534-541;
- mirror: `W_u_m_code, W_v_m_code (K,bind_r)`, `W_out_m_code (bind_r,K)`, `mirror_scale=0.1` — 544-549.

**Forward (формулы, factorized, строки):**

1. `h_conv, conv_state = conv(h)`; `h_norm = rms_norm(h + h_conv, ln_w)` — 634-635.
2. Проекция в K: `hp = h_norm @ V_shared` — 639.
3. Bind: `u = hp @ W_u_code`, `v = hp @ W_v_code` — 642-643.
4. Random features: `h_rf = hp @ R_frozen` (K→p), `K_val = einsum('blp,hpr->bhlr', h_rf, W_k_rf)`, `Q = ... W_q_rf` — 646-649. Без RF — напрямую из `hp` — 651-652.
5. Входной гейт (экспоненциальный, неограниченный): `i_raw = hp·W_i_code + b_i`, `i_gate = exp(i_raw)` — 654-655.
6. Затухание: `decay_raw = hp·W_decay_code + b_decay`, `decay = sigmoid(decay_raw)` — 657-658.
7. **Ковариационный инкремент:** `delta = (K_val ⊗ K_val) * i_gate` — 660-661 (внешнее произведение ранга 1).
8. **Параллельный скан:** `M_all, final = parallel_prefix_scan(decay, delta, cov_state)` — 663-665.
9. **Чтение:** `mem_r = Q @ M_all`; `mem_D = einsum('blhr,hrk->blhk', mem_r, W_read_code)`; `mem_sum = Σ_h mem_D` — 667-670.
10. Mirror (673-678): `disagreement = std_h(mem_D)`; `mirror_delta = ((disagreement@W_u_m) ⊙ (hp@W_v_m)) @ W_out_m`; `mem_sum += mirror_scale · mirror_delta`.
11. First moment (681-692): `K_mu = einsum(h_rf/hp, W_k_mu)`, `b_mu = K_mu * i_gate`, `mu_all = scan1d(decay, b_mu)`, `mu_read = mu_all · q_mu`, `mem_sum += mu_read @ W_mu_mem_code`.
12. Bind-enhance: `v_enh = v + mem_sum @ W_mem2v_code`; `h_adapt = hp + (u ⊙ v_enh) @ W_out_code` — 694-696.
13. Спектр: `h_scaled = h_adapt * lambda_k`; `delta_spec = h_scaled @ V_shared.T`; `h_out = h + delta_spec` — 698-700.

### 2.3. Параллельный скан (Hillis-Steele) — `core.py:387-437`

```
M[t] = a[t]·M[t-1] + b[t],  M[-1] = state
комбинатор: (A1,B1)∘(A2,B2) = (A1·A2, A2·B1 + B2)
```
- `parallel_prefix_scan` (387-412) — для матриц (B,L,H,r,r), `a` (B,L,H); цикл `stride*=2`, O(log L) шагов, autograd-safe (cat, без in-place).
- `parallel_prefix_scan_1d` (415-437) — для векторов µ (B,L,H,r).
- Учёт начального состояния: `M = M + A * state.unsqueeze(1)` — 410-411.

### 2.4. Спектральный оператор

- `compute_spectrum` (`core.py:113-173`): типы `fib_root`, `fib_seq`, `fib_ratio`, `linear`, `hybrid`; `fib_seq` даёт λ = F₂..F_{K+1}, нормированные в [0.8,1.8], и block_sizes ∝ F_k (напр. `[10,21,31,51,82,134,216,351]` при D=896, K=8).
- `fibonacci_roots` (90-101) — бисекция корней xᵏ=xᵏ⁻¹+…+1 (1.618, 1.839, 1.927, 1.966, …→2).
- `dct_basis` (187-194) — DCT-II, ортонормированные строки.
- `slide_lambda` (197-200): `λ_layer = λ · (0.5 + layer/(n-1))` — 0.5× внизу, 1.5× вверху.
- В factorized-режиме блоки не используются: λ применяется покоординатно в K-пространстве (698-700); в стандартном — по блокам (778-785).

### 2.5. Legacy `LDBlock`/`LDStack` (λ_d, `core.py:249-367, 891-952`)

- `LDBlock`: conv → rms_norm → `α = sigmoid((h_norm@W_gate + b_gate)·gate_scale)` (322-325) → `Δ = V_eff·diag(α⊙λ)·V_effᵀ·h_norm` (327-363); опционально `recurrent_scan` — последовательный цикл по токенам (337-351) и Cayley-вращение базиса `R=(I-S)⁻¹(I+S)`, O(D³) solve на forward (302-310).
- `LDStack`: `adaptive_gain = mean(α)` модулирует MLP-апдейт (934-939); `use_global_context` — двухпроходный контекст (946-948). В MemBind не используются.

### 2.6. `ZeckendorfReadout` (`ld_model/readout.py:44-252`)

Дерево битов Цекендорфа с обучаемыми центроидами:
- `codes (V',K)` — коды без соседних единиц (73-78); `centroids (K,2,2,D)`, невалидный (state=1,digit=1) занулён (84-89).
- `log P(i|h) = Σ_k log P(b_k | h, state_{k-1})`, `logit[k,state,digit] = h·c[k,state,digit]`, `log_softmax` по digit (107-125).
- `combined_idx = prev_bits*2 + codes` — gather по таблице (114-124).
- `forward_log_probs` — векторизация по всем V' (127-160); `predict` — жадный/сэмплинг обход дерева (162-202); `compare_with_lm_head` — top-k overlap и KL против обычной головы (204-252).

### 2.7. Факторизация весов и Fibonacci V_emb

- Вся сеть: `W(D×W) → V_shared(D×K) + W_code(K×W)`; MLP тоже (`mlp_up_codes (K,bottleneck)`, `mlp_down_codes (bottleneck,K)`) — `core.py:857-866`. Не факторизуются: conv (буфер D×48), rms-norm веса, `lm_head_bias`, `E_code` (frozen).
- `fib_bit_sizes` (794-811): битам 0..11 — размеры [1,2,3,5,8,13,21,34,55,89,144,233], остальным по 1; при K=24 сумма 620 ≤ D=896.
- `embed_fib`/`lm_head_fib` — `train_large.py:197-213`: биты размножаются `repeat_interleave` по размерам, `V_emb (D,620)` (init randn·0.01); обратно — посегментная сумма.

### 2.8. Конфиг (`LDConfig`, `core.py:16-58`) — ключевые поля

`D=896, n_layers=24, n_modes=8, vocab=50000, bottleneck=896, kernel_size=48, cov_heads=4, cov_r=16, bind_r=16, spectrum_type='fib_seq', spec_lo=0.8, spec_hi=1.8, dct_basis, lambda_sliding, cov_first_moment, cov_rf, cov_rf_dim=64, cov_multi_timescale, cov_tau_lo=3, cov_tau_hi=200, cov_mirror, factorized, factorized_K=24, fib_v, fib_v_Kstack=24`.

---

## 3. КОРРЕЛЯЦИОННЫЕ МЕТОДЫ (главный раздел)

> В FCP нет ни одной cosine-similarity функции и ни одного attention-слоя. Вся «корреляция» — это (а) второй момент (внешние произведения) с рекуррентным затуханием, (б) билинейное связывание (покомпонентное произведение) и (в) статистическая дисперсия между головами. Ниже — все механизмы.

### 3.1. Multi-head covariance memory — второй момент (ядро)

**Формула:**
```
delta_h[t] = i_h[t] · (k_h[t] ⊗ k_h[t])          # r×r, ранг 1
M_h[t]    = d_h[t] · M_h[t-1] + delta_h[t]       # рекуррентность 1-го порядка
mem_h[t]  = q_h[t] · M_h[t] · W_read_h           # чтение
```
**Где:** `ld_model/core.py:660-661` (delta), `663-665` (scan), `667-670` (read); стандартный вариант — `743-753`.
**Смысл:** ковариация проекций k по времени с экспоненциальным забыванием; замена KV-cache: состояние `H×r×r` (4×16×16 = 1024 числа на слой, ~4KB fp32). Аналог mLSTM/GLA/linear attention, но без нормализации и softmax. `M` хранит корреляции между компонентами k-пространства, а не сами значения (сжатие истории).
**История:** прототипы `experiments/test_cov_gate.py:188-204` (последовательный цикл, `M = decay*M + i*kᵀk`), `test_pscan_gate.py:185-193` (параллельный скан), `test_membind.py:183-198` (multi-head H=4,r=8), затем `core.py`.

### 3.2. Bilinear bind (VSA-binding) и memory feedback

**Формулы:**
```
u = h_norm · W_u          # (B,L,bind_r)
v = h_norm · W_v
v_enh = v + mem_sum · W_mem2v
h_adapt = h_norm + (u ⊙ v_enh) · W_out
```
**Где:** `core.py:642-643, 694-696` (factorized), `726-727, 775-776` (standard); концепция — `docs/LAMBDA_ARCHITECTURE.md:131-151` («билинейный bind в терминах VSA»); прототип — `experiments/test_bind_gate.py:146-195` (`BindGateBlock`, `W_out` инициализируется нулём → на старте identity).
**Смысл:** покомпонентное произведение двух low-rank проекций одного токена — нелинейность без softmax, «связывание» признаков. Память модулирует `v` (feedback). Это прямой VSA-bind, но в пределах одного токена (не между токенами).
**Важный эмпирический факт:** по `docs/REPORT_30K.md:63` норма `W_v` выросла +289% (лидер), а `W_read` −42%, `W_mem2v` −28% — **bind-путь доминирует, память при L=128 почти не используется**.

### 3.3. Memory read через корреляционную матрицу

**Формула:** `mem_r = q·M`, `mem_D = mem_r·W_read`, `mem_sum = Σ_h mem_D`.
**Где:** `core.py:667-670`; стандарт — `750-753`.
**Смысл:** запрос q «выбирает» из накопленной корреляционной матрицы те направления, что коррелировали с историей; `W_read` проецирует обратно. В отличие от attention — без softmax-конкуренции: все головы суммируются линейно.

### 3.4. Random Features (kernel-аппроксимация корреляции)

**Формулы:**
```
h_rf = hp · R_frozen                       # K→p (factorized) или D→p (standard)
k_h  = h_rf · W_k_rf,  q_h = h_rf · W_q_rf # p→r, обучаемые
R_frozen ~ N(0, 1/√K) (factorized: K=24,p=64), заморожен
```
**Где:** конструктор `core.py:509-518` (factorized), `591-596` (standard); forward `646-649`, `729-732`; прототип — `experiments/test_cov_enhanced.py:53-59, 93-99`.
**Смысл:** ёмкость ковариации r×r = 16×16 та же, но k-представление обогащается нелинейной (случайной) проекцией — аналог random-feature kernel approximation (Performer/FAVOR+). Заморозка R не даёт градиенту «схлопнуть» базис. Комментарий кода: «ёмкость ×4 в тех же 16×16» (`core.py:460`).

### 3.5. First moment µ — сохранение направления

**Формулы:**
```
b_mu[t] = i[t] · K_mu[t]
mu[t]   = d[t]·mu[t-1] + b_mu[t]         # r-вектор
mu_read = mu[t] · q_mu                   # (H,) скаляры
mem_sum += mu_read @ W_mu_mem
```
**Где:** `core.py:681-692` (factorized), `762-771` (standard); `parallel_prefix_scan_1d` `415-437`; прототип `experiments/test_cov_enhanced.py:122-131`.
**Смысл (из LAMBDA_ARCHITECTURE.md:281-291):** второй момент теряет знак (`k·kᵀ = (−k)(−k)ᵀ`), первый момент сохраняет направление. «Удваивает ёмкость 16×16 без роста r», +0 FLOPs относительно скана.

### 3.6. Cognitive Mirror — корреляция/согласие между головами

**Формулы:**
```
disagreement = std_h(mem_D)                       # (B,L,K) — разброс по H головам
mirror_delta = ((disagreement·W_u_m) ⊙ (hp·W_v_m)) · W_out_m
mem_sum += mirror_scale · mirror_delta            # mirror_scale init 0.1
```
**Где:** `core.py:672-678` (factorized), `755-760` (standard); конструктор `544-549`; прототип и варианты — `experiments/test_mirror.py:21-74` (linear / self_bind / hv_bind / std_bind), тест на синтетике `177-272`.
**Смысл:** если головы с разными τ расходятся — зеркало билинейно превращает дисперсию в коррекцию. Использует тот же bind-механизм (`u⊙v@W_out`), вход u — disagreement, а не h. Это единственная «статистическая корреляция в лоссе/архитектуре» — и то как std, а не корреляция.

### 3.7. Мультимасштабные головы (τ-спектр)

**Формулы:** `τ_h = exp(linspace(log τ_lo, log τ_hi, H))`, `d=1−1/τ`, `b_decay = −log(1/d − 1)`; значения — буферы (заморожены).
**Где:** `compute_timescales` `core.py:440-448`; init `523-529` (factorized), `579-585`; тест `experiments/test_multiscale.py:30-63, 94-111` (переопределяет `b_decay` на голову).
**Смысл:** H=4 головы получают τ=[3,12,49,200] — локальные n-граммы / фразы / предложения / абзацы; градиент не может их сдвинуть. Это временная, а не корреляционная структура, но именно она делает `M_h` многомасштабной корреляционной памятью.

### 3.8. Спектральный оператор `V·diag(λ)·Vᵀ`

**Формула:** `Δ = V_eff·diag(λ)·V_effᵀ·h`, λ из Фибоначчи; в factorized — `Δ = (h·V ⊙ λ)·Vᵀ`.
**Где:** `core.py:698-700` (factorized), `778-785` (standard), `LDBlock` 327-363.
**Смысл:** смешивание/фильтрация частотных компонент (при DCT-базисе — буквальный фильтрбанк; `experiments/analyze_math.py:164-176`). Не корреляция, но задаёт временные масштабы через λ>1 (усиление) / λ<1 (демпфирование).

### 3.9. Zeckendorf-коды как «разреженный ортогональный» код

**Формула:** токен i ↦ бинарный вектор без соседних 1 (теорема Цекендорфа), `E_code (V,K)`; embedding `E_code[i]·V_sharedᵀ`, lm_head `(h·V_shared)·E_codeᵀ`.
**Где:** `core.py:70-85`; `train_large.py:171-179, 223, 227`; тест `_test_factorization.py:11-36`.
**Смысл:** замена таблицы эмбеддингов на «кодовую книгу»; коды разреженные и почти ортогональные (перекрытие ограничено структурой), т.е. корреляция между токенами мала по построению. Дерево читаута (`readout.py`) факторизует вероятность токена по битам — по сути chain-факторизация без softmax по всему словарю.

### 3.10. Чего в FCP НЕТ (важно для EVA-CLM)

- нет cosine-similarity, dot-product attention, softmax-attention, key-value cache;
- нет корреляционных/статистических лоссов: только `F.cross_entropy` (`train_large.py:243-248`, `train_phase2.py:334`) и NLL Цекендорфа (`train_large.py:236-240`);
- нет нормализации ковариации (ни след, ни деление на число токенов) — `M` растёт как сумма затухающих внешних произведений;
- нет механизма «забывания по контенту» — decay чисто входной (sigmoid от h), не зависит от запроса;
- нет чтения памяти с температурой/конкуренцией — только линейная сумма голов.

---

## 4. Обучение и оптимизация

### 4.1. Пайплайн данных

- `prepare_corpus.py`: обход `main-russian/<жанр>/*.txt`, очистка, токенизация BPE (50K, `russian_tokenizer/tokenizer.json`), запись `token_stream_{GENRE}.bin` (int32) по жанрам; `build_index.py` → `token_index.json` (ACTION 549M токенов, DETECT 881M; полный `token_stream.bin` = 11.9B токенов, 39 жанров).
- `EpochDataset` (`train_large.py:94-115`): окна по seq_len, детерминированная перестановка на эпоху (`rng_seed=42`), val — `offset=n_train`, `rng_seed=0`.
- `train_phase2.py:96-112`: чанки (N,128) держатся на CPU, батч собирается на GPU.

### 4.2. Гиперпараметры и расписания

| | `train_phase2.py` (89M) | `train_large.py` (factorized) |
|---|---|---|
| Optimizer | AdamW, wd=0.01 (`215`) | AdamW, betas=(0.9,0.98), eps=1e-8, wd=0.01 (`262-264`) |
| LR | 1e-3, warmup 5% + **cosine→0** (`217-221`) | 1e-3, warmup 500, затем **константа** (`269-272`) |
| Grad clip | 1.0 (`348`) | 1.0 (`479`) |
| B/accum/seq | 4/8/128 | 8/4/128 (батчи ACTION) |
| NaN-политика | skip batch (`330-338`) | нет (падение) |
| AMP | нет | `GradScaler` при `--amp` (`264`), по факту отключён (NaN на шаге 220) |
| Чекпоинты | `phase2_stepN.pt` + `best`/`epoch` (`359-365, 417-422`) | `best_{genre}.pt` + последний step, **предыдущий step удаляется** (`384-395`) |
| Eval | по эпохам, 500 чанков | каждые `--eval_every`, early stop `--patience 3` (`507-530`) |
| Отчёты | HTML с нормами параметров (`240-314`) | HTML аналогично (`298-378`) |

### 4.3. Лоссы

- CE по всем токенам (weight tying embed/lm_head, `train_phase2.py:144`).
- Для Zeckendorf: `-mean log P(target|h)` — точная нормированная NLL через дерево (`train_large.py:236-240`).
- Aux-лоссов нет. Балансировка — только weight tying, grad clip, warmup.

### 4.4. Известные проблемы обучения (из логов/доков)

- `logs/training_phase2.log`: resume упал на несовпадении state_dict (сменилась архитектура), затем **CUDA OOM** на backward.
- `TRAINING_LOG.md`: grad norm = 0 на чётных чекпоинтах (отчёт считался до `zero_grad`, позже починено — `train_phase2.py:350-351`).
- AMP: «Disabled (NaN at step 220)» (`TRAINING_LOG.md:217`).
- 2.6M токенов на 89M параметров → переобучение (eval 308 vs train 138).

---

## 5. Что доказано, что мертво, что недоделано

### 5.1. Доказано (числами/логами/чекпоинтами)

| Факт | Доказательство |
|---|---|
| Архитектура MemBind обучается | `docs/REPORT_30K.md`: 89.1M, 30K шагов, train PPL 138 / eval PPL 308, grad 0.7–0.9, hidden σ=1.0 |
| Старый λ_d дообучен до PPL≈131 | `model_start.pt`: step=37500, epoch=3, best_ppl=131.53 (LDStack, D=896, L=12, K=4) |
| Covariance memory + скан работают | `experiments/test_pscan_gate.py`, `test_membind.py`, `test_cov_enhanced.py` (forward/backward, без NaN); VSA-скан совпадает с последовательным до 1e-5 (`test_wide_membind.py:319-360`) |
| MemBind быстрее/качественнее альтернатив | `experiments/analyze_architecture.py:215-222`: CovGate seq 438 PPL/260 tok/s; BindGate 793/1175; PScan 688/1212; **MemBind 466/962 (94% качества, 3.7×)** |
| DCT+slide не ломают forward | `experiments/test_dct_sliding.py:6-10`: loss 14.6/14.6/14.3/14.9 (baseline), nan=0 (числа — на случайных токенах, только sanity) |
| Первый момент и RF не ломают обучение | `test_cov_enhanced.py`, `test_ablation.py` (скрипты; результаты в репо не сохранены) |
| Цекендорф-коды корректны | `_test_factorization.py:32-34` (нет соседних 1), `readout.py:73-89` |
| Epoch reset даёт скачок PPL | `TRAINING_LOG.md:182-184` (−65% на границе эпох) |
| Гиперсеть невыгодна математически | `test_hyper_membind.py:270-296`: decoder всегда дороже хранения весов; выигрыш только при L > coord_dim |

### 5.2. Мертво/недоделано (следы в коде)

| Объект | След |
|---|---|
| Long-term memory (сессионная SVD `(p,q)` + адаптер) | `docs/LONGTERM_MEMORY.md` — концепция; в `core.py` нет `session_p/q`, нет SVD. Сама дока пишет «внедрять после PPL<50» (101) |
| Fibonacci V_emb run (12 дней, 549M токенов) | README:56 «🔄 В работе»; но `checkpoints/` содержит только `ACTION_step0.pt` **без `V_emb`** (flat K=24), финального fib_v чекпоинта нет; последний коммит 06.07.2026 |
| Заявленный PPL=1.8 (flat K=24, 20.5K шагов) | `docs/TRAINING_LOG.md:209-221`; чекпоинт `best_ACTION.pt` отсутствует (`debug_eval.py:11`, `_test_fib_factor.py:32` ссылаются на несуществующие файлы). PPL 1.8 (loss 0.59) на 1.6M параметров/4% данных выглядит как переобучение/утечка — **не подтверждено** |
| Пост-фактум факторизация обученных весов через V | `_test_cube.py`, `_test_fib_factor.py`, `outputs/_cube_test.txt`, `_fib_factor_test.txt`: rel err ≈ **0.996** для всех весов (block-mean и row-truncation). Факторизация возможна только с нуля |
| VSA-wide / Hybrid / Grouped-bilinear варианты | `test_wide_membind.py`, `test_wide_comparison.py` — только forward/backward/param-count, обучения нет |
| `learnable_V` (Cayley), `recurrent_scan`, `adaptive_gain`, `use_global_context` | Реализованы в `LDBlock`/`LDStack` (302-310, 337-351, 934-939, 946-948), но MemBind их не использует; `global_context` нигде не включён |
| Mirror варианты linear/self_bind/hv_bind/std_bind | `test_mirror.py` — тест 100 шагов на синтетике; результаты не сохранены; в модель вошёл только `std_bind` |
| Часть experiments | Ссылаются на отсутствующие чекпоинты: `phase2_step30000.pt`, `model_step25000.pt`, `phase2_best.pt`, `ACTION_step2500.pt`, `fresh_start_test.pt` — большинства нет |
| `outputs/bind_output.txt`, `cov_output.txt`, `fresh_start_test.log/err` | **Пустые файлы** |
| AMP | Отключён после NaN (`TRAINING_LOG.md:217`), хотя код поддержки есть (`train_large.py:57-58, 264`) |
| Resume | `logs/training_phase2.log` — RuntimeError несовпадения state_dict; `logs/training_phase2_full.log` — resume с `best_ppl=5104`, OOM |
| Legacy `LDBlock` | Помечен «legacy» (`core.py:6, 164`), но остаётся в коде |
| `_archive/`, `eva_ai/` | Старый EVA-Ai (ConceptNet, PDF, батники, `STUBS_REPORT.md`, `duplication_report.txt`) — не связано с MemBind |

### 5.3. Грабли (технические и методологические)

1. **Удаление предыдущего чекпоинта** (`train_large.py:390-394`): история шагов теряется, остаётся только последний + best. В результате нет ни одного промежуточного fib_v-чекпоинта.
2. **Bind-only коллапс:** при L=128 модель отказывается от памяти (`W_read` −42%, `W_mem2v` −28%, τ≈7.5 — не растёт; `REPORT_30K.md:63-67`). Нет лосса/регуляризации, заставляющей читать память.
3. **Экспоненциальный входной гейт** `i=exp(·)` с `b_i=1.0`: не ограничен, легко даёт взрыв/NaN в fp16 (AMP отключён на шаге 220).
4. **Нет нормализации ковариации:** `M` — сумма без деления на число токенов; при длинных контекстах масштаб растёт (спасает только decay и RMSNorm вокруг).
5. **Смена архитектуры ломает resume** (logs), а `save_checkpoint` пишет без версии конфига — старые чекпоинты несовместимы.
6. **Отчёт до zero_grad** → grad_norm=0 на чётных шагах (исправлено только в `train_phase2.py`).
7. **Разные LR-расписания** (`train_large` — константа, `train_phase2` — cosine) → чекпоинты и PPL несравнимы между скриптами.
8. **Конфиговая путаница:** в factorized-режиме `n_modes=8` игнорируется, λ считается по `factorized_K=24` (`core.py:488-499`); в `train_large.py` это нигде не сказано.
9. **Zeckendorf-кламп:** `target.clamp(0, Vp-1)` (`readout.py:121`) молча подменяет токены вне представимого диапазона (для vocab=50000 диапазон покрыт, но при vocab>75024 — тихая порча лосса).
10. **Хардкод путей:** `train_action.bat` (`cd C:\Users\black\OneDrive\Desktop\fcp` — строчными, ломается на регистрозависимой ФС), `prepare_corpus.py:15` (`SRC_ROOT`).
11. **Огромные артефакты в OneDrive:** 47.6 GB `token_stream.bin` + 3.4 GB `russian_chunks.npy` — риск синхронизации/конфликтов.
12. **Эпоха и «скачок» PPL:** −60% на границе эпох выглядит подозрительно (пересмотр тех же окон); для честной оценки нужен held-out, а не val-срез того же потока.
13. **Eval OOM** (`training_phase2.log`): eval на 500 чанках при 89M не влезал в 2GB вместе с обучением.

---

## 6. Переносимость в EVA-CLM

### 6.1. Что в EVA-CLM уже есть (проверено по коду)

`core/vsa_utils.py`: `dct_basis` (9), `zeckendorf_codes` (19), `fib_sigmoid_init` (39), `sparse_block_codes` (128), `twin_free_codes` (64), **`vsa_prefix_scan`** (168, chunked log-cumsum — численно устойчивее Hillis-Steele); `core/bind.py`: `BottleneckBind` (65) — билинейный bind со сдвигами golden/fibonacci и cascade; `core/mirror.py`: `GroupedCognitiveMirror` (80) с disagreement expert-vs-collective (596); `core/memory_bank.py`: L1/L2/L3 банки (slot-attention, **не** ковариация); `core/phantom.py`; `core/tau_api.py`. В `core/block.py:200` прямо зафиксировано: «Memory: VSA vector superposition (not covariance matrix)».

### 6.2. Топ переносимого

1. **Ковариационная память (матричное состояние).** Это главный дефицит EVA-CLM: векторная VSA-суперпозиция теряет кросс-размерные взаимодействия (это признаёт и `test_hyper_membind.py:233-243`). Перенос: `M_h[t]=d·M[t-1]+i·k⊗k`, чтение `q·M·W_read`, состояние `H×r×r`. Совместимо с `vsa_prefix_scan` EVA-CLM (он умеет только вектор; для матриц нужен вариант по последним двум осям — у FCP это `parallel_prefix_scan`, `core.py:387-412`). Рекомендуется как **опциональная голова** рядом с векторной суперпозицией, с ablation.
2. **Random features + first moment µ.** Дешёвое расширение ёмкости памяти: `R_frozen (D,p)` + обучаемые `p→r`; µ сохраняет знак/направление (`core.py:591-603, 681-692`). Портируется почти без изменений (p=64, r=16).
3. **ZeckendorfReadout (дерево вероятностей).** У EVA-CLM есть коды, но голова — `SigmoidCodedHead`. Дерево даёт точную нормированную `P(i|h)=Π_k P(b_k|h,state)`; центроиды `(K,2,2,D)`, невалидный слот занулён (фактически 3K·D параметров), O(K) на таргет — компактная альтернатива softmax-голове для ablation.
4. **Мультимасштабные головы (frozen τ_h).** `τ=[3,12,49,200]` на голову — простой способ задать горизонты памяти; может конфликтовать с τ-полем EVA-CLM (там τ — калибруемая величина), поэтому только как эксперимент.
5. **Cognitive mirror через std по головам.** В EVA-CLM уже есть mirror (expert-vs-collective), так что это скорее идея «второго сигнала рассогласования» — интегрировать в существующий `GroupedCognitiveMirror`, а не дублировать.
6. **Session/long-term memory (SVD resid).** `LONGTERM_MEMORY.md`: `M_final → SVD → (p,q)` → адаптер `(h·A)·Bᵀ`; запись без backprop. Хорошо ложится на банки/фантомы EVA-CLM, но требует проверки на масштабе (в FCP не реализовано).
7. **Weight factorization `V_shared + W_code`** — только как опция: доказано, что обученную dense-модель так сжать нельзя (rel err 0.996), но обучение с нуля работает (forward/backward), даёт ~82× сжатие. В EVA-CLM уже есть compression/коды; ценность — в K-bottleneck MLP и едином базисе.

### 6.3. Что переносить НЕЛЬЗЯ

| Механизм | Причина |
|---|---|
| Пост-фактум факторизация весов через V (DCT/random) | rel err ≈ 0.996 на всех весах (`outputs/_cube_test.txt`, `_fib_factor_test.txt`) — математически это сжатие произвольной матрицы в K=8..24 столбцов, информации не хватает |
| Гиперсеть-генерация весов | `test_hyper_membind.py:270-296` — decoder дороже хранения; выигрыш только при L>coord_dim, т.е. не для 24 слоёв |
| `recurrent_scan` (Python-цикл по L) | O(L) шагов, медленно; параллельный скан всегда быстрее |
| `LDBlock` sigmoid-gates / `adaptive_gain` / `global_context` | legacy λ_d; EVA-CLM ушёл дальше (softmax-free, но с другими механизмами) |
| Hillis-Steele `parallel_prefix_scan` как есть | по памяти O(L·H·r²) и хуже численно, чем chunked log-cumsum в `vsa_utils.py:168` EVA-CLM; переносить только при доказанной необходимости |
| CE-only рецепт FCP | привёл к bind-коллапсу; у EVA-CLM есть aux-лоссы — не откатываться |

### 6.4. Риски при переносе

1. **Взрыв `exp(i)`-гейта** (`b_i=1.0`): в fp16 даёт NaN; вводить clamp/log-space и следить за нормой `i_gate`.
2. **Дрейф масштаба M:** нет нормировки ковариации; при длинных сессиях `M` растёт — либо decay, либо делить на след/число токенов.
3. **Состояние-кортеж `(cov, mu, conv)`:** нужна версия/датакласс; иначе чекпоинты и resume сломаются (грабля FCP).
4. **Память обучения:** скан материализует `M_all (B,L,H,r,r)`; при B=2, L=2048, H=4, r=16 это ~4.2M чисел (~17 MB fp32) на слой — считать бюджет.
5. **Конфликт с τ-полем:** frozen τ_h и калибруемые τ EVA-CLM могут дать несогласованную динамику; начинать с одной головы/слоя.
6. **Слабое использование памяти при коротких L:** без aux-лосса «read usage» ковариация повторит судьбу FCP (W_read ↓). Предусмотреть диагностику/регуляризацию (напр., loss на предсказание по памяти или gate-штраф).
7. **Zeckendorf-кламп таргетов:** при vocab > max_representable — тихая порча лосса; проверять явно.
8. **Кросс-проектный артефакт:** в `checkpoints_micro/micro_step_5000.pt` лежит чекпоинт проекта **WideBind** (`core.config.WideBindConfig`, pickled-модель, 15.07.2026) — не путать с FCP-чекпоинтами; для загрузки требует модуль `core`.

---

## 7. Приложение A. Карта экспериментов (все 28)

| Файл | Что проверяет | Итог/статус |
|---|---|---|
| `test_bind_gate.py` | sigmoid-гейт vs bilinear bind (FCF) | прототип BindGate; сравнение PPL в конце (410-419); результаты не сохранены |
| `test_cov_gate.py` | bind + последовательная ковариационная память | прототип CovLDBlock; `W_read` zero-init (161) |
| `test_pscan_gate.py` | + параллельный скан (Hillis-Steele) | прототип PScan; 65-96 скан |
| `test_membind.py` | multi-head covariance (H=4,r=8) + feedback | reference-прототип; помечен «superseded» |
| `test_cov_enhanced.py` | first moment µ + random features | forward/backward OK; EnhancedMemBindBlock |
| `test_ablation.py` | µ / RF / оба / ничего | ablation-скрипт |
| `test_mirror.py` | 5 вариантов mirror | тест 100 шагов; не сохранено |
| `test_multiscale.py` | τ=[3,12,49,200] vs фикс. | тест 150 шагов; не сохранено |
| `test_dct_sliding.py` | DCT + λ-sliding на step30000 | результаты в шапке (6-10); только sanity |
| `test_dimensionality.py` | обучаемость D∈{512..2048} | скрипт; чекпоинт `phase2_best.pt` отсутствует |
| `test_fresh_start.py` | обучение с нуля + анализ гейтов | скрипт; результат не сохранён |
| `test_recurrent_scan.py` | parallel vs recurrent, экстраполяция L=128..2048 | скрипт |
| `test_generate.py`, `experiment_extrapolation.py` | генерация/экстраполяция | ссылаются на отсутствующие чекпоинты |
| `test_wide_membind.py` | чисто поэлементный VSA-wide | scan==sequential (1e-5); param-count |
| `test_wide_comparison.py` | Pure VSA vs Hybrid K=4/8/16 vs Grouped G=4/8/16 vs current | 4 теста (param, VRAM, grad flow, 100 шагов оптимизации); обучения нет |
| `test_hyper_membind.py` | гиперсеть-факторизация | математический вердикт: невыгодно (270-296) |
| `analyze_math.py` | K=8, DCT, λ-sliding, λ-tied heads, learnable δ | рекомендации (265-327) |
| `analyze_architecture.py` | HTML-отчёт + сравнение архитектур | таблица 215-222 (94%/3.7×) |
| `analyze_model.py`, `analyze_scaling.py`, `analyze_context.py`, `explore_*.py` | анализ масштабирования/контекста/спектров | аналитика, не обучение |
| `experiment_gate_analysis.py`, `experiment_importance.py`, `experiment_spectral_gates.py` | гейты/важность токенов/спектральные гейты | ссылаются на `model_step25000.pt` (нет) |
| `benchmark.py` | tok/s, латентность, VRAM | скрипт |
| `distill_zeckendorf.py` | дистилляция readout из обученной модели | скрипт; чекпоинт отсутствует |
| `experiment_tokenizer_compare.py` | Qwen3-146K vs Mistral-131K | скрипт |

## 8. Приложение B. Ключевые числа

- 89.1M (λ_d/MemBind, D=896, L=24, K=8): 30K шагов, train PPL 138.1, eval PPL 308.4, grad 0.7–0.9, hidden σ=1.00, τ≈7.5–7.8, `W_v` +289%, `W_read` −42% (`REPORT_30K.md`, `TRAINING_LOG.md`).
- λ_d 12 слоёв: `model_start.pt` step=37500, epoch=3, best_ppl=131.5.
- Factorized flat K=24: 1.6M параметров, 20.5K шагов, 7.9 ч, заявлен PPL=1.8 (не подтверждён чекпоинтом).
- fib_v: 2.0M параметров, V_shared 896×24 (DCT), V_emb 896×620, E_code 50000×24, ~530 tok/s (заявлено; прогон не подтверждён).
- Ковариационное состояние: H=4, r=16 → 1024 числа/слой (≈4KB fp32); 96KB на 24 слоя.
- Корпус: 11.9B токенов (39 жанров), ACTION 549M, DETECT 881M; seq_len=128, eff. batch 32.

---

*Отчёт составлен по коду на состоянии рабочего дерева (грязное, поверх `d8120d3`). Все ссылки `file:line` — на момент разбора; в незакоммиченных правках `core.py`/`train_large.py` номера строк могут отличаться от HEAD.*
