# EXT_WideBind_REPORT — полный разбор проекта WideBind (EVA v3)

> Внешний аудит кода проекта `C:\Users\black\OneDrive\Desktop\WideBind` (git HEAD `0397325`,
> 825 коммитов). Отчёт составлен по коду, логам, чекпоинтам и внутренним докам.
> Аудитория: команда EVA-CLM. Цель: извлечь корреляционные/связывающие методы и грабли.
> Дата аудита: 2026-09-19. Читать вместе с `ARCHITECTURE_REPORT.md` проекта WideBind (там
> авто-каталог классов) — здесь акцент на механике и на том, что реально работает.

---

## 0. Краткое резюме (TL;DR)

**Что это.** WideBind (внутреннее имя EVA, «Единая Вычислительная Архитектура») — исследовательская
языковая архитектура без полноценного attention: VSA-суперпозиция, нормосохраняющий bind
(спиральные вращения + hybrid HRR/elementwise), «когнитивное зеркало» (ансамбль 32 экспертов с
K-пространством и медленными сигналами), иерархическая память, τ-поле, гибридная голова
σ×(1+softmax). Обучается с нуля на русском корпусе, fp32, на Colab T4.

**Статус.** Активный R&D, не продукт. Каноническая «большая» конфигурация заявлена как
D=2560, 24 слоя, G=32, vocab=65536, **191 372 273 параметра**. Исторически задокументированы
val_loss 10.60 (step 2796), 9.674 (step 6757), 9.194 (step 10019), 8.769 (step 15844, старый
прогон). В репозитории лежит `checkpoints/best.pt` (1.93 ГБ) **свежего рестарта: step 233,
best_val 25.9** — то есть последний локальный запуск в самом начале (в фазе рампы). Корпус
обучения (`wb/token_stream_*.bin`) в этой копии **отсутствует** — есть только русский
токенизатор (~8 МБ ×2). Обучение из этой копии не воспроизводится без корпуса.

**Доказано кодом/тестами:** VSA prefix-scan (точный, с state), combinadic-коды (6 из 32,
C(32,6)=906192, детерминированы), DCT-ортогональность, forward/loss/градиенты, диапазоны
AdaptiveController, параметры зеркала; в логах — снижение val, работа memory bank, mat-рамп.
**Не доказано:** связная генерация, бенчмарки, перенос на 3B, реальная польза intent-моста
(на step 987 `w_intent` живёт только в L0, L12/L23 = 0), `mod_scale_mlp` больше не участвует
в forward (см. §5), прожектор (Projector) мёртв после замены CollectiveConceptLayer.

**Главный вывод для EVA-CLM.** Из WideBind переносимы: (1) единый гибридный гейт
`σ(l)·(1+softmax(l/τ))` как замена attention/гейтов; (2) VSA-bind как нормосохраняющий
примитив (спираль + hybrid HRR/elementwise с α-рампом) + когерентность фаз |Z|;
(3) chunked log-space prefix-scan и «медленные сигналы» зеркала (pred/temp/smooth/sym/help
с EMA-нормировкой и декорреляцией). Из обучения переносимы LossBalancer (PCGrad-проекция aux
на CE) и AGC. Не переносить: maturation-рамп как «готовность», manifold-beams в текущем
виде (Python-циклы), Projector, amp_optim, gradalign-костыль.

---

## 1. Инвентаризация и состояние

### 1.1. Дерево (без мусора)

```
WideBind/
├── README.md                  — 1619 строк, главный документ (по коду, но местами устарел)
├── train.bat                  — СЛОМАН: вызывает python train.py из корня (файла нет; он в scripts/)
├── requirements.txt           — torch>=2.5, numpy, tokenizers, tqdm, pymorphy3
├── core/                      — вся модель (~40 модулей)
│   ├── config.py              — WideBindConfig (150+ полей, λ_d-иерархия в __post_init__)
│   ├── lambda_utils.py        — λ_d, Fibonacci, производные пороги
│   ├── tau_config.py          — TauConfig: единое τ-поле
│   ├── embedding.py           — PartitionedEmbedding, SigmoidCodedHead, CognitiveCodedHead, RoPE
│   ├── vsa_utils.py           — combinadic-коды, DCT, Zeckendorf, chunked prefix-scan
│   ├── bind.py                — Bottleneck/Spiral/TrajectorySpiral/TrajectoryManifoldBind
│   ├── mirror.py              — GroupedCognitiveMirror (32 эксперта), BridgeGLU
│   ├── block.py               — WideBindBlock (VSA-память, conv, spectral, VPM)
│   ├── stack.py               — WideBindStack (1925 стр): intent bus, bridge, reasoning, losses, param_groups, AdaptiveController, MirrorLRScheduler
│   ├── concept_layer.py       — UnifiedConceptLayer (τ-driven концепты)
│   ├── memory_bank.py         — StreamingMemoryBank L1/L2/L3
│   ├── bridge.py              — SemanticBridge (per-layer, self-supervised)
│   ├── maturation.py          — MaturationController (time/τ-рамп)
│   ├── adaptation.py          — LossBalancer, DepthController, LRController, FailureDetector, AGC
│   ├── reasoning.py           — ReasoningMemory/Gate (адаптивная глубина)
│   ├── adaptive_gate.py       — hybrid_gate + AdaptiveGate (σ×softmax)
│   ├── layer_bridge_gate.py   — SpectrumGate/LayerBridgeGate (дубль spectrum_gate.py)
│   ├── compression.py         — FCF_CPR (8-бит квантизация чекпоинтов)
│   ├── migrate.py             — миграция state_dict (W_out +K, freq_scale)
│   ├── live_inference.py      — LiveInference + MirrorMonitor
│   ├── projector.py, word_num.py, curriculum.py, amp_optim.py, training_guard.py, model.py
│   └── spectrum_gate.py       — SpectrumGate (НЕ используется; дубль layer_bridge_gate)
├── data/                      — build_streams_eos.py, sentence_builder.py, stencil.json.gz, supervision.jsonl (6.5MB)
├── wb/russian_tokenizer/      — tokenizer.json, tokenizer_v65536.json (по ~8 МБ; корпуса НЕТ)
├── checkpoints/best.pt        — 1.93 ГБ, step 233 (свежий рестарт), best_val 25.90
├── docs/                      — ARCHITECTURE_REPORT.md (1197 стр), ARCHITECTURE_COMPARISON.md, JOURNAL, AGENT_BRIEF, AGENT_BOARD
├── scripts/                   — train.py, analyze.py (1614 стр), generate.py, smart_controller.py,
│                                 proj_read.py (мёртв), diag_*, test_* (~20 тестовых/диагностических)
├── tests/                     — test_model.py (56 тестов), test_infer.py
├── logs/mini_smf.log          — мини-прогон 12.6M (доказательство, что CE падает)
├── archive/                   — старый код (вне git): fcf_cpr, projector_net, zeckendorf_readout, скрипты
└── analyze_*.log/txt          — дампы analyze.py по чекпоинтам (числа ниже)
```

### 1.2. Метрики, зафиксированные в репозитории

| Источник | Step | val_loss | Комментарий |
|---|---|---|---|
| `checkpoints/best.pt` (загружен) | 233 | **25.897** | свежий рестарт (D=2560/24L, memory_bank=True, matur=True), val выше случайного ln(65536)=11.09 — рампа |
| `analyze_best.log` (143.01M) | 2796 | 10.604 | конфиг без memory bank; `pred` aux мёртв, cos_sim(diversity,CE) взорван |
| README §20 | 6757 | 9.674 | canonical run: D=2560/24L/191.4M, maturation 0.240, L1/L2=13468 |
| ARCHITECTURE_JOURNAL | 10019 | 9.194 | per-layer maturation + hybrid head |
| ARCHITECTURE_JOURNAL | 15844 | **8.769** | старый прогон, цель была val<8.5 (не достигнута в текущем) |
| `checkpoints/anomaly_track.json` | 15844 | — | последняя запись трекера (hp_max 6.34, predmse 3.58, gate_min 0.30) |
| `logs/mini_smf.log` | 0..? | CE 9.28 | 12.6M/16L/D=512/G=8 — smoke-прогон |

Связной генерации текста ни в одном логе не зафиксировано (в `archive/gen_*.txt` — обрывки).
Единственное прямое «доказательство генерации» — мини-прогон: `cat fox .`, `cat bird sat .`
(`scripts/mini_result_*.txt`) на 400 шагов, т.е. игрушечный уровень.

### 1.3. Git-гигиена

- 825 коммитов; HEAD — `docs: update README...`.
- В рабочем дереве: удалён корневой `ARCHITECTURE_REPORT.md` и `scripts/analyze_ckpt.py`
  (не закоммичено), появились untracked `docs/AGENT_BOARD.md`, `docs/ARCHITECTURE_REPORT.md`.
- `.gitignore` исключает `checkpoints/`, `*.pt`, `logs/`, `archive/`, `wb/russian_tokenizer/*.json`,
  `analyze_*.txt` — то есть чекпоинты/логи/корпус не в git.
- `train.bat` устарел (нет `--bottleneck` в текущем argparse; путь к `train.py` неверный).

---

## 2. Архитектура: поток данных и компоненты

### 2.1. Поток (canonical D=2560, 24 слоя)

```
tokens (B,L)
  → PartitionedEmbedding: combinadic code (6/32) → σ(2·M·code) → z⊗basis → RoPE(θ=1e6)
  → [если memory_bank] StreamingMemoryBank: read L1+L2+L3 на каждом токене (write на EOS=2)
  → n_layers × WideBindBlock:
        h = RMSNorm(h)
        h += depthwise causal Conv1d(kernel=48)                       (branch A)
        h += bind(h) [TrajectorySpiralBind]                           (branch B)
        VSA-память: 4 τ-шкалы, chunked prefix-scan, dual read+mu      (branch C)
        h += mirror(h, mem_all, global_state, intent, salience, mat)  (branch D)
        [UnifiedConceptLayer после L0: read+write концептов]
        [Variable Precision: ExactSequenceMemory по гейту]            (branch E)
        h += DCT-spectral(λ_k)                                        (branch F)
        h += GroupedMLP(SwiGLU, mirror-гейт mlp_mod)                  (branch G)
  → final RMSNorm
  → [если explicit_reasoning] адаптивная петля рассуждений (до 8 шагов)
  → [если triad_reason и не train] ре-циркуляция при conf<0.5
  → SigmoidCodedHead: 32 бита (6 активных) → hybrid_gate → logits
```

Интерфейс: `out, state, global_state, (reasoning_buffer, reasoning_count) = model(h, state, gs, step, tokens)`.
`state` — список per-layer кортежей `(mem_state, mu_state, conv_state, traj_state, pen)`.
Обучение — teacher forcing с переносом state между батчами; на EOS состояние умножается на 0.1
(`train.py:509-513`), на смене потока — сброс.

### 2.2. Конфигурация (канонический конфиг из README §20 и загруженного best.pt)

| Параметр | Значение |
|---|---|
| D / n_layers / G / bind_K / mirror_k | 2560 / 24 / 32 / 32 / 32 (staircase k=8/16/32) |
| vocab / code_dim / code_sparsity | 65536 / 32 / 6 |
| head_mode | `sigmoid_coded`, head_normalize=True |
| bind_twist_mode | `trajectory_spiral`, S=4, traj_dims=3, ocular=tied, gate=True |
| variable_precision / precision_threshold | True / 0.3 |
| explicit_reasoning / reasoning_adaptive / max_steps | True / True / 8 |
| intent_bridge / bridge_glu / bridge_conn | True / True / 0.1 |
| memory_bank (L1=3, L2=16…32, L3=8) | True |
| maturation_enabled / T0 / T_delay / delta | True / 8000 / 8000 / 4000 |
| softmax_free | True («режим Б») |
| use_amp | False (AMP ломал обучение) |
| seq_len (обучение) | 128 (но curriculum до 512) |

### 2.3. Ключевые формулы (по коду)

**Коды токенов** (`core/vsa_utils.py:57-84`): ровно S=6 единиц из K=32, combinadic-перестановка
с seed=42; гарантия C(K,S) ≥ vocab. Эмбеддинг (`core/embedding.py:101-111`):
`z = σ(2·(codes @ M))`, M — K×K ортогональная; `h = (z ⊗ basis)` (внешнее произведение
K сегментов × d), затем RoPE.

**Bind (spiral, `core/bind.py:365-454`):**
```
hp = RMSNorm(W_proj·h + b)                         # D→K
θ  = exp(W_freq)·freq_scale·hp + W_phase           # freq_scale init 2π
u' = u·cosθ − v·sinθ ;  v' = u·sinθ + v·cosθ       # комплексное вращение
out_s = [Re(u'⊙v'), Im(u'⊙v')]                     # 2K
hybrid = α·HRR(u_re, v_re) + (1−α)·(u_re⊙v_re)     # HRR = круговая свёртка (index-gather)
prod_re += 0.1·hybrid
|Z|² = ((Σcos)² + (Σsin)²)/(S·nd)²                 # когерентность фаз
out = [Σ_s out_s ; |Z|] @ W_out                    # W_out: (nd·2K+K, D)
```
α: 0.3→0.7 за 5000 шагов (`_hybrid_alpha`, bind.py:348-353).

**VSA-память** (`core/block.py:300-390`):
```
decay = clamp(exp(−1/τ_s)·σ(h·w_d + b_d), 0.01, 1.0)
i_gate = softplus(h·w_i + b_i + γ·‖pred_err‖) · (1 + bind_coh_gate·mean|Z|)
mem_t = decay⊙mem_{t−1} + i_gate⊙h
mem_read = mem_all·w_q + mem_leaf·w_q_leaf + mem_all·w_q_ctx + mu_read·w_mu_mem
```
Скан — в log-пространстве, chunk=32, double precision (`block.py:18-53, 356-388`);
4 шкалы τ передаются из stack как `exp(cumsum(softplus(_vsa_log_param)))+1`.

**Зеркало** (`core/mirror.py`):
```
hp = h_g @ W_proj (per-expert K-space) ⊛ pos_id (биполярный код)
signals: temp=hp−centroid; global=hp−gs; pred=(hp−α⊙hp_prev)/‖hp‖;
         smooth=hp−causal_conv(hp); sym=(hp·w_u)⊙(hp_prev·w_v); help=cross-expert recall
s_norm = s/EMA_rms(s); w=σ(log_w); delta=RMSNorm(Σ w_i·s_i)+tanh_bias
gate_logits = ⟨|pred_err|,w_gate⟩ + b_gate + gate_bias + ⟨delta,w_delta_gate⟩
              + grad_mod + dvar_mod + intent_gate + salience·w_sal
              + disagreement·w_contra + contra_expert − 0.5·meta_instability
expert_gate = σ(gate_logits); mirror = (tanh(linear)+skip·linear)·exp(log_scale)·adapt·expert_gate
mlp_mod = hybrid_gate(usefulness_logits)·(1 + β·(2·glu−1)·maturity)   # BridgeGLU
```
usefulness — соревновательный предиктор (MLP k→k→1), софт-порог по медиане, температура
3.0→0.3 по мере обучения (`mirror.py:578-597`).

**Голова** (`core/embedding.py:167-251`): `z_k=⟨h_k,readout_k⟩`; `T=exp(log_temp)`;
`zt = z/T + bit_bias (+bus_bias)`; `u,base = log(σ(zt)·(1+softmax(zt/T)))`; логиты
`u @ codesᵀ + base + token_bias`; нормализация logsumexp. `log_probs_for_target` считает
бернуллиеву сумму по активным/неактивным битам — CE без матрицы d×vocab.

---

## 3. КОРРЕЛЯЦИОННЫЕ МЕТОДЫ (главный раздел)

Ниже — все найденные механизмы корреляции/связывания/сопоставления, сгруппированные по
подсистемам. Для каждого: формула, ссылка `file:line`, назначение и статус.

### 3.1. Bind-семейство: связывание, сохраняющее норму

| Метод | Формула | Файл:строка | Назначение |
|---|---|---|---|
| Golden-shift cross | `left ⊙ roll(right, shift_s)`, shift = floor(s·K/φ) mod K | `core/bind.py:49-58,142-143` | билинейное скрещивание с золотым углом; разные «окуляры» |
| Bottleneck shift | `Σ_s (u_s⊙roll(v_s)) @ W_out[s]` | `bind.py:159-176` | legacy-режим (S=4, multi-ocular, ранг ≤ S·K) |
| Fibonacci cascade | `a_n = normalize(cross(a_{n−1}·w_u, a_{n−2}·w_v))·‖a_1‖`; смесь `mix = σ(mix_logit)·(1+softmax(mix_logit/e^logτ))` | `bind.py:179-214` | моночлены Фибоначчи + hybrid-смесь |
| SpiralBind | комплексное вращение: `u'=u·cosθ−v·sinθ`, `v'=u·sinθ+v·cosθ`; `θ=exp(W_freq)·hp+W_phase` | `bind.py:257-277` | фазовая интерференция каналов |
| TrajectorySpiralBind | то же + `traj = [hp_t, hp_{t−1}, hp_{t−2}]` (EMA 0.9/0.1) | `bind.py:365-454` | bind видит «траекторию», а не только текущий hp |
| **Hybrid HRR** | `α·HRR(u,v) + (1−α)·(u⊙v)`, HRR = круговая свёртка через `einsum(a, b[circ_conv_idx])`; α 0.3→0.7 за 5000 шагов | `bind.py:355-363`; таблицы индексов `bind.py:341-346` | компромисс: HRR (голографическая суперпозиция) + покомпонентное произведение |
| **Когерентность фаз \|Z\|** | `|Z|² = ((Σ_{s,d}cosθ)²+(Σ_{s,d}sinθ)²)/(S·nd)²`; добавляется K каналов в `W_out`; `i_gate *= (1+bind_coh_gate·mean|Z|)` | `bind.py:411-418, 338`; `block.py:309-311` | «точки скрещивания»: фазы всех спиралей совпали → резонанс усиливает запись в VSA |
| Manifold beams | переход `T=unbind(hp_t,hp_{t−1})`; кластеризация в лучи (greedy cosine > 0.5); затухание `θ=1/(1+len(zeck(age)))`; чтение `w=σ(sim·gain)·(1+softmax(sim·gain/τ))·decay` | `bind.py:536-648` | локальный контекст через «лучи» переходов вместо полной памяти |
| Zeckendorf-уровни | `_zeckendorf_levels`: веса `1/F_k`, нормированные | `bind.py:287-300` | иерархический распад (в FCF-наследии) |

**Оценка.** Нормосохраняющая часть (вращения + HRR) — математически здоровая: bind/unbind
изометрии, обратимые, что и декларируется как «лицензия на глубину». Но:
- README утверждает, что HRR считается «через FFT» (README §5.2) — **в коде FFT нет**:
  используется gather по таблицам `_circ_conv_idx` (память O(K²)=1024 на слой) и einsum.
  Док-дрифт.
- Hybrid HRR имеет `prod_re += 0.1·hybrid` — фиксированный коэффициент 0.1 (не обучаемый),
  т.е. HRR-ветвь входит в 10 раз слабее фазовой. Это скрытая константа вопреки манифесту
  «никаких магических чисел».
- `freq_scale` init 2π с claim «~0.5% позиций в резонансе»; миграция старых чекпоинтов
  ставит 1.0 (численно эквивалентно старому поведению) — аккуратный паттерн миграции.
- Manifold: `n_beams = ceil(sqrt(buffer))` (1024→32), буферы **неперсистентные**
  (`persistent=False`) — при resume манифолд пуст; пересборка — Python-цикл с `.item()`
  и `.tolist()` по всем переходам (`bind.py:559-601`) — на CPU, медленно, не масштабируется.
  `beam_age` заполняется только для новых центров; затухание Zeckendorf для старых лучей
  может считаться неверно после нескольких rebuild. Метод экспериментальный, дефолт
  `traj_manifold=False`.

### 3.2. Единый гибридный гейт σ×softmax — «attention-замена»

**Формула** (`core/adaptive_gate.py:26-72`, ядро — строка 58):
```
gate = sigmoid(logits) · (1 + softmax(logits / τ))          # normalize=True → L1
log-режим: u = log gate − log(1−gate);  base = Σ log(1−gate)
```
τ∈[0.1,10] (клэмп); при `τ→∞` — чистый sigmoid (независимые ворота, diversity), при `τ→0` —
почти winner-take-all (softmax). Использование:
- голова: `_su()` (`embedding.py:210-212`) — вероятностная интерпретация битов;
- cascade/mix в bind (`bind.py:188-196`);
- чтение лучей манифолда (`bind.py:626-638`);
- attention memory bank (`memory_bank.py:53-84`);
- концепт-слой (`concept_layer.py:292-299`);
- MLP-гейт зеркала `AdaptiveGate` (`mirror.py:283, 607-619`);
- LayerBridgeGate/SpectrumGate (`layer_bridge_gate.py:41-48`);
- ReasoningMemory: **чистый sigmoid**, без softmax (`reasoning.py:72`).

**Почему это ценно.** Это единственная сквозная «формула корреляции» проекта: она заменяет
softmax-конкуренцию на сумму «независимая активация + относительный фокус», сохраняя
ненулевой потенциал у проигравших (важно для EVA-CLM с её σ×softmax-головой). Градиент:
sigmoid-ветвь даёт включать/выключать признаки, softmax-ветвь — перераспределять вес.

**Грабли.** Формула продублирована в трёх местах: `adaptive_gate.hybrid_gate`,
`spectrum_gate.SpectrumGate` (не используется) и `layer_bridge_gate.SpectrumGate`
(используется). Дрейф гарантирован. В `hybrid_gate(log=True)` значение `gate` клэмпится в
[1e-7, 1−1e-7], и `base` — сумма логов; при K=32 это даёт численно устойчивый CE, но
`log_probs_for_target` при `normalize=True` всё равно вызывает полный forward (двойной проход
по голове) — потеря скорости.

### 3.3. VSA-память: корреляция через суперпозицию и prefix-scan

- **Decay-корреляция**: `decay = clamp(exp(−1/τ_s)·σ(h·w_d+b_d), 0.01, 1.0)` — содержание
  модулирует скорость забывания (`block.py:303-325`).
- **Surprisal-запись**: `i_gate = softplus(h·w_i+b_i+γ·‖pred_err‖)` — пишем то, что зеркало
  не смогло предсказать (`block.py:304-311`). γ инициализируется от τ слоя
  (`block.py:184-189`).
- **Prefix-scan**: ассоциативный скан `mem_t = a_t·mem_{t−1}+b_t` в log-пространстве,
  chunk=32, fp64 внутри (`vsa_utils.py:90-131`, `block.py:18-53`); двухуровневое склеивание
  чанков. Это точная (не приближённая) параллельная реализация рекуррентности — тесты
  `test_vsa_scan_exact/with_state/batched` подтверждают.
- **Dual read**: `mem_all·w_q + mem_leaf·w_q_leaf + mem_all·w_q_ctx` — внутричанковое
  покрытие + кросс-чанковый контекст (`block.py:370-371`).
- **Первый момент** `mu` — тот же скан с входом `mem_input·w_k_mu`, даёт «центр масс» записей
  (`block.py:374-388`).
- **Динамическая модуляция чтения/записи по K-пространству экспертов**:
  `read_mod = σ(hp_g @ w_q_dyn/√k)`, `write_mod = σ(hp_g @ w_i_dyn/√k)` (`block.py:327-338,
  415-425`) — корреляция «что эксперт понял» ↔ «что память отдаёт/берёт».
- **PrecisionGate + ExactSequenceMemory** (`block.py:56-87, 439-449`): локальный точный кэш с
  **сигмоид-нормированным средним** `A=σ(qkᵀ/√k); A=A/ΣA` (LaCUR-стиль, без softmax) —
  выпуклая комбинация, «проводимость, не конкуренция». Включается порогом
  `σ(precision_gate(h)).mean() > 0.3`.

### 3.4. Когнитивное зеркало: корреляции сигналов и экспертов

**5 медленных сигналов** (`mirror.py:330-338, 340-389, 514-523`):
1. `temp = (hp − mc_k)·w_temp` — отклонение от центроида памяти;
2. `global = (hp − gs_k)·w_global` — отклонение от кросс-слойного состояния;
3. `pred = (hp − α_diag⊙hp_prev)/‖hp‖ · scale` — ошибка само-предсказания во времени;
   `α_diag` — **обучаемая per-K-dim константа времени**, init из τ-лестницы [2,200],
   обновляется по residual-variance: `α_target=σ(2.2−log(rel_var))` (`mirror.py:365-388`);
4. `smooth = hp − causal_conv3(hp)` — локальная временная корреляция (ядро — диагональ
   x_{t−1}, `mirror.py:117-122`);
5. `sym = (hp·w_sym_u)⊙(hp_prev·w_sym_v)` — билинейная временная корреляция (аналог
   автокорреляции второго порядка).

**EMA-нормировка сигналов** (`mirror.py:525-538`): `s/EMA_rms(s)` — соизмеримость перед
суммированием; **декорреляция** (`mirror.py:543-558`): `Σ_{i<j} cos²(s_i,s_j)` (по центрированным
векторам) — штраф за коррелированные сигналы, отдаётся в aux `decorr`.

**Cross-expert recall (private memory read)** (`mirror.py:408-432`):
```
uncert = σ(|pred_error|);  q = hp·uncert
attn   = σ(q @ keysᵀ/√k)                 # независимые ворота, не softmax
help_k = (attn @ keys) · σ(w_help) · trust
disagreement = ‖hp − help_k‖/‖hp‖;  contra = σ(disagreement − 1);  trust = 1−contra
```
Это «корреляция экспертов»: уверенные состояния экспертов образуют банк `_private_mem`
(G×k), чтение — сигмоид-внимание по косинусной близости.

**Expert Knowledge Graph** (`mirror.py:434-467`), обновляется EMA 0.99:
```
concept_sim   = normalized(_private_mem) @ normalized(_private_mem)ᵀ     # (G,G)
behavior_div  = 1 − cos(hp_avg_g, hp_avg_g')                             # (G,G)
trust_matrix  = EMA(attn.mean(B,L))                                      # (G,G)
contra_graph  = concept_sim ⊙ behavior_div                               # (G,G)
dominance     = trust_matrix.sum(0);  isolation = 1 − trust_matrix.sum(1)/G
```
Все G×G-матрицы (32×32) — «социограмма» экспертов; используются для gate-bonus, записи
private memory (social pressure) и диагностики `debug_mind()`. Это самая развитая
корреляционная структура проекта, но масштаб O(G²) и почти вся — `no_grad` EMA.

**Запись private memory** (`mirror.py:469-512`): `conf = σ(−|pred_err|)`,
`conf_plastic = conf·(1−contra)·social_pressure`, мягкая конкуренция `conf^0.5·G/Σ`,
адаптивный decay 0.990–0.999, clamp ±10; гейт — maturation (иначе legacy step-delay).

**SMF-гейт** (`mirror.py:642-651`): `α = σ(⟨[ĥ, T̂], W_α⟩ + b_α)` — последовательно-параллельное
слияние (исправление доминирования L0; `log_skip_alpha` теперь 0).

### 3.5. UnifiedConceptLayer: концепты через косинус

`core/concept_layer.py` (заменил CollectiveConceptLayer + L3):
- **Запись**: `shared = mean_G(hp)`; `q = write_q_proj(shared)`; `sims = cos(q_n, concept_keys)`;
  обновление при `mat≥0.3` и `conf≥median`; `α = σ(−log τ_update)∈[0.001,0.5]`;
  рождение: `novelty = σ(τ_nov·(1−best_sim)) > 0.5` и `conf ≥ σ(τ_birth·0.5)`; вытеснение по
  `utility = confidence·count` (строки 132-240).
- **Чтение**: `scores = cos(q_n, keys)·τ_read`; `attn = σ(scores)·(1+softmax(scores/τ_read))`
  (гибрид), нормировка; `read = attn @ vals`, `out_proj` (строки 280-303).
- **Гейты выхода**: uncertainty `u=σ(κ(pen−τ_u))`, contradiction
  `c=σ(τ_c·cos(read,h))`; `out = read·u·c·σ(read_scale)` (строки 305-325).
- **Maturity**: `mat = σ((1/cv − λ_d)·τ_mat)`, `cv = std(resvar)/|EMA(resvar)|` (строки 104-124).

Градиент течёт через запись (нет `@torch.no_grad` на путях с `hp`), но фактически запись
мутирует `.data` буферов, т.е. градиент идёт только через `write_q_proj/write_v_proj` и
read-путь. Это честный, но компромиссный вариант «дифференцируемой памяти».

### 3.6. StreamingMemoryBank L1/L2/L3: attention-замена по слотам

`core/memory_bank.py`:
- **Единая attention-функция** (`_memory_attention`, строки 53-84):
  `scores = q·kᵀ/√d · temp; attn = σ(scores)·(1+softmax(scores/temp)); ·age_decay; /Σ`.
  Это тот же гибрид σ×softmax, что и везде.
- **L1** (3 слота): ring buffer эмбеддингов предложений, age decay `exp(−0.01·age)`.
- **L2** (16/32 слота): обучаемые keys/vals; запись — `novelty = σ(novelty_gate(emb))`,
  ключи `F.normalize`, значения `F.normalize·σ(val_log_scale)`; ring buffer с приоритетом
  «consumed» слотов.
- **L3** (8 концептов): кластеризация L2-ключей по косинусу: `best_sim > birth_threshold`
  (0.85 в config, 0.7 в прогоне) → обновление running mean с momentum 0.1; иначе рождение
  (при confidence > порога) или вытеснение; рождение помечает L2-слот consumed.
- **Fusion**: `[h, L1, L2, L3] → MLP → h + tanh(log_scale)·fused`; выходной слой fusion
  zero-init (no-op на старте).
- **Грабли**: чтение/запись — вложенные Python-циклы по батчу и токенам
  (`memory_bank.py:568-595`) → медленно; запись только на токене EOS=2 (в данных с
  `_eos.bin`); `val_norm` LayerNorm добавлен после инцидента с нормой std≈579 (README §16).

### 3.7. SemanticBridge и Intent Bus: кросс-слойная корреляция

**SemanticBridge** (`core/bridge.py`):
- probe `s_l = probe(h_l)` (общий для всех слоёв, 2-Linear+GELU, bridge_dim=256);
- инъекция: `h_l += tanh(stream_log_scale)·maturity·stream_proj(Σ w_n·bridge_stream[n])`,
  соседи i−1/i/i+1, веса — сигмоид-среднее (`stream_log_weights`, выпуклая комбинация);
- поток `bridge_stream (n_layers, 256)` — EMA 0.9/0.1, переносится между forward;
- **self-supervised loss**: `mean_l(1 − cos(s_l[:, :-1], emb_proj(embed(y[:,1:]))))`
  + `0.1·mean_offdiag_cos(layer_means)` (штраф за коллапс слоёв, `bridge.py:141-181`);
- readiness: `sat = 1 − EMA(loss)/init`, `readiness = σ((sat−r0)/rs) − base`.

**Intent Bus** (`core/stack.py:278-406`):
- `intent_probe: D → G·K_max`; salience-вес от головы (1-шаг задержки,
  `compute_salience = ‖σ(logits)‖`, нормировка на среднее);
- EMA-интеграция `α_l = 1 − exp(−τ_l/τ_min)` (глубокие — почти carried);
- шина: `Bus_i = (Σ_{j≤i} fresh_j + Σ_{j>i} carried_j)/n` — «поток, а не склад»;
- в зеркале: `intent_gate = ⟨hp−ik, w_intent⟩ + b_intent` (zero-init), плюс
  `gate_logits += salience·w_sal` (zero-init);
- фаза-2: `bus_head_proj` (zero-init) добавляет `bus_bias` в readout головы
  (`stack.py:857-869`).

**Оценка.** Идея «сжатого gist-потока» вместо KV-кэша — центральная новизна проекта.
Но доказательств пользы мало: в `analyze_987_bridge.txt` `w_intent` ненулевой только в L0
(0.054), L12/L23 = 0; `bus_head_proj` норм 0.156; salience-распределение почти равномерное
(H/Hmax не приведён, но max/min ≈ 2.9). В `ARCHITECTURE_COMPARISON.md` сами авторы
признают: «Phase-1 of the bridge is empirically not yet fully awake». Для EVA-CLM это
идея-кандидат, но не проверенный механизм.

### 3.8. Голова и эмбеддинг: корреляция через коды

- **Weight tying**: `SigmoidCodedHead.readout = embed.basis` (K×d) — энкодер и декодер
  используют один базис (`stack.py:28`, `embedding.py:177-181`).
- **Код-корреляция**: `logits = u @ codesᵀ` — скалярное произведение битовых логитов на
  бинарные коды; фактически корреляция «предсказание битов ↔ код токена».
- **token_bias** — частотный приор (|vocab| скаляров); **bit_bias** — маргинальные приоры
  битов; **bus_bias** — кросс-слойный stencil.
- **RoPE** (`embedding.py:12-44`): стандартный, θ=1e6, линейный scaling; D/2 частот.
- **CognitiveCodedHead** (не используется, `head_mode='cognitive_coded'`): добавляет
  `resonance = 1+w_energy·tanh(−‖h_g−readout‖²)` — корреляция «энергия совпадения»,
  `prior` от private_mem через `W_q_prior/W_k_prior`, social bias от dominance/contradiction,
  `token_shift_embed`. Интересный, но необученный прототип.

### 3.9. Корреляции в лоссах и градиентах

- **LossBalancer align** (`core/adaptation.py:492-544`): `g_final = g_CE + scale·g_aux`,
  `scale = min(cos(g_CE,g_aux)·cap, 1)·‖g_CE‖/(‖g_aux‖+ε)`; aux не может доминировать.
  Это корреляционный метод обучения (PCGrad-подобный), полностью переносимый.
- **cos_sim(diversity, CE)** — диагностика (`scripts/analyze.py:745-801`): косинус между
  градиентами diversity и CE. **Баг**: печатает 1e10+ («взорванные числа»), сам AGENT_BRIEF
  признаёт «игнорируй» (`docs/AGENT_BRIEF.md:37`). Причина — некорректная сборка карт
  градиентов (повторный обход `model.parameters()`), а не реальная корреляция.
- **gradalign** (`scripts/train.py:392-418`): `g_target = ‖∂CE/∂mlp_out‖` по экспертам,
  нормировка на max; `L = MSE(mlp_mod_norm, g_target_norm)` — «gradient-reactive
  governance loss». Оказался нужен только для разморозки `mod_scale_mlp`, который затем
  вообще перестал использоваться (см. §5.2). Костыль.
- **Декорреляции**: diversity (ковариация групповых выходов MLP → I,
  `stack.py:930-943`), decorr сигналов зеркала, orth/nuclear на W_proj bind
  (`stack.py:945-976`), balance (HHI по usage экспертов, `stack.py:917-928`),
  gate_repulse (−энтропия), alpha_novelty (var α).
- **Phase-ratio** (`train.py:448-472`): `ratio = ‖g_mirror‖/‖g_base‖` по слою, EMA+std,
  `mir_s = σ((ratio−EMA)/std)∈[0.2,2]` — корреляционная нормировка градиентов зеркала.
- **MirrorLRScheduler** (`stack.py:1701-1991`): `mult = (var_mult·alpha_mult·gate_mult)^{1/3}·mag_factor`;
  каждый множитель — `1/(текущее/EMA)` в клэмпе [0.5,2]; boost >1 только при
  `_val_improving`; `loss_lr_factor` — 0.5 при регрессе >5% от best, восстановление 1/200
  за шаг в плато.

### 3.10. Прочие корреляционные механизмы

- **Word arithmetic** (`core/word_num.py`): буква→простое число, слово→произведение;
  `morph_sim = 2·log НОД/(log N₁+log N₂)` — морфологическая корреляция слов без обучения.
  Переносимо как дешёвый морфологический сигнал (для русской морфологии), не требует
  чекпоинта.
- **CurriculumTracker** (`core/curriculum.py`): `p_i ∝ exp(L_i/τ)` — сэмплирование трудных
  чанков. Не используется в train.py.
- **FCF_CPR** (`core/compression.py`): удаление детерминированных буферов (`codes`, `V_dct`),
  свёртка uniform-скалярных `b_i/b_d` и 8-битная (per-tensor/per-channel) квантизация
  весов. Заявлено 8–16×; в generate.py — авто-декомпрессия.
- **analyze.py correlation-дампы**: `_grad_cos` (багованный), `run_bridge` cross-layer cosine
  потока intent (`analyze.py:806-920`), `run_metacog` (concept_sim/behavior_div/trust/
  meta_private_mem по слоям, `analyze.py:1129-1182`).
- **SmartController** (`scripts/smart_controller.py`): инференс-надстройка, где параметры
  сэмплинга — функции корреляций состояния: энтропия H, повторяемость n-грамм, `trust_max`
  из `debug_mind()`, tau-«личность» модели; режимы exploit/explore/confused/reason/
  recover-rep/recover-collapse. Идея переносима (адаптивный декодинг от внутренних
  корреляций), но эвристики (пороги 0.45/1.35, 0.82/0.96, окна 8/16) — магические числа.

---

## 4. Обучение и оптимизация

### 4.1. Потери

Основная: CE через `log_probs_for_target` (бернуллиев битовый CE, без softmax-матрицы),
маска PAD(0) и (опционально) EOS(2), surprisal-взвешивание (`stack.py:813-827`).

Aux-словарь (`compute_losses`, `stack.py:848-1266`; веса — только через LossBalancer):

| Имя | Формула/смысл | Статус |
|---|---|---|
| `pred` | MSE(pred_k, hp.detach()) — учит α-диагональ | **МЁРТВ** (в `_pred_cache` оба тензора detached, `stack.py:890-894`; analyze подтверждает `requires_grad=False`) |
| `gate_l1` | mean(expert_gate) — разреженность | жив |
| `reinforce` | MSE(usefulness, gate.detach()) | жив |
| `balance` | нормированный HHI usage экспертов | жив |
| `diversity` | MSE(cov(group_out), I) | жив, но шумный (аннилится `aux_anneal_tau`) |
| `nuc` | 𝔼‖W_proj·v‖ — ядерная норма bind | жив, вес 1e-5 |
| `orth` | ‖ŴᵀŴ−I‖² | по умолчанию 0 (VRAM) |
| `w_m2v` | иерархия вклада памяти по τ (тянет `_tau_dev`) | жив, target detached |
| `intent_tau` | иерархия α intent-потока | жив, target detached |
| `branch` | равенство log-var conv/bind/mirror | по умолчанию 0, аннилится |
| `div` | −(var_G σ(ls) + √(d/G)·var_k σ(ls)) | вес 10 (!) |
| `gate_repulse` | −энтропия usage | вес 0.3 |
| `alpha_novelty` | −var(α) по экспертам | вес 0.05 |
| `decorr` | Σcos² сигналов зеркала | жив |
| `signal_ent` | энтропия весов сигналов | жив |
| `ls_reg` | mean(max(0,ls−2.3)²) | жив |
| `mem_tau_reg` | prior + запрет инверсии L1>L3 | жив |
| `bridge_conn` | 1−cos (см. §3.7) | жив |
| `tau_dev_reg` | 0.01·mean(dev²) | центрирование τ |
| `lbg_diversity` | недостаток энтропии гейтов слоёв | жив |
| `gradalign` | MSE(mlp_mod, ‖∂CE/∂mlp_out‖) | только train.py, костыль |

Ключевое архитектурное решение: **в ядре нет весов aux** — все веса выводит LossBalancer.
Это правильный паттерн, который стоит перенять (но см. грабли: `pred` мёртв и всё равно
попадает в aux).

### 4.2. Контроллеры (core/adaptation.py)

- **DepthController**: разморозка блоков при плато val (`slope > −k·σ`), `init_active_layers=8`,
  +4 слоя за раз, максимум 24.
- **LRController** = warmup + MirrorLRScheduler + `rewind()` (ре-warmup при восстановлении).
- **FailureDetector**: SPC 3σ + относительный порог 15% + 3 последовательных нарушения →
  rollback на best.pt, свежий Adam, rewind LR.
- **GradientClipper (AGC)**: `clip iff ‖g‖ > 0.1·‖θ‖`; пропускает ‖θ‖<1e-3 (защита zero-init
  параметров — иначе мост/τ_dev зануляются).
- **build_optimizer**: AdamW (0.9,0.95), LLRD `0.9^layer` × роль-множители λ_d^p
  (embed/vsa λ⁻²≈0.296, mlp/bind λ⁻¹≈0.544, default 1.0, mirror/gate λ¹≈1.839), wd только
  на 2D.
- **Пороговое хозяйство**: `LambdaConfig(d=3)` выводит exploration_threshold=0.296,
  differentiation_threshold=0.087 и все диапазоны AdaptiveController.

### 4.3. Расписания

- warmup ~1000 шагов, `warmup_steps` переопределяется из λ_d;
- seq curriculum: 64 (до 32k шагов) → 256 (до 96k) → 512, длина выбирается
  `seq_pool[step % len]` (`train.py:348-365`);
- reasoning ramp `s = 1−exp(−t/1000)`;
- maturation time-ramp по τ_norm;
- aux annealing `branch/diversity` за 5000 шагов;
- multi-stream sampling: случайный поток и случайная позиция, state сбрасывается на границе
  документа.

---

## 5. Что доказано, что мертво

### 5.1. Доказано (тесты/логи/код)

- `tests/test_model.py` (56 тестов, 1 lastfailed — `test_inference`): точность VSA-скана,
  combinadic-свойства (ровно 6 бит, все биты используются, детерминизм, prefix-stable),
  ортогональность DCT, forward/loss/градиенты стека, диапазоны AdaptiveController,
  τ-иерархия λ_d, live_inference state.
- В логах: снижение val_loss на длинных прогонах (11.07→8.77), работа memory bank
  (L1=13468, L2=13468/7195 consumed, L3=8/5991 born), рост maturation 0.088→0.240,
  CE(random)≈25 при head с 65536 классов, `mod_mlp` дрейф 0.667→0.658 под gradalign
  (AGENT_BRIEF), `alpha_diag` ~0.90 (per-K τ init), `gate_ema`/`log_scale` специализируются.
- Mini-прогон 12.6M: CE 9.28 (16L, D=512) — падение CE подтверждено.
- Hybrid head: в журнале заявлено «entropy 10.5→0.58» и val 9.194→9.129 (гибридная голова),
  но эксперимент `hybrid_head_experiment.py` — отдельный скрипт hot-swap, не в основном коде
  (в `SigmoidCodedHead._su` уже используется hybrid_gate — т.е. фактически включено).

### 5.2. Мертво/недоделано (следы в коде)

1. **`mod_scale_mlp` — вестигиальный параметр.** С коммита `0c95cf5` MLP-гейт делает
   `AdaptiveGate(usefulness_logits)` (`mirror.py:619`) или BridgeGLU (`mirror.py:602-617`).
   `mod_scale_mlp` в forward **не участвует** (только `mod_scale_mem` на строке 620).
   При этом `scripts/analyze.py:268-308` (WAKE-детектор!), `train.py:529`, `light_analyze.py`,
   `diag_mlp_core.py`, `diag_grad_mlp.py`, `train_min.py` читают и логируют его как «главный
   гейт»; AGENT_BRIEF описывает его «заморозку» как проблему. То есть значительная часть
   диагностики и выводов — про мёртвый параметр.
2. **Projector/Прожектор мёртв.** `core/projector.py` и `scripts/proj_read.py` читают
   `layer.collective._write_event/_concept_id`, но `WideBindBlock.collective = None`
   (`block.py:227`), а `UnifiedConceptLayer` не выставляет эти атрибуты. `stack.projector_signals`
   всегда возвращает `(None, None)`. README §5.10 описывает несуществующую функциональность.
3. **`core/spectrum_gate.py`** — полный дубль SpectrumGate, нигде не импортируется.
   Используется версия из `layer_bridge_gate.py`.
4. **`core/amp_optim.py` (AmpAdam)** — для удалённой кодечной головы (SignedAmpCodec),
   не импортируется.
5. **`core/curriculum.py`** — CurriculumTracker не используется в train.py.
6. **`core/training_guard.py`, `core/model.py`** — shim-заглушки.
7. **`pred` aux-потеря мёртвая** (дет detached кэши) — «учит α», но градиента не даёт;
   analyze прямо печатает «МЁРТВЫЙ».
8. **`w_pred_scale_legacy`** — мёртвый параметр (std=0, «не получает градиента»,
   checkpoint_journal).
9. **Manifold-beams**: неперсистентные буферы, Python-циклы, `beam_age` частичный —
   экспериментальный тупик в текущем виде.
10. **`scale_w`** (VSA per-scale weights) почти не двигается (0.25→0.2500) из-за LR λ⁻² и
    слабого градиента; `alpha_diag` застыл ~0.90 (не дошёл до нижней границы 0.87).
11. **Intent Bridge**: `w_intent` живёт только в L0 (analyze_987); заявленный «фазовый
    прогрев» так и не завершён.
12. **`bridge_conn` collapse**: в ARCHITECTURE_JOURNAL зафиксирован «bridge_conn collapse
    на uniform maturation» (исправлялся per-layer maturation).
13. **Док-дрифт**: README утверждает (а) HRR через FFT — в коде gather; (б) maturation =
    `max(time_ramp, bridge_readiness)` — в коде `step_gate` **чистый time-ramp**, readiness
    только диагностика (`maturation.py:91-122`); (в) «без attention» — но
    `variable_precision=True` включает `ExactSequenceMemory` с softmax-ветвью (при
    `softmax_free=False`) / сигмоид-нормировкой (при True), что сами авторы честно называют
    локальным attention в ARCHITECTURE_COMPARISON.md; (г) T0=20000 в README §19.1 против
    T0=8000 в config/логах; (д) Projector (см. п.2).
14. **Мёртвые скрипты**: `train.bat` (неверный путь/аргументы), `scripts/archive/`,
    `archive/` (старый код, вне git), `scripts/analyze_ckpt.py` удалён из рабочего дерева.
15. **AGENT_BOARD** фиксирует незакрытый спор: middle-first maturation (L4-L8≈0.92) vs
    deep-first; предложена inference-аблация ΔCE по слоям и Spearman с mat/depthgrad — **не
    выполнена**; выводы о «хабе» отозваны как недоказанные. Это прямое признание, что
    приоритизация слоёв не откалибрована.

### 5.3. Известные баги (подтверждённые)

- `analyze.py` `cos_sim(diversity, CE)` печатает 1e10+ (AGENT_BRIEF: «игнорируй»).
- `scripts/train.py` использует глобальный `args` внутри `train()` (строки 283, 552) —
  функция не работает при импорте/вызове не из `__main__`.
- `optimizer.load_state_dict` позиционный (ломался при добавлении `freq_scale/W_out+K`) —
  починен через `param_names` в train.py, но старые чекпоинты требуют name-matching.
- AMP ломал «кризис согласования» → `use_amp=False` навсегда.
- L2 vals взрывались (std≈579) → `val_norm` LayerNorm при чтении (коммит 6740434).
- Echo chamber: запись private memory в случайные K-space состояния → фикс
  `pm_write_delay=5000` + maturation.
- `core/bind.py` содержит mojibake в docstring-ах TrajectoryManifoldBind (кодировка
  UTF-8↔cp1251) — косметика, но мешает чтению.

---

## 6. Переносимость в EVA-CLM

### 6.1. Переносить (высокая ценность)

1. **Единый гибридный гейт `σ(l)·(1+softmax(l/τ))`** (`adaptive_gate.py:26-72`) — как
   универсальный примитив: голова, память, концепты, гейты. У EVA-CLM уже есть σ×softmax
   голова — стоит унифицировать формулу и log-режим (`u, base`) для CE без softmax-матрицы.
   Обязательно свести к ОДНОЙ реализации (в WideBind их три).
2. **VSA-bind как изометрия**: комплексные вращения `θ=exp(W_freq)·freq_scale·hp+W_phase` +
   `u'=u cosθ−v sinθ` и hybrid HRR/elementwise с α-рампом (`bind.py:355-363, 385-418`).
   Для EVA-CLM («Bind: связывание, сохраняющее норму») это прямой референс: вращения
   нормосохраняющие, HRR даёт голографическую суперпозицию, α-расписание стабилизирует
   переход от elementwise к свёртке.
3. **Когерентность фаз |Z|** (`bind.py:411-418`) — дешёвая (сумма cos/sin) мера «резонанса»
   каналов; может служить сигналом важности/записи и диагностикой (аналог «фантомов»).
4. **Chunked log-space prefix-scan** (`vsa_utils.py:90-131`, `block.py:18-53`) — точная
   параллельная реализация линейной рекуррентности с fp64-защитой; переносится как есть
   для любых VSA/SSM-подобных состояний.
5. **Медленные сигналы зеркала** (`mirror.py:330-389, 514-558`): temp/pred/smooth/sym/help
   + EMA-нормировка + штраф декорреляции. Особенно **per-dim α (learned time constants) с
   residual-variance адаптацией** (`α_target=σ(2.2−log rel_var)`) — компактный и
   принципиальный механизм мультимасштабности.
6. **Cross-expert recall с сигмоид-вниманием и contradiction gate**
   (`mirror.py:408-432`) — переносимо в любую банковую память EVA-CLM (проверка согласия
   экспертов, trust=1−contra).
7. **LossBalancer (PCGrad-проекция aux на CE)** (`adaptation.py:492-544`) — переносится
   целиком, снижает нужду в ручных весах лоссов.
8. **AGC** (`adaptation.py:387-412`) с пропуском near-zero параметров — полезен для
   zero-init мостов.
9. **Combinadic sparse block codes** (`vsa_utils.py:57-84`) — детерминированные, ровно S из K,
   prefix-stable (добавление vocab не меняет старые коды). Для EVA-CLM block codes —
   готовая реализация с тестами.
10. **FCF_CPR** (`compression.py`) — 8-битная квантизация чекпоинтов с удалением
    детерминированных буферов; практично для хранения многих прогонов.
11. **τ-поле TauConfig** (`tau_config.py`) — если EVA-CLM ещё не имеет: один параметр
    `_tau_dev` (cumsum-монотонная лестница) + вывод всех порогов (maturation delays, gate
    temperatures, EMA α, LLRD). Но осторожно: WideBind-специфичные привязки (matur_T0,
    intent_alpha) — не догма.
12. **Word arithmetic** (`word_num.py`) — дешёвая морфологическая корреляция для русского
    (произведение простых, `morph_sim` через НОД) — можно использовать как признак/регуляризатор
    без обучения.

### 6.2. Не переносить / только как идею

- **MaturationController как «готовность»** — в коде это чистый time-ramp; readiness не
  влияет на гейт; доки противоречат коду; AGENT_BOARD показывает нерешённую калибровку
  порядка слоёв. Брать только концепцию «единый гейт для всех wake-up ветвей», не формулу.
- **Intent Bus / SemanticBridge в текущем виде** — не доказан (w_intent активен в 1 слое из
  24), добавляет 2M параметров и сложный streaming-state; для EVA-CLM лучше
  экспериментировать отдельно, а не тащить целиком.
- **TrajectoryManifoldBind** — Python-циклы, неперсистентные буферы, спорный beam_age.
- **gradalign** — лечил симптом уже мёртвого `mod_scale_mlp`; не нужен.
- **`mod_scale_mlp`, `w_pred_scale_legacy`, Projector, amp_optim, spectrum_gate.py,
  curriculum.py** — мёртвый код.
- **Вся диагностика WAKE >0.75** — измеряет мёртвый параметр, выводы о «пробуждении» по ней
  недостоверны.
- **Архитектурный стиль «стек всего»**: 24 слоя × (bind + VSA + conv + DCT + mirror + MLP +
  VPM + bridge) — вычислительно тяжело, гибрид не минимален; сами авторы признают
  (ARCHITECTURE_COMPARISON §5). Для EVA-CLM лучше изолированные механизмы.

### 6.3. Риски и грабли (сводка)

1. **Док-дрифт** — README описывает то, чего нет в коде (FFT-HRR, readiness-гейт,
   Projector, «без attention»). Любые заимствования проверять по коду, а не по README.
2. **Вестигиальные параметры в диагностике** — половина логов/анализов измеряет мёртвые
   величины (`mod_scale_mlp`), что породило ложные выводы (AGENT_BRIEF).
3. **Мёртвый aux (`pred`)** — дет detached кэши; паттерн «aux без градиента» незаметен, т.к.
   LossBalancer просто добавит 0.
4. **Багованный cos_sim-анализ** — не доверять «взорванным» корреляциям.
5. **AMP** — архитектура нестабильна в fp16; весь контур принудительно fp32.
6. **Python-циклы** в memory bank и manifold — узкое место; для масштаба переписывать
   векторно (chunked-сканы у них, наоборот, хороший пример).
7. **Чекпоинт-совместимость** — любое добавление параметров требует миграции
   (`migrate.py`), а оптимизатор — name-matching; иначе тихая порча resume (у них был
   инцидент с positional restore).
8. **«Магические» константы вопреки манифесту**: `+0.1·hybrid`, `γ=0.5`, пороги
   SmartController, `traj_gain=0.05`, `temp_write=0.5`, `_pm_coh_gate_std=0.02` — при
   переносе инвентаризировать.
9. **Статус обучения**: последний локальный best.pt — step 233/val 25.9 (хуже случайного),
   исторические рекорды 8.77–9.67 — на других прогонах; корпуса в копии нет. Не строить
   выводов о работоспособности на этом снапшоте.
10. **Дублирование кода** (hybrid_gate ×3, SpectrumGate ×2) — при переносе брать одну
    реализацию.

---

## 7. Приложение A. Карта «метод → файл:строка»

| Метод | Файл:строка |
|---|---|
| hybrid_gate σ×(1+softmax) | `core/adaptive_gate.py:26-72` |
| AdaptiveGate | `core/adaptive_gate.py:75-130` |
| combinadic-коды | `core/vsa_utils.py:57-84` |
| DCT-базис | `core/vsa_utils.py:9-15` |
| Zeckendorf-коды | `core/vsa_utils.py:19-35` |
| VSA prefix-scan | `core/vsa_utils.py:90-131`; `core/block.py:18-53` |
| PartitionedEmbedding | `core/embedding.py:64-111` |
| SigmoidCodedHead | `core/embedding.py:167-251` |
| CognitiveCodedHead (резонанс) | `core/embedding.py:254-375` |
| Golden shifts | `core/bind.py:49-58` |
| BottleneckBind | `core/bind.py:62-214` |
| SpiralBind | `core/bind.py:235-277` |
| TrajectorySpiralBind (HRR+coherence) | `core/bind.py:303-454` |
| TrajectoryManifoldBind (лучи) | `core/bind.py:457-648` |
| VSA-память (decay/i_gate/dual read/mu) | `core/block.py:300-390` |
| PrecisionGate/ExactSequenceMemory | `core/block.py:56-87, 439-449` |
| DCT-spectral | `core/block.py:453-459` |
| GroupedMLP (mirror-гейт) | `core/mlp.py:50-77` |
| Зеркало: сигналы | `core/mirror.py:330-338, 514-523` |
| Зеркало: pred/α | `core/mirror.py:340-389` |
| Зеркало: recall + contradiction | `core/mirror.py:408-432` |
| Зеркало: knowledge graph | `core/mirror.py:434-467` |
| Зеркало: запись private memory | `core/mirror.py:469-512` |
| Зеркало: декорреляция сигналов | `core/mirror.py:543-558` |
| Зеркало: usefulness/BridgeGLU/SMF | `core/mirror.py:578-651` |
| Зеркало: K-space gate | `core/mirror.py:653-724` |
| UnifiedConceptLayer | `core/concept_layer.py:26-353` |
| MemoryBank attention | `core/memory_bank.py:53-84` |
| MemoryBank L1/L2/L3 | `core/memory_bank.py:87-644` |
| SemanticBridge | `core/bridge.py:31-181` |
| Intent Bus | `core/stack.py:278-406` |
| Reasoning (adaptive) | `core/stack.py:669-775`; `core/reasoning.py` |
| Losses | `core/stack.py:848-1266` |
| param_groups (λ-иерархия) | `core/stack.py:1340-1488` |
| AdaptiveController | `core/stack.py:1494-1697` |
| MirrorLRScheduler | `core/stack.py:1701-1991` |
| LossBalancer/AGC/Depth/Failure | `core/adaptation.py:84-544` |
| TauConfig | `core/tau_config.py:42-267` |
| Maturation | `core/maturation.py:47-149` |
| FCF_CPR | `core/compression.py:13-361` |
| Word arithmetic | `core/word_num.py:41-92` |
| Анализ (bridge/metacog/grad) | `scripts/analyze.py:745-920, 1129-1182` |

## 8. Приложение B. Топ-числа для быстрой ориентации

- Параметров (canonical): **191 372 273** (README §20); в загруженном best.pt — 352 256 017
  элементов state_dict (включая буферы `codes` 65536×32 и позиционные коды).
- Контекст: `state` на слой = O(D·(S+1)) векторов, KV-кэша нет.
- Тесты: 56 (1 lastfailed).
- Коммитов: 825.
- Размер best.pt: 1.93 ГБ; токенизатор: 2×7.99 МБ; supervision.jsonl: 6.5 МБ.
- Корпус `wb/token_stream_*.bin`: **отсутствует**.

---

*Отчёт подготовлен по коду репозитория; все формулы сверены с реализацией. Места расхождения
документации и кода отмечены явно (§5.2). При переносе в EVA-CLM рекомендуется начинать с
пунктов §6.1.1–6.1.4 (гибридный гейт, bind, |Z|, prefix-scan) — они компактны, покрыты
тестами и не несут мёртвого груза WideBind.*
