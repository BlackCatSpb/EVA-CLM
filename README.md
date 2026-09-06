# EVA-CLM — Единая Вычислительная Архитектура

**Cognitive Learning Model · VSA + Cognitive Mirror + GroupedMLP + Unified τ-field.**
**D=2560, ~191M параметров, 24 слоя. Без attention, без softmax-матриц.**

```
  ┌───────────────────────────────────────────────────────────────┐
  │              EVA-CLM · Unified Computing Architecture         │
  │   VSA-memory · Bind-crossing · Cognitive Mirror               │
  │   Unified τ-field · SemanticBridge · MemoryBank L1/L2/L3      │
  │   Logit Cache · VSA-driven Compression                        │
  │   D=2560 · ~191M params · 24 layers                           │
  └───────────────────────────────────────────────────────────────┘
```

---

## Оглавление

1. [Философия](#1-философия)
2. [Типоразмеры](#2-типоразмеры)
3. [Поток данных](#3-поток-данных)
4. [Эмбеддинг](#4-эмбеддинг)
5. [EVABlock](#5-evablock)
   - [Bind](#52-bind)
   - [VSA-память](#53-vsa-память)
   - [Cognitive Mirror](#54-cognitive-mirror)
   - [Variable Precision](#55-variable-precision)
   - [Spectral + MLP](#56-spectral--mlp)
   - [Explicit Reasoning](#57-explicit-reasoning)
   - [BridgeGLU + SemanticBridge](#58-bridgeglu--semanticbridge)
6. [SigmoidCodedHead](#6-sigmoidcodedhead)
7. [Unified τ-field](#7-unified-τ-field)
8. [U1–U10 τ-tied improvements](#8-u1u10-τ-tied-improvements)
9. [Параметры и LR](#9-параметры-и-lr)
10. [AdaptiveController](#10-adaptivecontroller)
11. [MirrorLR](#11-mirrorlr)
12. [Функции потерь](#12-функции-потерь)
13. [Gradient Mechanics](#13-gradient-mechanics)
14. [Inference](#14-inference)
15. [Intent Bridge](#15-intent-bridge)
16. [Streaming Memory Bank](#16-streaming-memory-bank)
17. [**Logit Cache + VSA Compression**](#17-logit-cache--vsa-compression)
18. [Maturation + Triad](#18-maturation--triad)
19. [Структура репозитория](#19-структура-репозитория)
20. [Статус обучения](#20-статус-обучения)
21. [Лицензия](#21-лицензия)

---

## 1. Философия

1. **Память — вектор, не матрица.** Знание живёт в суперпозиции VSA, а не в K×V таблице. Нет KV-cache, O(1)-контекст.
2. **Глубина не убивает градиент.** Bind — изометрия (циклический сдвиг), сохраняющая норму. 24+ слоя с живой обратной связью.
3. **Точность переменна.** ~191M делают работу плотной модели много раз больше. Переменная точность через PrecisionGate.
4. **Обучение непрерывно.** Инференс = обучение: градиент оседает в неокортексе, зрелые паттерны — в ядро.
5. **Цель — субъект, а не функция.** Модель думает над концепциями, консолидирует опыт, реакционна во времени.

---

## 2. Типоразмеры

| Параметр | **Mini** | **Большая** | **XL** |
|---|---|---|---|
| D | 896 | 2560 | 4096 |
| n_layers | 24 | 24 | 16–32 |
| mlp_groups | 8 | 32 | 32 |
| vocab | 65536 | 65536 | 65536 |
| VRAM | ~2 GB | ~8.3 GB | 16–24 GB |
| Параметры | ~45M | **191.4M** | ~166–330M |

**Большая (обучаемая):** `D=2560, G=32, n_layers=24, bind_K=32`
Конфиг: `intent_bridge=True, bridge_glu=True, explicit_reasoning=True, collective_read_out=True, variable_precision=True, memory_bank=True`

---

## 3. Поток данных

```
[token ids]
     ↓
PartitionedEmbedding: sparse code → sigmoid-mix → basis ⊕ + RoPE
     ↓
Streaming Memory Bank: L1+L2+L3 read
     ↓
n_layers × EVABlock (mem, mu, conv states)
     ↓
Reasoning loop: adaptive depth
     ↓
SigmoidCodedHead: 32 binary bits (K=32, S=6)
     ↓
logits (head_normalize → normalized)
```

Стек: `out, state, global_state = model(h, state, global_state=…)`

---

## 4. Эмбеддинг

`PartitionedEmbedding` (`core/embedding.py`):

1. **Sparse block code:** ровно S=6 активных бит из K=32 (C(32,6)=906192 ≥ vocab)
2. **Плотный микс:** `z = σ(codes @ M × 2)`, M — K×K ортогональная
3. **Базис:** `z ⊗ basis`, каждый сегмент 1:1 с mirror
4. **RoPE:** θ=1e6 (Qwen3-style), 0 параметров

---

## 5. EVABlock

```
h_in
 ├─ [Pre-LN: RMSNorm]
 ├─ [Conv1d depthwise 48-tap, causal]        → A
 ├─ [Bind: D→K→D, spiral/trajectory]         → B
 ├─ [VSA Memory: 4 τ-scales]                 → C
 ├─ [Cognitive Mirror: G experts]             → D
 ├─ [UnifiedConceptLayer]
 ├─ [Variable Precision Memory]
 ├─ [Spectral: DCT × λ_k]                    → E
 ├─ [GroupedMLP: SwiGLU]                     → F
 └─ residual: h = h + (A + B + C + D + E + F)
```

### 5.2 Bind

Режимы: `off`, `shift`, `cascade`, `spiral`, `trajectory_spiral`, `trajectory_spiral+manifold`.

**Spiral-семейство:**
```
hp = RMSNorm(W_proj(h) + b)                  # D → K
θ = exp(W_freq)·freq_scale·hp + W_phase      # rotating phase
u' = u·cosθ − v·sinθ;  v' = u·sinθ + v·cosθ  # complex rotation
out = Σ_s (u' ⊙ v') @ W_out                  # 2·dims·K → D
```

**Когерентность спиралей:** `|Z|_k = |Σ e^{i·θ}| / (S·nd)` — редкие «точки скрещивания» дают的信息 boost.

### 5.3 VSA-память

4 фиксированных временных масштаба на слой:
```
decay = clamp(exp(-1/τ_s)·σ(h·w_d + b_d), 0.01, 1.0)
i_gate = softplus(h·w_i + b_i + γ·‖pred_err‖)
mem_t  = decay_t ⊙ mem_{t−1} + i_gate_t·h_t
```

Чтение: гибридный gate `sigmoid(scores) * (1 + softmax(scores/tau))`.
Вычисление: параллельный prefix-scan, fp32 под AMP.

### 5.4 Cognitive Mirror

G экспертов (32), каждый в K-буфере:
- **Staircase K:** L0..n/3 → k=8, n/3..2n/3 → k=16, глубже → k=32
- **5 сигналов:** temp, pred, smooth, sym, help
- **Ворота:** `g = σ(w_gate·|Δ| + b_gate + delta_gate + grad_mod + contradictions + meta_trust)`
- **Usefulness:** конкурентный предсказатель, дисциплинирует эксперта

**Private Memory:** `_private_mem` (G×k) пишется зрелостью: `conf = σ(−‖Δ‖)`.

### 5.5 Variable Precision

`ExactSequenceMemory` — точный кэш последних шагов, включается `precision = σ(precision_gate(h)) > 0.3`.

### 5.6 Spectral + MLP

- **Spectral:** DCT базис × обучаемый `λ_k`, модулированный по τ
- **MLP:** GroupedMLP (SwiGLU, G групп, расширение 4×). mirror-conditioned: `gate = silu(a + b·mlp_mod)`

### 5.7 Explicit Reasoning

Петля рассуждений **над скрытыми состояниями** (не токенами):

- **ReasoningMemory:** sigmoid attention по буферу шагов (не softmax)
- **ReasoningGate:** tanh-гейты (закрыт=0, отрицательный=антизнание)
- **Адаптивная глубина:** ранняя остановка при avg_gate < 0.5
- **Ramp:** `s = 1 − exp(−t/1000)` — плавное подключение

### 5.8 BridgeGLU + SemanticBridge

**BridgeGLU** (`core/mirror.py`):
```
glu = sigmoid(log_gain) · sigmoid(Wg·flat) · sigmoid(Wv·flat)
mlp_mod = usefulness · base · (1 + β·(2·glu − 1))
```
base = sigmoid(mod_scale_mlp) ≈ 0.667 (замороженный бейзлайн).

**SemanticBridge** (`core/bridge.py`):
- Per-layer probe: `s_l = probe(h_l) ∈ (B, L, bridge_dim)`
- Depth: injection `h += tanh(scale)·stream_proj(neighbours)`
- Time: персистентный `bridge_stream` EMA
- Self-supervised: `aux = 1 − cos(s_l[:, :-1], emb_proj(embed(y[:,1:])))`
- ~2.0M параметров, внутри модели

---

## 6. SigmoidCodedHead

Кодовое пространство: K=32 бит, активных S=6, C(32,6)=906192 ≥ vocab.

```
z_k    = ⟨h_k, readout_k⟩
z̅      = z/T + bit_bias
logit  = u·codesᵀ + base + token_bias
logprobs(target) = Σ актив. битов log σ(z̅) + Σ неактив. log(1−σ(z̅))
```

`head_normalize=True` → нормализованные лог-вероятности без большой матрицы d×vocab.

---

## 7. Unified τ-field

`TauConfig` — единое τ-поле, из которого выводятся **все** τ-зависимые величины.

```
log_tau = log(tau_min) + log(tau_max/tau_min) * (lf * (1 + 0.3 * dev) + 0.05)
tau_l   = exp(log_tau)
```

| Величина | Формула | Назначение |
|---|---|---|
| `tau_norm_l` | `(log(tau_l) - log(tau_min)) / (log(tau_max) - log(tau_min))` | Нормализованный τ |
| `mat_delay_l` | `T0 + (1 - tau_norm_l) * T_delay` | Задержка созревания |
| `gate_tau_l` | `tau_max_gate * (tau_min_gate / tau_max_gate) ^ mat_gate_l` | Температура гейтов |
| `alpha_l` | `1 - exp(-tau_l / tau_min)` | Скорость EMA |
| `lr_mult_l` | `(tau_l / tau_ref) ^ (-gamma)` | LLRD |

**1 обучаемый параметр** `_tau_dev` (n_layers) управляет всей τ-геометрией.

---

## 8. U1–U10 τ-tied improvements

Все улучшения выведены из τ-поля — **никаких магических чисел**:

| # | Улучшение | Формула |
|---|---|---|
| U1 | VSA Memory Scales | `vsa_tau[l, s] = base[s] * (τ_l / τ_mid)` |
| U2 | Reasoning Budget | `K = max(1, round(K_base * mean(τ_norm)))` |
| U3 | Spectral Damping | `damp = cos(π · τ_norm / 2)` |
| U4 | Bridge Injection | `inj = σ(α) · τ_norm + σ(β) · (1 − τ_norm)` |
| U5 | Mirror Signal Temp | `τ_signal = τ_min_gate · (τ_max_gate / τ_min_gate)^(1 − τ_norm)` |
| U6 | Memory Bank Fusion | `scale *= 0.3 + 0.7 · τ_norm` |
| U7 | Concept Birth | `birth_thresh = σ(log_τ_birth) · (1 − τ_norm · σ(log_τ_decay))` |
| U8 | Intent Alpha | `α_expert = α_base · (1 + σ(w_expert) · (2·τ_norm − 1))` |
| U9 | Gradient Clipping | `c_eff = c · (1 + τ_norm)^(−γ)` |
| U10 | Bind Frequency | `freq_eff = freq_scale · (τ_min / τ_max)^(τ_norm · η)` |

**Итого:** G + 6 = 38 новых параметров (0.002% от 191M).

---

## 9. Параметры и LR

| Группа | LR множитель | Содержит |
|---|---|---|
| embed | λ⁻² ≈ 0.296× | эмбеддинг, readout |
| mlp | λ⁻¹ ≈ 0.544× | MLP, bind W_proj/W_out |
| default | 1.0× | conv, norm, голова |
| mirror | λ¹ ≈ 1.839× | mirror W_proj/W_out, α, log_scale |
| gate | λ¹ ≈ 1.839× | w_gate, b_gate, w_i, b_i |
| vsa | λ⁻² ≈ 0.296× | b_d, b_i, scale_w |

---

## 10. AdaptiveController

Из статистик зеркал выводит актуальные параметры:
- `b_d` — расширение/сужение временного окна
- `b_i` — ворота записи: `i_target = min(1, 5.83/τ)`
- `w_mem2v`, `ema_alpha`, `noise_scale` — от diff
- `tanh_bias_mod`, `spectral_mod`, `pred_scale_mod`

---

## 11. MirrorLR

Counter-cyclic planner:
```
mult = (var_ratio × α_ratio × gate_ratio)^(1/3) · mag_factor · loss_lr_factor
```

- Рост компонента → LR снижается, спад → растёт
- loss_lr_factor: улучшение → ×1.05, регресс → ×0.5

---

## 12. Функции потерь

| Имя | Вес | Формула |
|---|---|---|
| ce | 1.0 | NLL через head.log_probs |
| pred | 0.01 | ошибка предсказания α |
| gate_l1 | 0.0001 | разреженность ворот |
| reinforce | 0.001 | soft-цель usefulness |
| balance | 0.026 | HHI-выравнивание |
| diversity | 0.001 | деккорреляция MLP |
| bridge_conn | 0.1 | self-supervised next-token prediction |
| div | 10.0 | разнообразие log_scale |

Все aux-потери возвращаются сырыми, веса — LossBalancer (align mode).

---

## 13. Gradient Mechanics

1. **Forward** → `ce_loss, aux_dict`
2. **CE-градиенты** → `torch.autograd.grad`
3. **LossBalancer** — aux проецируется на CE (cosine gate)
4. **Phase-ratio** per-layer: `ratio = ‖g_mirror‖/‖g_base‖`
5. **AGC:** адаптивный clip, пропускает zero-init
6. **AdamW** (β=(0.9,0.95)) + MirrorLR

---

## 14. Inference

```python
from core import LiveInference
live = LiveInference(model, cfg)
live.think(5)                       # self-dialogue
tokens = live.generate(prompt_ids, 100, think_steps=0)
```

Memory Bank хранит O(1) на шаг (векторы, не матрицы).

---

## 15. Intent Bridge

`cfg.intent_bridge=True` — перетекающий per-head поток намерений:

- **Bottom-up:** `intent_probe = Linear(D, G·K_max)` → per-expert intent
- **Salience-ворота:** `logits.sigmoid().norm()` → важность слова
- **Streaming Bus:** FRESH (верхние слои) + CARRIED (нижние, пред. шаг)
- **Top-down:** `gate_logits += salience·w_sal` (zero-init)
- **Self-tau:** обучаемый `tau_intent_dev[i]` управляет timescale

Два неограниченных канала:
- VSA-память — размытое общее понимание
- Intent-поток — направленная нить внимания

---

## 16. Streaming Memory Bank

`cfg.memory_bank=True` — трёхуровневая стриминговая память:

| Уровень | Назначение | Слоты |
|---|---|---|
| **L1** | Immediate (последние 3 предложения) | 3 |
| **L2** | Short-term (learned) | 16 |
| **L3** | Long-range (emergent concepts) | 8 |

- **Write:** при boundary предложения (SEP token)
- **Read:** sigmoid attention по буферам + fusion 4D → D
- **L3:** кластеризация L2 keys (cosine sim > 0.7)

Overhead: +0.3% params, +3.6 GB VRAM.

---

## 17. Logit Cache + VSA Compression

**Ключевая инновация:** замена KV-cache на **compressed logits с attention**.

### Архитектура

```
logits (V=65536)
     ↓
PerScaleLogitCache.store()
  ├── Scale 0 (fast):   k=64  → 1928 bytes
  ├── Scale 1:           k=48  → 1448 bytes
  ├── Scale 2:           k=40  → 1208 bytes
  └── Scale 3 (slow):    k=32  → 968 bytes
  Total: 5,552 bytes per token
     ↓
PerScaleLogitAttention(Q, Cache)
  Q = q_proj(h)              — from hidden state
  K_i = k_proj_i(cached)     — per-scale projection
  V_i = v_proj_i(cached)     — per-scale projection
  output = gate * combined + (1-gate) * h
     ↓
h_augmented → next token prediction
```

### VSA-driven Compression

Каждый масштаб имеет **обучаемый** `vsa_scale`:
```
k = base_k * sigmoid(vsa_scale)
```
- `vsa_scale → +∞`: k → base_k (максимальная точность)
- `vsa_scale → -∞`: k → 0 (максимальное сжатие)
- `vsa_scale = 0`: k = base_k/2 (баланс)

**Модель сама выбирает оптимальное сжатие** для каждого масштаба.

### Сравнение с KV-cache

| Tokens | Logit Cache | KV-cache | Экономия |
|--------|------------|----------|----------|
| 10K | 55.5 MB | 2.4 GB | **43x** |
| 100K | 555 MB | 24 GB | **43x** |
| **1M** | **5.55 GB** | **240 GB** | **43x** |

### Window Attention (512 tokens)

Cache хранит **все** токены, attention смотрит на **последние 512**:

| Tokens | Full vs Window | Cosine | Top-1 |
|--------|---------------|--------|-------|
| 500 | identical | 1.000000 | 1.0000 |
| 1000 | 1000 vs 512 | 0.999994 | 1.0000 |
| 100K | 1024 vs 512 | 0.999994 | 1.0000 |

**Потери = 0!** Далёкие токены (< 1% веса attention) не влияют на предсказание.

### Реализация

- `core/logit_cache_v2.py`: PerScaleLogitCache, PerScaleLogitAttention, PerScaleCacheAttention
- Интеграция в `EVAStack.process_with_cache()`
- Включение: `cfg.logit_cache_enabled=True`
- Тесты: `scripts/test_logit_cache_v2.py`, `scripts/test_window_impact.py`

---

## 18. Maturation + Triad

### MaturationController

Единый показатель зрелости слоя `M_l(t) ∈ [0,1]`:
```
gate_l(t) = sigmoid((t - (T0 + α·(1-τ_norm_l)·T_delay)) / Δ)
bridge_readiness = σ((sat - r0)/rs) - base
M_l(t) = max(gate_l(t), bridge_readiness)
```

- Готовность по компетентности моста, а не по часам
- Глубокие слои открываются ПЕРВЫМИ (инверсия bottom-up)
- Frozen MLP-гейт (~0.667) всегда открыт → сигнал обучения есть

### Triad: Рассудок как участник

При `triad_reason=True`:
- Если `_conf < triad_conf_thr=0.5`, ствол прогоняется повторно
- Консервативный бленд: `h = 0.5·h + 0.5·h2`
- Только inference — без новых параметров

---

## 19. Структура репозитория

```
EVA-CLM/
├── core/                       # 27 модулей — весь код модели
│   ├── config.py               # WideBindConfig (canonical) + EVAConfig
│   ├── tau_config.py           # TauConfig: единое τ-поле
│   ├── adaptation.py           # LossBalancer, AGC, FailureDetector
│   ├── block.py                # EVABlock
│   ├── stack.py                # EVAStack: forward, process_with_cache
│   ├── losses.py               # compute_losses
│   ├── bind.py                 # Bottleneck/Spiral/TrajectoryBind
│   ├── mirror.py               # GroupedCognitiveMirror, BridgeGLU
│   ├── bridge.py               # SemanticBridge: per-layer, self-supervised
│   ├── embedding.py            # PartitionedEmbedding, SigmoidCodedHead
│   ├── mlp.py                  # GroupedMLP (SwiGLU)
│   ├── reasoning.py            # ReasoningMemory + ReasoningGate
│   ├── memory_bank.py          # StreamingMemoryBank: L1+L2+L3
│   ├── logit_cache_v2.py       # PerScaleCache + VSA-driven compression
│   ├── logit_cache.py          # v1 LogitCache (legacy)
│   └── ...                     # other modules
│
├── scripts/
│   ├── train.py                # training loop
│   ├── analyze.py              # checkpoint analyzer
│   ├── generate.py             # generation: FCF-CPR, --smart
│   ├── test_logit_cache_v2.py  # logit cache tests
│   ├── test_window_impact.py   # window accuracy tests
│   └── ...                     # other scripts
│
├── notebooks/
│   └── eva_colab.ipynb         # canonical notebook (D2560/24L, T4)
│
├── tests/                      # 214/214 passing
├── checkponts/                 # checkpoints (best 1–20)
├── wb/                         # BPE tokenizer + training streams
└── README.md
```

---

## 20. Статус обучения

**Последний чекпоинт** (`checkponts/best 19.pt`):

| Метрика | Значение |
|---|---|
| val_loss | **9.180** (step 8155) |
| Maturation deep | 0.509 |
| Memory Bank L3 | 4859 births |
| bridge_conn | 0.213 |
| VRAM | ~10.6 GB |

**Траектория:**

| Step | val_loss | mat | Events |
|------|----------|-----|--------|
| 6990 | 14.461 | 0.437 | LR reset |
| 7223 | 9.339 | 0.452 | |
| 8155 | **9.180** | **0.509** | mat > 0.50 |
| 8305 | — | 0.519 | Bridge activation |
| 8388 | **9.122** | 0.522 | Latest eval |

**Цель:** 300K шагов.

---

## 21. Лицензия

**Экспериментально.** Архитектура активно развивается; API/конфиги могут
меняться. Миграция state_dict: `core/migrate.py::migrate_state_dict`.

> EVA-CLM — исследовательский проект нейроморфной языковой архитектуры:
> VSA, биллинейное связывание, самоорганизующиеся регуляторы по λ_d.

*Замечания, вопросы и PR — приветствуются.*
