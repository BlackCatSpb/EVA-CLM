# ARCHITECTURE REPORT — EVA (EVA-CLM)

Версия от 2026-09-05. Все формулы, классы, функции и дефолты извлечены напрямую из исходного кода.

---

## Содержание

1. [Обзор архитектуры](#1-обзор-архитектуры)
2. [Конфигурация EVAConfig](#2-конфигурация-evaconfig)
3. [Структура файлов](#3-структура-файлов)
4. [Embedding (embedding.py)](#4-embedding)
5. [Bind (bind.py)](#5-bind)
6. [Mirror (mirror.py)](#6-mirror)
7. [MLP (mlp.py)](#7-mlp)
8. [Block (block.py)](#8-block)
9. [Stack (stack.py) — прямой проход](#9-stack)
10. [Gate: AdaptiveGate, SpectrumGate, LayerBridgeGate](#10-gates)
11. [TauConfig (tau_config.py)](#11-tauconfig)
12. [Maturation (maturation.py)](#12-maturation)
13. [Bridge (bridge.py)](#13-bridge)
14. [Memory Bank (memory_bank.py)](#14-memory-bank)
15. [Reasoning (reasoning.py)](#15-reasoning)
16. [Concept Layer (concept_layer.py)](#16-concept-layer)
17. [VSA Utilities (vsa_utils.py)](#17-vsa-utilities)
18. [Adaptation (adaptation.py)](#18-adaptation)
19. [Lambda Utils (lambda_utils.py)](#19-lambda-utils)
20. [Scripts: analyze.py](#20-scripts)
21. [Текущие результаты обучения](#21-обучение)
22. [Известные проблемы и исправления](#22-проблемы)
23. [Полный отчёт по классам и функциям](#23-полный-отчёт)

---

## 1. Обзор архитектуры

EVA (EVA-CLM) — языковая модель на основе архитектуры Mixture-of-Experts (MoE) с когнитивным зеркалом. Ключевые особенности:

- **Кодирование токенов**: Sparse Block Codes (K=32, S=6) с внешним произведением для создания плотных эмбеддингов
- **Bind**: Траекторный спиральный bind (TrajectorySpiralBind) с комплексным скрещиванием
- **Mirror**: Ансамбль из G=32 экспертов, каждый в своём d=80-мерном подпространстве (D=2560)
- **Conv**: Depthwise свёртка с kernel=48
- **Spectral**: DCT-базис с индивидуальными масштабами частот
- **MLP**: Grouped SwiGLU с expand=4 (32 группы, по 80 dims)
- **Head**: SigmoidCodedHead — гибридный sigmoid-softmax гейтинг
- **Maturation**: Единый временной/τ-рамп для управления пробуждением слоёв (T0=8000)
- **Bridge**: Семантический мост (самосупервизия через предсказание следующего токена)
- **Intent Bridge**: Нисходяще-восходящая передача «намерения» экспертам
- **Reasoning**: Цепочка рассуждений (chain-of-thought) с адаптивной глубиной (max_steps=8)
- **Concept Layer**: Unified Concept Layer (S=8 концептов)
- **Memory Bank**: Иерархическая L1+L2+L3 память (softmax-free)
- **Private Memory**: Приватная память экспертов (per-expert K-space states)
- **Variable Precision Memory**: Precision-gated exact sequence memory

---

## 2. Конфигурация EVAConfig

Текущая production конфигурация (фактические значения из чекпоинтов):

### Основные размерности
| Параметр | Значение | Описание |
|----------|----------|----------|
| D | 2560 | Ширина модели |
| n_layers | 24 | Число слоёв |
| bind_K | 32 | Bottleneck K для bind |
| vocab | 65536 | Размер словаря |
| seq_len | 128 | Длина последовательности |
| mlp_groups | 32 | Число групп в MLP (G) |
| mlp_expand | 4 | Множитель расширения в MLP |
| d (выч.) | 80 | D // G = 2560 // 32 |
| Total params | 189.47M | Общее число параметров |

### Обучение
| Параметр | Значение | Описание |
|----------|----------|----------|
| lr | 3e-4 | Learning rate |
| warmup_steps | 1000 | Шаги разогрева |
| weight_decay | 0.01 | L2 регуляризация |
| grad_clip | 0.5 | Обрезка градиентов |
| max_steps | 300000 | Максимальное число шагов |
| batch_size | 1 | Размер батча |
| seq | 128 | Длина последовательности |

### Mirror
| Параметр | Значение | Описание |
|----------|----------|----------|
| mirror_k | 32 | K-space размерность (staircase: 8/16/32) |
| private_mem | True | Приватная память экспертов |
| expert_asymmetry | True | Асимметричная инициализация |
| meta_trust | True | Рекурсивное мета-доверие |
| intent_bridge | True | Intent Bridge включён |
| bridge_glu | True | BridgeGLU включён |
| bridge_glu_beta | 0.25 | Модуляция BridgeGLU |

### Maturation
| Параметр | Значение | Описание |
|----------|----------|----------|
| maturation_enabled | True | Включение maturation |
| matur_T0 | 8000.0 | Базовая задержка |
| matur_T_delay | 8000.0 | Доп. задержка для shallow |
| matur_delta | 4000.0 | Ширина рампы |

### Unified τ-field
| Параметр | Значение | Описание |
|----------|----------|----------|
| tau_enabled | True | Unified tau config |
| tau_min | 8.0 | Минимальный τ |
| tau_max | 512.0 | Максимальный τ |
| tau_dev_max | 0.3 | Макс. отклонение |
| tau_llrd_gamma | 0.65 | LLRD показатель степени |
| tau_mem_ref | 64.0 | Референс τ для памяти |

### Bridge
| Параметр | Значение | Описание |
|----------|----------|----------|
| bridge_conn | 0.1 | Вес aux loss bridge |
| bridge_dim | 256 | Размерность semantic bridge |
| bridge_depth | True | Cross-layer injection |

### Memory Bank
| Параметр | Значение | Описание |
|----------|----------|----------|
| memory_bank | True | Streaming memory bank |
| mem_l1_slots | 3 | L1 слоты (immediate) |
| mem_l2_slots | 32 | L2 слоты (short-term) |
| mem_l3_concepts | 8 | L3 концепты (long-range) |

### Reasoning
| Параметр | Значение | Описание |
|----------|----------|----------|
| explicit_reasoning | True | Цепочка рассуждений |
| reasoning_max_steps | 8 | Макс. шагов рассуждений |
| reasoning_adaptive | True | Адаптивная глубина |

---

## 3. Структура файлов

```
core/
├── __init__.py            — Экспорт всех публичных классов
├── config.py              — EVAConfig (dataclass с 150+ параметрами)
├── lambda_utils.py        — LambdaConfig (иерархия из λ_d)
├── tau_config.py          — TauConfig (единое τ-поле)
├── embedding.py           — PartitionedEmbedding, LmHead, SigmoidCodedHead, CognitiveCodedHead, RotaryEmbedding
├── bind.py                — BottleneckBind, SpiralBind, TrajectorySpiralBind, TrajectoryManifoldBind
├── mirror.py              — GroupedCognitiveMirror, BridgeGLU
├── mlp.py                 — GroupedMLP
├── block.py               — EVABlock, PrecisionGate, ExactSequenceMemory
├── stack.py               — EVAStack, AdaptiveController, MirrorLRScheduler
├── adaptive_gate.py       — AdaptiveGate, hybrid_gate()
├── spectrum_gate.py       — SpectrumGate
├── layer_bridge_gate.py   — LayerBridgeGate, SpectrumGate (per-layer)
├── bridge.py              — SemanticBridge
├── maturation.py          — MaturationController
├── memory_bank.py         — StreamingMemoryBank, L1Buffer, L2Bank, L3Concepts
├── reasoning.py           — ReasoningMemory, ReasoningGate, ThinkingTokenHead, ReasoningTokens
├── concept_layer.py       — UnifiedConceptLayer
├── vsa_utils.py           — dct_basis, zeckendorf_codes, sparse_block_codes, vsa_prefix_scan, fib_sigmoid_init
├── compression.py         — FCF_CPR (сжатие чекпоинтов)
├── adaptation.py          — LossBalancer, DepthController, LRController, FailureDetector, GradientClipper, build_optimizer
├── projector.py           — Projector (декодирование слов по сигналам концептов)
├── curriculum.py          — CurriculumTracker
├── word_num.py            — Буквенно-числовое кодирование слов
├── live_inference.py      — LiveInference, MirrorMonitor
├── model.py               — Deprecated shim
├── amp_optim.py           — AMP оптимизатор
├── training_guard.py      — Guard обучения
├── migrate.py             — Миграция чекпоинтов
└── archive/               — Архивные версии

scripts/
├── train.py               — Тренировочный цикл
├── analyze.py             — Анализатор чекпоинтов
├── light_analyze.py       — Лёгкий анализатор
├── live_gen.py            — Live-генерация
├── prune.py               — Прунинг
└── download_data.py       — Загрузка данных

tests/
├── test_math_audit.py     — 55 тестов математического аудита
├── test_model.py          — 56 тестов модели
├── test_tau_improvements.py — 63 теста τ-улучшений
├── test_gradient_flow.py  — 40 тестов градиентного потока
└── test_infer.py          — Тесты инференса
```

---

## 4. Embedding

### Файл: `core/embedding.py`

**PartitionedEmbedding**
```python
class PartitionedEmbedding(nn.Module):
    def __init__(self, cfg)
```
- D делится на K=32 сегментов
- Sparse block codes (S=6 активных бит)
- Dense mixing: sigmoid(scale · M · codes) — каждый бит влияет на все сегменты
- Basis: (K, d) learnable параметр
- Применяется RoPE

**SigmoidCodedHead**
```python
class SigmoidCodedHead(nn.Module):
    def __init__(self, cfg, embed_basis=None)
```
- Гибридный sigmoid-softmax гейтинг (hybrid_gate)
- log_probs_for_target для эффективного CE
- Градиент через гибридную формулу

**RotaryEmbedding**
```python
class RotaryEmbedding(nn.Module):
    def __init__(self, D, theta=1000000.0, scaling=1.0, max_len=65536)
```
- RoPE позиционное кодирование с кэшированием cos/sin

---

## 5. Bind

### Файл: `core/bind.py`

**TrajectorySpiralBind** (текущий production bind)
```python
class TrajectorySpiralBind(nn.Module):
    def __init__(self, D, K, cfg)
```
- Траектория: n_dims=3 предыдущих состояния
- Гибридный bind: alpha * HRR + (1-alpha) * element-wise
- Когерентность спиралей |Σ e^{iθ}|²
- Комплексные веса: w_u_re, w_u_im, w_v_re, w_v_im
- W_freq, W_phase — управление частотой и фазой
- hp_norm: _ExpRMSNorm если bind_qk_norm=True
- Возвращает (result, new_traj, coherence)

**BottleneckBind** (legacy)
```python
class BottleneckBind(nn.Module):
    def __init__(self, D: int, K: int, cfg)
```
- Билинейное cross-mixing с Fibonacci/golden-angle сдвигами
- tie_bind: W_out = W_proj^T

---

## 6. Mirror

### Файл: `core/mirror.py`

**GroupedCognitiveMirror**
```python
class GroupedCognitiveMirror(nn.Module):
    def __init__(self, D, G=32, k=32, ...)
```

Ключевые параметры:
- D=2560, G=32, k=32 (staircase: 8/16/32), d=80
- W_proj (G, d, k) — проекция в K-space
- W_out (G, k, d) — проекция обратно (= W_proj^T если tie_mirror_proj)
- alpha_diag (G, d) — диагональное соотношение
- log_scale (G, d) — масштаб коррекции
- tanh_bias (G, d) — смещение tanh
- w_temp, w_global, w_sym_u, w_sym_v — веса сигналов
- w_gate (G, k) — K-space gate
- w_delta_gate (G, k) — delta gate
- mod_scale_mlp (G,) — модуляция MLP гейта
- mod_scale_mem (G,) — модуляция памяти
- w_intent (G, k) — zero-init intent bridge
- b_intent (G,) — zero-init intent bridge
- w_sal (G,) — salience gate
- w_help (G, 1) — helpfulness signal
- w_contra (G,) — contrastive signal
- log_skip_alpha (G,) — skip connection
- _private_mem (G, k) — приватная память эксперта
- _signal_log_weights (5,) — sigmoid/softmax веса сигналов

**BridgeGLU**
```python
class BridgeGLU(nn.Module):
    def __init__(self, G, k)
```
- GLU-style gating: sigmoid(Wg·delta) * sigmoid(Wv·delta)
- log_gain: learnable параметр

**Forward pass:**
1. hp = einsum(h, W_proj) — K-space projection
2. pred_error = hp - hp_prev — temporal prediction error
3. Signals: temp_k, pred_error, smooth_k, sym_k, help_k
4. Weighted combination via _signal_log_weights
5. Delta = W_out(combined_signals)
6. mirror = (tanh(linear) + skip*linear) * log_scale * adapt_scale
7. expert_gate = sigmoid(gate_logits + intent_gate + delta_gate + grad_mod + dvar_mod)
8. mirror = mirror * expert_gate
9. Returns: (mirror_out, mlp_mod, mem_mod, hp, pred_error_norm)

---

## 7. MLP

### Файл: `core/mlp.py`

**GroupedMLP**
```python
class GroupedMLP(nn.Module):
    def __init__(self, D, expand, groups, swiglu=True, gate_b_init=0.25)
```
- G=32 группы, d=80, expand=4
- SwiGLU: gate = silu(W_gate·h) * (a + b·mirror_gate)
- mlp_gate_a (init=1.0), mlp_gate_b (init=0.25)

---

## 8. Block

### Файл: `core/block.py`

**EVABlock**
```python
class EVABlock(nn.Module):
    def __init__(self, cfg: EVAConfig, layer_idx: int)
```

Forward pass:
1. Pre-LN (RMSNorm)
2. Conv (depthwise, kernel=48) + residual
3. Bind (TrajectorySpiralBind)
4. VSA Memory (multi-scale prefix scan, S=4)
5. Mirror (GroupedCognitiveMirror)
6. Output: bind_gated + mem_modulated + mirror
7. Variable Precision Memory (PrecisionGate + ExactSequenceMemory)
8. Spectral (DCT basis scaling)
9. MLP (mirror-conditioned SwiGLU)

**PrecisionGate**
```python
class PrecisionGate(nn.Module):
    def __init__(self, D)
```
- sigmoid(linear(h)) — per-dim gate

**ExactSequenceMemory**
```python
class ExactSequenceMemory(nn.Module):
    def __init__(self, D, k, softmax_free=True)
```
- Self-attention: Q·K^T → attention → V
- softmax_free: LaCUR (sigmoid-normalized mean)

---

## 9. Stack

### Файл: `core/stack.py`

**EVAStack**
```python
class EVAStack(nn.Module):
    def __init__(self, cfg: EVAConfig)
```

Ключевые атрибуты:
- embed — PartitionedEmbedding
- lm_head — SigmoidCodedHead
- layers — nn.ModuleList[EVABlock] (24 слоя)
- reasoning_memory — ReasoningMemory (max_steps=8)
- thinking_head — ThinkingTokenHead (num_reasoning_tokens=4)
- reasoning_gate — ReasoningGate (adaptive depth)
- intent_probe — nn.Linear(D, n_experts * K_max)
- bus_head_proj — nn.Linear(n_experts * K_max, K_head)
- bridge — SemanticBridge (bridge_dim=256)
- layer_bridge_gate — LayerBridgeGate
- tau_config — TauConfig
- memory_bank — StreamingMemoryBank
- maturation — MaturationController

#### Прямой проход (forward)

1. **State initialization**: batch-mismatch guard
2. **Reasoning buffer**: training carries, eval resets
3. **Unified τ-field**: `tau_config.update(mat_gate)`
4. **VSA tau**: `vsa_tau = exp(cumsum(softplus(_vsa_log_param))) + 1.0`
5. **Adaptive gate biases**: per-layer b_i, b_d from AdaptiveController
6. **Global state**: running EMA of layer memory centroids
7. **Intent Bridge**: depth-flowing per-head intent stream
   - `fresh_i = probe_out.mean(dim=(0,1))` — gradient through probe
   - `bus_i = (running + (sum - carried)) / n_layers`
   - `intent_i = bus_i[..., :_ki]` — **НЕ масштабируется mat_gate** (исправлено)
8. **Maturation gate**: per-layer time ramp (deep-first)
9. **Per-layer loop**:
   - AdaptiveController: mem2v_scale, noise_scale, tanh_bias_mod, spectral_mod
   - Intent Bridge: intent_probe → fresh_i, bus_i, alpha_i blending
   - Semantic Bridge: inject_layer, probe_layer, record, update_stream
   - Memory Bank: read/write at sentence boundaries
   - EVABlock: forward (gradient checkpointing optional)
10. **Final norm**: RMSNorm (weight-only)
11. **Explicit Reasoning**: adaptive reasoning loop
12. **Triad**: confidence check → re-circulation if low (inference only)

#### Auxiliary Methods

```python
def compute_losses(self, h, targets, pred_weight=None, h_emb=None)
def compute_salience(self, logits)
def embed_tokens(self, tokens)
def _adaptive_reasoning(self, h, s, state, reasoning_buffer, reasoning_count)
```

---

## 10. Gates

### AdaptiveGate (`core/adaptive_gate.py`)

**Формула**:
```
gate = sigmoid(logits) * (1 + softmax(logits / tau))
```

### SpectrumGate (`core/spectrum_gate.py`)

**Режимы** (по значению tau):
- tau → ∞: pure sigmoid → diversity
- tau ≈ 1: balanced → diversity + precision
- tau → 0: pure softmax → final precision

### LayerBridgeGate (`core/layer_bridge_gate.py`)

- Per-layer SpectrumGate (24 штуки)
- effective_tau = tau_max * (1 - maturation) + tau_min * maturation
- 6 diagnostic features: pred_error_norm, gate_l1, mirror_norm, bridge_contribution, expert_entropy, diversity

---

## 11. TauConfig

### Файл: `core/tau_config.py`

**Формула τ-ladder**:
```python
base_inc = log_tau_range / max(n_layers - 1, 1)
_sp0 = log(2.0)  # softplus(0)
inc = base_inc * softplus(_tau_dev) / _sp0
log_tau = log_tau_min + cumsum(inc, dim=0)
tau_l = exp(log_tau)
```

**Выводимые величины**:
- `tau_norm = (log(tau_l) - log(tau_min)) / (log(tau_max) - log(tau_min))`
- `mat_delay = T0 + (1 - tau_norm) * T_delay`
- `gate_tau = exp(log(gate_tau_max) + (log(gate_tau_min) - log(gate_tau_max)) * mat_gate)`
- `intent_alpha = 1 - exp(-tau_l / tau_min)`
- `lr_mult = (tau_l / tau_mem_ref) ^ (-llrd_gamma)`
- `mem_tau = percentiles(tau_l)` → [L1, L2, L3]

**Текущие значения** (из чекпоинта best 10, step 4893):
- τ-ladder: 7.97 → 503.67 (63x range)
- intent_alpha: [0.70, 1.00]
- lr_mult spread: ~15x

---

## 12. Maturation

### Файл: `core/maturation.py`

**Формула maturation gate**:
```python
gate = sigmoid((step - (T0 + alpha * (1 - tau_norm) * T_delay)) / delta_t)
```

- Deep layers (tau_norm ≈ 1): open at T_eff = T0 = 8000
- Shallow layers (tau_norm ≈ 0): open at T_eff = T0 + T_delay = 16000
- Pure time ramp — bridge_readiness НЕ используется

**Текущие значения** (step ~4893, best 10):
- mat min: 0.064 (shallow)
- mat max: 0.315 (deep)
- readiness max: ~0.95

---

## 13. Bridge

### Файл: `core/bridge.py`

**SemanticBridge**
```python
class SemanticBridge(nn.Module):
    def __init__(self, D, n_layers, bridge_dim=256, depth=True, cfg=None)
```

Ключевые атрибуты:
- probe: Sequential(Linear(D, 256), GELU, Linear(256, 256))
- emb_proj: Linear(D, 256) — для потери
- stream_proj: Linear(256, D) — для инъекции
- stream_log_scale: nn.Parameter(zeros(1)) — injection strength
- stream_log_weights: nn.Parameter(3,) — sigmoid-веса соседей
- bridge_stream: buffer (n_layers, 256) — persistent stream

**Forward pass (per layer):**
1. `h = bridge.inject_layer(i, h, maturity, tau_norm)` — inject cross-layer stream
2. `_s_l = bridge.probe(h.detach())` — probe detached hidden state
3. `bridge.record(_s_l)` — record for loss
4. `bridge.update_stream(i, _s_l)` — EMA update persistent stream

**Injection formula:**
```python
neigh = [bridge_stream[i]]  # self
if i-1 >= 0: neigh.append(bridge_stream[i-1])  # bottom-up
if i+1 < n:  neigh.append(bridge_stream[i+1])  # top-down
sw = sigmoid(stream_log_weights[:len(neigh)])
w = sw / sw.sum()
combined = (stack * w).sum(0)
inj_strength = sigmoid(_inj_alpha) * tau_norm + sigmoid(_inj_beta) * (1 - tau_norm)
scale = tanh(stream_log_scale) * maturity * inj_strength
inj = scale * stream_proj(combined)
return h + inj
```

**Self-supervised loss:**
```python
loss = sum(1 - cosine_similarity(s_l[:, :-1], emb_proj(embed(y[:,1:]))))
```

**Исправление (2026-09-05):** Убрано масштабирование `intent_i` через `mat_gate`. Ранее `mat_gate[i]` (~0.09) масштабировал intent сигнал до ~0, делая gradient(w_intent) вырожденным. Теперь `intent_i` передаётся в mirror без масштабирования — mirror уже управляет зрелостью через `bridge_glu_net * maturity` и `expert_gate`.

---

## 14. Memory Bank

### Файл: `core/memory_bank.py`

**StreamingMemoryBank**
```python
class StreamingMemoryBank(nn.Module):
    def __init__(self, D, bridge_dim, l1_slots=3, l2_slots=32,
                 l3_concepts=8, ...)
```

Уровни (активны с step ~4620):
- **L1Buffer**: Rolling buffer (3 slots), overwrite oldest
- **L2Bank**: Learned memory bank (32 slots), novelty gate
- **L3Concepts**: Emergent concepts (8 slots), concept birth/update

**Fusion**: Linear(4D, D) → GELU → Linear(D, D)

---

## 15. Reasoning

### Файл: `core/reasoning.py`

**ReasoningMemory**
- step_encoder: Linear(D, D)
- step_query, step_key, step_value: Linear(D, D)
- output_proj: Linear(D, D)
- Fixed-size buffer: (B, max_steps, D) + count

**ReasoningGate**
- proj: Linear(D, max_steps) — init bias[0]=10, bias[1:]=0
- know_proj: Linear(know_dim, max_steps)
- r_proj: Linear(D, max_steps)
- Output: tanh(logits) ∈ (-1, 1)

**Adaptive reasoning (stack.py):**
```python
scale = 1 - exp(-step / ramp_steps)
for k in range(K):
    gate = reasoning_gate(h, knowledge)
    if gate.mean() < stop_threshold: break
    h = h + scale * reasoning_memory(h)
```

---

## 16. Concept Layer

### Файл: `core/concept_layer.py`

**UnifiedConceptLayer**
```python
class UnifiedConceptLayer(nn.Module):
    def __init__(self, D, k=256, S=8, ...)
```
- concept_keys (S, k) — prototypes
- concept_vals (S, D) — values
- q_proj, out_proj — attention readout
- write_q_proj, write_v_proj — concept writing
- log_tau_* — τ-gated thresholds

---

## 17. VSA Utilities

### Файл: `core/vsa_utils.py`

```python
def dct_basis(n)          — DCT-II basis (n, n)
def sparse_block_codes(vocab, K=32, S=6)  — Sparse block codes (V, K)
def vsa_prefix_scan(a, b, state=None)      — VSA associative parallel prefix scan
def fib_sigmoid_init(n)                    — Fibonacci-based sigmoid bias init
```

---

## 18. Adaptation

### Файл: `core/adaptation.py`

**DepthController**
```python
class DepthController:
    def __init__(self, model, n_layers, init_k=8, unfreeze_inc=4, ...)
```
- Progressive unfreezing via val-loss plateau detection
- Текущая динамика: 12/24 → 16/24 → 20/24 → 24/24 за ~1000 шагов

**LossBalancer**
- mode='align': PCGrad-style gradient projection
- mode='balance': dimensionless per-aux normalization

**FailureDetector**
- SPC 3σ rule for CE explosion detection

**GradientClipper**
- Adaptive Gradient Clipping (AGC)

---

## 19. Lambda Utils

### Файл: `core/lambda_utils.py`

```python
def lambda_d(d: int) -> float  — positive root of x^d = x^{d-1} + ... + 1
def spectral_radius(model, h)  — power iteration estimate of ρ(J)
```

**LambdaConfig** — все гиперпараметры выведены из λ_d

---

## 20. Scripts

### scripts/analyze.py

Единый анализатор чекпоинтов EVA. Методы:
- `load_ckpt(path)` — загрузка чекпоинта
- `run_static(ckpt, cfg, model, missing, unexpected)` — конфиг, per-layer параметры
- `run_inspector(model)` — сигналы, trust/concept/dominance
- `run_wake(model, ckpt)` — вердикт PASS/WATCH/WAKE
- `run_live(model, cfg)` — forward на случайном входе
- `run_head(model, ckpt, args, tok)` — декомпозиция bias vs контекст
- `run_bridge(model, cfg)` — runtime intent-bus metrics
- `parse_training_log(path)` — парсинг логов
- `render_log_html(data, outpath)` — HTML-отчёт

---

## 21. Текущие результаты обучения

### История чекпоинтов

| Чекпоинт | Step | val_loss |
|----------|------|----------|
| best 1 | 466 | 10.985 |
| best 2 | 1165 | 10.764 |
| best 3 | 1398 | 10.692 |
| best 4 | 1864 | 10.561 |
| best 5 | 2796 | 10.398 |
| best 6 | 3029 | 10.361 |
| best 7 | 3728 | 10.218 |
| best 8 | 3961 | 10.160 |
| best 9 | 4660 | 10.006 |
| best 10 | 4893 | 9.937 |

### Динамика ключевых метрик

- **val_loss**: 10.985 → 9.937 (Δ=-1.048 за ~4400 шагов, впервые < 10)
- **Memory Bank**: активировался на step ~4620, L1/L2/L3 работают
- **bridge_conn**: 0.12 → 0.13 (растёт)
- **Maturation**: 0.150 → 0.171 (deep layers 0.064 → 0.315)

---

## 22. Известные проблемы и исправления

### P0: bridge gradient killed by mat_gate (ИСПРАВЛЕНО)

**Проблема:** `intent_i` масштабировался `mat_gate[i]` (~0.09) перед передачей в mirror. Это делало `ik ≈ 0`, gradient(w_intent) ∝ (hp - ik) ≈ hp — вырожденный gradient, не зависящий от intent-сигнала.

**Исправление** (`stack.py:417-419`): Убрано масштабирование `intent_i` через `mat_gate`. Mirror уже управляет зрелостью через `bridge_glu_net * maturity` и `expert_gate`.

**Результат:** w_intent начал получать gradient. CE продолжает падать.

### P0.1: layer_bridge_gate.log_tau — мёртвые параметры (ИСПРАВЛЕНО)

**Проблема:** Все 24 `SpectrumGate.log_tau` имели `grad_norm=0`.

**Исправление:** Вычисление gate output перенесено вне `no_grad()` блока. Добавлен `lbg_diversity_loss`.

### P0.2: memory_bank.tau_config — τ-prior не применялся (ИСПРАВЛЕНО)

**Проблема:** StreamingMemoryBank не сохранял tau_config.

**Исправление:** Добавлено `self.tau_config = tau_config`.

### P0.3: _tau_dev — односторонний collapse (ИСПРАВЛЕНО)

**Проблема:** Оптимизатор толкал все значения в минус.

**Исправление:** Добавлен `tau_dev_reg` — L2-регуляризация к нулю.

### P0.4: cfg.llrd — мёртвый параметр (ИСПРАВЛЕНО)

**Проблема:** `cfg.llrd=0.9` существовал, но не использовался.

**Исправление:** Помечен как `DEPRECATED`.

### P1: reasoning K_full/K scaling (ИСПРАВЛЕНО)

**Проблема:** `_adaptive_reasoning` масштабировал K через τ_norm (8→4), но буфер создавался с K=4, а `ReasoningMemory` строил маску с `max_steps=8` → shape mismatch.

**Исправление** (`stack.py`): Буфер всегда создаётся с `K_full` (original max_steps), цикл запускает `K` (scaled) итераций.

### Тесты

- `test_math_audit.py`: 55/55 ✅
- `test_model.py`: 56/56 ✅
- `test_tau_improvements.py`: 63/63 ✅
- `test_gradient_flow.py`: 40/40 ✅
- `test_infer.py`: fixture error (pre-existing, не связан с архитектурой)
- **Итого: 214/214 тестов проходят**

---

## 23. Полный отчёт

### Все классы core/

| Класс | Файл | Описание |
|-------|------|----------|
| EVAConfig | config.py | Конфигурация модели |
| LambdaConfig | lambda_utils.py | Иерархия из λ_d |
| TauConfig | tau_config.py | Единое τ-поле |
| PartitionedEmbedding | embedding.py | Эмбеддинг через sparse block codes |
| SigmoidCodedHead | embedding.py | Sigmoid-coded голова |
| CognitiveCodedHead | embedding.py | Cognitive-coded голова |
| RotaryEmbedding | embedding.py | RoPE |
| BottleneckBind | bind.py | Билинейный bind |
| SpiralBind | bind.py | Спиральный bind |
| TrajectorySpiralBind | bind.py | Траекторный спиральный bind |
| TrajectoryManifoldBind | bind.py | Манифолд переходов |
| GroupedCognitiveMirror | mirror.py | Ансамбль 32 экспертов |
| BridgeGLU | mirror.py | GLU-style gating |
| GroupedMLP | mlp.py | Grouped SwiGLU MLP |
| EVABlock | block.py | Один слой модели |
| PrecisionGate | block.py | Variable precision gate |
| ExactSequenceMemory | block.py | Exact sequence memory |
| EVAStack | stack.py | Полная модель |
| AdaptiveGate | adaptive_gate.py | Sigmoid-softmax hybrid gate |
| SpectrumGate | spectrum_gate.py | Spectrum gate |
| LayerBridgeGate | layer_bridge_gate.py | Per-layer bridge gate |
| SemanticBridge | bridge.py | Semantic bridge |
| MaturationController | maturation.py | Maturation gate |
| StreamingMemoryBank | memory_bank.py | L1+L2+L3 память |
| L1Buffer | memory_bank.py | Rolling buffer |
| L2Bank | memory_bank.py | Learned memory bank |
| L3Concepts | memory_bank.py | Emergent concepts |
| ReasoningMemory | reasoning.py | Chain-of-thought memory |
| ReasoningGate | reasoning.py | Adaptive reasoning depth |
| ThinkingTokenHead | reasoning.py | Thinking token predictions |
| UnifiedConceptLayer | concept_layer.py | Concept layer |
| FCF_CPR | compression.py | Checkpoint compression |
| LossBalancer | adaptation.py | Multi-task balancing |
| DepthController | adaptation.py | Progressive unfreezing |
| LRController | adaptation.py | LR control |
| FailureDetector | adaptation.py | Divergence detection |
| GradientClipper | adaptation.py | AGC |
| Projector | projector.py | Word readout |
| CurriculumTracker | curriculum.py | Curriculum learning |
| MirrorMonitor | live_inference.py | Runtime tracer |
| LiveInference | live_inference.py | Stateful inference |

### Все standalone функции core/

| Функция | Файл | Описание |
|---------|------|----------|
| hybrid_gate() | adaptive_gate.py | Unified sigmoid-softmax gate |
| dct_basis() | vsa_utils.py | DCT-II basis |
| sparse_block_codes() | vsa_utils.py | Sparse block codes |
| vsa_prefix_scan() | vsa_utils.py | VSA prefix scan |
| lambda_d() | lambda_utils.py | Generalized golden ratio |
| spectral_radius() | lambda_utils.py | Spectral radius estimate |
| set_active_depth() | adaptation.py | Progressive unfreezing |
| build_optimizer() | adaptation.py | AdamW with LLRD |
| _memory_attention() | memory_bank.py | Hybrid memory attention |
