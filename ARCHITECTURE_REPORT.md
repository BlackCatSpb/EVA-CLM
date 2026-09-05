# ARCHITECTURE REPORT — WideBind (EVA v3)

Автоматически сгенерированный отчёт по архитектуре WideBind.
Версия от 2026-09-03. Все формулы, классы, функции и дефолты извлечены напрямую из исходного кода.

---

## Содержание

1. [Обзор архитектуры](#1-обзор-архитектуры)
2. [Конфигурация WideBindConfig](#2-конфигурация-widebindconfig)
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
18. [Compression (compression.py)](#18-compression)
19. [Adaptation (adaptation.py)](#19-adaptation)
20. [Lambda Utils (lambda_utils.py)](#20-lambda-utils)
21. [Projector (projector.py)](#21-projector)
22. [Curriculum (curriculum.py)](#22-curriculum)
23. [Word Num (word_num.py)](#23-word-num)
24. [Live Inference (live_inference.py)](#24-live-inference)
25. [Scripts: analyze.py](#25-scripts)
26. [Полный отчёт по классам и функциям](#26-полный-отчёт)

---

## 1. Обзор архитектуры

WideBind — это языковая модель на основе архитектуры Mixture-of-Experts (MoE) с когнитивным зеркалом. Ключевые особенности:

- **Кодирование токенов**: Sparse Block Codes (K=32, S=6) с внешним произведением для создания плотных эмбеддингов
- **Bind**: Билинейное скрещивание с золотым угловым сдвигом (Fibonacci) или спиральной траекторией
- **Mirror**: Ансамбль из G=32 экспертов, каждый в своём d-мерном подпространстве (D=4096 → d=128)
- **VSA Memory**: Multi-scale prefix scan (S=4 фиксированных τ: [8, 32, 128, 512])
- **Conv**: Depthwise свёртка с kernel=48
- **Spectral**: DCT-базис с индивидуальными масштабами частот
- **MLP**: Grouped SwiGLU с expand=4 (32 группы, по 128 dims)
- **Head**: SigmoidCodedHead — гибридный sigmoid-softmax гейтинг
- **Maturation**: Единый временной/τ-рамп для управления пробуждением слоёв
- **Bridge**: Семантический мост (самосупервизия через предсказание следующего токена)
- **Intent Bridge**: Нисходяще-восходящая передача «намерения» экспертам
- **Reasoning**: Цепочка рассуждений (chain-of-thought) с адаптивной глубиной
- **Concept Layer**: Коллективный слой концептов ( emergent concepts)
- **Memory Bank**: Иерархическая L1+L2+L3 память

---

## 2. Конфигурация WideBindConfig

Дефолты из `core/config.py` (строки 12–411):

### Основные размерности
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| D | 4096 | Ширина модели |
| n_layers | 32 | Число слоёв |
| bind_K | 64 | Bottleneck K для bind |
| vocab | 50000 | Размер словаря |
| seq_len | 128 | Длина последовательности |
| batch_size | 2 | Размер батча |
| mlp_groups | 32 | Число групп в MLP (G) |
| mlp_expand | 4 | Множитель расширения в MLP |

### Обучение
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| lr | 3e-4 | Learning rate |
| warmup_steps | 1000 | Шаги разогрева |
| weight_decay | 0.01 | L2 регуляризация |
| grad_clip | 0.5 | Обрезка градиентов |
| max_steps | 500000 | Максимальное число шагов |

### Binder
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| tie_bind | True | W_out = W_proj^T |
| bind_twist_mode | "trajectory_spiral" | Режим bind |
| bind_twist_S | 4 | Число螺旋 |
| bind_traj_dims | 3 | Размерность траектории |
| bind_qk_norm | True | RMSNorm на hp (QK-Norm) |

### Mirror
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| mirror_k | 32 | K-space размерность эксперта |
| mirror_k_staircase | True | k_l ∈ {8,16,32} по третям глубины |
| private_mem | True | Приватная память экспертов |
| expert_asymmetry | True | Асимметричная инициализация |
| meta_trust | True | Рекурсивное мета-доверие |

### Head
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| head_mode | "sigmoid_coded" | Тип головы |
| head_normalize | True | Нормализация логитов |
| code_dim | 32 | Размер кода |
| code_sparsity | 6 | Число активных бит |

### Maturation
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| maturation_enabled | True | Включение maturation |
| matur_T0 | 8000.0 | Базовая задержка |
| matur_T_delay | 8000.0 | Доп. задержка для shallow |
| matur_delta | 4000.0 | Ширина рампы |
| matur_r0 | 0.3 | Центр сигмоиды readiness |
| matur_rs | 0.2 | Наклон сигмоиды readiness |
| matur_write_thr | 0.3 | Порог зрелости для записи |

### Unified τ-field
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| tau_enabled | True | Unified tau config |
| tau_min | 8.0 | Минимальный τ |
| tau_max | 512.0 | Максимальный τ |
| tau_dev_max | 0.3 | Макс. отклонение |
| tau_llrd_gamma | 0.65 | LLRD показатель степени |
| tau_mem_ref | 64.0 | Референс τ для памяти |

### Bridge
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| bridge_conn | 0.1 | Вес aux loss bridge |
| bridge_dim | 256 | Размерность semantic bridge |
| bridge_depth | True | Cross-layer injection |
| intent_bridge | True | Intent Bridge включён |
| bridge_glu | True | BridgeGLU включён |
| bridge_glu_beta | 0.25 | Модуляция BridgeGLU |

### Memory Bank
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| memory_bank | False | Streaming memory bank |
| mem_l1_slots | 3 | L1 слоты (immediate) |
| mem_l2_slots | 16 | L2 слоты (short-term) |
| mem_l3_concepts | 8 | L3 концепты (long-range) |
| mem_l3_birth_threshold | 0.7 | Порог рождения концепта |

### Reasoning
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| explicit_reasoning | True | Цепочка рассуждений |
| reasoning_max_steps | 8 | Макс. шагов рассуждений |
| reasoning_adaptive | True | Адаптивная глубина |
| reasoning_gate_stop_threshold | 0.5 | Порог остановки |

### Triad (Рассудок)
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| triad_reason | True | Ре-циркуляция при низкой уверенности |
| triad_conf_thr | 0.5 | Порог уверенности |
| triad_max_passes | 3 | Бюджет ре-циркуляций |

### Scheduler
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| scheduler | 'mirror' | Тип планировщика |
| target_var | 0.1 | Целевая дисперсия |
| exploration_threshold | 0.25 | Порог исследования |
| differentiation_threshold | 0.08 | Порог дифференциации |

### Auxiliary Losses
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| gate_l1_weight | 0.0001 | L1 на гейты |
| balance_weight | 0.026 | Балансировка нагрузки |
| diversity_weight | 0.001 | Декорреляция |
| nuclear_weight | 1e-5 | Ядерная норма |
| orth_weight | 1e-4 | Ортогональность |
| surprisal_weight | 0.0 | Surprisal-weighted loss |
| branch_balance_weight | 0.0 | Баланс ветвей |
| gradalign_weight | 0.0 | Градиентное выравнивание |

### Qwen3-inspired
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| rope_theta | 1000000.0 | RoPE базовая частота |
| rope_scaling | 1.0 | RoPE масштаб |
| mlp_swiglu | True | SwiGLU gate в MLP |

### Spectrum
| Параметр | Дефолт | Описание |
|----------|--------|----------|
| spec_lo | 0.5 | Мин. спектральный масштаб |
| spec_hi | 1.5 | Макс. спектральный масштаб |

---

## 3. Структура файлов

```
core/
├── __init__.py            — Экспорт всех публичных классов
├── config.py              — WideBindConfig (dataclass с 150+ параметрами)
├── lambda_utils.py        — LambdaConfig (иерарифия из λ_d)
├── tau_config.py          — TauConfig (единое τ-поле)
├── embedding.py           — PartitionedEmbedding, LmHead, SigmoidCodedHead, CognitiveCodedHead, RotaryEmbedding
├── bind.py                — BottleneckBind, SpiralBind, TrajectorySpiralBind, TrajectoryManifoldBind
├── mirror.py              — GroupedCognitiveMirror, BridgeGLU
├── mlp.py                 — GroupedMLP
├── block.py               — WideBindBlock, PrecisionGate, ExactSequenceMemory
├── stack.py               — WideBindStack, AdaptiveController, MirrorLRScheduler
├── adaptive_gate.py       — AdaptiveGate, hybrid_gate()
├── spectrum_gate.py       — SpectrumGate
├── layer_bridge_gate.py   — LayerBridgeGate, SpectrumGate (per-layer)
├── bridge.py              — SemanticBridge
├── maturation.py          — MaturationController
├── memory_bank.py         — StreamingMemoryBank, L1Buffer, L2Bank, L3Concepts
├── reasoning.py           — ReasoningMemory, ReasoningGate, ThinkingTokenHead, ReasoningTokens
├── concept_layer.py       — CollectiveConceptLayer
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
```

---

## 4. Embedding

### Файл: `core/embedding.py`

#### Классы

**RotaryEmbedding**
```python
class RotaryEmbedding(nn.Module):
    def __init__(self, D, theta=1000000.0, scaling=1.0, max_len=65536)
```
- RoPE позиционное кодирование с кэшированием cos/sin
- Применяется после PartitionedEmbedding

**ZeckendorfEmbedding** (legacy)
```python
class ZeckendorfEmbedding(nn.Module):
    def __init__(self, cfg)
```
- Token → D-space через Zeckendorf codes + Linear проекцию
- Ранг матрицы эмбеддингов ≤ K=23

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

**LmHead** (legacy)
```python
class LmHead(nn.Module):
    def __init__(self, cfg)
```
- D-space → vocab logits через Zeckendorf code проекцию

**PartitionedHead**
```python
class PartitionedHead(nn.Module):
    def __init__(self, cfg, embed_basis=None)
```
- Segment-addressed readout + per-token bias
- weight tying с PartitionedEmbedding.basis (если передан)

**SigmoidCodedHead**
```python
class SigmoidCodedHead(nn.Module):
    def __init__(self, cfg, embed_basis=None)
```
- Гибридный sigmoid-softmax гейтинг (hybrid_gate)
- log_probs_for_target для эффективного CE
- Градиент через гибридную формулу

**CognitiveCodedHead**
```python
class CognitiveCodedHead(nn.Module):
    def __init__(self, cfg, embed_basis=None, k_mirror=32)
```
- Cognitive-coded голова с resonance/energy/social bias
- set_cognitive_state для передачи pred_error, private_mem, trust_matrix

---

## 5. Bind

### Файл: `core/bind.py`

#### Вспомогательные функции

```python
def _golden_shifts(K: int, S: int) -> list
def _fibonacci_shifts(K: int, S: int) -> list
def migrate_bind_state_dict(sd, n_layers, mode="off", S=1)
```

#### Классы

**_ExpRMSNorm**
```python
class _ExpRMSNorm(nn.Module):
    def __init__(self, K)
```
- RMSNorm через явную формулу (ONNX-exportable)

**BottleneckBind**
```python
class BottleneckBind(nn.Module):
    def __init__(self, D: int, K: int, cfg)
```
- Билинейное cross-mixing с Fibonacci/golden-angle сдвигами
- Режимы: "off" (legacy diagonal), "shift" (сумма S shifted products), "cascade" (Fibonacci-nested)
- Ocular: "tied" (shared W_out) или "multi" (per-shift W_out)
- Параметры: W_proj (D→K), w_u (S,K), w_v (S,K), w_bind_bias (K), W_out (K,D)
- hp_norm: _ExpRMSNorm если bind_qk_norm=True

**SpiralBind**
```python
class SpiralBind(nn.Module):
    def __init__(self, D, K, cfg)
```
- Комплексное спиральное скрещивание
- w_u_re, w_u_im, w_v_re, w_v_im — комплексные веса
- W_freq, W_phase — управление частотой и фазой
- Выход: (2K, D) → W_out проекция

**TrajectorySpiralBind**
```python
class TrajectorySpiralBind(nn.Module):
    def __init__(self, D, K, cfg)
```
- Траектория: n_dims=3 предыдущих состояния
- Гибридный bind: alpha * HRR + (1-alpha) * element-wise
- Когерентность спиралей |Σ e^{iθ}|²
- Возвращает (result, new_traj, coherence)

**TrajectoryManifoldBind**
```python
class TrajectoryManifoldBind(TrajectorySpiralBind):
    def __init__(self, D, K, cfg)
```
- Манифолд переходов + VSA-бандл чтение
- Лучи (beams) с Zeckendorf-затуханием по возрасту
- buffer_size, n_beams, cos_threshold, rebuild_interval, gain
- W_man: проекция манифолда

---

## 6. Mirror

### Файл: `core/mirror.py`

#### Классы

**BridgeGLU**
```python
class BridgeGLU(nn.Module):
    def __init__(self, G, k)
```
- GLU-style gating: sigmoid(Wg·delta) * sigmoid(Wv·delta)
- log_gain: learnable параметр (init ~2.7 → sigmoid ~0.667)

**GroupedCognitiveMirror**
```python
class GroupedCognitiveMirror(nn.Module):
    def __init__(self, D, G=32, k=32, log_scale_init_std=0.05,
                 delta_var_ema_min=0.8, delta_var_ema_max=0.99, tie_mirror_proj=False,
                 layer_idx=0, n_layers=32, has_private_mem=False,
                 expert_asymmetry=False, meta_trust=False,
                 gate_bias_scale=0.0, alpha_novelty_weight=0.0, seq_len=256,
                 intent_bridge=False, bridge_glu=False, bridge_glu_beta=0.25,
                 pm_write_delay=5000, pm_coh_gate_std=0.02, matur_write_thr=0.3)
```

Ключевые атрибуты:
- D, G, k, d = D//G — размерности
- W_proj (G, d, k) — проекция в K-space
- W_out (G, k, d) — проекция обратно
- alpha_diag (G, d) — диагональное соотношение
- log_scale (G, d) — масштаб коррекции
- tanh_bias (G, d) — смещение tanh
- w_temp, w_smooth, w_sym, w_help — веса сигналов
- mod_scale_mlp (G,) — модуляция MLP гейта
- mod_scale_mem (G,) — модуляция памяти
- w_intent (G, k) — zero-init intent bridge
- b_intent (G, k) — zero-init intent bridge
- w_sal (G,) — salience gate
- _private_mem (G, k, k) — приватная память эксперта
- _signal_log_weights (5,) — sigmoid/softmax веса сигналов

Forward возвращает: (mirror_out, mlp_mod, mem_mod, hp, pred_error_norm)

---

## 7. MLP

### Файл: `core/mlp.py`

**GroupedMLP**
```python
class GroupedMLP(nn.Module):
    def __init__(self, D, expand, groups, swiglu=True, gate_b_init=0.25)
```
- G=32 группы, d=D//G=128, expand=4
- SwiGLU: gate = silu(W_gate·h) · (a + b·mirror_gate)
- Параметры: W_gate, W_up, W_down, norm_w
- mlp_gate_a (init=1.0), mlp_gate_b (init=0.25) — когнитивное управление

---

## 8. Block

### Файл: `core/block.py`

#### Вспомогательные функции

```python
def _scan_chunk(b_chunk, d_chunk)  — параллельный чанк-скан из нулевого состояния
def _combine_chunks(chunk_data, initial_state)  — 2-уровневый cross-chunk скан
```

#### Классы

**PrecisionGate**
```python
class PrecisionGate(nn.Module):
    def __init__(self, D)
```
- sigmoid(linear(h)) — per-dim гейт

**ExactSequenceMemory**
```python
class ExactSequenceMemory(nn.Module):
    def __init__(self, D, k, softmax_free=True)
```
- Self-attention: Q·K^T → attention → V
- softmax_free: LaCUR (sigmoid-normalized mean)

**WideBindBlock**
```python
class WideBindBlock(nn.Module):
    def __init__(self, cfg: WideBindConfig, layer_idx: int)
```

Ключевые атрибуты:
- D, K = cfg.bind_K, layer_idx, tie_bind
- pre_ln_w (D,) — RMSNorm weight
- bind — BottleneckBind / SpiralBind / TrajectorySpiralBind / TrajectoryManifoldBind
- mirror — GroupedCognitiveMirror
- _n_scales = 4, _tau_s = [8, 32, 128, 512]
- w_i, w_d, w_q, w_q_leaf, w_q_ctx, w_mem2v — VSA параметры
- w_q_dyn, w_i_dyn, w_d_pen, w_bind_gate — dynamic memory
- scale_w — per-scale combination weights
- b_i, b_d — content-dependent gates
- gamma_surprisal — surprisal-gated write
- bind_coh_gate — coherence gate
- conv — depthwise Conv1d (kernel=48)
- V_dct, lambda_k — spectral (DCT basis + per-dim scale)
- mlp — GroupedMLP
- precision_gate, exact_memory — Variable Precision Memory
- collective — CollectiveConceptLayer (optional)

Forward pass (строки 242–493):
1. Pre-LN (RMSNorm)
2. Conv (depthwise) + residual
3. Bind (BottleneckBind / SpiralBind / TrajectorySpiralBind)
4. VSA Memory (multi-scale prefix scan S=4)
5. Mirror (GroupedCognitiveMirror)
6. Output: bind_gated + mem_modulated + mirror + collective
7. Variable Precision Memory (PrecisionGate + ExactSequenceMemory)
8. Spectral (DCT basis scaling)
9. MLP (mirror-conditioned SwiGLU)

---

## 9. Stack

### Файл: `core/stack.py`

**WideBindStack**
```python
class WideBindStack(nn.Module):
    def __init__(self, cfg: WideBindConfig)
```

Ключевые атрибуты:
- embed — PartitionedEmbedding
- lm_head — SigmoidCodedHead / CognitiveCodedHead / PartitionedHead
- layers — nn.ModuleList[WideBindBlock]
- reasoning_memory — ReasoningMemory (if explicit_reasoning)
- thinking_head — ThinkingTokenHead
- reasoning_gate — ReasoningGate (if reasoning_adaptive)
- intent_probe — nn.Linear(D, n_experts * K_max) (if intent_bridge)
- bus_head_proj — nn.Linear(n_experts * K_max, K_head) (if intent_bridge)
- bridge — SemanticBridge (if bridge_conn > 0)
- layer_bridge_gate — LayerBridgeGate (if bridge_conn > 0)
- tau_config — TauConfig
- _vsa_log_param — nn.Parameter (4,)
- _tau_l_dev — alias tau_config._tau_dev
- memory_bank — StreamingMemoryBank (if memory_bank)
- maturation — MaturationController (if maturation_enabled)
- final_norm_w — (D,) ones

#### Прямой проход (forward, строки 153–585)

```python
def forward(self, h, state=None, global_state=None, pred_weight=None, adaptive=True,
            context_mem=None, allow_write=None, step=None,
            reasoning_buffer=None, reasoning_count=None, intent_state=None,
            tokens=None, _triad_depth: int = 0)
```

1. **State initialization**: batch-mismatch guard
2. **Reasoning buffer**: training carries, eval resets
3. **Unified τ-field**: `tau_config.update(mat_gate)`
4. **VSA tau**: `vsa_tau = exp(cumsum(softplus(_vsa_log_param))) + 1.0`
5. **Adaptive gate biases**: per-layer b_i, b_d from AdaptiveController
6. **Global state**: running EMA of layer memory centroids
7. **Intent Bridge**: depth-flowing per-head intent stream
8. **Maturation gate**: per-layer time ramp (deep-first)
9. **Per-layer loop**:
   - AdaptiveController: mem2v_scale, noise_scale, tanh_bias_mod, spectral_mod
   - Intent Bridge: intent_probe → fresh_i, bus_i, alpha_i blending
   - Semantic Bridge: inject_layer, probe_layer, record, update_stream
   - Memory Bank: read/write at sentence boundaries
   - WideBindBlock: forward (gradient checkpointing optional)
   - Layer Bridge Gate: per-layer diagnostics
   - Global state update: alpha_l * gs_i + (1-alpha_l) * mem_avg
10. **Final norm**: RMSNorm (weight-only)
11. **Explicit Reasoning**: adaptive reasoning loop
12. **Triad**: confidence check → re-circulation if low

#### Auxiliary Methods

```python
def _knowledge_signal(self, h, state=None)  — (B, 8) confidence/entropy/representation metrics
def _last_conf(self, h)  — p1 of last position
def _adaptive_reasoning(self, h, s, state, reasoning_buffer, reasoning_count)
def reasoning_scale  — property: 1 - exp(-t/ramp_steps)
def reset_reasoning(self)
def embed_tokens(self, tokens)
def compute_loss(self, h, targets, pred_weight=None, h_emb=None)
def compute_losses(self, h, targets, pred_weight=None, h_emb=None)  — CE + aux dict
def _finalize_ce(self, ce, targets)  — mask PAD/EOS + surprisal weighting
def compute_salience(self, logits)  — word importance from head
def observe_output(self, logits)  — store salience for intent
```

---

## 10. Gates

### AdaptiveGate (`core/adaptive_gate.py`)

```python
class AdaptiveGate(nn.Module):
    def __init__(self, n_features: int, tau_init: float = 1.0, learnable_tau: bool = True)
```

**Формула**:
```
gate = sigmoid(logits) * (1 + softmax(logits / tau))
```

- sigmoid: independent activation per feature
- softmax: relative emphasis among features
- tau: temperature (learnable via log_tau)

**hybrid_gate** (standalone function):
```python
def hybrid_gate(logits, tau, dim=-1, log=False, normalize=False, eps=1e-7)
```

### SpectrumGate (`core/spectrum_gate.py`)

```python
class SpectrumGate(nn.Module):
    def __init__(self, n_features, tau_init=1.0, tau_min=0.1, tau_max=10.0, learnable_tau=True)
```

**Режимы** (по значению tau):
- tau → ∞: pure sigmoid → diversity
- tau ≈ 1: balanced → diversity + precision
- tau → 0: pure softmax → final precision

**Формула**: `gate = sigmoid(logits) * (1 + softmax(logits / tau))`

### LayerBridgeGate (`core/layer_bridge_gate.py`)

```python
class LayerBridgeGate(nn.Module):
    def __init__(self, n_layers, health_features=6, tau_min=0.3, tau_max=5.0)
```

- Per-layer SpectrumGate (n_layers штук)
- effective_tau = tau_max * (1 - maturation) + tau_min * maturation
- 6 diagnostic features: pred_error_norm, gate_l1, mirror_norm, bridge_contribution, expert_entropy, diversity
- Global readiness: bypass до созревания всех слоёв

---

## 11. TauConfig

### Файл: `core/tau_config.py`

```python
class TauConfig(nn.Module):
    def __init__(self, n_layers=24, tau_min=8.0, tau_max=512.0, tau_init_spread=0.5,
                 dev_max=0.3, T0=8000.0, T_delay=8000.0, delta_t=4000.0,
                 gate_tau_min=0.3, gate_tau_max=5.0, mem_tau_ref=64.0, llrd_gamma=0.65)
```

**Параметры**: _tau_dev (nn.Parameter, n_layers) — learnable deviation

**Формула τ-ladder** (строки 112–128):
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

---

## 12. Maturation

### Файл: `core/maturation.py`

```python
class MaturationController(nn.Module):
    def __init__(self, n_layers, tau_min, tau_max, cfg, tau_config=None)
```

**Формула maturation gate** (строка 115–116):
```python
gate = sigmoid((step - (T0 + alpha * (1 - tau_norm) * T_delay)) / delta_t)
```

- Deep layers (tau_norm ≈ 1): open at T_eff = T0
- Shallow layers (tau_norm ≈ 0): open at T_eff = T0 + T_delay
- Pure time ramp — bridge_readiness НЕ используется (scalar бы убил per-layer gradient)

**Свойства**:
- `global_ready`: True когда ВСЕ слои > bridge_control_threshold
- `global_readiness_ratio`: доля созревших слоёв

**Update** (строки 136–146):
```python
pen_ema = lerp(pen_ema, pred_err, 1 - ema)
sat = (1 - pen_ema / pen_init).clamp(0, 1)
readiness = sigmoid((sat - r0) / rs)
```

---

## 13. Bridge

### Файл: `core/bridge.py`

```python
class SemanticBridge(nn.Module):
    def __init__(self, D, n_layers, bridge_dim=256, depth=True, cfg=None)
```

Ключевые атрибуты:
- probe: Linear(D, bridge_dim) → GELU → Linear(bridge_dim, bridge_dim)
- emb_proj: Linear(D, bridge_dim) — для потери
- stream_proj: Linear(bridge_dim, D) — для инъекции
- stream_log_scale: nn.Parameter — bounded injection strength
- stream_log_weights: nn.Parameter(3,) — sigmoid-веса соседей
- bridge_stream: buffer (n_layers, bridge_dim) — persistent stream
- bridge_loss_init, bridge_loss_ema — readiness tracking

**Методы**:
```python
def readiness(self)  — scalar ∈ [0,1] по компетентности bridge
def start_forward(self)  — reset _preds list
def probe_layer(self, h_l)  — emit semantic vector
def inject_layer(self, i, h_l, maturity=None)  — add stream signal
def update_stream(self, i, s_l)  — EMA update
def record(self, s_l)  — record for loss
def reset_stream(self)
def loss(self, y, embed_fn)  — self-supervised: 1 - cos(s_l, emb_proj(next_embed))
```

**Readiness formula**:
```python
init = bridge_loss_init
sat = (1 - bridge_loss_ema / init).clamp(0, 1)
base = sigmoid(-r0 / rs)
readiness = sigmoid((sat - r0) / rs) - base
```

---

## 14. Memory Bank

### Файл: `core/memory_bank.py`

**_memory_attention** (standalone):
```python
def _memory_attention(q, k, temp, bridge_dim, softmax_free=True, age_decay=None)
```
- Hybrid: gate = sigmoid(scores) * (1 + softmax(scores / tau))

**L1Buffer**
```python
class L1Buffer(nn.Module):
    def __init__(self, D, bridge_dim, n_slots=3, softmax_free=True, tau_prior=0.5)
```
- Rolling buffer: overwrite oldest slot
- Hybrid attention read

**L2Bank**
```python
class L2Bank(nn.Module):
    def __init__(self, D, bridge_dim, n_slots=16, softmax_free=True, tau_prior=1.0)
```
- Learned memory bank with novelty gate
- Consumed slots (from L3) prioritized for overwrite

**L3Concepts**
```python
class L3Concepts(nn.Module):
    def __init__(self, D, bridge_dim, n_concepts=8, birth_threshold=0.7,
                 update_momentum=0.1, softmax_free=True, tau_prior=2.0)
```
- Emergent concepts from L2 clustering
- Concept birth/update via cosine similarity

**StreamingMemoryBank**
```python
class StreamingMemoryBank(nn.Module):
    def __init__(self, D, bridge_dim, l1_slots=3, l2_slots=16,
                 l3_concepts=8, l3_birth_threshold=0.7,
                 min_write_maturation=0.3, softmax_free=True, cfg=None, tau_config=None)
```
- L1 + L2 + L3 combined
- Fusion gate: Linear(4D, D) → GELU → Linear(D, D)
- log_scale injection strength

---

## 15. Reasoning

### Файл: `core/reasoning.py`

**ReasoningTokens** (static class):
```python
class ReasoningTokens:
    THINK = 65536
    STEP = 65537
    ANSWER = 65538
    END = 65539
```

**ReasoningMemory**
```python
class ReasoningMemory(nn.Module):
    def __init__(self, D, max_steps=8)
```
- step_encoder: Linear(D, D)
- step_query, step_key, step_value: Linear(D, D)
- output_proj: Linear(D, D)
- Fixed-size buffer: (B, max_steps, D) + count

**ReasoningGate**
```python
class ReasoningGate(nn.Module):
    def __init__(self, D, max_steps=8, know_dim=8)
```
- proj: Linear(D, max_steps) — init bias[0]=10, bias[1:]=0
- know_proj: Linear(know_dim, max_steps)
- r_proj: Linear(D, max_steps) — candidate influence
- Output: tanh(logits) ∈ (-1, 1)

**ThinkingTokenHead**
```python
class ThinkingTokenHead(nn.Module):
    def __init__(self, D, num_reasoning_tokens=4)
```

---

## 16. Concept Layer

### Файл: `core/concept_layer.py`

```python
class CollectiveConceptLayer(nn.Module):
    def __init__(self, D, k, S=8, uncert_theta=0.5, uncert_kappa=3.0,
                 contra_thresh=-0.1, contra_gain=6.0, birth_gap=0.55,
                 maturity_thresh=0.12, seed=0, cfg=None, softmax_free=None,
                 novelty_threshold=0.15)
```

Ключевые атрибуты:
- M: buffer (S, k) — concept prototypes
- U_s: buffer (S,) — usage
- N_s: buffer (S,) — count
- _mature: buffer — maturity flag
- W_o: Linear(S*k, D) — readout
- _read_scale, _temp: learnable params
- _write_event, _concept_id: buffers for projector

**Maturity detection**: CV = std/mean of residual variance, becomes mature when CV < 1/lambda_d for ceil(lambda_d) steps.

---

## 17. VSA Utilities

### Файл: `core/vsa_utils.py`

```python
def dct_basis(n)  — DCT-II basis (n, n)
def zeckendorf_codes(vocab=50000)  — Fibonacci Zeckendorf binary codes (V, K≈23)
def fib_sigmoid_init(n, fib_vals=None)  — Fibonacci-based sigmoid bias init
def sparse_block_codes(vocab=50000, K=32, S=6)  — Sparse block codes (V, K)
def vsa_prefix_scan(a, b, state=None)  — VSA associative parallel prefix scan
```

---

## 18. Compression

### Файл: `core/compression.py`

**FCF_CPR**
```python
class FCF_CPR:
    def compress_sd(self, sd)  — (compressed_dict, meta_dict)
    def decompress_sd(self, compressed, meta, cfg)  — full state dict
    def save_compressed(self, ckpt, save_path)  — inference-only artifact
    def load_compressed(self, load_path, cfg=None)
```

**Вспомогательные функции**:
```python
def is_removable(k)  — check if key is deterministic buffer
def is_scalar_gate(k, v=None)  — check if scalar-foldable
def quantize_tensor(t, n_bits=8)  — uniform quantization
def dequantize_tensor(indices, t_min, scale, dtype=torch.float32)
def quantize_tensor_channel(t, dim=0, n_bits=8)  — per-channel quantization
def dequantize_tensor_channel(indices, mins, scales, orig_shape, dtype=torch.float32)
def analyze_sd(sd)  — detailed analysis
```

---

## 19. Adaptation

### Файл: `core/adaptation.py`

**set_active_depth** (function):
```python
def set_active_depth(model, k)  — freeze blocks with index >= k
```

**build_optimizer** (function):
```python
def build_optimizer(model, base_lr, llrd_decay=0.9, weight_decay=0.01,
                    betas=(0.9, 0.95), lam=None)  — AdamW with LLRD
```

**DepthController**:
```python
class DepthController:
    def __init__(self, model, n_layers=None, init_k=8, unfreeze_inc=4,
                 warmup_steps=2000, k_sigma=1.0, eval_interval=1000, max_depth=None)
```
- Progressive unfreezing via val-loss plateau detection

**LRController**:
```python
class LRController:
    def __init__(self, model, optimizer, cfg, warmup=None, base_lr=None)
```
- Warmup + mirror-adaptive multiplier + recovery rewind

**FailureDetector**:
```python
class FailureDetector:
    def __init__(self, model, lr_controller, make_optimizer_fn, best_path,
                 base_lr, k_sigma=3.0, warmup=2000, recover_max=20, cooldown=50,
                 min_consecutive=3)
```
- SPC 3σ rule for CE explosion detection

**GradientClipper**:
```python
class GradientClipper:
    def __init__(self, c=0.01, eps=1e-3)
```
- Adaptive Gradient Clipping (AGC)

**LossBalancer**:
```python
class LossBalancer:
    def __init__(self, align=True, align_cap=10.0, eval_interval=1000)
```
- mode='align': PCGrad-style gradient projection
- mode='balance': dimensionless per-aux normalization

---

## 20. Lambda Utils

### Файл: `core/lambda_utils.py`

```python
def lambda_d(d: int) -> float  — positive root of x^d = x^{d-1} + ... + 1
def fib(n: int) -> int  — classical Fibonacci
def generalized_fib(n: int, d: int = 2) -> int  — d-step Fibonacci
```

**LambdaConfig**:
```python
class LambdaConfig:
    def __init__(self, d: int = 3)
```
- Все гиперпараметры выведены из λ_d
- Свойства: lam, lam_inv, lam_inv_sq, lam_inv_cu, lam_inv_4, lam_sq
- AdaptiveController thresholds: exploration_threshold, differentiation_threshold
- Memory/gate ranges: mem2v_scale, ema_alpha, noise_scale, delta_var_ema
- Learning/scheduler: warmup_steps, target_var, mag_threshold, lr_min_ratio
- Optimizer: gate_lr_mult
- Init values: log_scale_init_std, conv_init_std, w_d_init_std
- Buffer sizes: eval_interval, save_interval, log_interval, patience

**spectral_radius** (function):
```python
def spectral_radius(model, h, n_steps=20, n_iters=1)  — power iteration estimate of ρ(J)
```

---

## 21. Projector

### Файл: `core/projector.py`

```python
class Projector:
    def __init__(self, tokenizer: Tokenizer)
    def segment(write_event)  — (B, L) bool → list[(start, end)]
    def read_words(self, ids, write_event)  — decoded spans
    def concept_spans(self, concept_id, write_event)  — concept id per word
```

---

## 22. Curriculum

### Файл: `core/curriculum.py`

```python
class CurriculumTracker:
    def __init__(self, n_streams, tau_0=2.0, tau_min=0.1, decay_steps=50000, momentum=0.95)
    def to(self, device)
    @property tau  — τ_0 · exp(-t / T_decay) + τ_min
    def update(self, stream_idx, loss_val)
    def sample_probs(self, cur_step)  — p_i ∝ exp(L_i / τ)
```

---

## 23. Word Num

### Файл: `core/word_num.py`

```python
ALPHABET = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
PHI: Dict[str, int]  — буква → простое число
```

```python
def n_of(word: str) -> int  — состав слова (произведение φ)
def v_of(word: str) -> int  — порядок слова (полином)
def factors(n: int) -> str  — восстановление состава
def gcd_of(a: str, b: str) -> int
def lcm_of(a: str, b: str) -> int
def morph_sim(a: str, b: str) -> float  — морфологическая близость ∈ [0, 1]
def log_size(word: str) -> float
```

---

## 24. Live Inference

### Файл: `core/live_inference.py`

**MirrorMonitor**:
```python
class MirrorMonitor:
    def __init__(self, model: WideBindStack, max_history: int = 5000)
    def clear(self)
    def capture(self, global_state=None)  — read metrics after forward
```

**LiveInference**:
```python
class LiveInference:
    def __init__(self, model: WideBindStack, device='cpu')
    def think(self, steps=1)  — self-dialogue
    def respond(self, tokens)  — process input
    def reset(self)
```

---

## 25. Scripts

### scripts/analyze.py

Единый анализатор чекпоинтов WideBind. Методы:

- `load_ckpt(path)` — загрузка чекпоинта
- `run_static(ckpt, cfg, model, missing, unexpected)` — конфиг, per-layer параметры, VSA-лестница, зеркало, MLP, голова
- `run_inspector(model)` — сигналы (temp/pred/smooth/sym/help), trust/concept/dominance
- `run_wake(model, ckpt)` — вердикт PASS/WATCH/WAKE
- `run_live(model, cfg)` — forward на случайном входе
- `run_head(model, ckpt, args, tok)` — декомпозиция bias vs контекст + позиционная карта
- `run_anomaly(live, wake, ckpt)` — трекер аномалий
- `run_grad_info(model, ce_loss, aux)` — dead_pred + cos_sim(diversity, CE)
- `run_bridge(model, cfg)` — runtime intent-bus metrics
- `parse_training_log(path)` — парсинг логов
- `render_log_html(data, outpath)` — HTML-отчёт

---

## 26. Полный отчёт

### Все классы core/

| Класс | Файл | Описание |
|-------|------|----------|
| WideBindConfig | config.py | Конфигурация модели |
| LambdaConfig | lambda_utils.py | Иерархия из λ_d |
| TauConfig | tau_config.py | Единое τ-поле |
| PartitionedEmbedding | embedding.py | Эмбеддинг через sparse block codes |
| ZeckendorfEmbedding | embedding.py | Legacy эмбеддинг |
| LmHead | embedding.py | Legacy голова |
| PartitionedHead | embedding.py | Segment-addressed readout |
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
| WideBindBlock | block.py | Один слой модели |
| PrecisionGate | block.py | Variable precision gate |
| ExactSequenceMemory | block.py | Exact sequence memory |
| WideBindStack | stack.py | Полная модель |
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
| CollectiveConceptLayer | concept_layer.py | Concept layer |
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
| zeckendorf_codes() | vsa_utils.py | Fibonacci codes |
| sparse_block_codes() | vsa_utils.py | Sparse block codes |
| vsa_prefix_scan() | vsa_utils.py | VSA prefix scan |
| fib_sigmoid_init() | vsa_utils.py | Fibonacci sigmoid init |
| lambda_d() | lambda_utils.py | Generalized golden ratio |
| fib() | lambda_utils.py | Fibonacci |
| generalized_fib() | lambda_utils.py | d-step Fibonacci |
| spectral_radius() | lambda_utils.py | Spectral radius estimate |
| set_active_depth() | adaptation.py | Progressive unfreezing |
| build_optimizer() | adaptation.py | AdamW with LLRD |
| _memory_attention() | memory_bank.py | Hybrid memory attention |
| _scan_chunk() | block.py | Parallel chunk scan |
| _combine_chunks() | block.py | Cross-chunk scan |
| _golden_shifts() | bind.py | Golden angle shifts |
| _fibonacci_shifts() | bind.py | Fibonacci shifts |
| migrate_bind_state_dict() | bind.py | State dict migration |
| is_removable() | compression.py | Check removable key |
| is_scalar_gate() | compression.py | Check scalar-foldable |
| quantize_tensor() | compression.py | Uniform quantization |
| dequantize_tensor() | compression.py | Dequantization |
| quantize_tensor_channel() | compression.py | Per-channel quantization |
| dequantize_tensor_channel() | compression.py | Per-channel dequantization |
| analyze_sd() | compression.py | State dict analysis |

## 27. P0 Audit — найденные и исправленные баги

### P0.1: layer_bridge_gate.log_tau — мёртвые параметры

**Проблема:** Все 24 `SpectrumGate.log_tau` имели `grad_norm=0` — параметры были зарегистрированы в optimizer, но градиент не доходил. Корневая причина: гейт использовался только для diagnostics внутри `torch.no_grad()` в `compute_losses`.

**Исправление** (`stack.py:1094-1148`): Вычисление gate output перенесено **вне** `no_grad()` блока. Добавлен `lbg_diversity_loss` — энтропийная регуляция на gate weights, которая:
- Считает `softmax(gate_outputs)` по слоям
- Штрафует за collapse (негативная энтропия)
- Даёт `log_tau` дифференцируемый путь в loss

**Результат:** 24/24 ALIVE, 0 DEAD. `lbg_diversity` всегда присутствует в aux dict.

### P0.2: memory_bank.tau_config — τ-prior не применялся

**Проблема:** `StreamingMemoryBank` принимал `tau_config` в конструкторе и использовал его для вычисления priors (L1/L2/L3), но **не сохранял** как `self.tau_config`. Все три уровня имели `log_tau=0.1` (дефолт) вместо τ-differentiated priors.

**Исправление:**
1. `memory_bank.py:507`: Добавлено `self.tau_config = tau_config`
2. `stack.py:115-131`: `tau_config.update()` перенесён **перед** созданием `StreamingMemoryBank`, чтобы `mem_tau_cache` был заполнен

**Результат:** Prior-ы дифференцированы: L1=-1.18, L2=0.27, L3=1.61 (вместо.uniform -2.30 для всех).

### P0.3: _tau_dev — односторонний collapse

**Проблема:** `_tau_dev` имел `grad range: [-0.177, 0.000]` — оптимизатор толкал все значения в минус, сжимая τ-лесенку.

**Исправление** (`stack.py:1236-1243`): Добавлен `tau_dev_reg` — L2-регуляризация к нулю (uniform ladder):
```python
_tau_dev_reg = dev.pow(2).mean() * 0.01
aux_dict['tau_dev_reg'] = _tau_dev_reg
```
При `dev=0`: reg=0 (не мешает init). При отклонении: штраф в 0.01× квадрат.

**Результат:** `tau_dev_reg` присутствует в aux, `has_grad_fn=True`.

### P0.4: cfg.llrd — мёртвый параметр

**Проблема:** `cfg.llrd=0.9` существовал, но не использовался — LLRD управлялся только через `tau_llrd_gamma=0.65`.

**Исправление** (`config.py:122`): Помечен как `DEPRECATED`.

### P0.5: _tau_intent_dev — уже удалён (Phase 1)

**Статус:** ✅ Удалён в предыдущем раунде. Параметр не найден в checkpoint.

### Верификация (production config, 24 слоя, seq=32)

| Метрика | Результат |
|---------|-----------|
| layer_bridge_gate.log_tau | 24 ALIVE, 0 DEAD |
| memory_bank.tau_config | has=True, L1=-1.18, L2=0.27, L3=1.61 |
| tau_dev_reg | present, has_grad_fn=True |
| τ ladder | 9.59 → 613.48 (64x range) |
| intent_alpha | [0.70, 1.00] |
| lr_mult spread | 14.9x (γ=0.65) |
| Total params | 186.78M |
