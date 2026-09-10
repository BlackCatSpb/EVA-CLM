# EVA-CLM — Единая Вычислительная Архитектура

**Cognitive Learning Model · VSA + Cognitive Mirror + GroupedMLP + Unified τ-field.**
**D=2560 · 24 слоя · 32 эксперта · 212.3M параметров · vocab 65536 при 0 параметрах эмбеддинга.**
**Без attention-матриц, без softmax-узкого горла, без KV-cache.**

```
  ┌───────────────────────────────────────────────────────────────┐
  │              EVA-CLM · Unified Computing Architecture         │
  │   VSA-memory · Bind-crossing · Cognitive Mirror               │
  │   Unified τ-field · SemanticBridge · MemoryBank L1/L2 + UCL   │
  │   Code-space Logit Cache · VSA-driven Compression             │
  │   D=2560 · 212.3M params · 24 layers · attention-free         │
  └───────────────────────────────────────────────────────────────┘
```

> **Статус:** исследовательский альфа-проект, активный тренинг на Colab L4.
> 257 автотестов, полная трассируемость состояния обучения (см. §19–21).
> Всё, что заявлено в этом README, либо заперто тестом, либо помечено как непроверенное.

---

## Оглавление

1. [Философия](#1-философия)
2. [Числа прямо сейчас](#2-числа-прямой-сейчас)
3. [Поток данных](#3-поток-данных)
4. [Эмбеддинг](#4-эмбеддинг)
5. [EVABlock](#5-evablock)
6. [SigmoidCodedHead](#6-sigmoidcodedhead)
7. [Unified τ-field](#7-unified-τ-field)
8. [U1–U10 τ-tied улучшения](#8-u1u10-τ-tied-улучшения)
9. [Параметры и LR](#9-параметры-и-lr)
10. [AdaptiveController](#10-adaptivecontroller)
11. [MirrorLR](#11-mirrorlr)
12. [Функции потерь и телеметрия](#12-функции-потерь-и-телеметрия)
13. [Gradient Mechanics](#13-gradient-mechanics)
14. [Inference](#14-inference)
15. [Intent Bridge](#15-intent-bridge)
16. [Streaming Memory Bank + UCL](#16-streaming-memory-bank--ucl)
17. [Code-space Logit Cache](#17-code-space-logit-cache)
18. [Maturation + Triad](#18-maturation--triad)
19. [Дисциплина тренировки](#19-дисциплина-тренировки)
20. [Структура репозитория](#20-структура-репозитория)
21. [История аудита: M1–M13 и решения](#21-история-аудита-m1m13-и-решения)
22. [Честная оценка и фальсифицируемость](#22-честная-оценка-и-фальсифицируемость)
23. [Статус обучения и как запустить](#23-статус-обучения-и-как-запустить)
24. [Лицензия](#24-лицензия)

---

## 1. Философия

1. **Память — вектор, не матрица.** Знание живёт в суперпозиции VSA, а не в K×V таблице. Нет KV-cache, O(1)-контекст на шаг.
2. **Глубина не убивает градиент.** Bind — изометрия (циклический сдвиг/спираль), сохраняющая норму. 24 слоя с живой обратной связью.
3. **Словарь — код, не строка таблицы.** 65 536 токенов стоят **0 параметров**: sparse block code (K=32, S=6, C(32,6)=906 192) — детерминированный буфер, а не матрица эмбеддингов.
4. **Одна физическая величина — τ.** Все скорости, горизонты, температуры и задержки выводятся из единого τ-поля (λ-лестница по глубине). Магических чисел нет: константа либо привязана к τ/λ, либо удалена.
5. **Обучение непрерывно.** Инференс = обучение: зрелые, уверенные, новые паттерны на инференсе дописываются в концепт-стор (UCL).
6. **Измерительная честность.** Механизм считается существующим, только если его градиент доходит до целевой функции или он вызывается в реальном контуре. За этим следит автотест-детектор мёртвых параметров (§21). Телеметрия, не входящая в лосс, помечена как телеметрия.
7. **Один чекпоинт, полное состояние.** `best.pt` — единственный файл, и в нём всё для продолжения: веса, оптимизайзер, планировщик, baselines стража, курсор данных, RNG-потоки (§19).

---

## 2. Числа прямо сейчас

Измерено на реальной конфигурации ноутбука (fp32, L4 22.5GB):

| Метрика | Значение |
|---|---|
| Параметры (все, обучаемые) | **212 293 229** |
| — 24×EVABlock | 99.4M |
| — Streaming Memory Bank | 37.4M |
| — Reasoning Memory | 32.8M |
| — Logit Cache (code-space) | 30.7M |
| — UCL | 7.2M |
| — Bridge + intent | 4.7M |
| — Эмбеддинг / голова (всего) | **0.07M** |
| Словарь | 65 536 (буфер кодов, 0 параметров) |
| VRAM (seq=256, batch=1, fp32) | 12.1GB → **15.3GB** steady (Colab-пик 16.2/22.5) |
| Сквозная скорость | **~70 tok/s** (47 на старте, до разморозки глубины) |
| Корпус | 1.93B токенов (39 файлов) |
| Бюджетный цикл | 150k шагов ≈ 38.4M токенов ≈ 160–190 ч из ~307 ч Colab Pro |
| Тесты | 257 passed / 0 warnings |

---

## 3. Поток данных

```
[token ids]
     ↓
PartitionedEmbedding: sparse code → sigmoid-mix → basis ⊕ + RoPE   (0 параметров от vocab)
     ↓
UnifiedConceptLayer: глобальный концепт-стор (единственный)         (после эмбеддинга)
     ↓
Streaming Memory Bank: L1(свежее) + L2(обучаемые слоты) read+fusion
     ↓
24 × EVABlock: conv → bind → VSA-memory(4 τ-scale) → mirror → spectral → MLP
     │          + in-core SemanticBridge (per-layer probe/stream/predict)
     │          + intent bus (FRESH+carried), maturation gates, τ-лестница
     ↓
Logit Cache augment(h): 32 окна по seq, k/v write-time, gate ≈ 0
     ↓
SigmoidCodedHead: per-bit Bernoulli log-probs + factorized branch
     ↓
CE по ВСЕМ позициям + 18 aux-сигналов → LossBalancer (спектральное выравнивание)
```

Стек: `out, state, gs, res = model(h, state, step=…, tokens=…)`; потери — `model.compute_losses(out, y, h_emb=…)` (все τ-диагностики живут в `aux_dict`).

---

## 4. Эмбеддинг

`PartitionedEmbedding` (`core/embedding.py`):

1. **Sparse block code:** ровно S=6 активных бит из K=32; кодбук — детерминированный буфер (не параметр)
2. **Плотный микс:** `z = σ(codes @ M × 2)`, M — K×K ортогональная
3. **Базис:** `z ⊗ basis`, каждый сегмент 1:1 с зеркалом
4. **RoPE:** θ=1e6 (Qwen3-style), 0 параметров

Следствие: рост словаря **бесплатен** по параметрам; цена — в ёмкости кодового пространства (C(32,6)).

---

## 5. EVABlock

```
h_in
 ├─ [Pre-LN: RMSNorm]
 ├─ [Conv1d depthwise 48-tap, causal (двойной паддинг исправлен, M11)]   → A
 ├─ [Bind: D→K→D, trajectory_spiral]                                     → B
 ├─ [VSA Memory: 4 τ-масштаба, exact fp64 chunked prefix-scan (M3)]       → C
 ├─ [Cognitive Mirror: 32 эксперта, 5 сигналов + grad-mod (M5)]           → D
 ├─ [Spectral: DCT × λ_k, τ-модуляция]                                   → E
 ├─ [GroupedMLP: SwiGLU ×32, mirror-conditioned, τ-якорь (M5)]            → F
 └─ residual: h = h + (A + B + C + E + F)   (D — модуляция, не аддитив)
```

### 5.1 Bind

Режимы: `off`, `shift`, `cascade`, `spiral`, `trajectory_spiral(+manifold)`.

```
hp = RMSNorm(W_proj(h) + b)                  # D → K
θ  = exp(W_freq)·freq_scale·hp + W_phase     # вращающаяся фаза
u' = u·cosθ − v·sinθ;  v' = u·sinθ + v·cosθ
out = Σ_s (u' ⊙ v') @ W_out                  # 2·dims·K → D
```

Когерентность спиралей `|Z|_k` даёт усиление в точках скрещивания фаз.

### 5.2 VSA-память

```
decay = clamp(exp(-1/τ_s)·σ(h·w_d + b_d), 0.01, 1.0)
i_gate = softplus(h·w_i + b_i + γ·‖pred_err‖)     # γ-лестница из λ (M3)
mem_t  = decay_t ⊙ mem_{t−1} + i_gate_t·h_t
```

Чтение — гибридный гейт `σ(scores)·(1+softmax(scores/τ))`; скан — параллельный,
causal-точный (fp64 chunked scan, M3), под AMP — fp32/fp64-якоря.

### 5.3 Cognitive Mirror

G=32 экспертов, staircase-K (8/16/32 по глубине):
- сигналы: `temp, pred, smooth, sym, help` (+per-expert EMA сигналов, M4)
- ворота: `g = σ(w_gate·|Δ| + b + delta_gate + grad_mod + contra + meta_trust)`
- **usefulness** — конкурентный предсказатель, дисциплинирует эксперта
- private memory (G×k) пишется по зрелости; точное чтение — по STE-гейту precision (заблокирован и разблокирован в M11)

### 5.4 Explicit Reasoning

Петля над **скрытыми состояниями**: sigmoid-attention по буферу шагов,
tanh-гейты, адаптивная глубина, ramp `1−exp(−t/1000)`; знаменатель
консенсуса с нижним пределом 0.5 (M9) и отключённой thinking-надстройкой
(решение #1: `ThinkingTokenHead` удалён как необучаемый).

---

## 6. SigmoidCodedHead

Кодовое пространство: K=32 бит, активных S=6.

```
z_k   = ⟨h_k, readout_k⟩
z̄     = z/T + bit_bias + token_bias          # двойной bias (M1)
logprobs = Σ_акт. log σ(z̄) + Σ_неакт. log(1−σ(z̄))   по каждой позиции (M1)
branch     = factorized branch loss (живой член лосса, "branch" в логах)
```

Исправления M1: CE считался только по позиции 0 и с двойным учётом bias — теперь
полноценный per-token пер-позиционный NLL; это изменило шкалу CE (старые логи не
сравнимы, см. §23). Опционально: `CognitiveCodedHead` (mask-режим).

---

## 7. Unified τ-field

`TauConfig` — одно поле, из которого выводятся **все** τ-зависимые величины:

```
log_tau = log(tau_min) + log(tau_max/tau_min) · (lf·(1 + 0.3·dev) + 0.05)
```

| Величина | Формула | Потребитель |
|---|---|---|
| `tau_norm_l` | нормировка лог-τ | всё ниже |
| `mat_delay_l` | `T0 + (1−τ_norm)·T_delay` | wake-up слоёв |
| `gate_tau_l` | геометр. интерполяция | температуры гейтов |
| `α_l` (EMA) | `1 − exp(−1/τ_l)` классический горизонт | VSA, intent-потоки (исправлено в решениях: старая формула замораживала глубокие слои) |
| `lr_mult_l` | `(τ_l/τ_ref)^(−γ)` | τ-LLRD |
| `i_target` | `min(1, 5.83/τ)` | AdaptiveController |
| `c_eff` AGC | `c·(τ_ref/τ)^γ` | clipper, привязан к лестнице |

Управляется лестницей λ + **1 обучаемым параметром** `τ_dev` на слой.

---

## 8. U1–U10 τ-tied улучшения

| # | Улучшение | Формула | Статус |
|---|---|---|---|
| U1 | VSA-масштабы | `vsa_tau[l,s] = base[s]·(τ_l/τ_mid)` | живой |
| U2 | Reasoning budget | `K = max(1, round(K_base·mean(τ_norm)))` | живой |
| U3 | Spectral damping | `damp = cos(π·τ_norm/2)` | живой |
| U4 | Bridge injection | `inj = σ(α)·τ_norm + σ(β)·(1−τ_norm)` | живой (M2: один путь) |
| U5 | Mirror signal temp | `τ_sig = τ_min·(τ_max/τ_min)^(1−τ_norm)` | живой (wire в M6+) |
| U6 | Memory fusion scale | `scale = 0.3 + 0.7·τ_norm` | **сделан реальным в M11** (`_fusion_tau_alpha`) |
| U7 | Concept birth | `thr = σ(log_τ_birth)·(1−τ_norm·σ(log_τ_decay))` | живой |
| U8 | Intent alpha | carry `a = 1−1/τ_l`; девиация `(2σ(w_α)−1)(2τ_norm−1)` в вес bus-инъекции | **ожил в решениях #4** (был мёртв: насыщение α=1) |
| U9 | Gradient clipping | `c_eff = c·(τ_ref/τ)^γ` | живой |
| U10 | Bind frequency | `freq = fs·(τ_min/τ_max)^(τ_norm·η)` | живой |

Все τ-параметризованные константы проходят через `attach_tau`; `_vsa_tau_log`
намеренно вне оптимизатора (диагностика). Детектор мёртвых параметров следит,
чтобы ни один U-механизм снова не оказался декорацией.

---

## 9. Параметры и LR

| Группа | LR множитель | Содержит |
|---|---|---|
| embed | λ⁻² ≈ 0.296× | readout, basis-mix |
| mlp | λ⁻¹ ≈ 0.544× | MLP, bind W_proj/W_out |
| default | 1.0× | conv, norm, голова |
| mirror | λ¹ ≈ 1.839× | mirror-матрицы, α, log_scale |
| gate | λ¹ ≈ 1.839× | w_gate, b_gate, w_i, b_i |
| vsa | λ⁻² ≈ 0.296× | b_d, b_i, scale_w |

`build_optimizer` строит группы по ключу `(role, depth, wd)`; `MirrorLRScheduler`
позиционно индексирует `orig_lrs` и при несовпадении числа групп с чекпоинтом
берёт свежий снимок (M12) — старые чекпоинты безопасны при смене состава групп.

---

## 10. AdaptiveController

Из статистик зеркал выводит фактические управляющие параметры слоя:
`b_d` (окно), `b_i` (запись, `i_target=min(1, 5.83/τ)`), `w_mem2v`, `ema_alpha`,
`noise_scale`, `tanh_bias_mod`, `spectral_mod`, `pred_scale_mod`. Все пороги —
через τ/λ (аудит: γ-инициализация из лестницы, M3; стриминговый LossBalancer, M4).

---

## 11. MirrorLR

Counter-cyclic планировщик:

```
mult = (var_ratio · α_ratio · gate_ratio)^(1/3) · mag_factor · loss_lr_factor
```

Рост зеркального компонента → LR вниз, спад → вверх; `loss_lr_factor`:
улучшение ×1.05 / регресс ×0.5. После rollback — ре-бустрап (M8), ретроспектива
`orig_lrs` защищена (см. §9).

---

## 12. Функции потерь и телеметрия

Классификация честная: **лосс** — то, что входит в градиент через вес;
**телеметрия** — наблюдаемые сигналы (страж, логи), не имеющие веса.

| Имя | Роль | Суть |
|---|---|---|
| ce | лосс, вес 1.0 | per-bit Bernoulli NLL по всем позициям (M1) |
| branch | лосс | factorized-ветвь код-декомпозиции (M1) |
| pred, gate_l1, reinforce, balance, diversity, div, decorr, bridge_conn, gate_repulse, w_m2v, intent_tau, nuc, alpha_novelty, signal_ent, gradalign (bypass), mb_scale | лоссы-aux (веса τ-привязаны/из λ) | см. `core/losses.py` |
| layer_gate_*, lbg_*, ig_eff, mlp_out, usef, mat | телеметрия | кормит FailureDetector и логи |

Aux-потери возвращаются **сырыми**; балансировка — LossBalancer (режим align:
спектральная проекция `g_aux ≤ ‖g_CE‖`; режим balance — EMA-нормировка).
`gradalign` — bypass-член (прямой backward после align-прохода, M5).

---

## 13. Gradient Mechanics

1. Forward → `ce_loss, aux_dict` (включая τ-диагностику в `_cached_losses`)
2. CE-градиенты: `torch.autograd.grad` (без `.grad`-копий до align)
3. LossBalancer: косинус-гейт aux-проек onto CE; clone-обязательность (AGC мутирует in-place)
4. AGC: `‖g‖ > c·(τ_ref/τ)^γ·‖θ‖` → clip; zero-init параметры пропускаются
5. Оптимизайзер: три режима A/B — `adamw` (эталон, bit-identical torch), `eva`
   (медленные EMA-статы off-by-default, convex u-space), `eva_proj` (+AdamP
   null-band δ=2/√D, cautious floor; τ-привязки через `attach_tau`)
6. Watchdog (см. §19) → при дивергенции: откат + **свежий Adam** + LR-rewind

---

## 14. Inference

```python
from core import LiveInference
live = LiveInference(model, cfg)
tokens = live.generate(prompt_ids, 100, think_steps=0)
```

- Memory Bank: O(1) на шаг (векторы), не матрицы.
- **Inference = обучение (решение #5):** UCL-запись на инференсе не привязана к
  `model.train()` — она привязана к зрелости/уверенности/новизне; сгенерированные
  токены передаются в следующий шаг как `tokens=` (консолидация опыта).
- `generate` фикс: loopback-ошибка (M10) устранена; переменная точность — latch-поведение.

---

## 15. Intent Bridge

Перетекающий per-head поток намерений (`cfg.intent_bridge=True`):

- **Bottom-up:** `intent_probe(D→G·K)` — живой (подключён в решениях)
- **Bus:** `fresh_i + a·carried_i`, carry-коэффициент `a = 1−1/τ_l` (классический
  EMA-горизонт; старая v2-формула `1−exp(−τ/τ_min)` равнялась 1.0 на глубоких
  слоях — потоки были **заморожены на zero-init навсегда**; исправлено)
- **U8-девиация:** `(2σ(−1))·(2τ_norm−1)` доезжает до веса bus-инъекции —
  пер-экспертная τ-специализация реальна и обучается
- **Top-down:** `gate_logits += salience·w_sal` (zero-init; обучается через CE)

Два неограниченных канала: VSA-память (размытое общее) + intent-поток (нить внимания).

---

## 16. Streaming Memory Bank + UCL

Один концепт-стор вместо двух систем (решение #2): **L3-концепты банка удалены**
(дублировали UCL и не имели читаемого пути), fusion 4D→3D.

| Уровень | Назначение | Слоты |
|---|---|---|
| **L1** | Immediate (последние 3 предложения) | 3 |
| **L2** | Learned short-term | 32 |
| **UCL (глоб.)** | Emergent concepts после эмбеддинга; write по зрелости/новизне; novelty-gap обучаем (M6) | S=8 |

Чтение: sigmoid-attention + fusion (τ-масштаб U6 реален, M11). Счётчики
`mb_l3_*` в логах теперь читают UCL. Overhead: 37.4M + 7.2M параметров.

---

## 17. Code-space Logit Cache

**Дизайн (после решения #3 и follow-up-фикса L4-OOM):** кэш — это память о
предыдущих окнах, спроецированная в собственное код-пространство модели.

```
TRAINING (каждый шаг):
  h (B,L,D) ──store──> _h_cache (detached)      ≤32 окна
       │
       ├─ k_new = k_norm(k_proj_h(h))   ← единственная тяжёлая проекция за шаг (O(L))
       ├─ v_new = v_norm(v_proj_h(h))        (записанные окна хранят СВОИ k/v —
       │                                      память как записанный код)
       └─ attention(q(h) → [кэш k/v | k_new live]) → gate(≈4.5e-5) → h + ε·out
INFERENCE:
  logits → bit_profile = tanh(logits/10) @ codesᵀ / S  (V→K, 0 параметров)
         → sparse top-k компрессия (vsa_scales управляют k)  → ~43x экономия против KV-cache
```

Ключевые свойства:
- **Все обучаемые карты — K×D или D×D, ни одной V×D.** Старая версия несла
  3×65536×2560 ≈ **503M мёртвых параметров** (и это была половина «723M модели»).
- **Инициализация = тождество:** gate = MLP с нулевыми весами и bias −10 ⇒
  `σ(−10) ≈ 4.5e-5`; кэш начинает влиять только когда CE научится его спрашивать.
- **Детач при записи + live newest:** никаких «backward through graph a second time».
- **Окна:** `max_entries=32` ⇒ контекст кэша 32×256=8192 позиции; `pos_enc` — по модулю.
- **Границы очистки:** ротация документа, eval (до/после — кэш-лист не буфер,
  snapshot его не изолирует), rollback/resume.
- **R1 scheduled sampling** (5% шагов в inference-режиме) — через `process_with_cache`
  для вызывающих с логитами; **R6** — инвалидация на resume/LR-reset.
- Таблица экономии 43x/окно-attention в inference — унаследованные измерения
  старого режима; **для нового code-space пути подлежат переизмерению** (честно).

---

## 18. Maturation + Triad

```
gate_l(t) = σ((t − (T0 + α·(1−τ_norm)·T_delay))/Δ)
M_l(t)    = max(gate_l(t), bridge_readiness)
```

- Готовность — по компетентности моста, не по часам; глубокие слои открываются первыми.
- Frozen MLP-бейзлайн ≈0.667 всегда открыт → сигнал обучения не пропадает.
- **Triad** (`triad_reason`): при `conf < 0.5` ствол прогоняется повторно,
  консервативный бленд 0.5/0.5 — inference-only, 0 новых параметров.

---

## 19. Дисциплина тренировки

Инфраструктура, которая делает длинный прогон на арендованном GPU доверенным:

**Единственный чекпоинт `best.pt` (M12)** — пишется при улучшении val и при
чистом прерывании; содержит: model, optimizer+param_names (name-based
восстановление переживает смену архитектуры), scheduler (с val-EMA/LS-состоянием),
`best_val_loss`, `active_depth`, **state failure-детектора** (5 базлайнов,
серии, `ce_armed`, `recover_count`), состояние балансировщика, **курсор данных
(`stream_idx`, `offset`)** и обе RNG-ветки. `rollback.pt` выпилен: откат идёт к
самому `best.pt`; свежий прогон сеет step-0-заметку для дивергенции до первого eval.

**Val-честность (M8+M13):** hold-out = **последние 3 файла корпуса** (файловый
уровень, никогда не обучаем; при <8 файлов — семантика M8), среднее по 3×33
батчам из ¾-области каждого файла; hidden-state заново на батч ⇒ per-token CE.
Runtime-буферы (банки, bus, EMA, reasoning) снэпшотятся/восстанавливаются вокруг
eval — валидационные документы не отравляют рабочую память тренировки; кэш чистится.

**Страж (FailureDetector):** относительное правило `value > slow_EMA·(1+margin)`
+ healthy-band floor по каждому из `ce, diversity, gate_l1, mlp_ratio, ig_eff`;
CE вооружается после первого eval (известный ранний транзиент 66→11 не трогает
LR); не-конечный CE = откат безусловно; после отката — ре-бустрап базлайнов
(иначе «rollback каждые 50 шагов» — сценарий, который ловили в живых логах);
`recover_max=20`, cooldown, мин. 3 последовательных нарушения.

**Данные:** документ-ротация между файлами пула (фикс M8: `offset==0` больше не
мёртвая проверка), wrap = новая документ-сессия (сброс state/intent/bridge/cache,
узел `wrapped`). При смене схемы holdout курсор из старого чекпоинта клампится.

**Бюджетная арифметика:** см. §2/§23 — длина прогона = функция GPU-бюджета,
прогон разрежим на циклы без потери состояния.

---

## 20. Структура репозитория

```
EVA-CLM/
├── core/                       # ~27 модулей — вся модель
│   ├── config.py               # EVAConfig (dataclass; __post_init__ — λ-дома)
│   ├── tau_config.py           # TauConfig: единое τ-поле
│   ├── embedding.py            # PartitionedEmbedding, SigmoidCodedHead, CognitiveCodedHead
│   ├── block.py                # EVABlock: conv/bind/VSA/mirror/spectral/MLP
│   ├── bind.py  mirror.py  mlp.py  bridge.py  adaptive_gate.py
│   ├── concept_layer.py        # UnifiedConceptLayer (единственный концепт-стор)
│   ├── memory_bank.py          # StreamingMemoryBank L1+L2+fusion(3D)
│   ├── logit_cache.py          # code-space кэш: store/write-time kv/augment
│   ├── tau_compression.py  variable_precision.py  compression.py
│   ├── reasoning.py            # скрыто-состоянийные петли
│   ├── stack.py                # EVAStack: forward, compute_losses, snapshots, reset_cache
│   ├── losses.py               # compute_losses → (ce, aux_dict)
│   ├── adaptation.py           # LossBalancer/AGC/FailureDetector (реэкспорт)
│   ├── training_control.py     # …фактическая реализация стража/балансировщика
│   ├── lr_scheduler.py  optimizer.py   # MirrorLR; EVAAdamW/A (adamw-эталон проверен)
│   ├── live_inference.py       # generate + inference-обучение
│   ├── migrate.py              # state_dict-морфинг старых чекпоинтов (явный, не silent)
│   └── stream.py  data.py  …
├── notebooks/eva_colab.ipynb   # канон: 24L/D2560/L4, клетки 1–10 прогнаны E2E-харнессом
├── scripts/                    # train.py (зеркало ноута), analyze.py, generate.py, …
├── tests/                      # 257 тестов, вкл. детектор мёртвых параметров
├── checkponts/  wb/            # чекпоинты; BPE-токенизатор/потоки
└── README.md
```

---

## 21. История аудита: M1–M13 и решения

Пятиагентный глубокий аудит (5×«архитектор против кода») + живой лог Colab
давали находки каждая в свой milestone — всё заперто тестами:

| M | Суть находки → фикс |
|---|---|
| Adam | режим `adamw` bit-identical torch (100 шагов maxdiff 0); slow_ema по умолчанию off; AdamP null-band; τ-константы через `attach_tau` |
| M1 | голова: CE только по pos-0 + двойной bias → per-position + bit/token bias + factorized branch + маски |
| M2 | bridge-гейт M²→M; один путь инъекции; диагностика живая |
| M3 | VSA-скан: exact fp64 chunked scan, causal prefix_mean, γ-инициал из лестницы |
| M4 | per-expert сигнальные EMA; grad-mod; √k-кап скипов; governor parity с eval |
| M5 | `pred/w_m2v/intent_tau` были декорациями → подключены; gradalign bypass; MLP-якорь 1.5·σ |
| M6 | UCL: differentiable write (index_copy) + обучаемый novelty gap |
| M7 | точнотокенный routing (_VSA/_GATE/_MIRROR части); λ-лестница амплитуд dev_max tanh-bound |
| M8 | документ-ротация была мёртвой (`offset==0` никогда не срабатывал повторно); NaN-откат; ре-бустрап стража; **eval-изоляция буферов**; троттлинг записей |
| M9 | знаменатель консенсуса reasoning с floor 0.5; центрированная заметка U8; `python -m core.stack` |
| M10 | tau_compression устройства/vocab-гарды; строгий registry снятых модулей; verify-decompress; честность migrate |
| M11 | conv двойной паддинг; **precision-STE deadlock** (гейт не открывался никогда) → починен; U6 сделан реальным; **детектор мёртвых параметров** |
| M12 | единый `best.pt` = полное состояние (детектор/balancer/курсор/RNG); `weights_only`-фикс отката; `recover_count` перестал теряться |
| M13 | holdout 1 файл → 3 файла (val-история перестала быть доменной жеребьёвкой) |

**Решения владельца (D1–D5):** удалить thinking-токены; слить концепт-системы в
одну (UCL); включить logit cache (в переделанном code-space виде); сделать
`_w_alpha_expert` обучаемым (вскрыло заморозку intent-потоков — исправлено);
inference = обучение. Каждый пункт — с регрессионными локами; итог: **212.3M
реальных параметров вместо «723M» с 503M мёртвых**, 257 тестов, 0 предупреждений.

---

## 22. Честная оценка и фальсифицируемость

Архитектура не «лучшая по определению». Проверено/непроверено:

**Проверено инструментами:** связность τ-поля и U-механизмов (детектор + тесты),
численная честность (fp64-скан, bit-identical Adam-эталон), воспроизводимость
состояния (M12-резюме, E2E-прогон реальных клеток ноутбука + второй проход resume),
работоспособность на L4 (15.3GB / ~70 tok/s / нет ложных откатов на 1300+ шагов).

**Не проверено (это и есть ставка):**
1. **Бьёт ли baseline.** Ни один прогон пока не сравнивался с vanilla-трансформером
   равных параметров/шагов/данных. Это единственный вопрос, на который отвечает
   experiment, а не README.
2. **Нужен ли каждый U/aux-механизм.** kill-test-матрица (обнулить → измерить) в плане.
3. **Длинный контекст памяти** (retrieval-демо через bank/cache за пределами seq).

**Лифт фальсификации:**
- **Чекпоинт 1** (этот цикл, ~38M токенов): val-кривая на 3-файловом holdout;
  затем matched-vanilla при том же бюджете → EVA обязан выигрывать на токен, или тезис шатается.
- **Чекпоинт 2:** ablation τ-лестницы / coded head / memory — по одному на цикл.
- **Чекпоинт 3:** что-то, недостижимое baseline на этом бюджете (in-context retrieval
  за длину окна) — настоящий консервный выигрыш.

Если после чекпоинта 1 разрыв нулевой — это первый честный результат, а не провал:
врать в логах больше нечему.

---

## 23. Статус обучения и как запустить

**Текущий цикл** (перезапуск после M1–M13): шаг 0 → **150 000**, seq=256, fp32,
`optimizer='eva_proj'` (A2-рука), L4. Прежние цифры CE/val **несопоставимы** с
этой историей: изменилась шкала CE (M1: per-position кодовая NLL, не softmax-по-позиции-0)
и определение val (M8/M13: 3-файловый holdout). CE кодовой головы идёт в своей шкале —
не сравнивай его с ln(vocab) и со старыми логами; смотри на монотонность и на то,
что `ce` и `branch` затухают синхронно.

**Colab (канон):**
1. Загрузить `notebooks/eva_colab.ipynb` из репо, `FORCE_FRESH=True` только для чистого старта;
   далее любой рестарт — `False` (полное состояние в best.pt, M12).
2. Данные: `token_stream_*_clean.bin` на Drive; 3 последних по алфавиту файла = holdout (§19).
3. Клетки 1→10 по порядку; клетка 3 проверяет GPU, клетка 8 сама мигрирует старые чекпоинты
   (`core/migrate.py`, список операций — в выводе).
4. Голодные до памяти: `logit_cache_max_entries`, `seq_len`, `mlp_groups` — три первых рычага.

**train.py** — CLI-зеркало той же дисциплины (тот же payload/holdout/страж).

---

## 24. Лицензия

**Экспериментально.** Архитектура активно развивается; API/конфиги меняются.
Миграция state_dict — явная: `core/migrate.py::migrate_state_dict` (списывает/морфит
ключи поимённо, никогда молча).

> EVA-CLM — исследовательский проект неаттенционной языковой архитектуры:
> VSA, биллинейное связывание, кодовая голова, самоорганизующиеся регуляторы на τ-лестнице.

*Замечания, вопросы и PR — приветствуются. Особо — предложения kill-tests.*
