# Математический анализ EVA-CLM

**Источник:** `https://github.com/BlackCatSpb/EVA-CLM`  
**Локальная копия:** `C:\Users\black\OneDrive\Desktop\EVA CLM`  
**Важное уточнение:** в `C:\Users\black\OneDrive\Desktop\WideBind` находится отдельный репозиторий `widebind.git` — более ранняя/другая версия. Настоящий анализ выполнен по актуальному EVA-CLM.

**Дата анализа:** 2026-09-13  
**Метод:** параллельный анализ 4 специализированными агентами + верификация ключевых утверждений из первых рук.

---

## Содержание

1. [Общий математический вердикт](#1-общий-математический-вердикт)
2. [Единое τ-поле](#2-единое-τ-поле)
3. [VSA-память и Bind](#3-vsa-память-и-bind)
4. [Sparse Block Codes и SigmoidCodedHead](#4-sparse-block-codes-и-sigmoidcodedhead)
5. [Cognitive Mirror, MoE и спектральная ветвь](#5-cognitive-mirror-moe-и-спектральная-ветвь)
6. [LogitCache](#6-logitcache)
7. [Обучение и оптимизация](#7-обучение-и-оптимизация)
8. [Найденные несоответствия и ошибки](#8-найденные-несоответствия-и-ошибки)
9. [Рекомендации по исправлению](#9-рекомендации-по-исправлению)
10. [Итоговая оценка](#10-итоговая-оценка)

---

## 1. Общий математический вердикт

EVA-CLM — это **продвинутая исследовательская не-трансформерная языковая модель**, построенная вокруг трёх идей:

1. **Кодовое представление слов** — sparse block codes вместо классических эмбеддингов.
2. **VSA-память** — суперпозиция состояний вместо KV-cache.
3. **Единое τ-поле** — все временные константы архитектуры выводятся из одной лестницы временных масштабов.

По сравнению с версией `widebind.git`, в EVA-CLM исправлено множество математических и инженерных проблем (audit fixes B1-B12, M1-M7, M26, M34, M37-M38). В частности:
- Реализован **EVAAdamW** с cautious mask, AdamP-проекцией, slow EMA и trust.
- **τ-лестница** нормализована и корректно укладывается в `tau_max`.
- **Intent-шина** использует `1 - 1/τ` вместо насыщающейся экспоненты.
- **Матuration** переписана с log-time reparameterization.
- **Losses** переработаны: stable rank для nuc, корреляционная матрица для diversity, исправлены dead gradients.
- **VSA prefix scan** векторизован с правильным memory layout.

Тем не менее, **ряд декларируемых в README свойств не воспроизводится в коде в чистом виде**, и стабильность во многом всё ещё опирается на обширную систему aux-потерь и регуляризаторов.

---

## 2. Единое τ-поле

### 2.1 Конструкция τ-лестницы

Реальный код (`core/tau_config.py`) использует **кумулятивную сумму положительных приращений с нормализацией**:

```
Δ0 = ln(τ_max/τ_min) / (n-1)
dev_eff = dev_max · tanh(_tau_dev / dev_max) ∈ [-dev_max, +dev_max]
inc_l = Δ0 · softplus(dev_eff) / ln(2)
cs_l = cumsum(inc)_l
ln τ_l = ln τ_min + ln(τ_max/τ_min) · (cs_l / cs_{n-1})
τ_l = exp(ln τ_l)
```

**Ключевое следствие:** нормализация `cs / cs[-1]` жёстко фиксирует правый конец:

```
τ_{n-1} = τ_max = 512
```

Но левый конец при нулевых отклонениях сдвигается вверх:

```
τ_0 = τ_min · exp(ln(τ_max/τ_min) / n) ≈ 8 · e^(4.159/24) ≈ 9.51
```

Эффективный размах при инициализации — ~53.8×, а не декларируемые 64×.

### 2.2 Выводимые величины

| Величина | Формула | Назначение |
|---|---|---|
| `tau_norm` | `(ln τ_l - ln τ_min) / (ln τ_max - ln τ_min)` ∈ [0,1] | Нормализованный τ |
| `mat_delay` | `T_0 + (1 - tau_norm) · T_delay` | Задержка зрелости |
| `gate_tau` | `exp(ln τ_max^g + (ln τ_min^g - ln τ_max^g) · mat_gate)` | Температура гейтов |
| `intent_alpha` | `1 - 1/τ_l` | EMA интента |
| `lr_mult` | `(τ_l / τ_ref)^(-γ)` | LLRD-множитель |
| `mem_tau` | percentiles(τ_l): 17%, 50%, 83% | L1/L2/L3 температуры |

### 2.3 intent_alpha: 1 - 1/τ

EVA-CLM использует:

```
α_l = 1 - 1/τ_l,  τ_l ≥ 2
```

- При `τ_min=8`: `α_0 ≈ 0.875`
- При `τ_max=512`: `α_23 ≈ 0.998`

Это **исправляет критический баг** WideBind, где `1 - exp(-τ_l/τ_min)` давала exact 1.0 в fp32 уже при `τ_l ≥ 64`, убивая градиент fresh intent в глубоких слоях.

Недостаток: вся шина почти всегда несёт старый сигнал; поверхностные слои интегрируют лишь ~12% нового содержимого за шаг.

### 2.4 Maturation

Код (`core/maturation.py`):

```
T_eff,l = T_0 + α(1 - tau_norm_l) · T_delay
M_l(t) = σ((ln t - ln T_eff,l) · (T_eff,l / Δ_t))
```

**Log-time reparameterization** сохраняет локальный наклон при `t ≈ T_eff`, но градиент по времени не умирает при больших `t`.

**Расхождение с README:** README §15 и docstring `config.py` утверждают:

```
M_l(t) = max(time/τ-ramp, bridge_readiness)
```

**В коде** `MaturationController.step_gate()` использует **только time ramp**. `readiness` вычисляется от `pred_err` зеркала, но в `stack.py` финальный гейт может складываться как `max(time_gate, pred_err_readiness)`. При этом:
- `matur_bridge_readiness=True` в конфиге не влияет на `MaturationController`.
- `bridge.readiness()` применяется только к группировке параметров оптимизатора (`eva_optim.py`).
- README обещает `bridge_readiness`, код использует `pred_err` зеркала.

### 2.5 Параллельные τ-подсистемы

Несмотря на декларацию «единого поля», остались независимые τ-константы:

1. **`stack._vsa_log_param`** — 4 параметра для базовых масштабов VSA-памяти:
   ```
   τ_s(i,l) = (exp(cumsum(softplus(_vsa_log_param))) + 1) · (τ_l / τ_mid)
   ```

2. **`mirror_tau_min=2.0`, `mirror_tau_max=200.0`** — независимая лестница для инициализации per-K α в зеркале:
   ```
   τ_k = 2 · (200/2)^(k/(K-1))
   ```

3. **`c_ema = τ_mid / √D`** для global_state — смесь τ и размерности.

### 2.6 Риски τ-поля

1. **Чувствительность к `tau_min/tau_max`**: изменение любого конца сдвигает всю геометрию.
2. **Collapse `_tau_dev`**: все `dev_l` могут сдвинуться к одному знаку, сжав/растянув лестницу. `tau_dev_reg=0.01` — слабый якорь.
3. **Log-time maturation + LR scheduler**: рампа открывается поздно (8000-16000 шагов), в то время как warmup заканчивается на 1000. MirrorLR может войти в режим затухания раньше пробуждения слоёв.
4. **Несоответствие документации**: README описывает `bridge_readiness`, код использует time ramp / pred_err.

---

## 3. VSA-память и Bind

### 3.1 Bind

`TrajectorySpiralBind`:

```
hp = RMSNorm(W_proj · h + b) ∈ R^(B×L×K)
θ_s,d = exp(W_freq[s,d]) · freq_eff · hp + W_phase[s,d]
freq_eff = freq_scale · (τ_min/τ_l)^η,  η = σ(_eta)
```

Комплексное вращение:

```
(v'_re, v'_im) = (cos θ · v_re - sin θ · v_im, sin θ · v_re + cos θ · v_im)
```

Умножение:

```
prod = (u_re · v'_re - u_im · v'_im) + i(u_re · v'_im + u_im · v'_re)
```

Гибрид HRR/outer-product:

```
α = α_min + (α_max - α_min)(1 - τ_norm)
hybrid = α · HRR(a,b) + (1-α) · (a ⊙ b)
```

Поверхностные слои — outer-product, глубокие — циркулянтная HRR-свёртка.

**Утверждение README о нормосохранении верно только для изолированного поворота.** Полная операция включает обучаемые проекции и поэлементное умножение без условия ортогональности. Например, для `BottleneckBind`:

```
bind(h) = ((hp ⊙ w_u) ⊙ (hp ⊙ w_v)) · W_out
```

Норма выхода масштабируется как `‖W_out‖ · ‖hp‖²`, то есть зависит от входа и параметров.

### 3.2 VSA prefix scan

Рекуррентная память:

```
m_t = a_t ⊙ m_{t-1} + b_t
a_t = clamp(d_s · d_mod, d_s^k, 1)
d_s = exp(-1/τ_s)
d_mod = σ(h·w_d + b_d) / σ(b_d) ≤ 1
```

Chunked scan (`floor_log`, tail-referenced):

```
L_t = Σ log a_i
u_t = b_t · exp(A - L_t),  A = L_T
m_t = exp(L_t - A) · Σ u_i
```

Это алгебраически эквивалентно `exp(L_t) · Σ b_i · exp(-L_i)`, но избегает `1/cd ~ 10^25` и NaN в backward.

**M37/M38 vectorization fix:** `_scan_chunks` векторизует все чанки одним вызовом `(B·n_c, chunk, S, D)` с cumsum по оси chunk (contiguous layout). Ранее cumsum по strided axis `dim=2` был медленнее Python-цикла.

### 3.3 Ёмкость и SNR

Для одного масштаба с постоянным затуханием:

```
m_L = i · Σ d^j · h_{L-j}
N_eff ≈ 2τ
SNR ≈ D / N_eff ≈ D / (2τ)
```

| D | τ=8 | τ=128 | τ=512 |
|---|---|---|---|
| 2560 | 160 (22 dB) | 10 (10 dB) | 2.5 (4 dB) |
| 4096 | 256 (24 dB) | 16 (12 dB) | 4 (6 dB) |

Это идеальная оценка для ортогональных кодов; реальные эмбеддинги коррелированы.

---

## 4. Sparse Block Codes и SigmoidCodedHead

### 4.1 Кодовое пространство

Каждое слово кодируется вектором `c ∈ {0,1}^K` с ровно `S` единицами:

```
|C| = C(K,S)
```

Дефолт: `K=32, S=6` → `C(32,6) = 906 192 ≥ V=65 536`.

`codebook='twin_free'` требует `K=64` (иначе падает), даёт `C(64,6) = 74 974 368`.

### 4.2 Embedding

```
c ∈ {0,1}^(V×K)
z = σ(2 · c · M) ∈ (0,1)^(V×K)
e = z ⊗ B,  B ∈ R^(K×d), d = D/K
```

`embed_center=True` вычитает бегущее среднее `z`, убирая общую DC-составляющую.

### 4.3 Голова

```
z_k = <h_k, readout_k>
zt_k = z_k / T_k + b_k^bit + bus_bias_k
u, base = hybrid_gate(zt)
logits = u · C^T + base + b^tok
```

`bus_bias` приходит из `bus_head_proj(intent_bus)` и добавляется до нормализации.

Побитовая Бернулли-CE:

```
L = Σ_{k∈w*} -log σ(u_k) + Σ_{k∉w*} -log(1-σ(u_k))
```

### 4.4 Информационная ёмкость

```
H_max = log2 C(32,6) ≈ 19.79 бит
H(vocab=65k) = 16 бит
```

Параметров головы почти нет: `token_bias` (V) + разделяемый `readout`/`basis` (K×d).

---

## 5. Cognitive Mirror, MoE и спектральная ветвь

### 5.1 Cognitive Mirror

Дефолт конфликтует с README:
- `config.py`: `D=4096, n_layers=32` → `d = D/G = 128`
- README: `D=2560, n_layers=24` → `d = 80`

Пять сигналов:
- `temp` — отклонение от центроида памяти
- `pred` — ошибка самопредсказания
- `smooth` — локальная когерентность
- `sym` — билинейная временная симметрия
- `help` — приватная память

Gate:

```
g_e = σ(<|pred_error|, w_gate> + b_gate + gate_bias + <δ, w_δ> + grad_mod + dvar_mod + intent + contra - meta_trust)
```

Private memory `pm ∈ R^(G×k)`; запись только при зрелости.

### 5.2 MLP + BridgeGLU

```
gate = SiLU(h_g · W_gate) ⊙ (mlp_gate_a + mlp_gate_b · mirror_gate)
up = h_g · W_up
out = (gate ⊙ up) · W_down
```

BridgeGLU:

```
glu = σ(W_g · δ) ⊙ σ(W_v · δ) ⊙ σ(log_gain)
mlp_mod = base · 1.5σ(mod_scale_mlp) · (1 + β_glu(2·glu - 1)) · M_l
```

### 5.3 Спектральная ветвь

DCT-II базис `V` ортогонален: `VV^T = I`.

```
h_out = h + h · V^T · diag(λ_k · s_mod) · V
```

`λ_k` — обучаемые частотные масштабы; `s_mod` — модуляция от AdaptiveController.

---

## 6. LogitCache

### 6.1 Dual-mode

**Training:** хранит `h` (detached) + write-time `k,v`.
- Память: ~30 KB/token
- Стоимость шага: `O(L)` проекций текущего окна + `O(L·M)` скалярных произведений

**Inference:** хранит сжатые logits.
- Память: ~1.1 KB/token → 1M tokens = 1.1 GB (vs 240 GB KV-cache)

### 6.2 Компрессия

```
topk_vals, topk_idx = topk(logits, k)
scale = (max - min) / 255
idx_vals = uint8(round((topk_vals - min) / scale))
result[topk_idx] = idx_vals · scale + min
result[others] = min - 2
```

`k` управляется `vsa_scales`:

```
base_k = (128+96+80+64)/4 = 92
k = max(8, base_k · mean(sigmoid(vsa_scales)))
```

При инициализации `k ≈ 46` из 65536 слов (~0.07% словаря).

### 6.3 Риски

- Потеря хвоста: сохраняется только top-k.
- uint8 квантизация даёт ошибку до ±scale/2.
- Fill value `min-2` вносит смещение в хвост.
- Scheduled sampling 5% лишь частично сшивает train/inference представления.

---

## 7. Обучение и оптимизация

### 7.1 EVAAdamW

Реализован в `core/eva_optim.py`.

Update:

```
m ← b1·m + (1-b1)·g
v ← b2·v + (1-b2)·g²
u = m / (sqrt(v/bc2) + ε)
```

Модификаторы:
- **AdamP** (`projected_wd`): row-wise Gram-Schmidt при `|cos| ≥ 2/√fan_in`.
- **Slow EMA** (`slow_ema`): `u ← (u + c·m_s_hat) / (1+c)` — выпуклая комбинация.
- **Cautious**: `u ← u ⊙ [u⊙g > 0]`, ренормировка через `√(mean(mask))`.
- **Trust**: `u ← (floor + (1-floor)·trust) · u`.
- **Update cap** для τ-параметров: `cap = dev_max / (delta_t · lr)`.

`mode='adamw'` идентичен не-fused `torch.optim.AdamW`.

### 7.2 Ролевые группы

| Роль | Параметры | WD | Trust |
|---|---|---|---|
| `tau` | `tau_config.*`, `_tau_l_dev` | off | — |
| `scale_inv` | `*_log_*`, `log_temp/tau/gain/scale`, `embed_mix`, `bit_bias` | off | — |
| `zero_init` | `intent_probe.*`, `bus_head_proj.*` | off | \|readiness\| |
| `bridge` | `layers.*.mirror.bridge_glu_net.*` | on | bridge.readiness() |
| `mem` | `memory_bank.*` | on | \|readiness\| |
| `matrix` | остальные dim≥2 | on | — |
| `scalar` | остальные dim<2 | off | — |

### 7.3 MirrorLR

```
r_var = var(log_scale) / EMA[var]
m_var = clip(1/r_var, 0.5, 2.0)
mirror_mult = (m_var · m_alpha · m_gate)^(1/3) · m_mag
```

Val-gate: `mirror_mult > 1` разрешено только при `_val_improving`.

Val-damping:
- regression (`val > best·1.05`): `factor *= 0.5`
- improvement (`val < best·0.98`): `factor = 1.0`
- plateau: линейный warm-restart `factor += 1/τ_damp`

### 7.4 LossBalancer

`mode='align'`:

```
cos = <g_CE, g_aux> / (||g_CE|| · ||g_aux||)
s = max(0, cos) · ||g_CE|| / ||g_aux||
b = g_aux ⊙ [g_CE ⊙ g_aux > 0]  # per-coordinate sign mask
s_p = min(1, ||g_CE|| / ||b||)
g_final = g_CE + s_p · b
```

`gradalign` bypass'ит alignment и добавляется отдельно (backward hook на `h_mlp`).

### 7.5 Полная функция потерь

`losses.py` возвращает ~20 aux-потерь:

| Имя | Описание |
|---|---|
| `ce` | Bernoulli-NLL кодовой головы |
| `pred` | Live per-layer self-prediction |
| `gate_l1` | L1 гейтов зеркала |
| `reinforce` | MSE(usefulness, gate) |
| `balance` | Нормализованный HHI |
| `diversity` | Корреляционная матрица норм групп MLP |
| `nuc` | Stable-rank penalty для Bind W_proj |
| `orth` | MSE(W^T·W, I) |
| `w_m2v` | Иерархия w_mem2v по τ |
| `intent_tau` | Регуляризация intent_alpha |
| `branch` | Равенство log-variance ветвей |
| `signal_ent` | Максимизация энтропии весов сигналов |
| `gradalign` | Выравнивание mlp_mod с ‖∂CE/∂mlp_out‖ |
| `ls_reg` | L2 на log_scale > 2.3 |
| `div` | Дивергенция sigmoid(log_scale) |
| `gate_repulse` | Максимизация энтропии usage |
| `alpha_novelty` | Разнообразие α_diag |
| `decorr` | Кэшированная декорреляция зеркала |
| `lbg_diversity` | Энтропия layer-bridge gates |
| `mem_tau_reg` | Weight-decay log_tau |
| `bridge_conn` | Предсказание эмбеддинга next token |
| `tau_dev_reg` | L2 на _tau_dev |

### 7.6 Training_control

- `apply_tau_lr`: per-layer `lr_mult` после AGC.
- `FailureDetector`: смесь relative-outlier test (`value > slow_ema·(1+margin)`) и Page-Hinkley walk для дрейфов.
- `mirror_hyperparams`: τ-aware per-layer gains для AdaptiveController.

---

## 8. Найденные несоответствия и ошибки

### 8.1 Таблица несоответствий

| № | Утверждение README/ARCHITECTURE | Реальность в коде | Уровень |
|---|--------------------------------|-------------------|---------|
| E1 | `D=2560, n_layers=24, ~212.3M params` | `config.py`: `D=4096, n_layers=32`; при D=4096 только MLP ~201M параметров | Высокий |
| E2 | Bind сохраняет норму | Только изолированные подоперации изометричны; полная операция — произвольное билинейное преобразование | Средний |
| E3 | `M_l(t) = max(time_ramp, bridge_readiness)` | `step_gate` использует только time ramp; `bridge_readiness` не участвует; `pred_err` readiness используется в stack | Высокий |
| E4 | Все τ-константы выведены из одного поля | Остались `_vsa_log_param` и `mirror_tau_min/max` | Средний |
| E5 | `intent_alpha = 1 - exp(-τ_l/τ_min)` (README §10) | Код: `1 - 1/τ_l` | Низкий (код лучше) |
| E6 | Maturation линейная сигмоида | Код: log-time reparameterization | Низкий (код лучше) |
| E7 | Спектральный gate `cos(π·tau_norm/2)` | В коде не используется | Средний |
| E8 | «Ни одного магического веса» | Множество config-весов: `gate_l1_weight`, `balance_weight`, `diversity_weight`, `bridge_conn` и др. | Средний |
| E9 | Недостижимый legacy-код в `_adamp_project` | После `return u` остался мёртвый блок (строки 118-134) | Низкий |
| E10 | `codebook='legacy'` по умолчанию | README подразумевает 32-битные коды, но `twin_free` требует K=64 | Низкий |

### 8.2 Архитектурные риски

1. **Stateful память**: потоковые буферы переживают шаг; eval должен сбрасывать.
2. **~20 aux-потерь**: сложный ландшафт, потенциальные конфликты.
3. **Двойной LLRD**: `llrd_decay^l` в optimizer + `tau_lr_mult` в `apply_tau_lr`.
4. **LogitCache compression**: top-k ~46 из 65536; потеря хвоста.
5. **MirrorLR чувствителен к шуму val** на раннем обучении.
6. **bus_bias доминирование**: zero-init, но первые шаги Adam могут давать большие смещения.

---

## 9. Рекомендации по исправлению

### E1: Config vs README (D=4096 vs 2560, 32 vs 24 слоя)

**Проблема:** Самое серьёзное расхождение. README декларирует одну архитектуру, config — другую.

**Рекомендации:**
1. **Привести дефолты config.py к README:**
   ```python
   D: int = 2560
   n_layers: int = 24
   ```

2. **Или обновить README** под фактические дефолты 4096/32.

3. **Добавить assert** в `train.py`:
   ```python
   if args.D != cfg.D or args.n_layers != cfg.n_layers:
       logger.warning(f"CLI overrides config: D={args.D}!={cfg.D}, L={args.n_layers}!={cfg.n_layers}")
   ```

4. **Проверить параметрический бюджет** при D=2560/24 и убедиться, что он действительно ~212.3M.

### E2: Bind не сохраняет норму

**Рекомендации:**
1. **Добавить spectral regularization** на `W_proj`, `W_out`:
   ```python
   bind_spectral_reg = (W_proj.svd()[1].log().var() + W_out.svd()[1].log().var()) * 1e-4
   ```

2. **Использовать weight normalization** на `W_proj`/`W_out`.

3. **Или скорректировать README**, убрав утверждение о полной изометрии.

### E3: Maturation не использует bridge_readiness

**Рекомендации:**
1. **Реализовать документированное поведение:**
   ```python
   # В MaturationController.step_gate
   bridge_ready = bridge.readiness()  # scalar [0,1]
   M_l = torch.max(time_gate, bridge_ready)
   ```

2. **Или обновить README и config.py docstrings**, убрав упоминание `bridge_readiness` из maturation.

3. **Компромисс:** использовать `bridge_readiness` как ускоритель:
   ```python
   T_eff = T_eff * (1 - 0.5 * bridge_ready)
   ```

### E4: Параллельные τ-поля

**Рекомендации:**
1. **Убрать `_vsa_log_param`:**
   ```python
   vsa_base_taus = [tau_l[0], tau_l[n//3], tau_l[2*n//3], tau_l[-1]]
   ```

2. **Убрать `mirror_tau_min/max`:**
   ```python
   mirror_tau_k = tau_l[0] * (tau_l[-1]/tau_l[0])^(k/(K-1))
   ```

3. **Если параллельные поля нужны** — явно документировать их как «вторичные τ-подсистемы».

### E5-E6: intent_alpha и maturation в README

**Рекомендации:**
- Обновить README §10: заменить `1 - exp(-τ/τ_min)` на `1 - 1/τ`.
- Обновить README §15: заменить линейную сигмоиду на log-time reparameterization.

### E7: Спектральный gate

**Рекомендации:**
1. **Реализовать τ-модуляцию:**
   ```python
   tau_mask = cos(pi * tau_norm / 2)
   lambda_k = base_lambda * (1 + spectral_mod * tau_mask)
   ```

2. **Или убрать утверждение** из README.

### E8: Магические веса

**Рекомендации:**
1. Разделить веса на `λ_d-derived` и `experiment-specific`.
2. Добавить обоснование каждому весу в docstring.
3. Провести sensitivity analysis.

### E9: Мёртвый код в `_adamp_project`

**Рекомендации:**
- Удалить строки 118-134 в `core/eva_optim.py`.
- Добавить линтер/тест на недостижимый код.

### E10: Codebook

**Рекомендации:**
1. Либо сделать `codebook='twin_free'` дефолтом с `code_dim=64`.
2. Либо документировать, что `legacy` используется для совместимости.
3. Добавить assert:
   ```python
   if cfg.codebook == 'twin_free' and cfg.code_dim < 64:
       raise ValueError("twin_free requires code_dim >= 64")
   ```

### Дополнительные рекомендации

#### R1: Сократить число aux-потерь

Провести ablation study и удалить потери с:
- нулевым или отрицательным `cos(g_aux, g_CE)`;
- высокой корреляцией с другими aux-потерями;
- нестабильной динамикой (exploding EMA).

#### R2: Улучшить LogitCache

1. Увеличить начальное `k` (сейчас ~46 из 65536).
2. Использовать per-scale `k` вместо scalar.
3. Добавить оценку reconstruction loss между оригинальными и сжатыми logits.
4. Рассмотреть learned compression вместо top-k.

#### R3: Stateful память

1. Добавить автоматический `reset_cache()` перед каждым eval.
2. Проверить идентичность train/eval потоков (B10 уже частично).
3. Добавить тесты на continuity errors.

#### R4: MirrorLR

1. Увеличить требование для boost: не просто `_val_improving`, а несколько последовательных улучшений.
2. Добавить momentum в `_val_ema` с учётом длины eval interval.
3. Рассмотреть median вместо mean для mirror stats.

#### R5: Двойной LLRD

1. Убрать `llrd_decay` из `build_optimizer`, оставить только τ-LLRD.
2. Или сделать их взаимоисключающими через config flag.

#### R6: Тесты

Создать:
- `tests/test_tau_field.py`: монотонность, endpoints, градиенты.
- `tests/test_maturation.py`: deep-first inversion, bridge_readiness если задекларирован.
- `tests/test_bind_norm.py`: проверка нормы.
- `tests/test_vsa_scan.py`: equivalence с sequential scan.
- `tests/test_logit_cache.py`: reconstruction quality.
- `tests/test_config_consistency.py`: README defaults vs code defaults.

#### R7: Экспериментальная валидация

1. Замерить perplexity на стандартных корпусах.
2. Сравнить с трансформером равного размера.
3. Ablation по каждому компоненту.
4. Опубликовать training curves с разложением aux-потерь.

---

## 10. Итоговая оценка

### Сильные стороны

1. **Единая τ-геометрия** — элегантная попытка заменить набор констант одной размерностью.
2. **VSA prefix scan** — математически корректная ассоциативная операция с `O(1)` памятью контекста на инференс.
3. **Sparse block codes** — информационно ёмкое кодирование с минимальным числом параметров.
4. **EVAAdamW** — продвинутый оптимизатор с AdamP, cautious, trust, slow EMA.
5. **Множество audit fixes** — код активно развивается и исправляет найденные проблемы.
6. **Многомасштабная память** — соответствие когнитивной идее о разных темпах обработки.
7. **LogitCache** — амбициозная попытка заменить KV-cache компрессией.

### Слабые стороны и риски

1. **Config противоречит README** — самое серьёзное расхождение.
2. **Документация опережает код** в части maturation (bridge_readiness).
3. **Bind не является полностью нормосохраняющим**.
4. **Параллельные τ-подсистемы** подрывают идею «единого поля».
5. **~20 aux-потерь** создают сложный ландшафт.
6. **Stateful память** создаёт риски на eval и длинных контекстах.
7. **LogitCache компрессия** теряет хвост распределения.
8. **MirrorLR чувствителен к шуму val** на раннем обучении.

### Заключение

EVA-CLM — это **зрелая исследовательская архитектура** с глубокими математическими идеями и активной инженерной доработкой. Многие проблемы ранних версий исправлены. Однако для претензии на замену трансформерам необходимо:

- устранить расхождение между config и README;
- привести maturation в соответствие с документацией или обновить документацию;
- сократить число ручных констант и aux-потерь;
- провести сравнение с сильными трансформерными базлайнами на стандартных бенчмарках.

---

*Анализ завершён. Все проверки выполнены по актуальному состоянию репозитория EVA-CLM (origin: https://github.com/BlackCatSpb/EVA-CLM).*
