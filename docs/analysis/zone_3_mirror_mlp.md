# Зона 3 — GroupedCognitiveMirror, GroupedMLP, зрелость (MaturationController)

**Агент:** 3 из 6 (нерушимая цепочка, `docs/analysis/CHAIN.md`).
**Дерево:** commit `362a224` (M53e), torch 2.13.0+cpu, Python 3.14.6, Windows.
**Вход зоны:** `h` (B,L,D) — поток блока (нормированная копия `_ln(h).float()`), `mem_all` (B,L,D) —
VSA-чтение, `global_state`, `intent`, `salience`, `maturity`, `diff`, `tanh_bias_mod`, `pred_scale_mod`,
`step`, `allow_write`; плюс кэши предыдущего forward (`_cached_hp`, `_cached_pred_error_norm`).
**Выход зоны:** `mirror` (B,L,D), `mlp_mod` (B,L,G), `mem_mod` (B,L,G), `hp` (B,L,G,k),
`pred_error_norm` (B,L); плюс кэши/буферы зеркала и `mat_gate` (n_layers) от матчурации.
**Предыдущие отчёты:** прочитаны полностью (`zone_1_codes_embedding.md:1-595`,
`zone_2_block_core.md:1-716`); их выводы и открытые вопросы разобраны явно (§1.3, §2.13, §3, §6).

**Методика (правило полигона соблюдено).** Продакшн-модель не инстанцировалась, чекпойнты
не грузились. Анализ — чтение кода (`file:строка`) + история git (коммиты инцидентов).
Вычисления — только на мини-конфиге `SMALL = dict(n_layers=2, D=512, mlp_groups=4,
code_dim=16, code_sparsity=4, vocab=1820)` (`tests/test_gradient_flow.py:21`); четыре
скрипта в `%TEMP%\opencode\zone3_minirun{,2,3,4}.py`, все числа помечены **[mini-прогон]**.
Аналитические выводы — **[формула]**. Инциденты M47/M53d — по коммитам `778ea1e`, `0be800d`
(их текст цитируется) и по коду.

---

## 1. Граница зоны и входной контракт

### 1.1 Точка входа/выхода (фактический вызов)

```
# core/block.py:654-668
with torch.autocast(device_type=h.device.type, enabled=False):
    mirror, mlp_mod, mem_mod, hp, pred_error_norm = self.mirror(
        _ln(h).float(), mem_all.float(), global_state=_gs, diff=diff,
        tanh_bias_mod=tanh_bias_mod, pred_scale_mod=pred_scale_mod,
        context_mem=_ctx, allow_write=allow_write, step=step, intent=intent,
        salience=salience, maturity=maturity)
    mirror  = mirror.to(h.dtype)
    mlp_mod = mlp_mod.to(h.dtype)
    mem_mod = mem_mod.to(h.dtype)
```

Потребители внутри блока:

```
# block.py:672-694
mm = mem_mod.unsqueeze(-1)                          # (B,L,G,1)
read_mod = sigmoid(hp @ w_q_dyn / sqrt(k))          # (B,L,G,d)  — живой hp
mem_expert = mem_read.reshape(B,L,G,d) * read_mod
mem_modulated = (mem_expert * mm).reshape(B,L,D)
bind_gated    = (bind_out.reshape(B,L,G,d) * mm * sigmoid(w_bind_gate)).reshape(B,L,D)
enhanced = cap(bind_gated + mem_modulated*w_mem2v*mem2v_scale) + cap(mirror)
h = h + enhanced
...
# block.py:743, 772
h_mlp = self.mlp(_ln(h), mirror_gate=mlp_mod)
h = h + _stream_cap(h_mlp, self.branch_cap)
```

Матурити приходит из стека (`stack.py:423-451, 538-614`); формула — §2.10.
`pred_error_norm` (5-й выход) в блоке **не используется** — авторитетен кэш
`mirror._cached_pred_error_norm`, записываемый внутри зеркала (`mirror.py:537, 560`).

### 1.2 Что именно приходит от агента 2 (проверка контракта)

| Величина агента 2 | Как используется зоной 3 | Проверка |
|---|---|---|
| `_ln(h)` (B,L,D) fp32, per-pos RMS≈1 | вход `h` зеркала (`.float()`), вход MLP | подтверждено; у `_ln` нет обучаемого гейна (`block.py:429-433`) |
| `mem_all` (B,L,D) fp32, raw | второй аргумент зеркала (`mc_g=prefix_mean`, `mc_k`) | **не нормируется**; mini: 231 (L0) / 21665 (L1) — масштаб произвольный, компенсируется EMA-нормировкой сигналов (train-only) |
| `hp_cached` (B,L,G,k) detached | `write_mod = σ(hp_cached@w_i_dyn/√k)` (`block.py:576-586`) | shape-lock `(B,L)` есть (`block.py:577-578`) |
| `pen` (B,L) detached ≤~2 | VSA-decay `d_pen` (`block.py:548-560`), i_gate-boost (`:519-520`), UCL u_gate | **подтверждено: pen — кэш зеркала**, state[4] вестигиален (`block.py:775`) |
| `mat_gate` (n_layers,) | bridge/bank/UCL/private-mem/BridgeGLU | см. §2.10; intent НЕ гейтится (`stack.py:527-530`) |
| `diff`, `pred_scale_mod`, `tanh_bias_mod` | `AdaptiveController` (float / (G,)-тензор) | `pred_scale_mod` = `(1+0.5·tanh(dv−mean)).clamp(0.1,3)` (`adaptive_controller.py:189-199`) |
| `intent` (1,1,G,k) live | `gate_logits += ig·intent_alpha` (`mirror.py:886-922`) | zero-init `w_intent/b_intent/w_sal` ⇒ no-op на init |
| `salience` (B,L,1) detached | `w_sal` (только если `intent_bridge`) | при `observe_output` не вызывался — `_sal=None`, `w_sal` без градиента [mini-прогон] |
| `global_state` (1,1,D) detached | `temp_k += (hp−gs_k)·w_global` | — |

### 1.3 Геометрия: три независимые нарезки D

| | SMALL | cell-4 («продакшн») | CLI/config default |
|---|---|---|---|
| D | 512 | 2560 | 4096 |
| G = mlp_groups | 4 | 32 | 32 |
| d = D/G | 128 | 80 | 128 |
| mirror k (staircase) | 16 / 32 (n=2: `l<1→16`, иначе 32) | 8 / 16 / 32 | 8/16/32 |
| bind_K (блок) | 64 | 32 | 64 |
| code_dim (голова) | 16 (d_code=32) | 64 (d_code=40) | 32 (d_code=128) |

**Ответ агенту 2 (вопрос о нарезке):** K-пространство зеркала **не выравнивается ни с
`bind_K`, ни с `code_dim`** — это отдельный обучаемый автоэнкодер внутри каждой группы
`W_proj: (G,d,k)`, `W_out = W_projᵀ` при `tie_mirror_proj=True` (default; `mirror.py:150-157`).
В cell-4 группа `d=80` накрывает ровно **два** кодовых сегмента (40+40), а `k∈{8,16,32}` —
внутреннее сжатие 10:1/5:1/2.5:1. В SMALL: d=128 = 4 кодовых сегмента, k=16/32 → 8:1/4:1.
Три разбиения D (groups×d, code segments, bind-K) живут независимо.

---

## 2. Математика зоны

### 2.1 GroupedCognitiveMirror — структура

`GroupedCognitiveMirror(D, G, k, ...)` (`mirror.py:80-384`): ансамбль из `G` экспертов,
каждый в своём подпространстве `d=D/G`; у каждого эксперта свой K-space `k`.
`phi = log(1+layer_idx)/log(max(n_layers,2))` (`:137-139`) — координата глубины (fallback,
если нет τ-поля).

Инициализация:
- `W_proj ~ randn·proj_std`, `proj_std = 1/(d·k)^{1/4}` (`:141,149`); при `expert_asymmetry=True`
  (default `config.py:185`) — `nn.init.orthogonal_(W_proj[g])` (`:143-147`).
- `W_out`: при `tie_mirror_proj=True` — **буфер**, синхронизируемый из `W_projᵀ`
  forward-pre-hook'ом (`:150-155, 386-388`); иначе — независимый Parameter (`:157`).
- `alpha_diag`: per-(g,k) α по τ-лестнице `τ_k = τ_min·(τ_max/τ_min)^{k/(K−1)}`,
  `α_k=exp(−1/τ_k)`; `mirror_tau_min=2.0`, `mirror_tau_max=200.0` (`config.py:117-118`);
  при `expert_asymmetry` α_g переопределяется `0.85+0.14·g/(G−1)` (`:186-189`).
  [mini] α mean=0.92, range [0.85, 0.99].
- `log_scale`: при `expert_asymmetry` — **амплитудная лестница** `linspace(log(τ_min^eff/τ_max^eff), 0, G)`
  (`_amplitude_ladder`, `:27-39, 210-212`): `exp(log_scale)` [mini] = `[0.0618, 0.1568, 0.3925, 0.9891]`
  — эксперты имеют экспоненциально разные масштабы выхода (G-инвариантно; старый `0.05·1.5^g`
  взрывался ×14381 при G=32 — audit M4).
- `conv_smooth`: depthwise causal Conv1d(Gk→Gk, k=3) с нулями и center-dirac в весах
  (`:162-167`) → на init `hp_smooth = hp_{t−1}` (сдвиг на 1).
- `_pos_id_buf = sign(randn(1,4096,1,k; seed 12345))`, non-persistent (`:282-283`) —
  биполярное позиционное связывание; **при L>4096 буфер не покрывает последовательность**
  (`hp * self._pos_id_buf[:, :L]`, `:439` — broadcast упадёт при L>4096).
- Гейтовые параметры: `w_gate,b_gate` (`:219-221`), `w_delta_gate` (`:223`),
  `gate_bias = linspace(−scale, scale, G)` (`:224-225`), где scale = `0.5+1.5·layer_idx/(n−1)`
  при `gate_bias_scale_per_layer=True` (`block.py:273`, `config.py:206-207`).
  [mini] L0: `[−0.5,−1/6,+1/6,+0.5]`, L1: `[−2,−2/3,+2/3,+2]` — deep-слои стартуют сильнее
  дифференцированными.

### 2.2 K-space, позиции, hp_prev

```
h_g  = h.reshape(B,L,G,d);  mem_g = mem_all.reshape(B,L,G,d)
mc_g = prefix_mean(mem_g, dim=1)                       # причинное среднее по t
hp   = einsum('blgd,gdk->blgk', h_g, W_proj)           # (B,L,G,k)
hp  *= _pos_id_buf[:, :L]                              # ±1-модуляция (t, k)
hp_prev = cat([zeros_like(hp[:,0:1]), hp[:,:-1]], 1)   # сдвиг по времени
mc_k = einsum(mc_g, W_proj)
```

`prefix_mean` (`:14-24`): `out[t] = cumsum(x[0..t])/(t+1)` — O(L), без утечки будущего (audit M3).

**Найденный инвариант [mini-прогон]:** `_pos_id_buf` (±1, фиксированный по (t,k)) делает
`pen` функцией `α` и `k`, а не мерой предсказуемости. Точная модель:

```
raw_k = x_t,k·p_t,k − α_k·x_{t−1},k·p_{t−1},k,   E[p_t p_{t−1}] = 0
E[raw²] = σ_t² + α²·σ_{t−1}²   (кросс-член обнуляется знаком)
pen = sqrt( mean_{g,k}(raw²)/mean_k(hp²) ) = sqrt( (1 + α²·r)/k ),  r = σ²_{t−1}/σ_t²
```

Проверка [mini]: k=16, α=0.01/0.5/0.9/0.92/0.99 → pen = 0.2501/0.2813/0.3387/0.3420/0.3542
против предсказания `sqrt((1+α²)/k)` = 0.2500/0.2795/0.3363/0.3397/0.3518 (совпадение ≤1 %).
С выключенным `_pos_id_buf` (все +1) pen падает 0.342→0.118 (L0), 0.243→0.077 (L1).
**Следствие:** при α≈0.92 пол (floor) `pen ≥ sqrt(1/k) ≈ 0.25` (k=16) / 0.177 (k=32) —
даже при идеальном самопредсказании; `pen` почти не реагирует на реальную предсказуемость.

### 2.3 Пять сигналов, EMA-нормировка, τ_signal, decorr

```
temp_k   = (hp − mc_k)·w_temp  [+ (hp − gs_k)·w_global, если global_state]
pred_err = (hp − pred_k)/hp_norm · pred_scale_mod        # см. §2.5
smooth_k = hp − hp_smooth                                # hp_smooth = conv_smooth(pad2(hp)) = hp_{t−1}
sym_k    = (hp·w_sym_u)·(hp_prev·w_sym_v)                # билинейное временное взаимодействие
help_k   = trust·σ(w_help)·(attn @ keys)                 # private memory, см. §2.6
signals  = [temp_k, pred_err, smooth_k, sym_k, help_k]   # 5-й только при private_mem
```

EMA-нормировка (`:704-721`): `rms_i = s.detach().pow(2).mean(dim=(0,1)).sqrt()` — per-expert,
per-dim `(G,k)` (audit M4: раньше был один глобальный скаляр); EMA decay `0.01` (~100 шагов),
**только training**; `s_norm = s/(ema+1e-8)`. В eval читается тренировочная EMA
(режимного сдвига нет).

τ_signal (`:723-739`, U5):

```
tau_norm    = tau_config.tau_norm[layer]            # live, refresh в block.py:410-420
log_base    = log(tau_gate_min) + (log(tau_gate_max)−log(tau_gate_min))·(1−tau_norm)
tau_signal  = (log_base + _tau_signal_log).clamp(log 0.01, 2·log(max(tau_gate_max,1.01))).exp()
w           = sigmoid(_signal_log_weights / tau_signal)        # НЕ нормируются на 1
```

`tau_gate_min/max = 0.3/5.0` (τ-поле; fallback `mirror.py:336-337`), `_tau_signal_log` init 0
(identity к расписанию). [mini] `_signal_log_weights = fib_sigmoid_init(5) = [−2.398,−2.398,−1.609,−1.099,−0.336]`;
L0 (tau_norm=0.5) → `tau_signal=1.2247`, `w=[0.124,0.124,0.212,0.290,0.432]`;
L1 (tau_norm=1.0) → `tau_signal=0.3`, `w=[0.0003,0.0003,0.0047,0.025,0.246]` — **deep-слои
на init почти one-hot на `help` (private memory)**; остальные сигналы задавлены. `signal_ent`
(лосс, `losses.py:263-279`) максимизирует энтропию `w` и даёт градиент в `_signal_log_weights`
и `_tau_signal_log`.

Decorrelation (`:741-756`): для каждой пары i<j центрированные `(signals_normed[i]·w[i]).reshape(-1,G·k)`;
`cos².mean()`; усреднение по парам → `_cached_decorr` (**живой**, градиент в веса сигналов).

```
delta = Σ_i w_i·s_norm_i
delta = delta · rsqrt(mean_k(delta²)+1e-7)        # RMS-нормировка по k
delta = delta + tanh_bias·tanh_bias_mod           # tanh_bias init 0
```

### 2.4 Предсказание: α, override, adaptive τ, AR-damping, pen, pred_loss

```
alpha_eff = alpha_diag
if _alpha_override > 0:  alpha_eff = (1−o)·alpha_eff + o·1.0      # warmup-расписание scheduler
pred_k   = hp_prev · alpha_eff                                    # (B,L,G,k)
_pred_k_aux = pred_k                                              # без damping — для aux
hp_norm  = hp.norm(dim=-1, keepdim=True).clamp(min=1e-8)          # норма по k
raw_pred_error = hp − pred_k
if (not training) and _ar_mode:                                   # AR-декодирование (live_inference)
    damp = sigmoid(−raw_pred_error.norm(dim=-1).mean()/_damp_tau) # скаляр по (B,L,G,k)!
    alpha_eff = 1.0 + (alpha_eff−1.0)·damp;  pred_k = hp_prev·alpha_eff
pred_error = (hp − pred_k)/hp_norm · pred_scale_mod.view(G,1)
pred_error_norm = (raw_pred_error/hp_norm).pow(2).mean(dim=(−2,−1)).sqrt()   # pen, (B,L)
```

- `_damp_tau=0.1`; AR-режим включается только драйверами генерации (`live_inference.py:164-166`),
  не обычным eval (B3: иначе val-сигналы получают другую модель).
- `pred_scale_mod` из `AdaptiveController` — `(1+0.5·tanh(dv−mean)).clamp(0.1,3)`;
  fallback внутри зеркала (`:468-471`) другой: `(dv/dv_mean).clamp(0.1,3)` — standalone-путь.
- `pen` — per-position RMS по `(G,k)`; B10-фикс (было ~8.1 как норма по G·k; замок
  `tests/test_b10_spiral_and_ladder.py:48-57`: pen<2, decay-factor>0.75). [mini] 0.3425/0.2467
  (max 0.455/0.368) — совпадает с pos_id-моделью §2.2.
- Adaptive τ (train, `no_grad`, `:492-530`):
  `residual_var = pred_error.var(dim=(0,1))` → `_residual_var_ema.lerp_(·, 0.01)`;
  `rel_var = rv/(rv.mean(-1)+1e-10)`; `alpha_target = σ(2.2 − log(rel_var))`;
  запись **отложенная** (B13/F4-01): применяется на следующем train-forward с `lerp(·, 0.01)`,
  clamp `[0.01,0.99]`; `_ggeo_freeze` блокирует и flush, и pend (checkpoint-recompute).
  Плюс `alpha_novelty` push (при `alpha_novelty_weight>0`).
- `_pred_loss_term = F.mse_loss(_pred_k_aux, hp.detach())` — **живой** (градиент в
  `alpha_diag`/`W_proj` через `pred_k`), цель отцеплена; только training (`:538-547`).
- Кэши (пишутся в **обоих** режимах, M33): `_cached_pred_k`, `_cached_hp`, `_cached_pred_error_norm`
  (`:558-560`); плюс буферы `_cached_hp_buf/_cached_pred_k_buf/_cached_pred_error_norm_buf`
  для introspect (shape-lock, resize to (B,L), M41).

### 2.5 Private memory (кратко; полный разбор — зона 4)

Чтение (`:583-607`): `uncert = σ(|pred_error|)`; `q = hp·uncert`;
`keys = _private_mem.detach().clone()`; при `context_mem` — смесь 0.3/0.7 с ренормировкой к норме;
`attn = σ(q@keysᵀ/√k)` (**сигмоида, не softmax** — независимые гейты);
`help_base = attn@keys`; `disagreement = ||hp−help_base||/||hp||`;
`contra = σ(disagreement−1)`; `trust = 1−contra`; `help_k = help_base·σ(w_help)·trust`.

Knowledge graph (только при `_write`, `:609-646`): `concept_sim` EMA(0.99) из косинусов
нормированной private_mem; `behavior_div` из косинусов усреднённого hp; `_div_run`,
`_div_run_rec` (self-referenced divergence — вход `mirror_lstats`/AdaptiveController);
`_trust_matrix`; `contra_graph = concept_sim·behavior_div`; dominance/isolation.

Запись (`:648-691`): `conf = σ(−|pred_error|.mean(-1))`;
`conf_plastic = conf·(1−contra)·social_pressure`, `social_pressure = 1−0.5·σ(contra_expert+isolation)`;
`conf_soft = conf_plastic^{0.5}`; `conf_bc = conf_soft·G/Σconf_soft` (мягкая конкуренция);
`weighted_hp = mean_{B,L}(conf_bc·hp.detach())`;
`pm_decay = 0.999 − 0.009·σ(3−||pm||)`; `pm ← pm·pm_decay + weighted_hp·_write_scale·(1−pm_decay)`;
`clamp(±10)`. `_write_scale = σ((maturity−matur_write_thr)·10)` при наличии maturity
(`matur_write_thr=0.3`), иначе legacy-пол `_pm_write_delay`/`_pm_coh`.

**Важно (расхождение комментария и кода):** `_write_ok = bool(_write_scale > 0.01)`
(`:667`); при maturity=0 `_write_scale = σ(−3) = 0.0474 > 0.01` → запись **идёт всегда**
(с амплитудой 4.7 %), т.е. гейт мягкий, а не пороговый, вопреки докстрингу «открывается
ТОЛЬКО когда M_l пересекает matur_write_thr».

### 2.6 Гейты: grad_mod, dvar_mod, usefulness, hybrid, expert_gate, intent/contra/meta, boost

```
# grad_mod (audit M4): вход — 1-шагово-старая норма градиента hp
hp.register_hook(g → _prev_grad_norm ← g.detach().norm(-1).mean(dim=(0,1)))   # (G,)
_grad_norm_ema ← EMA(0.99/0.01, train-only)
grad_mod = exp(log_grad_mod_scale)·tanh(_prev_grad_norm/_grad_norm_ema.clamp(1e-8) − 1 + grad_mod_bias)

# dvar_mod
dvar = delta.var(dim=(0,1), unbiased=False).mean(-1)          # (G,), train-only
_delta_var ← EMA(rate = 0.8 + diff·(0.99−0.8) | 0.9)
dvar_mod = exp(log_dvar_mod_scale)·tanh(_delta_var + dvar_mod_bias)

# usefulness (self-organizing, конкурентный)
u_logits = usefulness_predictor(delta).squeeze(-1)            # MLP k→k→1 на эксперта
n_eff    = _fwd_count (train; persistent) | step (eval)
prog = 1−exp(−n_eff/200);  temp = clamp(3·exp(−2·prog), 0.3, 3.0)
if _usefulness_temp > 0: temp = _usefulness_temp              # override от MirrorLRScheduler
threshold = медиана u_logits по G (чётное G — среднее двух центральных)
usefulness = sigmoid((u_logits − threshold)/temp)             # ⇒ mean≈0.5 по построению

# hybrid gate (adaptive_gate.py:27-91)
base = sigmoid(u)·(1 + softmax(u/τ_h)),  τ_h = exp(hybrid_gate.log_tau) ST-clamped [0.1,10]

# BridgeGLU (mirror.py:55-77, 813-834)
glu = σ(log_gain)·σ(Wg·δ)·σ(Wv·δ),  log_gain = log(2.7) ⇒ σ=0.7297
live = (2·glu − 1) · maturity
mlp_mod = base · (1.5·σ(mod_scale_mlp)) · (1 + bridge_glu_beta·live)   # β=0.25
# без BridgeGLU: mlp_mod = base · (1.5·σ(mod_scale_mlp))
mem_mod = usefulness · σ(mod_scale_mem)          # без hybrid, без maturity
```

`mod_scale_mlp/mem` init `log 2` → σ=2/3; `mlp_mod` [mini] mean 0.670 (L0)/0.667 (L1), std 0.10/0.08;
`mem_mod` mean 0.3325/0.3330. `log_grad_mod_scale/log_dvar_mod_scale` init
`log_mod_init = −2.30 + 1.49·0.6^{layer_idx}` (L0 −0.81, L23 −2.30).

Expert gate (`:875-970`):

```
gate_signal = |pred_error|
gate_logits = gate_signal·w_gate + b_gate + gate_bias + delta·w_delta_gate
            + grad_mod + dvar_mod
if intent_bridge: ig = (hp−intent)·w_intent + b_intent + salience·w_sal
                  ig /= EMA_rms(ig);  gate_logits += ig·intent_alpha
if private_mem:   ctr = disagreement·w_contra + contra_expert
                       + (spec·w_spec + cons·w_cons)·w_contra·(1−intent_alpha)
                  ctr /= EMA_rms(ctr);  gate_logits += ctr·intent_alpha
if meta_trust:    gate_logits −= (1−intent_alpha)·_meta_private_mem
boost = active·intent_alpha·σ(3·ls_dev), active = (_ls_var_run < 0.05)  # train+eval
gate_logits += boost
expert_gate = σ(gate_logits)                       # (B,L,G)
```

`intent_alpha = 1−1/τ_l` (`tau_config.py:170-180`, v3) — τ-авторитет амплитуды;
для L0/L23 prod ≈ 0.875/0.998. [mini] `expert_gate` mean 0.448/0.458, std 0.185/0.291;
`gate_usage` (среднее по B,L) L0 [0.293…0.612], L1 [0.156…0.789] — гейты реально
дифференцированы уже на init (вклад `gate_bias`).

Выход зеркала:

```
linear   = delta @ W_out                                   # = delta @ W_projᵀ (tie)
mirror   = (tanh(linear) + exp(log_skip_alpha)·linear·min(1, √k/||linear||)) · exp(log_scale)
alpha    = σ(w_alpha·[normalize(h_g), normalize(mirror)] + b_alpha)   # SMF, w_alpha=0,b=0 ⇒ 0.5
mirror  *= alpha
mirror  *= expert_gate                                     # (B,L,G,1) broadcast
mirror   = mirror.reshape(B,L,D)
```

### 2.7 GroupedMLP: SwiGLU + mirror-гейтирование

`core/mlp.py:67-107`:

```
h = norm_w · h · rsqrt(mean(h², dim=-1, keepdim=True)+1e-7)      # обучаемый norm_w (D)
h → (B,L,G,d);  hg = (G, B·L, d)
gate = silu(hg @ W_gate)                                         # (G,BL,hidden), hidden=expand·d
if mirror_gate is not None:
    mg = mirror_gate.float() → (G,BL,1)
    gate = gate · (mlp_gate_a + mlp_gate_b · mg)                 # a=1.0, b=gate_b_init=0.25
up   = hg @ W_up
hf   = (gate·up) → (B,L,G,hidden)
out  = (hf @ W_down) → (B,L,G,d)
self._cached_group_out = out                                     # (B,L,G,d), для diversity-loss
return out.reshape(B,L,D)
```

- Init: `W_* ~ randn·sqrt(2/(d+hidden))` (SwiGLU); `norm_w=1`; `mlp_gate_a=1`, `mlp_gate_b=0.25`
  (`config.py:302`; при resume train.py:350-356 может «переоткрыть» гейт).
- **Пост-множитель убран** (`block.py:741-742`): раньше было `h_mlp *= mlp_mod` — двойное
  гейтирование (и внутри SwiGLU-гейта, и снаружи). Сейчас модуляция входит **только** в
  SwiGLU-гейт; `mlp_mod` ∈ (0, ~1.9), множитель `1+0.25·mg` ∈ (1, 1.47) — MLP можно только
  **усилить** относительно no-gate, подавить нельзя.
- [mini] мгновенный отклик: `mg=None → ‖out‖=57.44`, `mg=1.0 → 71.80`, `mg=1.5 → 78.99`
  (ровно ×(1+0.25·mg)); `mg=0` = no-gate.
- `_mlp_ratio` (`block.py:760-771`): fast (0.99) / slow (0.999) EMA нормы `‖h_mlp‖`, cold-start
  от первого наблюдения; **потребителей в дереве нет** (grep `_mlp_ratio` — только запись) —
  мёртвая телеметрия с `.item()`-синком каждый forward.
- `mlp_depth_lr_exp=0.10` (`config.py:307`, `stack.py:1294-1320`): backward-хук ×exp(0.1·i)
  на все параметры MLP (не на зеркальные гейты).

### 2.8 MaturationController: формулы, readiness, step_gate

`core/maturation.py`.

```
tau_norm_l = (log τ_l − log τ_min)/(log τ_max − log τ_min) ∈ [0,1]     # из tau_config.tau_norm_live()
T_eff_l   = T0 + alpha·(1 − tau_norm_l)·T_delay
gate_l(t) = σ( (log t − log T_eff_l.clamp(min=1)) · (T_eff_l/Δ) )      # log-time reparam (B3)
```

Defaults (`config.py:154-157`): `T0=8000, T_delay=8000, alpha=1.0, Δ=4000`; deep (τ_norm=1) →
`T_eff=8000`, shallow (τ_norm=0) → `T_eff=16000`. [mini] `step_gate`: step 8000 → [0.229 (L0,
τ_norm=0.5), 0.500 (L1)]; step 16000 → [0.703, 0.800]; step 22000 → [0.860, 0.883].

Readiness (`update`, `:163-178`):

```
warm (<300):  pen_init ← max(pen_init, pe);  pen_ema ← pen_init
else:         pen_ema ← lerp(pen_ema, pe, 1−0.999)
sat = clamp(1 − pen_ema/pen_init, 0, 1)
readiness = σ((sat − 0.3)/0.2)                       # r0=0.3, rs=0.2
```

**Критическая находка [mini + формула]:** `pen_init` — буфер, инициализированный **1.0**
(`:83`), и warm-фаза берёт `max` — после B10 (pen ≈ 0.25–0.45 < 1.0) `pen_init` **навсегда
остаётся 1.0**; warm-захват «random-regime» уровня не работает. Тогда `sat = 1 − pen_ema`, и
`readiness` — детерминированный таймер: `pen_ema` экспоненциально спускается от 1.0 к текущему
pen с τ≈1000 шагов. Симуляция (pen=0.34): readiness = 0.541 (1000), 0.768 (2000),
**0.833 (3135)**, 0.858 (8000+) — и **0.182 при step<300**. Это точно воспроизводит
наблюдение M47 «mat reached 0.82 around step 3135» (commit `778ea1e`). Вывод: `mat→0.82` —
следствие константы `pen_init=1.0`, а не реальной компетентности зеркала; при pen>1 (до B10,
шкала ~8.1) механизм работал бы как задумано. Верхний потолок readiness при pos_id-поле
[mini]: ~0.455 (α0=0.92) / 0.487 (α0=0.99) — если бы pen_init корректно ловил стартовый pen.

`global_ready` = все `gate > matur_bridge_control_threshold` (fallback 0.1, в `EVAConfig`
поля нет) (`:153-161`).

**Как mat_gate попадает в блоки** (`stack.py:423-451`):

```
if step is None:  mat_gate = maturation.gate            # eval: последний тренировочный
else:             mat_gate = max(step_gate(step), readiness)
                  maturation.gate.data.copy_(mat_gate)  # publish combined (M11/F2B-04)
tau_config.update(mat_gate_for_tau)                     # в НАЧАЛЕ forward, gate предыдущего шага
```

`tau_config.update` использует **буфер gate предыдущего forward** (1-шаг-старый) для
`gate_tau = exp(log τ_max^g + (log τ_min^g − log τ_max^g)·mat_gate)` (`tau_config.py:160-168`);
[mini] `gate_tau(0.82) = 0.4978` — точно наблюдённое M47 `lbg_tau→0.50`.

**Что гейтит maturity:**

| Потребитель | Формула | Порог/режим |
|---|---|---|
| bridge.inject_layer | `scale = tanh(stream_log_scale)·maturity·inj_strength(τ_norm)` | линейно (M2: двойное применение убрано) |
| memory_bank | stack **не вызывает** банк при `mat < mem_min_write_mat=0.3` (`stack.py:577-578`); внутри `_can_write` и read возвращает `h` без изменений | жёсткий порог 0.3 |
| UCL (concept_layer) | `_maybe_write` hard-skip при `mat<0.1`; birth skip; `_mat = mat_gate[0]` — только слой 0 | порог 0.1 |
| private mem (mirror) | `_write_scale=σ((mat−0.3)·10)`, `_write_ok = >0.01` ⇒ всегда true | мягкая шкала |
| BridgeGLU | `live ·= maturity` | линейно |
| LayerBridgeGate | pre-ready: `gate=mat`; post: `SpectrumGate·mat` | — |
| intent bus | **НЕ гейтится** (`stack.py:527-530`); зеркало использует `intent_alpha` | — |

### 2.9 Связь с depth-контроллером и τ-полем

- **DepthController** (`adaptation.py:87-169`) — отдельный контур: `set_active_depth`
  замораживает `requires_grad` слоёв ≥ k; разблокировка — по плато val-loss (`slope > −k·σ`),
  `init_active_layers=8`, `unfreeze_inc=4`. Прямой связи с mat_gate нет; оба контура
  пересекаются только через pred_err/val-loss и τ-поле (LLRD `apply_tau_lr`).
- **τ-поле**: `tau_l` монотонна по глубине (cumsum softplus от `_tau_dev`, нормированный на
  конец, B3); из неё выводятся `tau_norm`, `mat_delay`, `gate_tau`, `intent_alpha`, `lr_mult`,
  `mem_tau`. Зеркало обновляет `_tau_norm_layer`/`_intent_alpha` каждый forward блока
  (`block.py:410-420`) — M7-фикс против замороженных снапшотов.

### 2.10 Влияние на поток: количественно (mini) и разбор инцидента M47 / свежего прогона

**Порядки величин [mini-прогон, SMALL defaults, B=1, L=64, train, step=1000]** (per-position нормы):

| Величина | L0 | L1 |
|---|---|---|
| `h_in` | 2.448 | 69.99 |
| `conv_out` | 4.50 | 2.60 |
| `bind_out` | 2.73 | 3.30 |
| `mem_all` (raw) | 231.3 | 21665 |
| `enhanced_base` | 53.03 | 288.9 |
| `mirror` | **1.98** | **4.33** |
| `mlp_out` | 12.90 | 15.55 |
| `h_out` | 69.99 | 384.6 |

Гейтовые множители [mini]: `mem_mod`≈0.333, `read_mod`≈0.500, `σ(w_bind_gate)`=0.5 ⇒
memory-ветка в потоке умножается на ~0.17; `mlp_mod`≈0.67 ⇒ SwiGLU-гейт ×~1.17.
Сам `mirror` — малая аддитивная коррекция (2–4 против 70–385), т.е. зеркало управляет
потоком **через гейты**, а не собственной амплитудой.

**M47 (commit `778ea1e`):** «mlp_out collapsed 1400→424 and diversity 0.6→0.08 around step 3135
while mat reached 0.82 and lbg_tau fell to 0.50 — the MoE gates may be over-sharpening as
maturation opens». Проверка по коду/мини-прогонам:

1. `lbg_tau = 0.4978` — **точное** следствие `gate_tau(mat=0.82)` (формула §2.8), не отдельный
   сигнал.
2. `mat=0.82` — **артефакт таймера** `pen_init=1.0` (§2.8): при pen≈0.34 readiness=0.833 ровно
   на шаге 3135. Это не «MoE over-sharpening» и не компетентность.
3. `mlp_out` не может упасть от гейтов: вход MLP всегда `_ln(h)` с per-position нормой
   ровно `√D` (RMS≈1), а множитель гейта `1+0.25·mlp_mod` ∈ (1, 1.47). [mini] полный свип
   maturity 0→1 меняет `mlp_out` на **−2.8 %** (105.55→102.56); `mlp_mod` 0.672→0.558.
   Значит, 1400→424 (×0.30) — изменение **весов/усиления MLP**, не гейтирования. [mini]
   масштабирование `W_gate/W_up/W_down` на 0.5 даёт ×0.114 (≈0.5³) при неизменной diversity
   (0.0914→0.0895). В логе `mlp_out` — **полная** норма тензора (`notebooks/eva_colab.ipynb:1075-1076`),
   т.е. для cell-4 (B=2, L=224) 350 ↔ per-token 16.5 ↔ RMS 0.33; 1400 ↔ RMS 1.31. Это
   умеренные изменения усиления, а не «схлопывание».
4. `diversity` в логе — это aux-лосс `MSE(corr, I)` (`losses.py:118-136`), который **минимизируется**:
   меньше = выходы групп менее коррелированы. Метрика **инвариантна** к per-expert скалярному
   гейту (стандартизация убирает масштаб группы) — [mini] `mg` токен-варьирующий vs константный:
   div 0.0932/0.0932. 0.6→0.08 — это прогресс цели лосса (специализация), а не collapse-сигнал.
   Имя «diversity» вводит в заблуждение: логируется лосс, а не «разнообразие».
5. **Свежий прогон** (mlp_out ~350, diversity ~0.1, val 10.82): те же числа, что в M47/M53d, но
   val — лучший. Это разрешается неидентифицируемостью маркеров: (i) `mlp_out` — норм-функционал
   весов, а `final_norm` перед головой делает масштаб ствола CE-нерелевантным; (ii) `diversity` —
   сам aux-лосс, стремящийся к тем же малым значениям; (iii) `mat`/`lbg_tau` — часы
   (`pen_init=1.0`), а не здоровье. Различитель здорового и коллапсного режимов лежит **не здесь**,
   а на голове: в M53d коллапс вызвал SRL hard-refine (`u ← u + 0.7(logit(c)−u)` каждый forward,
   биты в ±9.2, sat→1.0, val→inf, commit `0be800d`); в свежем прогоне `head_srl_apply=False` по
   умолчанию — та же «картинка» MLP-маркеров без головного дефекта. Т.е. маркеры M47 — следствие,
   а не причина; в M47 крэш на шаге 3190 вызвали graph-pinned `_cache_*`/`_pred_loss_term` +
   retained-pass ggeo на checkpointed-графе (тот же коммит), а MLP-наблюдение было оставлено
   «на посмотреть» и не подтвердилось.

**Итог парадокса:** `mlp_out`/`diversity`/`lbg_tau` — прокси разных механизмов (усиление весов,
цель aux-лосса, часы матчурации) и не образуют диагностического признака; их совпадение в
здоровом и коллапсном прогонах ожидаемо. Диагностический канал — head-telemetry (`sat`,
`srl_expl`, wall) + val, что и было закрыто M53d/M53e.

### 2.11 Режимы, dtype, AMP

- Зеркало вызывается под `autocast(enabled=False)` (`block.py:656`), входы `.float()`;
  выходы кастуются в `h.dtype` (`block.py:664-667`); внутри всё fp32.
- Train/eval (M33/M44-доктрина «inference=learning»): кэши `_cached_hp/_cached_pred_k/_cached_pred_error_norm`
  пишутся в **обоих**; `_alpha_override` читается в обоих; anti-collapse governor — в обоих.
  Только training: `_pred_loss_term`, adaptive-α запись (deferred), EMA сигналов, `_grad_norm_ema`,
  `_delta_var`, `_gate_ema`, `_fwd_count`, KG-update, private-write. Только eval+`_ar_mode`:
  AR-damping. `allow_write=False` (eval без флага) выключает запись private/KG.
- `pred_error_norm` и `hp` — fp32; `mlp_mod/mem_mod/mirror` — dtype потока.

### 2.12 Градиентные пути (проверено [mini-прогон])

| Величина | Detach/live | Куда идёт градиент |
|---|---|---|
| `_cached_hp`, `_cached_pred_k`, `_cached_pred_error_norm`, `_cached_gate` | detached | только forward-потребители (write_mod/decay/UCL/read_mod-вход) |
| `_prev_grad_norm` | detached (hook) | вход `grad_mod` (без градиента) |
| `_pred_loss_term` | **live** | `mse(pred_k_aux, hp.detach())` → `alpha_diag`, `W_proj` |
| `_cached_decorr` | **live** | → `_signal_log_weights`, `_tau_signal_log` (через `w`) |
| `_cached_gate_l1`, `_cached_gate_usage` | **live** | → `w_gate/b_gate/w_delta_gate/gate_bias`, `usefulness_predictor` (через `usefulness`/`delta`) |
| `_cached_usefulness` | **live** | `reinforce = MSE(u, gate.detach())` → только predictor |
| `hp` (выход) | live | `read_mod = σ(hp@w_q_dyn/√k)` → `mem_modulated` → h → CE; плюс hook-норма |
| `mirror`, `mlp_mod`, `mem_mod` | live | прямо в h (зеркало/MLP/память) |

CE-градиент в зеркало идёт четырьмя путями: (1) `mirror → enhanced → h`; (2) `mlp_mod → SwiGLU-гейт`
(и `mlp_gate_a/b`); (3) `mem_mod → bind/memory`; (4) `hp → read_mod`. Aux-лоссы: `pred`
(§2.4), `decorr`, `gate_l1`, `reinforce`, `balance` (HHI `_cached_gate_usage`),
`gate_repulse` (−энтропия softmax(usage)), `alpha_novelty` (control-push в `alpha_diag`),
`signal_ent` (−H(w)), `div`/`ls_reg` (`log_scale`), `gradalign` (bypass, `mlp_mod`).
**Конфликт целей:** `balance` и `gate_repulse` оба тянут распределение использования экспертов
к равномерному (HHI↓ и −H↓), тогда как специализацию несут `diversity` (MLP-группы),
`alpha_novelty` и softmax-конкуренция usefulness.

### 2.13 Мёртвые/инертные каналы (найдено)

- `pred_weight` (`stack.py:301-302`) вычисляется и передаётся, но `losses.compute_losses`
  его **не использует** (только сигнатура `losses.py:10`) — pred aux сырой.
- `_mlp_ratio` — нет потребителей (§2.7).
- `mirror.cache_grad_norms` — вызывается только тестом; в цикле работает hook.
- `w_sal` без градиента, если `observe_output` не вызывался до forward.
- `_usefulness_temp`: после warmup+blend scheduler оставляет буфер на **0.5** (не сбрасывает в 0,
  `lr_scheduler.py:178-203`) ⇒ intrinsic-расписание `temp(n_eff)` не действует, usefulness всегда
  σ((u−median)/0.5).
- `pred_error_norm` (5-й выход зеркала) не используется блоком.
- `pen` в state-tuple вестигиален (подтверждение агента 2).

---

## 3. Что зона делает с входным контрактом (проверка/преобразование/потеря)

1. **`h`**: нормируется `_ln` до per-pos RMS≈1 (амплитуда потока не влияет на зеркало/MLP);
   при `|h|` у клапа 1e-6 `_ln` даёт норму до `√D/√(1e-7)` — теоретический выброс, но поток
   здорового режима далёк от нуля.
2. **`mem_all`**: не нормируется на входе; масштаб компенсируется EMA сигналов (train-only,
   τ≈100 шагов) — транзиент после resume/смены режима возможен (риск).
3. **`hp_cached`/`pen`**: 1-шагово-старые (M33); shape-lock есть; `state[4]` не обновляется.
4. **`intent`**: `intent_alpha` (τ-авторитет) умножает нормированный по RMS вклад — защита от
   runaway `‖w_intent‖` (fix run-A2, loss 3.8e22).
5. **`maturity`**: линейно масштабирует BridgeGLU-часть `mlp_mod` и мягко — запись private mem;
   гейты `expert_gate`/`mem_mod` от maturity **не зависят**.
6. **Потери**: `h_emb` не доходит (мёртв, зона 1); `pred_weight` мёртв (§2.13);
   `pen`-канал state мёртв; `hp` после блока живёт до `read_mod` и кэша UCL.

---

## 4. Передача следующему

### 4.1 Агенту 4 (память: UCL, банк, logit cache, intent/bridge, reasoning)

| # | Величина | Форма/dtype | Шкала (mini) | Режим/инвариант |
|---|---|---|---|---|
| 1 | `mem_mod` | (B,L,G) fp32→h.dtype | mean 0.333, std 0.015 | live; гейтит **и** bind, и mem_read: `mem·read_mod·mm`, `bind·mm·σ(w_bind_gate)` |
| 2 | `hp` (выход) | (B,L,G,k) fp32 | per-pos 3.93/4.59 | live; `read_mod=σ(hp@w_q_dyn/√k)`; UCL получает **detached** `_cached_hp` |
| 3 | `_cached_hp` | (B,L,G,k) detached | 1-шаг-старый | `write_mod` (w_i_dyn), UCL `_maybe_write` (gate=`_cached_gate`) |
| 4 | `_cached_pred_error_norm` (pen) | (B,L) detached | 0.34/0.25, max 0.46/0.37 | VSA `d_pen` (центрирован на `_pen_ema`), i_gate-boost `γ·pen`, UCL `u_gate=σ(3(pen−0.5))` |
| 5 | `_cached_gate` | (B,L,G) detached | mean 0.45 | per-expert веса записи UCL |
| 6 | `_cached_usefulness` | (B,L,G) live | mean 0.50 | memory-ветки не потребляют; reinforce-loss |
| 7 | `mat_gate` | (n_layers,) | см. §2.8 | bank: жёсткий порог 0.3; UCL: 0.1 + `mat_gate[0]` |
| 8 | state | `mem_state`(B,4D), `mu_state`(B,4D), `conv_state`(B,D,47), `traj_state`, `pen`(вестиг.) | — | все detach'нуты стеком (`stack.py:695`) |

Снапшот-контракт (`stack.py:1178-1220`): покрывает `_cached_hp/_cached_pred_k/_cached_pred_error_norm/
_cached_gate/_cached_usefulness` (включая `None`-состояние, M45) и `blk._traj_state`; **не** покрывает
`_pred_loss_term/_cached_decorr/_cached_gate_l1/_cached_gate_usage` (они освобождаются
`release_step_graph`, `stack.py:1058-1095`). `reset_cache` (`:1142-1176`) чистит `_cached_hp/_cached_pred_k/
_cached_pred_error_norm/_pred_loss_term/_cached_gate` и `reset_stream_bufs()`, но **не** `_cached_usefulness`
и не aux-кэши; `_traj_state` не чистится (дефект зоны 2, остаётся в силе).

### 4.2 Агенту 5 (голова)

- Прямых величин зеркало голове **не отдаёт**: голова видит `out` после `final_norm` (RMS≈1) +
  logit-cache augment. Влияние зеркала — только через h (mirror/MLP/mem_mod) и через VSA-decay (pen).
- `hp`/`pen`/`mlp_mod` в голову не пробрасываются; `mlp_mod` — только gradalign-aux (bypass,
  `losses.py:281-307`) и telemetry `_last_mlp_mod`.
- Для головы важны: `mem_mod` меняет долю «кодовой» памяти в h (mini: mem-ветка ×0.17),
  `mirror` — малая аддитивная коррекция (2–4), `mlp_out` — доминирующая ветка (13–16, на
  тренированных весах — сотни), но `final_norm` нормирует масштаб.
- Индикаторы для мониторинга из зоны 3: `_last_magnitude`, `_last_gates`, `_cached_gate_l1`,
  `_cached_decorr`, `_cached_usefulness`, `_last_mlp_mod`, `_mlp_ratio` (мёртв).

---

## 5. Связи с другими зонами

- **Зона 1:** `basis`/`code_dim` в зеркало не входят; `hp` — независимое K-пространство на группу.
  Лакуна/known-share (~24 %) — забота головы; зеркало пишет в D-симметричные координаты через
  W_out (dense) — как и все ветки.
- **Зона 2:** зеркало читает `_ln(h)` и `mem_all`; возвращает живые `mirror/mlp_mod/mem_mod/hp`;
  `pen` — единственный «surprise»-канал в VSA-decay; `mat_gate` гейтит bridge/bank/UCL; branch-loss
  читает `_cache_mirror_out` (mirror) и `_cache_bind_out` (=enhanced_base).
- **Зона 4:** UCL потребляет detached `_cached_hp/_cached_gate/pen`; bank — жёсткий порог mat 0.3;
  intent bus — `intent_alpha`; `write_mod` — `_cached_hp` (1-шаг).
- **Зона 5:** indirect через h; head-телеметрия (`sat`, `srl_expl`) — истинный различитель
  режимов M47/M53d.
- **Зона 6:** `pred`/`decorr`/`gate_l1`/`reinforce`/`balance`/`gate_repulse`/`alpha_novelty`/
  `signal_ent`/`div`/`ls_reg`/`gradalign` — сырые aux, спектрально-выровненные (`LossBalancer`,
  `BYPASS_AUX=('gradalign',)`); `apply_tau_lr` × exp(0.1·i) для MLP; `_alpha_override/_usefulness_temp`
  от `MirrorLRScheduler`; `flush_control_pending` (B16) — при сохранении.

---

## 6. Открытые вопросы и риски для следующего агента

1. **`pen_init=1.0` + B10 ⇒ readiness — таймер** (§2.8): `pen<1` не перебивает дефолт, warm-захват
   мёртв, `readiness(t)≈σ((1−pen_ema(t)−0.3)/0.2)`; `mat≈0.82` на ~3135 шаге — часы, не
   компетентность; τ-геометрия (deep-first) при этом перекрыта общим readiness до ~18–26k шагов.
   Ответ агенту 2 на «пере-калибровку порогов после B10»: проблема не только в порогах, а в
   семантике `pen_init` (нужно инициализировать 0 или от первого наблюдения; и/или нормировать
   pen к единой шкале). Не менять код здесь — зафиксировать для зоны 6.
2. **pos_id-пол pen** (§2.2): `pen ≈ sqrt((1+α²)/k)`; при k=8 (cell-4 shallow) пол ≈0.48, при
   k=32 ≈0.24. `pen` почти не различает реальную предсказуемость; pred-aux `mse(pred_k, hp.detach())`
   с α≈0.92 минимизируется относительно зашумлённой цели (sign-flip) — α-обучение частично
   «борется со знаком». Риск: predictive mirror учится α по случайному ±1-паттерну; readiness/
   VSA-decay наследуют эту шкалу.
3. **Private-mem write gate мягкий** (§2.5): при maturity=0 амплитуда 4.7 % — не ноль; при
   ресюме/долгом warm возможна медленная засевка памяти вне «когнитивного» окна (вопреки
   комментарию). Проверить в зоне 4.
4. **`mem_all` без входной нормировки** (§3): EMA сигналов train-only; после resume/смены
   масштаба потока возможен транзиент `temp_k` (до ~100 шагов).
5. **`_usefulness_temp` пиннится на 0.5** после warmup: intrinsic-расписание `temp(n_eff)`
   не работает; при этом в eval `step=None` даёт `n_eff=0` — но override 0.5 выравнивает режимы.
   Если убрать пиннинг — вернётся режимный сдвиг (train n_eff vs eval step).
6. **`_cached_usefulness` не в `reset_cache`** и aux-кэши не в снапшоте (только release) —
   асимметрия контракта документ-границы (M45-класс).
7. **`_mlp_ratio` мёртв**, но содержит `.item()` каждый forward (GPU-sync) — либо потребитель,
   либо удалить (зона 6).
8. **`pred_weight` мёртв** — AdaptiveController.pred_weight/`_pred_weight` не влияет на лоссы.
9. **pos_id буфер 4096**: при `seq_len>4096` (или OOM-recovery с длинным окном) broadcast упадёт;
   проверка длины отсутствует.
10. **M47-разбор** (§2.10): подтверждено, что маркеры `mlp_out/diversity/lbg_tau` неспецифичны;
    M47-крэш — graph-pinning + ggeo (закрыто M48/release_step_graph), а не «over-sharpening».
    Свежий прогон с теми же маркерами и val 10.82 — ожидаемая комбинация; для проверки здоровья
    смотреть head-telemetry (`sat`, `srl_expl`, wall) и val, а не MLP-маркеры.
11. **Конфликт aux-целей гейтов**: `balance` (HHI↓) и `gate_repulse` (−H↓) оба ведут к равномерному
    использованию экспертов — специализацию держат только diversity/alpha_novelty/softmax-usefulness;
    при M47-подобных разборах это стоит учитывать (зона 6).
12. **`cache_grad_norms` не вызывается циклом** — жив только hp-hook; при `hp.requires_grad=False`
    (freeze depth) `_prev_grad_norm` не обновляется (grad_mod «замерзает» на последнем значении).
