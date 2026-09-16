# Зона 4 — Память: UCL (концепты), StreamingMemoryBank (L1/L2), LogitCacheAttention (R1), intent-шина + SemanticBridge, reasoning, LayerBridgeGate

**Агент:** 4 из 6 (нерушимая цепочка, `docs/analysis/CHAIN.md`).
**Дерево:** commit `362a224` (M53e), torch 2.13.0+cpu, Python 3.14.6, Windows.
**Вход зоны:** `h` (B,L,D) — поток (на разных точках forward), `_cached_hp` (B,L,G,k), `pen`
(B,L), `_cached_gate` (B,L,G), `mat_gate` (n_layers,), `tokens` (B,L), `step`,
`_last_salience` (B,L,1), `_last_logits` (B,L,V), `_last_lacuna_rel` (scalar).
**Выход зоны:** `h` после UCL-инъекции, bank-read (через `_mem_dir`), reasoning, logit-cache
augment; `_last_bus`; `_intent_stream`; буферы UCL/банка/кэша/моста; телеметрия.
**Предыдущие отчёты:** прочитаны полностью (`zone_1_codes_embedding.md:1-595`,
`zone_2_block_core.md:1-716`, `zone_3_mirror_mlp.md:1-647`); их выводы и открытые вопросы
разобраны явно (см. §1.2, §3, §5, §6).

**Методика (правило полигона соблюдено).** Продакшн-модель не инстанцировалась,
чекпойнты не грузились. Анализ — чтение кода (`file:строка`). Вычисления выполнены только
на мини-геометрии `SMALL = dict(D=512, k=16, G=4, code_dim=16, code_sparsity=4, vocab=1820)`
на голых модулях `UnifiedConceptLayer` / `StreamingMemoryBank` / `LogitCacheAttention`
(скрипты `%TEMP%\opencode\zone4_minirun.py`, `zone4_ucl_dbg{,2}.py`); все такие числа
помечены **[mini-прогон]**. Аналитические выводы — **[формула]**. Числа предыдущих зон
цитируются с их метками ([audit 02a]/[audit 02b]).

---

## 1. Граница зоны и входной контракт

### 1.1 Точки входа/выхода (фактический поток `EVAStack.forward`, `stack.py:209-793`)

| Компонент зоны | Создание | Вызов в forward | Сколько раз за forward |
|---|---|---|---|
| `UnifiedConceptLayer` (UCL) | `stack.py:160-168` | `stack.py:619-636` (только `i==0`, после M50-cap блока 0) | **1** |
| `StreamingMemoryBank` | `stack.py:148-156` | `stack.py:577-584` (внутри цикла слоёв, до блока) | **до n_layers** (по слоям с `mat_gate[i]≥0.3` и `tokens≠None`) |
| `LogitCacheAttention` | `stack.py:196-207` | `stack.py:775-791` (в конце, после reasoning/триады) | **1** (+R1 внутри) |
| `SemanticBridge` | `stack.py:122-127` | `inject_layer`/`probe_layer`/`update_stream` — `stack.py:537-571`; `loss` — `losses.py:550-557` | **n_layers** (probe+inject+EMA) + 1 loss |
| `LayerBridgeGate` | `stack.py:131-136` | `layer_gate` — `stack.py:555-559`; диагностика — `stack.py:641-668`; aux — `losses.py:386-437` | n_layers (результат `layer_gate` **не используется**) |
| Intent-шина | `stack.py:76-115` | `stack.py:341-366, 406-422, 470-530, 697-699`; голова — `losses.py:20-29` | 1 (по слоям внутри цикла) |
| Reasoning | `stack.py:62-74` | `stack.py:717-731, 852-967` | K = `round(reasoning_max_steps·tau_norm_reasoning)` |
| Триада (inference) | cfg `triad_reason=True` (`config.py:53`) | `stack.py:733-773` | до `triad_max_passes=3` (рекурсивный forward) |

### 1.2 Что именно приходит от агента 3 (проверка контракта `zone_3_mirror_mlp.md:556-576`)

| Величина агента 3 | Как используется зоной 4 | Проверка |
|---|---|---|
| `mem_mod` (B,L,G), mean 0.333 | **напрямую не потребляется** зоной 4 (гейтит bind/mem_read в блоке до UCL; на UCL-инъекцию и bank-read не влияет) | подтверждено |
| `hp` (B,L,G,k) live | UCL получает **detached** `_cached_hp` (`stack.py:622`); зеркало — не отсюда | `_cached_hp` пишется зеркалом 1-шагово-старым (`mirror.py:559-560`, M33) |
| `_cached_pred_error_norm` (pen, B,L) 0.25–0.46 | UCL: `conf=σ(−pen)`, `u_gate=σ(3(pen−0.5))`; bank: не потребляет (только `lacuna`) | [mini] `u_gate(0.34)=0.382`, `u_gate(0.46)=0.470`, `u_gate(0.25)=0.320` |
| `_cached_gate` (B,L,G) detached | UCL write: per-expert веса `shared` | форма-чек `(B,L,G)` (`concept_layer.py:182`) |
| `mat_gate` (n_layers,) | bank: жёсткий порог 0.3 (и в стеке `:578`, и внутри `:420-421`); UCL: hard-skip при `mat<0.1`, `_mat=mat_gate[0]` | подтверждено (зона 3, §2.8) |
| `_residual_var_ema` зеркала | UCL `_update_maturity(resvar)` → `_mature` | `stack.py:624` |
| `_last_lacuna_rel` (из головы) | bank broadening M55a: `temp·(1+k·rel_excess)` | `stack.py:400-404`, `memory_bank.py:443` |
| `_last_read` (bank) | → `_head._mem_dir` для temper | `stack.py:582-584`; **обновляется только в training** (`memory_bank.py:471-473`) |
| `_cached_usefulness` | **не потребляется** зоной 4 | grep: только reinforce-loss (зона 3) |

### 1.3 Геометрия зоны 4 (три конфигурации дерева)

| | SMALL | cell-4 («продакшн») | CLI/config default |
|---|---|---|---|
| D | 512 | 2560 | 4096 |
| UCL: `k` = mirror.k слоя 0 | 16 (нет staircase при n=2 → l<1:16) | **8** (staircase: первая треть) | **8** (staircase, n=32) |
| UCL: `bridge_dim` | `bridge_dim` cfg (256) | 256 | 256 |
| UCL: `S` (`unified_concept_S`) | 8 | 8 | 8 |
| bank: `mem_bridge_dim`, L1/L2 | 256, 3/32 | 256, 3/32 | 256, 3/32 |
| logit cache: K_bits = code_dim | 16 | 64 | 32 |
| `G` (experts) | 4 | 32 | 32 |
| `K_max` (intent) | 32 | 32 | 32 |
| `cache_horizon_tokens` | 512 (=tau_max) | 512 | 512 |

Источники: `config.py:330-365,426`; staircase `block.py:253-262`; `stack.py:82-91,102-104,
162-164,199-206`. В SMALL UCL `k=16`, в cell-4/CLI — `k=8`: `write_q_proj: 8→256`.

### 1.4 Порядок операций внутри одного forward (только зона 4)

```
# до цикла:
_rt_snap ... (evaluate)                       # eval-изоляция (train.py:701)
self.bridge.start_forward()                   # stack.py:453-454: _preds=[]
_lac = max(0, _last_lacuna_rel − 1)           # stack.py:400-404 (M55b, rel-excess)
intent_streams = carried _intent_stream       # stack.py:346-366
# цикл по слоям i = 0..n−1:
  bridge.inject_layer(i, h, maturity=mat_gate[i], tau_norm=tau_norm[i])   # :541
  _s_l = bridge.probe_layer(h.detach()); bridge.record/update_stream      # :567-571
  if mat_gate[i] ≥ 0.3 and tokens≠None:
      h = memory_bank(h, tokens, step, mat_gate=_mg_l[i], lacuna=_lac)    # :577-581
      _head._mem_dir = memory_bank._last_read (если не None)              # :582-584
  intent_i = bus_i[..., :k_i] (до блока)                                  # :470-530
  h, s_out = layer(...)                                                   # :604-614
  h = _stream_cap(h, 1e3)                                                 # :615
  if i == 0: h = h + concept_layer(h, hp=_cached_hp, pen, resvar, ...)     # :619-636
  LBG-диагностика (no_grad)                                               # :641-668
# после цикла:
h = final_norm(h)                             # :710-715 (RMS≈1)
if explicit_reasoning and s>0: h = _adaptive_reasoning(h, s, ...)         # :717-731
if triad (inference): h = 0.5h + 0.5·forward(...)                         # :733-773
h = logit_cache.augment(h, novelty, logits=_last_logits if train)         # :775-791
```

Ключевое следствие: **UCL и банк стоят до головы и до `final_norm`** (кроме reasoning/cache),
т.е. их вклад нормируется `final_norm` вместе со всем потоком; reasoning и cache-augment
работают уже на RMS≈1.

---

## 2. Математика зоны

### 2.1 UnifiedConceptLayer (UCL) — `core/concept_layer.py`

#### 2.1.1 Слоты и состояние

```
S = 8 слотов;
concept_keys (S, bridge_dim) — буфер, init: F.normalize(randn(S,bridge_dim; seed=42))  :74-76
concept_vals (S, D)         — буфер, init: randn(S,D; seed=42)·0.02                    :77
concept_age, concept_count, concept_confidence (S,) — буферы (persistent)
_mature (1,) init 0.5; _resvar_ema=0; _resvar_var=1; _step (long)
_n_births/_n_updates/_n_skipped (long); _cached_birth_gate (non-persistent)
```

Обучаемые параметры: `write_q_proj: k→bridge_dim`, `write_v_proj: k→D`, `q_proj: D→bridge_dim`,
`out_proj: D→D`, `read_scale` (init **−4**), `log_tau_read` (init 1.0), `log_tau_update` (0.0),
`_log_tau_novelty_thr` (0.0), `log_tau_birth` (0.0), `log_tau_maturity` (1.0),
`log_tau_gate` (0.0 — **не используется**), `_log_tau_birth_thr` (0.0),
`_log_tau_decay_thr` (0.0), `log_tau_uncert` (**ln 0.5 = −0.6931**), `log_tau_contra` (1.0),
`uncert_kappa` (3.0). Все τ-пороги — через `sigmoid/exp` (`:52-118`).

#### 2.1.2 Непрерывная зрелость `_mature` (от resvar зеркала)

```
resvar = layer.mirror._residual_var_ema.mean()          # stack.py:624
lam    = lambda_d(cfg.lambda_d)                          # λ₃ ≈ 1.839
ema_rate = 1/λ ≈ 0.5437
_resvar_ema ← _resvar_ema + ema_rate·(rv − _resvar_ema)
_resvar_var ← (1−ema_rate)·_resvar_var + ema_rate·(rv−_resvar_ema)²
cv  = sqrt(_resvar_var)/(|_resvar_ema|+1e-8)
mat = σ((1/max(cv,1e-8) − λ)·τ_mat),  τ_mat = exp(log_tau_maturity).clamp(0.1,10) = 2.718 (init)
```
(`concept_layer.py:123-152`). Обновляется **в обоих режимах** (M33, `:313-317`).

[mini-прогон] cold-start, `resvar=0.01`: первый вызов `mat = 0.0068`, 5-й — `0.0081`,
200 вызовов с неизменным resvar → `mat = 1.0000`. При `resvar=0` → `cv=0` → `1/cv=1e8` → `mat=1`
мгновенно. Т.е. `_mature` — **второй таймер** (после readiness агента 3): он растёт от
стабильности `resvar` (не от `mat_gate` стека) и практически не зависит от качества концептов.

#### 2.1.3 Запись `_maybe_write` (`:156-279`)

```
if mat_gate < 0.1:  return (no-op, буферы не тронуты)                     # hard-skip
gate_w = gate (B,L,G) detached; shared = Σ_g gate_w·hp / Σ_g gate_w  (B,L,k)   # :182-187
q = write_q_proj(shared); q_n = normalize(q)                              # live
sims = q_n · normalize(keys)           (B,L,S)                            # live
best = argmax(sims); best_sim = max(sims)                                 # no_grad
val_proj = write_v_proj(shared)                                           # live
conf = σ(−pen)                                                            # (B,L)
alpha = σ(−log_tau_update).clamp(0.001, 0.5) = 0.5 (init)                 # live скаляр
mat = self._mature (не mat_gate!)
```
**Update (только `mat ≥ 0.3`)** — `:205-236`:
```
conf_floor = median_BL(conf).clamp(min=0.01)
для каждого слота s: mask = (best==s) & (conf ≥ conf_floor)
  new_key = normalize(mean(q_n[mask])); new_val = mean(val_proj[mask])
  count[s] < 3  →  слот заменяется целиком
  иначе         →  key ← normalize(key·(1−α) + new_key·α); val ← val·(1−α) + new_val·α
  запись через out-of-place index_copy → градиент течёт в write_q_proj/write_v_proj
  счётчики (count+=, age=0, _n_updates) — no_grad
```
**Birth (только `mat ≥ 0.1` и `novel.any()`)** — `:238-275`:
```
gap          = σ(_log_tau_novelty_thr) = 0.5 (init)
birth_thresh = σ(_log_tau_birth_thr)·(1 − tau_norm·σ(_log_tau_decay_thr)) = 0.5·(1−0.5·tau_norm)
novel = ((1 − best_sim) > gap) & (conf ≥ birth_thresh)
слот = первый count==0, иначе argmin(confidence·count)
keys[слот] ← normalize(mean(q_n[novel])); vals[слот] ← mean(val_proj[novel])   # live index_copy
confidence[слот] ← mean(conf[novel]); count=1; age=0; _n_births += 1
```
Возраст всех слотов `+= 1` в конце каждого вызова (рождённые получают `age=1`).

**Коммит буферов** — после read (`:393-396`): `concept_keys.copy_(keys_eff.detach())`,
`concept_vals.copy_(vals_eff.detach())`; `_step += 1` на каждом вызове.

#### 2.1.4 Чтение и инъекция (`:329-398`)

```
q = q_proj(h); q_n = normalize(q)
tau_read = exp(log_tau_read).clamp(0.1,10) = e ≈ 2.718 (init)
scores = (q_n · normalize(keys_eff))·tau_read
attn = σ(scores)·(1 + softmax(scores/tau_read));  attn ← attn/Σattn   (softmax_free=True)
read = attn @ vals_eff;  read = out_proj(read)
u_gate = σ(κ·(pen − τ_uncert)),  κ=3, τ_uncert=0.5 (init)     # :355-361
c_gate = σ(τ_contra·cos(normalize(read), normalize(h.detach())))  # τ_contra=e (init)
scale  = σ(read_scale) = σ(−4) = 0.01799 (init)               # :373
out = read·u_gate·c_gate·scale
out ← out·min(1, 0.25·‖h‖/(‖out‖+1e-8))                       # B1: потолок 25% локальной нормы
h ← h + out                                                    # stack.py:636 (после M50-cap)
```
[mini-прогон] `‖out‖/‖h‖ = 0.00107` (pen=0.34, случайные hp/h) — на порядок ниже потолка
0.25; градиенты: `q_proj 1.1e-8`, `out_proj 2.0e-8`, `read_scale 2.2e-8`,
`write_q_proj 8.8e-10`, `write_v_proj 1.2e-9` — т.е. **канал записи обучается на 1-2 порядка
слабее канала чтения** (σ(−4)·u_gate·c_gate·bound ≈ 0.0034 — общий множитель).

**Связь с лакуной и hp.** `vals` живут в полном D и читаются `out_proj` (D→D, dense) — инъекция
пишет и в known-bits, и в лакуну; никакого отдельного «лакунного» канала у UCL нет. Вход записи —
`hp` зеркала (K-пространство экспертов) через `shared`; вход чтения — `h`. `u_gate` связывает
чтение с pen зеркала (шкала 0.25–0.46 после B10): при типичном pen гейт ≈ 0.32–0.47, т.е.
концепты **почти всегда полуоткрыты**, а не «включаются по тревоге».

#### 2.1.5 Где UCL в стеке и что гейтит

- Вызов: `stack.py:619-636`, условие `i==0`, `_cached_hp.shape[:2]==h.shape[:2]`,
  `allow_write=True` **всегда** (решение #5: инференс=обучение; изоляция — снапшотом M8).
- `mat_gate[0]` — hard-skip записи при `<0.1` (`concept_layer.py:178`); внутренние update/birth
  гейтятся `self._mature` (0.3/0.1) — **не** mat_gate стека.
- Диагностика: `get_diagnostics()` (`:402-414`) → `losses.py:449-452` пишет
  `mb_l3_births`, `mb_l3_updates` (старые L3-метки); `_cached_birth_gate` — `birth_gate_mean()`.

### 2.2 StreamingMemoryBank — `core/memory_bank.py`

#### 2.2.1 L1Buffer (3 слота, кольцо) — `:90-172`

```
write(summary): slot = первый незаполненный, иначе argmax(age); buf[slot]=summary; age[slot]=0;
                age[остальных] += 1; _write_idx += 1                    # @torch.no_grad
read(h, temp_k):
  q = q_proj(h)                                  (B,L,bridge_dim)
  k = F.normalize(proj(buf))                     (n_slots,bridge_dim)
  v = out_proj(buf)                              (n_slots,D) — НЕ нормирован
  temp = exp(log_tau).clamp(0.1,10)·temp_k
  age_decay = exp(−0.01·buf_age)                 (n_slots,)
  attn = hybrid_gate(q@kᵀ/√bridge_dim, temp);  attn ← attn·age_decay;  attn ← attn/Σattn
  read = attn @ v
```

#### 2.2.2 L2Bank (32 слота, обучение ключей/значений) — `:175-327`

```
write(summary):
  novelty_score = σ(novelty_gate(summary))        # :241 — используется ТОЛЬКО для slot_novelty
  new_key = F.normalize(W_k(summary))·σ(key_log_scale)     (σ(0)=0.5)
  new_val = F.normalize(W_v(summary))·σ(val_log_scale)
  slot: первый незаполненный → иначе consumed-слот с max(age) → иначе max(age)  # :255-269
read(h, temp_k):
  q = q_proj(h); k = keys; v = val_norm(vals)
  temp = exp(log_tau).clamp(0.1,10)·temp_k
  age_decay = exp(−0.01·slot_age)
  attn = hybrid_gate(q@kᵀ/√bridge_dim, temp)·age_decay, нормировка
  read = W_o(attn @ v)
```
**Факт (расхождение с докстрингом):** novelty-гейт вытеснения **не реализован** — `slot_novelty`
нигде не читается, кроме `get_stats()`; вытеснение — consumed → oldest. [mini-прогон]:
`slot_novelty=[0.515,0.514,0.515,0.514]`, вытеснение по кругу `age=[3,2,1,0]`,
`grad novelty_gate = None` (параметры `novelty_gate` **мертвы** — вызов внутри `no_grad`).

#### 2.2.3 StreamingMemoryBank.forward (`:401-477`)

```
is_sep = (tokens == 2)                            # SEP
_can_write = (mat_gate is None) or (mat_gate ≥ 0.3)     # min_write_maturation=0.3
with no_grad: for b, for t: if is_sep: summary = h[b, sent_start:t+1].mean(0)
    → l1.write(summary); l2.write(summary)        # ЗАПИСЬ ПО КАЖДОМУ SEP НА КАЖДОМ СЛОЕ
_tk = 1 + lacuna_k·max(0, lacuna),  lacuna_k=0.5  # M55a: temp·(1+k·rel_excess)
mem_l1 = l1.read(h, _tk); mem_l2 = l2.read(h, _tk)
alpha = exp(_fusion_tau_alpha) ≈ 1 (zero-init)    # U6: per-level τ-modulation
combined = [α₀h; α₁mem_l1; α₂mem_l2]  (B,L,3D);  fused = fusion(combined)  # (D·3→D→D)
scale = tanh(log_scale) ∈ (−1,1);  scale *= (0.3 + 0.7·mean(tau_norm))     # τ-расписание
if training: _last_read = fused.detach()          # :471-473 — только train
if not _can_write: return h                       # полный no-op (даже без чтения)
return h + scale·fused
```
Порядок применения в стеке: **до** блока i (`stack.py:577-581`), т.е. чтение видит вход слоя
(после bridge-инъекции), запись — то же состояние; `h` на входе банка ещё не нормирован
(`_ln` нет) — [mini] `scale·fused` при init: `fusion` last layer zero-init → `fused=0` ⇒
**тождество на старте** (независимо от scale); обучается через CE.

**Многократность (важнейший факт):** банк вызывается в цикле слоёв, поэтому за один forward
на каждом слое с `mat_gate[i] ≥ 0.3` происходят (а) повторные записи всех SEP-предложений и
(б) чтение. [mini-прогон]: один вызов = 2 записи (2 SEP), два вызова (2 слоя) = 4 записи.
При 24 активных слоях одно предложение пишется ~24 раза (summary разные — h разных слоёв),
а L1 (3 слота) кольцуется каждым слоем.

#### 2.2.4 M55a broadening и лакуна

```
lacuna = max(0, _last_lacuna_rel − 1)            # stack.py:404 (1-шагово-старая, rel-excess)
temp   = exp(log_tau).clamp(0.1,10) · (1 + 0.5·lacuna)
```
[mini] при `lacuna=5` `temp` растёт ×3.5 → softmax-часть гибридного гейта размывается
(softmax(scores/τ) → равномернее), sigmoid-часть не меняется. Это единственный канал
«память ↔ лакуна»: broadening поиска по релевантности лакуны, ничего больше.

### 2.3 LogitCacheAttention (R1 / compressed logits) — `core/logit_cache.py`

#### 2.3.1 Хранилище и eviction

- `LogitCache`: `max_entries=64`; training-кольцо `_h_cache` (полные (B,L,D), detach на store),
  inference-кольцо `_logit_cache` (сжатые top-k логиты), `_p_cache` (M18 profile, fp16 (B,L,K)),
  `_kv_h` (write-time k/v), `_position`, счётчики novelty/длин.
- `store(h_or_logits, training, novelty)` (`:118-146`): training → `_h_cache.append(h.detach())`;
  иначе → `_compress(logits)` (sparse top-k, `k=get_k(0)`).
- `_evict` (`:148-183`): hard-cap `max_entries` (FIFO), затем **M34 horizon**:
```
while Σlens > horizon_tokens и len>1:
    age_i = Σ длин записей новее i;  retention_i = novelty_i·exp(−age_i/τ),  τ = horizon/2 = 256
    удалить запись с min(retention), кроме самой новой
```
- `get_k` (`:96-101`): `k = mean(BASE_K=[128,96,80,64])·σ(vsa_scales).mean = 92·σ(0)=46` [mini] 46.
- `bit_profile(logits)` (`:362-365`): `tanh(logits/10) @ codes_t / sparsity` — (…,V)→(…,K).
- `clear()` (`:253-263`) чистит всё, включая `_p_cache`, `_kv_h`, счётчики.

#### 2.3.2 LogitAttention (`:284-455`)

```
training (h-режим):
  Q = q_proj(h)                                   # live
  k_new = k_norm(k_proj_h(h)); v_new = v_norm(v_proj_h(h))     # live
  cache.push_kv(k_new.detach(), v_new.detach())   # write-time encodings
  K = cat(kv[:-1] + [k_new]); V = cat(kv[:-1] + [v_new])       # newest LIVE, older detached
inference (logits/profile-режим):
  prof = _p_cache (M18) или bit_profile(retrieve(logits))
  K = k_norm(k_proj_l(prof)); V = v_norm(v_proj_l(prof))       # K_bits→D
positions = arange(M) % max_cache_len(1024);  K += pos_enc(positions)
attn = softmax(QKᵀ/(√head_dim·τ)),  τ = exp(log_tau).clamp(0.1,10)
output = out_proj(attn @ V)
gate = σ(cache_gate(h))                           # MLP D→D/4→1, bias −10 ⇒ 4.54e-5 (init)
output = gate·output + (1−gate)·h                 # тождество на старте (в пределах 5e-5)
```

#### 2.3.3 LogitCacheAttention / R1 (`:458-577`)

```
use_inference_mode = training and ratio>0 and logits≠None and rand(1)<ratio   # 5% default
if use_inference_mode: _r1_steps += 1
store: train-h-режим → cache.store(h, training=True)
       иначе mode='profile' и codes_t → store_profile(bit_profile(logits))
             иначе → store(logits, training=False)
h_aug = attention(h, cache, training=(training and not use_inference_mode))
if (not training or use_inference_mode) and codes_t:
     cached_h = logit_to_hidden(bit_profile(новейшие логиты/профиль))  # K_bits→D
     h_aug += cached_h
augment(h, novelty, logits, training): out = forward(...)[0]; return out if finite else h
```
- Стек вызывает `augment(h, novelty=mean(pred_errs), logits=_last_logits if training else None,
  training=True)` — **жёстко `training=True`** (`stack.py:789-791`): в eval работает h-режим
  (store h + attend), R1 не срабатывает (logits=None).
- `_last_logits` — 1-шагово-старый detached (B,L,V) из `observe_output` (`stack.py:1017-1025`);
  комментарий: ~117 MB на один шаг. [mini] при `ratio=1.0`: `_r1_steps=1`, `_logit_cache`=1,
  `p_cache`=0 (mode='topk'), σ(−10)=4.54e-5, `‖augment−h‖=0.095` при норме входа ~2.9 (т.е. ~3%).
- `_r1_steps` — persistent буфер (`:484`), «definitive live check» (M56c).
- Очистка: document boundary (`train.py:491-492`), resume R6 (`train.py:368-370`),
  eval до/после (`train.py:702-704,743-744`), flush каждые 495 шагов (`train.py:677-678`).
- Что даёт: выравнивание train/inference — 5% тренировочных шагов кэш пишет/читает
  **compressed logits** (top-k=46 из V) вместо h, т.е. модель учится на том представлении,
  которое увидит в inference-режиме; `logit_to_hidden` (K_bits→D, xavier gain 0.01) даёт
  дополнительный вклад в h.

### 2.4 Intent-шина (`stack.py:76-115, 341-366, 406-422, 470-530`)

```
probe_out = intent_probe(h) → (B,L,G,K_max)          # K_max = max mirror.k по слоям
если _last_salience (B,L,1) формы совпадают: probe_out *= _last_salience.unsqueeze(-1)
fresh_i = mean_{B,L}(probe_out)                      # (1,1,G,K_max), LIVE (граф к probe)
carried_i = intent_streams[i].detach()               # прошлый шаг, detached
α_i = 1 − 1/τ_l,  τ_l = max(τ_l, 2)                  # τ-horizon (audit #4)
_expert_mod = (2σ(_w_alpha_expert)−1)·(2·tau_norm_i − 1)
α_per_expert = clamp(α_i·(1+_expert_mod), 0, 0.999)
intent_streams[i] = α·carried + (1−α)·fresh_i
_bus_running  += fresh_i + α·carried_i;   _bus_le_carried += carried_i
bus_i = (_bus_running + (_bus_sum − _bus_le_carried))/n_layers   # fresh j≤i, carried j>i
_bus_rms ← EMA(0.99/0.01) от RMS(bus_i);   _last_bus = bus_i/_bus_rms
intent_i = bus_i[..., :k_i]                          # в зеркало (intent=...)
```
- `_bus_sum = Σ carried` (detached, `:419`); `_bus_carried` — список; `_intent_stream` —
  список (n_layers,) detached-тензоров (1,1,G,K_max), сохраняется в конце forward (`:697-699`).
- **Сalience**: `compute_salience(logits) = σ(logits).norm(dim=-1,keepdim=True) / mean()`
  (`stack.py:1005-1015`) — L2-норма вектора сигмоид-вероятностей по V, нормированная к
  среднему 1 (O(1) per-position вес). `observe_output` (`:1017-1025`) сохраняет
  `_last_salience` (detached) и `_last_logits`; вызывается в `train.py:507` **после** forward
  (1-шаговая задержка). [mini] после нормировки `mean=1.0`, диапазон [0.983,1.012] для
  случайных логитов V=1820.
- `mat_gate` intent **не масштабирует** (`stack.py:527-530`); зрелость входит через
  `bridge_glu_net(delta)·maturity` и `expert_gate` зеркала (зона 3).
- В голову: `bus_bias = bus_head_proj(_last_bus.expand(B,L,·,·).reshape(B,L,−1))` →
  `zt += bus_bias` (`losses.py:20-29`; `embedding.py:363-366`) — zero-init per-bit stencil
  (U12: вход нормирован `_bus_rms`).
- `_last_bus` — detached (`:699`), значит градиент в `bus_head_proj` идёт от CE, в probe — от
  собственного self-term.

### 2.5 SemanticBridge (`core/bridge.py`)

```
probe: (D→bridge_dim→bridge_dim) GELU; s_l = probe(h.detach())            # :117-119
inject_layer(i, h, maturity, tau_norm):                                  # :121-169
  neigh = [stream[i]] + ([stream[i−1]] if i>0) + ([stream[i+1]] if i<n−1)
  w = σ(stream_log_weights[:|neigh|]);  w ← w/Σw
  combined = Σ w·neigh
  inj_strength = σ(_inj_alpha)·tau_norm + σ(_inj_beta)·(1−tau_norm)      # σ(1)=0.731, σ(0.5)=0.622
  scale = tanh(stream_log_scale)·maturity·inj_strength                   # init 0 ⇒ identity
  inj = scale·stream_proj(combined)
  h ← h + inj.view(1,1,D);   inj_ratio[i] = clamp(‖inj‖/‖h‖, 0, 1) (no_grad)
update_stream(i, s_l): stream[i] ← 0.9·stream[i] + 0.1·mean_{B,L}(s_l.detach())   # :171-175
loss(y, embed_fn):                                                       # :187-241
  tgt_raw = embed(y[:, :-1]) @ emb_proj_Wᵀ        # emb_proj_W — fixed orthogonal (seed 9)
  tgt = normalize(tgt_raw − _tgt_mean);  _tgt_mean — EMA (train-only)
  для каждого s_l: pred = normalize(s_l[:, :-1]);  InfoNCE(pred, tgt; temp=exp(nce_log_temp)=0.296)
     с маской false-negative по совпадению токенов
  loss = mean(InfoNCE) + 0.1·mean_pairwise_cos(layer_means)
  bridge_loss_init ← max(init, loss);  bridge_loss_ema ← lerp(·, 0.01)
readiness = σ((sat−0.3)/0.2) − σ(−1.5),  sat = clamp(1 − ema/init, 0, 1)  # :100-111
```
- `bridge_stream` — persistent буфер (n_layers, bridge_dim), EMA времени; при document
  boundary `train.py:488` зануляет; `reset_stream()` есть, но в reset_cache не вызывается.
- Проба читает **detached** h (градиент моста в ствол не идёт — M2-доктрина), инъекция —
  живой граф в `stream_proj`, `stream_log_scale`, `_inj_alpha/beta`, `stream_log_weights`.
- `readiness()` используется только trust-механизмом оптимизатора (`train.py:586-587`,
  `eva_optim.py:39,79,128`) — на поток не влияет.

### 2.6 LayerBridgeGate (`core/layer_bridge_gate.py`)

```
SpectrumGate: gate = σ(logits)·(1+softmax(logits/τ));  τ = τ_ext·exp(log_tau), оба clamp(0.1,10)
_effective_tau(mat) = exp(log τ_max + (log τ_min − log τ_max)·mat)   # geometric: mat=0→5, mat=1→0.3
layer_gate(i, health, maturation, global_ready, tau_external):
    pre-ready:  return maturation
    ready:      gated = SpectrumGate_i(health, τ_ext);  gate = mean(gated)·maturation; clamp(0,2)
diagnostics (6 features): [pen, gate_l1, ‖pred_k‖/1000, bridge_contrib, expert_entropy, 1−gate_l1]
```
- **В forward стека результат `layer_gate` не используется** (`stack.py:555-559`: `_gate_i`
  вычислен и забыт; комментарий `:560-566` — «for the diagnostics dashboard only»). На поток
  LBG не влияет вообще.
- Диагностика в стеке — **inline-дубль** (`stack.py:642-668`), где `_diag[3] = 0.5` (константа;
  `bridge.inj_ratio` не подставляется), а энтропия считается по всему батчу (в методе LBG —
  per-position; расхождение, отмеченное зоной 3).
- Единственный обучающий сигнал LBG — `lbg_diversity` (`losses.py:432-437`): `softmax(gates)`
  по слоям, `(log n − H)/log n`, градиент в `gates[l].log_tau` (через `_diag_grad`-копию).
  При `global_ready=False` diversity=0 и `log_tau` не обучается.
- `lbg_tau` в логах (`losses.py:421`): при ready = `_effective_tau(mat_l)` по слоям; при
  не-ready = `tau_max=5.0`. Связь с зоной 3: `mat=0.82 → 0.4978` — **точно** наблюдённое M47
  `lbg_tau→0.50` (подтверждено [mini]: `exp(log5+(log0.3−log5)·0.82)=0.4978`).

### 2.7 Reasoning (`core/reasoning.py` + `stack.py:852-967`)

**ReasoningMemory** (`reasoning.py:15-97`): буфер (B, max_steps=8, D); `current_step = step_encoder(h[:, -1:])`;
`attn = σ(qkᵀ/√D)·mask(count)`; `context = attn@v`; `output = output_proj(current+context)`;
если `count=0` → `output=current_step`; запись — сдвиг буфера + `current_step.detach()` в строку `count`.

**ReasoningGate** (`:100-172`): `logits = proj(h) + know_proj(know) + r_proj(r)`; `bias[0]=+10`
(гейт 0 ≈ tanh(10)≈1), остальные zero-init; выход `tanh(logits)`; есть `logits()` для ST.

**Петля `_adaptive_reasoning`** (`stack.py:852-967`):
```
K = max(1, round(8·tau_norm_reasoning))            # U2: τ-бюджет
s = 1 − exp(−reasoning_enabled_step/1000) (floor 1e-3)   # ramp (property :969-975)
know = _knowledge_signal(h)   # (B,8): p1/p2/margin/entropy last + средние + mem_agr + ‖h‖
for i in 0..K−1:
  r_i = ReasoningMemory(h, buf, count)             # кандидат
  l_i = gate.logits(h_acc, know, r_i);  a_i = tanh(l_i)  (i>0: ST: a_i = l_i + (a_i−l_i).detach())
  run = prev_open ≥ 0.5;  commit = run & (mean(a_i)≥0.5)   # буфер коммитится только при открытом гейте
  r_contrib = r_i (i=0) | normalize(r_i) (i>0)
  w_soft = 1 (i=0) | σ(20·clamp(Δconf/conf_base)) (i>0)    # валидация прироста уверенности головы
  contrib = a_i·r_contrib·w_soft·run
  weighted += contrib;  denom += a_i⁺·w_soft·run
  accum = weighted/denom.clamp(min=0.5);  h_acc = h + s·accum;  prev_open = mean(a_i)
_reasoning_gates[:K] ← mean(a_i)·w_soft·run (диагностика, no_grad)
```
- Включается при `explicit_reasoning=True` (cfg default **True**) и `s>0`; `reasoning_enabled_step`
  инкрементируется в train loop (`train.py:597-598`), восстанавливается из чекпойнта
  (`train.py:417,446`); в eval/val он остаётся последним тренировочным.
- Работает **после `final_norm`** (`stack.py:710-715` → `717-731`), т.е. на RMS≈1-шкале;
  `h = h_acc` заменяет h (в отличие от UCL-инъекции, здесь не «+» к исходному h, а результат
  взвешенного накопления при `s≈1`).
- Взаимодействие с гейтами/лакуной: `_knowledge_signal` и `_last_conf` — прямые вызовы головы
  (`stack.py:802-850`) на каждом reasoning-шаге (до K+1 лишних forward головы); лакуна в
  reasoning не входит.

### 2.8 Триада (re-pass, inference) — `stack.py:733-773`

```
if triad_reason and not training and step is not None and depth < 3:
    conf = mean(_last_conf(h));  если conf < 0.5: h2 = forward(h, ...) (рекурсивно)
    h = 0.5h + 0.5h2;  _triad_passes = depth+1
    восстанавливаются только bridge._preds и bridge.bridge_stream
```
Проводится **только** в inference/generation (`not self.training` и `step≠None`). Не
восстанавливаются: банк, UCL, logit cache, `_last_read`, `_last_salience`, `_intent_stream`,
`_last_bus` — ре-проход дублирует записи в банк и кэш (см. §6).

### 2.9 Градиентные пути и detach-барьеры зоны 4

| Величина | Граф | Куда идёт градиент |
|---|---|---|
| UCL read: `q_proj/out_proj/read_scale`, `attn` | live | в h (после блока 0) → CE |
| UCL write: `write_q_proj/write_v_proj` через `keys_eff/vals_eff` | live (index_copy) | в write-проекции; [mini] ~1e-9..1e-10 |
| `concept_keys/vals` буферы | detach на commit | не обучаются (by design, M6) |
| `_cached_hp/_cached_gate/pen` вход UCL | detached | только forward-потребители |
| Bank `fusion`, `log_scale`, `_fusion_tau_alpha`, L1/L2 проекции, `log_tau` | live (read-путь) | CE через `scale·fused` |
| Bank `keys/vals/buf` | буферы | не обучаются |
| Bank `novelty_gate` | **мёртв** (no_grad-контекст + `@torch.no_grad` write) | — |
| Logit cache `q/k/v/out_proj`, `log_tau`, `cache_gate`, `logit_to_hidden`, `k_proj_l/v_proj_l` | live | CE через augment; newest kv-entry live, старые detached |
| Bridge `probe` | detached вход h | только bridge-loss (InfoNCE) |
| Bridge `stream_proj/stream_log_scale/_inj_*/stream_log_weights` | live | CE через инъекцию |
| Bridge `bridge_stream` | EMA (no_grad) | — |
| Intent `intent_probe` | live (fresh_i self-term) | CE через `intent_i`→зеркало; carried detached |
| Intent `bus_head_proj` | live | CE через `zt += bus_bias` |
| Intent `_w_alpha_expert` | live | CE через carry-часть |
| Reasoning `reasoning_memory`, `reasoning_gate` | live | CE через `s·accum` (при `s>0`) |
| LBG `gates[*].log_tau` | live только в `lbg_diversity` | aux |
| UCL `_mature`, `concept_count/age/confidence` | no_grad | — |

### 2.10 Режимы, снапшоты, очистка

- **Train/eval:** UCL пишет в обоих режимах (решение #5); банк пишет в обоих (M33); bridge
  инъекция/проба/EMA — в обоих; logit cache в eval — h-режим (стек жёстко `training=True`);
  reasoning — в обоих (в eval `reasoning_enabled_step` тренировочный); триада — только
  inference; `_last_read` — только training.
- **Снапшот** `snapshot_runtime_buffers` (`stack.py:1178-1220`) покрывает: все буферы (включая
  non-persistent) UCL/банка/кэша/моста/интента + `_last_bus/_intent_stream/_reasoning_buffer/
  _reasoning_count` + per-layer `_cached_hp/_cached_pred_k/_cached_pred_error_norm/_cached_gate/
  _cached_usefulness/_traj_state`. **Не покрывает**: `_last_salience`, `_last_logits`,
  `_last_read`, `_last_conflict`, `_triad_passes`, `_layer_diagnostics`, `_preds` (но `_preds`
  сбрасывается `start_forward`).
- **`reset_cache`** (`stack.py:1142-1176`): чистит logit cache, `_mlp_cnt`, non-finite буферы,
  mirror-кэши (+`_traj_state` — дефект зоны 2), `reset_stream_bufs`. **Не чистит**: банк, UCL,
  `bridge_stream`, `_last_read`, `_last_salience`, `_last_logits`, `_intent_stream`, `_last_bus`,
  reasoning-буфер. При document boundary (`train.py:485-494`) чистятся: state/gs, `bridge_stream`,
  `memory_bank.reset()`, `logit_cache.cache.clear()`, `reset_reasoning()` — **UCL и intent-шина
  сохраняются** (концепты — долговременная память, by design).
- **evaluate** (`train.py:690-750`): снапшот → `logit_cache.cache.clear()` → по каждому
  hold-out документу `reset_reasoning` + `memory_bank.reset()` → `adaptive=False`,
  `tokens=x`; после — `cache.clear()` и restore. `UCL` в eval работает и покрыт снапшотом.

---

## 3. Что зона делает с входным контрактом (проверка/преобразование/потеря)

1. **`h`**: банк читает/пишет **ненормированный** h (нет `_ln`); UCL читает h как есть
   (Q-проекция), запись — из `hp`; bridge — `h.detach()`; reasoning — после `final_norm`.
2. **`pen`**: UCL использует в двух местах (`conf`, `u_gate`); пороги калиброваны под
   **до-B10-шкалу** (комментарий B12: `exp(1.0)=2.72` делал `u_gate` навсегда закрытым,
   перебазирован на `ln 0.5`). Остаточный риск: `u_gate` при pen∈[0.25,0.46] лежит в [0.32,0.47]
   — полуоткрыт, слабо различает режимы.
3. **`mat_gate`**: bank — жёсткий 0.3 (в стеке и внутри), UCL — 0.1 (только write-skip);
   bridge — линейный множитель; LBG — `gate=mat` до готовности. Intent — не гейтится.
4. **`tokens`**: bank требует `tokens≠None`; в `live_inference.think` (первый шаг) `tokens=None`
   → банк пропущен; в `generate` передаётся `_pending_tokens`.
5. **`_last_salience`**: применяется к probe только при совпадении (B,L); в eval не
   обновляется (observe_output не вызывается) — используется тренировочное значение.
6. **Потери**: `_cached_usefulness` не используется зоной 4; `_mem_dir` (train-only) в eval
   устаревает; `layer_gate` LBG не влияет на поток; `projector_signals`/`collective_stats`
   (`stack.py:986-998,1323-1341`) мертвы (`EVABlock.collective = None`, `block.py:388`);
   `live_inference.intent_state` всегда None (`_last_intent_state` не существует —
   grep: только live_inference.py:197,220).

---

## 4. Передача следующему (агенту 5: голова)

### 4.1 Точный выходной контракт зоны 4

| # | Величина | Форма / dtype | Шкала (mini/аналит.) | Инварианты и оговорки |
|---|---|---|---|---|
| 1 | Финальный `h` (вход головы) | (B,L,D) fp32 | после `final_norm` RMS≈1, затем reasoning `h += s·accum` (s→1), затем augment | порядок: final_norm → reasoning → триада (inference) → cache-augment |
| 2 | Cache-augment | (B,L,D) | `gate·attn + (1−gate)·h`, gate=σ(cache_gate(h)) | bias −10 ⇒ **σ(−10)=4.54e-5** на init; [mini] ‖Δh‖≈0.095 при ‖h‖≈2.9 |
| 3 | `_head._mem_dir` | (B,L,D) detached | `bank._last_read = fused` (до `scale`), **последнего слоя** с активным банком | **обновляется только в training**; в eval — последнее тренировочное значение или None; CE-путь (2D h) temper не применяет (shape-check `_md.shape[:-1]==u.shape[:-1]` не проходит для `(B,L)` vs `(N,1)`) |
| 4 | bus_bias (intent→голова) | (B,L,K_head) | `bus_head_proj(_last_bus_norm)`, zero-init | добавляется к `zt` per-bit (`embedding.py:363-366`), live-градиент в `bus_head_proj`; только при `intent_bridge` и `_last_bus≠None` |
| 5 | UCL-инъекция | (B,L,D) | `h += col_out` после блока 0; `‖col_out‖/‖h‖` [mini] ~0.001, потолок 0.25 | уже внутри `h`; отдельного канала в голову нет |
| 6 | `_last_lacuna_rel` | scalar (detached) | из головы, 1-шагово-старый | обратная связь: bank broadening `temp·(1+0.5·rel_excess)`; в первый шаг → 0 |
| 7 | Телеметрия `_cached_losses` | — | `mb_l1/l2_overwrites`, `mb_l2_consumed`, `mb_l3_births`, `mb_l3_updates`, `mb_scale` (`losses.py:439-453`); `layer_gate_mean/std/min/max`, `lbg_global_ready`, `lbg_diversity` | LBG-диагностика сбрасывается `stack._layer_diagnostics = {}` после каждого compute_losses |
| 8 | `_r1_steps` | long (1,) persistent | число срабатываний R1 (5%) | в чекпойнте; «definitive live check» (M56c) |
| 9 | `_reasoning_gates` | (max_steps,) non-persistent | `mean(a_i)·w_soft·run` | читается логом `train.py:608-612` |
| 10 | `_triad_passes` | int | 0..3 | inference-only |
| 11 | `concept_layer.get_diagnostics()` | dict | maturity, n_active, births, updates, birth_gate, novelty_gap, tau_birth, tau_read, confidence_mean | в `losses` идут только births/updates (метки L3) |
| 12 | `bank.get_diagnostics()` | dict | l1/l2 overwrites/fill/consumed/novelty_mean/age_mean/key_scale/val_scale/mem_scale | `novelty_mean` — единственный след мёртвого novelty-гейта |
| 13 | `bridge.inj_ratio` | (n_layers,) non-persistent | clamp(‖inj‖/‖h‖,0,1) | потребитель — только diagnostics-dashboard |
| 14 | `_last_conflict` (голова) | scalar | χ = relu(0.3−cos(h_implied, mem_dir)) | обновляется только на 3D-пути forward головы при `self.training`; CE-путь его не видит |

### 4.2 Что голова **не** получает

- `hp/pen/mlp_mod/mem_mod` напрямую не пробрасываются (только через h и через `u_gate/conf`
  внутри UCL, которые уже свёрнуты в `col_out`).
- `_cached_usefulness`, `_cached_gate` в голову не идут (только UCL-write).
- Направление чтения банка доступно только как `_mem_dir` (train-only) — в eval temper
  выключен (`_temper_active=False` при `step=None`), поэтому «конфликт голова↔память» в
  валидации не измеряется.

---

## 5. Связи с другими зонами

- **Зона 1 (коды/эмбеддинг):** logit cache получает **тот же объект `codes`**
  (`stack.py:202` ← `lm_head.codes`); `bit_profile` использует кодовую книгу головы
  (`K_bits=code_dim`). UCL-слоты в bridge_dim/D — от кодовой геометрии не зависят.
- **Зона 2 (ядро):** UCL-инъекция и bank-read **не покрыты M50/M51** (зона 2, §6.5) — их вклад
  добавляется после `branch_cap` и до следующего `_stream_cap(1e3)`; UCL сам ограничен 25%
  локальной нормы, банк — амплитудой `scale·fused`. `reset_cache` без `_traj_state` (зона 2)
  усугубляется тем, что reset_cache не чистит и зону 4 (см. §6).
- **Зона 3 (зеркало/матчурация):** UCL читает `_cached_hp/_cached_gate/pen` (detached,
  1-шагово-старые) и `_residual_var_ema`; bank — `mat_gate` (0.3); intent — `intent_alpha`
  зеркала (τ-авторитет); LBG `lbg_tau` = `gate_tau(mat)` — таймер зоны 3, не сигнал памяти.
  `pen_init=1.0` (зона 3) означает, что `mat_gate` приходит «по часам»: bank/UCL открываются
  по расписанию, а не по компетентности.
- **Зона 5 (голова):** см. §4; дополнительно: `_last_lacuna_rel` и `_mem_dir` — двусторонний
  контур память↔голова (broadening и temper), `bus_bias` — однонаправленный intent→голова.
- **Зона 6 (обучение):** `_role_lr_mult` (`adaptation.py:203-220`) для `memory_bank.*`,
  `concept_layer.*`, `logit_cache.*` = **1.0** (default; в `stack.param_groups` bridge-группа —
  базовый LR); `eva_optim._TRUST_MEM = ("memory_bank.",)` (trust = |maturation_readiness|);
  aux: `bridge_conn`, `mem_tau_reg`, `lbg_diversity`, `intent_tau`; `release_step_graph` НЕ
  освобождает `_pred_loss_term`-класс кэшей зоны 4 (их и нет), но `_last_logits` (117 MB)
  живёт ровно один шаг; R1 использует глобальный `torch.rand`.

---

## 6. Открытые вопросы и риски для следующего агента

1. **UCL-градиенты микроскопические.** [mini] `write_q_proj ~9e-10`, `q_proj ~1e-8` при
   `σ(−4)=0.018` (read_scale), `u_gate·c_gate≈0.19` и bound-множителе. При LR-роли 1.0 и
   Adam-масштабировании это ~1e-8-нормы шагов: канал записи может остаться «identity» очень
   долго (B1-доктрина сознательна, но замер «wake-up» в реальном прогоне нужен).
2. **`_mature` — второй таймер.** Растёт от стабильности `resvar` зеркала; cold-start
   `resvar>0` → `mat≈0.007` (запись выключена), `resvar=0` → `mat=1` (запись сразу). Это
   независимый от `mat_gate` контур: UCL может быть «зрелым» (update/birth идут), пока
   `mat_gate[0]<0.3`; и наоборот. Семантика `resvar` после B10 не перекалибрована (зона 3, §6).
3. **Банк: novelty-гейт вытеснения отсутствует.** Докстринг обещает novelty-based write
   gating; фактически вытеснение — consumed→oldest, `slot_novelty` — мёртвая диагностика,
   `novelty_gate` — мёртвые параметры (градиент None). Слот с высокой новизной может быть
   вытеснен следующим предложением.
4. **Банк: многократная запись на каждом слое.** За один forward одно SEP-предложение пишется
   столько раз, сколько активных слоёв (до 24), с разными `h`-summary; L1 (3 слота) при этом
   прокручивается почти полностью. Семантика «последние K предложений» не выполняется —
   фактически «последние K×(число активных слоёв) записей». Потребитель `get_diagnostics`
   (`l1_fill/l2_fill`) этого не различает.
5. **`_last_read`/`_mem_dir`:** обновляется только в training; в eval `_mem_dir` — stale/None,
   temper в eval выключен (`_temper_active=False` при `step=None`), а CE-путь (2D h) не
   проходит shape-check и **не темперируется вообще**. Т.е. M55a-конфликт «голова↔память»
   действует только на 3D-forward головы (observe_output) и только при `step≥1045`.
   `_mem_dir` при этом — направление чтения **последнего** активного слоя (перезапись в цикле).
6. **`reset_cache` неполон для зоны 4:** не чистит банк, UCL, `bridge_stream`, `_last_read`,
   `_last_salience`, `_last_logits`, `_intent_stream`, `_last_bus`. При NaN-rollback (watchdog
   вызывает `reset_cache`) рабочая память и концепты сохраняют «отравленное» содержимое;
   снапшот/restore eval-изоляции их покрывает, но rollback — нет.
7. **Триада загрязняет память:** ре-проход (inference, conf<0.5) пишет в банк и logit cache
   повторно и не восстанавливает их; восстанавливаются только `bridge._preds`/`bridge_stream`.
   При `triad_max_passes=3` возможны 2-3 дублирующие записи одного контента (плюс reasoning
   прогоняется дважды).
8. **Reasoning: K от `tau_norm_reasoning`** (`round(8·mean tau_norm)`): при tau_norm<0.0625
   K=1. Каждый шаг делает forward головы (`_knowledge_signal` + `_last_conf`) — до K+1
   лишних головных проходов за forward; в eval reasoning работает на тренировочном
   `reasoning_enabled_step` (ramp s≈1 к 5k шагов) — вклад `s·accum` в финальный h не
   изолирован снапшотом (буфер reasoning покрыт, но `h` — нет, это и есть выход).
9. **LBG декоративен для потока:** `layer_gate` в forward не используется; диагностика в стеке
   — inline-дубль (`_diag[3]=0.5`, batch-entropy); обучается только `lbg_diversity` при
   `global_ready`. `lbg_tau` в логах — `_effective_tau(mat)` (при ready) — не сигнал качества
   моста.
10. **Intent-каналы:** `intent_state` в `LiveInference` всегда None (`_last_intent_state`
    отсутствует); `_last_salience`/`_last_logits` не покрыты снапшотом (в eval не мутируются,
    но при rollback сохраняются); `bus_head_proj` получает градиент от CE, `_last_bus` detached.
11. **Мёртвые/инертные каналы зоны:** `projector_signals`, `collective_stats`,
    `log_tau_gate` (UCL, не читается), `novelty_gate` банка, `inj_ratio` (только dashboard),
    `_mlp_ratio` (зона 3), `w_sal` без `observe_output` (зона 3).
12. **R1-статистика:** `torch.rand` — глобальный RNG (не per-model generator); при resume
    последовательность R1-шагов зависит от порядка вызовов и сохраняется через `rng`-state
    чекпойнта (`train.py:661`), но в evaluate `torch.rand` не вызывается (logits=None).
13. **Horizon-эвикция слабо различает окна:** retention = novelty·exp(−age/τ), novelty =
    mean(pen)≈0.25–0.46 (узкий диапазон), τ=256; при коротких окнах (L=64) вытеснение почти
    всегда по возрасту — фактически FIFO, как и раньше (M34 задумывался иначе).
14. **UCL `tau_norm` для birth:** передаётся только `tau_norm[0]` (`stack.py:634`) — birth-порог
    привязан к первому слою, хотя `_mature` — глобальная величина.
