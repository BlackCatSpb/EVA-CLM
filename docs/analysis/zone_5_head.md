# Зона 5 — Голова: SigmoidCodedHead, гибридный гейт, лакуна, фантомы, SRL, tempering, контур памяти (M52–M56)

**Агент:** 5 из 6 (нерушимая цепочка, `docs/analysis/CHAIN.md`).
**Дерево:** commit `362a224` (M53e), torch 2.13.0+cpu, Python 3.14.6, Windows.
**Вход зоны:** финальный `h` (B,L,D) после `final_norm` (RMS≈1), reasoning, триады и
logit-cache augment; `_mem_dir` (направление чтения банка, training-only); `bus_bias`
(intent→голова); `_last_lacuna_rel` (1-шагово-старый); `codes`/`readout` (тай с
`PartitionedEmbedding.basis`).
**Выход зоны:** логиты (B,L,V) / (N,1,V) или log-probs по цели; CE; aux-термы
`head_wall`, `phantom_l1`; телеметрия (`lacuna*`, `sat`, `conflict`, `srl_*`, `ph_*`);
буферы фантом-банка в чекпойнте.
**Предыдущие отчёты:** прочитаны полностью (`zone_1_codes_embedding.md:1-595`,
`zone_2_block_core.md:1-716`, `zone_3_mirror_mlp.md:1-647`, `zone_4_memory.md:1-663`);
их выводы разобраны явно (см. §1.2, §1.3, §3, §5, §6).

**Методика (правило полигона соблюдено).** Продакшн-модель не инстанцировалась,
чекпойнты не грузились. Анализ — чтение кода (`file:строка`). Вычисления выполнены
только на мини-геометрии `SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16,
code_sparsity=4, vocab=1820)` (`tests/test_gradient_flow.py:21`): голова
`SigmoidCodedHead(EVAConfig(**SMALL))` без ствола (`%TEMP%\opencode\zone5_head_mini.py`)
и `EVAStack(SMALL)` для инвентаря `state_dict`; все такие числа помечены
**[mini-прогон]**. Числа аудитов/README помечены **[M52a]/[M53d]/[M53e]** (цитаты
`README.md:390-472`, `tests/test_m52a_gate_fixes.py:1-9`, commit `0be800d`), выводы
предыдущих зон — с их метками ([audit 02a], [audit 02b]).

---

## 1. Граница зоны и входной контракт

### 1.1 Где живёт голова и кто её вызывает

Голова строится стеком и делит readout с эмбеддингом (`stack.py:33`):

```python
self.lm_head = SigmoidCodedHead(cfg, embed_basis=self.embed.basis, rope=self.embed.rope)
```

Точек вызова головы за один тренировочный шаг **четыре** (это критично для
телеметрии/счётчиков, §2.4, §2.6):

| # | Вызов | Строка | Форма h | bus_bias | temper |
|---|---|---|---|---|---|
| 1 | `_knowledge_signal` (reasoning, при `explicit_reasoning` и `s>0`) | `stack.py:803` | (B,L,D) | нет | **да** (если shapes и `_mem_dir` совпали) |
| 2 | `_last_conf` (валидация шага reasoning, i>0; триада) | `stack.py:845`, `934`, `748` | (B,1,D) | нет | нет (shape-check) |
| 3 | `observe_output(model.lm_head(out))` | `train.py:507` | (B,L,D) | нет | **да** (shapes совпадают) |
| 4 | `compute_losses → log_probs_for_target` (CE-путь) | `losses.py:28-29` | (N,1,D), N=B·L | **да** | нет (shape-check) |

`_knowledge_signal`/`_last_conf` обёрнуты в `torch.no_grad()` (`stack.py:802, 844`),
но мутируют состояние головы (телеметрия, фантом-банк, счётчик SRL) — см. §3.4.

Порядок в тренировочном шаге (`train.py:501-508, 544, 594`):

```
h = model.embed_tokens(x)                 # autocast только здесь (train.py:501-502)
out, state, gs, _ = model(h, state, ...)  # внутри — вызовы головы #1/#2 (reasoning)
model.observe_output(model.lm_head(out))  # вызов #3: salience + _last_logits (R1)
ce_loss, aux_dict = model.compute_losses(out, y)  # вызов #4: CE (и последний _last_u/_last_p)
balancer.backward(ce_s, aux_s, model.parameters(), phase_model=model)
optimizer.step(); model.release_step_graph()      # отцепляет _last_u/_last_p (stack.py:1088-1094)
```

Следствие: `_last_u`/`_last_p`, которые читают aux-термы `head_wall`/`phantom_l1`
(`losses.py:499-513`), — это значения **последнего** вызова, т.е. CE-пути. Если порядок
вызовов изменится (например, `observe_output` после `compute_losses`), стена и L1 будут
считать другую `u`/`p` — контракт порядка зафиксирован здесь.

### 1.2 Что именно приходит от агента 4 (проверка контракта `zone_4_memory.md:552-580`)

| Величина агента 4 | Как используется головой | Проверка |
|---|---|---|
| Финальный `h` (B,L,D) RMS≈1 | `z = Σ⟨h_сегмент, R⟩`; лакуна `e_l = h − recon` | порядок final_norm → reasoning → триада → augment подтверждён (`stack.py:710-791`) |
| `_mem_dir` (B,L,D) detached = `bank._last_read = fused.detach()` **последнего** активного слоя | tempering M55a: `cos(ĥ_impl, _mem_dir)` | `memory_bank.py:471-473` — **только training**; `stack.py:582-584` перезаписывает в цикле слоёв |
| `bus_bias` (B,L,K) = `bus_head_proj(_last_bus)` zero-init | `zt += bus_bias` (`embedding.py:363-366`) | передаётся **только** из `losses.py:22-27` (CE-путь) |
| `_last_lacuna_rel` (scalar, detached) | broadening банка: `lacuna = max(0, rel−1)` (`stack.py:400-404`) | только при `step ≥ 1045` и `step is not None` |
| `_last_bus` (1,1,G,Kmax) detached | вход `bus_head_proj`; в голову — через `bus_bias` | `stack.py:699` |
| `codes` (V,K) — тот же объект, что у `logit_cache` | `u @ codes.T`; SRL shortlist; bit_profile кэша | `stack.py:202`, `logit_cache.py:227` |
| `_cached_usefulness`, `hp`, `pen`, `mlp_mod`, `mem_mod` | в голову **не пробрасываются** (только через h) | подтверждено; `pen` влияет лишь косвенно (VSA-decay/UCL) |

### 1.3 Что приходит от зон 1–3 (учёт выводов)

- **Readout — тай с basis (з1, `zone_1:304-311`).** Один `nn.Parameter` на кодирование и
  декодирование; голова видит ровно K сегментных скаляров `z_k = ⟨h_k, R_k⟩`. При
  K=64, D=2560: видимых измерений 64 (2.5 %), лакуна 2496 (97.5 %).
- **Эрозия margin (з1, F2A-09).** Нормы строк `basis` на чекпойнте упали 1.000 →
  [0.5146, 0.6423]; для головы это масштаб `z` (margin кодов сжимается ~2.4–3.8×),
  у блока этого канала нет (`zone_2:626`). Никто не логирует `‖R_k‖`.
- **Лакуна заливается стволом с блока 0 (з2, §3.2).** После блока 0 lacuna-share =
  0.9509, после блока 1 = 0.9432 → `ell = √0.9432 ≈ 0.971`, known-share ≈ 0.238
  (prod-комментарий `embedding.py:303-307`: `ell ≈ 0.97` → 0.243). Ствол пишет в лакуну
  всеми D-симметричными ветками (conv/bind/VSA/mirror/VPM/spectral/MLP).
- **`pen` — таймер (з3, §2.2/§2.8):** `pen ≈ √((1+α²)/k)` (пол √(1/k)), readiness —
  часы `pen_init=1.0`. Для головы это не вход, но `pen` определяет VSA-decay (и тем
  самым, что именно ствол запишет в лакуну) и `mat_gate`, гейтящий банк/UCL.
- **M50/M51 (з2):** `final_norm` даёт RMS≈1 независимо от шкалы потока; `stream_cap=1e3`
  и `branch_cap=1e4` — выше головы, но их срабатывание меняет направление h (а значит и
  `e_l`).
- **`_mem_dir`/`_last_lacuna_rel` — training-only (з4, §4.1):** в eval `_mem_dir`
  stale/None, `_temper_active=False` (при `step=None`), CE-путь не темперируется
  (shape-check `(N,1)` vs `(B,L)`).

### 1.4 Геометрии

| | SMALL (полигон) | cell-4 («продакшн») | CLI/config default |
|---|---|---|---|
| D | 512 | 2560 | 4096 |
| K = code_dim (голова) | 16 | 64 | 32 |
| d = D/K | 32 | 40 | 128 |
| Kp = head_phantom_bits | 32 | 32 | 32 |
| vocab | 1820 | 65536 | 65536 |
| `codebook` | legacy (C(16,4)=1820 ровно) | twin_free | legacy |
| лакуна D−K | 496 | 2496 | 4064 |

Замечание: в SMALL книга — **полное** множество 4-подмножеств (max overlap = S−1,
48 twin-кодов на код), поэтому выводы mini-прогонов о лакуне/SRL переносятся только
качественно (см. §2.4, §6).

---

## 2. Математика зоны

### 2.1 Полная формула головы (`SigmoidCodedHead`, `embedding.py:260-570`)

**Параметры и буферы (`embedding.py:273-343`):**

```python
readout = embed.basis                      # (K,d) — ТОТ ЖЕ Parameter, что basis эмбеддинга
_prop   = codes.mean(0)                    # (K,) persistent (только для init)
bit_bias = log(p̄/(1−p̄)), p̄ = clamp(_prop,1e-7,1−1e-7)     # (K,)
log_temp = zeros(K)                        # (K,)
emphasis_gain = ones(1)                    # λ, M52a (P2)
phantom_basis = _orth_rows(randn(Kp,D; seed 7))  # (Kp,D), Kp=32, ортонормированные строки
phantom_mix  = zeros(K,Kp)                 # zero-init ⇒ identity в forward на старте
lacuna_w = 30.0; lacuna_b = −3.0           # гейт фантомов
log_eta  = log(0.05); eta = exp(log_eta).clamp(0,0.2)
phantom_bank = PhantomBank(n_slots=16, D=D)  # M54, буферы в чекпойнте
token_bias = zeros(V); normalize = True
```

**Forward (`embedding.py:500-540`), точный порядок:**

```python
# 0) 2D вход разворачивается в (N,1,D), обратно squeeze (embedding.py:501-505)
B, L, D = h.shape
h_g = h.reshape(B, L, K, d)
z   = (h_g * readout[None,None]).sum(-1)                      # (B,L,K) — СЫРОЙ z (pre-T)

# 1) температура бит: ST-кламп (M52a P4)
T   = exp(log_temp);  T = T + (T.clamp(0.1,10.0) − T).detach()
if temp_factor is not None: T = T * temp_factor                # не используется стеком
z_data = z / T                                                 # источник эмфазы (P3)

# 2) логит-оддсы бит
zt = z_data + bit_bias
if bus_bias is not None: zt = zt + bus_bias                    # (B,L,K) или (N,1,K)

# 3) гибридный гейт (log-режим, tau=1.0): u, base = hybrid_gate(zt, 1.0, log=True,
#                                 emph_logits=z_data, gain=emphasis_gain)
r    = softmax(z_data)                                         # БЕЗ прайора (M52a P3)
u    = zt + λ·(log1p(r) − log1p(1/K))
base = Σ_k logsigmoid(−u_k)                                    # (B,L)

# 4) лакуна + фантомный канал (M52b/M55b)
e_l  = h − (z[...,None] * readout).reshape(B,L,D)              # сырой z, не z_data
ell  = ‖e_l‖ / (‖h‖ + 1e-6)
ell_rel = (ell / (ell_ema + 1e-6)).clamp(0, 5)
g    = sigmoid(lacuna_w·(ell_rel − 1) + lacuna_b)              # = σ(30(rel−1) − 3)
e_in = e_l + η·noise  (training) | e_l (eval)                  # η = exp(log_eta).clamp(0,0.2)
p    = tanh(e_in @ phantom_basis.T) · g                        # (...,Kp)
u    = u + p @ phantom_mix.T                                   # (...,K)

# 5) SRL (опционально, M53): диагностика или u ← u_refined (apply)
# 6) логиты
logits = u @ codes.T + (base[...,None] if not normalize else 0.0) + token_bias
# 7) tempering M55a (только если _temper_active и shapes совпали)
χ = relu(0.3 − cos(Σ_k σ(u_k)·R_k, _mem_dir));  logits = logits / (1 + 0.5·χ)
# 8) нормализация
if normalize: logits = logits − logsumexp_v(logits)
```

**Проверка формулы [mini-прогон]:** ручная реконструкция `z/T + bit_bias +
λ(log1p softmax(z/T) − log1p(1/K))` и `logits = u@Cᵀ + token_bias − logsumexp`
совпала с `forward` **бит-в-бит** (`maxdiff = 0.0`); `base.grad_fn` живой (но мёртв в
нормированном пути, §2.2). `d = D//K`, `assert D % K == 0` (`embedding.py:271`).

**dtype/режимы.** Все параметры головы fp32; `codes` — fp32-буфер (`persistent=False`);
`logits` fp32. В `scripts/train.py` autocast обёрнут только вокруг `embed_tokens`
(`train.py:501-502`), ствол и голова идут в fp32. В Colab-ноутбуке (cell 10, строки
122-126) `torch.autocast('cuda', dtype=_AMP_DTYPE, enabled=_USE_AMP)` обёрнут вокруг
**всего** forward+head+compute_losses, поэтому при `use_amp=True` (cell-4: False)
логиты головы могут считаться в bf16 — отдельный риск (§6.14).

**`log_probs_for_target` (`embedding.py:542-570`).**
- `normalize=True` (production): `forward(h2, bus_bias)` → gather по цели. Значит CE-путь
  проходит **все** стадии (phantom, SRL-apply, temper при совпадении shapes).
- `normalize=False`: факторизованный скор
  `lp = Σ_k [c_k logσ(u_k) + (1−c_k) logσ(−u_k)] + token_bias[t]` — тождественен
  `logit_t` forward (без temper); ветки согласованы (`zone_1:436-441`).
- 2D-баг M1 закрыт: `t = targets.reshape(-1)`, `h2 = h.reshape(-1,D)`, один gather,
  token_bias не удваивается (`embedding.py:543-549`).

### 2.2 Гибридный гейт и аудит M52a

**Функция (`adaptive_gate.py:27-91`):**

```python
tau_t = tau + (tau.clamp(0.1,10) − tau).detach()               # ST-кламп (M52a P4)
independent = sigmoid(logits)
relative    = softmax((logits if emph_logits is None else emph_logits) / tau_t)
gate        = independent * (1 + relative)                      # не-лог режим
if normalize: gate /= gate.sum(-1,keepdim=True).clamp(min=1e-7)
if log:                                                         # режим головы
    u    = logits + gain * (log1p(relative) − log1p(1/logits.shape[-1]))
    base = logsigmoid(−u).sum(-1)
```

Голова вызывает `hybrid_gate(zt, 1.0, log=True, emph_logits=z_data, gain=emphasis_gain)`
(`embedding.py:389-390`): τ=1.0 (ST-кламп — тождество), акцент читает `z_data = z/T`,
центрирован на `log1p(1/K)`. Не-лог режим `σ(z)(1+softmax(z/τ))` используется другими
гейтами архитектуры (L1/L2-память, SpectrumGate) — зона 5 его только определяет.

**Числа аудита M52a [M52a] (`README.md:399-411`, `tests/test_m52a_gate_fixes.py:1-9`):**

| Дефект | Измерение | Фикс |
|---|---|---|
| Мёртвый `base` в нормированном пути | градиент ровно 0 (константа по V сокращается в logsumexp); [mini] `forward == raw−logsumexp`, `atol 1e-5` | `base` добавляется только при `normalize=False` (P5); «неизвестное» — в лакуне |
| Акцент заражён прайором | `corr(bit_bias, бонус) = +0.90` (софтмакс читал `z + bit_bias`) | софтмакс читает только `z/T` (P3); тест: сдвиг прайора на 3.0 меняет `u` ровно на 3.0 |
| λ фиксирована | — | `emphasis_gain` обучаемый, init 1.0 (P2), forward на init = старой форме бит-в-бит |
| Насыщение σ | `d logsigmoid/du` при u=30: **9.36e-14** [mini]; аудит: ~9.4e-14; `sigmoid(30)=1.0` в fp32 | линейная стена `w·relu(|u|−6)²`, `dL/du = 2w(|u|−6)/N` (N — число элементов среднего); при w=1e-3, u=30: `4.8e-2/N` (README фиксирует 1.9e-4 как измеренный эффективный градиент, что соответствует N≈253 либо пост-балансерной нормировке) |
| τ-клампы | `exp(log_temp).clamp(0.1,10)` вне рельсов: `dτ/dlog_temp = 0` | ST-кламп (форвард тот же, backward = identity); тест: градиент `log_tau` жив при τ=20 |

Константы конфига: `head_u_wall=1e-3`, `head_u_wall_u0=6.0` (`config.py:62-65`);
aux-терм: `head_wall = 1e-3·relu(|u|−6)².mean()` (`losses.py:506-513`). Важно:
`_last_u` записывается в `_su` **до** фантомного микса (§2.1 шаг 4) — стена и `sat`
видят только «известные» биты и не ограничивают фантомный канал (§6.8).

### 2.3 Лакуна и фантомный канал (M52b/M55b)

```python
e_l  = h − Σ_k z_k·(R_k ⊗ e_k)                  # embedding.py:371
ell  = ‖e_l‖/(‖h‖+1e-6)                          # :452-453
ell_rel = (ell/(ell_ema+1e-6)).clamp(0,5)        # :466
g    = σ(30·(ell_rel−1) − 3)                     # :473 (lacuna_w=30, lacuna_b=−3)
p    = tanh(e_in @ phantom_basis.T)·g            # :472
u   += p @ phantom_mix.T                         # :486 (mix zero-init)
```

**Ортогональность.** `e_l` — ортогональный остаток к span{R_k⊗e_k}: `⟨e_l, R_k⟩ = 0`
точно, т.к. строки `readout` живут в непересекающихся сегментах (K≤d — QR-ортонормированы,
K>d — всё равно дизъюнктны). [mini] max |⟨e_l, R_k⟩| = **5.96e-7** (fp32); в-span h
(`h = Σ z_k R_k`) даёт `ell = 1.6e-7`; комментарий кода фиксирует 7e-7.

**Величина ℓ.** Для изотропного случайного h ожидание `ell ≈ √(1−K/D)` — это
«типичный уровень», а не жёсткий пол (h в span readout → ell=0):

| Состояние | K/D | ell | known-share = √(1−ell²) |
|---|---|---|---|
| random h, [mini] | 16/512 | 0.9850 | 0.172 |
| √(1−K/D) формула | 16/512 | 0.9843 | — |
| после блока 0 [mini, з2] | 16/512 | 0.9751 (share 0.9509) | 0.221 |
| после блока 1 [mini, з2] | 16/512 | 0.9712 (share 0.9432) | 0.238 |
| prod, комментарий | 64/2560 | ≈0.97 | ≈0.243 |
| prod, изотропная формула | 64/2560 | 0.9874 | 0.158 |

**Относительная новизна.** `ell_ema` (non-persistent, lazy-init в обоих режимах,
обновление EMA 0.99/0.01 **только в training**, `embedding.py:456-465`) калибрует
уровень; `ell_rel ≈ 1` на текущем уровне. [mini] после первого train-forward
`ell_ema = 0.985`; тест M55b запирает `0.9 < rel < 1.1` и `gate < 0.5` на уровне.

**Гейт фантомов** `g = σ(30(rel−1)−3)` [формула/мини]:

| rel | 0.90 | 0.98 | 1.00 | 1.02 | 1.10 | 1.50 |
|---|---|---|---|---|---|---|
| g | 0.0025 | 0.0266 | **0.0474** | **0.0832** | 0.500 | 1.000 |

**Шум η.** `log_eta` init `log(0.05)`, `η = exp(log_eta).clamp(0,0.2)`, шум из
**отдельного** генератора (seed 1234, device-local, `embedding.py:488-498`), только в
training; глобальный RNG не сдвигается (замок `test_m55b_lacuna_rel.py:62-72`). Смесь
`phantom_mix` zero-init ⇒ forward на init **бит-в-бит** равен голове «только известные
биты» (замок `test_m52b_lacuna.py:54-62`, atol=0.0).

**Где именно 2496 dim и кто их пишет (стык з1/з2/з5).** В h₀ (з1) 2496 координат
буквально нулевые, потому что h₀ = Σ a_k B_k⊗e_k живёт в span readout. В финальном h
(з5) «2496» — это не фиксированный набор координат, а **ортогональное дополнение**
span{R_k⊗e_k} в R^D размерности D−K; любое состояние общего положения имеет туда
проекцию. Пишут туда (каждый своей D-плотной картой):
- ствол: conv (depthwise D→D), bind (D→K→D через W_out), VSA-чтение, mirror (W_out),
  VPM (proj D→D), spectral (DCT-поворот), MLP (SwiGLU по группам) — [mini, з2]
  lacuna-share веток 0.94–0.99;
- зона 4: UCL `out_proj` (D→D, `concept_layer.py`), банк `fusion` (3D→D),
  logit-cache augment, reasoning `output_proj`, bridge `stream_proj`;
- голова: сам readout (эрозия/обучение R_k) и фантомный канал (`phantom_basis` читает
  лакуну, `phantom_mix` возвращает её в K-биты).
Структурный **читатель** лакуны ровно один — `phantom_basis` (Kp=32 из 2496 ≈ 1.3 %);
остальные «читают» её лишь косвенно, через следующую D→K→D-операцию ствола
(`zone_2:610-617`).

### 2.4 SRL: State Resolution Loop (M53/M53d/M53e)

**Аннилинг-EM (`embedding.py:396-443`):**

```python
M   = min(srl_shortlist=64, V)
idx = topk(uf @ (2C−1).T, M)                     # shortlist первого прохода
Cs, Ss = C[idx], (2C−1)[idx]
tau = tau0 = 1.0
for t in range(steps):                           # steps=3 (cfg), cell-4: 2
    p    = softmax( (uf·Ss).sum(-1) / max(tau, 0.05) )
    chat = Σ_m p_m·Cs_m
    uf   = uf + α·(logit(chat.clamp(1e-4, 1−1e-4)) − uf)     # α = 0.7
    tau  = tau·γ                                             # γ = 0.6
star  = argmax p;  cstar = Cs[star]
conf  = p.max;  ent = H(p)
expl  = −Σ_k [c*_k·logσ(u0_k) + (1−c*_k)·logσ(−u0_k)] / K     # цена объяснения, нат/бит
```

Замечания кода: `a0 = sigmoid(u0)` (`:437`) вычисляется и **не используется** (мёртвая
строка); `expl` считается по **исходному** `u0`, не по `uf`; shortlist фиксирован после
первого прохода (уточнение не выходит за него); `logit(1e-4) ≈ ±9.21` — именно туда
«щёлкают» биты при `apply=True` (M53d).

**Классификация (docstring/README §13.3):** concept `max_p>0.9 и expl<thr`;
contradiction `max_p≤0.9`; lacuna `expl≥thr`, `thr = head_srl_expl_thr = 0.7`.
**В production-коде эти пороги нигде не применяются**: `srl_expl_thr` читают только
тесты и `analyze.py:577` (печать); голова возвращает сырые `conf/ent/expl`. Прототипные
числа [M53]: чистый код → conf 1.0 / expl 0.03 нат/бит; смесь двух кодов → conf 0.50;
шум → expl 1.08. [mini, K=16] чистый код conf 1.0 / expl 0.069; смесь conf 0.363;
шум conf **1.0** / expl 0.649 — на мини-книге (полное множество 4-подмножеств) у шума
всегда есть близкий код, поэтому пороги калиброваны под prod K=64 (комментарий теста).

**M53d — катастрофа hard-refine (commit `0be800d`).** С `head_srl_apply=True` каждый
forward делает `u ← u + 0.7(logit(c) − u)`: неподвижная точка — code-consistent биты,
CE-градиент не может её перебить. Живой лог: `ce_raw 11 → 31 (шаг 1045, прогрев
SRL/temper/broadening) → 78 (1100) → 91 (1210)`; `sat 0 → 0.5 → 1.0`; `srl_expl
0.72 → 27.1`; биты садятся на `logit(1e-4) ≈ ±9.2`; затем M47-каскад (diversity 0.08,
`lbg_tau≈0.49`, val→inf; best.pt остался на 250/11.1584). Фикс: `head_srl_apply=False`
по умолчанию — SRL считает телеметрию, forward **не трогает** (замок
`test_m53_srl.py:123-135`: `atol 1e-6` бит-идентично SRL-off). Прогрев: `head_srl_after
= 1045` (на init петля коммитится к случайным кодам: CE 11.47 → 22.39 [README §13.3]);
`_srl_active = srl_on and step ≥ srl_after` выставляется стеком на каждом forward
(`stack.py:394`), в eval (`step=None`) — False.

**M53e — каденс (commit `362a224`).** `head_srl_every = 50`; условие
`_srl_step % every == 0` (по **числу вызовов головы**, pre-increment), `_srl_step`
non-persistent. Стек сам вызывает голову (#1/#2), поэтому один forward модели
продвигает счётчик несколько раз; тест зовёт голову напрямую
(`test_m53_srl.py:138-148`). Цена: ~25 % tok/s и ~2.5 ГБ на forward (topk по V=65536
на каждом срабатывании) [M53e].

### 2.5 Tempering и broadening (M55a)

**Tempering (`embedding.py:525-535`):**

```python
_md = getattr(self, '_mem_dir', None)
if (_md is not None and _md.shape[-1] == D and _md.shape[:-1] == u.shape[:-1]):
    a      = sigmoid(u)                                    # (...,K)
    h_impl = Σ_k a_k·R_k                                   # «какое h подразумевают биты»
    cos    = cosine_similarity(h_impl, _md, eps=1e-6)
    chi    = relu(temper_cos − cos)                        # cos_thr = 0.3
    if training: self._last_conflict = chi.mean()
    logits = logits / (1 + temper_k·chi)                   # k = 0.5
```

`_mem_dir` = `bank._last_read = fused.detach()` **последнего** слоя с активным банком
(перезапись в цикле `stack.py:577-584`), обновляется **только в training**
(`memory_bank.py:471-473`). Термперинг — это per-position температура: дележ не меняет
argmax, а «размягчает» распределение на конфликте. Гард `_temper_active` (стек,
`stack.py:396`): `temper_on and step ≥ temper_after(1045)`; в eval `step=None` → False.

**Broadening (`stack.py:400-404`, `memory_bank.py:442-445`):**

```python
lacuna = max(0.0, _last_lacuna_rel − 1.0)        # 1-шаг-старый; только step≥1045 и step≠None
_tk    = 1.0 + mem_lacuna_k·lacuna               # mem_lacuna_k = 0.5
temp   = exp(log_tau).clamp(0.1,10)·_tk          # на L1 и L2
```

[mini, з4] при `lacuna=5` temp ×3.5 → софтмакс-часть гибридного внимания размывается
(сигмоид-часть не меняется). В eval `lacuna=None` → `_tk=1`.

**Следствия training-only `_mem_dir` (стык с з4):**
1. CE-путь (2D h, `u.shape[:-1]=(N,1)`) не проходит shape-check `(N,1) vs (B,L)` →
   **градиент CE никогда не видит tempering**; темперируются только вызовы #1/#3
   (logits для salience/R1 и reasoning-верификатора).
2. В eval temper выключен полностью (`_temper_active=False`), даже если `_mem_dir`
   остался с тренировки; конфликт «голова↔память» в валидации не измеряется.
3. `_mem_dir` — обычный атрибут: не буфер, не в `state_dict`, не в
   `snapshot_runtime_buffers` (`stack.py:1178-1220`). После resume/generation из
   свежего процесса он `None` → temper выключен до первого training-чтения банка;
   `reset_cache` его тоже не чистит (`stack.py:1142-1176`).

### 2.6 Фантом-банк (M54) и живой факт `ph_*=0`

**Лайфцикл (`phantom.py:21-121`):**

```python
observe(e_l, ell_rel, thr):
    sel = ell_rel > thr                       # cfg head_phantom_thr = 1.1 (RELATIVE)
    E = e_l[sel][:32]                          # бюджет 32 позиции, детерминированный
    для каждой E_j: s, i = max_j cos(E_j, directions)
        if s ≥ merge(0.7) and filled[i]:
            directions[i] ← 0.95·directions[i] + 0.05·E_j     # EMA 0.05
            confidence[i] = min(1, confidence[i] + 0.05); count[i] += 1
        else:
            слот = первый свободный | argmin(confidence)
            directions[i] = E_j; confidence[i] = 0.5; count[i] = 1; filled[i] = True
    archive: filled & conf < 0.25 & count > 3 → слот очищается
decay(): confidence *= 0.999                  # каждый training-вызов головы
```

Cadence: `_pb_step % phantom_every(25) == 0` (`embedding.py:481-485`), `decay()` — каждый
training-вызов. Буферы `directions (16,D)`, `confidence`, `count`, `filled`, `_births`,
`_obs` — persistent, едут в чекпойнт [mini: `state_dict` содержит все шесть].
`confirmed_directions()` (`phantom.py:116-121`) не имеет ни одного потребителя в дереве
(grep: только тест) — **M55 (рост бит) не реализован**; банк сегодня — чистый
накопитель-диагностика.

**Живой факт `ph_*=0` (rel ≤ 1.02, всплесков нет) — разбор.**
`observe` вызывается только для позиций с `ell_rel > 1.1`; per-position `ell_rel`
само-калибруется к 1 (EMA 0.99 следит за средним), поэтому при живом уровне 1.0–1.02
`sel` пуст → `_obs = 0`, `phantoms = 0`, `births = 0` (замок
`test_m55b_lacuna_rel.py:52-59`). Гейт `g` при этом почти закрыт: σ(30(1.0−1)−3)=0.047,
σ(30(1.02−1)−3)=0.083. Что это значит:
- банк не накапливает направления → `confirmed_directions()` пуст → **у M55 нет входа**;
- единственный реально обучаемый лакунный канал — `phantom_mix` (zero-init), его
  CE-градиент жив сразу [mini: max |grad| = 0.107 на init]; а `phantom_basis`,
  `lacuna_w/b`, `log_eta` получают **ровно нулевой** градиент, пока `phantom_mix == 0`
  (grad через p равен нулю) — обучение канала двухфазное: сначала mix, потом базис/гейт
  (замок `test_m52b_lacuna.py:65-83`);
- порог 1.1 требует относительного всплеска >10 % — в живых прогонах его нет; либо
  EMA сглаживает слишком быстро (τ≈100 вызовов головы), либо семантических всплесков
  лакуны в данных действительно нет. M55 требует либо пере-калибровки статистики
  новизны, либо реализации потребителя банка (§6.5).

### 2.7 Режимы, клампы, dtype — сводка

| Механизм | train | eval | строка |
|---|---|---|---|
| шум η в `_phantom_mix` | да | нет | `embedding.py:467-471` |
| `ell_ema` lazy-init | да | да | `:456-460` |
| `ell_ema` EMA-обновление | да | нет (M55c) | `:461-465` |
| телеметрия `_last_lacuna*`, `_last_p` | да | нет | `:475-479` |
| фантом-банк (`decay`/`observe`) | да | нет | `:480-485` |
| `_last_u`, `_last_sat` | да | нет | `:391-393` |
| SRL (`_srl_active` от step) | step≥1045 | нет | `stack.py:394`, `embedding.py:509-517` |
| temper (`_temper_active`) | step≥1045 | нет | `stack.py:396` |
| `_last_conflict` | да | нет | `embedding.py:533-534` |
| broadening `lacuna` | step≥1045 | нет | `stack.py:400-404` |
| `_mem_dir` обновление | да | нет | `memory_bank.py:471-473` |

Клампы/эпсилоны: `T.clamp(0.1,10)` ST; `eta.clamp(0,0.2)`; `ell_rel.clamp(0,5)`;
`ell_ema+1e-6`; `‖h‖+1e-6`; `chat.clamp(1e-4,1−1e-4)`; `max(tau,0.05)`;
`temper cos eps=1e-6`; `sat` — доля `|u|>12.0`; `_prop.clamp(1e-7,1−1e-7)`.
Dtype: голова fp32 (кроме notebook-AMP); `codes` fp32; `logits` fp32.

### 2.8 Градиентные пути головы (что получает CE, что detached)

| Величина | Граф | Куда течёт CE-градиент |
|---|---|---|
| `readout` (=`embed.basis`) | live | через `z` (все K сегментов) и через `e_l = h − ΣzR` (фантомный путь: `∂e_l/∂R_k = −z_k e_k`) |
| `bit_bias` | live | через `zt → u → logits` |
| `log_temp` | live | ST-кламп пропускает identity; `z_data = z/T` |
| `emphasis_gain` (λ) | live | через `log1p(r)`; r=softmax(z/T) — живой |
| `token_bias` | live | прямо в логиты |
| `phantom_mix` | live | сразу ненулевой (p≠0) |
| `phantom_basis`, `lacuna_w`, `lacuna_b`, `log_eta` | live | **нуль, пока `phantom_mix=0`**; после — через p/g/шум |
| `bus_head_proj` (з4) | live | только CE-путь (`losses.py:22-27`) |
| `h` (ствол) | live | через `z` (сегментные проекции) и фантомный путь (h входит в `e_l`) |
| `codes`, `_prop` | буферы | не обучаются |
| `ell_ema`, банк, `cstar/p/conf/expl`, `_last_*` | detach/no_grad | — |
| `_mem_dir` | detached (`fused.detach()`) | temper влияет на forward, но не даёт градиента памяти |
| SRL-apply `_u_srl` | live (topk-индексы — нет) | при `apply=True`; по умолчанию граф строится и выбрасывается |
| `_last_u` / `_last_p` | live-пины | aux `head_wall`/`phantom_l1`; `release_step_graph` отцепляет после шага (`stack.py:1088-1094`) |

Баллистика aux: `head_wall = 1e-3·relu(|u|−6)².mean()` и
`phantom_l1 = 1e-4·|p|.mean()` возвращаются сырыми (`losses.py:499-513`); дальше
`LossBalancer.backward` применяет per-parameter sign-mask + норменный предел
`‖g_aux‖ ≤ ‖g_CE‖` (bypass только у `gradalign`, и тот под тем же пределом,
`training_control.py:499-507, 649-689`). Вес `head_u_wall`/`head_phantom_l1` — не
магические множители лосса, а часть сырого значения; в `_cached_losses` они **не
попадают** (см. §4).

---

## 3. Что зона делает с входным контрактом (проверка/преобразование/потеря)

1. **`h`** принимается как (B,L,D) или (N,D); 2D разворачивается в (N,1,D) и
   схлопывается обратно. Никакой нормировки/центрирования в голове нет — весь
   масштаб/направление приходят от `final_norm` (RMS≈1).
2. **Readout-тай** (з1) означает: (а) эрозия basis бьёт по масштабу `z`; (б) голова
   физически не может читать 2496 лакунных измерений — только через `phantom_basis`.
3. **`_mem_dir`** используется только при точном совпадении shapes; в CE-пути не
   совпадает → temper теряется для обучения; в eval выключен флагом. Первый
   train-forward после resume/reload идёт без temper (`_mem_dir=None`).
4. **`bus_bias`** подаётся только из `losses.py` (CE-путь). `observe_output`,
   `_knowledge_signal`, `_last_conf` зовут `lm_head(h)` без него → salience,
   `_last_logits` (R1) и reasoning-верификатор видят голову **без** intent-стенсила.
5. **`_last_lacuna_rel`** читается стеком 1-шагово-старым и только при `step≥1045`;
   в eval/на первом шаге broadening = 0.
6. **Счётчики и состояние** мутируются **всеми** вызовами головы (включая no_grad
   reasoning-вызовы): `_srl_step` (каденс), `_pb_step`/`phantom_bank` (наблюдения),
   `ell_ema` (в training), `_last_u/_last_p/_last_sat/_last_lacuna*` (в training).
   Порядок вызовов в шаге фиксирован (§1.1).
7. **Потери**: `base` мёртв в нормированном пути (P5); классификация SRL не
   материализуется; `head_wall` не видит фантомный канал (`_last_u` — pre-phantom);
   `sat` считает только pre-phantom биты; M55-рост отсутствует.

---

## 4. Передача следующему (агенту 6: обучение)

### 4.1 Точный выходной контракт

| # | Величина | Форма / dtype | Шкала, единицы | Инварианты и оговорки |
|---|---|---|---|---|
| 1 | `lm_head(h)` | (B,L,V) или (N,1,V) fp32 | log-prob (normalize=True) | `head_normalize=True` (cfg/cell-4); gather-совместимо |
| 2 | `log_probs_for_target(h,t)` | (N,) fp32 | log p(t) | normalized-путь = forward+gather; unnormalized = факторизованный скор |
| 3 | `ce_loss` | scalar | нат/токен, surprisal-взвешенный | `mask = t≠0` (+EOS если `mask_eos`); `w=σ(2·sw·(ce/mean−1))` только в training и sw>0 (`losses.py:35-55`) |
| 4 | `ce_raw` | scalar | нат/токен, **без** весов | единственная честная сравнимая метрика (M46); в `_cached_losses['ce_raw']` |
| 5 | `head_wall` | scalar | `1e-3·relu(|u|−6)².mean()`, u = pre-phantom | только если `head_u_wall>0` и `_last_u` есть; читается балансером; **не логируется** train.py |
| 6 | `phantom_l1` | scalar | `1e-4·|p|.mean()`, p = post-gate | только если `head_phantom_l1>0` и `_last_p` есть; **не логируется** train.py |
| 7 | `_last_srl` | dict conf/ent/expl | conf∈[0,1], expl нат/бит | обновляется только в training и только на срабатывании каденса (every 50 вызовов головы); в eval — stale |
| 8 | `_last_u` | (N,K) live | log-odds pre-phantom | пин графа; отцепляется `release_step_graph` |
| 9 | `_last_p` | (N,Kp) live | post-gate firing | пин графа; отцепляется `release_step_graph` |
| 10 | `_last_sat` | scalar | доля `|u|>12` (pre-phantom) | train-only |
| 11 | `_last_conflict` | scalar | mean χ = relu(0.3−cos) | train-only, только на 3D-вызовах с совпавшими shapes |
| 12 | `_last_lacuna` / `_last_lacuna_rel` / `_last_lacuna_gate` | scalars | ℓ≈0.97 / rel~1 / g≈0.05 | train-only; `rel` питает broadening (1-шаг) |
| 13 | `head_telemetry()` | dict | `lacuna, lacuna_rel, lacuna_gate, sat, conflict, srl_conf, srl_ent, srl_expl, ph_phantoms, ph_confirmed, ph_conf, ph_births, ph_obs` | ключи появляются только если соответствующий атрибут уже создан; в eval — тренировочные значения. Печатается только в `analyze.py:899` и ноутбуке (`eva_colab.ipynb:1104`), **не** в train.py |
| 14 | Буферы в чекпойнте | `_prop (K,)`, `phantom_bank.{directions,confidence,count,filled,_births,_obs}`, `_embed_rope._freqs` | — | persistent; `verify_identity_resume` проверяет формы `lm_head.*` |
| 15 | Non-persistent | `codes`, `ell_ema (1,)`, `_pb_step`, `_srl_step` | — | `ell_ema` после resume lazy-init'ится на первом же forward (оба режима) к текущему среднему, дальше EMA τ≈100 вызовов; `_srl_step` сброшен → SRL сработает на первом же вызове |
| 16 | Plain attrs (не сохраняются/не снапшотятся) | `_mem_dir`, `_last_lacuna*`, `_last_srl`, `_last_u`, `_last_p`, `_last_sat`, `_last_conflict` | — | после resume = None; eval их не сбрасывает; `reset_cache` не чистит |

### 4.2 Что теряется на стыке (обязательно учесть)

1. **CE-путь не темперируется** (shape-check `(N,1)` vs `(B,L)`): temper M55a действует
   только на 3D-вызовы (`observe_output`, `_knowledge_signal`), т.е. на salience/R1 и
   reasoning-верификатор, но не на обучающий сигнал. Это осознанная (по факту) развилка:
   если цель — учить голову «сомневаться на конфликте», сейчас это не происходит.
2. **`bus_bias` отсутствует в 3D-вызовах**: intent-стенсил учится только через CE, а
   salience/R1/reasoning видят голову без него (несогласованность представления
   «внутри шага»).
3. **Wall/L1 не в `_cached_losses`**: train.py печатает только `_cached_losses`; aux
   значения `head_wall`/`phantom_l1` видны только ноутбуку (merged aux) и `analyze.py`.
4. **`_last_u` — pre-phantom**: стена и `sat` не ограничивают фантомный вклад; при
   росте `phantom_mix` эффективные биты могут насыщаться вне контроля стены.
5. **`ell_ema` non-persistent**: lazy-init на первом forward после resume, далее
   EMA-хвост (~100 вызовов; M55b-комментарий — ~1000 шагов); в переходном окне
   `ell_rel` смещён → гейт/банк/broadening ведут себя иначе.
6. **`_mem_dir` не персистентен и не снапшотится**: generation из свежего процесса —
   temper выключен; NaN-rollback (`reset_cache`) не сбрасывает stale `_mem_dir`.
7. **Классификация SRL не материализуется**: в телеметрии только `conf/ent/expl`;
   labels (concept/contradiction/lacuna) должен выводить потребитель (пороги 0.9/0.7).
8. **`ph_*` пусты by design**: банк не наблюдает при rel≤1.02; M55-рост отсутствует,
   потребителя у `confirmed_directions()` нет.

### 4.3 Роли оптимизатора/чекпойнта (для зоны 6)

- `lm_head.readout` → группа `embed` (λ⁻² роль, `adaptation.py:207-209`;
  `stack.py:1409-1411`); остальные параметры головы (`bit_bias`, `log_temp`,
  `emphasis_gain`, `phantom_basis`, `phantom_mix`, `lacuna_w/b`, `log_eta`,
  `token_bias`) в `_GATE_PARTS`/`_MIRROR_PARTS` не входят → `default`/`default_wd`
  (LR×1.0; wd только для ndim≥2: `phantom_basis`, `phantom_mix`).
- `release_step_graph()` обязателен после `optimizer.step()` (пины `_last_u/_last_p`);
  вызывать после каждого шага, как в train.py:594.
- Порядок «forward → observe_output → compute_losses» фиксирует, какие `u`/`p` увидит
  wall/L1 (последний вызов = CE-путь).

---

## 5. Связи с другими зонами

- **Зона 1:** readout-тай (один Parameter на кодирование/декодирование); `_prop`/`bit_bias`
  (mean-field прайор, 83 % массы вне книги — `zone_1:186-195`); эрозия basis → масштаб
  `z`; `codes` — тот же объект, что у logit-cache; fingerprint B8 покрывает `lm_head.*`
  формы; `S` (sparsity) в sigmoid-голове — мёртвое поле (`zone_1:379-380`).
- **Зона 2:** `final_norm` → RMS≈1 — единственная нормировка входа головы; лакуна
  заливается стволом с блока 0 (share 0.95 → ell 0.971); M50/M51 меняют направление h
  (и `e_l`), но не масштаб у головы.
- **Зона 3:** `pen`/`hp`/`mlp_mod` в голову не пробрасываются; reasoning-петля зовёт
  голову (`_knowledge_signal`/`_last_conf`) и продвигает SRL-каденс; readiness-«часы»
  (pen_init=1.0) определяют, когда UCL/банк начнут писать в лакуну.
- **Зона 4:** `_mem_dir` (train-only, последний активный слой), `bus_bias`, broadening
  (`_last_lacuna_rel`), R1-логиты и salience из вызова #3; UCL/банк пишут в D-плотные
  направления (в лакуну); `reset_cache` не чистит ни `_mem_dir`, ни `_last_*`.
- **Зона 6:** aux-балансер (сырые значения, per-param предел), `release_step_graph`,
  роль-LR, `verify_identity_resume`, логирование (train.py не печатает head-телеметрию
  и aux wall/L1 — ноутбук печатает).

---

## 6. Открытые вопросы и риски для следующего агента

1. **CE-путь не темперируется** (§2.5): обучение не чувствует M55a-конфликт; если
   temper задуман как обучающий сигнал, нужен способ прокинуть `_mem_dir` в 2D-путь
   (например, reshape до (B,L) до вызова) — но это меняет контракт shapes.
2. **`bus_bias` не согласован между вызовами**: salience/R1/reasoning видят голову без
   intent-стенсила; при включённом `intent_bridge` представление «головы» различается
   между CE и observe_output.
3. **`_mem_dir`/`_last_*` — не персистентны и не снапшотятся** (§2.5, §4.2): generation
   после reload без temper; rollback не чистит stale; eval-изоляция их не покрывает.
4. **SRL-каденс по числу вызовов головы**: reasoning-вызовы (#1/#2) продвигают
   `_srl_step`; фактический интервал в оптимизационных шагах зависит от `s`/`K`
   reasoning; `_srl_step` non-persistent → после resume SRL срабатывает на первом
   вызове (шумный первый forward).
5. **`ph_*=0` by design** (§2.6): банк не набирает направления при rel≤1.02; M55-рост
   не реализован; `confirmed_directions()` без потребителя. Порог 1.1 (relative) и
   статистика EMA требуют пере-калибровки, иначе банк — мёртвый буфер в чекпойнте.
   Плюс расхождение: fallback `getattr(cfg,'head_phantom_thr',0.1)` в коде vs cfg 1.1
   (`embedding.py:317` vs `config.py:101`); `head_phantom_merge` cfg **не передаётся**
   в `PhantomBank` (жёсткий 0.7, `embedding.py:315-316`).
6. **Двухфазное обучение фантомного канала**: `phantom_basis`/`lacuna_w/b`/`log_eta`
   имеют нулевой градиент при `phantom_mix=0`; пока mix не вырастет, гейт и базис не
   учатся; wall не ограничивает фантомный вклад (§2.8, §4.2 п.4).
7. **Классификация SRL не применяется** (§2.4): `srl_expl_thr` мёртв в production;
   [mini] на K=16 шум даёт conf 1.0 — пороги 0.9/0.7 калиброваны под prod K=64.
   `a0` в `srl()` — мёртвая строка.
8. **Стена невидима**: `head_wall`/`phantom_l1` не логируются train.py и не попадают в
   `_cached_losses`; при w=1e-3 per-element градиент `2w(|u|−6)/N` мал (README: 1.9e-4
   при u=30 — эффективная величина), фактический масштаб задаёт балансер.
9. **`sat` и `_last_u` — pre-phantom**: телеметрия насыщения недооценивает фантомный
   канал; порог 12 (телеметрия) vs ~17 (fp32-подпор σ′).
10. **`ell_ema` после resume** (non-persistent): lazy-init на первом forward даёт
    мгновенную пере-съёмку уровня, но EMA-хвост (~100 вызовов, M55b-комментарий говорит
    о ~1000 шагах) — окно, в котором `ell_rel` смещён → гейт/банк/broadening в
    переходном режиме.
11. **`_noise_gen` (seed 1234) не в чекпойнте**: после resume поток exploration-шума
    воспроизводится с начала; сам шум не сдвигает глобальный RNG (замок), но
    детерминизм сквозь resume не гарантирован.
12. **Геометрия SMALL ≠ prod для лакуны/SRL**: K/D 0.031 vs 0.025; книга mini —
    полная (twin-коды), margin-выводы не переносятся (`zone_1:573-577`).
13. **AMP**: в ноутбуке голова внутри bf16-autocast (`eva_colab.ipynb` cell 10:122-126)
    при `use_amp=True` (cell-4: False); `logsigmoid`/`σ` в bf16 могут насыщаться
    раньше fp32 — стена (relu/квадрат) остаётся живой, но telemetry-пороги 12/17
    калиброваны по fp32. В train.py голова fp32 (autocast только вокруг embed).
14. **`token_bias` без weight decay?** `token_bias` — ndim=1 → группа `default` без wd
    (`stack.py:1434-1435`); при этом в лоссах нет отдельного приора частот — риск
    дрейфа редких токенов не закрыт ничем (зона 6).
