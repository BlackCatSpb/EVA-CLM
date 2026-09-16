# Зона 2 — Ядро блока: остаточный поток, conv, bind, VSA, спектр, VPM, предохранители M31/M50/M51

**Агент:** 2 из 6 (нерушимая цепочка, `docs/analysis/CHAIN.md`).
**Дерево:** commit `362a224` (M53e), torch 2.13.0+cpu, Python 3.14.6, Windows.
**Вход зоны:** `h₀` (B,L,D) float32 + контракт кодового пространства агента 1
(`docs/analysis/zone_1_codes_embedding.md`), состояния слоёв.
**Выход зоны:** поток `h` после всех блоков, `final_norm`, UCL-инъекция, bridge-инъекция,
состояния/кэши ветвей, инварианты шкалы.
**Предыдущий отчёт:** прочитан полностью (`zone_1_codes_embedding.md:1-595`); все его
открытые вопросы разобраны явно (см. §1.3, §3.3, §4, §6).

**Методика (правило полигона соблюдено).** Продакшн-модель не инстанцировалась,
чекпойнты не грузились. Анализ — чтение кода (`file:строка`). Вычисления выполнены
только на мини-конфиге `SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16,
code_sparsity=4, vocab=1820)` и помечены **[mini-прогон]** (два прогона:
«core-only» с выключенными мостами/UCL/матчурацией/кэшем и «SMALL defaults»;
`%TEMP%\opencode\zone2_minirun.py`, `zone2_minirun2.py`). Числа прошлых аудитов
помечены **[audit 02b]** / **[audit 02a]**, аналитические выводы — **[формула]**.

---

## 1. Граница зоны и входной контракт

### 1.1 Где начинается и кончается зона

Поток обучения (`scripts/train.py:501-508`):

```
h = model.embed_tokens(x)                        # ЗОНА 1
out, state, gs, _ = model(h, state, ...)         # ЗОНА 2: EVAStack.forward
ce_loss, aux_dict = model.compute_losses(out, y, h_emb=h)  # ЗОНА 5/6
```

Зона 2 — тело `EVAStack.forward` (`stack.py:209-793`) **за вычетом** embed/head:
- bridge-инъекция до каждого блока (`stack.py:537-541`);
- цикл слоёв `for i, (layer, s) in enumerate(zip(self.layers, state))` (`stack.py:455`);
- сам `EVABlock.forward` (`block.py:390-775`) — предмет §2;
- M50-cap после каждого блока (`stack.py:615`);
- UCL-инъекция после блока 0 (`stack.py:619-636`);
- `final_norm` (`stack.py:710-715`);
- logit-cache augment (`stack.py:778-791`, зона 4) — последнее преобразование `h`
  перед головой.

### 1.2 Что именно приходит от агента 1 (используемые величины)

| Величина агента 1 | Как используется в зоне 2 |
|---|---|
| `h₀` (B,L,D) float32, `‖h₀‖≈4.40`, RMS≈0.087 (prod) | вход блока 0; участвует в residual-сумме как есть |
| common-mode 95.11 %, pair-cos 0.9512 | уходит в ствол без изменений (`embed_center=False`); M31 нормирует только входы ветвей |
| `basis`/`readout` (K,d), тай с головой | **блоком не читается вообще**; z-масштаб головы определяется только `embed.basis` |
| K=64/S=6/twin_free (cell-4) vs K=32/legacy (CLI) | геометрия блока задаётся **не** `code_dim`, а `cfg.bind_K` (§1.3) |
| эрозия basis min 0.2993 → [0.5146,0.6423] | на ствол не влияет (basis не в графе блока); влияет на голову (зона 5) |
| bridge-инъекция до блока 0 | реализована в `stack.py:537-541`; на init тождество (см. §2.12) |
| финальная норма → RMS≈1 | `stack.py:710-715` — конец зоны 2 |

### 1.3 Какая геометрия у «прогона» и какая у блока (ответ на вопрос (а) агента 1)

Геометрия кодов (`code_dim`, `codebook`) и геометрия **блока** (`bind_K`, `conv_kernel`)
— разные поля. Блок использует:

```
block.K = cfg.bind_K                 # block.py:210; размерность bind-бутылочного горла
mirror k = cfg.mirror_k_staircase    # block.py:253-262: 8 / 16 / 32 по третям глубины
           иначе cfg.mirror_k
```

| Источник | D | n_layers | `bind_K` (блок) | code_dim (голова) | codebook |
|---|---|---|---|---|---|
| `EVAConfig` defaults (`config.py:13-16,106-113`) | 4096 | 32 | **64** | 32 | legacy |
| `scripts/train.py` CLI (`train.py:760-773`) | 4096 | 24 | **64** (`--bind-K`) | 32 (флага нет) | legacy |
| Notebook cell-4 = «production» (`eva_colab.ipynb` cell 3) | 2560 | 24 | **32** (`bind_K=32`!) | 64 | twin_free |
| Единственный чекпойнт `best.pt` [audit 02a:396-399] | 2560 | 24 | legacy-эпоха | 32 | legacy |
| Полигон `SMALL` (`tests/test_gradient_flow.py:21`) | 512 | 2 | **64** (дефолт, не переопределён) | 16 | legacy |

**Явная фиксация анализа:** все формулы ниже справедливы для любого `bind_K`; числа
[mini-прогон] получены при `bind_K=64`, `D=512`, `mlp_groups=4` (d_group=128),
`code_dim=16` (d_code=32), mirror k=16/32 (staircase для n=2: `l<1 → 16`, иначе 32).
Продакшн-числа [audit 02b] сняты на D=256, bind_K=16 либо аналитически для 24 слоёв;
числа cell-4 (D=2560, bind_K=32, d_group=80, mirror k=8/16/32) помечены отдельно.
**Важно:** одна mirror-группа cell-4 (d=80) накрывает **два** кодовых сегмента
(`d_code = D/code_dim = 40`) — сегментная структура агента 1 и группы зеркала не 1:1.

### 1.4 Состояния на входе (per-layer state tuple)

`state[i] = (mem_state, mu_state, conv_state, traj_state, pen)` (`block.py:395-401`):

| Поле | Форма | dtype | Смысл |
|---|---|---|---|
| `mem_state` | (B, S·D), S=4 | fp32 | последнее состояние VSA-скана (multi-scale) |
| `mu_state` | (B, S·D) | fp32 | то же для первого момента |
| `conv_state` | (B, D, 47) | `h.dtype` | последние 47 позиций `LN(h)` для causal-conv |
| `traj_state` | (B, nd−1, L, K), nd=3 | fp32 | перенос траектории спирали (только streaming) |
| `pen` | (B,L) или None | fp32 | **вестигиальный канал**: см. §2.13 |

При `state=None` все поля None; на первом окне документа VSA стартует с нуля,
conv — с нулей, траектория — из in-graph сдвигов (`bind.py:411-420`).
Состояния **детерминированно отцепляются** стеком после каждого слоя:
`s_out = tuple(t.detach() ...)` (`stack.py:695`) — BPTT между окнами нет.

---

## 2. Математика зоны

### 2.1 Формула остаточного потока и порядок операций (M31)

**Определение потока.** Внутри блока `h` — истинный residual-аккумулятор; ни одна
ветка его не перезаписывает, каждая добавляет свой вклад:

```
# block.py:466-772, порядок строго по коду
h  ← h + _stream_cap( conv(_ln(h)),                      branch_cap )   # :469-477
bind_out, new_traj, coh ← bind(_ln(h), traj_state)                      # :496
h_v = _ln(h)                          # VSA-запись/скан читают ТОТ ЖЕ поток   # :517
… VSA scan → mem_read (mem_all/mem_leaf/mu_all)                         # :592-651
mirror, mlp_mod, mem_mod, hp, pen_new ← mirror(_ln(h), mem_all)         # :659-663
enhanced_base = bind_gated + mem_modulated · w_mem2v · mem2v_scale      # :692
enhanced = _stream_cap(enhanced_base, branch_cap)
         + _stream_cap(mirror,       branch_cap)                        # :693-694
h  ← h + enhanced                                                       # :700
h  ← h + _stream_cap( (precision · exact · soft_gate), branch_cap )     # :704-720
h  ← h + _stream_cap( V_dct·(λ_k ⊙ (V_dctᵀ·_ln(h))·damp·spectral_mod), branch_cap )  # :729-737
h  ← h + _stream_cap( mlp(_ln(h), mirror_gate=mlp_mod), branch_cap )    # :743-772
```

**Где читается `_ln(h)`** (какие ветки видят нормированный вход) — все семь точек,
и каждая видит поток **со всеми уже добавленными предыдущими ветками**:

| Точка | Строка | Что видит |
|---|---|---|
| conv | `block.py:469` | `h` на входе блока (после bridge-инъекции стека) |
| bind | `block.py:496` | `h + cap(conv)` |
| VSA (запись и decay) | `block.py:517` | `h + cap(conv)` |
| mirror | `block.py:659-663` | `h + cap(conv)` (enhanced ещё не добавлен) |
| VPM | `block.py:708,718` | `h + cap(conv) + enhanced` |
| spectral | `block.py:730` | `h + … + cap(VPM)` |
| MLP | `block.py:743` | `h + … + cap(spectral)` |

Сырой `h` (без LN) читают только: `_stream_cap` (кап), `final_norm`, bridge-инъекция,
UCL-инъекция и logit-cache. Ни одна ветка не видит ненормированный вход.

**`_ln` (M50-безопасная форма)** — `block.py:429-433`:

```python
_m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
_x = x / _m
return self.pre_ln_w * _x * torch.rsqrt(_x.pow(2).mean(dim=-1, keepdim=True) + 1e-7)
```

Это RMS-норма (масштабно-инвариантная), но записанная через amax-rescale: `x²` не
переполняется даже при `|x|~1e20`. `pre_ln_w` — **буфер**, не параметр
(`block.py:233`), нигде не обучается: у `_ln` нет обучаемого аффинного гейна (в
отличие от `bind.hp_norm.weight` и `mlp.norm_w` — они Parameters). Выход `_ln` —
per-position RMS≈1 (при `|x|` не у клапа 1e-6).

**Почему `h` не перезаписывается и что это даёт градиентно (M31).** Комментарий
`block.py:449-464` фиксирует замеры 1045-шагового аудита:

```
Старая форма: ОДИН LN на входе блока, ветки копятся на нормированную копию,
              выход LN(h)+Σ(ветки):
   обратный спад 2.3x/слой; CE-grad L0 3.2e-8 vs L23 2.6 (отношение ~1.2e-8);
   Adam SNR ~1e-10 — мелкая половина заморожена.
Два неудачных фикса на реальном чекпойнте:
   (i) end-compensation (±h_n гасит прямую conv-ветку),
   (ii) straight-through LN (восстанавливает identity, но открывает сырой
        каскадный гейн ~50x/сабслой -> inf).
Правильная форма (классический pre-LN для внутриблочного каскада):
   h_out = h_in + Σ f_s(LN(·)) с identity-коэффициентом 1 на каждом сабслое;
   каждая ветка кормится входом единичного масштаба.
```

Градиентно: `∂h_out/∂h_in = I + Σ_s ∂f_s/∂LN · ∂LN/∂h_in`. Единичная матрица даёт
CE-градиенту «шоссе» до слоя 0 (никакого 2.3x/слой спада), а якобиан LN ограничен
(нормировка RMS), поэтому ветки не могут раздуть каскад. Побочный эффект честно
зафиксирован в коде: forward-значения отличаются от старого блока («fresh run
required», `block.py:462-464`). Замок: `tests/test_m31_depth_highway.py:36` требует
`L0/L5 > 1e-3` (старое 1e-8). **[mini-прогон, core-only, CE-backward]:**
‖grad‖ слоя 0 = 15.36, слоя 1 = 10.14 (отношение 1.52); ни один параметр не
получил ноль/NaN.

**Что это значит для остальных зон.** Residual-шоссе не «обнуляет» ветки: вклад
каждой ветки виден CE-градиенту через её собственный путь; при этом DC/common-mode
из h₀ не нормируется — он живёт в `h` как есть, а ветки получают его нормированную
проекцию. Отсюда требование агента 1 (h₀ common-mode 95 %) удовлетворяется
структурно: ствол несёт DC, ветки его «видят» через LN.

### 2.2 Conv-ветка

```
h_perm = _ln(h).transpose(1,2)                      # (B, D, L)      :469
_full  = cat([conv_state, h_perm], dim=-1)          # causal-склейка :470
h_conv = self.conv(_full)[..., -L:].transpose(1,2)  # (B, L, D)      :473
conv_state_out = h_perm[:, :, -47:]                 # (B, D, 47)     :474
h ← h + _stream_cap(h_conv, 1e4)                                    # :477
```

- `nn.Conv1d(D→D, kernel_size=48, padding=0, groups=D, bias=False)` (`block.py:350-352`),
  depthwise, kaiming_normal fan_in/linear (`block.py:353`). `conv_kernel=48`
  (`config.py:266`).
- Причинность реализована **явно** состоянием (M11, `block.py:342-349`): на старте
  потока `conv_state = 0` (47 нулей = left-pad), затем переносятся последние 47
  `LN(h)`; выход позиции t зависит от t−47..t. Комментарий фиксирует старый дефект:
  `padding=47 + cat(47)` давал окно t−94..t−47 (слепота к последним 47 токенам и
  строгий ноль для L<94 — источник нулевого градиента conv).
- B1-защиты: левый defensive pad (`:471-472`) и fixed-width carry (`:475-476`).
- dtype: `conv_state` наследует `h.dtype` (`block.py:468`) — при AMP перенос между
  окнами разных dtype падает на `torch.cat` **[audit 02b §4.2]**. При `use_amp=False`
  (дефолт и cell-4) всё fp32.
- Формы: вход (B,D,L+47) → выход (B,D,L) → (B,L,D). Параметров: D·48.
- [mini-прогон] норма выхода conv: 4.58 (L0), 2.23 (L1) — мала относительно потока
  (см. §3.1); lacuna-share 0.941/0.951.

### 2.3 Bind-ветка: TrajectorySpiralBind / TrajectoryManifoldBind

**Выбор** (`block.py:236-246`): `bind_twist_mode` (default `"trajectory_spiral"`,
`config.py:406`); при `traj_manifold=True` (`config.py:416`, default False; cell-4 не
включает) — `TrajectoryManifoldBind`, иначе `TrajectorySpiralBind`. `SpiralBind` и
`BottleneckBind` в текущих прогонах не задействованы.

**Фазовая спираль** (`bind.py:310-509`):

```
hp = hp_norm(W_proj(_ln(h)) + w_bind_bias)                  # (B,L,K)  :392
# hp_norm = _ExpRMSNorm(K): weight * x * rsqrt(mean(x²)+1e-7)           :14-22
traj = cat([hp.unsqueeze(1), traj_state], dim=1)            # (B,nd,L,K) :423
u  = hp · (w_u_re + i·w_u_im)                               # :445-446
v  = traj · (w_v_re + i·w_v_im)                             # :448-449
θ  = (exp(W_freq) · freq_eff) · hp + W_phase                 # :440-442
vr = v · e^{iθ}                                              # :450-451
prod = u · vr                                                # :452-453
hybrid = α·(u ⊛ v) + (1−α)·(u ⊙ v)                          # :456-461
prod_re += 0.1 · hybrid                                      # :462
out_acc = Σ_{s,d} cat[prod_re, prod_im]  ⊕  coherence        # :463-473
result = out_acc @ W_out                                     # :507
```

- `W_proj`: Linear(D→K, bias=True) **плюс** отдельный `w_bind_bias` (K) —
  двойной bias (`bind.py:317-318`).
- `W_freq` инициализирован лог-сеткой частот: `τ_base = log(arange(1,K+1)/K)`,
  сдвиг `(frac−0.5)·2·log K`, `frac = (s·nd+d+0.5)/(S·nd)` (`bind.py:327-336`);
  `freq_scale` init `2π` (`bind.py:340`).
- **U10 τ-coherent schedule**: `freq_eff = freq_scale · (τ_min/τ_max)^{τ_norm·σ(_eta)}`
  (`bind.py:426-433`); `τ_norm` обновляется **живьём** каждый forward блоком
  (`block.py:410-416`), `_eta` init 0.5.
- **Затухание** в спирали — не свободный τ, а фазовое вращение `e^{iθ}`; отдельного
  экспоненциального decay-гейта у bind нет.
- **Hybrid HRR**: `α = α_min + (α_max−α_min)·(1−τ_norm)` (`bind.py:363-377`),
  `α_max=0.7, α_min=0.3` (`config.py:409-410`) — статично по τ, train≡eval (B3).
  [mini-прогон]: L0 τ_norm=0.5 → α=0.5; L1 τ_norm=1.0 → α=0.3 (совпадает с
  [audit 02b] табл. стр. 14).
- **Когерентность** `coherence = |Σ_{s,d} e^{iθ}|²/(S·nd)² ∈ [0,1]` (`bind.py:466-472`)
  — идёт в выход bind (K каналов) и в VSA как усилитель записи (§2.4).
- `W_out`: (nd·2K+K, D) = (7K, D), xavier gain 0.5.
- **Траектория (in-graph shift, B2)**: при `traj_state is None` (обучение) каналы
  d=1..nd−1 строятся сдвигами `hp` по времени `F.pad(_hc,(dd,0))` (`bind.py:411-420`)
  — это даёт живую производную `∂out_t/∂h_{t−d}`. При streaming — перенос
  `traj_state`; в **обучении** detached-кэш не читается никогда (B10,
  `block.py:483-504`): `if traj_state is None and not self.training: traj_state =
  getattr(self,'_traj_state',None)`, запись кэша — только training/`_stream_mode`.
  [audit 02b] F2B-01 (замороженная траектория, eval зануляет кэш) — **закрыт B10**;
  замки: `tests/test_b10_spiral_and_ladder.py:20-45`.
- **Zeckendorf/манифолд** (`TrajectoryManifoldBind`, `bind.py:512-710`, default OFF):
  кольцевой буфер переходов `T = unbind(hp_t, hp_{t−1})` (1024×K, fp32, no_grad),
  кластеризация в лучи (n_beams=ceil(√1024)=32, cos_threshold 0.5, rebuild каждые
  128 переходов), чтение — hybrid attention `sigmoid·(1+softmax/τ)` по косинусам
  q↔луч с **Zeckendorf-затуханием по возрасту** `θ(age)=1/(1+len(zeck(age)))`
  (`bind.py:664-668`), вклад `result + gain·clamp(man@W_man, ±8)`, gain=0.05.
  Буферы non-persistent; `_push_transitions` вызывается в **обоих** режимах
  (`bind.py:708`) — на eval-форвардах буферы пишутся, но покрыты снапшотом стека
  (`stack.py:1202-1218`).
- **Мёртвые гейты**: `bind_twist_gate=True` (cell-4) не реализован в
  TrajectorySpiralBind [audit 02b F2B-05]; живые гейты — per-expert
  `w_bind_gate` (σ(0)=0.5) и `mem_mod` зеркала (§2.6).
- [mini-прогон]: ‖bind‖ = 3.10/3.41, lacuna-share 0.984/0.976.
  [audit 02b] (pre-B10): bind gain ‖out‖/‖h‖ = 0.451, coherence mean 0.443,
  max 0.996.

### 2.4 VSA-ветка (multi-scale prefix scan, 4 темпа τ)

**Лестница τ.** В `EVAStack`:
`_base_vsa = exp(cumsum(softplus(_vsa_log_param))) + 1` (`stack.py:276`),
`_vsa_log_param = [1.7918, 1.2321, 1.1304, 1.1065]` (`stack.py:139`) →
**[формула]** `_base_vsa = [8.0, 32.0, 128.0, 512.0]`. На слой:
`tau_s_i = _base_vsa · τ_l/τ_mid` (`stack.py:591-607`), `τ_mid = √(τ_0·τ_23)`.
В блоке: `d_s = exp(−1/tau_s)` (`block.py:515`). Собственный
`_vsa_tau_log` блока (`block.py:294-295`) в EVAStack **не используется** (tau_s
всегда передан) и исключён из оптимизатора [audit M7].

[формула] Продакшн (24 слоя, dev=0): `tau_norm_l = (l+1)/24`, `τ_l = 8·64^{(l+1)/24}`,
`τ_0 = 9.51`, `τ_23 = 512`, `τ_mid = 69.79`, отношения `τ_l/τ_mid ∈ [0.136, 7.34]`.
**[mini-прогон]:** L0 `tau_s = [2.83, 11.31, 45.26, 181.03]`,
L1 `tau_s = [22.63, 90.51, 362.05, 1448.25]` (2-слойный τ = [64, 512], τ_mid=181).

**Запись:**

```
h_v = _ln(h)
i_gate = softplus(h_v·w_i + b_i + γ_surprisal·pen) · (1 + bind_coh_gate·coh_mean)
write_mod = σ( hp_cached @ w_i_dyn / √k )            # только если hp_cached есть
mem_input = h_v · write_mod · i_gate                  # иначе h_v · i_gate
```

(`block.py:516-524, 571-588`). Init: `b_i=−2.5` (i_gate≈0.08), `w_i~N(0,1)`,
`gamma_surprisal = 0.5·σ(log(τ_l/τ_mid))` (мелкие слои ~0.06, глубокие ~0.44,
`block.py:323-333`), `bind_coh_gate=0.5` (`block.py:335`), `hp_cached` — детач-кэш
**предыдущего** forward (M10: один путь в train/eval).

**Decay:**

```
d_mod   = σ(h_v·w_d + b_d) / σ(b_d).clamp(min=1e-3),  clamp(max=1.0)   # B18 rest=1
d_pen   = 1 − (σ(pen − _pen_ema + w_d_pen) − σ(w_d_pen))               # B18b centered
decay   = clamp( d_s · d_mod · d_pen, min=0.01, max=1.0 )              # :565
floor_log = k · log(d_s), k = vsa_decay_floor_k = 2.0                  # :606-607
log_a = log(decay.clamp(min=1e-6)).clamp_min(floor_log)                # :125
```

т.е. содержимое может **укоротить** память до `τ_s/k`, но не ниже (floor). `b_d`
init `2.0 + 3·layer_frac` (`block.py:320-322`), `w_d ~ N(0, 0.1²)` (`config.py:262`).
B18/B18b закрыли «схлопывание лестницы» [audit 02b F2B-02]: старая форма
`σ(h·w_d+b_d)` без rest-нормировки умножала decay каждого токена на ~0.88–0.99 и
давала window-carry r=0 на всех 8 layer×scale (τ_eff ≈ 1.4 токена, ×2.1…×1013 от
номинала). Замки: `tests/test_b18_ladder_floor.py:30-41` (rest carry r>0.02).

**Prefix scan** (`block.py:44-157`):
- `_scan_chunk` (M26): tail-referenced форма
  `intra_t = (cd_t/cd_last)·cumsum_i(b_i·cd_last/cd_i)` — все экспоненты ≤0, ни один
  операнд не превышает `|b|`, fp32-стабильна в fwd и bwd при любом floor. При
  `floor_log=None` — легаси fp64 reciprocal (оракул M3, для точных тестов).
- `_scan_chunks` (M37/M38): векторный чанковый скан, chunk=32, ось чанков вложена в
  batch; tail-pad `b=0/decay=1` не двигает реальные префикс-суммы.
- `_combine_chunks`: межчанковый скан по финальным состояниям, возвращает
  `combined, final_state, leaf`; `mem_state` (перенос) входит сюда.
- `vsa_utils.vsa_prefix_scan` (`vsa_utils.py:168-209`) — отдельный fp64-оракул,
  блоком не вызывается (используется тестами).

**Чтение** (`block.py:624-651`):

```
w = σ(scale_w)                                        # (S,D), init fib → [1/7,1/7,2/7,3/7]
mem_all  = Σ_s w_s · scan_s
mem_leaf = Σ_s w_s · leaf_s                           # без кросс-чанк контекста
mem_read = mem_all·w_q + mem_leaf·w_q_leaf + (mem_all−mem_leaf)·w_q_ctx
mu_all   = Σ_s w_s · scan_mu_s                        # первый момент, тот же decay
mem_read += mu_all·w_q_mu·w_mu_mem
```

`w_q = w_q_leaf = 1/√D`, `w_q_ctx = 0.5/√D` (константы, но Parameters),
`w_k_mu, w_q_mu, w_mu_mem ~ N(0,1)` (`block.py:299-304`). Затем per-expert
read-модуляция зеркала (§2.6).

**dtype/режимы:** весь скан — fp32 (`decay.float()`, `input_vec.float()`,
`mem_state.float()`, `block.py:596-602`); комментарий `:622` — «Keep VSA in fp32».
`noise_scale>0` умножает `i_gate` на шум **только в training** (`:542-544`).
[mini-прогон]: raw-норма `mem_all` = 342.4 (L0) / 478.9 (L1) — **самая большая
ветка блока**; с учётом read-весов именно VSA-чтение (особенно первый момент с
`|w_q_mu|~O(1)`) даёт основной прирост потока (см. §3.1).

### 2.5 Зеркало (интерфейс; сама математика — зона 3)

Вызов под `autocast(enabled=False)`, входы `_ln(h).float()` и `mem_all.float()`
(`block.py:654-663`):

```
mirror, mlp_mod, mem_mod, hp, pred_error_norm = self.mirror(...)
```

- `hp` (B,L,G,k) — **живой** (в графе); `_cached_hp`, `_cached_pred_error_norm` —
  детачи предыдущего forward (`mirror.py:559-560`), пишутся в обоих режимах (M33).
- `pred_error_norm = (raw_pred_error / hp_norm).pow(2).mean(dim=(-2,-1)).sqrt()`
  (`mirror.py:537`) — per-position RMS по (G,k), O(1); B10-фикс единиц (было ~8.1
  как норма по G·k). Замок: `tests/test_b10_spiral_and_ladder.py:48-57` (pen<2,
  factor>0.75).
- `pen` (из состояния или кэша) входит в VSA-decay (`block.py:548-560`) и в
  i_gate-boost (`:519-520`).
- k зеркала: staircase 8/16/32 (`block.py:253-262`); `G=cfg.mlp_groups`.
- `mem_mod` (B,L,G) гейтит **и** память, **и** bind: `bind_gated = bind·mm·σ(w_bind_gate)`,
  `mem_modulated = mem_read·read_mod·mm` (`block.py:672-692`). Выход зеркала
  добавляется в поток **без** `mm` (только кап): `enhanced = cap(base) + cap(mirror)`.

### 2.6 VPM-ветка (Variable Precision Memory)

```
precision = σ(precision_gate(_ln(h).float()))                 # (B,L,1), :707-708
hard      = (precision.mean() > 0.3).to(h.dtype)              # скалярный гейт
soft_gate = hard + (precision.mean() − precision.mean().detach())   # M11 ST
exact     = exact_memory(_ln(h).float())
h ← h + _stream_cap( (precision · exact · soft_gate).to(h.dtype), 1e4 )
```

`ExactSequenceMemory` (`block.py:169-191`, `exact_k = min(64, D//4)`):
`A = σ(qkᵀ/√k)`, `A = A/A.sum(-1).clamp(min=1e-6)`, `out = proj(A@v)` — LaCUR
(«проводимость, не конкуренция», выпуклая оболочка, без взрыва). Под `autocast`
выключен (softmax/sigmoid в fp16 переполняются). Straight-through M11: при
закрытом гейте forward=0, но `∂/∂precision.mean()` жив — открыть можно всегда.
**Наблюдение:** attention **не маскирована причинно** (полная L×L): в окне обучения
позиция t читает и будущие позиции. При streaming L=1 вырождается в `v_t`.
[mini-прогон]: precision.mean = 0.599 (L0) / 0.332 (L1) при пороге 0.3 — гейт
открыт/на грани; норма выхода 7.3/7.0.

### 2.7 Спектральная ветка (DCT + Chebyshev-демпфирование)

```
h_dct = _ln(h).float() @ V_dct.T                       # (B,L,D)      :730
damp  = cos(π · τ_norm / 2)                            # U3            :731-734
h_dct = h_dct · λ_k · spectral_mod · damp
h ← h + _stream_cap( (h_dct @ V_dct).to(h.dtype), 1e4 )               :736
```

- `V_dct = dct_basis(D)` — ортонормированный DCT-II (`vsa_utils.py:9-15`);
  строка 0 = 1/√D.
- `λ_k = base + per_dim`, `base = 0.5 + layer_frac`, `per_dim = linspace(1.0,0.5,D)·0.2`
  (`block.py:366-372`) → DC получает `base+0.2`, Найквист `base+0.1`;
  init-диапазон [0.6, 1.7]. Параметр `lambda_k` (D).
- `damp = cos(π·τ_norm/2)`: [формула] по 24 слоям `[0.998, …, 0.707 (l=11), …, 0.0 (l=23)]`
  — глубокие слои спектрально почти выключены на init; [mini-прогон] `[0.7071, 0.0]`.
- `spectral_mod` — float от AdaptiveController (без градиента), `block.py:735`.
- DCT-ветка — **плотное** смешивание всех D координат: она разносит DC/common-mode
  по всему спектру, включая лакуну.

### 2.8 MLP-ветка (GroupedMLP, SwiGLU)

`mlp.py:67-107`:

```
h = norm_w · h · rsqrt(mean(h²)+1e-7)                  # обучаемый norm_w (D)
h → (B,L,G,d),  hg = (G, B·L, d)
gate = silu(hg @ W_gate)
if mirror_gate: gate *= (mlp_gate_a + mlp_gate_b · mg)  # a=1, b=0.25 init
up   = hg @ W_up
out  = (gate·up) @ W_down → (B,L,D)
```

`mirror_gate = mlp_mod` (B,L,G) — **живой** граф. Init: `randn·√(2/(d+hidden))`.
`_cached_group_out` (`mlp.py:106`) — сырой выход до reshape, для diversity-loss.
[mini-прогон]: ‖h_mlp‖ = 12.5/15.2; lacuna-share 0.994 — самая «лакунная» ветка.

### 2.9 Предохранители M50/M51

**`_stream_cap` (M50/M51)** — `block.py:22-32`:

```
if cap <= 0: return x
m = x.abs().amax(dim=-1, keepdim=True)
return x * (cap / m.clamp_min(cap))        # при m>cap: y = x·cap/m
```

- Scale-инвариантный (направление сохраняется), якобиан O(cap/m) — не зануляется:
  тест `tests/test_m50_stream_fuse.py:12-20` (x=1e4·N(0,1), cap=1e3 → grad>1e-3).
- **M50** — на потоке между блоками: `h = _stream_cap(h, 1e3)` (`stack.py:615`),
  `stream_cap=1e3` (`config.py:439`). Стоит **после** блока, **до** UCL-инъекции
  (`stack.py:619-636`) и до bridge следующего слоя. Комментарий `stack.py:376-385`:
  измерено `h~1e20` на выходах слоёв → `mean(h²)=inf` → `rsqrt(inf)=0` → выход
  ровно ноль → CE-градиент в ствол ровно ноль → deadlock, пока aux-лоссы
  (bridge/bank) раздували ветки (`bridge._inj_alpha/beta` — топ-движеры Adam
  мёртвого чекпойнта). C=1e3 выбран так, что «healthy streams measure 30-70».
- **M51** — на каждую инъекцию ветки: `branch_cap=1e4` (`config.py:441`,
  `block.py:217`), применён к conv (`:477`), bind+VSA-группе (`enhanced_base`, `:693`),
  зеркалу (`:694`), VPM (`:719`), spectral (`:736`), MLP (`:772`). Суммарно до
  ~6e4 на блок, но следующий M50-cap режет поток до 1e3. Комментарий `block.py:213-216`:
  «healthy branches measure O(1)–O(1e3), so 1e4 is a no-op until something runs away».
- **M51-анкер** в branch-loss (`losses.py:224-261`): три log-ratio члена
  scale-free, поэтому «равномерный рост всех веток x10/слой был невидим — поток
  дошёл до ~1e16, а 'branch' остался конечным»; `branch_var_anchor=0.5` тянет
  log-дисперсию каждой ветки к медленному EMA (эталон детач). Замок:
  `tests/test_m51_branch_guards.py:80-84`.
- **UCL-инъекция** (`stack.py:619-636`) и **memory-bank** (`stack.py:577-581`)
  M51/M50 **не** покрыты: UCL ограничен изнутри 25 % локальной нормы
  (`concept_layer.py:375-380`), банк — своей амплитудой; до следующего cap их вклад
  живёт без потолочного ограничения.

### 2.10 Финальная норма и выход из зоны

`stack.py:710-715`:

```
_fm = h.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
_hs = h / _fm
h = final_norm_w · _hs · rsqrt( mean(_hs²) + 1e-7 )
```

`final_norm_w` — **буфер** (`stack.py:75`), не параметр: аффинного обучения нет.
Выход — per-position RMS≈1: [mini-прогон] ‖out‖ = 22.627 = √512 (точно), min/max
22.6269/22.6273. Старая форма `mean(h²)` переполнялась при h>1e19 (см. M50);
amax-rescale это снимает (тест `test_m50_stream_fuse.py:29-34`, 1e20 → finite).

Порядок после цикла: `final_norm` → reasoning-ветка (если включена, `stack.py:718-728`)
→ triad re-pass (только inference) → `logit_cache.augment(h, ...)` (`stack.py:778-791`).
`augment` возвращает `gate·attn_output + (1−gate)·h`, `gate=σ(−10)≈4.5e-5` при init
(`logit_cache.py:348-360, 449-450`) — тождество на старте; в training передаётся
1-шагово-старый `_last_logits` (M56, R1 5 %).

### 2.11 Градиентные пути внутри блока (что получает CE-градиент, где detach)

| Ветка | Путь CE-градиента | Detach-барьеры |
|---|---|---|
| conv | `conv.weight` (у `_ln` обучаемых параметров нет — только Якобиан к `h`) | — |
| bind | `W_proj`, `w_bind_bias`, `w_u/v_re/im`, `W_freq`, `W_phase`, `W_out`, `freq_scale`, `_eta`, `hp_norm.weight` | `traj_state`-кэш (в train не читается), `traj_state_out` detach (streaming) |
| VSA | `w_i, b_i, w_d, b_d, w_d_pen, gamma_surprisal, bind_coh_gate, scale_w, w_q*, w_k_mu, w_q_mu, w_mu_mem`; `w_i_dyn` (вход `hp_cached` — константа), `w_q_dyn` (вход `hp` живой) | `pen` (кэш), `hp_cached`, `mem_state`/`mu_state` (нет BPTT), `_pen_ema` (no_grad) |
| зеркало | все параметры зеркала; `h` через `_ln(h)`; `mem_all`; `mlp_mod` → MLP-гейт | `global_state` (`stack.py:470`), `salience` (голова), `diff/tanh/pred/spectral_mod` (float) |
| VPM | `precision_gate`, `exact_memory.*` (только при hard=1) | — |
| spectral | `lambda_k` | `V_dct` — буфер; `spectral_mod` — float |
| MLP | `W_gate/W_up/W_down`, `norm_w`, `mlp_gate_a/b`; обратно в зеркало через `mlp_mod` | — |
| UCL | read-путь (`q_proj`, `out_proj`, `read_scale`); write — функциональный, гейтед | запись в буферы через `.detach()` (после чтения) |
| bridge | `stream_proj`, `stream_log_scale`, `_inj_alpha/beta`, `stream_log_weights`; **не** в `h` (инъекция не имеет h-градиента) | `probe_layer(h.detach())` — probe учится своим InfoNCE-лоссом |
| intent | `intent_probe` (fresh_i живой), `w_intent` в зеркале | carried-поток detach, `_bus_sum` detach |

Кэши для лоссов (только training): `_cache_conv_out`, `_cache_bind_out`,
`_cache_mirror_out`, `_cache_mlp_out` (`block.py:479-480, 697-699, 744`) — с графом,
потребляются branch-loss при `branch_balance_weight>0` (cell-4: 0.1);
`_cache_mlp_mod` (`:666`) — для gradalign (train.py default 0.0, Colab-arm 0.3);
хук на `h_mlp` пишет `_gradalign_tgt = ‖∂CE/∂mlp_out‖` по экспертам
(`block.py:752-758`), `_ga_record=False` защищает от aux/bypass-фаз.
При gradient_checkpointing (cell-4: ON) `_checkpointed_block` пере-исполняет forward
на backward, явно протаскивая кэши зеркала (`stack.py:1271-1289`); `_ggeo_freeze`
гасит повторные control-записи.

### 2.12 Bridge-инъекция до блока (зона 4, но входит в поток зоны 2)

`stack.py:537-541` вызывает `bridge.inject_layer(i, h, maturity=mat_gate[i],
tau_norm=tau_config.tau_norm[i])` **перед каждым слоем**:

```
combined = Σ_neigh w · bridge_stream[neighbour]        # EMA-поток (0.9/0.1)
scale = tanh(stream_log_scale) · maturity · inj_strength(τ_norm)
inj   = scale · stream_proj(combined)
return h_l + inj                                        # bridge.py:121-169
```

На init `stream_log_scale=0` → `tanh(0)=0` → `inj=0` (тождество), но **после
прогрева** `inj≠0`: вход блока 0 уже не равен `h₀` (вопрос (д) агента 1 подтверждён;
[M33-комментарий] поток пишется в обоих режимах). `probe_layer(h.detach())` —
проба отцеплена; её обучение идёт через `bridge.loss` (InfoNCE). Матчур-гейт
`mat_gate` (train: max(ramp, readiness), eval: последний ramp) умножает инъекцию —
см. F2B-04 у агента 3/4.

### 2.13 Состояния, кэши, NaN-предохранитель, режимы

- **NaN-guard `_chk`** (`block.py:421-427`): только training; при NaN/inf любого
  промежуточного тензора возвращает `h*NaN` и NaN-состояния (быстрый fail, не тихая
  порча). Записывает `self._nan_at` (диагностика).
- **`pen`-канал состояния вестигиален**: `block.py:444-445` при `pen is None` берёт
  `mirror._cached_pred_error_norm`; возвращаемый state-tuple несёт **тот же входной
  `pen`** (`block.py:775`), т.е. `state[4]` навсегда остаётся тем, что пришло
  (обычно None) — фактический источник pen — кэш зеркала (1-шагово-старый). Это
  согласуется с M10 (одинаковость train/eval), но означает, что поле `pen` в state
  можно считать мёртвым.
- **Режимные различия внутри блока**: (i) `_chk` — только train; (ii) шум `i_gate`
  — только train; (iii) чтение `_traj_state` — только `not training`; (iv) запись
  кэшей зеркала — оба (M33); (v) `_mlp_ratio` EMA — оба; (vi) gradalign-хук — только
  train. Остальное train≡eval.
- **dtype**: блок в основном fp32; `autocast(enabled=False)` вокруг зеркала, VPM и
  spectral (fp32-якоря); `_ln` overflow-safe; VSA-скан fp32. При `use_amp=True`
  (дефолт False, cell-4 False) landmines: `conv_state` dtype между окнами
  [audit 02b §4.2], `index_copy` в UCL [audit 02b §4.1, закрыт B10 `.to()`].
- **Снапшот/восстановление**: `snapshot_runtime_buffers` покрывает все буферы +
  `_last_bus/_intent_stream/_reasoning_*` + per-layer `_cached_hp/_cached_pred_k/
  _cached_pred_error_norm/_cached_gate/_cached_usefulness` + `_traj_state`
  (`stack.py:1178-1220`). **Найденный дефект**: `reset_cache` пытается обнулить
  `_traj_state` **у зеркала** (`stack.py:1168-1176`, цикл по `mir`), но живой
  атрибут живёт на **блоке** (`EVABlock._traj_state`), у зеркала его нет
  (`hasattr(mir,'_traj_state')==False`) — т.е. `reset_cache()` траекторный перенос
  не чистит (снапшот — чистит/восстанавливает). При `load_state_dict` (resume в
  новом процессе) атрибут и так None; риск — same-process rollback/`NaN`-skip:
  следующий eval прочитает до-rollback carry. Отдать агенту 5.

---

## 3. Что зона делает с входным контрактом (проверка/преобразование/потеря)

### 3.1 Динамика шкалы потока: как растёт/падает ‖h‖

[mini-прогон, core-only, eval, B=1, L=64, seed 0] — нормы (mean, min, max):

| Точка | ‖·‖ mean | min | max | pair-cos |
|---|---|---|---|---|
| `h₀` | 2.447 | 2.389 | 2.499 | 0.9243 |
| после блока 0 | **66.87** | 24.37 | 83.18 | 0.8597 |
| после блока 1 | **83.46** | 27.80 | 161.21 | 0.7441 |
| выход (final_norm) | 22.627 | 22.627 | 22.627 | 0.7441 |

Прирост ×27 уже на первом блоке. Декомпозиция [mini-прогон + формула]:
raw-нормы веток L0 — conv 4.58, bind 3.10, **mem_all 342.4**, mirror 2.96, VPM 7.30,
MLP 12.54; read-веса `w_q≈1/√512=0.0442` дают из `mem_all` лишь ~15, но первый
момент читается `w_q_mu ~ N(0,1)` без 1/√D-подавления, поэтому именно VSA-чтение
(и первый момент) даёт основной вклад в `enhanced`. Ни один branch_cap (1e4) не
срабатывает; M50 (1e3) тоже (66–83 ≪ 1e3). Это и есть источник продакшн-диапазона
«healthy streams 30-70» (`stack.py:385`): скачок происходит на первом блоке, дальше
рост медленный (2 слоя: +25 %; на 24 слоях — суммарный дрейф, ограниченный cap'ом).
[mini-прогон, SMALL defaults: 63.66 → 74.99 → 22.626, lacuna 0.9456] — включение
bridge/UCL/intent/maturation на init почти не меняет шкалу (их гейты нулевые/малые).

Связь с контрактом агента 1: `h₀≈4.40` (prod) / 2.45 (mini) — на порядок ниже
порога 1e3; шкала h₀ не сохраняется до головы (final_norm → RMS≈1), но ствол несёт
её DC/common-mode (pair-cos 0.95) через все блоки (mini: 0.9243 → 0.7441 — часть
common-mode размывается ветками, но DC доминирует и на выходе).

### 3.2 Что происходит с 2496 «лакунными» dim (количественно)

- `h₀`: `e_l = 0` **точно** — [mini-прогон] lacuna-share = 1.2e-7 ≈ 0 (подтверждение
  агента 1: h₀ лежит в span{readout}).
- После блока 0: lacuna-share = **0.9509**; после блока 1 = **0.9432**; на выходе
  0.9432. [mini-прогон defaults: 0.9456]. То есть ствол **обильно пишет в лакуну**
  с первого же блока.
- Доля каждого блока в лакуну [mini-прогон, lacuna-share выхода ветки]:
  conv 0.941/0.951, bind 0.984/0.976, mirror 0.981/0.986, VPM 0.989/0.979,
  MLP 0.994/0.984. Все ветки — плотные в D (bind/VPM — D→K→D, spectral — DCT-поворот,
  conv/MLP — по своим осям), поэтому их энергия преимущественно вне K-мерного
  «известного» подпространства.
- Для головы (агент 5): известная доля нормы = `√(1−ell²)`; [mini] 0.329 при
  ell=0.9432 (K=16/D=512). В продакшне комментарий M55b (`embedding.py:303-307`)
  даёт `ell ≈ 0.97` → known-share ≈ 0.243. Т.е. z-канал головы (K=64 скаляров) видит
  лишь ~24 % нормы финального потока; остальное — лакуна. Лакуна не «мёртвая»: она
  (i) питает следующий слой (conv/spectral/mirror/VPM/bind перемешивают D→K→D и
  возвращают часть лакуны в K-сегменты), (ii) читается phantom-basis (Kp=32
  ортонормированных направления в полном D; `phantom_mix` zero-init — на старте
  канал выключен). **Структурного** канала «лакуна → known-bits» кроме phantom нет.

### 3.3 Проверка/потеря входного контракта агента 1

| Пункт агента 1 | Статус в зоне 2 |
|---|---|
| (а) K/S/codebook прогона | блок использует `bind_K`, а не `code_dim`; cell-4: bind_K=32/code_dim=64/twin_free; CLI: bind_K=64/code_dim=32/legacy; чекпойнт: legacy K=32. Явно зафиксировано (§1.3) |
| (б) h₀ 4.40 / RMS 0.087 / DC 95.11 % | подтверждено; DC не нормируется, идёт в ствол как есть; M31 нормирует только входы ветвей |
| (в) readout тай, голова видит K=64, лакуна 2496 | подтверждено: блок не читает basis; лакуна после блока 0 — 95 % (mini); e_l(h₀)=0 |
| (г) эрозия basis 1.000→[0.5146,0.6423] | **не видна блоку** (basis не в графе блока); влияет только на z-масштаб головы. Риск для зоны 5 |
| (д) bridge меняет h до блока 0 | подтверждено; на init inj=0, после прогрева ≠0 (`tanh(stream_log_scale)·maturity·inj_strength`) |
| (е) финальная норма | формула и позиция подтверждены; RMS≈1; `final_norm_w` — буфер |

**Потери контракта:** позиций в h₀ нет (conv/scan дают их внутри блока); `h_emb`
мёртв (подтверждено агентом 1, блок его не касается); DC/common-mode не подавляется
(`embed_center=False`); K-кодовая структура h₀ после блока 0 перемешивается со
всеми D-координатами (лакуна).

---

## 4. Передача следующему (агенту 3: Mirror + GroupedMLP + зрелость)

Точный выходной контракт зоны 2 (величины, которые видит/производит `mirror.py`,
`mlp.py`, `maturation.py`):

| # | Величина | Форма / dtype | Шкала, единицы | Инварианты и оговорки |
|---|---|---|---|---|
| 1 | `h_in` слоя (поток) | (B,L,D) fp32 | mini: 2.45→66.9→83.5; prod healthy 30–70; cap 1e3 | true residual accumulator; после блока капнут M50; до UCL следующего слоя — нет |
| 2 | `_ln(h)` | (B,L,D) fp32 | per-position RMS≈1 | amax-rescaled; без обучаемого гейна (`pre_ln_w` — буфер); вход зеркала (`.float()`), conv, bind, VSA, VPM, spectral, MLP |
| 3 | `mem_all` | (B,L,D) fp32 | mini 342/479 (raw) | VSA-чтение всех 4 шкал, `w=σ(scale_w)`; вход `mirror(h, mem_all)` |
| 4 | `hp_cached` | (B,L,G,k) detached | k∈{8,16,32} | 1-шагово-старый; shape-lock (B,L), иначе игнор; вход write_mod |
| 5 | `pen` | (B,L) detached | per-position RMS, ≤~2 (замок B10) | 1-шагово-старый; decay-модуляция и i_gate-boost; state[4] вестигиален |
| 6 | `gs_i` | (1,1,D) detached clone | — | global_state слоя, EMA-источник |
| 7 | `diff`, `tanh_bias_mod`, `pred_scale_mod`, `spectral_mod` | python floats | — | от AdaptiveController; без градиента |
| 8 | `intent` / `salience` / `maturity` | (1,1,G,k_i) live / (B,L,1) detached / tensor | — | intent — от intent_probe (граф), salience — от головы (detach), maturity — `mat_gate[i]` |
| 9 | `mirror` | (B,L,D) fp32→h.dtype | mini 2.96/4.45 | добавляется в поток напрямую `cap(mirror)`; **не** гейтится `mem_mod` |
| 10 | `mlp_mod`, `mem_mod` | (B,L,G) | σ-гейты | mlp_mod → SwiGLU-гейт MLP; mem_mod → bind **и** память |
| 11 | `hp` | (B,L,G,k) live | — | вход `read_mod` (w_q_dyn) |
| 12 | `pred_error_norm` (новый) | (B,L) detached | per-position RMS | пишется в `mirror._cached_pred_error_norm` (оба режима) |
| 13 | Состояния | `mem_state` (B,4D) fp32, `mu_state` (B,4D) fp32, `conv_state` (B,D,47) h.dtype, `traj_state` (B,2,L,K)/None | — | все `detach()`-нуты стеком (`stack.py:695`); conv_state dtype = h.dtype (AMP-landmine) |
| 14 | Кэши блока | `_cache_conv_out`, `_cache_bind_out`, `_cache_mirror_out`, `_cache_mlp_out`, `_cache_mlp_mod`, `_gradalign_tgt` | live-граф | train-only; branch-loss (weight 0.1 cell-4), gradalign (0.3 Colab / 0.0 train.py) |
| 15 | Выход зоны | `h` (B,L,D) → final_norm → RMS≈1; `out` в head | fp32 | после final_norm + logit-cache augment (gate≈4.5e-5 → identity на init) |

**Ничего из перечисленного не теряется на стыке зона 2 → зона 3**, кроме явно
отмеченного: `pen`-канал state мёртв (реальный pen — кэш зеркала), basis/readout
блоком не читается, лакуна (~95 % нормы) в known-bits головы не попадает.

---

## 5. Связи с другими зонами

- **Зона 1:** h₀ — вход; DC/common-mode 95 % не нормируется; `basis`/`code_dim` в
  блоке не используются (только `bind_K`); bridge меняет h до блока 0.
- **Зона 3 (mirror/MLP/maturation):** зеркало вызывается внутри блока под
  `autocast(enabled=False)`; его `mlp_mod`/`mem_mod`/`hp` — живые управляющие
  сигналы; `_cached_hp`/`pen` — 1-шагово-старые; `mat_gate` гейтит bridge/UCL/банк;
  `mirror k` staircase 8/16/32 против `bind_K`/`code_dim` — разная нарезка D.
- **Зона 4 (память/UCL/bridge):** bridge-инъекция до каждого блока (scale =
  tanh·maturity·inj_strength); UCL-инъекция после блока 0 (`h + col_out`, ≤25 %
  локальной нормы); logit-cache augment в конце forward; intent-bus питает зеркало.
- **Зона 5 (голова):** получает `out` после final_norm (RMS≈1) + cache-augment;
  лакуна ≈ 95 % (prod ≈ 97 %) — phantom-basis (Kp=32) единственный структурный
  канал лакуны; `head_temper` использует `_mem_dir` из банка (зона 4).
- **Зона 6 (обучение):** branch-loss анкер M51 (`losses.py:224-261`); gradalign-хук
  (`block.py:752-758`); gradient_checkpointing пере-исполняет forward
  (`stack.py:1271-1289`); AGC/LLRD роли; `stream_cap`/`branch_cap` — конфиг.

---

## 6. Открытые вопросы и риски для следующего агента

1. **`exact_memory` (VPM) не причинна**: `A = σ(qkᵀ/√k)` без маски — позиция t
   читает будущие позиции окна. В streaming L=1 безвредно; в обучении это утечка
   будущего в ветку (гейт `precision.mean()>0.3` на mini открыт: 0.599/0.332).
   Решение — за зонами 5/6 (или явная causal-маска).
2. **`pen` в state-tuple мёртв** (`block.py:775`): реальный источник — кэш зеркала;
   при аудите/переносе состояния не полагаться на `state[4]`.
3. **`conv_state` dtype = h.dtype** (`block.py:468`): при включении AMP перенос
   между окнами может упасть на `torch.cat`; при `use_amp=False` — не проявляется.
4. **`pre_ln_w` — буфер**: у M31-LN нет обучаемого гейна; если понадобится
   per-branch learnable scale — это изменение контракта (сейчас его нет).
5. **M51 не покрывает UCL/bank-инъекции**: их вклад добавляется после cap'а и до
   следующего M50; аномалия в UCL/bank может превысить 1e3 на один слой.
6. **Зеркальные пороги после B10**: pen теперь per-position RMS (≤2), но
   downstream-пороги (UCL u_gate ~2.72, pred_scale_mod) калибровались под старую
   шкалу ~8.1 [audit 02b F2B-06]; нужна пере-калибровка (зона 3).
7. **`_vsa_log_param` получает градиент** ([mini-прогон]: 0.225) — в продакшне
   tau_s идёт из него (`stack.py:276`), роль LR/оптимизатора не проверена здесь
   (зона 6).
8. **M18-вердикт**: age-addressing спирали закрыт (purity 0.389→0.367 при K=32→64);
   не тратить бюджет на фазовые теги.
9. **Лакуна ≈95 % (prod ≈97 %)**: любые D-симметричные операции (spectral DCT,
   conv, dense W_out) автоматически пишут в лакуну; для зоны 5 — phantom-basis
   остаётся единственным структурным каналом, его Kp=32 из 2496 — узкое место.
10. **`_traj_state`/`_cached_*` покрыты снапшотом** (M33/M45/B10) — при написании
    `reset_document_state()` агенту 5 учесть, что `_traj_state` уже там; но
    `reset_cache()` его **не чистит** (атрибут на блоке, цикл — по зеркалу,
    `stack.py:1168-1176`; см. §2.13) — same-process rollback оставит до-rollback
    carry, который прочитает следующий eval. При generation-путях (без снапшота)
    манифолд-буферы (если `traj_manifold=True`) пишутся на eval-форвардах.
