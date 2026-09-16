# Зона 6 — Обучение: сборка лосса, LossBalancer, контроллеры, оптимизатор, данные, замкнутый контур

**Агент:** 6 из 6 (замыкающий, `docs/analysis/CHAIN.md`).
**Дерево:** commit `362a224` (M53e), torch 2.13.0+cpu, Python 3.14.6, Windows.
**Вход зоны:** `ce`/логиты и телеметрия головы (зона 5), aux-кэши зон 2–5, `mat_gate`/`tau_config`
(зона 3), `_mem_dir`/`bus_bias` (зона 4), `h` (зоны 1–4), данные (токены), чекпойнт.
**Выход зоны:** обновления весов (grad → optimizer.step), расписания (LR, depth, warmup),
сборка лосса и балансер, чекпойнт/resume, замкнутый контур обратно в зону 1.
**Предыдущие отчёты:** прочитаны полностью (`zone_1_codes_embedding.md:1-595`,
`zone_2_block_core.md:1-716`, `zone_3_mirror_mlp.md:1-647`, `zone_4_memory.md:1-663`,
`zone_5_head.md:1-637`); их выводы, открытые вопросы и контракты учтены явно (§1.2, §2.7–2.8, §3, §6).

**Методика (правило полигона соблюдено).** Продакшн-модель не инстанцировалась, чекпойнты
не грузились, `scripts/analyze.py` не запускался. Анализ — чтение кода (`file:строка`).
Вычисления — только на мини-конфиге `SMALL = dict(n_layers=2, D=512, mlp_groups=4,
code_dim=16, code_sparsity=4, vocab=1820)` (`%TEMP%\opencode\zone6_groups.py`): измерено
число параметр-групп `build_optimizer` (14/25/8) и роль `token_bias`; помечено **[mini-прогон]**.
Аналитические выводы — **[формула]**. Числа инцидентов цитируются из отчётов зон 2–5 и
комментариев кода с их метками (M47/M53d/2970/A2).

---

## 1. Граница зоны и входной контракт

### 1.1 Где начинается зона: два фактических боевых контура

В дереве **два независимых тренировочных цикла**, и они не эквивалентны:

| | `scripts/train.py` (CLI) | `notebooks/eva_colab.ipynb` (боевой) |
|---|---|---|
| Геометрия | D=4096, 24 слоя, `bind_K=64`, `code_dim=32` legacy, `mlp_groups=8` | D=2560, 24 слоя, `bind_K=32`, `code_dim=64` twin_free, `mlp_groups=32` |
| Оптимизатор | **всегда AdamW** (`cfg.optimizer='adamw'`; CLI-флага `--optimizer` нет) | `eva_proj` (EVAAdamW + AdamP) |
| `memory_bank` | **выключен** (config default False; CLI не включает) | включён (`mem_l1_slots=3, mem_l2_slots=32`) |
| `surprisal_weight` | 0.0 (config default) | 0.3 |
| `branch_balance_weight` | 0.0 (M51-анкер не работает) | 0.1 |
| `gradalign_weight` | 0.0 | 0.3 (ставится в цикле, ipynb:827) |
| `per_layer_ls_lr` | жёстко False (ipynb:160=True) | True |
| `mlp_depth_lr_exp`-буст | вызывается (`train.py:230`) | **не вызывается** |
| `torch.manual_seed` до init | **нет** | 1234 (M56b, ipynb:244) |
| AMP | `autocast` только вокруг `embed_tokens` (`train.py:501-502`) | `autocast` вокруг всего forward+head+loss (ipynb:887-891), `use_amp=False` в cell-4 |
| Ctrl+C | чекпойнт **не пишется** (`train.py:682-685`) | `best.pt` перезаписывается (`ipynb:1235-1242`) |
| OOM-ladder | только стартовый блок B19 (`train.py:840-846`, см. §6.1) | runtime-ладдер recompute→batch→window + regrow (ipynb:899-950, 1019-1032) |

Всё ниже, если не сказано иное, описывает **оба** цикла, различия помечены явно. «Продакшн»-
числа ниже — cell-4 (боевой ноутбук), как в зонах 1–5.

### 1.2 Что именно приходит от агента 5 (проверка контракта `zone_5_head.md:512-531`)

| Величина зоны 5 | Как используется зоной 6 | Проверка |
|---|---|---|
| `ce_loss` (surprisal-взвешенный в train) | первый аргумент `balancer.backward`; `_cached_losses['ce']`; печать `loss=` | подтверждено; **не сравним с val** (M46) |
| `ce_raw` | `_cached_losses['ce_raw']` — честная метрика | подтверждено |
| `head_wall`, `phantom_l1` | aux-термы → балансер; **не** в `_cached_losses` → train.py их не печатает | подтверждено (ipynb печатает merged aux) |
| `_last_u`, `_last_p` | входы wall/L1 (пины графа; отцепляются `release_step_graph`) | подтверждено |
| `_last_srl`, `_last_sat`, `_last_conflict`, `_last_lacuna*`, `ph_*` | только `head_telemetry()`; train.py её не зовёт; в чекпойнт не идут | подтверждено |
| `lm_head` параметры (`bit_bias`, `log_temp`, `emphasis_gain`, `phantom_*`, `lacuna_w/b`, `log_eta`, `token_bias`) | `build_optimizer` роли (matrix/scalar/scale_inv) | `token_bias` — `scalar`, wd=0 в обоих путях [mini-прогон] |
| `readout` (=`embed.basis`) | группа λ⁻² (LR 8.87e-5 при base 3e-4) | [mini-прогон] |
| `release_step_graph()` | обязателен после `optimizer.step()` — оба цикла зовут | подтверждено |

### 1.3 Что приходит от зон 1–4 (сводно, с их открытыми вопросами)

- **Коды не обучаемы** (з1): `embed.codes` — non-persistent буфер; обучение может менять
  только `embed_mix`, `basis/readout`, `bit_bias`, `log_temp`, `emphasis_gain`, `token_bias`.
  Fingerprint B8 (`training_control.py:419-465`) страхует resume. Эрозия `‖B_k‖` 1.000→[0.5146,
  0.6423] (з1, F2A-09) в логах не отслеживается (§6).
- **Шкала потока** (з2): M50 `stream_cap=1e3`, M51 `branch_cap=1e4`; healthy 30–70; финальная
  норма → RMS≈1. Зона 6 не нормирует h повторно.
- **`pen` — таймер, readiness — часы** (з3): `pen_init=1.0` ⇒ `readiness(t)` детерминирован
  (mat≈0.82 к ~3135); `mat_gate` гейтит bank (0.3)/UCL (0.1)/bridge; **зона 6 использует это
  как расписание, не как компетентность**.
- **Мёртвые каналы** (з3/з4/з5): `pred_weight`, `novelty_gate`, `_mlp_ratio`,
  `cache_grad_norms`, M55-рост, `layer_gate` LBG — все перечислены в §2.7.
- **`reset_cache` неполон** (з2/з4): не чистит `_traj_state`, банк, UCL, `bridge_stream`,
  `_last_*`; в циклах он вызывается на flush (каждые 495 шагов), R6-resume и в watchdog-ветке
  (последней в бою нет — §2.4).

---

## 2. Математика зоны

### 2.1 Полная сборка CE (`core/losses.py:10-71`)

**Coded-путь (production, `head_normalize=True`), `losses.py:20-55`:**

```python
bus_bias = bus_head_proj(_last_bus.expand(B,L,·,·).reshape(B,L,−1))   # только тут
lp = lm_head.log_probs_for_target(h.reshape(-1,D), t.reshape(-1), bus_bias=bus_bias)
ce = −lp                                       # (N,), N = B·L
mask = (t != 0)                                # PAD=0
if cfg.mask_eos: mask &= (t != 2)              # False ⇒ EOS учится
ce_raw = Σ(ce·mask)/Σmask                      # M46: честная метрика
if training and sw > 0:                        # sw = cfg.surprisal_weight
    w = σ( sw·2·( ce/mean(ce) − 1 ) )          # no_grad; mean(ce) = Σce/Σmask
    ce_loss = Σ(ce·w·mask)/Σmask
else: ce_loss = ce_raw
```

**Legacy-путь (`losses.py:56-71`, `head_mode != sigmoid_coded`):** `F.cross_entropy(...,
reduction='none')`, та же маска, но `w = σ(2·(ce/mean(ce) − 1))` — **без множителя `sw`**
(сдвиг масштаба взвешивания между путями; в бою не проявляется, coded-путь).

Инварианты: `w ∈ (0,1)`; `mean(w)≈0.5` при sw=0.3 ⇒ train-CE систематически **ниже** eval-CE
в ~2.0–2.3× (M46-комментарий `losses.py:42-47`); поэтому и логируются оба (`ce`, `ce_raw`).
`_cached_losses['ce']` — взвешенный в train, в eval `ce_loss = ce_raw` (`stack.training=False`).

**Голова вызывается 4 раза за train-шаг** (з5, §1.1): `_knowledge_signal`/`_last_conf`
(no_grad), `observe_output` (no_grad, но аргумент `model.lm_head(out)` вычисляется ДО вызова —
строится и тут же выбрасывается полный граф головы, лишний forward), `compute_losses` (CE-путь).
Wall/L1 читают `_last_u/_last_p` **последнего** вызова — т.е. CE-путь (порядок зафиксирован
`train.py:507-508`, `ipynb:890-891`).

### 2.2 Все aux-термы: формула, шкала, что тянет

Все значения возвращаются **сырыми** (комментарий `losses.py:465-470`): никаких per-loss
множителей, кроме явно «запечённых» в значение (nuc/wall/L1/gradalign). `*_weight`-поля
конфига в большинстве случаев — только on/off.

| Терм | Формула (код) | Источник | Что тянет / градиент | Условие |
|---|---|---|---|---|
| `pred` | `mean_layers mse(pred_k_aux, hp.detach())` | `_pred_loss_term` (з3) | `alpha_diag`, `W_proj` | train; всегда |
| `gate_l1` | `mean_layers expert_gate.mean()` (**не L1 логарифмов**, `mirror.py:972`) | `_cached_gate_l1` (live) | `w_gate/b_gate/w_delta_gate/gate_bias` | всегда |
| `reinforce` | `mean_layers mse(u, g)` | `_cached_usefulness` live, `_cached_gate` detach | только `usefulness_predictor` | всегда |
| `balance` | `mean_layers (HHI−1/G)/(1−1/G), clamp≥0`, `usage=expert_gate.mean(0,1)` | `_cached_gate_usage` live | `w_gate…`, predictor | всегда (cfg `balance_weight` не читается) |
| `diversity` | `mean_layers α_l·mse(corr(‖group_out‖), I)`; `α_l=1−e^{−τ_l/τ_min}` | `mlp._cached_group_out` live | `W_gate/up/down`, `norm_w` | всегда; **τ-тай старой формы** (не v3) |
| `nuc` | `mean_layers (1−SR/rank)`; `SR=clamp(‖W‖_F²/σ̂max²,1,rank)`; 4 power-iter `no_grad` | `bind.W_proj.weight` | Frobenius-член W | `nuclear_weight=1e-5` → **значение уже ×1e-5** |
| `orth` | `mean mse(ŴᵀŴ, I)`, `Ŵ=W/‖col‖` | bind W_proj | W_proj | `orth_weight>0`; cell-4 = 0 |
| `w_m2v` | `mean_layers (w_mem2v.mean()−target)²`; `target=m/(1+e^{−(logτ_l−logτ_mid)})` (detach) | `w_mem2v` | `w_mem2v` | `w_m2v_hierarchy_weight=0.01` (только гейт) |
| `intent_tau` | `mean_layers (α_act−α_tgt)²`; `α_act=clamp(1−τ_mid/(√D·τ_l),0)`, `α_tgt=0.3/(1+e^{−(logτ_l−logτ_mid)})` | `tau_config._tau_dev` (live) | `_tau_dev` | `intent_bridge` и вес>0 (0.01) |
| `branch` | `mean_layers [log vc−log vb]²+[log vc−log vm]²+[log vb−log vm]²` + `anchor·Σ(log v_s−log ref)²` | `_cache_conv/bind/mirror_out` live | ветки ствола | `branch_balance_weight>0` (**cell-4 0.1**); `anchor=0.5` |
| `signal_ent` | `mean_layers Σ p·log p`, `p=norm(σ(_signal_log_weights/τ_signal_used))` ⇒ минимизация −H | mirror | `_signal_log_weights`, `_tau_signal_log` | всегда |
| `gradalign` | `weight·mean_layers mse(m̂t, ĝt)`, `ĝt=detach(‖∂CE/∂mlp_out‖)/max` | `_gradalign_tgt` (hook), `_cache_mlp_mod` live | `mlp_mod`-путь | `gradalign_weight>0`; **ipynb 0.3, train.py 0**; BYPASS |
| `ls_reg` | `mean_layers relu(log_scale−2.3)²` | mirror | `log_scale` | всегда |
| `div` | `−(var_G(σ(log_scale))·mean + √(d/G)·var_d(σ(log_scale)))` | mirror | `log_scale` | `div_weight>0` (10.0) — только гейт |
| `gate_repulse` | `mean_layers −H(softmax(usage))` | usage | гейты | `gate_repulse_weight>0` (0.3) |
| `alpha_novelty` | `mean_layers −var_G(α_diag.mean(-1))` | α_diag | `alpha_diag` | `alpha_novelty_weight>0` (0.05); сам вес применяется **в зеркале** (adaptive-α push, з3) |
| `decorr` | `mean_layers mean_pairs cos²(центр. взвеш. сигналов)` | `_cached_decorr` live | `_signal_log_weights`, `_tau_signal_log` | всегда |
| `mem_tau_reg` | `Σ_{L1,L2}(log_tau−init)².mean() + 0.1·relu(l1.log_tau−l2.log_tau).mean()` | банк | `log_tau` уровней | `memory_bank` (только ipynb) |
| `bridge_conn` | `mean_layers InfoNCE(pred_l, tgt; temp=σ-клэмп e^{nce}) + 0.1·mean_pairwise_cos` | `bridge._preds`, цель — detach-центрированный embedding | `stream_proj/probe` (не в h) | `bridge_conn>0` (0.1) — только гейт |
| `tau_dev_reg` | `0.01·mean(_tau_dev²)` | TauConfig | `_tau_dev` | всегда |
| `lbg_diversity` | `(log n − H(softmax(gates)))/log n` | LBG-диагностика | `gates[l].log_tau` | LBG и `global_ready` |
| `head_wall` | `1e-3·relu(|u|−6)².mean()` (u pre-phantom) | `_last_u` | голова | `head_u_wall>0`; не логируется train.py |
| `phantom_l1` | `1e-4·|p|.mean()` | `_last_p` | phantom_mix/basis | `head_phantom_l1>0`; не логируется train.py |
| `pred_w` | `mse(head.pred_w, I)` | — | — | **мёртв**: у `SigmoidCodedHead` нет `pred_w` (`losses.py:456-463`) |

**Сборка aux_dict (`losses.py:471-569`):** терм попадает в словарь только если он `!= 0`
(тензорные скаляры). Итог cell-4: ~24 живых терма; train.py — без `branch`, `gradalign`,
`surprisal`, `bridge`? (bridge есть — config default 0.1), без `mem_tau_reg` (bank off).

**Логирование:** `_cached_losses` (`losses.py:372-385`) содержит фиксированный набор
`ce, ce_raw, pred, gate_l1, reinforce, balance, div, gate_repulse, alpha_novelty, signal_ent,
ls_reg, decorr` (+ `lbg_*`, `mb_*`). **Не попадают**: `nuc, orth, w_m2v, intent_tau, branch,
gradalign, diversity, tau_dev_reg, bridge_conn, mem_tau_reg, head_wall, phantom_l1`.
`train.py:605-606` печатает только `_cached_losses` ⇒ половина aux-ледера невидима в CLI-логе;
ноутбук печатает `aux_dict ∪ _cached_losses` (`ipynb:1091-1103`).

### 2.3 LossBalancer: точная математика (`training_control.py:481-756`)

Классовый docstring (`:481-505`) описывает **устаревшую** формулу
`g_final = g_CE + max(cos,0)·(‖g_CE‖/‖g_aux‖)·g_aux`; фактический код — per-параметрическая
(B2, аудит C). Точный `backward` (`:636-756`):

```python
params = [p for p in parameters if p.requires_grad]
bypass = {k: aux.pop(k) for k in BYPASS_AUX if tensor&requires_grad}   # BYPASS_AUX=('gradalign',)
# 1) aux/bypass-фаза: заморозить gradalign-хук
for l in phase_model.layers: l._ga_record = False
gce = autograd.grad(ce, params, retain_graph=True, allow_unused=True)
gau = autograd.grad(Σ aux_tensors, params, retain_graph=True, allow_unused=True)
# cos и нормы — попарными dot (без flat-копий); cos → self.last_cos (диагностика)
cos = ⟨gce,gau⟩/(‖gce‖·‖gau‖);  scale = min(1,max(0,cos))·‖gce‖/‖gau‖   # ← scale НЕ используется
# 2) применение (no_grad):
p.grad = gce.clone() if gce is not None else zeros
b = gau · 1[gce·gau > 0]                       # per-coordinate sign-agreement
s = clamp(‖gce‖/(‖b‖+1e-12), max=1.0)
p.grad += b·s                                  # per-param: ‖Δ_aux‖ ≤ ‖g_CE‖
# 3) bypass (gradalign) — ПОСЛЕ, под тем же правилом, но относительно p.grad:
gb = autograd.grad(Σ bypass, params, retain_graph=retain_graph)
b = gb · 1[p.grad·gb > 0]; s = clamp(‖p.grad‖/(‖b‖+1e-12), max=1.0); p.grad += b·s
# 4) разморозить хук
for l in phase_model.layers: l._ga_record = True
```

Ключевые свойства и что было исправлено аудитами:

1. **Глобальный cos-гейт заменён per-параметрическим** (B2): старая форма обнуляла весь aux,
   если суммарный aux-градиент ортогонален CE; игрушечный тест 8.000 vs 2.000. `scale`
   вычисляется, но в применении не участвует (мёртвая переменная; в `last_cos` уходит только
   `cos`).
2. **Per-параметрический предел `‖Δ‖≤‖g_CE‖`** — буквально (комментарий `:640-648`); для
   bypass — предел относительно **уже собранного** `p.grad` (B14: раньше raw `.backward()`
   давал 51×CE и попадал в gradalign-цель).
3. **`clone()` выходов `autograd.grad`** обязателен: AGC делает `p.grad.mul_` in-place, а
   autograd отдаёт view внутренних буферов (`:668-671`).
4. **Один общий aux-градиент**: `aux_total = Σ aux_tensors` — относительные сырые величины
   термов задают направление суммы; per-термовые ограничения отсутствуют (ограничивается
   только суммарный sign-agreeing компонент).
5. **`phase_model._ga_record=False`** на время aux/bypass: hook в блоке записывает
   `‖∂CE/∂mlp_out‖` только на CE-фазе (иначе n−1 слоёв хранили aux-градиент, rel-err 1.0).
6. **`_update_balance`/`loss()` — мёртвый режим**: `loss()` (balance-mode) циклы не вызывают;
   `align`-флаг в `backward` не читается (всегда align-путь); `ema_ce/ema_A/ema_aux` остаются
   `None` и такими же пишутся в `state_dict` (`:522-538`). `align_cap` принимается и не
   используется. `grad_geometry` (B6) из циклов удалён (M48), но жив для офлайн-диагностики.
7. **Порядок и фазы**: `balancer.backward(ce_s, aux_s, model.parameters(), phase_model=model)`
   → `apply_tau_lr` (τ-LLRD × ls_m) → `clipper.clip` → `optimizer.step()`. Границы
   «‖aux‖≤‖CE‖» действуют в пространстве **до** τ-LR/AGC; τ-LR умножает CE- и aux-компоненту
   одного параметра одинаково (соотношение сохраняется), AGC затем режет абсолютную норму.
8. **Взаимодействие с wall/L1 (з5)**: сырые значения (1e-3/1e-4 уже внутри) идут в общий aux-сумме;
   `_last_u` — **pre-phantom**, поэтому стена не ограничивает фантомный канал; `_last_p` —
   post-gate. Ни wall, ни L1 не попадают в `_cached_losses` (train.py их не видит).
9. **Взаимодействие с M51-анкером**: `_ref = stack._branch_var_ref` — медленная EMA (0.999/0.001),
   засевается первым наблюдением, обновляется в `compute_losses` под `no_grad` (`losses.py:249-261`);
   `_ref` — **plain-атрибут**: не в `state_dict`, не в снапшоте, не в чекпойнте ⇒ после resume
   анкер пере-базируется на текущий масштаб (дыра проброса, §2.8).
10. **τ-поля внутри aux**: `diversity` использует `α_l=1−e^{−τ_l/τ_min}` (старая форма),
    `intent_tau` — `α_act=1−τ_mid/(√D·τ_l)` (v2-подобная), тогда как зеркало/шина используют
    v3 `intent_alpha = 1−1/τ_l` (`tau_config.py:170-180`) — «единый авторитет» τ нарушен:
    регуляризаторы регулируют **не ту** α, что потребляет модель.

### 2.4 Контроллеры

#### FailureDetector (`training_control.py:131-412`) — детектор-сирена D6

```python
a = 0.99; a_slow = 1 − (1−a)/10 = 0.999; min_samples = 100
# состояние сигнала: [fast, prev, n, slow, dvar, ph, phmin]
σ̂_real = sqrt(EMA_0.99(Δ²)/2);  σ = max(σ̂_real, 0.02·slow)
thresh = max(margin·slow, k_sigma·σ),  margin=0.15 (CE) | 1.0 (mlp_ratio/ig_eff/diversity)
viol = (v > slow + thresh) and (v ≥ floor)          # floor: 2.0/1.5/5.0; CE floor=−inf
# Page-Hinkley канал (медленные дрейфы):
ph = max(0, ph + Δ − 0.02·σ);  phmin = min(phmin, ph)
h  = max(230·σ̂_real²/σ, 0.05·slow);  viol |= (ph−phmin) > h and (v ≥ floor)
# подтверждение: 3 подряд (min_consecutive); cooldown=50 подавляет триггеры, но наблюдение идёт
# warmup: до cfg.warmup_steps (1200 ipynb / 500 train.py) — observe молча копит, не сигналит
# CE дополнительно под ce_armed: arm_ce() взводит ПОСЛЕ первого val-eval и пере-бутстрапит baseline
# non-finite CE → принудительный ALARM (NaN-сравнения не срабатывают)
# триггер: recover_count++, _strikes++, cooldown, print, return True  (НИКАКОГО rollback — D6)
```

`state_dict` (`:212-227`) персистит strikes/recover_count/ce_armed/cooldown/viol/stats;
`load_state_dict` терпит старый 4-полосный формат (B13). **КРИТИЧНО:** `check()` и `arm_ce()`
не вызываются **ни в train.py, ни в ноутбуке** (grep: только создание/load/state_dict) —
детектор полностью инертен, его baselines никогда не накапливаются. Комментарий
`train.py:511-514` («Audit M8: train.py never passed the protective metrics and never armed
CE — здесь watchdog был декорацией») описывает уже исправленное состояние, но фактически
вызов так и не появился. `hard_veto_ceiling` (`:468-478`) импортируется обоими циклами и
никогда не вызывается (только тесты). `nonfinite_gradient_names` (`adaptation.py:485-494`)
импортируется и не вызывается: единственный фильтр не-конечных градиентов — внутри AGC
(per-param drop).

#### DepthController (`adaptation.py:87-169`)

```python
init_k=8, inc=4, warmup=2000, k_sigma=1.0, eval_interval (1045 ipynb / 1000 train)
update(step) каждый шаг; val_loss — только на canonical eval
val_ema/val_var — EWMA с a = 1−1/max(eval_interval,100); std = sqrt(var)+1e-8
slope = val − prev
плато ⇔ slope > −k·std  И  step−last_depth_step ≥ eval_interval  И  active < max
active += inc; set_active_depth(model, active)   # requires_grad = (i < active)
```

`set_active_depth` (`:76-84`) замораживает слои ≥k по `requires_grad`. Оба цикла строят
оптимизатор **до** заморозки (все параметры в группах) и восстанавливают depth из чекпойнта
(`active_depth` + `depth_state`, B14: `_last_depth_step` персистится). Внутри одного eval
порядок: `depth.update(step, val_loss)` → `scheduler.report_val_loss(val_loss)`
(`train.py:640-641`; ipynb:1192-1194). Прямой связи с `mat_gate` нет (з3, §2.9).

#### LRController / MirrorLRScheduler (`adaptation.py:308-398`, `lr_scheduler.py:24-315`)

Фаза warmup+blend (`step < warmup+50`), `lr_scheduler.py:180-200`:

```python
mult = step/warmup (линейно);  blend=50
_alpha_override = 1 − 0.7·mult                # warmup; затем 0.3·(1−blend)
_usefulness_temp = 2.0 → 0.5 линейно; затем 0.5 + 0.45·(1−blend) → навсегда 0.5
```

После warmup (`:204-248`):

```python
var=mean_l var(log_scale); mag=mean_l |mirror|; alpha=mean_l (1−α_diag); gate_var=mean_l var(gates)
EMA τ_ema=0.99 от каждого; ratio = cur/ema
var_mult/alpha_mult/gate_mult = clamp(1/ratio, 0.5, 2.0)
mag_factor = clamp(1/mag_ratio, 0.2, 1.0)
mirror_mult = (var_mult·alpha_mult·gate_mult)^(1/3) · mag_factor
m = max(0.2, mirror_mult);  if m>1 and not _val_improving: m = 1.0;  m = min(m, lr_boost_max=2.0)
mult = m · _loss_lr_factor
pg['lr'] = orig_lr[i] · mult          # только группы, бывшие при init
```

`report_val_loss` (`:119-176`) — best-anchored damping: регресс `val > best·1.05` → `×0.5`
(пол 0.05); улучшение `val < best·0.98` → `1.0` и best обновляется; плато-зона → warm-restart
`+1/200` за eval; `_val_improving` — гистерезис (tol 0.002). `rewind()` (`adaptation.py:376-398`)
существует, но **не вызывается** (D6: rollback'а нет). `report_train_loss` — пустой pass.
`per_layer_ls_lr` (ipynb=True): `_ls_mult[l]=clamp(1/(fast/slow), 0.5, 2.0)` от fast/slow EMA
`var(log_scale)` и применяется через `apply_tau_lr`; train.py жёстко False.
**Баг train.py:** `ls_mults` применяются к `layer.base_parameters` дважды — отдельным блоком
(`train.py:557-565`) и внутри `apply_tau_lr` (`:577`); в бою дремлет из-за `per_layer_ls_lr=False`.

#### GradientClipper (AGC, `adaptation.py:414-482`)

```python
c = 0.1 (трансформерный режим; train.py:260, ipynb:754)
attach(model): c_eff(p) = c · (τ_ref/τ_l)^γ,  τ_ref=mem_tau_ref=64, γ=llrd_gamma=0.65
              (только для имён layers.<l>.*; вне слоёв масштаб 1); карта перестраивается
              раз в eval_interval шагов (U9)
clip(p): не-конечный grad → p.grad = None (M28); ‖p‖<eps=1e-3 → skip (zero-init защита);
         ‖g‖ > c_eff·‖p‖ → g *= c_eff·‖p‖/(‖g‖+eps)
```

`c_eff = c·(τ_ref/τ_l)^γ` — **та же степень, что `lr_mult=(τ_l/τ_ref)^{−γ}`** τ-LLRD: клип и
LR-распределение согласованы (одна формула, два применения).

#### τ-регуляризаторы

- `intent_tau` (`losses.py:206-223`): см. §2.2/§2.3 п.10 — регулирует v2-форму α, а не v3.
- `tau_dev_reg` (`losses.py:562-567`): `0.01·mean(_tau_dev²)` — тянет лестницу к равномерной;
  градиент в `_tau_dev` живой, но `intent_tau`/`w_m2v` тянут в другую сторону.
- `mem_tau_reg` (`losses.py:533-549`): якорь `log_tau` к init + запрет инверсии L1>L2
  (только при включённом банке; в train.py не активен).
- `diversity` τ-тай (`losses.py:130-133`): `α_l=1−e^{−τ_l/τ_min}` — третья форма α.
- `w_m2v` (`losses.py:188-204`): τ-иерархия `w_mem2v`; цель detach, параметр live (M5-фикс).

#### Прочее, что меняет сигнал

- `apply_mlp_depth_gradient_boost` (`stack.py:1294-1320`): backward-хук `grad × exp(0.1·i)` на
  все параметры MLP (не на зеркальные гейты); вызывается **только train.py:230**.
- `stack.param_groups` (`stack.py:1343-1493`, λ-таблица 14 групп) — **мёртвый дубль**:
  оба цикла строят оптимизатор через `core.adaptation.build_optimizer`.
- `model._phase_ratio_ema/_std` (`train.py:226-227`) — записываются один раз, не читаются.
- `_full_env`/`_atomic42` (`train.py:269-292`) — определены, не вызываются (сохранение
  best.pt продублировано inline, `:646-664`).

### 2.5 Оптимизатор и планировщик

**Группы (`adaptation.py:223-301`).** Ключ группы: `(role_mult, depth_mult, [role, wd, trust, cap])`.
`role_mult` (`_role_lr_mult`, `:203-220`) — λ-иерархия от `lambda_d(cfg.lambda_d)` (λ≈1.839):
`λ⁻²≈0.296` (VSA-части, `embed.*`, `lm_head.readout/proj`), `λ⁻¹≈0.544` (MLP-ядра, bind
`W_proj/W_out`), `1.0`, `λ≈1.839` (зеркальные проекции/гейты, `reasoning_gate`).
`depth_mult = llrd_decay^{max(li,0)}` (индексный LLRD; CLI `--llrd` default 1.0, cfg.llrd=0.9 —
но CLI побеждает; при ≠1.0 включается двойной LLRD и печатается предупреждение `train.py:296-297`).
`wd`: AdamW-путь — `ndim≥2`; EVA-путь — по роли (`_resolve_role`, `eva_optim.py:120-137`).
Исключён из оптимизатора только `_vsa_tau_log` (audit M7); `_vsa_log_param` включён
(з2, вопрос 7 — подтверждено: role=scale_inv, lr=base).

**[mini-прогон] число групп (SMALL):**

```
eva_proj, llrd=1.0 : 14 групп
adamw,    llrd=1.0 :  8 групп
eva_proj, llrd=0.9 : 25 групп
token_bias: ndim=1 → eva role=scalar wd=0.0; adamw wd=0.0  (в обоих путях WD НЕТ)
tau-группа: cap = dev_max/(delta_t·lr) = 0.3/(4000·3e-4) = 0.25
lr-примеры: 8.868e-05 = base·λ⁻², 5.518e-04 = base·λ, 1.631e-04 = base·λ⁻¹
```

Для cell-4 (bank/UCL/logit_cache) добавляется mem-роль и часть scale_inv-подгрупп — итог
≈15–16 групп; «16 групп» из задания — это фактическая раскладка боeвого прогона, точное число
зависит от конфига (формула выше).

**EVAAdamW (`eva_optim.py:140-292`, mode eva/eva_proj):**

```python
u = m/denom
projected (eva_proj): AdamP-проекция строк при |cos(u,W)| ≥ δ=2/√fan_in, ре-нормировка к ‖W_row‖
cautious: u *= 1[u·g > 0]; u /= sqrt(mean(mask)).clamp_min(0.5·max(0.5, 1−2/√D))
trust:    u *= floor + (1−floor)·trust, floor=0.5, trust из set_trust() по роли
          (bridge.readiness / |maturation.gate.mean|; train.py:584-592 — только eva-путь и без AMP;
          ноутбук set_trust НЕ зовёт ⇒ trust≡1.0)
slow_ema: OFF (beta_slow=0.9999, slow_mix=0.25 — параметры, но флаг False)
update_cap: u.clamp_(-cap, cap) для τ-группы (0.25 на base LR)
wd: p *= 1−lr·wd только role.wd и dim≥2
```

**Name-based restore (`train.py:101-173`):** матчинг `param_names` из чекпойнта → слот состояния
по имени; **новые параметры (M52–M56) в старом `param_names` отсутствуют → `skipped`** (слот не
восстанавливается: Adam стартует с нулей, но градиенты и обновления идут — параметр жив);
shape-mismatch → `skipped` + печать; спец-случай `bind.W_out` (частичное восстановление строк
с `exp_avg_sq=1.0` на новых); **нет `param_names` → оптимизатор не восстанавливается вовсе**
(свежий Adam). LR старых групп копируется позиционно. Затем `LRController` создаётся заново
(снапшот `_orig_lrs` от restored-значений) и `load_state_dict` восстанавливает step/EMA/
`orig_lrs` (позиционно, только при совпадении числа групп). `verify_identity_resume` (B8) —
фатальность при shape-mismatch `embed./lm_head./final_norm` или fingerprint-дрейфе.
**Баг train.py (B15-регрессия):** `mlp_gate_b_init>0` (default 0.25) **всегда** перезаписывает
`mlp_gate_b` и `hybrid_gate.log_tau` на resume (`train.py:350-357`); ноутбук это чинит гардом
«только для чекпойнтов без `mlp_gate_b`» (`ipynb:541-546`).

**Состояние планировщика (`lr_scheduler.py:271-315`):** step, last_log, `tau_*`, `orig_lrs`
(позиционно, с гардом числа групп), best_val_loss, loss_lr_factor, val_ema, val_improving,
ls_fast/slow. `_usefulness_temp` после warmup закреплён на 0.5 (з3, F4-07/12) — intrinsic-
расписание `temp(n_eff)` не действует.

### 2.6 Данные и eval-гигиена

**TokenStream (`train.py:46-74`, ipynb:352-374):** memmap uint16 → `.long()`; x = tokens[o:o+B·L],
y = x+1; возвращает `(x, y, offset+B·L, wrapped)`. Громкая проверка `id ≥ vocab` — **только
если передан `vocab`**; train.py передаёт (`:496`), ноутбук — **нет** (`ipynb:878, 926, 1178`) ⇒
B7-проверка в бою инертна.

**Батчи/куррикулум:** train.py: `B=2`, окна `seq_pool=[64,128,256,512]` по ступеням
`tau_short=32000/tau_long=96000` (`:453-470`); ноутбук: `B=2, L=224` фиксировано.
**Курсор/док-границы:** ротация потока, когда `offset==0 или offset+_need>len`; сброс
`state/gs`, `bridge.bridge_stream.zero_()`, `memory_bank.reset()`, `logit_cache.cache.clear()`,
`reset_reasoning()` (`train.py:472-496`; ipynb:857-877). **`_intent_stream`/`_last_bus` при
смене документа не сбрасываются** (з4-контракт: снапшот их покрывает, док-граница — нет).
**Seed:** data-RNG `torch.Generator().manual_seed(42)` — оба (`train.py:422`, ipynb:818);
M56b `torch.manual_seed(1234)` перед init модели — **только ноутбук** (`ipynb:244`); train.py
инициализирует модель несеяно (непроверенная воспроизводимость), глобальный RNG (шум i-gate,
R1 `torch.rand`, dropout) тоже не сеется явно.

**Eval (`train.py:690-750`, ipynb:1136-1227):** hold-out — последние `_hold_n=3` файла (при ≥8);
снапшот `snapshot_runtime_buffers()` → `logit_cache.cache.clear()` → на каждый файл
`reset_reasoning()`+`memory_bank.reset()`, state `est/ogs` свежие, но переносятся между окнами
внутри файла; `adaptive=False`; `@torch.no_grad`; `compute_loss(es)` → CE (eval ⇒ ce_raw).
Регион чтения: train.py `len//2`, ноутбук `len//4` (в комментарии M13 — «3/4-region»).
Покрыто снапшотом: все буферы (вкл. non-persistent), `_last_bus`, `_intent_stream`,
`_reasoning_buffer/count`, per-layer `_cached_hp/_cached_pred_k/_cached_pred_error_norm/
_cached_gate/_cached_usefulness`, `blk._traj_state` (M33/M45/B10).
**НЕ покрыто снапшотом и не сбрасывается**: `_mem_dir`, `_last_salience`, `_last_logits`,
`_last_read`, head plain-attrs (`_last_u/_last_p/_last_sat/_last_conflict/_last_srl/
_last_lacuna*`), `_branch_var_ref`, `_layer_diagnostics`, `_pred_loss_term`-класс (release,
не reset). В eval они не мутируются (train-only), но rollback/stale-сценарии не закрыты (з4/з5).

**M49 ранние eval'ы:** `step<3000 и step%250==0` — пишут `best.pt`/val, **не трогают**
контроллеры (depth/scheduler только на canonical `step%eval_interval==0`); оба цикла
(`train.py:624-641`, ipynb:1136-1194). `best_val_loss`-политика: улучшение → `best.pt`
(атомарно: tmp+move, `train.py:20-25`); train.py при Ctrl+C **не сохраняет** (`:682-685`),
ноутбук перезаписывает `best.pt` (`ipynb:1235-1242`). Состав чекпойнта: step, model, `code_fp`,
optimizer (+`param_names`), scheduler, best_val_loss, cfg, `reasoning_enabled_step`,
`active_depth`, `recover_count`, `detector`, `depth_state`, `balancer`, `stream_idx`, `offset`,
`rng`, `data_rng`, `stream_state`, `stream_gs`, `cuda_rng`. **M56b-политика**: seed init — только
ноутбук; `best.pt` хранит RNG-состояния, но не seed модели (при fresh-старте train.py
недетерминирован).

### 2.7 Замкнутый контур

**(а) Что обучение возвращает в зону 1.** Коды не обучаемы (индексация int64, non-persistent
буфер). Обучение меняет: `embed_mix` (λ⁻², lr 8.87e-5), `basis`/`readout` (тот же Parameter —
один градиент с энкодера и головы), `bit_bias`, `log_temp`, `emphasis_gain`, `token_bias`,
`phantom_basis/mix`, `lacuna_w/b`, `log_eta`. Т.е. **идентичность токена (книга кодов)
заморожена; адаптируется только метрика чтения/записи** (`basis`) и аффинные приоры. Контур
«CE → h → basis → z → CE» замкнут, но он не может переопределить сам код; при эрозии
`‖B_k‖` (F2A-09) обучение может лишь перемасштабировать `z` через `basis`/`log_temp`.

**(б) Замкнутые петли, которые реально работают:**

1. `h → CE → (балансер) → параметры ствола/головы` — основная петля (без BPTT между окнами:
   состояния detach'нуты `stack.py:695`).
2. salience→intent: `observe_output(logits)` → `_last_salience` → `intent_probe` → `_last_bus` →
   `bus_bias` → `zt` головы (1-шаговая задержка; bus_bias только в CE-пути, з5).
3. `_last_lacuna_rel` → broadening банка `temp·(1+0.5·rel_excess)` (train, 1-шаг, `step≥1045`).
4. `_mem_dir` → temper головы (train, 3D-вызовы; CE-путь не проходит shape-check).
5. `pen`/`hp` (1-шаг кэши зеркала) → VSA-decay/i_gate/UCL-write.
6. Внешний контур контроллеров: `val_loss` → DepthController/LRController; mirror-статистики →
   MirrorLRScheduler; `mat_gate` → банк/UCL/bridge.
7. `watchdog` — **разомкнут** (не вызывается).

**(в) Мёртвые каналы (замкнутые «в никуда»):** `pred_weight` (вычисляется `stack.py:301-302`,
нигде не потребляется), `pred_w`-aux (головы `pred_w` нет), `novelty_gate` банка (no_grad +
не читается, з4), `_mlp_ratio` (нет потребителя, з3), `cache_grad_norms` (только тест, з3),
M55-рост `confirmed_directions()` (нет потребителя, з5), `layer_gate` LBG (не влияет на поток,
з4), `projector_signals`/`collective_stats` (з4), `intent_state` в LiveInference (з4),
`hard_veto_ceiling`, `nonfinite_gradient_names`, `alarm_strikes`, `_full_env`/`_atomic42`,
`_phase_ratio_*`, `mirror_hi` (все — зона 6).

**(г) Опасные петли (с числами):**

- **2970-взрыв:** три branch-члена — log-отношения (scale-free): «поток дошёл до ~1e16, а
  'branch' остался конечным» (`losses.py:228-233`); M51-анкер (0.5) добавлен как абсолютный
  якорь; чекпойнт 2970 был однажды авто-возобновлён и «стоил целой сессии» (ipynb:466-468),
  поэтому rolling `latest.pt` удалён — единственный `best.pt`.
- **M47 (step 3135/3190):** маркеры `mlp_out 1400→424`, `diversity 0.6→0.08`, `mat=0.82`,
  `lbg_tau→0.50` — следствия таймера `pen_init=1.0` (з3), а крэш на 3190 вызван
  graph-pinned `_cache_*`/`_pred_loss_term` + retained-pass `ggeo` (8 backward'ов на лог);
  закрыто `release_step_graph` в ckpt-fallback + удаление ggeo (M48).
- **M53d:** SRL hard-refine `u ← u+0.7(logit(c)−u)` каждый forward → `ce 11→31→78→91`,
  `sat 0→0.5→1.0`, `srl_expl 0.72→27.1`, val→inf, `best.pt` остался на 250/11.1584; закрыто
  `head_srl_apply=False`.
- **A2:** `‖w_intent‖→62k`, loss 3.8e22, diversity 3.8e22; закрыто RMS-нормировкой входа гейта и
  `intent_alpha` (з3).
- **Run B LR-коллапс:** старый `report_val_loss` (ratchet ×0.5 к полу 0.05 без достижимого
  restore) — заменён best-anchored правилом.
- **UCL-запись:** градиенты ~9e-10 (з4) — канал может остаться «identity» очень долго.
- **NaN-zombie:** `watchdog.check` не вызывается; при NaN-лоссе AGC дропает все градиенты
  (`p.grad=None`), `optimizer.step()` не меняет веса, лог печатает NaN — петля может крутиться
  молча до ближайшего eval; единственная защита — `best.pt`.
- **Surprisal-разрыв:** train-CE (взвешенный) vs val-CE (сырой) ×2.0–2.3 — «фантомные 8 нат»
  (M46); сравнение метрик без `ce_raw` ложно.

### 2.8 Сверка контрактов: сквозной проброс «величина → откуда → куда → сохраняется/теряется»

| Величина | Откуда | Куда (зона 6/дальше) | Судьба |
|---|---|---|---|
| `tokens` (B,L) | з1/данные | TokenStream → embed/forward/CE | ok; клэмп ≥vocab невидим (B7 inert в ipynb) |
| `codes` (V,K) | з1 | logit_cache, SRL, fingerprint | non-persistent; fingerprint в ckpt (B8) |
| `basis`/`readout` | з1↔з5 | λ⁻² группа, lr 8.87e-5 | обучается; эрозия норм не логируется |
| `h₀`/DC/лакуна | з1/з2 | forward (не нормируется) | ok |
| `h` (out, RMS≈1) | з2/з4 | CE + aux-кэши | ok |
| `h_emb` | з1→з6 | `compute_losses(h_emb=h)` | **обрыв: не используется** |
| `pen`, `hp` | з3 (1-шаг кэши) | VSA-decay/i_gate/UCL | **в чекпойнт не идут** (stream_state несёт только `state`-tuple с вестигиальным `pen`); после resume — холодный первый forward |
| `_pen_ema` | з2 | decay-центрирование | **persistent buffer → ckpt ok** |
| `mem_all`, `mem_mod` | з2/з3 | внутри forward | не сохраняются (runtime) |
| `_mem_dir` | з4 (bank._last_read) | temper головы | **train-only; не ckpt, не снапшот, `reset_cache` не чистит**; CE-путь не темперируется |
| `bus_bias`/`_last_bus` | з4 | CE-путь головы | `_last_bus` — plain attr: снапшот да, ckpt нет → после resume intent-стенсил отсутствует |
| `_intent_stream` | з4 | зеркало (carry) | снапшот да; ckpt/resume нет; док-граница не сбрасывает |
| `mat_gate`/`readiness` | з3 | банк/UCL/bridge/τ | timer; в ckpt `gate` (буфер) + `step` |
| `tau_config` (τ_l, lr_mult, mem_tau) | з3/з6 | apply_tau_lr/AGC/bank | `_tau_dev` в ckpt; кэши пересчитываются |
| `gate_tau` | з3 | LBG `lbg_tau` | пересчитывается от `mat_gate` |
| `head_wall`/`phantom_l1` | з5 | aux → балансер | **в `_cached_losses` не попадают** → train.py не логирует; не в ckpt |
| `_last_u/_last_p` | з5 | wall/L1 | пины графа, `release_step_graph`; не ckpt/снапшот |
| `_last_srl/_last_sat/_last_conflict` | з5 | head_telemetry | train.py не печатает; не персистятся |
| `_last_lacuna_rel` | з5 | broadening банка | 1-шаг; не ckpt; eval использует stale |
| `ph_*` | з5 | head_telemetry | =0 by design; не персистятся |
| `_r1_steps` | з4 | M56c-контроль | persistent buffer → ckpt ok |
| `ell_ema`, `_srl_step`, `_pb_step` | з5 | пороги лакуны/SRL/фантомов | non-persistent → после resume lazy-init/сброс |
| `_noise_gen` (seed 1234) | з5 | шум η | не в ckpt → поток шума воспроизводится с начала |
| `_traj_state` | з2 | streaming carry | снапшот да; **`reset_cache` не чистит** (дефект); ckpt нет |
| mirror-кэши `_cached_*` | з3 | write_mod/UCL/read_mod/aux | снапшот (eval) да; **ckpt нет** → после resume холодный старт |
| `_branch_var_ref` | з6 | M51-анкер | plain attr; не ckpt/снапшот → пере-базируется на resume |
| aux-значения | з2–з6 | балансер | живут один шаг; в ckpt не идут |
| `stream_state`/`gs` | з2 | resume непрерывности | `_dstate/_tstate` → ckpt ok (B15) |
| RNG (`rng`, global, cuda) | з6 | воспроизводимость | в ckpt ok (кроме seed init train.py) |
| optimizer/scheduler/watchdog/depth/balancer state | з6 | resume | в ckpt; balancer-EMA=None (мёртвый режим) |
| `val_history.jsonl` | з6 | история val | только ноутбук; train.py не пишет |
| `code_fp` | з1/з6 | защита resume | в ckpt ok (B8) |

**Полный список обрывов проброса, найденных зоной 6:**

1. `h_emb` — «двухконечное чтение» мёртво (подтверждение з1).
2. `pred_weight` — вычисляется, не потребляется (з3 + з6).
3. `pred_w` aux — нет параметра у sigmoid-головы.
4. `head_wall`/`phantom_l1` — не в `_cached_losses` ⇒ нет в логах train.py.
5. `head_telemetry` (sat/srl/conflict/lacuna/ph_*) — train.py не зовёт; в чекпойнт не идут.
6. `_mem_dir` — train-only, не персистится, CE-путь не темперируется (з5).
7. `bus_bias` — отсутствует в 3D-вызовах головы (з5); `_last_bus` не в ckpt.
8. mirror 1-шаговые кэши (`_cached_hp/_cached_pred_k/_cached_pred_error_norm/_cached_gate/
   _cached_usefulness`) — не в `stream_state` → теряются при resume в новом процессе.
9. `_intent_stream` — не сбрасывается на док-границе и не в ckpt.
10. `_traj_state` — `reset_cache` не чистит (з2), в ckpt нет.
11. `_branch_var_ref` — не персистится (M51-анкер пере-базируется).
12. `_last_lacuna_rel` — не персистится; broadening выключен в eval и на первом шаге.
13. `ell_ema`/`_srl_step`/`_pb_step`/`_noise_gen` — non-persistent (з5).
14. `watchdog` — не вызывается (check/arm_ce) в обоих циклах.
15. `hard_veto_ceiling`, `nonfinite_gradient_names` — импортированы, не вызываются.
16. `novelty_gate` (з4), `_mlp_ratio` (з3), `cache_grad_norms` (з3), M55-рост (з5),
    `layer_gate` LBG (з4), `projector_signals`/`collective_stats` (з4) — мёртвые каналы.
17. `balancer.loss()`/balance-EMAs — не используются (align-путь всегда).
18. `_full_env`/`_atomic42`/`alarm_strikes`/`_phase_ratio_*`/`mirror_hi` — мёртвый код train.py.
19. CLI-флаги без эффекта: `--head`, `--amp-obj`, `--no-amp-pred`, `--save-interval`,
    `--scheduler`, `--stage-steps`, `--stage-mode`, `--readiness-full`, `--per-layer-ls-lr`
    (перетирается `per_layer_ls_lr=False`); нет флага `--optimizer`.
20. `token_bias` — без wd и без частотного приора (з5, вопрос 14) — дрейф редких токенов
    не закрыт ничем.
21. train.py: нет seed-init модели (M56b — только ноутбук).
22. Ноутбук: нет vocab-проверки TokenStream, нет mlp-depth-буста, нет `set_trust`.
23. train.py B19 (`:840-846`): при `gradient_checkpointing=True` (default) **делит `cfg.seq_len`
    на 2** на старте — не OOM-реакция, а безусловный сдвиг eval-окна; train-куррикулум всё
    равно берёт литералы 64/256/512 ⇒ train/eval окна расходятся (64 vs 128).
24. train.py: двойное применение `ls_mult` к `base_parameters` (дремлет при `per_layer_ls_lr=False`).
25. Legacy-CE-путь игнорирует `sw` в весах (в бою не используется).
26. `intent_tau`/`diversity` регулируют α в формах, отличных от v3-`intent_alpha` зеркала.
27. `_last_logits` (117 MB) — plain attr, не снапшотится; в eval не нужен (logits=None), но
    rollback-сценарий не чистит (з4).
28. `_pred_loss_term`-класс aux-кэшей: release, но не reset (в eval не пересчитывается; в
    `_cached_losses` после eval могут попасть eval-значения — косметика лога).

---

## 3. Что зона делает с входным контрактом (проверка/преобразование/потеря)

1. **CE**: маскирует PAD (`t!=0`), обучает EOS (`mask_eos=False`), взвешивает surprisal
   (coded-путь), отдаёт `ce_raw` отдельно; **train-CE не сравним с val-CE** без `ce_raw`.
2. **Aux**: собирает ~24 сырых терма, суммирует и ограничивает суммарный aux per-param
   (не per-term); cfg-веса в большинстве — только on/off; часть термов невидима в логе train.py.
3. **Градиенты**: CE- и aux-градиенты собираются `autograd.grad` (не `.backward()`); знак и
   норма ограничены; не-конечные дропаются AGC; затем τ-LLRD × ls_m, затем AGC.
4. **Оптимизатор**: AdamW или EVAAdamW (боевой); τ-cap, trust (не в бою), AdamP, cautious;
   новые параметры при resume получают свежий Adam; `mlp_gate_b`-реопен в train.py затирает
   обученные значения (B15-регрессия).
5. **Расписания**: warmup/blend `_alpha_override`/`_usefulness_temp`, mirror-adaptive LR,
   plateau-depth, τ-LLRD; `rewind` не используется.
6. **Данные**: memmap-стримы, окна 64–512 (train.py) / 224 (ipynb), док-границы сбрасывают
   не весь стейт (`_intent_stream` остаётся); seed 42 для данных, 1234 для init — только ноутбук.
7. **Eval**: изоляция снапшотом (полная по буферам и большинству кэшей), но `_mem_dir`/head-
   plain-attrs/`_branch_var_ref` не покрыты; регион чтения у циклов разный; ранние eval'ы
   пишут `best.pt`, не трогая контроллеры.
8. **Чекпойнт**: единый `best.pt` (улучшение val) + полный restart-state; в CLI нет
   interrupt-сейва; `code_fp`/B8-защита; компрессия FCF-CPR (`compression.py`) стрипает
   optimizer/scheduler и пересобирает коды legacy-билдером (латентный дефект з1).
9. **Потери**: telemetry головы и половина aux-ледера не доходят до логов/чекпойнта; pen/hp/
   mem_dir/bus/branch-ref теряются на resume; watchdog/эскалация не работают.

---

## 4. Передача следующему (финальному отчёту): выходной контракт зоны 6

**Что обучение отдаёт обратно (обновления/расписания):**

| # | Величина | Форма/шкала | Инварианты |
|---|---|---|---|
| 1 | Обновления всех trainable-параметров, кроме `_vsa_tau_log` | — | grad-конвейер: `balancer → (ls_m) → apply_tau_lr(τ-LLRD×ls_m) → AGC → optimizer` |
| 2 | `p.grad` после балансера | per-param | `‖Δ_aux‖ ≤ ‖g_CE‖` (sign-agreeing), bypass — под тем же правилом от собранного grad |
| 3 | LR по группам | 14–16 групп (cell-4, eva_proj) | `base·λ^p·llrd^depth` × mirror-mult × `_loss_lr_factor`; τ-LLRD — grad-скейл |
| 4 | `_alpha_override`/`_usefulness_temp` | per-layer буферы зеркала | warmup+blend; после — 0.0/0.5 навсегда |
| 5 | `active_depth` | int | плато val, шаг +4, интервал ≥ eval_interval |
| 6 | `mat_gate`/`readiness` | (n_layers,) | таймер `pen_init=1.0` (з3) |
| 7 | R1-каденс | 5% шагов | глобальный `torch.rand` (не пер-model) |
| 8 | `best.pt` | полный restart-state | атомарно; val-улучшение; `code_fp`; B8-verify |

**Что НЕ доходит до логов/чекпойнта:** `head_wall`, `phantom_l1`, `nuc`, `orth`, `w_m2v`,
`intent_tau`, `branch`, `gradalign`, `diversity`, `tau_dev_reg`, `bridge_conn`, `mem_tau_reg`
(в train.py); вся `head_telemetry` (в train.py); `_branch_var_ref`; mirror 1-шаг кэши;
`_mem_dir`; `_last_bus`/`_intent_stream`; `_last_lacuna*`; `_noise_gen`; `ell_ema`/`_srl_step`/
`_pb_step`; `_phase_ratio_*`; `watchdog`-baselines (не накапливаются); `balancer`-EMAs
(мёртвый режим); `val_history.jsonl` (только ноутбук); seed init (train.py).

---

## 5. Связи с другими зонами

- **Зона 1:** codes заморожены; обучение меняет `embed_mix`/`basis`/`bit_bias`/`token_bias`;
  B8-fingerprint; `_sig_mean` (non-persistent) при resume переучивается ~1000 шагов (вопрос з1.7)
  — флаг `embed_center=False` в бою, дремлет.
- **Зона 2:** M50/M51 caps и `_stream_cap` определяют шкалу, к которой применяются aux;
  `_cache_conv/bind/mirror_out` — вход M51-анкера; `_pen_ema` — persistent; `_traj_state` не
  чистится `reset_cache` (з2) — flush каждые 495 шагов и R6-resume оставляют carry.
- **Зона 3:** `pen`/`hp`/`readiness` — расписание зоны 6; aux `pred/gate_l1/reinforce/balance/
  diversity/decorr/signal_ent/div/gate_repulse/alpha_novelty/ls_reg`; `_pred_loss_term` — live-пин;
  `_gradalign_tgt` — CE-фаза; `_usefulness_temp` пиннится на 0.5; `_mlp_ratio` мёртв.
- **Зона 4:** `bridge_conn` (InfoNCE), `mem_tau_reg`, `lbg_diversity`; `_mem_dir`/`_last_lacuna_rel`/
  `bus_bias` — training-only контуры; UCL-write ~9e-10; `novelty_gate` мёртв; `reset_cache`
  неполон — flush/resume не чистят банк/UCL/bridge_stream (з4) ⇒ возможна кросс-документная утечка
  памяти через flush-границы (в бою банк сбрасывается на док-границе, UCL — нет by design).
- **Зона 5:** CE-сборка, surprisal, wall/L1, `_last_u/_last_p`-пины, `release_step_graph`;
  telemetry не персистится; `ph_*=0`; M55-рост отсутствует.

---

## 6. Открытые вопросы и риски

1. **FailureDetector полностью инертен.** `check()`/`arm_ce()` не вызываются ни в CLI, ни в
   ноутбуке; `hard_veto_ceiling`/`nonfinite_gradient_names` тоже. При NaN-лоссе AGC дропает
   градиенты, шаг «пустой», лог печатает NaN — петля может идти до eval. Решение (код не
   менялся): либо вернуть вызов `watchdog.check(ce, step, metrics)` + `arm_ce()` на первом eval,
   либо удалить мёртвый контур, чтобы не создавать ложного ощущения защиты.
2. **B19 в train.py (`:840-846`) — не OOM-реакция, а порча окна**: при `gradient_checkpointing=True`
   (default) `cfg.seq_len //= 2` на старте; train-куррикулум берёт литералы ⇒ eval-окно ≠ train-окно.
3. **B15-регрессия в train.py**: `mlp_gate_b_init` (0.25) затирает обученные `mlp_gate_b` и
   `hybrid_gate.log_tau` на каждом resume (ноутбук чинит гардом).
4. **Несовпадение контуров train.py/ipynb**: bank off, AdamW vs eva_proj, surprisal 0 vs 0.3,
   branch-anchor 0 vs 0.1, gradalign 0 vs 0.3, mlp-depth boost on/off, seed init off/on,
   vocab-check off/on, interrupt-save off/on, регион eval `len//2` vs `len//4`. Любые выводы
   «по логам» должны указывать, какой контур их породил.
5. **`_branch_var_ref` не персистится** — M51-анкер после resume пере-базируется на текущий
   (возможно, уже сдвинутый) масштаб; при резюме из «плохого» состояния анкер закрепит его.
6. **τ-авторитет размножен**: три формы α (`v3=1−1/τ` в зеркале/шине, `1−e^{−τ/τ_min}` в
   diversity, `1−τ_mid/(√D·τ_l)` в intent_tau) — регуляризаторы оптимизируют не то, что
   потребляет модель; при этом `tau_dev_reg` (к 0) и `intent_tau` (к сигмоиде) тянут `_tau_dev`
   в разные стороны.
7. **`token_bias` без wd и без приора частот** — дрейф редких токенов не закрыт (з5.14).
8. **Surprisal-разрыв train/val** (×2.0–2.3) — любой анализ «train CE vs val CE» без `ce_raw`
   ложен (M46); в train.py surprisal выключен, в ноутбуке включён ⇒ логи двух контуров
   несравнимы ещё и по этому признаку.
9. **Eval-гигиена**: `_mem_dir`, head plain-attrs, `_branch_var_ref` не покрыты снапшотом;
   rollback-сценарий (`reset_cache` неполон, з2/з4) оставляет stale-память/концепты; `_intent_stream`
   не сбрасывается на док-границе.
10. **Resume-холодный старт зеркала**: `_cached_hp/pen/_cached_gate` теряются (не в ckpt) —
    первый forward после resume идёт без write_mod/VSA-boost/UCL-весов; в сочетании с
    `_usefulness_temp=0.5` и `mat_gate`-таймером это может давать транзиент ~1 eval.
11. **Мёртвый aux-ледер**: `head_wall`/`phantom_l1`/`branch`/`gradalign` не видны в CLI-логе,
    а `nuc/orth/w_m2v/intent_tau` — вообще нигде (кроме ipynb merged). Диагностика wall/L1/M51
    в бою существует только в ноутбуке.
12. **Компрессия чекпойнтов** (`compression.py`): стрипает optimizer/scheduler (осознанно,
   inference-only), но пересобирает коды legacy-билдером (`:232`) — при persistent-кодах станет
   тихой подменой книги (латентно, з1 F2A-12).
13. **AMP**: train.py-AMP обёрнут только вокруг embed, ipynb-AMP — вокруг всего forward+loss;
   при включении AMP в ноутбуке `scaler=None` (bf16) и grad-конвейер не меняется, но telemetry-
   пороги и `conv_state`-landmine (з2) остаются.
14. **`_last_logits` (117 MB) и `observe_output(lm_head(out))`**: второй граф головы строится
    и выбрасывается каждый шаг (лишний forward+память); при OOM-ладдере это может быть
    триггером, хотя R1-контракт от этого не страдает.
15. **`_vsa_log_param` обучается** (з2.7): роль scale_inv, lr=base, в группе с `embed_mix` —
    реальный контроль лестницы τ через `_base_vsa`; в аудитах не измерялся его дрейф.
