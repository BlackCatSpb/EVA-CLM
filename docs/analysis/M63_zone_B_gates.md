# M63 / агент B — гейты и контроллеры

> Цепочка M63 (2026-09-17). Метод: чтение кода (продакшн-конфиг не инстанцировался, чекпойнт не грузился — контракт CHAIN.md:17-23); телеметрия из брифа/журнала.
> Классификация: «таймер» = выход функция только step; «измерение» = меняется при изменении измеряемой величины при фиксированном step; «инертен» = константа/не подключён.

## 0. Метод, границы, оговорки

- Источник истины — реальный код (Read/Grep); состояние `best.pt` проверено только листингом ключей zip-архива.
- Телеметрия прогона: `val 8.4877@7315` (журнал) — в брифе указано `8.3915@7315`; расхождение не разрешено (журнал описывает момент сохранения чекпойнта, бриф — резюм-прогон).
- Все выводы об инертности LBG/spectrum опираются на чтение кода; где нужен живой лог (tempering `chi`, wall `sat`), помечено.

---

## 1. Полная инвентаризация

| № | Гейт/контроллер | Вход | Формула (file:line) | Класс | Доказательство |
|---|---|---|---|---|---|
| 1 | `maturation.step_gate` | `step`, `tau_norm` | `σ((ln t − ln T_eff)·T_eff/Δ)`, `T_eff=T0+α(1−τ_norm)T_delay` — `core/maturation.py:151-153` | **ТАЙМЕР** | чистая функция `step`; `_tn` из `tau_config.tau_norm_live()` (maturation.py:138-139) |
| 2 | `maturation.update/readiness` | `pred_err_l` | `sat=1−pen_ema/pen_init`; `readiness=σ((sat−r0)/rs)` — `core/maturation.py:176-183` | **ИЗМЕРЕНИЕ** | M58c: `pen_init` стартует с 0 и сеется первым наблюдением (maturation.py:83-89) |
| 3 | Итоговый `mat_gate` | ramp + readiness | `max(ramp, readiness)` — `core/stack.py:478-479` | гибрид | mat=0.637 [0.26,0.92]: deep-слой рампа на 7315 даёт `σ(ln(7315/8000)·2)=0.455`, shallow `σ(−3.13)=0.042` → измерение доминирует |
| 4 | `DepthController.update` | `val_loss` | `slope > −k·σ` → `+inc` — `core/adaptation.py:150-166` | **ИЗМЕРЕНИЕ** | вызывается только на каноничном eval; `depth.update(step)` каждый шаг — no-op (adaptation.py:142-143) |
| 5 | `lbg` pre-ready | `mat_gate` | `return maturation` — `core/layer_bridge_gate.py:124-125` | **ТАЙМЕР** (через mat) | `global_ready` требует `gate>0.1` (maturation.py:159-161) |
| 6 | `lbg` ready (SpectrumGate) | health(6) × tau | `σ(logits)·(1+softmax(logits/τ))` — `core/adaptive_gate.py:63-69`; сведение — `layer_bridge_gate.py:129-132` | **ИНЕРТЕН** | stack считает `_gate_i` (stack.py:591-595), но probe читает сырой `h.detach()` (stack.py:603); единственный потребитель — `lbg_diversity` (losses.py:432-437,568-569) на **detached-clone** диагностик (losses.py:403) |
| 7 | `lbg_tau` | `mat_gate` | `exp(log τ_max+(log τ_min−log τ_max)·mat)` — `layer_bridge_gate.py:110-111`; `tau_config.py:160-168` | **ТАЙМЕР** | lbg_tau=0.97 = `gate_tau(mat)`; lbg_diversity=0.0000 → aux-ключ не эмитится (`!= 0`, losses.py:568) → `log_tau` без градиента |
| 8 | UCL `read_scale` | обучение + floor | `σ(read_scale)`, `clamp(min=floor)` — `core/concept_layer.py:373-378`; floor по `step<5000` — `stack.py:401-403` | **ТАЙМЕР** (floor) + обучаемый | read_scale=0.0179 после снятия floor=0.1 = само-закрытие (уточнено агентом C: clamp-freeze, градиент 0) |
| 9 | UCL `_mature` | `resvar` зеркала | `cv=√var/|ema|`; `σ((1/cv−λ)·τ_mat)` — `core/concept_layer.py:123-148` | **ИЗМЕРЕНИЕ** | порядок `M6`: λ из `lambda_utils`, не размерность d (concept_layer.py:134-137) |
| 10 | UCL write hard-skip | `mat_gate[0]` | `if mat_gate < 0.1: return` — `concept_layer.py:178` | **ТАЙМЕР** | + собственный `mat>=0.3` (concept_layer.py:205) |
| 11 | UCL births | `best_sim`, `conf`, `tau_norm` | `gap=σ(_log_tau_novelty_thr)`; `novel=((1−best_sim)>gap)&(conf≥birth_thresh)` — `concept_layer.py:243-248,250-275` | **ИЗМЕРЕНИЕ (вырождено)** | births 307–353 статичны, 8/8 слотов занято → `empty` пуст, рождение идёт через `utility.argmin` (concept_layer.py:256-258), но `novel` не срабатывает: gap≈0.5 |
| 12 | phantom `decay` | — | `confidence *= 0.999` каждый forward — `core/phantom.py:53-56` | **ТАЙМЕР** | не зависит от данных; τ_распада≈1000 forward-ов |
| 13 | phantom `observe/confirm` | `ell_rel`, cos | `sel = L>1.1`; merge cos≥0.7; `+0.05`; confirm 0.75/archive 0.25 — `phantom.py:29-33,59-101` | **ИЗМЕРЕНИЕ** | alive 16, confirmed 0–2: за 25 шагов decay=0.975, а +0.05 при повторном совпадении — едва удерживает |
| 14 | SRL | `u` (log-odds) | annealed EM: `p=softmax(u·(2C−1)/τ_t)`, `τ_t=τ0·γ^t` — `core/embedding.py:409-456` | **ИНЕРТЕН** | `srl_apply=False`; srl_conf=1.0/srl_ent=0: `u` насыщен, τ после 2-3 шагов (0.36-0.216) → softmax one-hot |
| 15 | tempering | `cos(implied_h, mem_dir)` | `χ=relu(0.3−cos)`; `logits/=1+0.5χ` — `core/embedding.py:589-597` | **ИЗМЕРЕНИЕ (условно)** | shape-fix M58b; `_mem_dir` есть только после записи банка (stack.py:619-621); при cos≥0.3 χ=0 → identity |
| 16 | bus cap | `bus_bias` | `bus_bias·(3/amax.clamp_min(3))` — `embedding.py:374-379` | **СТАТ. ПРЕДОХРАНИТЕЛЬ** | порог фиксирован 3.0; инертен, пока не связывает |
| 17 | `emphasis_gain` | `relative` softmax | `u = zt + gain·(log1p(r)−log1p(1/K))` — `adaptive_gate.py:87`; параметр — `embedding.py:288,402-403` | обучаемый | gain=−1.25: модель инвертирует конкуренцию; центрирование даёт ровно 0 при равномерном softmax |
| 18 | wall | `|u|` | `1e-3·relu(|u|−6)².mean()` — `core/losses.py:506-513` | **ИЗМЕРЕНИЕ**, порог фикс | эмитится только при `>0`; при `sat=0` ключ отсутствует → инертен в этом прогоне |
| 19 | LossBalancer | градиенты CE/aux | per-coordinate sign-mask + `‖aux·s‖≤‖g_CE‖` per-param — `core/training_control.py:425-436`; bypass gradalign — 437-448 | **ИЗМЕРЕНИЕ** | `align_cap` не используется (207); агрегатный cos только логируется (422) |
| 20 | ReasoningGate | `h, know, r` | `tanh(proj(h)+know_proj(know)+r_proj(r))` — `core/reasoning.py:137-158` | **ИНЕРТЕН** | `proj.weight/know_proj/r_proj` zero-init, bias[0]=+10 → gates i>0 = 0; stop_thr=0.5 → цикл исполняет 1 шаг |
| 21 | seq curriculum | `step` | `step<32000→64; <96000→256; else 512` — `scripts/train.py:476-485` | **ТАЙМЕР** | в ноутбуке отсутствует, seq фикс 224 — живой прогон шёл без curriculum |
| 22 | OOM-лестница | событие OOM | ckpt→batch/2→seq/2, regrow через 200, ckpt-probe через 500 | **событийный ТАЙМЕР** | нет измерения свободной VRAM до OOM |
| 23 | `lr_boost`/`ls_mult` | EMA-отношения | `mirror_mult=(var·alpha·gate)^{1/3}·mag_factor`, boost только при `_val_improving` — `core/lr_scheduler.py:204-242`; `ls_mult` fast/slow var(log_scale) — 75-113 | **ИЗМЕРЕНИЕ** | self-referenced EMAs; применяется `apply_tau_lr` |
| 24 | AdaptiveController per-layer | `|mirror|`, divergence | `expl=min(1,|mirror|/thr)`; `diff=div/run_rec` — `core/adaptive_controller.py:60-75` | **ИЗМЕРЕНИЕ** | `var(log_scale)` выведен из контура |

### Пояснения

- **maturation — единственный гейт с реальным приводом**: `maturity` масштабирует инъекцию bridge (`core/bridge.py:162-163`), запись private-mem (`core/mirror.py:664-666`) и live-модуляцию (`mirror.py:824-827`). Но 2/3 его аргумента — часы. Дефект: `update()` вызывается только если `len(pred_errs)==n_layers` (stack.py:744-745), иначе молча пропускается.
- **`max(ramp, readiness)` — односторонний**: гейт может только открываться (stack.py:478-479); закрытие по измерению невозможно by construction.
- **LBG — «привод без нагрузки»**: поток берёт `probe_layer(h.detach())` (stack.py:603); единственный градиент — `lbg_diversity` через detached-clone (losses.py:403); при =0 ключ не эмитится (losses.py:568). Плюс `losses.py` использует инлайновую копию диагностик с `diag[3]=0.5` и ненормированной энтропией (stack.py:699) — рассинхрон с `layer_diagnostics` (layer_bridge_gate.py:134-168).
- **UCL births**: статичность 307–353 — корректное поведение измерения на однородном корпусе (новизна не превышает gap). Неопределённость: без гистограмм `1−best_sim` нельзя отличить «нет новизны» от «порог завышен».
- **phantom**: `decay` — чистый таймер, делает банк наблюдателем: между наблюдениями (25 forward) теряется 2.5% уверенности, `confirm=0.75` достижим только при почти непрерывных повторных наблюдениях. `decay` не знает, наблюдался ли слот (уточнено агентом C: подтверждение арифметически невозможно).
- **tempering**: `χ` зависит от `cos` — при типично высоком согласии `χ=0`. Неопределённость: решается логом `head_telemetry()['conflict']` (stack.py:1152-1154).
- **reasoning**: `reasoning_scale` — таймер-рампа, но к 7316 уже 0.999; инертность от нулевого `r_proj` (reasoning.py:130-132) и порога остановки.
- **OOM-лестница** меняет seq/batch по событию, а не по измерению headroom; regrow через 200 — источник пилообразности.
- **соседние каналы**: `usef=0.502` — выход `usefulness_predictor(delta)` с порогом-медианой (mirror.py:781-808), статичность = коллапс `delta`/предиктора (уточнено агентом E: медианное центрирование даёт mean≡0.5 структурно). `nuc=0` — при `nuclear_weight=1e-5` (config.py:314). `cache_gate` — zero-init + bias −10 (logit_cache.py:348-360,449-450).

---

## 2. Шина доказательств

### 2.1 Сигналы

| Сигнал | Формула | Стоимость | Шум/оценка |
|---|---|---|---|
| S1 per-layer pred-error ε_l | существующий `pred_error=(hp−α·hp_prev)/‖hp‖·pred_scale_mod` (mirror.py:474,488), среднее — `_cached_pred_error_norm` (mirror.py:560; stack.py:675-677) | уже посчитан; +O(n) | EMA(0.999) ≈ 1000 шагов |
| S2 grad-SNR γ_l | `μ_l/σ_l`, EMA(0.99) от `‖g_l‖` (один стекованный `.norm()` на 24 слоя = 1 sync) | O(n) + 1 device-sync | σ стабильна после ~30 шагов |
| S3 effective rank r_l | `r_l=exp(H(σ_i/Σσ_i))` или `#{σ_i>0.1·σ_max}` по сингулярным числам carry/private-mem | SVD дорого; round-robin 1 слой/шаг, K=50 → O(1) амортиз. | окно ~10 замеров |
| S4 relative novelty ν_l | `ε_l/EMA(ε_l)` — тот же принцип, что `ell_rel` головы и `sat` maturation | O(1) | EMA(0.99) ~100 шагов |
| S5 var(log_scale)_l | `ls.var()` — уже считается (lr_scheduler.py:87-89) | 0 | малый |
| S6 mirror consistency κ_l | `cos(mirror_out, h)` или `cos(bridge_stream, h)` | O(D) на слой | устойчив per-batch |

Инварианты: все сигналы детерминированы в eval (M44-доктрина), считаются в обоих режимах, используют только существующие буферы; читаются с лагом 1 шаг (иначе гейт влияет на собственный вход).

### 2.2 Математика отображения

1. **Почему монотонные.** Гейт в контуре с обратной связью: `z_{t+1}=f(θ; g_t)`. Немонотонность даёт релейные переключения (предельные циклы). Монотонность + ограниченность (σ) — неотрицательная петлевая чувствительность. Для открывающих гейтов: `g_t = max(g_{t-1}, G(z_t))` — иначе измерение может закрыть уже работающий канал.
2. **Калибровка без магических чисел:** (а) квантильная нормировка (PIT): `z = F̂_l(s)` по своей истории (резервуар 512 значений/слой) — выход равномерен, «порог» = «верхние q% своей истории»; (б) EMA-z-score: `z=(s−EMA(s))/(√(EMA(s²)−EMA(s)²)+ε)`, `g=σ(z/κ)`.
3. **Квантили предпочтительнее для открывающих гейтов**: распределение ε_l нестационарно (старт → сдвиг), абсолютный порог `r0=0.3` (maturation.py:95) — магическое число, зависящее от масштаба pen.
4. **Что заменяет step**: `g = max(quantile_gate(z), deadline_gate(t))`, где deadline включается поздно (например `t > 2(T0+T_delay)`) — страховка от deadlock (maturation.py:17-23).

### 2.3 Learned monotone vs квантильные пороги

| Критерий | Learned `σ(w·z+b), w>0` | Квантильные пороги |
|---|---|---|
| Адаптация к задаче | да, если есть путь до CE | нет (контур сам меняет распределение) |
| Устойчивость | риск коллапса/взрыва w; trust region | нет параметров |
| Нестационарность | нужна нормировка входа | самокалибруется |
| Проверяемость | монотонность по построению, обучение невидимо | порог = «верхние q%» |
| Прецедент | SpectrumGate, emphasis_gain | maturation `r0/rs`, `ls_mult_min/max` |

Рекомендация: **квантильные пороги для гейтов безопасности** (maturation, depth, UCL floor, phantom decay) и **learned monotone с положительным весом** только в CE-пути. LBG — контрпример: без CE-пути learned-гейт умер.

---

## 3. Протокол фальсификации

| Тест | Объект | Воздействие | Метрика | Порог |
|---|---|---|---|---|
| T1 «сигнал двигает выход» | каждый гейт | `z → shuffle(z)`, `z → 0` | `Δ=max|G(z)−G(perm)|` | `Δ > 10⁻³` и `Δ > 10·fp_noise` |
| T2 «не функция step» | каждый гейт | фикс `z`, `step ∈ {0,1k,10k,50k,150k}` | `std_step(G)/mean(G)`, `ρ(G,step | z)` | `std/mean < 10⁻⁶` ⇒ таймер; `|ρ|<0.1` для измерения |
| T3 «причинность сигнала» | mat/ucl/phantom/depth | подмена `z_t` на `z_{t-500}` | `max|G(z_t)−G(z_old)|` | `>10⁻³` |
| T4 «инвариантность» | все | `h→c·h (c>0)` | `max|G−G'|` | `=0` для scale-invariant |
| T5 «живой градиент» | обучаемые | 100 шагов на toy-лоссе | `‖∂L/∂θ_gate‖` | `>10⁻⁸`; `=0` ⇒ мёртвый канал |
| T6 «мертвый канал» | все | `state_dict` за 1000 шагов | `max|θ_t−θ_{t-1000}|` | `=0` ⇒ удалить/пометить diagnostic |
| T7 «односторонность» | mat/depth/UCL | монотонность открытия | `min_t (g_t − g_{t−1})` | `≥0` для открывающих |

Эскиз:

```python
def test_gate_is_a_measurement(gate_fn, signal):
    z = torch.randn(24, 6); z2 = z.clone(); z2[:] = z2[torch.randperm(24)]
    g1, g2 = gate_fn(z, step=5000), gate_fn(z2, step=5000)
    assert (g1 - g2).abs().max() > 1e-3
    g_step = torch.stack([gate_fn(z, step=s) for s in (0, 1000, 10000, 100000)])
    assert g_step.std(0).max() / g_step.mean().clamp_min(1e-6) > 1e-6  # иначе таймер
```

**Мета-тест T8**: занулить сигнал на живом чекпойнте (только чтение, мини-модель): если после 100 шагов гейт тот же — канал декоративен. Для влияющих на поток — counterfactual replay.

---

## 4. Миграция и риски

### Порядок замены

1. **Только инструментация (0 правок поведения)**: логировать S1–S6 рядом с гейтами. Ноль риска, даёт распределения для квантилей. 500–1000 шагов.
2. **Триаж мёртвых**: LBG — либо подключить как привод (заменить `probe_layer(h)` на `probe_layer(gate_i·h)`), либо удалить; SRL — убрать annealing до 0.216 или floor τ; reasoning gate — ненулевой init `r_proj` (ε=1e-3) или удалить адаптивный цикл.
3. **phantom decay**: `conf *= decay(age_since_last_obs)` — измерение вместо часов. Конфиг-совместимо.
4. **UCL**: floor по step → floor по квантилю novelty/utility; `mat_gate<0.1` hard-skip → `_mature` с калиброванным порогом.
5. **maturation**: `max(quantile(readiness), deadline(t))`. Самый рискованный шаг — последним, с rate-limit.
6. **depth**: slope → grad-SNR + per-layer CE-прирост; rate-limit «+4 не чаще eval_interval».
7. **OOM-лестница/seq**: измерять `torch.cuda.mem_get_info()` и γ_l.

### Ранний этап

- До `matur_warm=300` — timer-floor; до 100 eval-ов квантили неполны → blend `g=(1−w(t))·ramp + w(t)·quantile`, `w(t)=min(1,n_obs/100)`.
- Safety-deadline: при мёртвых сигналах гейт открывается к `2(T0+T_delay)=32k`.
- Rate-limit на открытие (+Δg за eval_interval).

### Совместимость с чекпойнтами

- В `best.pt` ключи: `maturation.pen_init/pen_ema/gate/readiness/tau_norm`, `layer_bridge_gate.gates.N.log_tau`, `concept_layer.read_scale/_mature/_resvar_*`, `phantom_bank.*`. Замена должна сохранять имена; новые буферы — `persistent=False` или мягкая инициализация; identity-ключи блокируются (training_control.py:155-173).
- Удаление LBG: unexpected keys (не фатально, train.py:381), но `lbg_tau` пропадёт из логов — не сломать analyze.py.
- `DepthController.get_state/put_state` (adaptation.py:120-137) — сериализация полная (B14); замена метрики плато требует новой версии state с fallback.

### Риски

| Риск | Механизм | Митигация |
|---|---|---|
| Открытие слишком рано | readiness шумит на малых n | квантиль + rate-limit + min_dwell |
| Открытие слишком поздно | сигнал замер после спада | safety-deadline + односторонность |
| Гейт «взламывает» сигнал | гейт влияет на ε/γ | лаг 1 шаг + detach + EMA |
| Дорогие измерения | SVD/CE-пробы | round-robin, только существующие буферы |
| Регрессия eval/train | сигналы только в train | M44-доктрина |
| Дрейф чекпойнтов | новые ключи/формы | persistent=False, strict=False, fallback |

---

## 5. Эскиз

```python
# core/evidence_bus.py — только чтение существующих сигналов, никаких новых графов
class EvidenceBus(nn.Module):
    S = 6  # eps, grad_snr, eff_rank, novelty, var_log_scale, consistency
    def __init__(self, n_layers, q=512):
        self.register_buffer('ema',  torch.zeros(n_layers, self.S), persistent=False)
        self.register_buffer('emsq', torch.zeros(n_layers, self.S), persistent=False)
        self.register_buffer('qbuf', torch.zeros(n_layers, q), persistent=False)
    @torch.no_grad()
    def update(self, sig):                       # sig: (n_layers, S), лаг 1 шаг
        self.ema.mul_(0.999).add_(sig, alpha=0.001)
        self.emsq.mul_(0.999).add_(sig*sig, alpha=0.001)
    def z(self, sig):
        mu, var = self.ema, (self.emsq - self.ema**2).clamp_min(1e-12)
        return (sig - mu) / var.sqrt()
    def quantile(self, sig, i):
        return (self.qbuf[:, :] <= sig[:, i:i+1]).float().mean(-1)
```

Гейты: `g = σ(z/κ)` для обучаемых (κ=1), `g = max(q_hist, deadline(t))` для открывающих. `persistent=False` для колец, `persistent=True` только для `ema/emsq` (иначе холодный resume — та же болезнь, что `_branch_var_ref`).

---

## 6. Вердикты

| Пункт | Вердикт | Обоснование |
|---|---|---|
| maturation ramp→измерение | **переформулировать** | readiness уже есть; ramp оставить дедлайном, не `max` |
| depth | **делать (уточнить)** | уже измерение; slope → grad-SNR+CE-прирост, rate-limit |
| LBG/SpectrumGate | **не делать как гейт** | привод не подключён (stack.py:603), `log_tau` без градиента |
| UCL read_scale | **делать** | обучаемый само-закрывающийся сигнал; floor → квантиль (с поправкой агента C: clamp-freeze) |
| UCL `_mature`/births | **переформулировать** | `_mature` — измерение, но hard-skip mat_gate — таймер; births диагностировать гистограммой |
| phantom decay | **делать** | decay по возрасту наблюдения (с поправкой агента C: сначала эвикция) |
| SRL | **не делать (apply)** | вырожден (conf=1/ent=0); оставить диагностику |
| tempering | **переформулировать** | сначала лог `conflict`; при χ≡0 — убрать или калибровать cos по квантилю |
| bus cap / wall | **делать (низкий приоритет)** | пороги — константы; квантили, сохранив предохранитель |
| emphasis_gain | **делать** | обучаемый, измеренный −1.25; логировать и ограничить (агент A: канал мёртв — per-bit или удалить) |
| LossBalancer | **делать** | уже измерение; вернуть в лог `last_cos` (training_control.py:422) |
| reasoning gate | **переформулировать** | нулевой init = мёртвый; оживить или удалить |
| seq curriculum | **не делать** | в живом контуре отсутствует |
| lr_boost/ls_mult | **делать** | уже измерения; добавить T2-тест |
| OOM-лестница | **переформулировать** | измерять VRAM headroom до OOM |

**Итог.** Из 24 позиций: 7 — измерения, 6 — таймеры, 5 — инертны/вырождены (LBG, SRL, reasoning, wall при sat=0, lbg_tau), 3 — обучаемые/предохранители, 3 — вне контура. Шина S1–S6 строится на существующих величинах, без новых параметров, совместима с `best.pt` при сохранении имён.
