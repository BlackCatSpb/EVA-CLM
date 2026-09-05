
## Резюме: Fresh Start с P0 fixes — шаг 3685 (2026-09-03)

### Состояние
- **Step:** 3685 (Colab запущен, fresh start)
- **Val:** 3.07e4@3495 — **ТЕКУЩИЙ BEST**
- **CE:** 9.96@3685 — **впервые под 10!**
- **Maturation:** 0.134 [0.048, 0.254]
- **bridge_conn:** ~0.21 (стабильный)
- **Memory bank:** L1=0 L2=0 L3=0 (ждёт mat≥0.3)
- **NaN:** 0, Inf: 0

### Траектория val (fresh start)

| Step | Val | Delta |
|------|-----|-------|
| 233 | 9.53e6 | — |
| 466 | 6.02e4 | — |
| 699 | 5.70e4 | -5% |
| 932 | 5.42e4 | -5% |
| 1165 | 5.10e4 | -6% |
| 1398 | 4.84e4 | -5% |
| 1631 | 4.62e4 | -5% |
| 1864 | 4.38e4 | -5% |
| 2097 | 4.20e4 | -4% |
| 2330 | 3.98e4 | -5% |
| 2563 | 3.77e4 | -5% |
| 2796 | 3.58e4 | -5% |
| 3029 | 3.40e4 | -5% |
| 3262 | 3.23e4 | -5% |
| 3495 | 3.07e4 | -5% |

### Ожидаемые рубежи

| Рубеж | Ожидаемый step | Значение |
|-------|---------------|----------|
| mat≥0.3 (memory bank) | ~6000-7000 | L1/L2/L3 начнут записывать |
| mat≥0.5 (полная зрелость) | ~12000-14000 | Все компоненты активны |
| val<1000 | ~10000-12000 | Модель начинает предсказывать |

### Ключевые коммиты
- `7f40202` — P0 Audit Fixes (gate grad, tau_config, tau_dev_reg, dead llrd)
- `136472c` — Notebook param routing (double LLRD fix, _tau_l_dev routing)
- `f712145` — CopyBackwards fix (.data.copy_() + .detach() на register_buffer)

### Наблюдения
- **CE впервые под 10!** — 9.96@3685
- val_ppl снижается на 5% каждые 230 шагов — стабильный темп
- Maturation 0.134 → deep слои mature ~0.25, shallow ~0.05
- bridge_conn стабилизировался ~0.21 — нет коллапса
- Memory bank L1/L2/L3 = 0 — ждёт mat≥0.3
- **Diversity** выросла до 149@3685 (было 60-80 ранее)
- **intent_w** подскочил до 1.66@3465 (было 1.54)
- L0 anomalies — predMSE=8.8 vs median=2.5, ||hp||=6.3 vs median=2.2
- Bridge стабилен: bus_head_proj weight_norm=0.318, cross-layer cosine=0.035

### Checkpoint best 6 (шаг 3495, val=3.07e4)
- **L22 pm_norm=0.149** —显著 вырос (было 0.113@3262)
- **L23 pm_norm=0.114** —显著 вырос (было 0.099@3262)
- **token_bias.mean=-0.88** — модель запоминает токены активнее (было -0.82)
- **bus_head_proj=0.343** — стенсил обучается (было 0.318)
- **CE(random)=44.3** — вырос (было 28.9), модель лучше различает случайные входы
- **entropy(pos)=8.3** —显著 вырос (было 6.5), модель лучше различает позиции
- **Diversity reconciliation** — cos_sim=-1.00, scale=0, diversity пока не помогает

### Новые данные: шаг 3300→4015
- **CE=9.35@4015** — **НОВЫЙ ALL-TIME LOW** (было 9.96@3685)
- **Val:** 2.84e4@3961 (↓2.7% от 3.07e4@3495)
- **DepthController активировался@3961** — `val plateau -> active_depth=12/24`
- **bridge_conn** резко упал: 0.214→0.115@3960 → 0.123@4015 (transient)
- **intent_w** вырос: 1.537→1.722@4015
- **mod_mlp** вырос: 0.508→0.519 — modulation gate открывается
- **Memory bank** — всё ещё dormant (mat<0.3)

### Checkpoint best 8 (шаг 5359, val_loss=9.95)
- **Maturation=0.187** —显著 вырос (было 0.129@3495)
- **L23 pm_norm=0.351** — deep-first maturation СРАБОТАЛА! (было 0.114@3495)
- **L17 pm_norm=0.193**, L21=0.176, L22=0.166 — глубокие слои резко выросли
- **CE(random)=12.5** — модель различает random vs real (было 44.3@3495)
- **token_bias mean=-1.38** — сильная мемоизация токенов (было -0.88)
- **Bridge L12 intent_w=0.061** — intent stream проникает в глубокие слои (было 0.000)
- **Memory bank L2 vals** —首次非零: mean=-11.37, std=633.9
- **Gate variance (L18-L23)** вырос ×3: 0.04→0.13-0.15
- **Diversity cos_sim** нормализовался: -1.0→0.0
- **L0 anomaly** — ||hp||=5.8 (×2.4 median), predMSE=7.9

### Что делать дальше
1. Продолжить мониторинг — maturation 0.187, ждём mat≥0.3
2. При mat≥0.3: проверить что memory bank L1/L2/L3 начали записывать
3. При mat≥0.5: оценить качество генерации текста
4. После 10K шагов: проанализировать auxiliary losses

---

## Резюме: Best 9 (шаг 13980) — Memory Bank взорвался (2026-09-04)

### Состояние
- **Step:** 13980
- **Val:** 6.49e3 — **НОВЫЙ BEST** (было 3.07e4@3495)
- **CE:** 7.73 — **НОВЫЙ ALL-TIME LOW** (было 9.35@4015)
- **Maturation:** 0.630 — **ПОЛНАЯ ЗРЕЛОСТЬ** (было 0.187@5359)
- **Memory bank:** L1=1,705,276 L2=1,705,276 L3=134,093 — **АКТИВНЫ**
- **NaN:** 0, Inf: 0
- **Params:** 191.37M

### Траектория val (свежий restart)

| Step | Val | Delta | CE | Mat | L1 |
|------|-----|-------|----|-----|-----|
| 0 | 9.53e6 | — | 11.91 | 0.059 | 0 |
| 233 | 9.53e6 | — | 12.21 | 0.062 | 0 |
| 466 | 6.02e4 | — | 10.99 | 0.066 | 0 |
| 1398 | 4.84e4 | — | 10.67 | 0.081 | 0 |
| 2563 | 3.77e4 | — | 10.40 | 0.105 | 0 |
| 3495 | 3.07e4 | — | 10.22 | 0.128 | 0 |
| 4620 | 2.40e4 | — | 9.19 | 0.162 | 196 |
| 5359 | 2.10e4 | — | 9.56 | 0.186 | 28,142 |
| 6058 | 1.82e4 | — | 9.52 | 0.213 | 70,829 |
| 7922 | 1.30e4 | — | 9.07 | 0.299 | 335,353 |
| 9553 | 9.90e3 | — | 8.69 | 0.379 | 621,584 |
| 10951 | 8.52e3 | — | 8.50 | 0.459 | 926,865 |
| 12815 | 6.86e3 | — | 7.87 | 0.566 | 1,364,452 |
| 13980 | **6.49e3** | — | 8.19 | 0.630 | 1,653,940 |

### Вехи DepthController
- 3961: active_depth=12/24
- 4427: active_depth=16/24
- 4660: active_depth=20/24
- 4893: active_depth=24/24 (полная глубина)

### Checkpoint best 9 (шаг 13980, val=6.49e3)

**Per-layer (L0 vs L23):**
| Метрика | L0 | L23 | Наблюдение |
|---------|-----|-----|------------|
| alpha | 0.9051 | 0.9073 | стабильно |
| |1-a| | 0.0949 | 0.0927 | стабильно |
| ls_std | 3.447 | 3.558 | L23 выше |
| w_help | 0.7395 | 0.7436 | uniform |
| skip | 1.003 | 1.001 | стабильно |
| pm_norm | 0.210 | 0.573 | L23 выше (.deep-first maturation) |

**VSA tau:** tau[0]=8.09, tau[-1]=532.02, ratio=65.8x, tau_l_dev=0.0000

**Layer Bridge Gate:** log_tau ≈ ±0.002 (sigmoid ≈ 0.50 на всех слоях)
- Runtime gates: L0=0.44, L4=0.70, L17=0.26, L23=0.52

**Memory bank:**
- L1: log_tau=-1.18 (tau ~0.3), out_proj working
- L2: log_tau=0.27 (tau ~1.3), 16 slots occupied
- L3: log_tau=1.61 (tau ~5.0), 8 concept slots
- L3 vals: [-1071, +1075] — экстремальный диапазон
- fusion.0.weight: [2560, 10240] — 4×2560 input dims

**LM HEAD:**
- token_bias: mean=-3.71, std=0.69 (сильная мемоизация)
- log_temp: mean=-0.007, t_eff=0.806

**REASONING:** enabled_step=13981, все gates=0 (ещё не включился)

**INSPECTOR:**
- help=0.503, w_help(sigmoid)=0.740
- trust: mean=0.0156, diag=0.5000
- pm_step: 0/0

**WAKE DETECTOR:**
- MLP W_std: mean=0.0694 (expected 0.0691, +0.0003 dev) — PASS
- concept-birth: mean=0.0000 max=0.0000 — WATCH mode
- slots: 83/192, 9 full layers, 11 empty layers
- maturation: min=0.397, max=0.817, mean=0.630

**LIVE forward:**
- CE(random)=26.25 — модель различает random vs real
- gate_l1=0.476 — хороший разброс
- Per-layer ||hp||: L0=3.4 → L23=7.0 (градиенты растут через глубину)
- Per-layer |mirror|: L7=67 → L23=196

**ANOMALY TRACK:**
- max ||hp|| L23=7.02 (median 4.38, ×1.6)
- max predMSE L23=2.8
- min gate L17=0.26
- No births in empty layers

**BRIDGE:**
- SALIENCE: mean=0.381, H/Hmax=0.997 (почти равномерно)
- INTENT STREAM: per-layer norm 0-2240
- Cross-layer cosine: 0.0376 (low = good, слои дополняют друг друга)
- BUS norm=1746.6, bus_bias norm/pos=204.9
- intent_probe: W_norm=19.7, b_norm=0.48

### Генерация текста (CPU, 15-20 токенов)

| Промпт | Генерация |
|--------|-----------|
| `Привет` | `Казах травмission высоцкогооде базой аллегори вскочилйн...` |
| `Москва` | `оак превратили Хар227 1773 предупредили туристический...` |
| `Искусственный интеллект` | `разрезал помешает телефону скобках физическимиодами...` |

**Вывод:** Генерация случайная — **ожидаемо** при val=6.49e3 (CE=7.73). Модель ещё не выучила статистику языка.

### Ключевые изменения от best 8 к best 9
| Метрика | best 8 (5359) | best 9 (13980) | Δ |
|---------|---------------|----------------|---|
| Val | 2.10e4 | 6.49e3 | **↓69%** |
| CE | 9.56 | 7.73 | **↓19%** |
| Maturation | 0.187 | 0.630 | **×3.4** |
| L1 | 28,142 | 1,705,276 | **×60** |
| L3 | 36 | 134,093 | **×3,724** |
| CE(random) | 12.5 | 26.25 | **×2.1** |
| token_bias mean | -1.38 | -3.71 | **×2.7** |
| bridge gate | active | ~0.50 | neutral |

### Что делать дальше
1. **Продолжить обучение** — val=6.49e3, цель val<1000
2. **Включить reasoning** — enabled_step=13981, возможно улучшит генерацию
3. **При val<1000** — сделать финальный inference тест
4. **Investigate L3 vals** — экстремальный диапазон [-1071, +1075]
5. **Colab budget** — оценить оставшиеся кредиты

---

## Резюме: Best 11 (шаг 15611) — val=6.37e3, token_bias=-4.16 (2026-09-04)

### Состояние
- **Step:** 15611
- **Val:** 6.37e3 — **НОВЫЙ BEST** (было 6.49e3@13980)
- **CE(random):** 16.80 — **НОВЫЙ ALL-TIME LOW** (было 26.25@13980)
- **Maturation:** 0.713 — **ПОЛНАЯ ЗРЕЛОСТЬ** (было 0.630@13980)
- **Memory bank:** L1=2,179,876 L2=2,179,876 L3=181,561 — **АКТИВНЫ**
- **NaN:** 0, Inf: 0

### Ключевые улучшения от best 9 к best 11

| Метрика | best 9 (13980) | best 11 (15611) | Δ |
|---------|---------------|-----------------|---|
| Val | 6.49e3 | 6.37e3 | **↓1.8%** |
| CE(random) | 26.25 | 16.80 | **↓36%** |
| out_norm | 1035.2 | 57.2 | **↓94%** |
| token_bias mean | -3.71 | -4.16 | ↓12% |
| token_bias std | 0.69 | 0.76 | ↑10% |
| Maturation | 0.630 | 0.713 | ↑13% |
| L1 | 1,653,940 | 2,179,876 | ↑32% |
| L3 | 134,093 | 181,561 | ↑35% |

### Token Bias — подробный анализ
- **mean:** -3.71 → -4.16 (модель стала более уверена в предсказаниях)
- **std:** 0.69 → 0.76 (разница между частыми и редкими токенами выросла)
- **Топ токены:** `','`, `''`, `'.'`, `' и'`, `' в'` — частотные токены русского языка
- **Интерпретация:** модель выучила prior probability каждого токена

### Concept Birth — ожидание
- Maturation mean=0.713, max=0.870
- Порог concept birth: ~0.75
- **Осталось ~1200 шагов** до активации concept birth

### Что делать дальше
1. **Продолжить обучение** — val=6.37e3, цель val<5000
2. **Ждать concept birth** — maturation 0.713, порог 0.75
3. **При val<5000** — сделать inference тест
4. **Investigate lbg_diversity=0** — все gates одинаковые

---

## τ-Gated Ranking — F4 (2026-09-05)

### Проблема
Ranking loss монотонно рос: 3917→7970 за 700 шагов (×2), загрязняя `aux_total` и доминируя в `||g_aux||`.

### Решение
~~τ-gated ranking~~ **REMOVED** (ranking_weight=0.0). Ranking loss was unbounded,
grew 4000→15000+ while CE stayed 7-9, τ-gating stuck at 0.5. Effective contribution
was dominated by LossBalancer spectral alignment. div + gate_repulse + signal_ent
provide equivalent regularization (all bounded). Ranking added no unique signal.

### Результаты (smoke test)
| Config | ranking |
|--------|---------|
| τ-gated (default) | 648 |
| Unbounded (gate=1) | 672 |
| Gated-off (gate≈0) | 391 |
| Gradients | log_tau_ranking: 76.0, ranking_threshold: -23.0 |

### Изменения в коде
- `stack.py:114-119`: Added `log_tau_ranking` and `ranking_threshold` params
- `stack.py:1024-1039`: Ranking loss gated by `sigmoid(tau_rank * (ls_spread - rank_thresh))`
- `stack.py:1128-1130`: `_cached_losses` includes `tau_rank_gate` and `ranking_threshold` diagnostics
- `stack.py:1289-1292`: L2 reg on `log_tau_ranking` (0.01)
- `stack.py:1418-1419`: Routed to `tau_dev` param group (0.2× LR)
