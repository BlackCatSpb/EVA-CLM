# EVA-CLM — предлагаемые патчи (P0/P1/P2)

Источник: внешний математический аудит (`EVA-CLM_math_report.md`, commit 3bccfc2).
Все патчи следуют доктрине проекта: **identity-at-init / checkpoint-safe / A-B-рука / тест-лок**.
Каждый патч независим; порядок применения — по приоритету. В конце каждого — критерий приёмки.

Легенда приоритетов:
- **P0** — корректность: чинит измеренные режимные/численные дефекты живых путей.
- **P1** — потолок качества: расширяет выразительность головы и даёт инструмент измерения потолка.
- **P2** — научная чистота: минимальная абляция + baseline для атрибуции тезиса.
- **P3** — метакогнитивная инструментация (по заявке автора): леджер регуляторов,
  леджер рождений с MDL-ценой, census двойных писателей + двухвременна́я проверка.

---

## P0-1. F2: предохранитель fp32-скана (τ_s-кламп)

**Суть.** Tail-referenced скан переполняется в fp32, когда динамический диапазон чанка
`32·k/τ_s > ln(FLT_MAX) ≈ 88.7` (измерено: τ_s=0.3 → NaN). `τ_s = _base_vsa·τ_l/τ_mid`
зависит от обучаемого без ограничений `_vsa_log_param` → зона NaN достижима дрейфом.

**Файл:** `core/block.py`, `EVABlock.forward`, сразу после
`tau_s = torch.exp(self._vsa_tau_log) if tau_s is None else tau_s` (~строка 577).

```python
        tau_s = torch.exp(self._vsa_tau_log) if tau_s is None else tau_s
        # ─── P0-1 (F2): fp32-scan safety floor ───
        # Условие конечности tail-referenced скана: 32·|floor_log| < 88.7,
        # floor_log = −k/τ_s ⇒ τ_s > 32k/88.7 ≈ 0.36k. Берём 0.5k (запас ×1.4).
        # Кламп живёт ЗДЕСЬ (единственная точка входа τ_s), не в _scan_chunks:
        # семантика пола τ_s/k не меняется молча — меняется сам τ_s, и это
        # телеметрируется (bind срабатывает = лестница ушла в опасную зону).
        if self._vsa_floor_k > 0:
            _tau_safe = 0.5 * self._vsa_floor_k          # k=2 ⇒ τ_s ≥ 1.0
            _bound = tau_s < _tau_safe
            tau_s = tau_s.clamp(min=_tau_safe)
            if self.training and bool(_bound.any()):
                self._scan_floor_bound = True            # телеметрия (сбрасывается стеком)
```

**Дополнительно (пояс, не замена):** в `_scan_chunks` floored-ветвь — одноразовое предупреждение вместо молчаливого NaN:

```python
    if floor_log is not None:
        # P0-1 guard: экспонента чанка не должна превышать ln(FLT_MAX)−запас
        if float(floor_log.min()) * chunk < -80.0:
            import warnings
            warnings.warn('scan: floor_log·chunk < −80 — fp32 overflow zone', RuntimeWarning)
```

**Тест** (`tests/test_p0_scan_floor.py`):

```python
def test_scan_finite_at_extreme_tau():
    cfg = EVAConfig(D=256, n_layers=2, mirror_k=8, mlp_groups=8, lambda_d_enabled=False,
                    gradient_checkpointing=False, logit_cache_enabled=False)
    blk = EVABlock(cfg, 0)
    with torch.no_grad():
        blk._vsa_tau_log.fill_(-50.0)        # τ_s → e^-50 (дрейф)
    h = torch.randn(1, 64, 256)
    out, st = blk(h)
    assert torch.isfinite(out).all() and all(
        t is None or torch.isfinite(t).all() for t in st if isinstance(t, torch.Tensor))
```

**Приёмка:** тест зелёный; при штатных τ_s (≥8·0.125≈1) кламп никогда не связывает
(проверить `_scan_floor_bound` отсутствует за 100 шагов малой модели).

---

## P0-2. F3: weight tying в графе, а не в значениях

**Суть.** `_tie_hook` копирует `W_proj.weight.data → W_out.data` под `no_grad`:
градиент выходного пути теряется (измерено: ‖W_out.grad‖≈51.5 в никуда), собственные
обновления W_out стираются следующим forward, Adam-моменты W_out — мёртвый груз.

### 2a. `core/bind.py`, `BottleneckBind`

```python
    def __init__(self, D, K, cfg):
        ...
        if self.mode != "off" and self.ocular == "multi" and self.S > 1:
            self.W_out = nn.Parameter(torch.empty(self.S, K, D))
            nn.init.xavier_uniform_(self.W_out, gain=0.5)
            self._tied = False
        else:
            self._tied = tie_bind
            if self._tied:
                # P0-2: tied-окуляр — НЕ параметр; чтение in-graph из W_proj.weight.
                # W_out больше не создаётся (старые чекпойнты: ключ игнорируется
                # при загрузке — значения и так равны W_proj, см. migrate-заметку).
                self.W_out = None
            else:
                self.W_out = nn.Parameter(torch.empty(K, D))
                nn.init.xavier_uniform_(self.W_out, gain=0.5)
            # _tie_hook удалён

    def _ocular(self):
        # единственный источник окуляра; градиент выходного пути достигает W_proj
        return self.W_proj.weight if self._tied else self.W_out
```

Во всех ветвях forward заменить `... @ self.W_out` → `... @ self._ocular()`.

### 2b. `core/mirror.py`, `GroupedCognitiveMirror`

Буфер `W_out` остаётся (checkpoint-совместимость), но **не участвует в вычислениях**:

```python
    def _W_out_eff(self):
        # P0-2 (F3): tie = in-graph транспозиция W_proj, а не копия значений.
        # Реконструкторский градиент (delta @ W_out → mirror → CE) теперь
        # формирует W_proj — «K-space автоэнкодер» учится обеими сторонами.
        if self.tie_mirror_proj:
            return self.W_proj.permute(0, 2, 1)      # (G, k, d) — тот же einsum
        return self.W_out
```

В forward: `linear = torch.einsum('blgk,gkd->blgd', delta, self._W_out_eff())`.
Хук `_sync_W_out` оставить (буфер нужен старым чекпойнтам/инспекторам), но в
`param_groups`/`build_optimizer` связанных параметров не появится автоматически —
W_out и так буфер.

**A/B-рука (рекомендую):** флаг `cfg.tie_grad: bool = False` по умолчанию; `True`
включает `_W_out_eff`, `False` — старое чтение буфера. На resume это **новая**
градиентная тропа — запускать только с `tie_grad=True` с начала или с явным A/B.

**Миграция:** в `core/migrate.py` — при загрузке старого ckpt с tied-bind:
ключ `bind.W_out` отбрасывать (значения равны W_proj по инварианту хука);
логировать как skipped-by-design, не как unexpected.

**Тест:**

```python
def test_tie_gradient_reaches_proj():
    cfg = EVAConfig(D=64, bind_K=16, bind_twist_mode='off', tie_bind=True,
                    lambda_d_enabled=False)
    b = BottleneckBind(64, 16, cfg)
    x = torch.randn(1, 4, 64)
    b(x).pow(2).sum().backward()
    g_hook = b.W_proj.weight.grad.norm().item()
    # эталон: in-graph tie
    b.W_proj.weight.grad = None
    hp = b.hp_norm(b.W_proj(x) + b.w_bind_bias)
    prod = (hp * b.w_u[0]) * (hp * b.w_v[0])
    (prod @ b.W_proj.weight).pow(2).sum().backward()   # окуляр = W_proj
    assert b.W_proj.weight.grad.norm().item() >= g_hook * 1.01
```

**Приёмка:** в tied-режимах `W_proj.grad` строго больше прежнего; `W_out` отсутствует
в `named_parameters()` bind; forward побитово равен прежнему (значения окуляра те же).

---

## P0-3. F6: корректный HRR-unbind

**Суть.** `_circ_corr_idx[t,n] = (t−n) mod K` даёт `unbind(a, a⊛b) = K·reverse(b)`
(измерено: cos с reverse = 0.79, с b = −0.07). Для циркулярной свёртки правильный
unbind — корреляция с индексом `(t+n) mod K` (в DFT: `irfft(conj(A)·C)/K = b`).

**Файл:** `core/bind.py`. В `TrajectorySpiralBind.__init__` (рядом с `_circ_conv_idx`):

```python
        # P0-3: индекс unbind — (t+n) mod K, НЕ (t−n):
        # Σ_t a_t·c_{(t+n)} = K·b_n при c = a⊛b (автокорреляция биполярного a ≈ K·δ).
        circ_unbind = torch.tensor(
            [[(t + n) % K for n in range(K)] for t in range(K)], dtype=torch.long)
        self.register_buffer('_circ_unbind_idx', circ_unbind, persistent=False)
        # _circ_corr_idx оставить (старые чекпойнты), но unbind читает новый
```

В `TrajectoryManifoldBind._hrr_unbind`:

```python
    def _hrr_unbind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        bg: torch.Tensor = b[..., self._circ_unbind_idx]
        return torch.einsum('blt,bltn->bln', a, bg)
```

**Тест:**

```python
def test_hrr_unbind_inverts_bind():
    torch.manual_seed(0); K = 64
    a = torch.sign(torch.randn(1, 1, K)); b = torch.sign(torch.randn(1, 1, K))
    bind = TrajectorySpiralBind.__new__(TrajectorySpiralBind)  # или через cfg
    # ... собрать индексы как в __init__ ...
    c = bind._hrr_bind(a, b)
    rec = bind._hrr_unbind(a, c) / K
    assert F.cosine_similarity(rec, b, dim=-1).item() > 0.7
```

**Приёмка:** cos(unbind(a,bind(a,b))/K, b) > 0.7 на биполярных векторах; ветвь
`traj_manifold=True` кластеризует переходы в том же пространстве, что и запросы чтения.
Влияет только на `TrajectoryManifoldBind` (default off) — безопасно.

---

## P0-4. F4: восстановить класс `LambdaConfig`

**Суть.** `spectral_radius` определён на col 0 внутри тела класса; `summary` и
`print_comparison` после него стали недостижимыми вложенными функциями (проверено:
`hasattr(LambdaConfig, 'summary') is False`).

**Файл:** `core/lambda_utils.py`. Механическая правка:
1. Вырезать блок `def spectral_radius(...)` (строки ~306–344) и перенести **в конец файла** на уровень модуля (он и так модульный — оставить как есть, но ПОСЛЕ класса, чтобы класс не разрывался).
2. `def summary(self)` / `@staticmethod def print_comparison(d)` — дедентировать на один уровень (в тело класса `LambdaConfig`), разместить перед перенесённым `spectral_radius`.
3. В `spectral_radius` убрать неиспользуемые `device`/`dtype`; `n_iters` по умолчанию поднять до 3 (усреднение по Рэлею для неэрмитова J на одной итерации зашумлено).

**Тест:** `assert hasattr(LambdaConfig(3), 'summary') and callable(LambdaConfig.print_comparison)` + `test_tau_lint` остаётся зелёным.

---

## P0-5. F1a: стриминг L=1 — перенос hp_prev, позиции и rel-pos

**Суть.** При L=1 (LiveInference/после prefill): `hp_prev ≡ 0` ⇒ pen-экономика
мертва (pred_error ≡ 0.25, измерено); `pos_id` застревает на позиции 0 (hp-расхождение
~150%); `sent_masks` дают rel≡0/bos≡True; sentence-ring пулит «предложения» из одного
токена. Чинится переносом четырёх скаляров/тензоров — по образцу `conv_state`.

### 5a. `core/mirror.py` — перенос hp_prev и позиции (только AR-режим)

```python
    # __init__ (рядом с _ar_mode-потребителями):
    self.register_buffer('_stream_hp_prev', torch.zeros(1, 1, G, k), persistent=False)
    self.register_buffer('_pos_ptr', torch.zeros(1, dtype=torch.long), persistent=False)

    # forward, замена строки hp_prev = cat([zeros, hp[:, :-1]]):
    def _hp_prev_with_carry(self, hp):
        z = torch.zeros_like(hp[:, 0:1])
        if getattr(self, '_ar_mode', False) and hp.shape[1] == 1:
            c = self._stream_hp_prev
            if c.shape[0] == hp.shape[0] and c.shape[-1] == hp.shape[-1]:
                z = c.to(hp.dtype)
        return torch.cat([z, hp[:, :-1]], dim=1)

    # там же — позиционная маска:
    def _pos_slice(self, L):
        if getattr(self, '_ar_mode', False) and L == 1:
            p = int(self._pos_ptr.item()) % min(self.seq_len, self._pos_id_buf.shape[1])
            self._pos_ptr += 1                      # AR-only: training/eval не трогают
            return self._pos_id_buf[:, p:p + 1]
        return self._pos_id_buf[:, :L]

    # в конце forward (после кэшей, оба режима — «streaming correctness»):
    if getattr(self, '_ar_mode', False):
        self._stream_hp_prev.copy_(hp[:, -1:].detach())

    # reset_stream_bufs(): добавить
    self._stream_hp_prev.zero_(); self._pos_ptr.zero_()
```

**Почему `% seq_len`:** тренировка видит только паттерны позиций 0…seq_len−1;
циклирование держит стриминг внутри выученного распределения масок.

### 5b. `core/embedding.py` — streaming-вариант `sent_masks`

```python
    # PartitionedEmbedding: буферы переноса (non-persistent)
    self.register_buffer('_sent_rel_ptr', torch.zeros(1, dtype=torch.long), persistent=False)

    def sent_masks_stream(self, tokens):
        """L=1-эквивалент sent_masks с переносом rel-позиции.
        Контракт тренировки (побитово): rel(SEP) = длина его предложения − 1,
        после SEP счётчик обнуляется, bos = (rel == 0) — включая SEP подряд."""
        assert tokens.shape == (1, 1), 'stream sent_masks: B=L=1 (AR-декод)'
        sep = (tokens == 2)
        rel = self._sent_rel_ptr.clamp(max=self._sent_pos_max - 1)
        bos = (rel == 0)
        if bool(sep.any()):
            self._sent_rel_ptr.zero_()
        else:
            self._sent_rel_ptr += 1
        return sep, bos, rel.view(1, 1)

    # forward: если self._ar_mode и L == 1 → sent_masks_stream, иначе sent_masks
```

(`_ar_mode` на эмбеддинг пробрасывает стек там же, где ставит `_head._tokens`:
`self.embed._ar_mode = <флаг из generate/LiveInference>`. Мутация буфера вне
no_grad безопасна: путь активен только в AR-декоде — training/gradient-checkpointing
его не исполняют, двойного тика при рекомпьюте нет.)

### 5c. `core/logit_cache.py` — sentence-ring для L=1

В `LogitAttention.forward` training-ветвь: при `L == 1` вместо scatter-пулинга —
накопитель на модуле:

```python
        # P0-5c: стриминговый аккумулятор предложения (L=1)
        if L == 1 and self.sentence_ring and tokens is not None:
            if not hasattr(self, '_sent_acc'):
                self._sent_acc = [torch.zeros(B, 1, self.kv_dim, device=h.device),
                                  torch.zeros(B, 1, self.kv_dim, device=h.device),
                                  torch.zeros(B, 1, device=h.device)]
            self._sent_acc[0] += k_new; self._sent_acc[1] += v_new; self._sent_acc[2] += 1
            if bool((tokens == 2).any()):
                n = self._sent_acc[2].clamp_min(1.0)
                cache.push_kv_sent((self._sent_acc[0] / n).detach(),
                                   (self._sent_acc[1] / n).detach(),
                                   int(n.max().item()))
                self._sent_acc[0].zero_(); self._sent_acc[1].zero_(); self._sent_acc[2].zero_()
            # в оконном режиме — прежний scatter-путь
```

### 5d. Snapshot-контракт (M8)

`core/stack.py`, `snapshot_runtime_buffers`: `_stream_hp_prev`/`_pos_ptr`/`_sent_rel_ptr` —
registered buffers ⇒ покрываются автоматически; `_sent_acc` кэша — добавить в
`__attrs__`-обход (или зарегистрировать как буферы). `reset_streams()`/`reset_cache()`
должны вызывать `reset_stream_bufs()` и обнулять `_sent_acc`/`_pos_ptr`/`_sent_rel_ptr`
(граница документа = холодные стримы, контракт T7).

**Тест** (`test_p0_stream_parity.py`) — ключевой:

```python
def test_ar_stream_matches_window_tail():
    """Окно [t0..t7] teacher-forced против 8 AR-шагов с переносом:
    предиктивные сигналы зеркала обязаны совпасть (pen, hp), а не разойтись на 150%."""
    model = _small_model(); model.eval()
    for l in model.layers: l.mirror._ar_mode = True
    toks = torch.randint(3, V, (1, 8))
    with torch.no_grad():
        h = model.embed_tokens(toks)
        out_w, *_ = model(h, None, adaptive=False, tokens=toks)
        pen_w = model.layers[0].mirror._cached_pred_error_norm[0].clone()
        model.reset_streams(); model.reset_cache()
        for l in model.layers: l.mirror.reset_stream_bufs()
        state = gs = None; pens = []
        for t in range(8):
            h1 = model.embed_tokens(toks[:, t:t+1])
            out_s, state, gs, _ = model(h1, _carry(state), global_state=gs,
                                        adaptive=False, tokens=toks[:, t:t+1])
            pens.append(model.layers[0].mirror._cached_pred_error_norm[0, -1].item())
    pen_s = torch.tensor(pens)
    rel = ((pen_s - pen_w) .abs() / (pen_w.abs() + 1e-6))[1:]   # позиция 0 — холодный старт
    assert rel.mean() < 0.15, f'AR-стриминг расходится с окном: {rel.mean():.3f}'
```

**Приёмка:** средний rel-diff pen между AR-стримингом и окном < 0.15 (было: pen
заморожен на 0.25, hp-расхождение 1.5); тренировочный путь (L>1, не _ar_mode)
побитово не изменился — запереть отдельным assert'ом.

---

## P0-6. F1b: генерация без L-кратной перезаписи памяти

**Суть.** `generate.py` пере-подаёт перекрывающиеся окна с переносом state:
каждый токен пишется в VSA/банк/UCL ≈ L раз (измерено: медленная шкала ×3.52 при
L=8; при L=384 — кратно хуже). После P0-5 переходим на **prefill + инкрементный
L=1-декод** — каждый токен пишется ровно один раз, как в тренировке.

**Файл:** `scripts/generate.py`, цикл генерации:

```python
    # ─── Prefill: одно окно промпта, state инициализируется один раз ───
    ctx = tokens[-L:].unsqueeze(0)
    h = model.embed_tokens(ctx)
    out, state, gs, rb = model(h, None, global_state=None, adaptive=False,
                               step=base_step, tokens=ctx,
                               intent_state=intent_state,
                               reasoning_buffer=None, reasoning_count=None)
    model.observe_output(model.lm_head(out))
    # AR-режим для зеркал/эмбеддинга (включается ДО декода, как в LiveInference):
    for _mm in [l.mirror for l in model.layers]: _mm._ar_mode = True
    model.embed._ar_mode = True

    # ─── Декод: строго L=1, state несётся, окна НЕ перекрываются ───
    for step in range(max_new_tokens):
        logits = head(out[:, -1:, :])[0, 0]
        next_token = ...  # сэмплер без изменений
        tok1 = next_token.view(1, 1)
        h1 = model.embed_tokens(tok1)
        out, state, gs, rb = model(h1, state, global_state=gs, adaptive=False,
                                   step=base_step + step + 1, tokens=tok1,
                                   intent_state=intent_state,
                                   reasoning_buffer=rb[0] if rb else None,
                                   reasoning_count=rb[1] if rb else None)
        intent_state = getattr(model, '_last_intent_state', None)
        model.observe_output(model.lm_head(out))
        # salience/intent: форма (B,1) — стек уже поддерживает
```

**Альтернатива (если оконный декод хочется сохранить):** маска записи — передавать
в блок `write_mask (B,L)` и домножать `i_gate` (и запись UCL/банка) на маску
«только новый суффикс». НЕ рекомендую как основную: перенос state через перекрытие
всё равно пере-применяет decay ко всему окну (память гаснет в L раз быстрее) —
полная консистентность достигается только инкрементным путём.

**Тест-лок:** на toy-потоке из 16 токенов (окно 8) — per-scale нормы VSA-состояния
после генерации совпадают с нормами после train-style стриминга тех же токенов
в пределах 10% (до фикса: расхождение ×3.5 на медленной шкале). Плюс
`test_t9_generate_parity` обновить: gs/intent по-прежнему ведутся, но окна не
пере-подаются.

**Приёмка:** память/банк/UCL/кэш получают каждый токен ровно один раз; CE на
hold-out в AR-режиме сходится к оконному CE тех же документов (±5%), а не
расходится, как сейчас.

---

## P1-1. Измерить потолок факторизованной головы (диагностика, до всяких фиксов)

**Суть.** Голова — ранг-K log-линейное семейство над кодами; битовая независимость
не выражает внутрикодовых корреляций. Их же замер «CE code-only 12.84 > bias-only
8.65» говорит, что битовый путь пока проигрывает unigram-биасу. Прежде чем
расширять голову — измерить, **сколько CE семейство теряет в принципе** на
биграммном уровне.

**Новый файл:** `scripts/probe_head_ceiling.py`

```python
"""Потолок PoB-головы: сравнение CE на корпусе:
  (1) unigram, (2) bigram (полный softmax), (3) проекция bigram на семейство
      q(v|u) ∝ exp(Σ_k c_vk u_k) — ровно то, что голова МОЖЕТ выразить.
(3) решается как выпуклая задача (CE выпукла по u: logsumexp — выпуклая, p — константы) LBFGS по 64-м
координатам на контекст. Разница (3)−(2) = цена битовой независимости;
если она велика — P1-2 (парный канал) обязателен, а не опционален."""
import sys, math, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from core.config import EVAConfig
from core.vsa_utils import build_codes

def bigram_counts(tokens: torch.Tensor, V: int):
    idx = tokens[:-1] * V + tokens[1:]
    return torch.bincount(idx, minlength=V * V).reshape(V, V).double()

def fit_pob(logp_row: torch.Tensor, C: torch.Tensor, iters=300):
    """logp_row: (V,) log целевого условного распределения; C: (V,K). Минимизация
    KL(p || q_u), q_u = softmax(C u) — выпукло по u."""
    u = torch.zeros(C.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([u], max_iter=iters, line_search_fn='strong_wolfe')
    p = logp_row.exp()
    def closure():
        opt.zero_grad()
        loss = -(p * (C @ u - torch.logsumexp(C @ u, dim=-1))).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        ce = -(p * (C @ u - torch.logsumexp(C @ u, dim=-1))).sum()
    return float(ce)

def main(stream_path: str, n_ctx=400):
    import numpy as np
    cfg = EVAConfig(code_dim=64, code_sparsity=6, vocab=65536, codebook='twin_free')
    C = build_codes(cfg).double()
    data = torch.from_numpy(np.fromfile(stream_path, dtype=np.uint16).astype(np.int64))
    cnt = bigram_counts(data, cfg.vocab)
    p_next = cnt / cnt.sum(1, keepdim=True).clamp_min(1)
    # контексты — топ по частоте (репрезентативные)
    freq = cnt.sum(1)
    ctx = freq.argsort(descending=True)[:n_ctx]
    w = freq[ctx].double(); w /= w.sum()
    ce_full = ce_pob = 0.0
    for i, c in enumerate(ctx):
        p = p_next[c]
        lp = (p + 1e-12).log()
        ce_full += w[i] * float(-(p * lp).sum())
        ce_pob  += w[i] * fit_pob(lp, C)
    print(f'bigram CE (полный softmax): {ce_full:.4f}')
    print(f'bigram CE (потолок PoB-головы): {ce_pob:.4f}')
    print(f'цена битовой независимости: {ce_pob - ce_full:+.4f} нат')
```

**Приёмка:** число в логе/whiteboard. Интерпретация: gap < 0.1 нат — потолок не
связывает, приоритет у ствола; gap > 0.3 нат — P1-2/P1-3 в критический путь.

---

## P1-2. Контекстно-зависимый парный канал головы (rank-r Ising-поправка)

**Суть.** Расширить семейство головы с Π σ(u_k)^{c_vk} до
`logit_v = u·c_v + (C z₁)_v ⊙ (C z₂)_v + b_v`, где `z₁ = u V₁`, `z₂ = u V₂`,
`V ∈ R^{K×r}` — ранговая r контекстная поправка на **пары битов**. Стоимость:
+2 matmul'а (V,K)×(K,r) на позицию (~+30% головы), +2Kr = 2048 параметров (r=16).
Нормировка logsumexp остаётся точной (семейство замкнуто). Zero-init **только V₂**
(V₁ — малый случайный): произведение двух нулей — седло с нулевым градиентом
в обе стороны; асимметричный init даёт живую тропу (тот же урок, что phantom_mix:
«mix zero-init, но p≠0»).

**Файл:** `core/embedding.py`, `SigmoidCodedHead`.

В `core/config.py` — новое поле:

```python
    head_pair_rank: int = 0   # 0 = off (identity), A/B-рука: 16
```

В `SigmoidCodedHead.__init__` (после phantom-блока):

```python
        self.pair_r = int(getattr(cfg, 'head_pair_rank', 0) or 0)
        if self.pair_r > 0:
            assert self.normalize, 'pair-канал требует head_normalize=True'
            _g = torch.Generator().manual_seed(11)
            self.pair_V1 = nn.Parameter(torch.randn(self.K, self.pair_r, generator=_g) * 0.02)
            self.pair_V2 = nn.Parameter(torch.zeros(self.K, self.pair_r))  # zero → identity
```

В `forward` — замена блока построения logits:

```python
        logits = u @ self.codes.T \
                 + (base[..., None] if not self.normalize else 0.0) \
                 + self.token_bias
        if self.pair_r > 0:
            z1 = u @ self.pair_V1                          # (…, K)@(K,r) → (…, r)
            z2 = u @ self.pair_V2
            logits = logits + (z1 @ self.codes.T) * (z2 @ self.codes.T)   # (…, V)
```

**Тест:** (a) `pair_r=16` на init: logits побитово равны `pair_r=0` (V₂=0);
(b) один шаг SGD на синтетической цели, где истинный logit содержит член
`c_3·c_7` (взаимодействие битов): с `pair_r=16` CE падает ниже, чем с `pair_r=0`
за N шагов; (c) градиент `pair_V2.grad` ненулевой на первом шаге.

**Приёмка:** на P1-1-диагностике парный канал закрывает ≥30% gap'а битовой
независимости (проверить: расширить probe суммой `u·c + (Cz₁⊙Cz₂)` и пере-фитить).

---

## P1-3. Морфологические коды вместо случайных (индуктивный bias для головы)

**Суть.** Коды сейчас случайны (combinadic/twin_free) — голова не получает никакой
морфологической структуры. `word_num` уже содержит нужную алгебру (буква→простое,
log-состав). Кодируем слово через random-projection его log-буквенного состава и
top-S — похожие слова (высокий morph_sim) разделят больше битов; факторизованная
голова получает способность обобщать по морфологии, а d_min-структура остаётся
управляемой жадным ремонтом (их же twin_free-упаковка, но приоритет = семантика).

**Новый файл:** `core/codebooks_morph.py` (или расширение `vsa_utils.build_codes`)

```python
import math, torch
from .word_num import PHI, _norm, ALPHABET

def morph_codes(word_list, K=64, S=6, seed=42, max_overlap=None, repair=True):
    """word_list: list[str] длины vocab (порядок = id токена).
    Возвращает (V,K) 0/1, детерминированно."""
    A = len(ALPHABET)
    X = torch.zeros(len(word_list), A)
    for i, w in enumerate(word_list):
        for ch in _norm(w):
            X[i, ALPHABET.index(ch)] += math.log(PHI[ch])
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)      # ≈ их morph_sim-геометрия
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(A, K, generator=g)
    scores = X @ R                                            # (V,K)
    codes = torch.zeros(len(word_list), K)
    seen = {}
    for v in range(len(word_list)):
        order = scores[v].argsort(descending=True)
        pick = order[:S].tolist()
        if repair:
            # детерминированный ремонт коллизий: следующая по score позиция
            key = tuple(sorted(pick))
            while key in seen:
                nxt = int(order[len(pick)])
                pick[-1] = nxt                                # заменяем наименьший из S
                key = tuple(sorted(pick))
            seen[key] = v
        codes[v, torch.tensor(pick)] = 1.0
    if max_overlap is not None:                               # опц.: гарантия d_min
        codes = _greedy_overlap_repair(codes, scores, max_overlap)  # их twin_free-цикл,
        # но кандидаты упорядочены по scores (семантика), а не случайны
    return codes
```

**Требуется:** словарь id→слово из токенизатора (`tokenizer.get_vocab()`); для
спецтокенов (PAD/BOS/SEP) — любые выделенные коды (фиксированные, не из проекции).

**Тесты:** (1) вес ровно S у всех; уникальность; (2) `corr(morph_sim(w1,w2),
overlap(c1,c2)/S) > 0.3` на выборке пар (структура перенеслась в коды);
(3) при `max_overlap=S−2` — максимальный оверлап ≤ 4 (их же проверка twin_free);
(4) roundtrip головы на init = 1.000 (ортогональный базис не зависит от кодов).

**Риск/метрика A/B:** семантические близнецы уменьшают margin конкретных пар —
логировать гистограмму margin'ов; сравнивать CE + зонд «предсказать форму
незнакомой леммы» (морфологическое обобщение). Это **геометрическая** замена
кодовой книги — только fresh run (как embed_center).

---

## P2-1. Минимальная абляция: `minimal_cfg` + GRU-baseline на их же данных

**Суть.** Тезис «суперпозиция кодовых состояний > внимание» сейчас конфундирован
с ~30 надстройками. Нужен один флаг, оставляющий голый ствол, и честный baseline
того же параметрного бюджета на том же `TokenStream`.

**Файл:** `core/config.py`

```python
    @classmethod
    def minimal(cls, **kw):
        """P2-1: голый ствол — embed + блок(conv+bind+VSA+spectral+MLP+mirror-core)
        + coded head. Все когнитивные надстройки off. Только для A/B-науки."""
        cfg = cls(**kw)
        cfg.variable_precision = False; cfg.explicit_reasoning = False
        cfg.triad_reason = False;           cfg.private_mem = False
        cfg.meta_trust = False;             cfg.collective_layer = False
        cfg.unified_concept_layer = False;  cfg.logit_cache_enabled = False
        cfg.memory_bank = False;            cfg.cov_memory = False
        cfg.bridge_conn = 0.0;              cfg.intent_bridge = False
        cfg.bridge_glu = False;             cfg.head_lacuna = False
        cfg.head_srl = False;               cfg.head_temper = False
        cfg.maturation_enabled = False;     cfg.softmax_free = True
        cfg.bind_twist_mode = 'trajectory_spiral'   # ядро bind остаётся
        cfg.lambda_d_enabled = False                # плоские расписания — меньше связности
        return cfg
```

**Baseline** (`scripts/baseline_gru.py`, ~40 строк): 2-слойная GRU (hidden=2560,
vocab-эмбеддинг 65536×64 tied с их кодовой головой — тот же выходной слой для
честности), обучается их `TokenStream`, их CE-метрикой (`ce_raw`), их же LR-сеткой.
Отчёт: две кривые `ce_raw(tokens)` на 1%/5%/10% корпуса.

**Приёмка:** одна страница в `docs/WHITEBOARD.md`: минимальная EVA vs GRU vs
полная EVA на одинаковых токенах. Если минимальная EVA ≥ GRU — тезис получает
первое чистое подтверждение/опровержение; дельта «полная − минимальная» — цена
и ценность когнитивной надстройки, измеренная впервые.

---

## P2-2. Гигиена (одним коммитом, без поведения)

1. **Дубли полей dataclass** в `core/config.py`: `head_bus_cap`, `head_phantom_max`,
   `mem_lacuna_k`, `head_temper`, `head_temper_k`, `head_temper_cos`,
   `head_temper_after`, `orth_weight` объявлены дважды — оставить одно объявление
   (значения совпадают; поведение не меняется). Тест-лок: `len(fields) == len(set(fields))`.
2. **BOM + mojibake** в `core/bind.py`: пересохранить UTF-8 без BOM; docstring'и
   `TrajectoryManifoldBind` перечитать из git-истории до двойной перекодировки
   (или восстановить вручную — сейчас там cp1251-кракозябры).
3. **Мёртвый код**: `LmHead`/`ZeckendorfEmbedding`/`PartitionedHead` — под флаг
   «legacy» или в `archive/`; `_zeckendorf_levels`/`_fib_sequence` в bind.py не
   используются (проверить grep'ом).
4. **`_pos_id_buf` 4096**: assert `seq_len <= 4096` в конструкторе зеркала (сейчас
   L>4096 — тихий index error).

---

## P3 — Метакогнитивная инструментация (по заявке автора)

Мотивация (позиция автора): метакогнитивность заложена целенаправленно и неотделима
от параметров; регуляторы нужны; ожидание — более эффективное использование
параметров + рождение новых параметров в метакогнитивной части. Три инструмента
ниже переводят каждый пункт этой позиции из тезиса в **непрерывно измеряемый факт**
в вашей же методологии (kill-switch, ledger, τ-lint):

- **P3-1 RegulatorLedger** — «все регуляторы нужны» становится таблицей измеренных ΔCE;
- **P3-2 BirthLedger + MDL** — «рождение параметров» становится задачей отбора модели
  (новорождённый обязан окупить свою BIC-цену, иначе архив + блэклист направления);
- **P3-3 DualWriterCensus** — «неотделимость от параметров» формализуется как
  двухвременна́я стохастическая аппроксимация (Боркар) с тестом разделения масштабов;
- **P3-4 ParamVelocity** — KPI «эффективность использования параметров».

Все четыре не меняют обучение (кроме двух однострочных флагов в P3-1) и ложатся
на существующую инфраструктуру: `snapshot_runtime_buffers`/`restore` (M8),
чекпойнт-карусель balancer'а, eval-каденцию.

---

## P3-1. RegulatorLedger — kill-switch для регуляторов forward-пути

**Суть.** Обобщение `AuxKillSwitch` (M64.12) с aux-лоссов на управляющие законы.
Каждый регулятор периодически переводится в **identity** на фиксированной probe-партии,
парная разница CE приписывается ему:

```
delta_i = CE(identity_i) − CE(active)     [нат/токен, probe-партия, no_grad]
```

Round-robin по 2–3 регулятора на точку (каденция AuxKillSwitch); Шмитт-триггер
с dwell: `delta_EMA < eps_off = max(2σ_re, 0.01·σ_b, 1e-4)` подряд `dwell` раз ⇒
статус **DORMANT** (рекомендация к пенсии); `> 2·eps_off` ⇒ **ACTIVE**.
Две сигмы: σ_re — шум повторного прогона одной партии (детерминизм forward),
σ_b — разброс CE между двумя probe-партиями; порог «1% от batch-шума» — это
порог РЕШЕНИЯ (вклад меньше 1% фоновой вариативности партий = регулятор-кандидат
на пенсию), а не порог измерения (никаких абсолютных магических констант). Автоотключение **запрещено**
(доктрина `aux_kill_disable`): ledger рекомендует, человек/новое A/B решает через cfg-флаг.

**Новый файл:** `core/regulator_ledger.py`

```python
"""core/regulator_ledger.py — P3-1: леджер регуляторов (kill-switch для контроллеров).

Протокол измерения (на каждый регулятор, всё под snapshot/restore M8):
  1. snap = model.snapshot_runtime_buffers()
  2. ce_a = probe(batch_a); ce_r = probe(batch_a)     # σ_re — детерминизм
     ce_b = probe(batch_b)                            # σ_b — batch-swap разброс
  3. reg.identity(); ce_off = probe(batch_a); reg.restore()
  4. model.restore_runtime_buffers(snap)
  5. delta = ce_off − ce_a;  eps_off = max(2σ_re, 0.01·σ_b, 1e-4)
Замечания:
  - probe — eval/no_grad, state=None (оконный teacher-forced, как evaluate());
  - запись в накопительные состояния (кэш-кольцо, банк) в active-ноге дрейфует
    на ≤1 окна за прогон — ring/eviction это переваривают, buffers покрыты snap;
  - регуляторы, не имеющие атрибутной identity (grad_mod/dvar_mod, governor,
    intent-авторитет), требуют однострочных флагов — см. «два флага» ниже;
    v1 работает с атрибутными.
"""
from __future__ import annotations
import torch
from typing import Callable, Dict, List, Optional


class Reg:
    """Регулятор с обратимым identity-зажимом (атрибутный уровень, без граф-хирургии)."""
    def __init__(self, name, enter: Callable[[], None], leave: Callable[[], None]):
        self.name = name; self._enter = enter; self._leave = leave
    def identity(self): self._enter()
    def restore(self):  self._leave()


def _attr_reg(name, obj, attr, ident):
    """Reg для простого атрибута (bool/float/None).

    Атрибут может отсутствовать (флаги _damp_on/_pen_decay_on до приземления
    патча block.py): identity становится no-op, restore удаляет временный
    атрибут — леджер безопасен на любом состоянии кода (до введения флагов
    строки spec_damp/pen_decay меряют ~0 и осмысленны только после патча).
    """
    box = {}
    def enter():
        box['had'] = hasattr(obj, attr)
        box['old'] = getattr(obj, attr, None)
        setattr(obj, attr, ident)
    def leave():
        if box['had']:
            setattr(obj, attr, box['old'])
        else:
            try:
                delattr(obj, attr)
            except AttributeError:
                pass
    return Reg(name, enter, leave)


def _multi(name, regs: List[Reg]) -> Reg:
    return Reg(name, lambda: [r.identity() for r in regs],
                     lambda: [r.restore() for r in regs])


def build_registry(model, cfg) -> List[Reg]:
    """v1: 11 атрибутных регуляторов + 2 флаговых (см. патч block.py ниже)."""
    R: List[Reg] = []
    hd = getattr(model, 'lm_head', None)
    if hd is not None and getattr(hd, 'temper_on', False):
        R.append(_attr_reg('head_temper', hd, '_temper_active', False))
    if hd is not None and getattr(hd, 'Kp', 0) > 0:
        kp = hd._kp_active
        box = {}
        def _ph_enter(box=box, kp=kp):
            box['v'] = int(kp.item()); kp.fill_(0)      # _phantom_mix: (…,0)@(0,K) → 0
        def _ph_leave(box=box, kp=kp):
            kp.fill_(box['v'])
        R.append(Reg('head_phantom', _ph_enter, _ph_leave))
    ucl = getattr(model, 'concept_layer', None)
    if ucl is not None:
        rs = ucl.read_scale; box = {}
        def _ucl_enter(box=box, rs=rs):
            box['v'] = rs.detach().clone(); rs.data.fill_(-30.0)   # σ(−30) ≈ 1e−13
        def _ucl_leave(box=box, rs=rs):
            rs.data.copy_(box['v'])
        R.append(Reg('ucl_read', _ucl_enter, _ucl_leave))
    br = getattr(model, 'bridge', None)
    if br is not None:
        R.append(_attr_reg('bridge_inject', br, 'depth', False))
    if getattr(model, 'intent_bridge', False) and getattr(model, 'bus_head_proj', None) is not None:
        w = model.bus_head_proj.weight; box = {}
        def _bus_enter(box=box, w=w):
            box['v'] = w.detach().clone(); w.data.zero_()
        def _bus_leave(box=box, w=w):
            w.data.copy_(box['v'])
        R.append(Reg('bus_stencil', _bus_enter, _bus_leave))
    if getattr(model, 'explicit_reasoning', False):
        R.append(_attr_reg('reasoning', model, 'reasoning_scale_override', 0.0))
    R.append(_attr_reg('triad', cfg, 'triad_reason', False))
    # пер-слойные — агрегируются в ОДИН регулятор (иначе round-robin на 24×3 застрянет)
    R.append(_multi('vpm',       [_attr_reg(f'vpm_L{i}', l, 'variable_precision', False)
                                  for i, l in enumerate(model.layers)]))
    R.append(_multi('spec_damp', [_attr_reg(f'sd_L{i}', l, '_damp_on', False)
                                  for i, l in enumerate(model.layers)]))
    R.append(_multi('pen_decay', [_attr_reg(f'pd_L{i}', l, '_pen_decay_on', False)
                                  for i, l in enumerate(model.layers)]))
    if getattr(model, 'logit_cache', None) is not None:
        R.append(_attr_reg('logit_cache', model, 'logit_cache', None))
    if getattr(model, 'memory_bank', None) is not None:
        R.append(_attr_reg('memory_bank', model, 'memory_bank', None))
    box = {}
    def _m2v_enter(box=box):
        box['v'] = (cfg.w_mem2v_scale_min, cfg.w_mem2v_scale_max)
        cfg.w_mem2v_scale_min = cfg.w_mem2v_scale_max = 1.0   # scale ≡ 1 при любом diff
    def _m2v_leave(box=box):
        cfg.w_mem2v_scale_min, cfg.w_mem2v_scale_max = box['v']
    R.append(Reg('mem2v_adapt', _m2v_enter, _m2v_leave))
    return R


class RegulatorLedger:
    def __init__(self, model, cfg, probe: Callable, regs: Optional[List[Reg]] = None,
                 per_round: int = 3, dwell: int = 3):
        """probe(x, y) -> ce_raw float (no_grad, model.eval). batches — 2 фикс. партии."""
        self.regs = regs if regs is not None else build_registry(model, cfg)
        self.probe = probe
        self.batches: List = []            # [(x_a, y_a), (x_b, y_b)] — заполняет train.py
        self.per_round = per_round; self.dwell = dwell
        self.state: Dict[str, dict] = {r.name: dict(delta_ema=0.0, n=0, off=0, on=0,
                                                    status='PROBATION') for r in self.regs}
        self.sigma_re: Optional[float] = None    # шум повторного прогона (детерминизм)
        self.sigma_b: Optional[float] = None     # batch-swap разброс CE
        self.ptr = 0

    @staticmethod
    def _probe_state(model):
        """Python-состояние, НЕ покрытое snapshot_runtime_buffers (M8):
        кольца логит-кэша (списки тензоров на модуле) и счётчики каденций головы.
        Без него probe недетерминирован: active-нога пушит в кольцо, следующая
        нога читает другой кэш (σ_re = 0.57 ната на незакрытых состояниях —
        измерено). Тензоры в кольцах уже detached — копируем только списки."""
        ex = {}
        lc = getattr(model, 'logit_cache', None)
        if lc is not None:
            c = lc.cache
            ex['cache'] = (list(c._kv_h), list(c._kv_sent), list(c._sent_lens),
                           list(c._h_scores), list(c._h_lens),
                           {k: list(v) for k, v in c._kv_ms.items()},
                           {k: list(v) for k, v in c._ms_lens.items()},
                           list(getattr(c, '_p_cache', [])), c._position)
        hd = getattr(model, 'lm_head', None)
        if hd is not None and hasattr(hd, '_srl_step'):
            ex['srl_step'] = int(hd._srl_step.item())
        return ex

    @staticmethod
    def _probe_restore(model, ex):
        lc = getattr(model, 'logit_cache', None)
        if lc is not None and 'cache' in ex:
            c = lc.cache
            (c._kv_h, c._kv_sent, c._sent_lens, c._h_scores, c._h_lens,
             _kv_ms, _ms_lens, c._p_cache, c._position) = ex['cache']
            c._kv_ms = {k: _kv_ms[k] for k in c._kv_ms}
            c._ms_lens = {k: _ms_lens[k] for k in c._ms_lens}
        hd = getattr(model, 'lm_head', None)
        if hd is not None and 'srl_step' in ex:
            hd._srl_step.fill_(ex['srl_step'])

    @staticmethod
    def _restore_all(model, snap, ex):
        RegulatorLedger._probe_restore(model, ex)
        model.restore_runtime_buffers(snap)

    def _one(self, model, reg: Reg):
        # ВАЖНО: восстановление между КАЖДОЙ ногой — probe-forward сам мутирует
        # буферы (_bus_rms EMA, _intent_stream, кэши зеркала): без межногового
        # restore σ_re = 0.57 ната (измерено), с ним — fp-шум.
        snap = model.snapshot_runtime_buffers()
        ex = self._probe_state(model)
        try:
            ce_a = self.probe(*self.batches[0])
            self._restore_all(model, snap, ex)
            ce_r = self.probe(*self.batches[0])     # rerun: шум детерминизма (≈0)
            self._restore_all(model, snap, ex)
            ce_b = self.probe(*self.batches[1])     # перестановка партии: естественный разброс CE
            self._restore_all(model, snap, ex)
            reg.identity()
            try:
                ce_off = self.probe(*self.batches[0])
            finally:
                reg.restore()
        finally:
            self._restore_all(model, snap, ex)
        return ce_off - ce_a, abs(ce_r - ce_a), abs(ce_b - ce_a)

    def measure_round(self, model) -> Dict[str, float]:
        out = {}
        for _ in range(min(self.per_round, len(self.regs))):
            reg = self.regs[self.ptr % len(self.regs)]; self.ptr += 1
            d, s_re, s_b = self._one(model, reg)
            self.sigma_re = s_re if self.sigma_re is None else 0.9 * self.sigma_re + 0.1 * s_re
            self.sigma_b = s_b if self.sigma_b is None else 0.9 * self.sigma_b + 0.1 * s_b
            st = self.state[reg.name]
            st['delta_ema'] = d if st['n'] == 0 else 0.8 * st['delta_ema'] + 0.2 * d
            st['n'] += 1
            # eps_off: значим против ШУМА ПРОГОНА (2σ_re) И против 1% естественного
            # batch-разброса CE (0.01·σ_b) — регулятор, чей весь вклад < 1% фоновой
            # вариативности партий, — кандидат на пенсию (порог решения, не шума).
            eps_off = max(2.0 * (self.sigma_re or 0.0), 0.01 * (self.sigma_b or 0.0), 1e-4)
            if st['delta_ema'] < eps_off and st['n'] >= self.dwell:
                st['off'] += 1; st['on'] = 0
            elif st['delta_ema'] > 2.0 * eps_off:
                st['on'] += 1; st['off'] = 0
            st['status'] = ('DORMANT' if st['off'] >= self.dwell else
                            'ACTIVE' if st['on'] >= 1 else 'PROBATION')
            out[reg.name] = round(st['delta_ema'], 5)
        return out

    def suggestions(self) -> List[str]:
        return sorted(k for k, v in self.state.items() if v['status'] == 'DORMANT')

    def state_dict(self) -> dict:
        return {'state': self.state, 'ptr': self.ptr,
                'sigma_re': self.sigma_re, 'sigma_b': self.sigma_b}

    def load_state_dict(self, sd: Optional[dict]) -> None:
        if not sd: return
        for k, v in (sd.get('state') or {}).items():
            if k in self.state: self.state[k].update(v)
        self.ptr = int(sd.get('ptr', 0))
        self.sigma_re = sd.get('sigma_re'); self.sigma_b = sd.get('sigma_b')
```

**Два однострочных флага** (единственная правка forward — `core/block.py`):

```python
# спектральная ветвь (U3-демпфирование) — замена одной строки:
_cheb_damp = (math.cos(math.pi * self._tau_norm / 2.0)
              if getattr(self, '_damp_on', True) else 1.0)

# pen-модуляция затухания (B18b) — замена условия (тело блока без изменений):
if pen is not None and getattr(self, '_pen_decay_on', True):
    ...  # прежнее тело: _pc = pen - self._pen_ema; d_pen_factor = ...
```

Оба default-True ⇒ forward побитово прежний (ваш контракт identity).

**Probe-функция и интеграция** (`scripts/train.py`, рядом с balancer):

```python
@torch.no_grad()
def _ledger_probe(model, x, y, step):
    h = model.embed_tokens(x)
    out, *_ = model(h, None, global_state=None, adaptive=False, step=step, tokens=x)
    model.compute_losses(out[:, :-1], y[:, 1:])
    return float(model._cached_losses['ce_raw'])

ledger = RegulatorLedger(model, cfg, probe=lambda x, y: _ledger_probe(model, x, y, step))
ledger.batches = [_fixed_batch(streams, i, cfg) for i in (0, 1)]   # 2 фикс. партии hold-out

# в eval-блоке (step % eval_interval == 0), КАЖДЫЙ 4-й eval:
if step % (cfg.eval_interval * 4) == 0:
    _was_training = model.training; model.eval()
    print('  [ledger]', ledger.measure_round(model), '| dormant:', ledger.suggestions())
    if _was_training: model.train()

# чекпойнт (рядом с 'balancer'):
state['ledger'] = ledger.state_dict()
# resume: ledger.load_state_dict(ckpt.get('ledger'))
```

**Стоимость:** 3 probe-forward на регулятор, 3 регулятора на точку, точка каждые
4 eval — ~9 окон на 4×eval_interval ≈ пренебрежимо на фоне align-пути (3 обхода графа).

**Тест** (`tests/test_p3_regulator_ledger.py`):

```python
def test_ledger_classifies_dead_and_alive():
    model = _small_model(); model.eval()
    # мёртвый: свежий logit_cache (gate=4.5e-5) → delta ≈ 0 → DORMANT
    # живой: bus_head_proj.weight заполнен большим → delta > 0 → ACTIVE
    with torch.no_grad():
        model.bus_head_proj.weight.fill_(5.0)
    ledger = RegulatorLedger(model, model.cfg, probe=..., per_round=len(...))
    ...
    assert ledger.state['logit_cache']['status'] == 'DORMANT'
    assert ledger.state['bus_stencil']['status'] == 'ACTIVE'

def test_identity_is_reversible():
    """После полного круга measure_round все атрибуты/веса побитово равны исходным,
    snapshot/restore покрыл буферы (сравнить state_dict до/после)."""
    ...  # собрать модель; клонировать state_dict и ключевые атрибуты cfg;
         # прогнать measure_round по всем регуляторам; сравнить побитово
```

**Приёмка:** (1) оба теста зелёные; (2) за 20 точек на живой модели каждый регулятор
имеет статус ≠ PROBATION; (3) `suggestions()` пуст на конфигах, где всё регуляторы
дали инцидент-происхождение (или непустой — и тогда это повод для A/B, а не для спора).

---

## P3-2. BirthLedger + MDL-цена рождения

**Суть.** Новорождённые структуры (выросший фантом-бит, UCL-концепт из
подтверждённого фантома) заносятся в леджер с ценой `r` (число добавленных в
состояние величин) и измеряются **контрфактически** той же парной пробой:

```
delta_i = CE(без новорождённого i) − CE(со всеми)          [нат/токен]
gain    = delta_i · N_tok(i)                                 [нат суммарно с рождения]
cost    = λ · r_i · ln N_tok(i)                              [BIC-цена, λ=1]
вердикт = gain − cost
```

Арифметика для документации (ваш стиль — выводимые числа): фантом-бит
r = D + K = 2560 + 64 = 2624; горизонт 2000 шагов × 384 токена = 768k токенов;
cost ≈ 2624·ln(768k) ≈ 35.6k нат ⇒ **порог окупаемости ≈ 46 миллинат/токен**.
UCL-концепт: r = D + bridge_dim = 2816, порог ≈ 49 мнат/токен.

Два подряд отрицательных вердикта ⇒ **архив + блэклист направления**
(cos > 0.9 ⇒ запрет повторного рождения до истечения cooldown). Блэклист —
главный MDL-механизм: он убивает churn «рождение/архивация одного и того же»,
который вы уже чинили в банке (M64: births ≈ observations).

**Новый файл:** `core/birth_ledger.py`

```python
"""core/birth_ledger.py — P3-2: леджер новорождённых параметров + MDL-отбор.

Контрфактическое изъятие (probe под snapshot/restore M8):
  phantom_bit j: mix[:, j] ← 0  (канал точно выключается: вклад = p @ mixᵀ);
                 строка базиса остаётся, но умножается на нулевой столбец.
  ucl_slot s:    vals[s] ← 0    (внимание к мёртвому слоту остаётся —
                 консервативно: занижает gain, никогда не завышает).
Прямое сопоставление слота: entry хранит юнит-направление d; слот ищется
argmax|cos| > 0.99 среди активных строк/слотов на момент замера (API не меняется).
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


class BirthLedger:
    def __init__(self, lam: float = 1.0, horizon_steps: int = 2000,
                 dwell: int = 2, blacklist_steps: int = 20000):
        self.lam = lam; self.horizon = horizon_steps
        self.dwell = dwell; self.bl_steps = blacklist_steps
        self.entries: List[dict] = []    # kind, step, d (cpu unit), r, fails, retired
        self.blacklist: List[dict] = []  # d, until_step

    # ─── регистрация ───
    def record(self, kind: str, direction: torch.Tensor, step: int, r: int,
               slot: Optional[int] = None) -> None:
        """slot/row — ОБЯЗАТЕЛЬНО фиксируется в момент рождения: UCL-слоты
        дрейфуют между record и measure (записи концептов идут и в eval —
        «инференс = обучение»), cos-рематчинг позже теряет новорождённого
        (измерено на смоук-тесте). Fallback-матчинг — только для старых записей."""
        d = F.normalize(direction.detach().float().cpu().reshape(-1), dim=-1)
        self.entries.append(dict(kind=kind, step=int(step), d=d, r=int(r),
                                 slot=slot, fails=0, retired=False, verdicts=[]))

    def allow_birth(self, direction: torch.Tensor, step: int, thr: float = 0.9) -> bool:
        d = F.normalize(direction.detach().float().cpu().reshape(-1), dim=-1)
        for b in self.blacklist:
            if int(b['until']) > int(step) and float(abs(b['d'] @ d)) > thr:
                return False
        return True

    # ─── MDL-вердикт ───
    def mdl(self, delta_ce: float, n_tok: float, r: int) -> float:
        gain = delta_ce * n_tok
        cost = self.lam * r * math.log(max(n_tok, 2.0))
        return gain - cost

    # ─── контрфактическое изъятие (контекст-менеджер) ───
    def _locate(self, model, e: dict):
        """(объект, индекс) новорождённого: slot из записи, иначе argmax|cos|
        БЕЗ жёсткого порога (слоты дрейфуют; S мал, argmax стабилен)."""
        head = getattr(model, 'lm_head', None)
        ucl = getattr(model, 'concept_layer', None)
        if e['kind'] == 'phantom_bit' and head is not None and head.Kp > 0:
            kp = int(head._kp_active.item())
            if e.get('slot') is not None and e['slot'] < kp:
                return head, int(e['slot'])
            if kp == 0:
                return None, None
            sim = (head.phantom_basis.data[:kp].float()
                   @ e['d'].to(head.phantom_basis.device)).abs()
            return head, int(sim.argmax())
        if e['kind'] == 'ucl' and ucl is not None:
            if e.get('slot') is not None:
                return ucl, int(e['slot'])
            sim = (F.normalize(ucl.concept_vals.detach().float(), dim=-1)
                   @ e['d'].to(ucl.concept_vals.device)).abs()
            return ucl, int(sim.argmax())
        return None, None

    def _excise(self, model, e: dict):
        obj, idx = self._locate(model, e)
        if obj is None:
            return None
        if e['kind'] == 'phantom_bit':
            col = obj.phantom_mix.data[:, idx].clone()
            obj.phantom_mix.data[:, idx].zero_()
            return lambda: obj.phantom_mix.data[:, idx].copy_(col)
        row = obj.concept_vals.data[idx].clone()
        obj.concept_vals.data[idx].zero_()          # внимание к мёртвому слоту
        def undo():                                  # остаётся — консервативно:
            obj.concept_vals.data[idx].copy_(row)    # занижает gain, не завышает
        return undo

    # ─── замер (вызывается из train.py рядом с RegulatorLedger) ───
    @torch.no_grad()
    def measure(self, model, probe, batch, step: int, tokens_per_step: int):
        try:
            from .regulator_ledger import RegulatorLedger as _RL
        except ImportError:                      # standalone-использование
            from regulator_ledger import RegulatorLedger as _RL
        out = {}
        for e in self.entries:
            if e['retired'] or step - e['step'] < self.horizon:
                continue
            snap = model.snapshot_runtime_buffers()
            ex = _RL._probe_state(model)
            undo = None
            try:
                ce_on = probe(*batch)              # ВСЕ новорождённые активны
                _RL._probe_restore(model, ex)
                model.restore_runtime_buffers(snap)
                undo = self._excise(model, e)      # изъять новорождённого i
                if undo is None:
                    continue                       # направление не найдено — пропуск
                ce_off = probe(*batch)
                undo(); undo = None
            finally:
                if undo is not None:
                    undo()                         # страховка от исключения
                _RL._probe_restore(model, ex)
                model.restore_runtime_buffers(snap)
            n_tok = (step - e['step']) * tokens_per_step
            v = self.mdl(ce_off - ce_on, n_tok, e['r'])
            e['verdicts'].append(round(v, 1))
            if v < 0: e['fails'] += 1
            else:     e['fails'] = 0
            if e['fails'] >= self.dwell:
                e['retired'] = True
                self.blacklist.append(dict(d=e['d'], until=step + self.bl_steps))
                self._retire(model, e)
            out[e['kind']] = out.get(e['kind'], 0) + 1
        return out

    def _retire(self, model, e: dict) -> None:
        obj, idx = self._locate(model, e)
        if obj is None:
            return
        if e['kind'] == 'phantom_bit':
            obj.phantom_mix.data[:, idx].zero_()   # канал навсегда выключен
            obj.phantom_basis.data[idx].zero_()
            e['slot'] = idx                        # ёмкость переиспользуема (см. хук)
        else:
            obj.concept_vals.data[idx].zero_(); obj.concept_keys.data[idx].zero_()
            obj.concept_count[idx] = 0; obj.concept_confidence[idx] = 0   # слот снова «пуст»

    def free_rows(self, kind: str) -> List[int]:
        return [e.get('slot', -1) for e in self.entries
                if e['retired'] and e['kind'] == kind and e.get('slot', -1) >= 0]

    def stats(self) -> dict:
        return dict(births=len(self.entries),
                    retired=sum(1 for e in self.entries if e['retired']),
                    blacklisted=len(self.blacklist))

    def state_dict(self) -> dict:
        return dict(entries=[{**e, 'd': e['d'].tolist()} for e in self.entries],
                    blacklist=[{'d': b['d'].tolist(), 'until': b['until']}
                               for b in self.blacklist])

    def load_state_dict(self, sd: Optional[dict]) -> None:
        if not sd: return
        self.entries = [{**e, 'd': torch.tensor(e['d'])} for e in sd.get('entries', [])]
        self.blacklist = [{'d': torch.tensor(b['d']), 'until': b['until']}
                          for b in sd.get('blacklist', [])]
```

**Хуки в `core/stack.py`** (точные якоря: линк (A) ~строка 432, линк (C) ~строка 437):

```python
# линк (A) — рождение UCL-концепта из подтверждённого фантома:
if _sim < 0.9 and (_birth_ledger is None or _birth_ledger.allow_birth(_d, _st)):
    if _ucl.birth_from_direction(_d, confidence=0.6):
        if _birth_ledger is not None:
            # слот фиксируется В МОМЕНТ рождения (до дрейфа записей):
            _dn = torch.nn.functional.normalize(_d.detach().float().reshape(-1), dim=-1)
            _slot = int((torch.nn.functional.normalize(
                _ucl.concept_vals.detach().float(), dim=-1) @ _dn).abs().argmax())
            _birth_ledger.record('ucl', _d, _st,
                                 r=_ucl.D + _ucl.bridge_dim, slot=_slot)

# линк (C) — рост фантом-канала (строки известны точно — передаём индекс):
_prev_kp = int(_head._kp_active.item())
_grew = _head.grow_phantom_bits(_dirs)
if _grew and _birth_ledger is not None:
    for _j in range(_prev_kp, _prev_kp + _grew):
        _birth_ledger.record('phantom_bit', _head.phantom_basis.data[_j], _st,
                             r=_head.D + _head.K, slot=_j)
```

`_birth_ledger` живёт на стеке (атрибут, не подмодуль — как `_ga_record`);
train.py создаёт его, кладёт `state['birth_ledger'] = _birth_ledger.state_dict()`
в чекпойнт и вызывает `measure(...)` в тот же eval-блок, что и RegulatorLedger
(один snapshot на оба — экономия).

**Опционально (переиспользование ёмкости):** в `grow_phantom_bits` перед добавлением
консультировать `ledger.free_rows('phantom_bit')` — рождение в освобождённую строку
вместо хвоста ёмкости. Без этого retirement просто гасит канал (тоже корректно).

**Тест** (`tests/test_p3_birth_ledger.py`):
1. **Арифметика MDL**: `mdl(delta=0.046, n_tok=768_000, r=2624) ≈ 0` (порог окупаемости
   фантом-бита ≈ 46 мнат/токен — зафиксировать число тестом, как вы фиксируете τ).
2. **Синтетика**: два новорождённых — «полезный» (вручную поднять его вклад:
   заполнить mix-столбец так, чтобы он снижал CE probe-партии) и случайный;
   после horizon полезный ACTIVE (verdict > 0), случайный retired + в блэклисте;
   `allow_birth(тот же вектор)` → False до cooldown.
3. **Обратимость**: после measure() все изъятые столбцы/строки восстановлены
   побитово (кроме осознанно retired).

**Приёмка:** на живом прогоне в телеметрии появляется строка
`births=N retired=M blacklisted=K` и медианный verdict; «рождение параметров»
впервые получает бухгалтерию: сколько рождено, сколько окупилось, какова цена.

---

## P3-3. DualWriterCensus + двухвременна́я проверка

**Суть.** «Метакогнитивность неотделима от параметров» формально означает: часть
параметров имеет **двух писателей** — оптимизатор (быстрый внутренний контур) и
контрольный закон (медленный внешний). Это схема двухвременной стохастической
аппроксимации (Borkar): разделима и устойчива, когда `τ_control ≥ SEP·τ_adam`,
SEP=10, τ_adam = 1/(1−β₁) = 10 шагов ⇒ **τ_control ≥ 100 шагов**. Реестр +
AST-сканер + тест разделения масштафов — в стиле `test_tau_lint` (новый писатель
без регистрации ⇒ красный тест).

**Новый файл:** `core/param_writers.py`

```python
"""core/param_writers.py — P3-3: реестр двойных писателей параметров.

Категории:
  optimizer-only   — обычный параметр (реестр не нужен);
  dual-writer      — оптимизатор + контрольный закон (.data-мутация);
                     условие разделения: tau_ctrl >= SEPARATION * TAU_ADAM;
  quasi-static     — редкие событийные записи (birth/index_copy) вне lerp-класса:
                     tau=None + обязательное обоснование.
Буферы (private_mem, concept_* до birth, bank keys/vals) — НЕ параметры:
их писатели покрыты snapshot/restore (M8), в census не входят.
"""
TAU_ADAM = 10.0        # 1/(1-0.9)
SEPARATION = 10.0

DUAL_WRITERS = {
    # leaf-имя параметра: (писатель, tau_ctrl шагов, обоснование/заметки)
    'b_i':            ('AdaptiveController→stack.forward lerp', 1000.0,
                       'vsa_b_d_smooth=0.999; Adam-сторона замедлена vsa_b_lr_mult=0.1'),
    'b_d':            ('AdaptiveController→stack.forward lerp', 1000.0, 'то же'),
    'alpha_diag':     ('mirror self-regulation (pend/flush, 1/step)', 100.0,
                       'lerp 0.01; B13 F4-01: ровно одна запись на шаг'),
    'phantom_basis':  ('EMA-steering confirmed directions', 770.0,
                       '0.01 на observe, observe каждые 100 head-forward (~12 шагов)'),
    'concept_keys':   ('functional write + birth_from_direction', None,
                       'квазистатика: редкие события, не lerp-класс'),
    'concept_vals':   ('functional write + birth_from_direction', None, 'то же'),
}
```

**Тест** (`tests/test_dual_writer_census.py`):

```python
import ast, pathlib
from core.param_writers import DUAL_WRITERS, TAU_ADAM, SEPARATION

MUTATORS = {'copy_', 'lerp_', 'add_', 'mul_', 'fill_', 'zero_', 'sub_', 'clamp_'}
PARAM_LEAVES = set(DUAL_WRITERS) | {'W_out', 'W_proj', 'readout', 'basis',
                                    'embed_mix', 'log_scale', 'log_temp'}

def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr); node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return '.'.join(reversed(parts))

def _scan():
    hits = []
    for p in sorted(pathlib.Path(__file__).parent.parent.joinpath('core').glob('*.py')):
        tree = ast.parse(p.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            tgt = None
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in MUTATORS):
                tgt = n.func.value
            elif isinstance(n, ast.Assign):
                tgt = next((t for t in n.targets if isinstance(t, ast.Subscript)), None)
            if tgt is None:
                continue
            ch = _chain(tgt)
            if '.data' not in ch:
                continue
            leaf = ch.split('.data')[0].split('.')[-1]
            if leaf in PARAM_LEAVES:
                hits.append((p.name, n.lineno, ch, leaf))
    return hits

def test_every_dual_writer_registered():
    unreg = [(f, ln, ch) for f, ln, ch, leaf in _scan()
             if leaf not in DUAL_WRITERS]
    assert not unreg, (f'незарегистрированные писатели параметров '
                       f'(зарегистрируйте в core/param_writers.py или уберите): {unreg}')

def test_timescale_separation():
    for leaf, (writer, tau, note) in DUAL_WRITERS.items():
        if tau is None:
            assert note, f'{leaf}: квазистатика без обоснования'
            continue
        assert tau >= SEPARATION * TAU_ADAM, \
            f'{leaf}: tau_ctrl={tau} < {SEPARATION * TAU_ADAM} (двухвременна́я схема не разделима)'

def test_control_writers_alive():
    """Runtime-канарейка: медленный контур действительно пишет (при живом оптимизаторе)."""
    model = _small_model(); model.train()
    b0 = model.layers[0].b_d.detach().clone()
    x = torch.randint(3, model.cfg.vocab, (1, 16))
    h = model.embed_tokens(x)
    model(h, None, step=1, tokens=x, adaptive=True)
    model(h, None, step=2, tokens=x, adaptive=True)
    assert not torch.equal(b0, model.layers[0].b_d.detach()), 'b_d-лерп мёртв'
```

Замечание для кодера: `_tie_hook` (bind) и `_sync_W_out` (mirror) пишут в `W_out.data` —
после **P0-2** первый исчезает, второй пишет в буфер (не параметр) и попадёт в скан;
в этом случае тест-лок ожидает его в разделе «buffer-writers» реестра (добавить
`BUFFER_WRITERS = {'W_out': ('tie sync hook', 'значения, не градиент; P0-2')}`
и пропускать leaf'ы из него с пометкой).

**Приёмка:** тесты зелёные; в `docs/WHITEBOARD.md` таблица двойных писателей
(6 записей v1) с τ_ctrl — формальное доказательство, что «неотделимость от
параметров» является разделимой двухвременной схемой, а не гонкой двух писателей.

---

## P3-4. ParamVelocity — KPI эффективности параметров

**Суть.** Постоянная телеметрия «кто из параметров реально двигается и что это
стоит»: относительное перемещение ‖Δθ‖/‖θ‖ по модулям за log_interval, без
полноразмерных копий (детерминированная подвыборка координат, бюджет ~7MB на
production-модели). Ваш замер M64.7 («readout сдвинулся на −1.9% за 7315 шагов»)
был ручным постмортемом — это та же метрика, поставленная на поток.

**Новый файл:** `core/param_velocity.py`

```python
from __future__ import annotations
import torch
from typing import Dict


class ParamVelocity:
    """‖Δθ‖/‖θ‖ по модулям через детерминированную подвыборку координат.

    sample() вызывается раз в log_interval (после optimizer.step()).
    Память: frac·numel·4B (1% от 175M ≈ 7MB). Индексы сидированы — воспроизводимо.
    Агрегаты: head / embed / mirror / mlp / bind / bridge / cache / reasoning / trunk
    + отдельная строка newborn (phantom_mix, pair_V*, UCL out_proj) — прямо отвечает
    на вопрос «учатся ли рождённые параметры».
    """
    def __init__(self, model, frac: float = 0.01, min_coords: int = 64, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        self.idx: Dict[str, torch.Tensor] = {}
        self.snap: Dict[str, torch.Tensor] = {}
        self.vel: Dict[str, float] = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            k = max(min_coords, int(p.numel() * frac))
            self.idx[n] = torch.randperm(p.numel(), generator=g)[:min(k, p.numel())]

    def sample(self, model, ema: float = 0.8) -> None:
        for n, p in model.named_parameters():
            i = self.idx.get(n)
            if i is None:
                continue
            v = p.detach().reshape(-1)[i.to(p.device)]
            if n in self.snap:
                den = float(self.snap[n].norm())
                # near-zero-init параметры: относительная скорость не имеет смысла
                # (тот же eps-класс, что пропуск AGC при ‖θ‖<1e-3; на смоук-тесте
                # trunk/bridge давали 1e4 именно на этом) — пропускаем.
                if den > 1e-6:
                    rel = float((v - self.snap[n]).norm()) / den
                    self.vel[n] = rel if n not in self.vel \
                        else ema * self.vel[n] + (1 - ema) * rel
            self.snap[n] = v.clone()

    @staticmethod
    def _bucket(n: str) -> str:
        if n.startswith('lm_head'):   return 'head'
        if n.startswith('embed'):     return 'embed'
        if '.mirror.' in n:           return 'mirror'
        if '.mlp.' in n:              return 'mlp'
        if '.bind.' in n:             return 'bind'
        if 'bridge' in n or 'intent' in n or 'bus_head' in n: return 'bridge'
        if 'logit_cache' in n:        return 'cache'
        if 'reasoning' in n:          return 'reasoning'
        if 'concept_layer' in n:      return 'ucl'
        return 'trunk'

    def report(self) -> Dict[str, float]:
        agg: Dict[str, list] = {}
        for n, v in self.vel.items():
            agg.setdefault(self._bucket(n), []).append(v)
        return {k: round(sum(v) / len(v), 6) for k, v in sorted(agg.items())}
```

**Интеграция:** `pvel = ParamVelocity(model)` при сборке; `pvel.sample(model)` после
`optimizer.step()` раз в `cfg.log_interval`; строка лога `pvel head=… mirror=… trunk=…`;
`state['pvel'] = {'vel': pvel.vel}` в чекпойнт (индексы детерминированы — не возим).

**Метрика-производная для whiteboard:** «эффективность параметров» =
Δ(ce_raw за интервал) / Σ_buckets vel·numel — падение CE на единицу относительного
перемещения параметрической массы. Рост этого отношения при P3-1/2/3 — прямое
подтверждение вашего ожидания «более эффективное использование параметров».

**Тест:** на малой модели два sample() вокруг одного шага SGD: vel > 0 у всех
групп с ненулевым lr; замороженная группа (set_active_depth) даёт vel = 0.

**Приёмка:** память < 10MB на production; числа воспроизводимы между запусками
(сидированные индексы); строка pvel в логе с первого же интервала.

---

## Порядок применения и A/B-дисциплина

```
Неделя 1:  P0-1, P0-3, P0-4, P2-2        (безопасно, поведение не меняют*)
Неделя 2:  P0-2 (tie_grad-флаг, A/B)     (*кроме tied-режимов: там градиент и должен измениться)
Неделя 3:  P0-5 (stream-parity тест — главный лок)
Неделя 4:  P0-6 (переписать generate.py; сравнить AR-CE до/после на hold-out)
Параллельно: P1-1 (диагностика потолка — ничего не меняет, решает судьбу P1-2/P1-3)
Параллельно: P3-3, P3-4 (тесты/телеметрия — поведение не меняют вовсе)
После P0:  P3-1 (леджер регуляторов — сначала как телеметрия, пенсии только через A/B)
           P3-2 (birth ledger — включается вместе с линками M59)
           P2-1 и только потом — содержательные A/B надстроек на чистом стволе.
```

Каждый P0-патч меняет **режим**, а не инициализацию: свежие чекпойнты не нужны,
но замеры val до/после обязательны (особенно P0-5/P0-6: ожидается падение AR-CE
при неизменном оконном CE — это и есть доказательство, что фикс попал в цель).
P3-патчи не меняют обучение вообще (два getattr-флага в block.py default-True,
всё остальное — no_grad-измерения): их можно включать на живом прогоне в любой момент.

Общий принцип всех правок — ваш же: zero-init/identity там, где меняется forward,
явный флаг там, где меняется обучение, тест-лок на каждый инвариант.

