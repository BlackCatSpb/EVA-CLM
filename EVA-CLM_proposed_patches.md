# EVA-CLM — предлагаемые патчи (P0/P1/P2)

Источник: внешний математический аудит (`EVA-CLM_math_report.md`, commit 3bccfc2).
Все патчи следуют доктрине проекта: **identity-at-init / checkpoint-safe / A-B-рука / тест-лок**.
Каждый патч независим; порядок применения — по приоритету. В конце каждого — критерий приёмки.

Легенда приоритетов:
- **P0** — корректность: чинит измеренные режимные/численные дефекты живых путей.
- **P1** — потолок качества: расширяет выразительность головы и даёт инструмент измерения потолка.
- **P2** — научная чистота: минимальная абляция + baseline для атрибуции тезиса.

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

## Порядок применения и A/B-дисциплина

```
Неделя 1:  P0-1, P0-3, P0-4, P2-2        (безопасно, поведение не меняют*)
Неделя 2:  P0-2 (tie_grad-флаг, A/B)     (*кроме tied-режимов: там градиент и должен измениться)
Неделя 3:  P0-5 (stream-parity тест — главный лок)
Неделя 4:  P0-6 (переписать generate.py; сравнить AR-CE до/после на hold-out)
Параллельно: P1-1 (диагностика потолка — ничего не меняет, решает судьбу P1-2/P1-3)
После P0:  P2-1 и только потом — содержательные A/B надстроек на чистом стволе.
```

Каждый P0-патч меняет **режим**, а не инициализацию: свежие чекпойнты не нужны,
но замеры val до/после обязательны (особенно P0-5/P0-6: ожидается падение AR-CE
при неизменном оконном CE — это и есть доказательство, что фикс попал в цель).

Общий принцип всех правок — ваш же: zero-init/identity там, где меняется forward,
явный флаг там, где меняется обучение, тест-лок на каждый инвариант.
