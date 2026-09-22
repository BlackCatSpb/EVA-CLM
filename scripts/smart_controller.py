"""
smart_controller.py — экспериментальный «умный инференс» для EVA.

SmartController сам подбирает параметры генерации ПЕРЕТОКЕНОВО, опираясь на
собственные сигналы модели:
  - энтропия головы H           -> неопределённость / где модель «не уверена»
  - mirror.debug_mind():         -> метакогнитивные сигналы («понимает себя»)
      trust_max      — уверенность модели в своём выводе
      gate_ema_mean  — вовлечённость экспертов
  - недавние токены              -> детектор повторений
  - скользящая энтропия         -> детектор «коллапса» (зацикливание)
  - tau-зависимости             -> темпоральная «личность» модели (VSA-таймскейлы)

Режимы: exploit / explore / confused / reason / recover-rep / recover-collapse.
Используется из scripts/generate.py (флаг --smart) и scripts/smart_infer.py.
"""
import os, sys, math, torch
import torch.nn.functional as F
from core import EVAStack


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# скалярные метакогнитивные ключи, по которым усредняем по слоям (с весом tau)
SCALAR_KEYS = {'trust_max', 'gate_ema_mean', 'w_help', 'private_mem_norm', 'ls_var_mean'}


class SmartController:
    def __init__(self, model, vocab, reasoning_on=True, no_trunc=False):
        self.reasoning_on = reasoning_on
        self.vocab = vocab
        self.no_trunc = no_trunc
        self.recent = []
        self.hist = []
        self.recov = 0
        self.decisions = []
        # --- tau-зависимости: темпоральная «личность» модели ---
        self._compute_tau(model)
        # M40 (architecture-fit): normalization of generation temperature
        # against the head's LEARNED log_temp — without it, base_temp means
        # different things at different training stages (the coded head sums
        # K=64 bit-log-odds, its raw scale drifts as the ladder and log_temp
        # train). Mirrors generate.py AdaptiveSampler.norm_log_temp.
        _lt = getattr(getattr(model, 'lm_head', None), 'log_temp', None)
        self._temp_norm = _lt is not None
        self._temp_ref = float(_lt.detach().mean().item()) if self._temp_norm else 0.0
        # M40: RELATIVE entropy — the absolute Hn=H/ln(V) thresholds were
        # calibrated for a softmax head. Under SigmoidCodedHead the coded
        # logits are sums of Bernoulli log-odds: H sits at ~1-4 nats ALWAYS
        # (Hn<0.4), so mode/top_k/temp were frozen in 'exploit' and top-k
        # never actually adapted — the reported «top-k не отсекается». Now
        # H is judged against its own EMA, exactly as this module's docstring
        # always promised.
        self._H_ema = None
        # настраиваемые пороги (модулируются tau)
        self.temp_lo, self.temp_hi = 0.45, 1.35
        self.p_lo, self.p_hi = 0.82, 0.96
        self.rep_base, self.rep_max = 2.0, 5.0
        self.rep_window = 8
        self.rep_ngram = 3
        self.alarm_window = 16
        self.trust_thr = 0.25
        self.collapse_H = 0.6
        self.reason_thr = 0.6
        # модуляция tau: длинная память -> холоднее + реже reasoning (выше порог самосомнения)
        bias = (0.5 - self.tau_norm) * 0.5
        self.temp_lo += bias
        self.temp_hi += bias
        self.trust_thr = min(max(0.25 + self.tau_norm * 0.5, 0.1), 0.95)
        self.alarm_window_base = self.alarm_window

    def _compute_tau(self, model):
        """M40: the per-layer profile comes from the UNIFIED tau field
        (tau_config.tau_l) — single source, like everything else. The old
        local re-implementation duplicated a retired ladder formula
        (tmin*(tmax/tmin)^(lf*(1+0.1*tanh(dev)))) — a parallel τ-subsystem
        of its own (math-audit E4 territory)."""
        tc = getattr(model, 'tau_config', None)
        n = len(model.layers)
        if tc is not None and getattr(tc, 'tau_l', None) is not None:
            self.tau_l_vec = [float(x) for x in tc.tau_l.detach().cpu()]
            tau_n = float(tc.tau_norm.detach().mean())
        else:  # mock/fallback: geometric ladder over the VSA base
            vsa = torch.exp(torch.cumsum(F.softplus(model._vsa_log_param), 0)) + 1.0
            tmin, tmax = vsa[0].item(), vsa[-1].item()
            self.tau_l_vec = [tmin * (tmax / tmin) ** (i / max(n - 1, 1))
                             for i in range(n)]
            tau_n = 0.5
        self.tau_personality = sum(self.tau_l_vec) / max(len(self.tau_l_vec), 1)
        self.tau_norm = min(max(tau_n, 0.0), 1.0)

    def _entropy(self, logits):
        p = torch.softmax(logits.float(), -1)
        p = torch.clamp(p, min=1e-12)  # avoid 0*log(0)=NaN on overconfident (hot) logits
        return float(-(p * p.log()).sum().item())

    def _repetition(self):
        toks = self.recent[-self.rep_window:]
        n = self.rep_ngram
        if len(toks) < n * 2:
            return False
        last = tuple(toks[-n:])
        cnt = sum(1 for i in range(len(toks) - n + 1) if tuple(toks[i:i + n]) == last)
        return cnt >= 2

    def _rep_pressure(self):
        """Непрерывный сигнал зацикливания: доля токенов в окне, повторяющихся
        ранее (0 = чисто, 1 = сплошные повторы). Окно = адаптивный alarm_window."""
        look = self.recent[-self.alarm_window:]
        if len(look) < 4:
            return 0.0
        cnt = {}
        for t in look:
            cnt[t] = cnt.get(t, 0) + 1
        dup = sum(v - 1 for v in cnt.values() if v > 1)
        return min(dup / len(look), 1.0)

    def decide(self, logits, mind, step):
        H = self._entropy(logits)
        # M40: Hn is the entropy RELATIVE to its own running EMA (see init).
        # r=2 -> Hn~0.73 (explore); r=0.5 -> Hn~0.2 (exploit); r=1 -> ~0.52.
        if self._H_ema is None:
            self._H_ema = max(H, 1e-3)
        _r = H / self._H_ema
        Hn = 1.0 / (1.0 + math.exp(-3.0 * (math.log(max(_r, 1e-6)) - 0.15)))
        self._H_ema = 0.92 * self._H_ema + 0.08 * max(H, 1e-3)
        trust = mind.get('trust_max', 0.5)
        self.hist.append(H)
        if len(self.hist) > 64:
            self.hist = self.hist[-32:]
        # collapse = sustained DEEP drop below the model's own baseline, not a
        # WideBind-era absolute constant (collapse_H kept as a fallback floor):
        collapse = len(self.hist) >= 4 and (
            max(self.hist[-4:]) < min(self.collapse_H, 0.45 * self._H_ema))
        rep = self._repetition()

        h = smooth(Hn)
        temp = lerp(self.temp_lo, self.temp_hi, h)
        top_p = lerp(self.p_hi, self.p_lo, h)   # уверен -> уже (exploit)
        # адаптивный top_k: уверен -> сужаем (фокус), неуверен -> ядро (nucleus)
        top_k = 0 if Hn > 0.55 else int(round(lerp(50, 12, h)))
        reason = 0.0
        mode = 'exploit' if Hn < 0.4 else ('explore' if Hn < 0.7 else 'confused')

        # --- непрерывно-адаптивные штрафы и окна (penalties etc.) ---
        loop_p = self._rep_pressure()                          # доля РЕАЛЬНЫХ повторов
        # давление = повторы + неуверенность (низкий trust) + высокая энтропия
        pressure = min(max(loop_p, 0.5 * (1.0 - trust), 0.45 * Hn), 1.0)
        rep_pen = lerp(self.rep_base, self.rep_max, smooth(pressure))
        # окно и n-грамма штрафа растут с давлением: короткое/униграммное когда
        # чисто, длинное/триграммное когда ловим петли
        self.rep_window = int(round(lerp(6, 18, smooth(pressure))))
        self.rep_ngram = 1 + int(round(smooth(pressure) * 2))  # 1..3
        # адаптивное окно «тревоги»: дальше смотрим назад при давлении
        self.alarm_window = int(round(lerp(8, 24, smooth(pressure))))
        # адаптивный bias_alpha: снимаем learned prior под реальными петлями/неуверенностью
        alpha = min(max(1.0 - 1.5 * loop_p - 0.2 * (1.0 - trust), 0.0), 1.0)

        if self.reasoning_on and trust < self.trust_thr:
            reason = 1.0
            mode = 'reason'

        if rep:
            rep_pen = max(rep_pen, self.rep_base + 2.5)
            top_k = 50
            self.rep_window = max(self.rep_window, 12)
            self.rep_ngram = 3
            self.alarm_window = max(self.alarm_window, 16)
            reason = 1.0 if self.reasoning_on else reason
            mode = 'recover-rep'

        if collapse:
            temp = min(self.temp_hi + 0.4, temp + 0.5)
            top_k = 40
            self.rep_window = 18
            self.rep_ngram = 3
            self.alarm_window = 24
            mode = 'recover-collapse'

        if self.recov > 0:
            self.recov -= 1
            temp = max(temp, 1.15)
            rep_pen = max(rep_pen, self.rep_base + 1.5)
            reason = 1.0 if self.reasoning_on else reason

        if rep or collapse:
            self.recov = 3

        if self.no_trunc:
            top_p = 1.0
            top_k = 0

        self.model_reason_override = reason
        self.decisions.append((step, mode, round(H, 2), round(trust, 2),
                               round(temp, 2), round(top_p, 2), int(top_k),
                               round(rep_pen, 2), int(reason),
                               int(self.rep_window), int(self.rep_ngram),
                               round(alpha, 2), int(self.alarm_window)))
        return temp, top_p, top_k, rep_pen, alpha

    def sample(self, logits, temp, top_p, top_k, rep_pen):
        logits = logits.clone()
        # M40: base_temp means the SAME thing at any training stage once the
        # head's learned bit-temperature scale divides it out (AdaptiveSampler
        # parity); exp(clamp) guards degenerate log_temp.
        if self._temp_norm:
            temp = max(temp, 1e-3) / max(math.exp(min(self._temp_ref, 10.0)), 1e-3)
        # order-preserving window (audit M10): set() iterated in arbitrary
        # order — the penalty hit random tokens instead of the recent ones
        for rid in self.recent[-self.rep_window:]:
            logits[rid] -= rep_pen
        if temp != 1.0:
            logits = logits / temp
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.numel()))
            logits[logits < v[-1:]] = -float('inf')
        if top_p < 1.0:
            s = torch.sort(logits, descending=True)[0]
            cum = torch.cumsum(torch.softmax(s, -1), -1)
            mask = cum <= top_p
            cut = s[mask][-1:] if mask.any() else s[0:1]
            logits[logits < cut] = -float('inf')
        probs = F.softmax(logits, -1)
        return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def smart_generate(model, prompt, controller, max_new_tokens=64, rep_window=8,
                   set_reason=True, no_trunc=False, allow_write=None,
                   context_mem=None, reset_reasoning=False, base_step=0):
    """Генерация под управлением SmartController. set_reason=True -> переключает
    model.reasoning_scale_override перетокеново (по решению контроллера).
    allow_write=True -> разрешает записи в memory bank. context_mem -> внешний контекст."""
    from scripts.generate import load_russian_tokenizer
    controller.no_trunc = no_trunc
    model.eval()
    # P0-6 (F1b): prefill + incremental L=1 decode — the SAME pattern as
    # generate() and the training loop. The old loop re-fed the whole sliding
    # window with the carried state, so VSA/bank/UCL/cache received ~L writes
    # per token (the math audit measured the slow VSA scale x3.52 at L=8; at
    # L=384 proportionally worse) and the generating model was NOT the function
    # validated on the hold-out. This smart path had missed the fix.
    for _l in model.layers:
        _mm = getattr(_l, 'mirror', None)
        if _mm is not None:
            _mm._ar_mode = True
    if getattr(model, 'embed', None) is not None:
        model.embed._ar_mode = True
    if hasattr(model, '_last_salience'):
        model._last_salience = None
    tok = load_russian_tokenizer()
    det = lambda ids: tok.decode(ids, skip_special_tokens=True)
    ids = tok.encode(prompt).ids
    # T9.11: границы предложений как в обучении (SEP id=2)
    if not ids or ids[-1] != 2:
        ids = ids + [2]
    device = next(model.parameters()).device
    tokens = torch.tensor(ids, dtype=torch.long, device=device)
    L = model.cfg.seq_len
    state = None
    gs = None            # T9.11: кросс-слойный self-model EMA между шагами
    intent_state = None
    rb = None
    head = model.lm_head
    tb = getattr(head, 'token_bias', None)

    out_ids = list(ids)
    n = len(model.layers)

    # Prefill: the prompt as ONE window; the recurrent state starts cold.
    ctx = tokens[-L:].unsqueeze(0)
    h = model.embed_tokens(ctx)
    out, state, gs, rb = model(h, None, global_state=None, adaptive=False,
                               context_mem=context_mem, allow_write=allow_write,
                               step=base_step, intent_state=None, tokens=ctx)
    intent_state = getattr(model, '_last_intent_state', None)  # T9.11
    model.observe_output(head(out))   # salience of the prompt's last token

    for step in range(max_new_tokens):
        if reset_reasoning:
            model.reset_reasoning()
            rb = None
        logits = head(out[:, -1:, :])[0, 0]
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            state = None
            rb = None

        # tau-weighted trust_max aggregated from all layers WITHOUT per-layer
        # host syncs: debug_mind() calls .item() ~10x/layer, which under
        # no_grad generation stalls the GPU. meta_signals() returns tensors.
        trust_ts = []
        gate_ts = []
        for layer in model.layers:
            t, g = layer.mirror.meta_signals()
            trust_ts.append(t)
            gate_ts.append(g)
        trust_stack = torch.stack(trust_ts)
        gate_stack = torch.stack(gate_ts)
        wvec = torch.tensor(controller.tau_l_vec, dtype=trust_stack.dtype,
                            device=trust_stack.device)
        wsum = wvec.sum()
        trust_val = ((trust_stack * wvec).sum() / wsum).item() if wsum > 0 else 0.5
        if not math.isfinite(trust_val):
            trust_val = 0.5
        mind = {
            'trust_max': trust_val,
            'gate_ema_mean': gate_stack[-1].item(),
        }

        temp, top_p, top_k, rep_pen, alpha = controller.decide(logits, mind, step)
        if set_reason:
            model.reasoning_scale_override = controller.model_reason_override
        if tb is not None:
            logits = (logits - tb) + alpha * tb   # адаптивный bias_alpha
        nt = controller.sample(logits, temp, top_p, top_k, rep_pen)
        controller.recent.append(nt)
        max_recent = max(controller.rep_window + controller.rep_ngram,
                         controller.alarm_window)
        if len(controller.recent) > max_recent:
            controller.recent = controller.recent[-max_recent:]
        out_ids.append(nt)
        tokens = torch.cat([tokens, torch.tensor([nt], dtype=torch.long, device=device)])

        # P0-6: incremental decode — ONE new token per step, the state carried.
        tok1 = torch.tensor([[nt]], dtype=torch.long, device=device)
        h1 = model.embed_tokens(tok1)
        out, state, gs, rb = model(h1, state, global_state=gs, adaptive=False,
                                   context_mem=context_mem, allow_write=allow_write,
                                   step=base_step + step + 1,
                                   intent_state=intent_state,
                                   reasoning_buffer=rb[0] if rb is not None else None,
                                   reasoning_count=rb[1] if rb is not None else None,
                                   tokens=tok1)
        intent_state = getattr(model, '_last_intent_state', None)  # T9.11
        model.observe_output(head(out))  # salience of THIS step -> next intent
    return det(out_ids), controller.decisions
