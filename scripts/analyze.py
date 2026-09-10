"""Единый анализатор чекпоинтов EVA.

Объединяет все методы анализа в один прогон:
  - static    — конфиг, per-layer параметры, VSA-лестница, зеркало, MLP, голова, NaN/Inf
  - inspector — сигналы (temp/pred/smooth/sym/help), trust/concept/dominance (private memory)
  - wake      — вердикт PASS/WATCH/WAKE (пробуждение MLP по маркерам watchlist)
  - live      — forward на случайном входе: hp/predMSE/|mirror|/gate/ls/skip, сигналы, параметры
  - anomaly   — трекер аномалий: max ||hp||/predMSE/min gate/births + дельты к прошлому чекпоинту
  - gradinfo  — dead_pred (pred_loss без градиента из-за detach в _pred_cache) + cos_sim(diversity, CE)
  - head      — декомпозиция bias vs контекст + позиционная карта головы + A/B reasoning
  - cmp       — сравнение нескольких чекпоинтов на одинаковом входе

Usage:
  python scripts/analyze.py <ckpt...> [--no-live] [--no-head] [--quick]
                                       [--windows N] [--prompt TEXT] [--temp T]

  Один чекпоинт — полный разбор; несколько — разбор каждого + сравнение.
"""
import sys, os, math, re, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

import argparse
import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals

sys.path.insert(0, BASE)
from core import EVAConfig, EVAStack
from core.config import WideBindConfig
add_safe_globals([EVAConfig, WideBindConfig])

PUNCT = re.compile(r'^[\s\.,:;!?\-—–…"«»()\[\]{}]+$')
WORD = re.compile(r'[а-яёА-ЯЁ]')


# ─────────────────────────── загрузка ───────────────────────────

def load_ckpt(path):
    """Потоковая загрузка с минимальным пиком RAM.

    torch.load возвращает ЧЕКПОИНТ-словник целиком: стейт-дикт модели
    (~212M params ≈ 0.9GB) + optimizer (m+v ≈ 1.7GB) + scheduler. Если
    держать всё одновременно с моделью (~6.7GB с буферами), пик ~16GB —
    это OOM на машинах с малым RAM. Здесь:
      1) optimizer/scheduler/param_names выбрасываются СРАЗУ после load,
      2) стейт-дикт копируется в модель поэлементно и освобождается по ходу
         (пик во время копирования ~ модель + один тензор вместо 2 дублей).
    weights_only=False (M12): конверт best.pt несёт cfg-датакласс и
    dict-состояния стража/балансировщика; на torch<2.4 нет самого kwarg.
    """
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    cfg = ckpt['cfg']
    # M12-конверт проверяется ДО того, как тяжёлые части выброшены ради RAM:
    _sch = ckpt.pop('scheduler', None)
    _opt = ckpt.pop('optimizer', None) or {}
    _pn = ckpt.pop('param_names', None) or []
    ckpt['_sched_present'] = _sch is not None
    ckpt['_opt_groups'] = len(_opt.get('param_groups', []))
    ckpt['_opt_slots'] = len(_opt.get('state', {}))
    ckpt['_param_names_n'] = len(_pn)
    del _sch, _opt, _pn
    sd = ckpt.pop('model') or {}
    ckpt['_n_tensors'] = len(sd)
    sd_keys = set(sd.keys())
    model = EVAStack(cfg)
    params_d = dict(model.named_parameters())
    buffers_d = dict(model.named_buffers())
    with torch.no_grad():
        for k in list(sd.keys()):
            v = sd.pop(k)
            if k in params_d:
                d = params_d[k]
                if d.shape == v.shape:
                    d.copy_(v)
                else:
                    print(f'  [warn] shape-mismatch {k}: ckpt={list(v.shape)} model={list(d.shape)}')
            elif k in buffers_d:
                b = buffers_d[k]
                if b.shape == v.shape:
                    b.copy_(v)
            del v
    model_keys = set(model.state_dict().keys())  # persistent-буферы; не-persistent не «missing»
    missing = sorted(model_keys - sd_keys)
    unexpected = sorted(sd_keys - model_keys)
    model.train()
    if model.explicit_reasoning:
        model.reasoning_enabled_step = int(ckpt.get('reasoning_enabled_step', 0))
    return ckpt, cfg, model, missing, unexpected


def sec(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


# ───────────────────────── ENVELOPE (M12/M13) ─────────────────────────

def run_envelope(ckpt):
    """Конверт единственного чекпоинта: best.pt обязан нести ВСЁ состояние
    перезапуска (M12) — иначе resume это не resume, а холодный старт."""
    sec('ENVELOPE (M12 single-checkpoint)')
    step = ckpt.get('step', '?')
    if ckpt.get('seed') and step == 0:
        print('  SEED (step 0, model-only) — цель отката до первого EVAL; '
              'для resume не годится. Ждём первую eval-запись.')
        return {'seed': True}
    need = ['step', 'model', 'optimizer', 'param_names', 'scheduler', 'best_val_loss',
            'cfg', 'reasoning_enabled_step', 'recover_count', 'active_depth',
            'detector', 'balancer', 'stream_idx', 'offset', 'rng', 'data_rng']
    # load_ckpt streaming-pop выбрасывает тяжёлые части; восстанавливаем факт их
    # наличия из метаданных, иначе они вечно ложно падают как MISSING.
    present = {k for k in ckpt if not k.startswith('_')}
    if ckpt.get('_n_tensors'):
        present.add('model')
    if ckpt.get('_sched_present'):
        present.add('scheduler')
    if ckpt.get('_opt_groups') or ckpt.get('_opt_slots'):
        present.add('optimizer')
    if ckpt.get('_param_names_n'):
        present.add('param_names')
    miss = [k for k in need if k not in present]
    extra = [k for k in present if k not in need]
    d = ckpt.get('detector') or {}
    b = ckpt.get('balancer') or {}
    print(f'  step={step}  best_val={ckpt.get("best_val_loss", float("nan")):.4f}  '
          f'depth={ckpt.get("active_depth")}  reasoning_ramp={ckpt.get("reasoning_enabled_step")}')
    print(f'  data cursor: stream_idx={ckpt.get("stream_idx")} offset={ckpt.get("offset")}')
    print(f'  detector: recover={d.get("recover_count", ckpt.get("recover_count"))} '
          f'armed={d.get("ce_armed")} baselines={sorted((d.get("stats") or {}).keys())} '
          f'viol={d.get("viol")}')
    print(f'  balancer: ema_ce={b.get("ema_ce")} align={b.get("align")}')
    print(f'  rng: cpu={("rng" in present)} data_gen={("data_rng" in present)}  '
          f'optimizer: groups={ckpt.get("_opt_groups", 0)} slots={ckpt.get("_opt_slots", 0)} '
          f'(model tensors={ckpt.get("_n_tensors", 0)})')
    if miss:
        print('  [MISSING] ' + ', '.join(map(str, miss)))
        print('   -> пред-M12 чекпоинт: resume холодный в недостающих частях '
              '(страж пере-бустрапится, курсор данных/val-эпоха начнутся заново)')
    if extra:
        print('  extra keys (ok): ' + ', '.join(map(str, extra)))
    print('  VERDICT: ' + ('полный побитовый resume возможен' if not miss else 'частичный resume'))
    return {'seed': False, 'missing': miss}


# ───────────────────────── MECHANISMS (post-M1..M13) ─────────────────────────

def run_mech(model, cfg):
    """Живой инвентарь механизмов, которые аудиты оживляли: состояние
    init-гейтов и обученных управляющих весов по состоянию чекпоинта."""
    import math
    sec('MECHANISMS (cache / UCL / mirror-signals / head / tau-ladder)')
    out = {}
    lc = getattr(model, 'logit_cache', None)
    if lc is None:
        print('  logit_cache: DISABLEN конфигом (logit_cache_enabled=False)')
    else:
        a = lc.attention
        gb = float(a.cache_gate[-2].bias.detach())
        gate = 1.0 / (1.0 + math.exp(-gb))
        status = ('identity — CE кэш ещё не спрашивает' if gate < 1e-3 else
                  'приоткрыт' if gate < 0.05 else 'АКТИВНО востребован')
        at = float(a.log_tau.detach().exp().clamp(0.1, 10.0))
        vs = torch.sigmoid(lc.cache.vsa_scales.detach()).tolist()
        print(f'  cache: windows={lc.cache.max_entries} gate_bias={gb:+.2f} (σ={gate:.5f}) -> {status}')
        print(f'         attn τ={at:.2f}; inference k-фракции шкал={[round(x, 2) for x in vs]}')
        out['cache'] = {'gate': gate, 'bias': gb}
    ucl = getattr(model, 'concept_layer', None)
    if ucl is not None:
        nb = int(ucl._n_births); nu = int(ucl._n_updates); ns = int(ucl._n_skipped)
        S = int(ucl.concept_count.numel()); used = int((ucl.concept_count > 0).sum())
        thr = float(torch.sigmoid(ucl._log_tau_birth_thr.detach()))
        nov = float(torch.sigmoid(ucl._log_tau_novelty_thr.detach()))
        conf = float(ucl.concept_confidence.max())
        step = int(ucl._step)
        warn = '  ⚠ не рождал концептов за 5k+ шагов — novelty ниже порога?' if (nb == 0 and step > 5000) else ''
        print(f'  UCL: births={nb} updates={nu} skipped={ns} занято={used}/{S} '
              f'step={step}; thr σ(birth)={thr:.3f} σ(novelty)={nov:.3f} max_conf={conf:.3f}{warn}')
        out['ucl'] = {'births': nb, 'updates': nu, 'used': used, 'step': step}
    n_sig = int(model.layers[0].mirror._signal_log_weights.numel())
    hs = []
    for layer in model.layers:
        w = torch.sigmoid(layer.mirror._signal_log_weights.detach())
        p = w / (w.sum() + 1e-10)
        hs.append(float(-(p * (p + 1e-10).log()).sum()))
    hmax = math.log(n_sig)
    print(f'  mirror signal entropy (n_sig={n_sig}, max={hmax:.3f}): '
          f'avg={sum(hs) / len(hs):.3f} min={min(hs):.3f} max={max(hs):.3f}')
    if min(hs) < 0.35 * hmax:
        print('   ⚠ near-collapse на каком-то слое (когда-то +H-знак вёл именно туда; M5)')
    out['mirror_h'] = hs
    head = getattr(model, 'lm_head', None)
    tb = getattr(head, 'token_bias', None); bb = getattr(head, 'bit_bias', None)
    if tb is not None:
        print(f'  head: token_bias rms={float(tb.detach().pow(2).mean().sqrt()):.4f} '
              f'bit_bias std={float(bb.detach().std()):.4f} (приор {float(bb.detach().mean()):+.3f})')
    dev = model.tau_config._tau_dev.detach()
    print(f'  τ-ladder dev: |max|={float(dev.abs().max()):.4f} '
          f'[{float(dev.min()):+.4f}, {float(dev.max()):+.4f}] (лестница ещё прямая = мало движения)')
    return out


# ──────────────────────────── HEALTH (log rules) ───────────────────────────

def run_health(data):
    """Детерминированные вердикты по логам — то, что при разборе смотрели
    глазами (наклон branch/bridge_conn, направление signal_ent, жизнь intent,
    healthy-band diversity, ползучесть mem)."""
    import statistics as _stt
    sec('HEALTH RULES')
    steps = data['steps']
    if not steps:
        print('  ⚠ main-строки не найдены — формат лога не распознан '
              '(проверь intent_eff/55-интервал)')
        return
    main = data['main']; aux = data['aux']
    def med(x):
        x = list(x)
        return _stt.median(x) if x else float('nan')
    n = len(steps); q = max(1, n // 4)
    res = []
    ce = main.get('ce', [])
    res.append(('ce без NaN/inf', all(v == v and abs(v) < 1e30 for v in ce)))
    br = aux.get('branch', [])
    if br:
        res.append(('branch затухает (factorized head)', med(br[-q:]) < med(br[:q])))
    bc = aux.get('bridge_conn', [])
    if bc:
        res.append(('bridge_conn падает (мост предсказывает)', med(bc[-q:]) < med(bc[:q])))
    se = aux.get('signal_ent', [])
    if se:
        m = med(se)
        res.append((f'signal_ent в −H-шкале (~{m:.2f})', -1.75 < m < -0.5))
    gl = aux.get('gate_l1', [])
    if gl:
        res.append(('gate_l1 редеет/стабилен', med(gl[-q:]) <= med(gl[:q]) * 1.05))
    dv = aux.get('diversity', [])
    if dv:
        res.append(('diversity в healthy band (<5.0)', max(dv) < 5.0))
    ie = main.get('intent_w', [])
    if ie:
        res.append((f'intent-шина жива (late med={med(ie[n - q:]):.4f})', med(ie[n - q:]) > 0))
    mem = main.get('mem', [])
    if len(mem) > 4:   # creep = медиана последних четверти против второй (первые
        q = max(1, len(mem)//4)   # ~10% — штатный разогрев: кэш, Adam-слоты)
        _d = med(mem[-q:]) - med(mem[q:2*q])
        res.append((f'mem без ползучего роста (Δплато={_d:+.2f}GB)', _d < 1.5))
    us = main.get('usef', [])
    if steps[-1] > 5000 and len(us) > q:
        res.append(('usef разошёлся после 5k', _stt.pstdev(us[-q:]) > 1e-3))
    ok_all = True
    for name, ok in res:
        ok_all = ok_all and ok
        print(f'  {"✓" if ok else "⚠"} {name}')
    ev = data['eval']
    if ev:
        vals = [v for _, v, _ in ev]
        arrow = '↓' if vals[-1] <= vals[0] else '⚠ нет снижения'
        print(f'  eval: {len(vals)} точек | first={vals[0]:.4f} last={vals[-1]:.4f} {arrow}')
    lr = main.get('lr', []); tk = main.get('tok_s', [])
    print(f'  steps={n} | med tok/s={med(tk):.0f} | lr∈[{min(lr):.2e}..{max(lr):.2e}]'
          if lr else f'  steps={n}')
    print('  VERDICT: ' + ('здорово' if ok_all else 'есть ⚠ — смотреть вручную'))


# ─────────────────────────── STATIC ───────────────────────────

def run_static(ckpt, cfg, model, missing, unexpected, tok=None):
    st = {'config': [], 'param_groups': [], 'per_layer': [], 'vsatau': {},
          'mirror': [], 'lm_head': [], 'tensors': {}, 'reasoning': None,
          'inspector': None, 'file': os.path.basename(ckpt.get('_path', '?')),
          'step': ckpt.get('step', '?'),
          'best_val': float(ckpt.get('best_val_loss', float('inf'))),
          'missing': [k for k in missing[:10]], 'unexpected': [k for k in unexpected[:10]]}
    sec('STATIC')
    print(f'File:       {st["file"]}')
    print(f'Step:       {st["step"]}')
    print(f'Best val:   {st["best_val"]}')
    print(f'Params:     {model.param_count() / 1e6:.2f}M')
    print(f'Tensors:    {ckpt.get("_n_tensors", "-")}')
    print(f'Missing:    {len(missing)}   Unexpected: {len(unexpected)}')
    if missing:
        for k in missing[:5]:
            print(f'  MISS: {k}')
    if unexpected:
        for k in unexpected[:5]:
            print(f'  UNEXP: {k}')

    imp = ['D', 'n_layers', 'G', 'k', 'vocab', 'seq_len', 'mlp_groups', 'mlp_expand',
           'bind_K', 'mirror_k', 'lr', 'weight_decay', 'explicit_reasoning',
           'reasoning_max_steps', 'reasoning_adaptive', 'variable_precision',
           'collective_read_out', 'private_mem', 'head_mode', 'bind_twist_mode']
    print('\nCONFIG:')
    for a in imp:
        v = getattr(cfg, a, 'N/A')
        st['config'].append((a, v))
        print(f'  {a:24s} = {v}')

    print('\nPARAM GROUPS (LR multipliers):')
    for g in model.param_groups():
        lr = g.get('lr', cfg.lr)
        wd = g.get('weight_decay', 0)
        n = len(g['params'])
        st['param_groups'].append({'lr': lr, 'wd': wd, 'n': n})
        print(f'  lr={lr:.6f} wd={wd:.4f} n_params={n:4d}')

    print('\nPER-LAYER (params, без forward):')
    hdr = f'{"L":>3s} {"alpha":>7s} {"|1-a|":>7s} {"ls_std":>7s} {"w_help":>7s} ' \
          f'{"skip":>7s} {"scale_w":>7s} {"conv_std":>8s} {"pm_norm":>8s}'
    print(hdr)
    print('-' * len(hdr))
    for i, layer in enumerate(model.layers):
        m = layer.mirror
        a = m.alpha_diag.data
        ls = m.log_scale.data
        wh = torch.sigmoid(m.w_help.data)
        lsa = torch.sigmoid(m.log_skip_alpha.data)
        sw = torch.sigmoid(layer.scale_w.data)
        cw = layer.conv.weight.data
        pm = m._private_mem.norm(dim=-1).mean().item() if m._has_private_mem else float('nan')
        st['per_layer'].append({'layer': i, 'alpha': a.mean().item(), 'a1': (1 - a).abs().mean().item(),
                                'ls_std': ls.std().item(), 'w_help': wh.mean().item(),
                                'skip': lsa.mean().item(), 'scale_w': sw.mean().item(),
                                'conv_std': cw.std().item(), 'pm_norm': pm})
        print(f'{i:3d} {a.mean().item():7.4f} {(1 - a).abs().mean().item():7.4f} '
              f'{ls.std().item():7.4f} {wh.mean().item():7.4f} {lsa.mean().item():7.4f} '
              f'{sw.mean().item():7.4f} {cw.std().item():8.4f} {pm:8.3f}')

    vsa_log = model._vsa_log_param
    vsa_tau = torch.exp(torch.cumsum(F.softplus(vsa_log), dim=0)) + 1.0
    td = model._tau_l_dev.data
    b_d = model.b_d.data if hasattr(model, 'b_d') else None
    b_i = model.b_i.data if hasattr(model, 'b_i') else None
    st['vsatau'] = {'tau0': float(vsa_tau[0].item()), 'taun': float(vsa_tau[-1].item()),
                    'ratio': float(vsa_tau[-1].item() / vsa_tau[0].item()),
                    'tau_l_dev': float(td.mean().item()), 'tau_l_std': float(td.std().item()),
                    'b_d': (float(b_d.mean().item()), float(b_d.min().item()), float(b_d.max().item())) if b_d is not None else None,
                    'b_i': (float(b_i.mean().item()), float(b_i.min().item()), float(b_i.max().item())) if b_i is not None else None}
    print(f'\nVSA TAU: tau[0]={vsa_tau[0].item():.2f}  tau[-1]={vsa_tau[-1].item():.2f}  '
          f'ratio={vsa_tau[-1].item() / vsa_tau[0].item():.1f}x  '
          f'tau_l_dev={td.mean().item():.4f} (std {td.std().item():.4f})')
    # _tau_intent_dev removed: tau_config.intent_alpha is the single source of truth
    if b_d is not None:
        print(f'  b_d: mean={b_d.mean().item():.4f} range=[{b_d.min().item():.4f}, {b_d.max().item():.4f}]')
    if b_i is not None:
        print(f'  b_i: mean={b_i.mean().item():.4f} range=[{b_i.min().item():.4f}, {b_i.max().item():.4f}]')

    print('\nMIRROR INTERNALS (L0, mid, last) — ВСЕ параметры зеркала:')
    for i in (0, len(model.layers) // 2, len(model.layers) - 1):
        m = model.layers[i].mirror
        lrow = {'layer': i, 'params': []}
        print(f'  L{i}:')
        for attr, t in m.named_parameters():
            t = t.data
            stdv = 0.0 if t.numel() <= 1 else t.std().item()
            info = f'shape={list(t.shape)} mean={t.mean().item():.4f} std={stdv:.4f}'
            if t.numel() <= 12:
                info += f' vals={[round(v, 3) for v in t.flatten().tolist()]}'
            elif attr == 'tanh_bias':
                info += f' min={t.min().item():.4f} max={t.max().item():.4f}'
            lrow['params'].append({'name': attr, 'shape': list(t.shape),
                                   'mean': float(t.mean().item()), 'std': float(stdv),
                                   'min': float(t.min().item()), 'max': float(t.max().item())})
            print(f'    {attr:18s}: {info}')
        st['mirror'].append(lrow)

    lm = getattr(model, 'lm_head', None)
    if lm is not None:
        print('\nLM HEAD:')
        for name, p in lm.named_parameters():
            d = p.data
            st['lm_head'].append({'name': name, 'shape': list(d.shape),
                                  'mean': float(d.mean().item()), 'std': float(d.std().item())})
            print(f'  {name:24s}: shape={list(d.shape)} mean={d.mean().item():.6f} '
                  f'std={d.std().item():.6f}')

    params = list(model.parameters())
    numel = 0
    s_sum = torch.zeros((), dtype=torch.float64)
    s_sq = torch.zeros((), dtype=torch.float64)
    t_min = float('inf')
    t_max = float('-inf')
    n_nan = n_inf = 0
    sample = []
    SAMPLE_CAP = 2_000_000
    cap_per = max(1, SAMPLE_CAP // max(len(params), 1))
    for p in params:
        d = p.data
        f = d.detach().float().flatten()
        n = f.numel()
        if n == 0:
            continue
        numel += n
        s_sum += f.sum(dtype=torch.float64)
        s_sq += (f * f).sum(dtype=torch.float64)
        if numel - n == 0:
            t_min = float(d.min())
            t_max = float(d.max())
        else:
            t_min = min(t_min, float(d.min()))
            t_max = max(t_max, float(d.max()))
        n_nan += int(torch.isnan(d).sum())
        n_inf += int(torch.isinf(d).sum())
        stride = max(1, n // cap_per)
        sample.append(f[::stride] if stride > 1 else f)
    mean = (s_sum / numel).item()
    std = max(float(s_sq / numel) - mean * mean, 0.0) ** 0.5
    st['tensors'] = {'numel': numel, 'mean': mean, 'std': std, 'min': t_min, 'max': t_max,
                     'nan': n_nan, 'inf': n_inf, 'quantiles': []}
    print(f'\nTENSORS: {numel} scalars | mean={mean:.6f} '
          f'std={std:.6f} | min={t_min:.6f} '
          f'max={t_max:.6f}')
    q = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])
    try:
        all_vals = torch.cat(sample)
        quants = torch.quantile(all_vals, q)
        st['tensors']['quantiles'] = [(round(qi * 100), float(qv)) for qi, qv in zip(q.tolist(), quants.tolist())]
        print('  quantiles: ' + '  '.join(f'Q{qi * 100:3.0f}={qv:.6f}'
                                           for qi, qv in zip(q.tolist(), quants.tolist())))
    except Exception:
        pass
    print(f'  NaN: {n_nan}  Inf: {n_inf}')

    print('\nALL PARAMETERS (grouped по шаблону имени — ВСЕ, вкл. новые):')
    for g in _grouped_param_stats(model):
        print(f'  {g["name"]:46s} {str(g["shape"]):24s} n={g["n"]:9d} '
              f'mean={g["mean"]:+.5f} std={g["std"]:+.5f} [{g["min"]:+.4f}, {g["max"]:+.4f}]')

    if model.explicit_reasoning:
        g = getattr(model, '_reasoning_gates', None)
        st['reasoning'] = {'enabled_step': model.reasoning_enabled_step,
                           'scale': float(model.reasoning_scale),
                           'gates': [round(float(x), 4) for x in g] if g is not None else []}
        print(f'\nREASONING: enabled_step={model.reasoning_enabled_step} '
              f'scale={model.reasoning_scale:.4f}  last gates={g}')

    if model.layers[0].mirror._has_private_mem:
        st['inspector'] = run_inspector(model)

    return st


# ─────────────────────────── INSPECTOR (private memory) ───────────────────────────

def run_inspector(model):
    m0 = model.layers[0].mirror
    w = torch.softmax(m0._signal_log_weights, dim=0)
    sig = {'temp': float(w[0].item()), 'pred': float(w[1].item()),
           'smooth': float(w[2].item()), 'sym': float(w[3].item()),
           'help': float(w[4].item())}
    insp = {'signals': sig, 'w_help': float(torch.sigmoid(m0.w_help).mean().item()),
            'w_contra': float(m0.w_contra.mean().item()),
            'concept_sim': None, 'behavior_div': None, 'trust': None,
            'dominance': None, 'isolation': None, 'contra_expert': None,
            'pm_step': None}
    print('\nINSPECTOR (private memory):')
    print(f'  signals: ' + '  '.join(f'{n}={v:.3f}' for n, v in sig.items()))
    print(f'  w_help(sigmoid)={torch.sigmoid(m0.w_help).mean().item():.3f}  '
          f'w_contra={m0.w_contra.mean().item():.4f}')
    if m0._concept_sim_ema is not None:
        cs = m0._concept_sim_ema
        insp['concept_sim'] = {'mean': float(cs.mean().item()), 'std': float(cs.std().item()),
                               'diag': float(cs.diag().mean().item())}
        print(f'  concept_sim: mean={cs.mean().item():.4f} std={cs.std().item():.4f} '
              f'diag={cs.diag().mean().item():.4f}')
    if m0._behavior_div_ema is not None:
        insp['behavior_div'] = float(m0._behavior_div_ema.mean().item())
        print(f'  behavior_div: {m0._behavior_div_ema.mean().item():.4f}')
    if m0._trust_matrix is not None:
        tr = m0._trust_matrix
        insp['trust'] = {'mean': float(tr.mean().item()), 'diag': float(tr.diag().mean().item())}
        print(f'  trust: mean={tr.mean().item():.4f} diag={tr.diag().mean().item():.4f}')
    if m0._cached_dominance is not None:
        insp['dominance'] = [round(float(x), 3) for x in m0._cached_dominance.tolist()]
        print(f'  dominance: {[round(x, 3) for x in m0._cached_dominance.tolist()]}')
    if m0._cached_isolation is not None:
        insp['isolation'] = [round(float(x), 3) for x in m0._cached_isolation.tolist()]
        print(f'  isolation: {[round(x, 3) for x in m0._cached_isolation.tolist()]}')
    if m0._cached_contra_expert is not None:
        insp['contra_expert'] = [round(float(x), 3) for x in m0._cached_contra_expert.tolist()]
        print(f'  contra_expert: {[round(x, 3) for x in m0._cached_contra_expert.tolist()]}')
    if hasattr(m0, '_pm_write_delay') and hasattr(m0, '_pm_step'):
        insp['pm_step'] = int(m0._pm_step.item())
        print(f'  pm_step: {int(m0._pm_step.item())}/{m0._pm_write_delay}')
    return insp


def _grouped_param_stats(model):
    """Сводка по ВСЕМ параметрам модели, сгруппированная по шаблону имени (без номера слоя).
    Гарантирует, что ни один параметр — в т.ч. новые (w_sal, _tau_intent_dev, ...) — не потеряется."""
    groups = {}
    for name, p in model.named_parameters():
        key = re.sub(r'layers\.(\d+)\.', 'layers.L.', name)
        t = p.data
        n2 = t.numel()
        mean2 = float(t.mean().item())
        var2 = 0.0 if n2 <= 1 else float(t.var().item())
        mn2 = float(t.min().item())
        mx2 = float(t.max().item())
        if key not in groups:
            groups[key] = {'shape': list(t.shape), 'n': n2, 'mean': mean2,
                           'M2': var2 * (n2 - 1), 'min': mn2, 'max': mx2}
        else:
            g = groups[key]
            n1, m1, M2a = g['n'], g['mean'], g['M2']
            n = n1 + n2
            delta = mean2 - m1
            m = m1 + delta * n2 / n
            M2 = M2a + var2 * (n2 - 1) + (delta * delta) * n1 * n2 / n
            g['n'], g['mean'], g['M2'] = n, m, M2
            g['min'] = min(g['min'], mn2)
            g['max'] = max(g['max'], mx2)
    out = []
    for key, g in groups.items():
        std = math.sqrt(g['M2'] / (g['n'] - 1)) if g['n'] > 1 else 0.0
        out.append({'name': key, 'shape': g['shape'], 'n': g['n'],
                    'mean': g['mean'], 'std': std, 'min': g['min'], 'max': g['max']})
    out.sort(key=lambda d: d['name'])
    return out


# ─────────────────────────── WAKE ───────────────────────────

REF_STEP = 1398
W_STD_REF = 0.0705
DECAY_RATE = 1.6e-6
EMPTY_SLOT_LAYERS = set()  # dynamic: populated from model at runtime
SIG = {'PASS': 'PASS ', 'WATCH': 'WATCH', 'WAKE': 'WAKE '}


def _verdict(report, flag, hits, text):
    report.append(f'  [{SIG[flag]}] {text}')
    for h in hits:
        report.append(f'        {h}')


def run_wake(model, ckpt):
    step = int(ckpt.get('step', 0))
    n_layers = len(model.layers)
    report = [f'Wake-up scan: step={step} layers={n_layers}']

    mlp_wstd, gate_mlp, gate_mem, slot_occ, temp_vals, mat_counts, birth_gates = [], [], [], {}, [], [], []
    for i, layer in enumerate(model.layers):
        mlp = layer.mlp
        if hasattr(mlp, 'W_up'):
            mlp_wstd.append(mlp.W_up.data.std().item())
        elif hasattr(mlp, 'W_gate'):
            mlp_wstd.append(mlp.W_gate.data.std().item())
        mir = layer.mirror
        if hasattr(mir, 'mod_scale_mlp'):
            gate_mlp.append(torch.sigmoid(mir.mod_scale_mlp.data).mean().item())
        if hasattr(mir, 'mod_scale_mem'):
            gate_mem.append(torch.sigmoid(mir.mod_scale_mem.data).mean().item())
        cl = getattr(layer, 'collective', None)
        if cl is not None:
            slot_occ[i] = int((cl.N_s.data > 0).sum().item())
            if hasattr(cl, '_temp') and cl._temp.ndim == 0:
                temp_vals.append(cl._temp.data.item())
            mat = getattr(cl, '_mature_count', None)
            if mat is not None:
                mat_counts.append(mat)
            bg = cl.birth_gate_mean()
            birth_gates.append(bg.item() if hasattr(bg, 'item') else float(bg))

    wstd = sum(mlp_wstd) / len(mlp_wstd)
    expected = W_STD_REF * torch.exp(torch.tensor(-DECAY_RATE * (step - REF_STEP))).item()
    dev = wstd - expected
    report.append(f'  MLP W_std mean={wstd:.4f} decay-expected={expected:.4f} dev={dev:+.4f}')
    _verdict(report, 'WAKE' if dev > 0.001 else ('WATCH' if dev > 0.0005 else 'PASS'), [],
             f'MLP W_std vs decay curve (marker #1, dev {dev:+.4f})')

    per_layer = sorted(((i, mlp_wstd[i] - expected) for i in range(n_layers)), key=lambda t: -t[1])
    top = per_layer[:3]
    _verdict(report, 'WAKE' if top[0][1] > 0.002 else ('WATCH' if top[0][1] > 0.001 else 'PASS'),
             [f'L{li}: W_std dev {d:+.5f}' for li, d in top],
             f'per-layer W_std deviation (marker #1b, worst {top[0][1]:+.5f})')

    g_max = sorted(((i, torch.sigmoid(l.mirror.mod_scale_mlp.data).max().item())
                    for i, l in enumerate(model.layers)), key=lambda t: -t[1])
    gtop = g_max[:3]
    _verdict(report, 'WAKE' if gtop[0][1] > 0.75 else ('WATCH' if gtop[0][1] > 0.72 else 'PASS'),
             [f'L{li}: gate max {v:.3f}' for li, v in gtop],
             f'per-layer gate max (marker #2b, worst {gtop[0][1]:.3f}, wake >0.75)')

    g_mlp = sum(gate_mlp) / len(gate_mlp)
    g_mem = sum(gate_mem) / len(gate_mem) if gate_mem else float('nan')
    report.append(f'  sigmoid(mod_scale_mlp) mean={g_mlp:.3f}  '
                  f'sigmoid(mod_scale_mem) mean={g_mem:.3f}')
    _verdict(report, 'WAKE' if g_mlp > 0.75 else ('WATCH' if g_mlp > 0.72 else 'PASS'), [],
             f'modulation gate (marker #2, >0.75 = WAKE, baseline 0.668)')
    if abs(g_mlp - g_mem) > 0.2:
        _verdict(report, 'WAKE', [], f'mlp/mem gate divergence {abs(g_mlp - g_mem):.2f}')

    if birth_gates:
        bmean = sum(birth_gates) / len(birth_gates)
        bmax = max(birth_gates)
        report.append(f'  concept-birth gate mean={bmean:.4f} max={bmax:.4f}')
        _verdict(report, 'PASS' if bmean > 1e-4 else 'WATCH',
                 [f'L{i}: {bg:.4f}' for i, bg in sorted(enumerate(birth_gates),
                                                          key=lambda t: -t[1])[:3]],
                 f'concept-birth (режим Б): mean {bmean:.4f} (спит если ~0)')

    empty_layers = {i for i, o in slot_occ.items() if o == 0}
    births = [i for i in empty_layers if i in slot_occ and slot_occ[i] > 0]
    full = [i for i, o in slot_occ.items() if o >= 8]
    report.append(f'  slots occupied: {sum(slot_occ.values())}/192, full layers: {len(full)}, '
                  f'empty layers: {sorted(empty_layers)}, births in empty: {births}')
    _verdict(report, 'WAKE' if births else 'PASS', [],
             f'slot births in empty layers: {births or "none"}')
    if temp_vals:
        t_mean = sum(temp_vals) / len(temp_vals)
        report.append(f'  _temp mean={t_mean:.3f}')
        if t_mean < 1.95:
            _verdict(report, 'WATCH', [], f'_temp dropped from 2.0 baseline ({t_mean:.3f})')

    if mat_counts:
        n_zero = sum(1 for c in mat_counts if c == 0)
        report.append(f'  _mature_count: {n_zero}/{len(mat_counts)} layers locked (0)')
        if n_zero < len(mat_counts) * 0.5:
            _verdict(report, 'WATCH', [], 'maturity lock released in >half layers')

    lm = getattr(model, 'lm_head', None)
    if lm is not None:
        log_temp = None
        for name, p in lm.named_parameters():
            if 'log_temp' in name:
                log_temp = p.data
        if log_temp is not None:
            report.append(f'  log_temp mean={log_temp.mean().item():+.4f} (ref +0.0188 at 1398)')
            if log_temp.mean().item() > 0.25:
                _verdict(report, 'WATCH', [], 'log_temp rising (softmax flattening)')

    mat = getattr(model, 'maturation', None)
    mat_info = None
    if mat is not None:
        g = mat.gate
        gl = g.tolist()
        gmin, gmax = min(gl), max(gl)
        gmean = sum(gl) / len(gl)
        rmax = float(mat.readiness.max().item()) if hasattr(mat, 'readiness') else 0.0
        report.append(f'  MATURATION gate: min={gmin:.4f} max={gmax:.4f} mean={gmean:.4f} '
                      f'(ramp T0={mat.T0:.0f} delta={mat.delta_t:.0f} T_delay={mat.T_delay:.0f})')
        report.append(f'  readiness max={rmax:.4f}  tau_norm max={mat.tau_norm.max().item():.3f}')
        if gmax < 0.05:
            _verdict(report, 'WATCH', [],
                     f'maturation gates still ~closed (max {gmax:.3f}) — bridge not engaged yet')
        else:
            _verdict(report, 'PASS', [],
                     f'maturation gate opening (max {gmax:.3f}) — bridge engaging')
        mat_info = {'gate': gl, 'gmin': gmin, 'gmax': gmax, 'gmean': gmean,
                    'readiness_max': rmax, 'tau_norm_max': float(mat.tau_norm.max().item()),
                    'T0': mat.T0, 'delta': mat.delta_t, 'T_delay': mat.T_delay}

    hdr = 'OK' if not any(r.startswith('  [WAKE') for r in report) else 'WAKE-CANDIDATE'
    print(f'\nWAKE DETECTOR (verdict: {hdr})')
    for line in report:
        print(line)

    return {
        'verdict': hdr,
        'report': report,
        'step': step,
        'mlp_wstd': mlp_wstd,
        'gate_mlp_mean': gate_mlp,
        'gate_max_layers': [i for i, _ in g_max],
        'gate_max': [v for _, v in g_max],
        'dev_layers': [i for i, _ in per_layer],
        'dev': [d for _, d in per_layer],
        'wstd': wstd, 'expected': expected, 'dev_mean': dev,
        'g_mlp': g_mlp, 'g_mem': g_mem,
        'slots': sum(slot_occ.values()), 'full': len(full), 'births': births,
        'temp_mean': (sum(temp_vals) / len(temp_vals)) if temp_vals else None,
        'mat_locked': n_zero if mat_counts else None,
        'mat': mat_info,
    }


# ─────────────────────────── LIVE ───────────────────────────

@torch.no_grad()
def run_live(model, cfg, batch=1, seq=128, gradinfo=True):
    device = 'cpu'
    model.to(device)
    model.train()
    x = torch.randint(0, cfg.vocab, (batch, seq), device=device)
    h = model.embed_tokens(x)
    h_out, _, _, _ = model(h, tokens=x)

    sec('LIVE (forward на случайном входе)')
    print(f'INPUT/OUTPUT:  in_norm={h.norm(dim=-1).mean().item():.3f}  '
          f'out_norm={h_out.norm(dim=-1).mean().item():.3f}  '
          f'out_std={h_out.std().item():.4f}')
    tgt = torch.randint(1, cfg.vocab, (batch, seq), device=device)
    ls, aux = model.compute_losses(h_out[:, :-1], tgt[:, 1:])
    print(f'CE(random)={ls.item():.3f}  pred={aux["pred"]:.3f}  '
          f'gate_l1={aux["gate_l1"]:.5f}  balance={aux["balance"]:.5f}')

    print()
    hdr = f'{"L":>3s} {"||hp||":>8s} {"predMSE":>8s} {"|mirror|":>9s} {"gate":>7s} ' \
          f'{"gate_var":>9s} {"|1-a|":>7s} {"ls_expM":>8s} {"ls_std":>7s} {"skip":>6s} ' \
          f'{"w_delta":>7s}'
    print(hdr)
    print('-' * len(hdr))
    rows = []
    for i, layer in enumerate(model.layers):
        m = layer.mirror
        hp = m._cached_hp
        pk = m._cached_pred_k
        rows.append({
            'layer': i, 'hp': hp.norm(dim=-1).mean().item(),
            'predMSE': F.mse_loss(pk, hp.detach()).item(),
            'mirror': m._last_magnitude.item(),
            'gate': m._cached_gate_l1.item(),
            'gate_var': m._last_gates.var().item(),
            'a1': (1 - m.alpha_diag.data).abs().mean().item(),
            'ls_expM': m.log_scale.data.exp().max().item(),
            'ls_std': m.log_scale.data.std().item(),
            'skip': math.exp(m.log_skip_alpha.data.mean().item()),
            'w_delta': m.w_delta_gate.data.abs().mean().item(),
        })
        print(f'{i:3d} {hp.norm(dim=-1).mean().item():8.3f} '
              f'{F.mse_loss(pk, hp.detach()).item():8.3f} '
              f'{m._last_magnitude.item():9.3f} {m._cached_gate_l1.item():7.5f} '
              f'{m._last_gates.var().item():9.5f} '
              f'{(1 - m.alpha_diag.data).abs().mean().item():7.4f} '
              f'{m.log_scale.data.exp().max().item():8.2f} '
              f'{m.log_scale.data.std().item():7.4f} '
              f'{math.exp(m.log_skip_alpha.data.mean().item()):6.3f} '
              f'{m.w_delta_gate.data.abs().mean().item():7.5f}')

    print('\nSIGNALS (sigmoid + softmax share) L0, mid, last:')
    names = ['temp', 'pred', 'smooth', 'sym'] + (
        ['help'] if model.layers[0].mirror._has_private_mem else [])
    signals = []
    for i in (0, len(model.layers) // 2, len(model.layers) - 1):
        m = model.layers[i].mirror
        w = torch.sigmoid(m._signal_log_weights)
        p = torch.softmax(m._signal_log_weights, dim=0)
        signals.append({'layer': i, 'sigmoid': w.tolist(), 'share': p.tolist(),
                        'names': names})
        print(f'  L{i}: sigmoid={["%.3f" % v for v in w.tolist()]}  '
              f'share={["%.3f" % v for v in p.tolist()]}  names={names}')

    print('\nMIRROR PARAMS L0, mid, last:')
    mirror_rows = []
    for i in (0, len(model.layers) // 2, len(model.layers) - 1):
        m = model.layers[i].mirror
        row = {
            'layer': i,
            'alpha_diag': (m.alpha_diag.data.mean().item(), m.alpha_diag.data.min().item(),
                           m.alpha_diag.data.max().item()),
            'log_scale': (m.log_scale.data.mean().item(), m.log_scale.data.max().item()),
            'tanh_bias': (m.tanh_bias.data.mean().item(), m.tanh_bias.data.max().item()),
            'b_gate': m.b_gate.data.mean().item(),
            'gate_bias': m.gate_bias.data.mean().item(),
            'mod_mlp': torch.sigmoid(m.mod_scale_mlp.data).mean().item(),
            'mod_mem': torch.sigmoid(m.mod_scale_mem.data).mean().item(),
            'gate_ema': m._gate_ema.data.mean().item(),
            'w_sal': torch.sigmoid(m.w_sal.data).mean().item() if hasattr(m, 'w_sal') else None,
            'w_intent': torch.sigmoid(m.w_intent.data).mean().item() if hasattr(m, 'w_intent') else None,
        }
        if m._has_private_mem:
            row['w_help'] = torch.sigmoid(m.w_help.data).mean().item()
            row['w_contra'] = m.w_contra.data.mean().item()
            row['pm_norm'] = m._private_mem.norm(dim=-1).mean().item()
        mirror_rows.append(row)

    vsa_log = model._vsa_log_param
    vsa_tau = torch.exp(torch.cumsum(F.softplus(vsa_log), dim=0)) + 1.0
    td = model._tau_l_dev.data
    print(f'\nGLOBAL: vsa tau=[{vsa_tau[0].item():.2f}, {vsa_tau[-1].item():.2f}]  '
          f'tau_l_dev={td.mean().item():.4f} (std {td.std().item():.4f})')
    pm = getattr(model, '_private_mem', None)
    if pm is not None:
        print(f'  model private_mem norm: {pm.norm(dim=-1).mean().item():.3f}')

    grad_info = None
    if gradinfo:
        try:
            grad_info = run_grad_info(model, cfg, seq=min(seq, 16))
        except Exception as e:
            print(f'  [warn] grad info: {e}')

    usef_vals = [l.mirror._cached_usefulness.mean().item() for l in model.layers
                 if hasattr(l.mirror, '_cached_usefulness')]
    return {
        'in_norm': h.norm(dim=-1).mean().item(),
        'out_norm': h_out.norm(dim=-1).mean().item(),
        'out_std': h_out.std().item(),
        'ce_random': ls.item(), 'pred': aux['pred'], 'gate_l1': aux['gate_l1'],
        'balance': aux['balance'],
        'usef': (sum(usef_vals) / len(usef_vals)) if usef_vals else float('nan'),
        'rows': rows, 'signals': signals, 'mirror': mirror_rows,
        'vsa_tau': [vsa_tau[0].item(), vsa_tau[-1].item()],
        'tau_l_dev': td.mean().item(), 'tau_l_dev_std': td.std().item(),
        'pm_norm': pm.norm(dim=-1).mean().item() if pm is not None else None,
        'grad_info': grad_info,
    }


# ─────────────────────────── HEAD ───────────────────────────

def _head_forward(model, window, tok):
    with torch.no_grad():
        h = model.embed_tokens(window.unsqueeze(0))
        out, _, _, _ = model(h, None, adaptive=False)
        return model.lm_head(out[0])


def _cat(tid, tok):
    tstr = tok.decode([int(tid)]).strip()
    return 'punct' if PUNCT.match(tstr) else ('word' if WORD.search(tstr) else 'other')


def run_head(model, ckpt, args, tok):
    sec('HEAD: декомпозиция bias vs контекст + позиционная карта + A/B reasoning')
    model.eval()
    text = 'Москва — столица России, и в ней живут миллионы людей. ' * 60
    ids = tok.encode(text).ids
    L = model.cfg.seq_len
    n_w = max(args.windows, 1)
    log_temp = model.lm_head.log_temp.data.mean().item()
    t_eff = args.temp / math.exp(log_temp)
    tb = model.lm_head.token_bias.data
    tb_argmax = int(tb.argmax())
    print(f'  log_temp={log_temp:.4f} -> t_eff={t_eff:.4f} | '
          f'token_bias top: {[repr(tok.decode([int(i)])) for i in torch.topk(tb, 5).indices.tolist()]}')

    stats = {'match': 0, 'total': 0, 'ctx_word': 0, 'ctx_punct': 0,
             'full_word': 0, 'full_punct': 0}
    posmap = {}
    for start in range(0, min(len(ids) - L, n_w * L), L):
        window = torch.tensor(ids[start:start + L], dtype=torch.long)
        logits = _head_forward(model, window, tok)
        ctx = logits - tb.unsqueeze(0)
        pr = F.softmax((logits.double() / t_eff), dim=-1)
        top1 = pr.max(dim=-1)
        H = -(pr * torch.log2(pr.clamp_min(1e-12))).sum(dim=-1)
        for pos in range(L):
            d = posmap.setdefault(pos, {'top1': 0.0, 'H': 0.0, 'cnt': 0, 'punct': 0, 'word': 0})
            d['top1'] += top1.values[pos].item()
            d['H'] += H[pos].item()
            d['cnt'] += 1
            if _cat(top1.indices[pos], tok) == 'punct':
                d['punct'] += 1
            elif _cat(top1.indices[pos], tok) == 'word':
                d['word'] += 1
        for pos in range(0, L, 8):
            lt = logits[pos]
            stats['total'] += 1
            if int(lt.argmax()) == tb_argmax:
                stats['match'] += 1
            for tid, cat, k in ((int(ctx[pos].argmax()), 'ctx', 'ctx'),
                                (int(lt.argmax()), 'full', 'full')):
                c = _cat(tid, tok)
                stats[f'{k}_word'] += c == 'word'
                stats[f'{k}_punct'] += c == 'punct'

    print(f'\n  BIAS-DECOMP: argmax==bias в {stats["match"]}/{stats["total"]} '
          f'({100.0 * stats["match"] / max(stats["total"], 1):.1f}%) | '
          f'ctx: word={stats["ctx_word"]} punct={stats["ctx_punct"]} | '
          f'full: word={stats["full_word"]} punct={stats["full_punct"]}')

    print('\n  POSMAP: pos | top-1 avg | H avg | punct% | word%')
    bins = [(0, 0, 15), (16, 15, 33), (32, 31, 65), (64, 63, 129), (128, 127, 193),
            (192, 191, 225), (224, 223, 241), (240, 239, 249), (250, 249, L - 1)]
    bins_out = []
    for label, lo, hi in bins:
        span = hi - lo + 1
        top1 = sum(posmap[p]['top1'] / posmap[p]['cnt'] for p in range(lo, hi + 1)) / span
        H = sum(posmap[p]['H'] / posmap[p]['cnt'] for p in range(lo, hi + 1)) / span
        pc = sum(posmap[p]['punct'] for p in range(lo, hi + 1)) / span * 100
        wc = sum(posmap[p]['word'] for p in range(lo, hi + 1)) / span * 100
        bins_out.append({'label': label, 'top1': top1, 'H': H, 'punct': pc, 'word': wc})
        print(f'  {label:3d} | {top1:.4f} | {H:.3f} | {pc:5.1f}% | {wc:5.1f}%')

    print('\n  A/B REASONING (одно окно, top-1/H + KL):')
    win = torch.tensor(ids[:L], dtype=torch.long)
    res = {}
    for mode, override in (('OFF', 0.0), ('natural', None), ('FULL', 1.0)):
        model.reasoning_scale_override = override
        model.reset_reasoning()
        logits = _head_forward(model, win, tok)[-1]
        p = F.softmax(logits.double(), dim=-1)
        pe = F.softmax((logits.double() / t_eff), dim=-1)
        Hr = -(p * torch.log2(p.clamp_min(1e-12))).sum().item()
        He = -(pe * torch.log2(pe.clamp_min(1e-12))).sum().item()
        res[mode] = (p.max().item(), Hr, pe.max().item(), He, p)
        print(f'    {mode:8s} top-1={p.max().item():.4f} H={Hr:.3f} bit | '
              f'sampler top-1={pe.max().item():.4f} H={He:.3f} bit')
    model.reasoning_scale_override = None
    pn, po = res['natural'][4].clamp_min(1e-12), res['OFF'][4].clamp_min(1e-12)
    print(f'    KL natural||OFF = {(pn * torch.log2(pn / po)).sum().item():.4f} bit | '
          f'KL OFF||natural = {(po * torch.log2(po / pn)).sum().item():.4f} bit')

    return {
        'log_temp': log_temp, 't_eff': t_eff,
        'token_bias_top': [tok.decode([int(i)]) for i in torch.topk(tb, 5).indices.tolist()],
        'bias_decomp': {'match': stats['match'], 'total': stats['total'],
                        'pct': 100.0 * stats['match'] / max(stats['total'], 1),
                        'ctx_word': stats['ctx_word'], 'ctx_punct': stats['ctx_punct'],
                        'full_word': stats['full_word'], 'full_punct': stats['full_punct']},
        'posmap': bins_out,
        'ab': {m: {'top1': v[0], 'H': v[1], 's_top1': v[2], 's_H': v[3]}
               for m, v in res.items()},
    }


# ─────────────────────────── CMP ───────────────────────────

def run_cmp(models, args, tok):
    sec('CMP: сравнение на одинаковом входе')
    prompt = args.prompt or 'В один холодный зимний вечер'
    ids = tok.encode(prompt).ids
    for ckpt_path, (model, _state) in models.items():
        model.eval()
        L = model.cfg.seq_len
        ctx = torch.tensor(ids[-L:], dtype=torch.long).unsqueeze(0)
        log_temp = model.lm_head.log_temp.data.mean().item()
        tb = model.lm_head.token_bias.data
        ro = model.lm_head.readout.data
        print(f'\n  {os.path.basename(ckpt_path)} | step={_state.get("step", "?")} | '
              f'reasoning t={model.reasoning_enabled_step} scale={model.reasoning_scale:.4f}')
        print(f'    head: log_temp={log_temp:.4f} | tb_max={tb.abs().max().item():.3f} '
              f'tb_std={tb.std().item():.3f} | readout_norm={ro.norm().item():.3f}')
        for ov, tag in ((0.0, 'reasoning OFF'), (None, 'natural'), (1.0, 'FULL')):
            model.reasoning_scale_override = ov
            model.reset_reasoning()
            with torch.no_grad():
                h = model.embed_tokens(ctx)
                out, _, _, _ = model(h, None, adaptive=False)
                logits = model.lm_head(out[:, -1:, :])[0, 0]
            eff = args.temp / math.exp(log_temp)
            for tname, z in (('raw', logits.double()), ('sampler', logits.double() / eff)):
                pr = F.softmax(z, dim=-1)
                topk = torch.topk(pr, 8)
                toks = [tok.decode([int(i)]) for i in topk.indices.tolist()]
                print(f'    [{tag}] {tname:7s} top-1={pr.max().item():.4f} '
                      f'H={-(pr * torch.log2(pr.clamp_min(1e-12))).sum().item():.3f} bit | '
                      f'{" | ".join(f"{t!r}({v:.3f})" for t, v in zip(toks, topk.values.tolist()))}')
        model.reasoning_scale_override = None


# ─────────────────────────── ANOMALY TRACK ───────────────────────────

def _track_path(ckpt_path):
    return os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'anomaly_track.json')


def run_anomaly(live, wake, ckpt):
    step = int(ckpt.get('step', 0))
    rows = live['rows']
    hp_max = max(rows, key=lambda r: r['hp'])
    pe_max = max(rows, key=lambda r: r['predMSE'])
    gate_min = min(rows, key=lambda r: r['gate'])
    med_hp = sorted(r['hp'] for r in rows)[len(rows) // 2]
    births = wake.get('births') or []

    report = [f'Anomaly track: step={step} layers={len(rows)}']
    report.append(f'  max ||hp|| L{hp_max["layer"]} = {hp_max["hp"]:.3f} '
                  f'(median {med_hp:.3f}, x{hp_max["hp"] / max(med_hp, 1e-6):.1f})')
    report.append(f'  max predMSE L{pe_max["layer"]} = {pe_max["predMSE"]:.1f}')
    report.append(f'  min gate L{gate_min["layer"]} = {gate_min["gate"]:.4f} '
                  f'(gate_var {gate_min["gate_var"]:.4f}, |mirror| {gate_min["mirror"]:.1f})')
    report.append(f'  births in empty layers: {births or "none"}')

    flags = []
    if hp_max['hp'] > 20.0:
        flags.append(('WAKE', f'||hp|| L{hp_max["layer"]} раздут ({hp_max["hp"]:.1f}, '
                              f'медиана {med_hp:.1f})'))
    elif hp_max['hp'] > 10.0:
        flags.append(('WATCH', f'||hp|| L{hp_max["layer"]} повышен ({hp_max["hp"]:.1f})'))
    if pe_max['predMSE'] > 1000.0:
        flags.append(('WAKE', f'predMSE L{pe_max["layer"]} огромен ({pe_max["predMSE"]:.0f})'))
    elif pe_max['predMSE'] > 50.0:
        flags.append(('WATCH', f'predMSE L{pe_max["layer"]} повышен ({pe_max["predMSE"]:.0f})'))
    if gate_min['gate'] < 0.25:
        flags.append(('WATCH', f'глубокая модуляция L{gate_min["layer"]} '
                               f'(gate {gate_min["gate"]:.3f})'))
    if births:
        flags.append(('WATCH', f'births {births}'))
    for flag, text in flags:
        _verdict(report, flag, [], text)
    hdr = 'OK' if not flags else ('ANOMALY' if any(f == 'WAKE' for f, _ in flags) else 'WATCH')
    print(f'\nANOMALY TRACK (verdict: {hdr})')
    for line in report:
        print(line)

    path = _track_path(ckpt.get('_path', 'best.pt'))
    hist = []
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                hist = json.load(f)
        except Exception:
            hist = []
    prev = next((e for e in reversed(hist) if e['step'] < step), None)
    if prev is not None:
        dh, dp, dg = (hp_max['hp'] - prev['hp_max'],
                      pe_max['predMSE'] - prev['predmse_max'],
                      gate_min['gate'] - prev['gate_min'])
        print(f'  vs prev step {prev["step"]}: ||hp||max {dh:+.3f} | '
              f'predMSEmax {dp:+.1f} | gate_min {dg:+.4f} | '
              f'births {prev["births"]} -> {births}')
    else:
        dh = dp = dg = None
    hist.append({'step': step, 'hp_max': hp_max['hp'], 'hp_layer': hp_max['layer'],
                 'predmse_max': pe_max['predMSE'], 'predmse_layer': pe_max['layer'],
                 'gate_min': gate_min['gate'], 'gate_min_layer': gate_min['layer'],
                 'births': births})
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False, indent=1)
        print(f'  track: {path}')
    except Exception as e:
        print(f'  [warn] track not saved: {e}')

    return {'verdict': hdr, 'report': report, 'hp_max': hp_max['hp'],
            'hp_layer': hp_max['layer'], 'predmse_max': pe_max['predMSE'],
            'predmse_layer': pe_max['layer'], 'gate_min': gate_min['gate'],
            'gate_min_layer': gate_min['layer'], 'births': births,
            'prev_step': prev['step'] if prev else None,
            'dhp': dh, 'dpe': dp, 'dgate': dg}


# ─────────────────────────── GRAD INFO ───────────────────────────

def _chunk_params(params, max_bytes=96 * 1024 * 1024):
    chunks, cur, acc = [], [], 0
    for p in params:
        sz = p.numel() * p.element_size()
        if cur and acc + sz > max_bytes:
            chunks.append(cur)
            cur, acc = [], 0
        cur.append(p)
        acc += sz
    if cur:
        chunks.append(cur)
    return chunks


def run_grad_info(model, cfg, seq=16):
    """GRAD INFO на крошечном forward (SEQ мал и память-лёгкий).

    Основной forward анализатора идёт ВНЕ автограда (no_grad), поэтому
    здесь делаем отдельный grad-прогон на seq<=16 токенов. Градиенты CE и
    diversity считаются по ОДНОМУ графу и накапливаются попараметно по чанкам:
    пиковая память ~ граф + 1 чанк градиентов вместо двух полных наборов
    (2×212.3M×4B ≈ 1.7GB). Автограф по всем 24 слоям всё равно нужен, но при
    seq=16 он мизерный."""
    import gc
    sec('GRAD INFO: dead_pred + cos_sim(diversity, CE)')
    info = {'dead_pred': None, 'cos_global': None, 'cos_l16': None,
            'scale': None, 'n_ce': None, 'n_div': None}
    try:
        # run_live вызывается под @torch.no_grad(); тут возвращаем градиенты,
        # чтобы forward и autograd.grad считали настоящие градиенты.
        with torch.enable_grad():
            params = list(model.parameters())
            x = torch.randint(0, cfg.vocab, (1, seq), device='cpu')
            tgt = torch.randint(1, cfg.vocab, (1, seq), device='cpu')
            h = model.embed_tokens(x)
            h_out, _, _, _ = model(h, tokens=x)
            ls, aux = model.compute_losses(h_out[:, :-1], tgt[:, 1:])

        pred = aux.get('pred')
        if pred is None:
            info['dead_pred'] = True
            print('  pred: МЁРТВЫЙ — отсутствует в aux (pred_k/hp детачатся в _pred_cache), '
                  'градиента НЕ даёт')
        else:
            dead = not (isinstance(pred, torch.Tensor) and pred.requires_grad)
            info['dead_pred'] = dead
            print(f'  pred: aux={pred.item():.4f} requires_grad={pred.requires_grad}'
                  + (' → БЕЗ ГРАДИЕНТА (dead)' if dead else ''))

        div = aux.get('diversity')
        if isinstance(div, torch.Tensor) and div.requires_grad:
            l16_set = set(id(p) for p in model.layers[16].parameters()) \
                if len(model.layers) > 16 else None
            a2 = b2 = dot = 0.0
            a2l = b2l = dotl = 0.0
            chunks = _chunk_params(params)
            for k, grp in enumerate(chunks):
                keep = k < len(chunks) - 1
                gce = torch.autograd.grad(ls, grp, retain_graph=keep, allow_unused=True)
                gdv = torch.autograd.grad(div, grp, retain_graph=keep, allow_unused=True)
                for p, g1, g2 in zip(grp, gce, gdv):
                    if g1 is None or g2 is None:
                        continue
                    f1 = g1.flatten().float()
                    f2 = g2.flatten().float()
                    a2 += float((f1 * f1).sum())
                    b2 += float((f2 * f2).sum())
                    dot += float((f1 * f2).sum())
                    if l16_set is not None and id(p) in l16_set:
                        a2l += float((f1 * f1).sum())
                        b2l += float((f2 * f2).sum())
                        dotl += float((f1 * f2).sum())
                del gce, gdv
            del x, tgt, h, h_out, ls
            gc.collect()
            n_ce, n_div = math.sqrt(a2), math.sqrt(b2)
            cos = (dot / (n_ce * n_div)) if n_ce > 0 and n_div > 0 else 0.0
            cos16 = (dotl / (a2l ** 0.5 * b2l ** 0.5)) if a2l > 0 and b2l > 0 else 0.0
            scale = max(0.0, min(10.0, cos)) * n_ce / max(n_div, 1e-8)
            info.update(cos_global=cos, cos_l16=cos16, scale=scale,
                        n_ce=n_ce, n_div=n_div, diversity_aux=div.item())
            print(f'  diversity aux={div.item():.3f}')
            print(f'  ||gCE||={n_ce:.3e}  ||gDIV||={n_div:.3e}')
            print(f'  cos_sim(diversity, CE): global={cos:.4f}  L16={cos16:.4f}')
            print(f'  scale(align) = cos·||CE||/||DIV|| = {scale:.4f} (cap 10)')
            if cos < 0.2:
                print('  [WATCH] diversity слабо выровнен с CE — её градиент почти не добавляется')
        else:
            print('  diversity: не требует градиента или отсутствует — пропущено')
    except Exception as e:
        print(f'  [warn] grad info: {e}')
    return info


# ─────────────────────────── BRIDGE ───────────────────────────

def run_bridge(model, cfg, batch=1, seq=128):
    """Runtime-метрики Intent Bridge: салайенс от головы, поток intent по слоям,
    кросс-слойная избыточность шины, head-stencil (bus_head_proj) и τ-лесенка
    интеграции intent. Требует intent_bridge=True (иначе возвращает None)."""
    sec('BRIDGE: runtime intent-bus metrics')
    if not getattr(model, 'intent_bridge', False):
        print('  Intent Bridge ОТКЛЮЧЁН (cfg.intent_bridge=False) — метрики недоступны')
        return None
    model.train()
    x = torch.randint(0, cfg.vocab, (batch, seq), device='cpu')
    with torch.no_grad():
        h = model.embed_tokens(x)
        out, _, _, _ = model(h)                   # заполняет _intent_stream, _last_bus
    B, L, D = out.shape

    # --- Salience из РЕАЛЬНОГО выхода головы ---
    with torch.no_grad():
        logits = model.lm_head(out)
    raw = logits.sigmoid().norm(dim=-1, keepdim=True)        # (B,L,1)
    sal = raw.flatten()
    sal_norm = raw / raw.mean().clamp_min(1e-6)
    p = sal_norm.flatten() / sal_norm.flatten().sum().clamp_min(1e-9)
    H = -(p * torch.log2(p.clamp_min(1e-12))).sum().item()
    Hmax = math.log2(sal.numel())
    print(f'  SALIENCE (от head logits): mean={sal.mean().item():.4e} '
          f'min={sal.min().item():.4e} max={sal.max().item():.4e} std={sal.std().item():.4e}')
    print(f'    max/min={sal.max().item() / max(sal.min().item(), 1e-12):.1f}  '
          f'норм.энтропия H/Hmax={H / Hmax:.3f}  (1=равномерно, 0=сфокус на токенах)')

    # --- Intent stream (per-layer carried gist) ---
    stream = model._intent_stream
    if not isinstance(stream, list) or len(stream) != len(model.layers):
        print('  [warn] _intent_stream не заполнен после forward — пропуск stream-метрик')
        layer_norms, expert_norm, off = [], [], 0.0
    else:
        layer_norms = [s.norm().item() for s in stream]
        stacked = torch.stack([s.reshape(-1, model._n_experts, model._K_max)
                               for s in stream], 0)        # (L,G,Kmax)
        expert_norm = stacked.norm(dim=-1).mean(dim=0)     # (G,)
        flat = stacked.reshape(len(stream), -1)
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cos = torch.matmul(flat, flat.t())
        off = cos[~torch.eye(len(stream), dtype=torch.bool)].mean().item() \
            if len(stream) > 1 else 0.0
        print(f'  INTENT STREAM: per-layer norm mean={sum(layer_norms)/len(layer_norms):.4f} '
              f'min={min(layer_norms):.4f} max={max(layer_norms):.4f}')
        en = expert_norm.flatten().tolist()
        if model._n_experts > 8:
            print(f'    per-expert mean norm (первые 8): '
                  f'{[round(v, 4) for v in en[:8]]} ...')
        else:
            print(f'    per-expert norm: {[round(v, 4) for v in en]}')
        print(f'    cross-layer cosine (redundancy): mean_offdiag={off:.4f} '
              f'(1=слои несут одно и то же, 0=дополняют)')

    # --- Bus ---
    if model._last_bus is not None:
        bus = model._last_bus
        print(f'  BUS (_last_bus): norm={bus.norm().item():.4f}  '
              f'per-expert mean='
              f'{bus.reshape(-1, model._n_experts, model._K_max).norm(dim=-1).mean().item():.4f}')

    # --- Head stencil (bus_head_proj) ---
    bhp = model.bus_head_proj
    w = bhp.weight.data
    print(f'  HEAD STENCIL bus_head_proj: weight_norm={w.norm().item():.4f} '
          f'(zero-init => 0 на старте, рост = стенсил обучается)')
    if model._last_bus is not None:
        _bus = model._last_bus.expand(B, L, -1, -1).reshape(B, L, -1)
        bus_bias = bhp(_bus)
        print(f'    bus_bias (вклад в логиты головы): norm/pos='
              f'{bus_bias.norm(dim=-1).mean().item():.5f}')

    # --- Bridge params per layer ---
    print('  BRIDGE PARAMS (L0 / mid / last):')
    for i in (0, len(model.layers) // 2, len(model.layers) - 1):
        m = model.layers[i].mirror
        wi = m.w_intent.norm().item() if hasattr(m, 'w_intent') else 0.0
        bi = m.b_intent.norm().item() if hasattr(m, 'b_intent') else 0.0
        ws = torch.sigmoid(m.w_sal).mean().item() if hasattr(m, 'w_sal') else 0.0
        print(f'    L{i}: ||w_intent||={wi:.4f} ||b_intent||={bi:.4f} w_sal(sig)={ws:.4f}')

    ip = model.intent_probe
    print(f'  intent_probe: W_norm={ip.weight.data.norm().item():.4f} '
          f'b_norm={ip.bias.data.norm().item() if ip.bias is not None else 0.0:.4f}')

    # --- τ-ladder (intent integration timescales) ---
    vsa_tau = torch.exp(torch.cumsum(F.softplus(model._vsa_log_param), 0)) + 1.0
    tau_min, tau_max = vsa_tau[0], vsa_tau[-1]
    c_ema = (1.0 / math.sqrt(cfg.D)) * (tau_min * tau_max).sqrt()
    n = len(model.layers)
    # Use tau_config.intent_alpha (v2 formula: 1 − exp(−tau_l/tau_min))
    if hasattr(model, 'tau_config'):
        taus = model.tau_config._tau_l_cache.tolist()
        alphas = model.tau_config._alpha_cache.tolist()
    else:
        taus, alphas = [], []
        for i in range(n):
            taus.append(0.0)
            alphas.append(0.0)
    print(f'  TAU LADDER (intent): tau[0]={taus[0]:.3f} tau[-1]={taus[-1]:.3f} '
          f'alpha[0]={alphas[0]:.4f} alpha[-1]={alphas[-1]:.4f} '
          f'(alpha = доля carried intent в EMA)')

    return {
        'salience': {'mean': sal.mean().item(), 'min': sal.min().item(),
                     'max': sal.max().item(),
                     'ratio': sal.max().item() / max(sal.min().item(), 1e-12),
                     'entropy_norm': H / Hmax},
        'stream_layer_norm': layer_norms,
        'stream_expert_norm': expert_norm.tolist() if isinstance(expert_norm, torch.Tensor) else [],
        'cross_layer_cos': off,
        'bus_norm': model._last_bus.norm().item() if model._last_bus is not None else None,
        'stencil_w_norm': w.norm().item(),
        'tau': taus, 'alpha': alphas,
    }


# ─────────────────────────── ALL-METRICS LOG PARSER ───────────────────────────

import re as _re

_MAIN_RE = _re.compile(
    r'step=\s*(\d+)\s+loss=([-\d.eE+]+)\s+ce=([-\d.eE+]+)\s+'
    r'mod_mlp=([-\d.eE+]+)\s+mod_std=([-\d.eE+]+)\s+lr=([-\d.eE+]+)\s+'
    r'tok/s=(\d+)\s+mem=([\d.]+)GB\s+intent_eff=([-\d.eE+]+)\s+'
    r'mlp_out=([-\d.eE+]+)\s+usef=([-\d.eE+]+)\s+mat=([\d.]+)\[([\d.]+),([\d.]+)\]')
_AUX_RE = _re.compile(r'aux:\s+(.*)')
_AUX_KV = _re.compile(r'(\w+)=([-\d.eE+]+)')
_EVAL_RE = _re.compile(r'EVAL step=(\d+):\s*val_loss=([-\d.eE+]+)\s*val_ppl=([-\d.eE+]+)')
_DEPTH_RE = _re.compile(r'\[DepthController\].*?->\s*active_depth=(\d+)/(\d+)')
_BRIDGE_RE = _re.compile(r'In-core SemanticBridge active\s*\((.*?)\)')
_SAVE_RE = _re.compile(r'(?:EVAL )?[Ss]aved (best|latest)[^\n]*?step\s*=?\s*(\d+)')


def parse_training_log(path):
    """Парсит лог обучения Colab и возвращает ВСЕ метрики в структурированном виде.

    Возвращает dict:
      steps : list[int]                       — шаги с основной строкой
      main  : {metric: [val,...]}             — loss, ce, mod_mlp, mod_std, lr,
                                                tok_s, mem, intent_eff, mlp_out, usef,
                                                mat, mat_min, mat_max (по шагам)
      aux   : {metric: [val,...]}             — ВСЕ aux: ключи (alpha_novelty, balance,
                                                branch, bridge_conn, decorr, div, diversity,
                                                gate_l1, gate_repulse, gradalign, intent_tau,
                                                 ls_reg, nuc, pred, reinforce,
                                                signal_ent, w_m2v, ...)
      eval  : [(step, val_loss, val_ppl), ...]
      depth : [(step, active, total), ...]
      bridge: str | None                      — строка In-core SemanticBridge active (...)
      saves : [(kind, step), ...]
    """
    data = {'steps': [], 'main': {}, 'aux': {}, 'eval': [], 'depth': [],
            'bridge': None, 'saves': []}
    MAIN_KEYS = ['loss', 'ce', 'mod_mlp', 'mod_std', 'lr', 'tok_s', 'mem',
                 'intent_w', 'mlp_out', 'usef', 'mat', 'mat_min', 'mat_max']  # internal key; log prints intent_eff
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = _MAIN_RE.search(line)
            if m:
                step = int(m.group(1))
                vals = [float(x) for x in m.groups()[1:]]
                # vals order: loss, ce, mod_mlp, mod_std, lr, tok_s, mem, intent_w,
                #             mlp_out, usef, mat, mat_min, mat_max  (13 чисел)
                data['steps'].append(step)
                for k, v in zip(MAIN_KEYS, vals):
                    data['main'].setdefault(k, []).append(v)
                continue
            a = _AUX_RE.search(line)
            if a:
                for k, v in _AUX_KV.findall(a.group(1)):
                    try:
                        data['aux'].setdefault(k, []).append(float(v))
                    except ValueError:
                        pass
                continue
            e = _EVAL_RE.search(line)
            if e:
                data['eval'].append((int(e.group(1)), float(e.group(2)), float(e.group(3))))
                continue
            d = _DEPTH_RE.search(line)
            if d:
                data['depth'].append((data['steps'][-1] if data['steps'] else 0,
                                      int(d.group(1)), int(d.group(2))))
                continue
            b = _BRIDGE_RE.search(line)
            if b:
                data['bridge'] = b.group(1)
                continue
            s = _SAVE_RE.search(line)
            if s:
                step_s = s.group(2)
                data['saves'].append((s.group(1), int(step_s) if step_s else None))
                continue
    return data


def _svg_spark(steps, vals, w=640, h=70, color='#58a6ff'):
    if not steps or len(vals) < 2:
        return '<div class="dim">нет данных</div>'
    xs = list(range(len(vals)))
    x0, x1 = xs[0], xs[-1]
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    pad = 4
    def px(i, v):
        xx = pad + (w - 2 * pad) * (xs[i] - x0) / max(1, (x1 - x0))
        yy = (h - pad) - (h - 2 * pad) * (v - vmin) / (vmax - vmin)
        return xx, yy
    pts = ' '.join(f'{px(i, vals[i])[0]:.1f},{px(i, vals[i])[1]:.1f}'
                   for i in range(len(vals)))
    last = px(len(vals) - 1, vals[-1])
    return (f'<svg width="{w}" height="{h}" style="display:block">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}"/>'
            f'<circle cx="{last[0]:.1f}" cy="{last[1]:.1f}" r="2.5" fill="{color}"/>'
            f'</svg><div class="dim">min={vmin:.4g} max={vmax:.4g} last={vals[-1]:.4g}</div>')


def render_log_html(data, outpath):
    import html as H
    steps = data['steps']
    main = data['main']
    aux = data['aux']

    # сводные карточки (последнее значение каждой ключевой метрики)
    def _last(d, k):
        return d[k][-1] if (k in d and d[k]) else float('nan')

    cards = [
        ('Steps', f'{steps[0] if steps else "?"}–{steps[-1] if steps else "?"}'),
        ('CE', f'{_last(main, "ce"):.4f}'),
        ('val_loss', f'{data["eval"][-1][1] if data["eval"] else float("nan"):.4f}'),
        ('val_ppl', f'{data["eval"][-1][2]:.0f}' if data["eval"] else '—'),
        ('mat', f'{_last(main, "mat"):.4f}'),
        ('mat_min', f'{_last(main, "mat_min"):.4f}'),
        ('bridge_conn', f'{_last(aux, "bridge_conn"):.4f}'),
        ('branch', f'{_last(aux, "branch"):.2f}'),
        ('pred', f'{_last(aux, "pred"):.3f}'),
        ('gradalign', f'{_last(aux, "gradalign"):.2f}'),
        ('intent_eff', f'{_last(main, "intent_w"):.3f}'),
        ('usef', f'{_last(main, "usef"):.3f}'),
        ('active_depth', f'{data["depth"][-1][1]}/{data["depth"][-1][2]}' if data['depth'] else '—'),
        ('tok/s', f'{_last(main, "tok_s"):.0f}'),
    ]
    ch = []
    ch.append('<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><title>EVA training log</title><style>')
    ch.append('body{background:#0d1117;color:#c9d1d9;font:14px/1.5 Consolas,monospace;margin:24px}'
              'h1{color:#f0f6fc}h2{color:#79c0ff;border-bottom:1px solid #30363d;padding-bottom:4px;margin-top:26px}'
              '.cards{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}'
              '.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 14px;min-width:96px}'
              '.card b{display:block;font-size:17px;color:#f0f6fc}.card span{font-size:11px;color:#8b949e;text-transform:uppercase}'
              'table{border-collapse:collapse;margin:10px 0;font-size:12px}th,td{border:1px solid #30363d;padding:3px 8px;text-align:right}'
              'th{background:#161b22;color:#8b949e;position:sticky;top:0}td:first-child,th:first-child{text-align:left}'
              '.scroll{max-height:420px;overflow:auto;border:1px solid #30363d;border-radius:8px}'
              '.dim{color:#8b949e}.g{color:#7ee787}.y{color:#e3b341}.r{color:#ff7b72}')
    ch.append('</style></head><body>')
    ch.append('<h1>EVA — Training Log Dashboard</h1>')
    if data['bridge']:
        ch.append(f'<div class="dim">bridge: {H.escape(data["bridge"])}</div>')
    if data['saves']:
        sv = ', '.join(f'{k}@{s}' for k, s in data['saves'])
        ch.append(f'<div class="dim">saves: {H.escape(sv)}</div>')
    ch.append('<div class="cards">')
    for k, v in cards:
        ch.append(f'<div class="card"><b>{v}</b><span>{k}</span></div>')
    ch.append('</div>')

    # графики ключевых метрик
    ch.append('<h2>ДИНАМИКА КЛЮЧЕВЫХ МЕТРИК</h2>')
    chart_metrics = [('ce', '#ff7b72'), ('mat', '#79c0ff'), ('mat_min', '#a5d6ff'),
                     ('mod_mlp', '#7ee787'), ('bridge_conn', '#d2a8ff'), ('branch', '#e3b341'),
                     ('pred', '#ffa657'), ('gradalign', '#56d364'), ('diversity', '#79c0ff'),
                     ('gate_l1', '#58a6ff'), ('intent_w', '#ff7b72'),  # intent_w = intent_eff series
                     ('usef', '#7ee787'), ('div', '#e3b341'), ('decorr', '#79c0ff'),
                     ('reinforce', '#d2a8ff'), ('ls_reg', '#56d364'), ('nuc', '#f0883e'),
                     ('tok_s', '#8b949e'), ('lr', '#a5d6ff')]
    for k, col in chart_metrics:
        src = main if k in main else aux
        if k in src and len(src[k]) >= 2:
            ch.append(f'<h3>{k} ({len(src[k])} точек)</h3>')
            ch.append(_svg_spark(steps, src[k], color=col))

    # таблица ВСЕХ метрик по шагам
    ch.append('<h2>ВСЕ МЕТРИКИ ПО ШАГАМ (полная таблица)</h2>')
    all_keys = list(main.keys()) + [k for k in aux.keys() if k not in main]
    ch.append('<div class="scroll"><table><tr><th>step</th>')
    for k in all_keys:
        ch.append(f'<th>{H.escape(k)}</th>')
    ch.append('</tr>')
    nrows = len(steps)
    for i in range(nrows):
        ch.append(f'<tr><td>{steps[i]}</td>')
        for k in all_keys:
            src = main if k in main else aux
            _ser = src.get(k) or ()          # aux-ряды могут быть короче (gradalign
            v = _ser[i] if i < len(_ser) else None  # появляется не с первой строки)
            if v is None:
                ch.append('<td>—</td>')
            else:
                ch.append(f'<td>{v:.4g}</td>')
        ch.append('</tr>')
    ch.append('</table></div>')

    if data['eval']:
        ch.append('<h2>VAL LOSS (EVAL)</h2>')
        ch.append('<table><tr><th>step</th><th>val_loss</th><th>val_ppl</th></tr>')
        for s, vl, vp in data['eval']:
            ch.append(f'<tr><td>{s}</td><td>{vl:.4f}</td><td>{vp:.0f}</td></tr>')
        ch.append('</table>')
    if data['depth']:
        ch.append('<h2>ACTIVE DEPTH (DepthController)</h2>')
        ch.append('<table><tr><th>step</th><th>active</th><th>total</th></tr>')
        for s, a, t in data['depth']:
            ch.append(f'<tr><td>{s}</td><td>{a}</td><td>{t}</td></tr>')
        ch.append('</table>')

    ch.append('<p class="dim">generated by analyze.py --log</p>')
    ch.append('</body></html>')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ch))
    print(f'\nLOG HTML report: {outpath}')


def run_metacog(model, cfg):
    """Извлекает ВСЕ мета-когнитивные буферы модели (per-layer + глобальные)."""
    layers = []
    for i, layer in enumerate(model.layers):
        m = layer.mirror
        row = {'layer': i}
        try:
            row['pm_norm'] = (m._private_mem.norm(dim=-1).mean().item()
                              if m._has_private_mem else None)
        except Exception:
            row['pm_norm'] = None
        try:
            w = torch.sigmoid(m._signal_log_weights)
            row['signal_w'] = (w / (w.sum() + 1e-10)).tolist()
        except Exception:
            row['signal_w'] = None
        for bname in ('gate_ema', 'w_help', 'w_contra'):
            try:
                row[bname] = float(getattr(m, bname).mean().item() if hasattr(m, bname) else getattr(m, '_' + bname).mean().item())
            except Exception:
                row[bname] = None
        try:
            row['trust_diag'] = m._trust_matrix.diag().mean().item()
        except Exception:
            row['trust_diag'] = None
        try:
            row['trust_max'] = m._trust_matrix.max().item()
        except Exception:
            row['trust_max'] = None
        try:
            row['concept_sim'] = m._concept_sim_ema.mean().item()
        except Exception:
            row['concept_sim'] = None
        try:
            row['behavior_div'] = m._behavior_div_ema.mean().item()
        except Exception:
            row['behavior_div'] = None
        try:
            row['meta_private_mem'] = (m._meta_private_mem.mean().item()
                                        if m._meta_trust else None)
        except Exception:
            row['meta_private_mem'] = None
        layers.append(row)
    mat = model.maturation
    out = {
        'layers': layers,
        'mat_gate': mat.gate.tolist(),
        'mat_readiness': mat.readiness.tolist(),
        'mat_pen_init': mat.pen_init.tolist(),
        'mat_pen_ema': mat.pen_ema.tolist(),
        'bridge_readiness': (float(model.bridge.readiness())
                             if model.bridge is not None else None),
    }
    return out


# ─────────────────────────── HTML ───────────────────────────

def _hue(v, vmin, vmax):
    t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    return 120 * (1.0 - t)


def save_html_report(ckpt, cfg, model, wake, live, head, anomaly=None, bridge=None, metacog=None,
                     static=None):
    import html as H
    path = ckpt.get('_path', '?')
    step = ckpt.get('step', '?')
    stem = os.path.splitext(path)[0]
    out = f'{stem}_{step}_report.html' if step != '?' else stem + '_report.html'
    best = ckpt.get('best_val_loss', float('inf'))
    params = model.param_count() / 1e6

    w_sal_val = None
    if hasattr(model.layers[0].mirror, 'w_sal'):
        w_sal_val = torch.stack([torch.sigmoid(l.mirror.w_sal.data).mean()
                                 for l in model.layers]).mean().item()
    tau_intent_dev_val = (model.tau_config._tau_dev.data.mean().item()
                          if hasattr(model, 'tau_config') else None)

    if live is None:
        live = {'ce_random': float('nan'), 'pred': float('nan'), 'tau_l_dev': float('nan'),
                'gate_l1': float('nan'), 'usef': float('nan'), 'rows': [], 'signals': [],
                'mirror': [], 'grad_info': None}
    live_available = live.get('rows') or live.get('signals') or live.get('mirror')

    badge = lambda flag: (f'<span class="bdg b-{flag}">{flag}</span>'
                          if flag in ('PASS', 'WATCH', 'WAKE') else flag)
    ch = []
    ch.append('<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">')
    ch.append('<title>EVA report step ' + str(step) + '</title><style>')
    ch.append('''body{background:#0d1117;color:#c9d1d9;font:14px/1.5 Consolas,monospace;margin:24px}
h1{font-size:20px;color:#f0f6fc}h2{font-size:16px;color:#79c0ff;border-bottom:1px solid #30363d;padding-bottom:4px;margin-top:28px}
h3{font-size:13px;color:#79c0ff;margin-top:16px;margin-bottom:4px}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 14px;min-width:110px}
.card b{display:block;font-size:18px;color:#f0f6fc}.card span{font-size:11px;color:#8b949e;text-transform:uppercase}
table{border-collapse:collapse;margin:10px 0}th,td{border:1px solid #30363d;padding:4px 10px;text-align:right}
th{background:#161b22;color:#8b949e;font-size:11px;text-transform:uppercase}
td:first-child,th:first-child{text-align:left}
.bdg{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:bold}
.b-PASS{background:#1f6f2f;color:#7ee787}.b-WATCH{background:#6b4d00;color:#e3b341}.b-WAKE{background:#6e1a1a;color:#ff7b72}
.heat{display:flex;gap:3px;margin:10px 0}.hc{flex:1;text-align:center;font-size:10px;color:#8b949e;border-radius:4px;padding:4px 0}
.bar{background:#21262d;border-radius:4px;height:14px;position:relative;margin:2px 0;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;background:#58a6ff;border-radius:4px}
.bar span{position:absolute;left:6px;top:0;font-size:10px;color:#e6edf3}
.g{color:#7ee787}.y{color:#e3b341}.r{color:#ff7b72}.dim{color:#8b949e}''')
    ch.append('</style></head><body>')
    ch.append(f'<h1>EVA — {H.escape(os.path.basename(path))}</h1>')
    ch.append(f'<div class="dim">step={step} &nbsp; best_val={best:.4f} &nbsp; params={params:.2f}M '
              f'&nbsp; {wake["verdict"]}</div>')

    ch.append('<div class="cards">')
    cards = [
        ('Step', str(step)), ('Best val', f'{best:.4f}'),
        ('MLP W_std', f'{wake["wstd"]:.4f}'), ('dev', f'{wake["dev_mean"]:+.4f}'),
        ('mod_mlp σ', f'{wake["g_mlp"]:.3f}'), ('mod_mem σ', f'{wake["g_mem"]:.3f}'),
        ('slots', f'{wake["slots"]}/192'), ('full L', str(wake["full"])),
        ('births', str(wake["births"])),
    ]
    if live_available:
        cards.append(('CE(rand)', f'{live["ce_random"]:.3f}'))
        cards.append(('pred', f'{live["pred"]:.3f}'))
        cards.append(('gate_l1', f'{live["gate_l1"]:.5f}'))
        cards.append(('tau_l_dev', f'{live["tau_l_dev"]:.4f}'))
    else:
        cards.append(('CE(rand)', '—'))
        cards.append(('pred', '—'))
        cards.append(('gate_l1', '—'))
        cards.append(('tau_l_dev', '—'))
    if wake.get('mat') is not None:
        m = wake['mat']
        cards.append(('mat gate', f'{m["gmax"]:.3f}'))
        cards.append(('mat mean', f'{m["gmean"]:.3f}'))
    if not math.isnan(live.get('usef', float('nan'))):
        cards.append(('usef', f'{live["usef"]:.3f}'))
    for k, v in cards:
        ch.append(f'<div class="card"><b>{v}</b><span>{k}</span></div>')
    if w_sal_val is not None:
        ch.append(f'<div class="card"><b>{w_sal_val:.4f}</b><span>w_sal &sigma;</span></div>')
    if tau_intent_dev_val is not None:
        ch.append(f'<div class="card"><b>{tau_intent_dev_val:.4f}</b><span>tau_intent_dev</span></div>')
    if bridge is not None:
        b = bridge['salience']
        ch.append(f'<div class="card"><b>{b["entropy_norm"]:.3f}</b><span>salience H/Hmax</span></div>')
        ch.append(f'<div class="card"><b>{bridge["cross_layer_cos"]:.3f}</b><span>bus cos</span></div>')
        ch.append(f'<div class="card"><b>{bridge["stencil_w_norm"]:.4f}</b><span>stencil ‖W‖</span></div>')
        ch.append(f'<div class="card"><b>{bridge["alpha"][-1]:.4f}</b><span>intent α[-1]</span></div>')
    ch.append('</div>')

    if static is not None:
        ch.append('<h2>STATIC</h2>')
        ch.append(f'<div class="dim">file={H.escape(static.get("file", "?"))} &nbsp; '
                  f'step={static.get("step", "?")} &nbsp; best_val={static.get("best_val", "?")} '
                  f'&nbsp; missing={len(static.get("missing", []))} '
                  f'unexpected={len(static.get("unexpected", []))}</div>')
        if static.get('config'):
            ch.append('<h3>CONFIG</h3><table><tr><th>param</th><th>value</th></tr>')
            for k, v in static['config']:
                ch.append(f'<tr><td>{H.escape(str(k))}</td><td>{H.escape(str(v))}</td></tr>')
            ch.append('</table>')
        if static.get('param_groups'):
            ch.append('<h3>PARAM GROUPS (LR multipliers)</h3>')
            ch.append('<table><tr><th>lr</th><th>wd</th><th>n_params</th></tr>')
            for g in static['param_groups']:
                ch.append(f'<tr><td>{g["lr"]:.6f}</td><td>{g["wd"]:.4f}</td><td>{g["n"]}</td></tr>')
            ch.append('</table>')
        if static.get('per_layer'):
            ch.append('<h3>PER-LAYER (params, без forward)</h3>')
            ch.append('<table><tr><th>L</th><th>alpha</th><th>|1-a|</th><th>ls_std</th>'
                      '<th>w_help</th><th>skip</th><th>scale_w</th><th>conv_std</th>'
                      '<th>pm_norm</th></tr>')
            for r in static['per_layer']:
                ch.append(f'<tr><td>{r["layer"]}</td><td>{r["alpha"]:.4f}</td>'
                          f'<td>{r["a1"]:.4f}</td><td>{r["ls_std"]:.4f}</td>'
                          f'<td>{r["w_help"]:.4f}</td><td>{r["skip"]:.4f}</td>'
                          f'<td>{r["scale_w"]:.4f}</td><td>{r["conv_std"]:.4f}</td>'
                          f'<td>{r["pm_norm"]:.3f}</td></tr>')
            ch.append('</table>')
        if static.get('vsatau'):
            vt = static['vsatau']
            ch.append('<h3>VSA TAU</h3><div class="dim">'
                      f'tau[0]={vt["tau0"]:.2f} &nbsp; tau[-1]={vt["taun"]:.2f} &nbsp; '
                      f'ratio={vt["ratio"]:.1f}x &nbsp; tau_l_dev={vt["tau_l_dev"]:.4f} '
                      f'(std {vt["tau_l_std"]:.4f})')
            if vt.get('b_d'):
                b = vt['b_d']
                ch.append(f'<br>b_d: mean={b[0]:.4f} range=[{b[1]:.4f}, {b[2]:.4f}]')
            if vt.get('b_i'):
                b = vt['b_i']
                ch.append(f'<br>b_i: mean={b[0]:.4f} range=[{b[1]:.4f}, {b[2]:.4f}]')
            ch.append('</div>')
        if static.get('mirror'):
            ch.append('<h3>MIRROR INTERNALS (L0, mid, last) — ВСЕ параметры зеркала</h3>')
            for row in static['mirror']:
                ch.append(f'<div class="dim">L{row["layer"]}:</div>')
                ch.append('<table><tr><th>param</th><th>shape</th><th>mean</th><th>std</th>'
                          '<th>min</th><th>max</th></tr>')
                for p in row['params']:
                    ch.append(f'<tr><td>{H.escape(p["name"])}</td><td>{p["shape"]}</td>'
                              f'<td>{p["mean"]:.4f}</td><td>{p["std"]:.4f}</td>'
                              f'<td>{p["min"]:.4f}</td><td>{p["max"]:.4f}</td></tr>')
                ch.append('</table>')
        if static.get('lm_head'):
            ch.append('<h3>LM HEAD</h3>')
            ch.append('<table><tr><th>name</th><th>shape</th><th>mean</th><th>std</th></tr>')
            for p in static['lm_head']:
                ch.append(f'<tr><td>{H.escape(p["name"])}</td><td>{p["shape"]}</td>'
                          f'<td>{p["mean"]:.6f}</td><td>{p["std"]:.6f}</td></tr>')
            ch.append('</table>')
        if static.get('tensors'):
            t = static['tensors']
            ch.append('<h3>TENSORS</h3><div class="dim">'
                      f'{t["numel"]} scalars &nbsp; mean={t["mean"]:.6f} &nbsp; '
                      f'std={t["std"]:.6f} &nbsp; min={t["min"]:.6f} &nbsp; '
                      f'max={t["max"]:.6f} &nbsp; NaN={t["nan"]} &nbsp; Inf={t["inf"]}</div>')
            if t.get('quantiles'):
                ch.append('<div class="dim">' + '  '.join(
                    f'Q{q}={v:.6f}' for q, v in t['quantiles']) + '</div>')
        if static.get('reasoning'):
            r = static['reasoning']
            ch.append('<h3>REASONING</h3><div class="dim">'
                      f'enabled_step={r["enabled_step"]} &nbsp; scale={r["scale"]:.4f} &nbsp; '
                      f'last gates={r["gates"]}</div>')
        if static.get('inspector'):
            ins = static['inspector']
            ch.append('<h3>INSPECTOR (private memory)</h3>')
            ch.append('<div class="dim">signals: ' + '  '.join(
                f'{n}={v:.3f}' for n, v in ins['signals'].items()) + '</div>')
            ch.append(f'<div class="dim">w_help(sigmoid)={ins["w_help"]:.3f} &nbsp; '
                      f'w_contra={ins["w_contra"]:.4f}</div>')
            if ins.get('concept_sim'):
                cs = ins['concept_sim']
                ch.append(f'<div class="dim">concept_sim: mean={cs["mean"]:.4f} '
                          f'std={cs["std"]:.4f} diag={cs["diag"]:.4f}</div>')
            if ins.get('behavior_div') is not None:
                ch.append(f'<div class="dim">behavior_div: {ins["behavior_div"]:.4f}</div>')
            if ins.get('trust'):
                tr = ins['trust']
                ch.append(f'<div class="dim">trust: mean={tr["mean"]:.4f} '
                          f'diag={tr["diag"]:.4f}</div>')
            for key, label in (('dominance', 'dominance'), ('isolation', 'isolation'),
                               ('contra_expert', 'contra_expert')):
                if ins.get(key):
                    ch.append(f'<div class="dim">{label}: {ins[key]}</div>')
            if ins.get('pm_step') is not None:
                ch.append(f'<div class="dim">pm_step: {ins["pm_step"]}</div>')

    ch.append('<h2>WAKE DETECTOR</h2>')
    for line in wake['report']:
        flag = ''
        for f in ('WAKE', 'WATCH', 'PASS'):
            if line.strip().startswith(f'[{f}]') or (f + ' ') in line[:12]:
                flag = f
                break
        esc = H.escape(line)
        if flag:
            esc = esc.replace(f'[{flag}]', badge(flag), 1)
        ch.append(f'<div>{esc}</div>')

    ch.append('<h2>Per-layer gate max (wake &gt; 0.75)</h2>')
    ch.append('<div class="heat">')
    for lyr, v in zip(wake['gate_max_layers'], wake['gate_max']):
        hue = _hue(v, 0.60, 0.80)
        ch.append(f'<div class="hc" title="L{lyr}: {v:.4f}" style="background:hsl({hue:.0f},75%,32%)">'
                  f'{lyr}<br>{v:.2f}</div>')
    ch.append('</div>')

    ch.append('<h2>Per-layer W_std dev (marker #1b)</h2>')
    ch.append('<div class="heat">')
    for lyr, d in zip(wake['dev_layers'], wake['dev']):
        hue = _hue(d, 0.0, 0.004)
        ch.append(f'<div class="hc" title="L{lyr}: {d:+.5f}" style="background:hsl({hue:.0f},75%,32%)">'
                  f'{lyr}<br>{d:+.4f}</div>')
    ch.append('</div>')

    if wake.get('mat') is not None:
        m = wake['mat']
        ch.append(f'<h2>MATURATION gate (live wake-up ramp; T0={m["T0"]:.0f} '
                  f'delta={m["delta"]:.0f} T_delay={m["T_delay"]:.0f})</h2>')
        ch.append('<div class="heat">')
        for lyr, v in enumerate(m['gate']):
            hue = _hue(v, 0.0, 0.8)
            ch.append(f'<div class="hc" title="L{lyr}: {v:.4f}" style="background:hsl({hue:.0f},75%,32%)">'
                      f'{lyr}<br>{v:.2f}</div>')
        ch.append('</div>')
        ch.append(f'<div class="dim">readiness max={m["readiness_max"]:.4f} '
                  f'&nbsp; tau_norm max={m["tau_norm_max"]:.3f} '
                  f'(gate opens deeper layers later via tau-geometry)</div>')

    ch.append('<h2>LIVE — per-layer dissection</h2>')
    if live['rows']:
        ch.append('<table><tr><th>L</th><th>||hp||</th><th>predMSE</th><th>|mirror|</th>'
                  '<th>gate</th><th>gate_var</th><th>|1-a|</th><th>ls_expM</th><th>ls_std</th>'
                  '<th>skip</th><th>w_delta</th></tr>')
        for r in live['rows']:
            ch.append('<tr><td>' + str(r['layer']) + '</td><td>' + f"{r['hp']:.3f}" +
                      '</td><td>' + f"{r['predMSE']:.3f}" + '</td><td>' + f"{r['mirror']:.3f}" +
                      '</td><td>' + f"{r['gate']:.4f}" + '</td><td>' + f"{r['gate_var']:.4f}" +
                      '</td><td>' + f"{r['a1']:.4f}" + '</td><td>' + f"{r['ls_expM']:.2f}" +
                      '</td><td>' + f"{r['ls_std']:.4f}" + '</td><td>' + f"{r['skip']:.3f}" +
                      '</td><td>' + f"{r['w_delta']:.5f}" + '</td></tr>')
        ch.append('</table>')
    else:
        ch.append('<div class="dim">live пропущен (--no-live) — данные не собраны</div>')

    ch.append('<h2>SIGNALS (softmax share)</h2>')
    if live['signals']:
        for s in live['signals']:
            ch.append(f'<div class="dim">L{s["layer"]}: ' +
                      ' '.join(f'{n}={v:.3f}' for n, v in zip(s['names'], s['share'])) + '</div>')
            for n, v in zip(s['names'], s['share']):
                w = v * 100
                ch.append(f'<div class="bar"><i style="width:{w:.1f}%"></i>'
                          f'<span>{n} {v:.3f}</span></div>')
    else:
        ch.append('<div class="dim">live пропущен (--no-live) — данные не собраны</div>')

    ch.append('<h2>MIRROR PARAMS (L0 / mid / last)</h2>')
    if live['mirror']:
        ch.append('<table><tr><th>L</th><th>α_diag</th><th>log_scale</th><th>tanh_bias</th>'
                  '<th>b_gate</th><th>gate_bias</th><th>mod_mlp σ</th><th>mod_mem σ</th>'
                  '<th>gate_ema</th><th>w_sal σ</th><th>w_intent σ</th><th>w_help σ</th>'
                  '<th>w_contra</th><th>pm_norm</th></tr>')
        for r in live['mirror']:
            ch.append('<tr><td>' + str(r['layer']) + '</td><td>' +
                      f"{r['alpha_diag'][0]:.3f}" + '</td><td>' + f"{r['log_scale'][0]:.2f}" +
                      '</td><td>' + f"{r['tanh_bias'][0]:.3f}" + '</td><td>' +
                      f"{r['b_gate']:.3f}" + '</td><td>' + f"{r['gate_bias']:.3f}" +
                      '</td><td>' + f"{r['mod_mlp']:.3f}" + '</td><td>' +
                      f"{r['mod_mem']:.3f}" + '</td><td>' + f"{r['gate_ema']:.3f}" + '</td><td>' +
                      (f"{r.get('w_help', 0):.3f}" if r.get('w_help') is not None else '—') +
                      '</td><td>' + (f"{r.get('w_contra', 0):.4f}" if r.get('w_contra') is not None else '—') +
                      '</td><td>' + (f"{r.get('pm_norm', 0):.3f}" if r.get('pm_norm') is not None else '—') +
                      '</td><td>' + (f"{r.get('w_sal', 0):.4f}" if r.get('w_sal') is not None else '—') +
                      '</td><td>' + (f"{r.get('w_intent', 0):.4f}" if r.get('w_intent') is not None else '—') +
                      '</td></tr>')
        ch.append('</table>')
    else:
        ch.append('<div class="dim">live пропущен (--no-live) — данные не собраны</div>')

    if metacog is not None:
        ch.append('<h2>META-COGNITION (ВСЕ буферы)</h2>')
        if metacog.get('bridge_readiness') is not None:
            ch.append(f'<div class="cards">'
                      f'<div class="card"><b>{metacog["bridge_readiness"]:.4f}</b>'
                      f'<span>bridge readiness</span></div>'
                      f'<div class="card"><b>{max(metacog["mat_gate"]):.4f}</b>'
                      f'<span>mat gate max</span></div>'
                      f'<div class="card"><b>{sum(metacog["mat_gate"]) / len(metacog["mat_gate"]):.4f}</b>'
                      f'<span>mat gate mean</span></div>'
                      f'<div class="card"><b>{max(metacog["mat_readiness"]):.4f}</b>'
                      f'<span>readiness max</span></div></div>')
        sig_names = ['temp', 'pred', 'smooth', 'sym', 'help']
        ch.append('<table><tr><th>L</th><th>pm_norm</th><th>gate_ema</th>'
                  '<th>w_help</th><th>w_contra</th><th>trust_diag</th><th>trust_max</th>'
                  '<th>concept_sim</th><th>behavior_div</th><th>meta_pm</th>'
                  '<th>signal temp</th><th>signal pred</th><th>signal smooth</th>'
                  '<th>signal sym</th><th>signal help</th></tr>')
        for r in metacog['layers']:
            sw = r.get('signal_w') or [None] * 5
            def _c(x):
                return f'{x:.4f}' if isinstance(x, (int, float)) else '—'
            ch.append('<tr><td>' + str(r['layer']) + '</td><td>' + _c(r.get('pm_norm')) +
                      '</td><td>' + _c(r.get('gate_ema')) + '</td><td>' + _c(r.get('w_help')) +
                      '</td><td>' + _c(r.get('w_contra')) + '</td><td>' + _c(r.get('trust_diag')) +
                      '</td><td>' + _c(r.get('trust_max')) + '</td><td>' + _c(r.get('concept_sim')) +
                      '</td><td>' + _c(r.get('behavior_div')) + '</td><td>' + _c(r.get('meta_private_mem')) +
                      '</td>' + ''.join(f'<td>{_c(sw[j] if j < len(sw) else None)}</td>'
                                       for j in range(5)) + '</tr>')
        ch.append('</table>')
        ch.append('<h2>MATURATION per-layer</h2>')
        ch.append('<table><tr><th>L</th><th>gate</th><th>readiness</th>'
                  '<th>pen_init</th><th>pen_ema</th></tr>')
        for i in range(len(metacog['mat_gate'])):
            def _f(x):
                return f'{x:.4f}'
            ch.append('<tr><td>' + str(i) + '</td><td>' + _f(metacog['mat_gate'][i]) +
                      '</td><td>' + _f(metacog['mat_readiness'][i]) + '</td><td>' +
                      _f(metacog['mat_pen_init'][i]) + '</td><td>' +
                      _f(metacog['mat_pen_ema'][i]) + '</td></tr>')
        ch.append('</table>')

    ch.append('<h2>ANOMALY TRACK</h2>')
    if anomaly:
        for line in anomaly['report']:
            flag = ''
            for f in ('WAKE', 'WATCH', 'PASS'):
                if line.strip().startswith(f'[{f}]') or (f + ' ') in line[:12]:
                    flag = f
                    break
            esc = H.escape(line)
            if flag:
                esc = esc.replace(f'[{flag}]', badge(flag), 1)
            ch.append(f'<div>{esc}</div>')
        if anomaly.get('prev_step') is not None:
            ch.append(f'<div class="dim">vs prev step {anomaly["prev_step"]}: '
                      f'||hp||max {anomaly["dhp"]:+.3f} | predMSEmax {anomaly["dpe"]:+.1f} '
                      f'| gate_min {anomaly["dgate"]:+.4f} | '
                      f'births {anomaly["births"]}</div>')
    elif not live_available:
        ch.append('<div class="dim">live пропущен (--no-live) — трекер недоступен</div>')
    else:
        ch.append('<div class="dim">нет данных</div>')

    ch.append('<h2>GRAD INFO</h2>')
    gi = live.get('grad_info') if live else None
    if gi:
        dead = gi.get('dead_pred')
        ch.append('<div>pred: ' +
                  ('<span class="bdg b-PASS">DEAD</span> — нет в aux / без градиента '
                   '(pred_k/hp детачатся в _pred_cache), на обучение не влияет'
                   if dead else '<span class="bdg b-WATCH">ALIVE</span> — требует градиента') +
                  '</div>')
        if gi.get('cos_global') is not None:
            cls = 'g' if gi['cos_global'] >= 0.2 else 'r'
            ch.append(f'<div>diversity aux={gi.get("diversity_aux", 0):.0f} — '
                      f'cos_sim(div, CE): global=<b class="{cls}">{gi["cos_global"]:.4f}</b> '
                      f'L16={gi["cos_l16"]:.4f} | ||gCE||={gi["n_ce"]:.2e} '
                      f'||gDIV||={gi["n_div"]:.2e} | scale(align)={gi["scale"]:.4f} (cap 10)</div>')
        else:
            ch.append('<div>diversity: не требует градиента или отсутствует</div>')
    else:
        ch.append('<div class="dim">grad info пропущен (--no-gradinfo)</div>')

    if bridge:
        ch.append('<h2>BRIDGE (intent bus)</h2>')
        s = bridge['salience']
        ch.append(f'<div>salience (от head): mean={s["mean"]:.2e} max/min={s["ratio"]:.1f} '
                  f'норм.энтропия={s["entropy_norm"]:.3f} '
                  f'(1=равномерно, 0=сфокус на токенах)</div>')
        ch.append(f'<div>intent stream cross-layer cosine={bridge["cross_layer_cos"]:.4f} '
                  f'(bus redundancy) &nbsp; bus_norm={bridge["bus_norm"]}</div>')
        ch.append(f'<div>head stencil bus_head_proj weight_norm='
                  f'{bridge["stencil_w_norm"]:.4f} (0 = не обучен)</div>')
        ch.append(f'<div>tau_intent ladder: alpha[0]={bridge["alpha"][0]:.4f} '
                  f'alpha[-1]={bridge["alpha"][-1]:.4f}</div>')

    if head:
        ch.append('<h2>HEAD</h2>')
        bd = head['bias_decomp']
        cls = 'g' if bd['pct'] < 90 else 'r'
        ch.append(f'<div>BIAS-DECOMP: argmax==bias <b class="{cls}">{bd["match"]}/{bd["total"]} '
                  f'({bd["pct"]:.1f}%)</b> &nbsp; ctx: word={bd["ctx_word"]} punct={bd["ctx_punct"]} '
                  f'&nbsp; full: word={bd["full_word"]} punct={bd["full_punct"]}</div>')
        ch.append('<table><tr><th>pos</th><th>top-1</th><th>H</th><th>punct%</th><th>word%</th></tr>')
        for b in head['posmap']:
            ch.append(f'<tr><td>{b["label"]}</td><td>{b["top1"]:.4f}</td><td>{b["H"]:.3f}</td>'
                      f'<td>{b["punct"]:.1f}%</td><td>{b["word"]:.1f}%</td></tr>')
        ch.append('</table>')
        ch.append('<table><tr><th>mode</th><th>top-1</th><th>H</th><th>sampler top-1</th><th>sampler H</th></tr>')
        for m, v in head['ab'].items():
            ch.append(f'<tr><td>{m}</td><td>{v["top1"]:.4f}</td><td>{v["H"]:.3f}</td>'
                      f'<td>{v["s_top1"]:.4f}</td><td>{v["s_H"]:.3f}</td></tr>')
        ch.append('</table>')
        ch.append('<div class="dim">token_bias top: ' +
                  ', '.join(H.escape(repr(t)) for t in head['token_bias_top']) +
                  f' &nbsp; log_temp={head["log_temp"]:.4f} t_eff={head["t_eff"]:.4f}</div>')

    ch.append('<h2>ALL PARAMETERS (grouped — ВСЕ, вкл. новые)</h2>')
    ch.append('<table><tr><th>name</th><th>shape</th><th>n</th><th>mean</th>'
              '<th>std</th><th>min</th><th>max</th></tr>')
    for g in _grouped_param_stats(model):
        ch.append('<tr><td>' + H.escape(g['name']) + '</td><td>' + str(g['shape']) +
                  '</td><td>' + f"{g['n']}" + '</td><td>' + f"{g['mean']:+.5f}" +
                  '</td><td>' + f"{g['std']:+.5f}" + '</td><td>' + f"{g['min']:+.4f}" +
                  '</td><td>' + f"{g['max']:+.4f}" + '</td></tr>')
    ch.append('</table>')

    ch.append('<p class="dim">generated by analyze.py &mdash; ' +
              f'step {step}, best val {best:.4f}, {params:.2f}M params</p>')
    ch.append('</body></html>')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ch))
    print(f'\nHTML report: {out}')


# ─────────────────────────── MAIN ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description='EVA checkpoint analyzer (все методы + лог)')
    ap.add_argument('checkpoints', nargs='*', help='path(s) to .pt (optional if --log given)')
    ap.add_argument('--no-live', action='store_true', help='skip live forward dissection')
    ap.add_argument('--no-gradinfo', action='store_true',
                    help='skip dead_pred/cos_sim(diversity,CE) grad diagnostics (slower)')
    ap.add_argument('--no-head', action='store_true', help='skip head analysis (bias/posmap/A-B)')
    ap.add_argument('--quick', action='store_true', help='static + wake only')
    ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--prompt', type=str, default='')
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--seq', type=int, default=128, help='live forward seq_len')
    ap.add_argument('--no-html', action='store_true', help='skip HTML report generation')
    ap.add_argument('--log', type=str, default='',
                    help='path to Colab training log (.txt) -> полный HTML-дашборд ВСЕХ метрик')
    args = ap.parse_args()
    if not args.checkpoints and not args.log:
        ap.error('нужен хотя бы один checkpoint или --log PATH')

    tok = None
    if not args.no_head and not args.quick:
        try:
            from scripts.generate import load_russian_tokenizer
            tok = load_russian_tokenizer()
        except Exception as e:
            print(f'[warn] токенизатор недоступен ({e}), head-анализ пропущен')
            tok = None

    models = {}
    for path in args.checkpoints:
        print(f'\n# {path}')
        try:
            ckpt, cfg, model, missing, unexpected = load_ckpt(path)
        except Exception as e:
            print(f'[error] не удалось загрузить: {e}')
            continue
        ckpt['_path'] = path
        models[path] = (model, ckpt)
        try:
            run_envelope(ckpt)
        except Exception as e:
            print(f'[error] envelope: {e}')
        static_data = None
        try:
            static_data = run_static(ckpt, cfg, model, missing, unexpected, tok)
        except Exception as e:
            print(f'[error] static: {e}')
        try:
            mech_data = run_mech(model, cfg)
            if static_data is not None:
                static_data['mech'] = mech_data
        except Exception as e:
            print(f'[error] mech: {e}')
        try:
            wake_data = run_wake(model, ckpt)
        except Exception as e:
            print(f'[error] wake: {e}')
            wake_data = None
        live_data = None
        head_data = None
        anomaly_data = None
        if not args.quick:
            if not args.no_live:
                try:
                    live_data = run_live(model, cfg, seq=args.seq, gradinfo=not args.no_gradinfo)
                except Exception as e:
                    print(f'[error] live: {e}')
                if live_data is not None and wake_data is not None:
                    try:
                        anomaly_data = run_anomaly(live_data, wake_data, ckpt)
                    except Exception as e:
                        print(f'[error] anomaly: {e}')
            if tok is not None:
                try:
                    head_data = run_head(model, ckpt, args, tok)
                except Exception as e:
                    print(f'[error] head: {e}')
        bridge_data = None
        if not args.quick and getattr(model, 'intent_bridge', False):
            try:
                bridge_data = run_bridge(model, cfg, seq=args.seq)
            except Exception as e:
                print(f'[error] bridge: {e}')
        if not args.no_html and wake_data is not None:
            try:
                metacog = run_metacog(model, cfg) if not args.quick else None
            except Exception as e:
                print(f'[error] metacog: {e}')
                metacog = None
            try:
                save_html_report(ckpt, cfg, model, wake_data, live_data, head_data,
                                 anomaly_data, bridge_data, metacog=metacog,
                                 static=static_data)
            except Exception as e:
                print(f'[error] html: {e}')

    if len(models) > 1 and tok is not None:
        try:
            run_cmp(models, args, tok)
        except Exception as e:
            print(f'[error] cmp: {e}')

    if args.log:
        try:
            print(f'\n# parsing log: {args.log}')
            log_data = parse_training_log(args.log)
            n = len(log_data['steps'])
            print(f'  parsed steps={n}  aux_metrics={len(log_data["aux"])}  '
                  f'eval={len(log_data["eval"])}  depth={len(log_data["depth"])}')
            run_health(log_data)
            if n == 0:
                print('[warn] основные строки step= не найдены — проверьте формат лога')
            out = os.path.splitext(args.log)[0] + '_log_report.html'
            render_log_html(log_data, out)
        except Exception as e:
            print(f'[error] log: {e}')


if __name__ == '__main__':
    main()