"""probe_branch_visibility.py — где теряется сигнал к ветвям?

MOTIVATION
----------
`docs/ARCHITECTURE_JOURNAL.md:2288-2321` (эксперименты C1/v1) сообщает на
чекпоинте 18920:

    dCE/dh_final = 0.2115 ,  bind L0 = 5.95e-14 ,  mirror L0 = 3.57e-13
    -> "сигнал к ветке на 9-13 порядков слабее, перекос в глубокие слои ~5000x"
    -> "ветки СТРУКТУРНО невидимы лоссу"

Однако README (`Текущее состояние`) фиксирует, что зрелость МЕЛКИХ слоёв равна
0.18-0.21 ("L0-L3 стали проекционными"), а `mat_gate` масштабирует mirror, MLP,
bridge и запись memory bank. Если причина в гейтировании, то `5.95e-14` — не
структурное свойство архитектуры, а следствие закрытого гейта на мелких слоях.

Этот скрипт проверяет гипотезу ОДНИМ различием: тот же forward/backward, тот же
батч, тот же чекпоинт — с `maturation_enabled=True` и `=False`.

ЧТО ИЗМЕРЯЕТСЯ (по слоям)
-------------------------
  * `|dCE/dh_l|`          — норма градиента на ВЫХОДЕ слоя l (реальный сигнал,
                            доходящий до этого места ствола);
  * `|dCE/dh_l| / mean`   — баланс по глубине (README: "перекос deep-heavy");
  * `grad(bind)`, `grad(mirror)`, `grad(mlp)`, `grad(conv)` — нормы градиента
    параметров ветвей этого слоя;
  * `mat`                 — значение maturation-гейта слоя.

ЗАПУСК
------
    python scripts/probe_branch_visibility.py --ckpt checkponts/best.pt \
        --stream wb/token_stream_WAR_eos.bin --windows 6 --seq 384

Ожидаемое поведение при верной гипотезе:
  * `maturation_enabled=True`  -> |dCE/dh_l| на мелких слоях близко к нулю,
    градиент ветвей там же на порядки ниже, чем у головы;
  * `maturation_enabled=False` -> мелкие слои получают сигнал того же порядка,
    что и глубокие, и отношение к голове становится ~O(1).

ВАЖНО
-----
  * Скрипт НИЧЕГО не обучает: только forward + backward + счёт норм.
  * Он НЕ меняет файлы проекта и чекпоинт не перезаписывает.
  * `gradient_checkpointing` принудительно выключается ТОЛЬКО на время замера
    (иначе хуки исполняются дважды при recompute и нормы удваиваются).
    В каноническом профиле он и так `False`.
  * Замер требует видеопамяти как один training-шаг (B=1, без оптимизатора).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:                                        # noqa: E402
    from core.config import EVAConfig
except ImportError:                         # WideBind Mini naming
    from core.config import WideBindConfig as EVAConfig
try:                                        # noqa: E402
    from core.stack import EVAStack
except ImportError:                         # WideBind Mini naming
    from core.stack import WideBindStack as EVAStack


# ─────────────────────────── аргументы ───────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='путь к best.pt / step_*.pt')
    ap.add_argument('--stream', default=None,
                    help='uint16-поток токенов (*_eos.bin); если не задан — '
                         'используются случайные токены (менее информативно)')
    ap.add_argument('--seq', type=int, default=384, help='длина окна (default 384)')
    ap.add_argument('--windows', type=int, default=6, help='сколько окон усреднить')
    ap.add_argument('--start', type=int, default=0,
                    help='смещение в потоке, в токенах (default 0 — начало файла)')
    ap.add_argument('--steps', type=int, default=None,
                    help='шаг(и) обучения для maturation-гейта; default — шаг чекпоинта. '
                         'Можно задать несколько через запятую: 18920,6160')
    ap.add_argument('--device', default=None, help="cuda | cpu (default: авто)")
    ap.add_argument('--json', default=None, help='куда сохранить сводку JSON')
    ap.add_argument('--no-null-test', action='store_true',
                    help='не делать зануление ветвей (быстрее)')
    return ap.parse_args()


# ─────────────────────────── данные ───────────────────────────

def load_windows(args, vocab):
    """Возвращает список (1, seq) тензоров токенов."""
    if args.stream:
        arr = np.memmap(args.stream, dtype=np.uint16, mode='r')
        n = len(arr)
        if n < args.seq + 1:
            raise SystemExit(f'поток слишком мал: {n} токенов < {args.seq + 1}')
        w = []
        for i in range(args.windows):
            off = args.start + i * args.seq
            if off + args.seq + 1 > n:
                off = max(0, n - args.seq - 1)
            t = torch.from_numpy(np.asarray(arr[off:off + args.seq + 1], dtype=np.int64))
            w.append(t.unsqueeze(0))
        print(f'  данные: {args.stream}  ({n:,} токенов), окон={len(w)}, seq={args.seq}')
        return w
    g = torch.Generator().manual_seed(1234)
    w = [torch.randint(1, vocab, (1, args.seq), generator=g) for _ in range(args.windows)]
    print(f'  данные: СЛУЧАЙНЫЕ токены (--stream не задан), окон={len(w)}')
    return w


# ─────────────────────────── замер ───────────────────────────

def measure_arm(cfg, model, windows, device, step, null_test=True):
    """Один прогон замера. Возвращает dict с таблицами по слоям."""
    n_layers = len(model.layers)

    h_grads = defaultdict(list)          # layer -> [ |dCE/dh_l| по окнам ]
    h_norms = defaultdict(list)          # layer -> [ |h_l| ]
    branch_g = defaultdict(lambda: defaultdict(list))   # branch -> layer -> [norm]
    handles = []

    def make_hook(layer_idx):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.retain_grad()
                _store[layer_idx] = t
            return None
        return hook

    _store = {}
    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    model.train()
    torch.set_grad_enabled(True)

    for wi, tok in enumerate(windows):
        x = tok.to(device)
        model.zero_grad(set_to_none=True)
        _store.clear()

        h_in = model.embed_tokens(x)
        out = model(h_in)
        h_out = out[0] if isinstance(out, tuple) else out
        ce = model.compute_loss(h_out, x)
        ce.backward()

        # |dCE/dh_l| — реальный сигнал, дошедший до выхода слоя l
        ref = None
        for i in range(n_layers):
            t = _store.get(i)
            if t is None or t.grad is None:
                h_grads[i].append(float('nan'))
                continue
            g = float(t.grad.norm())
            h_grads[i].append(g)
            h_norms[i].append(float(t.detach().norm()))
            if i == n_layers - 1:
                ref = g
        # нормируем на ГЛУБОКИЙ слой (как в журнале: перекос к глубине)
        if ref:
            h_grads.setdefault('_ref', []).append(ref)

        # нормы градиента параметров ветвей, по слоям
        for i, layer in enumerate(model.layers):
            for name, p in layer.named_parameters():
                if p.grad is None:
                    continue
                v = float(p.grad.norm()) ** 2
                # Toplevel submodules of a layer in BOTH projects:
                # bind / mirror / conv / mlp (+ precision_gate, exact_memory).
                # NB: `mirror.conv_smooth.weight` must stay under `mirror`,
                # so match only the FIRST path segment.
                top = name.split('.')[0]
                if top == 'conv':
                    branch_g['conv'][i].append(v)
                elif top in ('bind', 'mirror', 'mlp'):
                    branch_g[top][i].append(v)
                else:
                    branch_g['other'][i].append(v)

    for hd in handles:
        hd.remove()

    def mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return sum(xs) / len(xs) if xs else float('nan')

    result = {
        'h_grad': {i: mean(h_grads[i]) for i in range(n_layers)},
        'h_norm': {i: mean(h_norms[i]) for i in range(n_layers)},
        'branch': {b: {i: math.sqrt(mean(v)) for i, v in d.items()}
                   for b, d in branch_g.items()},
        'ce': None,
    }

    # head градиент для опорного отношения
    head_sq = sum(float(p.grad.norm()) ** 2 for n, p in model.named_parameters()
                  if p.grad is not None and n.startswith('lm_head'))
    result['head'] = math.sqrt(head_sq)

    # зануление ветвей: |dh|/|h| и dCE
    if null_test:
        result['null'] = {}
        for attr in ('bind', 'mirror'):
            hs = []

            def mk(_attr):
                def hook(_m, _i, out):
                    if isinstance(out, tuple):
                        return (torch.zeros_like(out[0]),) + tuple(out[1:])
                    return torch.zeros_like(out)
                return hook

            for layer in model.layers:
                if hasattr(layer, attr):
                    hs.append(getattr(layer, attr).register_forward_hook(mk(attr)))
            model.zero_grad(set_to_none=True)
            x = windows[0].to(device)
            with torch.no_grad():
                h0 = model.embed_tokens(x)
                o0 = model(h0)
                base_h = o0[0] if isinstance(o0, tuple) else o0
                base_ce = float(model.compute_loss(base_h, x))
            for hd in hs:
                hd.remove()
            # с занулением (нужен граф только для CE)
            hs = []
            for layer in model.layers:
                if hasattr(layer, attr):
                    hs.append(getattr(layer, attr).register_forward_hook(mk(attr)))
            with torch.no_grad():
                h1 = model.embed_tokens(x)
                o1 = model(h1)
                n_h = o1[0] if isinstance(o1, tuple) else o1
                n_ce = float(model.compute_loss(n_h, x))
            for hd in hs:
                hd.remove()
            rel = float((n_h - base_h).norm() / base_h.norm().clamp_min(1e-12))
            result['null'][attr] = {'rel': rel, 'dCE': n_ce - base_ce}
            result.setdefault('_base_ce', base_ce)

    return result


# ─────────────────────────── вывод ───────────────────────────

def print_arm(tag, cfg, res, mat_gate=None):
    n = len(res['h_grad'])
    ref = max(v for v in res['h_grad'].values() if not math.isnan(v)) if n else float('nan')
    print()
    print('=' * 104)
    print(f'ARM: {tag}   (maturation_enabled={getattr(cfg, "maturation_enabled", None)})')
    print('=' * 104)
    print(f"{'layer':>5} {'|dCE/dh_l|':>13} {'/max':>8} {'|h_l|':>12} "
          f"{'grad bind':>12} {'grad mirror':>12} {'grad mlp':>12} "
          f"{'grad conv':>12} {'mat':>7}")
    for i in range(n):
        hg = res['h_grad'][i]
        b = res['branch'].get('bind', {}).get(i, float('nan'))
        m = res['branch'].get('mirror', {}).get(i, float('nan'))
        ml = res['branch'].get('mlp', {}).get(i, float('nan'))
        cv = res['branch'].get('conv', {}).get(i, float('nan'))
        mg = mat_gate[i] if mat_gate is not None else float('nan')
        print(f'{i:>5} {hg:>13.5e} {hg/ref if ref else float("nan"):>8.4f} '
              f'{res["h_norm"].get(i, float("nan")):>12.5e} '
              f'{b:>12.5e} {m:>12.5e} {ml:>12.5e} {cv:>12.5e} {mg:>7.4f}')
    print(f"\n  lm_head grad norm = {res['head']:.5e}")
    for br in ('bind', 'mirror', 'mlp', 'conv'):
        d = res['branch'].get(br, {})
        if d:
            tot = math.sqrt(sum(v * v for v in d.values()))
            print(f"  total {br:<7} grad norm = {tot:.5e}   head/{br} = "
                  f"{res['head'] / max(tot, 1e-30):.4e}")


def clone_cfg(cfg):
    """Копия конфига только по полям dataclass.

    `cfg.__dict__` несёт ещё и РАНТАЙМ-атрибуты, которых нет в конструкторе
    (`matur_bridge_readiness` и родственные, дописанные __post_init__/стеком) —
    их нельзя передавать в EVAConfig(**...). Фильтруем по dataclasses.fields,
    иначе TypeError: unexpected keyword argument.
    """
    import dataclasses
    names = {f.name for f in dataclasses.fields(cfg)}
    dropped = sorted(set(cfg.__dict__) - names)
    if dropped:
        print(f'  [info] clone_cfg: отброшено {len(dropped)} runtime-атрибутов '
              f'(например {dropped[:3]})')
    return EVAConfig(**{k: v for k, v in cfg.__dict__.items() if k in names})


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    ck_step = ck.get('step')
    print(f'checkpoint: {args.ckpt}')
    print(f'  step={ck_step}  best_val={ck.get("best_val_loss")}')
    print(f'  D={cfg.D} L={cfg.n_layers} vocab={cfg.vocab} G={cfg.mlp_groups} '
          f'K={cfg.bind_K} head_normalize={cfg.head_normalize}')
    import dataclasses
    _fields = {f.name for f in dataclasses.fields(cfg)}
    _has_mat = 'maturation_enabled' in _fields
    print(f'  maturation_enabled: {"field=" + str(getattr(cfg, "maturation_enabled", None)) if _has_mat else "НЕ ПОЛЕ этого конфига"}')
    print(f'  gradient_checkpointing={getattr(cfg, "gradient_checkpointing", None)}')
    if not _has_mat:
        print('  [WARN] у этого конфига нет поля maturation_enabled: обе руки будут')
        print('         идентичны, и замер НЕ проверяет гипотезу. Скрипт рассчитан')
        print('         на конфиг основного проекта (core/config.py:263).')

    steps = ([int(s) for s in str(args.steps).split(',')] if args.steps
             else [int(ck_step or 0)])

    windows = load_windows(args, cfg.vocab)

    summary = {'ckpt': args.ckpt, 'ckpt_step': ck_step, 'arms': {}}

    for step in steps:
        for mat in (True, False):
            # СВЕЖАЯ конфигурация на каждую руку: __post_init__ мутирует cfg,
            # а мы меняем флаг — порядок обхода не должен влиять на результат.
            cfg_arm = clone_cfg(cfg)
            cfg_arm.maturation_enabled = bool(mat)
            gc_saved = getattr(cfg_arm, 'gradient_checkpointing', False)
            cfg_arm.gradient_checkpointing = False   # хуки не должны дублироваться

            torch.manual_seed(0)
            model = EVAStack(cfg_arm)
            sd = ck.get('model') or ck.get('state_dict') or {}
            # Авто-миграция: чекпоинты WideBind Mini несут bind.W_out старой
            # ширины (128 при текущих 160) и падают на load_state_dict. В
            # основном проекте это делает train.py при resume; здесь вызываем
            # migrate_state_dict сами, чтобы probe работал на любом чекпоинте.
            try:
                from core.migrate import migrate_state_dict
                sd, changed = migrate_state_dict(dict(sd), model)
                if changed:
                    print(f'  [info] migrate_state_dict: мигрировано {changed} ключей')
            except Exception as e:                       # noqa: BLE001
                print(f'  [info] migrate_state_dict недоступен: '
                      f'{type(e).__name__}: {str(e)[:60]}')
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f'  [warn] load_state_dict: missing={len(missing)} '
                      f'unexpected={len(unexpected)}')
                if missing[:3]:
                    print('          missing sample:', missing[:3])
            model.to(device)

            mat_gate = None
            if getattr(model, 'maturation', None) is not None:
                with torch.no_grad():
                    g = model.maturation.step_gate(step, model._tau_l_dev.detach())
                    g = torch.maximum(g, model.maturation.readiness.detach().clone())
                mat_gate = [float(v) for v in g.cpu().tolist()]

            # ПРОВЕРКА, что переключатель вообще сработал. Без неё две руки
            # могут молча дать идентичные числа (флага нет в конфиге, или
            # контроллер не создан) — и результат будет ложно истолкован.
            ctrl = getattr(model, 'maturation', None)
            if ctrl is None and mat:
                print('  [WARN] maturation_enabled=True, но model.maturation is None '
                      '— контроллер не создан для этого конфига;')
                print('         рука maturation=True НЕ ОТЛИЧАЕТСЯ от maturation=False.')
            if ctrl is not None and not mat:
                print('  [WARN] maturation_enabled=False, но model.maturation существует '
                      '— флаг не читается этим стеком.')

            tag = f'{device} step={step} maturation={mat}'
            res = measure_arm(cfg_arm, model, windows, device, step,
                              null_test=not args.no_null_test)
            print_arm(tag, cfg_arm, res, mat_gate)

            if res.get('null'):
                print()
                print('  зануление ветвей (первое окно):')
                print(f"    baseline CE = {res.get('_base_ce'):.6f}")
                for k, v in res['null'].items():
                    print(f"    {k:<7} |dh|/|h| = {v['rel']:.3e}   dCE = {v['dCE']:+.6f}")

            summary['arms'][f'mat={mat}_step={step}'] = {
                'steps': step,
                'maturation_enabled': mat,
                'mat_gate': mat_gate,
                'h_grad': {str(k): v for k, v in res['h_grad'].items()},
                'h_norm': {str(k): v for k, v in res['h_norm'].items()},
                'head_grad': res['head'],
                'branch_total': {
                    b: math.sqrt(sum(v * v for v in d.values()))
                    for b, d in res['branch'].items()},
                'null': res.get('null'),
            }
            cfg_arm.gradient_checkpointing = gc_saved

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f'\nJSON: {args.json}')

    print()
    print('=' * 104)
    print('КАК ЧИТАТЬ')
    print('=' * 104)
    print("""
  * `|dCE/dh_l| / max` — баланс сигнала по глубине. Если при maturation=True
    мелкие слои дают ~1e-3 и меньше, а глубокие ~1 — это и есть "перекос
    deep-heavy" из журнала.
  * Если при maturation=False мелкие слои выравниваются к ~O(1), гипотеза
    ПОДТВЕРЖДЕНА: 5.95e-14 — следствие закрытого гейта, а не структурное
    свойство архитектуры. Тогда лечить надо РАСПИСАНИЕ созревания, а не ветки.
  * Если и при maturation=False мелкие слои остаются ~1e-14, гипотеза
    ОТВЕРГНУТА: причина в самом bind-пути (тогда смотреть w_bind_gate /
    капы / _stream_cap), и вывод "структурно невидимы" приобретает силу.
  * `head/bind` и `head/mirror` — прямые аналоги чисел журнала (0.2115 против
    5.95e-14 = 3.6e12). Здесь печатаются отношения норм по всему слою.
""")


if __name__ == '__main__':
    main()
