"""Сравнение динамического пер-экспертного гейта mlp_mod с градиент-реактивной целью
gradalign: g_target = ||dCE/d mlp_out|| (по экспертам).

Для двух чекпоинтов (напр. step_2961 — после gradalign, и best.pt — до):
  - per-layer модуляция mlp_mod: mean и std ПО ЭКСПЕРТАМ (G) — насколько гейт варьируется
    между экспертами (при заморозке все эксперты ~одинаковы, std мал);
  - alignment = 1 - MSE( mlp_mod_norm , g_target_norm ) после per-group нормировки
    (как в L_gradalign). Чем выше — тем лучше гейт отслеживает, где выход MLP реально
    меняет лосс.

Использование:
  python scripts/diag_gate_gradalign.py <ckpt_A> <ckpt_B> [--batch 2] [--seq 128] [--nb 2]
"""
import sys, os, math
import torch
from torch.serialization import add_safe_globals

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from core import WideBindConfig, WideBindStack
add_safe_globals([WideBindConfig])

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass


def load(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ckpt['cfg']
    model = WideBindStack(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.train()
    return ckpt, cfg, model


def probe_by_layer(model, cfg, batch=2, seq=128, nb=2):
    rows = {i: [] for i in range(len(model.layers))}
    for _ in range(nb):
        x = torch.randint(0, cfg.vocab, (batch, seq))
        h = model.embed_tokens(x)
        out, _, _, _ = model(h)
        tgt = torch.randint(1, cfg.vocab, (batch, seq))
        ce, _ = model.compute_losses(out[:, :-1], tgt[:, 1:])

        outs, mods = [], []
        for l in model.layers:
            o = getattr(l, '_cache_mlp_out', None)
            m = getattr(l, '_cache_mlp_mod', None)
            if o is not None and m is not None:
                outs.append(o); mods.append(m)
            else:
                outs.append(None); mods.append(None)

        valid = [(o, m) for o, m in zip(outs, mods) if o is not None and m is not None]
        grads = torch.autograd.grad(ce, [o for o, _ in valid], retain_graph=True, allow_unused=True)
        gi = 0
        for i, (o, m) in enumerate(zip(outs, mods)):
            if o is None or m is None:
                continue
            g = grads[gi]; gi += 1
            if g is None:
                g = torch.zeros_like(o)
            gt = g.reshape(g.shape[0], g.shape[1], -1, g.shape[-1]).pow(2).sum(-1).sqrt()
            gt_n = gt / (gt.max().detach() + 1e-8)
            m_n = m / (m.max().detach() + 1e-8)
            align = 1.0 - ((m_n - gt_n) ** 2).mean().item()
            mod_mean = m.mean().item()
            mod_std = m.std(dim=2).mean().item()
            rows[i].append((mod_mean, mod_std, align))
    return {i: (sum(r[0] for r in v) / len(v),
                sum(r[1] for r in v) / len(v),
                sum(r[2] for r in v) / len(v)) for i, v in rows.items() if v}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--nb', type=int, default=2)
    args = ap.parse_args()

    results = {}
    for path in args.ckpts:
        ckpt, cfg, model = load(path)
        rows = probe_by_layer(model, cfg, batch=args.batch, seq=args.seq, nb=args.nb)
        results[path] = rows
        del model

    keys = list(results.keys())
    print(f'\n{"L":>3s} | ' + ' | '.join(
        f'{os.path.basename(k):<34s}' for k in keys))
    print(f'{"":>3s} | ' + ' | '.join(
        f'{"mod_mean mod_std align":<34s}' for _ in keys))
    print('-' * (6 + 38 * len(keys)))
    n = len(next(iter(results.values())))
    for i in range(n):
        cells = []
        for k in keys:
            mm, ms, al = results[k].get(i, (float('nan'),) * 3)
            cells.append(f'{mm:7.4f} {ms:7.4f} {al:7.4f}')
        print(f'{i:3d} | ' + ' | '.join(f'{c:<34s}' for c in cells))

    # сводка по моделям
    print('\nСВОДКА (усреднено по слоям):')
    for k in keys:
        vals = list(results[k].values())
        am = sum(v[0] for v in vals) / len(vals)
        ast = sum(v[1] for v in vals) / len(vals)
        aa = sum(v[2] for v in vals) / len(vals)
        print(f'  {os.path.basename(k):<34s} mod_mean={am:.4f} mod_std(expert)={ast:.4f} '
              f'align(gate~g_target)={aa:.4f}')


if __name__ == '__main__':
    main()
