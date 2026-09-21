"""scripts/probe_introspection.py — P4-2: подписной эксперимент «верная интроспекция».

Протокол (ничего не дообучается на тесте):
  1. Стриминг hold-out в AR-режиме (prefill окна + инкрементный L=1, state
     несётся; P0-5/6: зеркала/эмбеддинг в AR-режиме, один токен = одна запись).
  2. K позиций t*: инъекция коррупции двух видов —
     (a) токенная: x[t*] ← случайный токен;
     (b) латентная: выход слоя l в позиции t* += eps·randn/√D (forward-hook).
  3. Ряды истинных сигналов: ell_rel, conflict, pen, gate_mean, chi_time, ‖h‖.
  4. Метрики против размеченных t*:
     - z-пик (max в [t*, t*+64] в робастных σ контроля), задержка пика;
     - FAR = доля контрольных окон той же длины с z > 2;
     - AUC = P(z-пик инъекции > z-пик контроля) — оконный, без sklearn;
     - R²(h→сигнал): ridge-проба, fit на первой 60% позиций, оценка на второй.
       (Сигнал с высоким R² «вербализуем через h»; низкий R² при высоком AUC —
       «состояние знает, токенный канал — нет» — тезис привилегированного канала.)

Usage:
  python scripts/probe_introspection.py --ckpt checkponts/best.pt \\
      --file data/holdout.bin --seq 384 --steps 256 \\
      --inject-token 6 --inject-latent 4 --eps 1.0 --layer 12
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig          # noqa: E402
from core.stack import EVAStack            # noqa: E402

SIGNALS = ('ell_rel', 'conflict', 'pen', 'gate', 'chi', 'hnorm')
WIN = 64          # окно детекции после t*
Z_THR = 2.0       # порог спайка (робастные σ контроля)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _signals(model, out, head):
    sig = {
        'ell_rel': _f(getattr(head, '_last_lacuna_rel', None)),
        'conflict': _f(getattr(head, '_last_conflict', None)),
        'pen': float('nan'), 'gate': float('nan'), 'chi': float('nan'),
        'hnorm': _f(out.norm(dim=-1).mean()),
    }
    pens, gates, chis = [], [], []
    for l in model.layers:
        m = getattr(l, 'mirror', None)
        v = getattr(m, '_cached_pred_error_norm', None) if m is not None else None
        if isinstance(v, torch.Tensor):
            pens.append(float(v.mean()))
        v = getattr(m, '_cached_gate', None) if m is not None else None
        if isinstance(v, torch.Tensor):
            gates.append(float(v.mean()))
        v = getattr(l, '_chi_time', None)
        if isinstance(v, torch.Tensor):
            chis.append(float(v.mean()))
    if pens:
        sig['pen'] = sum(pens) / len(pens)
    if gates:
        sig['gate'] = sum(gates) / len(gates)
    if chis:
        sig['chi'] = sum(chis) / len(chis)
    return sig


def _latent_hook(eps, D):
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            h = h + eps * torch.randn_like(h) / math.sqrt(D)
            return (h,) + tuple(out[1:])
        return out + eps * torch.randn_like(out) / math.sqrt(D)
    return hook


@torch.no_grad()
def stream_run(model, tokens, seq, steps, device, inject=None):
    """Один AR-прогон. inject = {'kind': 'token'|'latent', 't': int, 'tok': int,
    'eps': float, 'layer': int}. Возвращает (rec: dict[str,list], hs: (T,D))."""
    model.eval()
    model.reset_streams()
    model.reset_cache()
    head = model.lm_head
    ctx = tokens[:seq].view(1, -1).to(device)
    h = model.embed_tokens(ctx)
    out, state, gs, rb = model(h, None, global_state=None, adaptive=False,
                               step=0, tokens=ctx)
    model.observe_output(head(out))
    for l in model.layers:
        m = getattr(l, 'mirror', None)
        if m is not None:
            m._ar_mode = True
    if getattr(model, 'embed', None) is not None:
        model.embed._ar_mode = True
    rec = {k: [] for k in SIGNALS}
    hs = []
    hook = None
    if inject is not None and inject['kind'] == 'latent':
        hook = model.layers[int(inject['layer'])].register_forward_hook(
            _latent_hook(float(inject['eps']), model.cfg.D))
    try:
        for i in range(steps):
            t = seq + i
            tok = tokens[t].view(1, 1).to(device)
            if (inject is not None and inject['kind'] == 'token'
                    and i == int(inject['t'])):
                tok = torch.tensor([[int(inject['tok'])]], device=device)
            h1 = model.embed_tokens(tok)
            out, state, gs, rb = model(h1, state, global_state=gs,
                                       adaptive=False, step=i + 1, tokens=tok)
            model.observe_output(head(out))
            sig = _signals(model, out, head)
            for k in SIGNALS:
                rec[k].append(sig[k])
            hs.append(out[0, 0].float().cpu())
    finally:
        if hook is not None:
            hook.remove()
    return rec, torch.stack(hs)


def _local_z(series, warm=WIN, W=128):
    """Локальный z: (x_t − rolling_median)/(1.4826·rolling_MAD) по трейлингу W.
    Глобальная робастная σ ловит дрейф как ложные спайки; локальная — честная
    детекция ТРАНЗИЕНТА относительно собственного рабочего уровня (доктрина
    самокалибровки M55b). Прогрев (первые warm позиций) — nan."""
    x = torch.tensor(series, dtype=torch.float64)
    z = torch.full_like(x, float('nan'))
    for i in range(max(warm, 16), x.numel()):
        s = x[max(0, i - W):i]
        s = s[torch.isfinite(s)]
        if s.numel() < 16:
            continue
        med = s.median()
        mad = (s - med).abs().median() * 1.4826
        sd = float(mad) if float(mad) > 1e-9 else float(s.std()) + 1e-9
        z[i] = (x[i] - med) / sd
    return z


def _peak_z_series(z, t0, win=WIN):
    w = z[max(0, t0):min(z.numel(), t0 + win)]
    w = w[torch.isfinite(w)]
    if w.numel() == 0:
        return float('nan'), -1
    k = int(torch.argmax(w))
    return float(w[k]), k


def _window_peaks(z, win=WIN):
    out = []
    for s in range(WIN, max(WIN + 1, z.numel() - win), win // 2):
        w = z[s:s + win]
        w = w[torch.isfinite(w)]
        if w.numel() == win:
            out.append(float(w.max()))
    return out


def _auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    n = 0.0
    for a in pos:
        for b in neg:
            n += 1.0 if a > b else (0.5 if a == b else 0.0)
    return n / (len(pos) * len(neg))


def _ridge_r2(X, y, frac=0.6, lam=1e-2):
    m = torch.isfinite(y)
    X, y = X[m], y[m]
    if X.shape[0] < 32:
        return float('nan')
    X = X - X.mean(0, keepdim=True)
    y = y - y.mean()
    sd = X.std(0, keepdim=True).clamp_min(1e-6)
    X = X / sd
    n = int(X.shape[0] * frac)
    Xt, yt = X[:n], y[:n]
    Xv, yv = X[n:], y[n:]
    A = Xt.T @ Xt + lam * torch.eye(X.shape[1], dtype=X.dtype)
    w = torch.linalg.solve(A, Xt.T @ yt)
    pred = Xv @ w
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--file', required=True, help='hold-out .bin (uint16)')
    ap.add_argument('--seq', type=int, default=384)
    ap.add_argument('--steps', type=int, default=256)
    ap.add_argument('--inject-token', type=int, default=6)
    ap.add_argument('--inject-latent', type=int, default=4)
    ap.add_argument('--eps', type=float, default=1.0)
    ap.add_argument('--layer', type=int, default=-1)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck.get('cfg')
    if not isinstance(cfg, EVAConfig):
        raise SystemExit('чекпойнт без cfg (EVAConfig) — probe невозможен')
    model = EVAStack(cfg)
    miss, unexp = model.load_state_dict(ck['model'], strict=False)
    if unexp:
        print(f'[probe] unexpected keys: {len(unexp)}')
    model.to(args.device)
    model.eval()
    layer = args.layer if args.layer >= 0 else cfg.n_layers // 2
    print(f'[probe] ckpt step={ck.get("step")} val={ck.get("best_val_loss")} '
          f'| D={cfg.D} L={cfg.n_layers} | latent layer={layer}')

    data = np.fromfile(args.file, dtype=np.uint16).astype(np.int64)
    off = max(len(data) // 4, args.seq + args.steps + 8)
    if off + args.seq + args.steps >= len(data):
        raise SystemExit('файл слишком мал для hold-out сегмента')
    toks = torch.from_numpy(data[off:off + args.seq + args.steps])

    # 1. контрольный прогон: локальные z-ряды (без прогрева)
    clean, hs = stream_run(model, toks, args.seq, args.steps, args.device)
    z_clean = {k: _local_z(clean[k]) for k in SIGNALS}

    # 2. инъекции (равномерно по доступным позициям)
    lo, hi = WIN, args.steps - WIN - 1
    pos_t = np.linspace(lo, hi, max(args.inject_token, 1)).astype(int).tolist()
    pos_l = np.linspace(lo, hi, max(args.inject_latent, 1)).astype(int).tolist()
    runs = []
    for i, t0 in enumerate(pos_t):
        runs.append(('token', t0, stream_run(
            model, toks, args.seq, args.steps, args.device,
            inject={'kind': 'token', 't': t0,
                    'tok': int(torch.randint(3, cfg.vocab, (1,)).item())})[0]))
    for i, t0 in enumerate(pos_l):
        runs.append(('latent', t0, stream_run(
            model, toks, args.seq, args.steps, args.device,
            inject={'kind': 'latent', 't': t0, 'eps': args.eps,
                    'layer': layer})[0]))

    # контрольные окна: максимумы z по скользящим окнам чистого прогона
    ctrl_windows = {k: _window_peaks(z_clean[k]) for k in SIGNALS}

    print('\n=== детекция коррупции (z-пик / задержка / AUC / FAR) ===')
    print(f'{"signal":<9} {"kind":<6} {"z_peak":>8} {"delay":>6} {"AUC":>6} {"FAR":>6}')
    for kind in ('token', 'latent'):
        for k in SIGNALS:
            zs, ds = [], []
            for kd, t0, rec in runs:
                if kd != kind:
                    continue
                z, d = _peak_z_series(_local_z(rec[k]), t0)
                if math.isfinite(z):
                    zs.append(z)
                    ds.append(d)
            if not zs:
                continue
            far = sum(1 for z in ctrl_windows[k] if z > Z_THR) / max(len(ctrl_windows[k]), 1)
            auc = _auc(zs, ctrl_windows[k])
            print(f'{k:<9} {kind:<6} {np.mean(zs):>8.2f} {np.mean(ds):>6.1f} '
                  f'{auc:>6.3f} {far:>6.3f}')

    # 3. R²(h→сигнал): ridge-проба без обучения ствола
    print('\n=== декодируемость из h (ridge, fit 60% / eval 40%) ===')
    for k in SIGNALS:
        y = torch.tensor(clean[k], dtype=torch.float32)
        r2 = _ridge_r2(hs, y)
        extra = ''
        mh = getattr(model, 'meta_head', None)
        if mh is not None and k != 'hnorm':
            with torch.no_grad():
                pred = mh(hs.to(args.device)).cpu()
            idx = {'ell_rel': 0, 'conflict': 1, 'pen': 2, 'gate': 3, 'chi': 4}.get(k)
            if idx is not None:
                m = torch.isfinite(y)
                if int(m.sum()) > 32:
                    p, t = pred[m, idx], y[m]
                    ss = float(((t - p) ** 2).sum())
                    st = float(((t - t.mean()) ** 2).sum()) + 1e-12
                    extra = f' | MetaHead R²={1.0 - ss / st:+.3f}'
        print(f'  {k:<9} R²(h)={r2:+.3f}{extra}')
    print('\nИнтерпретация: AUC>0.9 — «модель замечает коррупцию»; низкий R²(h) '
          'при высоком AUC — «состояние знает, вербальный канал — нет».')


if __name__ == '__main__':
    main()
