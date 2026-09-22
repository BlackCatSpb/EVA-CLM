"""scripts/probe_frozen_head.py — EXT §6-B: frozen-h head probe.

Сколько РЕАЛЬНАЯ голова может выжать из текущего ствола? Собираем h (выход
ствола) на hold-out окнах, замораживаем ствол и дообучаем ТОЛЬКО голову
(readout/bit_bias/token_bias/log_temp/emphasis_gain/phantom/lacuna) на парах
(h, next-token) ровно тем же CE-путём, что обучение
(`lm_head.log_probs_for_target(h, targets, bus_bias)` — включая стенсил шины).

Интерпретация (EXT §6, таблица 2×2):
  CE_B ≪ val  -> голова недоучена/в хедж-ловушке (патология h→u; лечится
                 head-фокусом: LR головы, снятие gain<0, pair);
  CE_B ≈ val  -> ствол несёт мало сигнала (лечится данными/стволом).

Сравнение с потолком семейства (probe_head_ceiling.py --bias: 6.75 на WAR)
даёт вторую ось: достижима ли вообще такая CE на реальном h.

Run:
  python scripts/probe_frozen_head.py --ckpt "checkponts\\best 13640.pt" \
      --file wb/token_stream_WAR_eos.bin --windows 12 --steps 600 --lr 3e-4
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


def collect(model, tokens, seq, n_windows, step0, device):
    """(h, bus_bias, targets) по окнам; состояние сбрасывается как в eval."""
    xs, bs, ys = [], [], []
    model.reset_streams()
    model.reset_cache()
    state = gs = None
    with torch.no_grad():
        for i in range(n_windows):
            off = i * seq
            x = tokens[off:off + seq].view(1, -1).to(device)
            y = tokens[off + 1:off + seq + 1].view(1, -1).to(device)
            h_emb = model.embed_tokens(x)
            # как eval: состояние НЕСЁТСЯ между окнами документа, step один
            out, state, gs, _ = model(h_emb, state, global_state=gs,
                                      adaptive=False, step=step0, tokens=x)
            bb = None
            _bus = getattr(model, '_last_bus', None)
            if _bus is not None and getattr(model, 'bus_head_proj', None) is not None:
                B, L = x.shape
                _b = _bus.expand(B, L, -1, -1).reshape(B, L, -1)
                bb = model.bus_head_proj(_b).reshape(B * L, 1, -1).detach().cpu()
            xs.append(out.reshape(-1, out.shape[-1]).detach().cpu())
            ys.append(y.reshape(-1).detach().cpu())
            bs.append(bb)
            print(f'  window {i + 1}/{n_windows}: h {tuple(xs[-1].shape)} '
                  f'bus={"yes" if bb is not None else "no"}', flush=True)
    return xs, bs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--file', required=True)
    ap.add_argument('--seq', type=int, default=384)
    ap.add_argument('--windows', type=int, default=12)
    ap.add_argument('--train-windows', type=int, default=8)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck.get('cfg')
    step0 = int(ck.get('step', 0) or 0)
    model = EVAStack(cfg)
    miss, unexp = model.load_state_dict(ck['model'], strict=False)
    model.to(args.device)
    model.eval()
    print(f'[frozen-h] ckpt step={step0} val={ck.get("best_val_loss")} '
          f'| missing={len(miss)} unexpected={len(unexp)}')

    import numpy as np
    data = np.fromfile(args.file, dtype=np.uint16).astype(np.int64)
    off0 = max(len(data) // 4, args.seq + 1)
    toks = torch.from_numpy(data[off0:off0 + args.windows * args.seq + 1])
    print(f'[frozen-h] tokens from {off0:,}: {toks.numel():,} '
          f'({args.windows} окон x {args.seq})')

    t0 = time.time()
    xs, bs, ys = collect(model, toks, args.seq, args.windows, step0, args.device)
    print(f'[frozen-h] collection: {time.time() - t0:.0f}s', flush=True)
    # интерлив: чётные окна -> train, нечётные -> eval (без контентного сдвига)
    tr_i = [i for i in range(args.windows) if i % 2 == 0]
    ev_i = [i for i in range(args.windows) if i % 2 == 1]
    X_tr = torch.cat([xs[i] for i in tr_i]); Y_tr = torch.cat([ys[i] for i in tr_i])
    X_ev = torch.cat([xs[i] for i in ev_i]); Y_ev = torch.cat([ys[i] for i in ev_i])
    B_tr = torch.cat([bs[i] for i in tr_i]) if bs and bs[0] is not None else None
    B_ev = torch.cat([bs[i] for i in ev_i]) if bs and bs[0] is not None else None
    print(f'[frozen-h] train positions={X_tr.shape[0]} eval={X_ev.shape[0]}')

    head = model.lm_head
    params = list({id(p): p for p in head.parameters()}.values())
    print(f'[frozen-h] head params: {sum(p.numel() for p in params):,} '
          f'({len(params)} тензоров)')

    def eval_ce():
        head.eval()
        with torch.no_grad():
            lp = head.log_probs_for_target(X_ev, Y_ev, bus_bias=B_ev)
            return float(-lp.mean())

    ce0 = eval_ce()
    print(f'[frozen-h] CE до дообучения: eval={ce0:.4f} (val модели=7.93)')

    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=args.lr)
    head.train()
    n = X_tr.shape[0]
    t0 = time.time()
    for st in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch,))
        bb = B_tr[idx] if B_tr is not None else None
        lp = head.log_probs_for_target(X_tr[idx], Y_tr[idx], bus_bias=bb)
        loss = -lp.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], 1.0)
        opt.step()
        if st % 50 == 0:
            print(f'  step {st}/{args.steps} train CE={float(loss.detach()):.4f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    ce1 = eval_ce()
    print('-' * 62)
    print(f'CE_B (frozen-h, голова дообучена): eval={ce1:.4f}')
    print(f'CE   (та же голова без дообучения): eval={ce0:.4f}')
    print(f'val модели (полный forward)       : 7.9260')
    print(f'потолок семейства (probe --bias)  : 6.7527 (bigram-статистика WAR)')
    print('-' * 62)
    if ce1 < 7.6:
        print('VERDICT: голова ВЫЖИМАЕТ из текущего h заметно больше, чем модель '
              '(head-optimization gap) — патология h→u / хедж-ловушка.')
    else:
        print('VERDICT: даже дообученная голова не выжимает больше — ствол несёт '
              'мало сигнала на этом объёме данных (лечится данными).')


if __name__ == '__main__':
    main()
