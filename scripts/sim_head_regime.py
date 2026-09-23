"""scripts/sim_head_regime.py — мини-симуляция: выходит ли голова в режим чтения
контекста при разных вариантах readout. Ревью-версия (по замечаниям внешнего
аудита EXT):

  * позитивные контроли ДО выводов: (a) эмпирическая биграмма (train->eval) —
    «сигнал в задаче есть»; (b) MLP(embed(prev)) -> softmax — harness учится;
    (c) EVA-ствол + обычная Linear(D,V)-голова — отделяет «кодовая голова
    блокирует» от «ствол не маршрутизирует»;
  * maturation открыта рано (matur_T0/delta/T_delay = 100) — иначе весь warmup
    идёт при зрелости ~0.1 (гейты зеркал закрыты) и застой не интерпретируем;
  * knob-liveness: пертурбация readout обязана менять CE (защита от «мёртвой
    ручки»);
  * eval каждые 500 шагов + ряды (CE, dctx, ||rec||/||h||, std(z), gain, log_temp);
  * руки: joint с нуля (block_tied / block_untied / full) и joint-продолжение
    из «застрявшего» warmup-состояния (block / full) — резюм-сценарий;
  * печать missing/unexpected при load_state_dict.

Задача: режимная цепочка (mode-switch): внутри режима токен берётся из своего
среза словаря с вероятностью conc (иначе uniform). Униграмма = uniform по
симметрии (~ln V); режимная цена ~ ln(slice); bigram раскрывает режим.

Run: python scripts/sim_head_regime.py [--conc 0.75] [--warmup 3000] [--joint 3000]
"""
import argparse
import math
import os
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


def make_stream(n, V, modes, mode_len, conc, seed):
    g = torch.Generator().manual_seed(seed)
    slice_sz = V // modes
    toks = torch.empty(n, dtype=torch.long)
    m = 0
    for i in range(n):
        if i % mode_len == 0:
            m = int(torch.randint(0, modes, (1,), generator=g))
        if float(torch.rand((), generator=g)) < conc:
            t = m * slice_sz + int(torch.randint(0, slice_sz, (1,), generator=g))
        else:
            t = int(torch.randint(0, V, (1,), generator=g))
        toks[i] = t
    return toks


def bigram_ce(tr, ev, V):
    bi = tr[:-1] * V + tr[1:]
    keys, cnts = torch.unique(bi, return_counts=True)
    uni = torch.bincount(tr, minlength=V).double()
    bt = ev[:-1] * V + ev[1:]
    pos = torch.searchsorted(keys, bt).clamp(0, keys.numel() - 1)
    vals = torch.where(keys[pos] == bt, cnts[pos],
                       torch.zeros(1, dtype=cnts.dtype)).double()
    pu = uni[ev[:-1]].double()
    return float(-torch.log((vals + 1.0) / (pu + V)).mean())


def _det(o):
    if torch.is_tensor(o):
        return o.detach()
    if isinstance(o, (list, tuple)):
        return type(o)(_det(x) for x in o)
    return o


def make_cfg(args, read_full=False):
    cfg = EVAConfig(D=args.D, vocab=args.vocab, n_layers=2, seq_len=args.seq,
                    code_dim=args.K, code_sparsity=args.S,
                    logit_cache_enabled=False, gradient_checkpointing=False)
    cfg.head_read_full = read_full
    # открыть зрелость рано (иначе гейты закрыты весь warmup — ревью аудита)
    cfg.matur_T0 = 100.0
    cfg.matur_T_delay = 100.0
    cfg.matur_delta = 100.0
    return cfg


def collect_h(model, toks, seq, step0, device):
    hs, ys = [], []
    model.reset_streams()
    model.reset_cache()
    state = gs = None
    with torch.no_grad():
        for off in range(0, toks.numel() - seq - 1, seq):
            x = toks[off:off + seq].view(1, -1).to(device)
            y = toks[off + 1:off + seq + 1].view(1, -1).to(device)
            h_emb = model.embed_tokens(x)
            out, state, gs, _ = model(h_emb, state, global_state=gs,
                                      adaptive=False, step=step0, tokens=x)
            hs.append(out.reshape(-1, out.shape[-1]).detach().cpu())
            ys.append(y.reshape(-1).detach().cpu())
    return torch.cat(hs), torch.cat(ys)


def head_ce(head, X, Y, bs=256):
    with torch.no_grad():
        tot, n = 0.0, 0
        for i in range(0, X.shape[0], bs):
            lp = head.log_probs_for_target(X[i:i + bs], Y[i:i + bs])
            tot += float(-lp.sum())
            n += int(lp.numel())
    return tot / max(n, 1)


def bias_ce(head, Y):
    with torch.no_grad():
        b = head.token_bias.detach().float()
        lz = torch.logsumexp(b, dim=-1)
        return float((lz - b[Y]).mean())


def geometry(head, X, bs=256):
    with torch.no_grad():
        rec_n, h_n, z_std = 0.0, 0.0, 0.0
        for i in range(0, X.shape[0], bs):
            xb = X[i:i + bs]
            if getattr(head, '_read_full', False):
                z = xb @ head.readout_full
                rec = z @ head.readout_full.T
            else:
                h_g = xb.reshape(xb.shape[0], head.K, -1)
                z = (h_g * head.readout.unsqueeze(0)).sum(-1)
                rec = (z.unsqueeze(-1) * head.readout).reshape(xb.shape[0], -1)
            rec_n += float(rec.norm(dim=-1).sum())
            h_n += float(xb.norm(dim=-1).sum())
            z_std += float(z.std()) * xb.shape[0]
        return rec_n / max(h_n, 1e-9), z_std / max(X.shape[0], 1)


def full_report(model, Xev, Yev, tag):
    hd = model.lm_head
    ca = head_ce(hd, Xev, Yev)
    cb = bias_ce(hd, Yev)
    geo, zstd = geometry(hd, Xev)
    gain = float(hd.emphasis_gain.detach().item()) if hasattr(hd, 'emphasis_gain') else float('nan')
    lt = float(hd.log_temp.detach().mean())
    print(f'  [{tag}] val={ca:.4f} bias-only={cb:.4f} dctx={cb - ca:+.4f} | '
          f'||rec||/||h||={geo:.4f} std(z)={zstd:.4f} gain={gain:+.3f} '
          f'log_temp={lt:+.3f}', flush=True)
    return dict(val=ca, dctx=cb - ca, geo=geo, gain=gain, lt=lt)


def train_joint(model, tr, args, steps, every, Xev, Yev, tag, log_every=1000):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = gs = None
    off = 0
    for st in range(1, steps + 1):
        if off + args.seq + 1 > tr.numel():
            off = 0
            state = gs = None
        x = tr[off:off + args.seq].view(1, -1).to(args.device)
        y = tr[off + 1:off + args.seq + 1].view(1, -1).to(args.device)
        h_emb = model.embed_tokens(x)
        out, state, gs, _ = model(h_emb, state, global_state=gs,
                                  adaptive=False, step=st, tokens=x)
        state = _det(state)
        gs = _det(gs)
        lp = model.lm_head.log_probs_for_target(out.reshape(-1, out.shape[-1]),
                                                y.reshape(-1))
        loss = -lp.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        off += args.seq
        if log_every and st % log_every == 0:
            print(f'    {tag} step {st}: train CE={float(loss.detach()):.4f}', flush=True)
        if every and st % every == 0:
            Xe, Ye = collect_h(model, Xev, args.seq, st, args.device)
            full_report(model, Xe, Ye, f'{tag}@{st}')
    return model


def control_mlp(tr, ev, args):
    """Позитивный контроль: MLP(embed(prev_token)) -> softmax (bigram-level)."""
    V = args.vocab
    emb = torch.nn.Embedding(V, 64)
    net = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.SiLU(),
                              torch.nn.Linear(256, V))
    opt = torch.optim.Adam(list(emb.parameters()) + list(net.parameters()), lr=3e-3)
    for st in range(1, 2001):
        i = torch.randint(0, tr.numel() - 2, (256,))
        x = emb(tr[i])
        logits = net(x)
        loss = torch.nn.functional.cross_entropy(logits, tr[i + 1])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        i = torch.arange(0, min(ev.numel() - 2, 20000))
        logits = net(emb(ev[i]))
        ce = float(torch.nn.functional.cross_entropy(logits, ev[i + 1]))
    print(f'[control] MLP(embed(prev)): eval CE={ce:.4f} (bigram-уровень)', flush=True)


def control_plain_head(tr, ev, args):
    """Позитивный контроль: тот же EVA-ствол + обычная Linear(D,V)-голова."""
    torch.manual_seed(args.seed + 7)
    model = EVAStack(make_cfg(args)).to(args.device)
    lin = torch.nn.Linear(args.D, args.vocab).to(args.device)
    opt = torch.optim.Adam(list(model.parameters()) + list(lin.parameters()), lr=args.lr)
    state = gs = None
    off = 0
    for st in range(1, args.joint + 1):
        if off + args.seq + 1 > tr.numel():
            off = 0
            state = gs = None
        x = tr[off:off + args.seq].view(1, -1).to(args.device)
        y = tr[off + 1:off + args.seq + 1].view(1, -1).to(args.device)
        h_emb = model.embed_tokens(x)
        out, state, gs, _ = model(h_emb, state, global_state=gs,
                                  adaptive=False, step=st, tokens=x)
        state = _det(state)
        gs = _det(gs)
        logits = lin(out.reshape(-1, args.D))
        loss = torch.nn.functional.cross_entropy(logits, y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(lin.parameters()), 1.0)
        opt.step()
        off += args.seq
    # eval
    model.eval()
    Xe, Ye = collect_h(model, ev, args.seq, st, args.device)
    with torch.no_grad():
        logits = lin(Xe)
        ce = float(torch.nn.functional.cross_entropy(logits, Ye))
    print(f'[control] EVA-ствол + Linear(D,V): eval CE={ce:.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warmup', type=int, default=3000)
    ap.add_argument('--joint', type=int, default=3000)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--D', type=int, default=256)
    ap.add_argument('--K', type=int, default=32)
    ap.add_argument('--S', type=int, default=3)
    ap.add_argument('--vocab', type=int, default=512)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--mode-len', type=int, default=24)
    ap.add_argument('--conc', type=float, default=0.75)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tr = make_stream(200_000, args.vocab, args.modes, args.mode_len, args.conc, 1)
    ev = make_stream(12_000, args.vocab, args.modes, args.mode_len, args.conc, 2)
    print(f'[sim] unigram ~ ln V = {math.log(args.vocab):.3f} | '
          f'режимная ~ ln(slice) = {math.log(args.vocab // args.modes):.3f} | '
          f'bigram(train->eval) = {bigram_ce(tr, ev, args.vocab):.4f}', flush=True)

    # ── позитивные контроли ──
    control_mlp(tr, ev, args)
    control_plain_head(tr, ev, args)

    # ── warmup (совместный, block+tied) ──
    torch.manual_seed(args.seed + 1)
    warm = EVAStack(make_cfg(args)).to(args.device)
    Xw, Yw = collect_h(warm, ev, args.seq, 0, args.device)
    print('[sim] warmup старт:', end=' ')
    full_report(warm, Xw, Yw, 'warm@0')
    train_joint(warm, tr, args, args.warmup, 1000, ev, None, 'warmup')
    warm_sd = {k: v.clone() for k, v in warm.state_dict().items()}
    Xw, Yw = collect_h(warm, ev, args.seq, args.warmup, args.device)
    warm_res = full_report(warm, Xw, Yw, 'warm@final')

    # ── joint с нуля: block_tied / block_untied / full ──
    print('[sim] === joint с нуля ===')
    for arm, full, untied in (('joint_block_tied', False, False),
                              ('joint_block_untied', False, True),
                              ('joint_full', True, False)):
        torch.manual_seed(args.seed + 3)
        m = EVAStack(make_cfg(args, read_full=full)).to(args.device)
        if untied:
            m.lm_head.readout = torch.nn.Parameter(m.lm_head.readout.detach().clone())
        # knob-liveness: пертурбация readout обязана менять CE
        X0, Y0 = collect_h(m, ev, args.seq, 0, args.device)
        ce_a = head_ce(m.lm_head, X0, Y0)
        with torch.no_grad():
            if getattr(m.lm_head, '_read_full', False):
                m.lm_head.readout_full.mul_(1.01)
            else:
                m.lm_head.readout.mul_(1.01)
        ce_b = head_ce(m.lm_head, X0, Y0)
        assert abs(ce_a - ce_b) > 1e-6, f'{arm}: readout-ручка мёртвая ({ce_a}=={ce_b})'
        with torch.no_grad():
            if getattr(m.lm_head, '_read_full', False):
                m.lm_head.readout_full.mul_(1.0 / 1.01)
            else:
                m.lm_head.readout.mul_(1.0 / 1.01)
        train_joint(m, tr, args, args.joint, 1000, ev, None, arm)
        Xe, Ye = collect_h(m, ev, args.seq, args.joint, args.device)
        full_report(m, Xe, Ye, f'{arm}@final')

    # ── joint-продолжение из warmup-состояния (резюм-сценарий) ──
    print('[sim] === joint-продолжение из warmup (резюм) ===')
    for arm, full in (('resume_block', False), ('resume_full', True)):
        torch.manual_seed(args.seed + 5)
        m = EVAStack(make_cfg(args, read_full=full)).to(args.device)
        miss, unexp = m.load_state_dict(warm_sd, strict=False)
        if full:
            W = m.lm_head.readout_full.data
            R = m.lm_head.readout.data
            d, K = int(R.shape[-1]), m.lm_head.K
            _ok = all(torch.equal(W[k * d:(k + 1) * d, k], R[k]) for k in range(K))
            assert _ok, 'readout_full обязан пере-синхронизироваться из загруженного readout'
            print('  [resume_full] readout_full re-synced from loaded readout: OK', flush=True)
        train_joint(m, tr, args, args.joint, 1000, ev, None, arm)
        Xe, Ye = collect_h(m, ev, args.seq, args.joint, args.device)
        full_report(m, Xe, Ye, f'{arm}@final')

    print('-' * 100)
    print('VERDICT: режим «контекст читается» = val < bias-only и dctx > 0, причём '
          'на уровне биграммного ориентира.')


if __name__ == '__main__':
    main()
