"""scripts/probe_bias_decomp_multi.py — BIAS-DECOMP/Δctx на РАЗНЫХ текстах:
отвечает, «скачут данные» из-за модели или из-за выборки одного текста.

Для каждого чекпоинта × текста: CE(full), CE(bias-only), dctx, доля позиций
argmax(full)==argmax(bias), категории full-argmax (word/punct), энтропия.

Run: python scripts/probe_bias_decomp_multi.py --ckpts "a.pt" "b.pt" \
        --war wb/token_stream_WAR_eos.bin
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402
from scripts.generate import load_russian_tokenizer  # noqa: E402

PUNCT = re.compile(r'^[\s\W_]+$')
WORD = re.compile(r'\w', re.UNICODE)


def _cat(tid, tok):
    t = tok.decode([int(tid)]).strip()
    return 'punct' if PUNCT.match(t) else ('word' if WORD.search(t) else 'other')


def run_text(model, tok, window, step, device, n_windows=2):
    L = int(model.cfg.seq_len)
    ids = window[:n_windows * L]
    tb = model.lm_head.token_bias.data
    tb_argmax = int(tb.argmax())
    lz_b = float(torch.logsumexp(tb.float(), dim=-1))
    tot_ce = tot_b = tot_n = 0.0
    match = npos = 0
    cats = {'word': 0, 'punct': 0, 'other': 0}
    H_sum = 0.0
    model.eval()
    with torch.no_grad():
        model.reset_streams()
        model.reset_cache()
        state = gs = None
        for off in range(0, len(ids) - L, L):
            x = torch.tensor(ids[off:off + L], dtype=torch.long, device=device).view(1, -1)
            y = torch.tensor(ids[off + 1:off + L + 1], dtype=torch.long,
                             device=device).view(1, -1)
            h = model.embed_tokens(x)
            out, state, gs, _ = model(h, state, global_state=gs,
                                      adaptive=False, step=step, tokens=x)
            logits = model.lm_head(out)
            lp = logits.reshape(-1, logits.shape[-1])
            tgt = y.reshape(-1)
            ce = float(-lp[torch.arange(tgt.numel()), tgt].mean())
            bce = float((lz_b - tb[tgt]).mean())
            tot_ce += ce * tgt.numel()
            tot_b += bce * tgt.numel()
            tot_n += tgt.numel()
            for pos in range(0, L, 8):
                npos += 1
                if int(lp[pos].argmax()) == tb_argmax:
                    match += 1
                cats[_cat(int(lp[pos].argmax()), tok)] += 1
                p = torch.softmax(lp[pos].double(), -1)
                H_sum += float(-(p * torch.log2(p.clamp_min(1e-12))).sum())
    return dict(ce=tot_ce / max(tot_n, 1), bce=tot_b / max(tot_n, 1),
                dctx=(tot_b - tot_ce) / max(tot_n, 1), match=match, n=npos,
                cats=cats, H=H_sum / max(npos, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--war', default='wb/token_stream_WAR_eos.bin')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    tok = load_russian_tokenizer()
    texts = {}
    texts['analyzer'] = tok.encode('Москва — столица России, и в ней живут миллионы людей. ' * 60).ids
    texts['war_holdout'] = np.fromfile(args.war, dtype=np.uint16).astype(np.int64)[
        :2000].tolist()
    _w = np.fromfile(args.war, dtype=np.uint16).astype(np.int64)
    texts['war_eval_region'] = _w[len(_w) // 4:len(_w) // 4 + 2000].tolist()
    texts['prompt'] = tok.encode('На рассвете началось наступление, и тогда').ids * 20

    for path in args.ckpts:
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        cfg = ck.get('cfg')
        model = EVAStack(cfg)
        miss, unexp = model.load_state_dict(ck['model'], strict=False)
        model.to(args.device)
        step = int(ck.get('step', 0) or 0)
        print(f'\n=== {os.path.basename(path)} step={step} '
              f'best_val={float(ck.get("best_val_loss", -1)):.4f} '
              f'(missing={len(miss)}, unexpected={len(unexp)}) ===')
        for name, ids in texts.items():
            r = run_text(model, tok, ids, step, args.device)
            print(f'  {name:16s} CE={r["ce"]:.4f} bias-only={r["bce"]:.4f} '
                  f'dctx={r["dctx"]:+.4f} | argmax==bias {r["match"]}/{r["n"]} '
                  f'({100.0 * r["match"] / max(r["n"], 1):.0f}%) | full: '
                  f'w={r["cats"]["word"]} p={r["cats"]["punct"]} o={r["cats"]["other"]} '
                  f'| H={r["H"]:.2f} bit', flush=True)


if __name__ == '__main__':
    main()
