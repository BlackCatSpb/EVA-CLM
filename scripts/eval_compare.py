"""
eval_compare.py — количественное сравнение режимов генерации.
Сравнивает baseline (фикс. temp, без truncation, без reasoning) против
smart (адаптивный temp + reasoning, без truncation) на одинаковых RNG-сидax.
Метрики (по словам/субтокенам от пробела):
  distinct-1 / distinct-2 — разнообразие (выше = лучше)
  repetition% — доля токенов, повторяющих недавний (окно 8) (ниже = лучше)
"""
import os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate import (load_inference_checkpoint, load_russian_tokenizer, generate)
from scripts.smart_controller import SmartController, smart_generate

PROMPTS = ['Привет, как дела?', 'Москва — столица', 'В начале было Слово']
TOKENS = 24
K = 3


def words(text):
    return text.split()


def distinct(ws):
    if len(ws) < 2:
        return 0.0, 0.0
    d1 = len(set(ws)) / len(ws)
    bg = [(ws[i], ws[i + 1]) for i in range(len(ws) - 1)]
    d2 = len(set(bg)) / len(bg)
    return d1, d2


def repetition(ws, window=8):
    if len(ws) <= window:
        return 0.0
    rep = sum(1 for i in range(window, len(ws)) if ws[i] in ws[i - window:i])
    return rep / (len(ws) - window)


def evaluate():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/step_11844_fcf.pt')
    ap.add_argument('--tokens', type=int, default=TOKENS)
    ap.add_argument('--k', type=int, default=K)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = load_inference_checkpoint(a.checkpoint, skip_compression=False, device='cpu')
    cfg = state['cfg']
    model = __import__('core').WideBindStack(cfg).to(device)
    model.load_state_dict(state['model'], strict=False)
    vocab = cfg.vocab

    def run_baseline():
        model.reasoning_scale_override = 0.0
        out = []
        for pi, p in enumerate(PROMPTS):
            for k in range(a.k):
                torch.manual_seed(1000 + pi * 10 + k)
                t = generate(model, p, a.tokens, temperature=0.9, top_k=0,
                             rep_penalty=2.0, rep_window=5)
                out.append(t)
        return out

    def run_smart():
        out = []
        ctrl = SmartController(model, vocab, reasoning_on=True, no_trunc=True)
        for pi, p in enumerate(PROMPTS):
            for k in range(a.k):
                ctrl.recent = []
                ctrl.recov = 0
                torch.manual_seed(1000 + pi * 10 + k)
                t, _ = smart_generate(model, p, ctrl, a.tokens, no_trunc=True)
                out.append(t)
        return out

    def agg(texts):
        d1 = d2 = rep = 0.0
        for t in texts:
            ws = words(t)
            x1, x2 = distinct(ws)
            d1 += x1; d2 += x2; rep += repetition(ws)
        n = len(texts)
        return d1 / n, d2 / n, rep / n, n

    base = run_baseline()
    smart = run_smart()
    bd1, bd2, brep, bn = agg(base)
    sd1, sd2, srep, sn = agg(smart)
    print(f'{"config":10} {"distinct1":>10} {"distinct2":>10} {"repetition%":>12}  (n={bn})')
    print(f'{"baseline":10} {bd1:10.4f} {bd2:10.4f} {brep*100:11.2f}%')
    print(f'{"smart":10} {sd1:10.4f} {sd2:10.4f} {srep*100:11.2f}%')
    print()
    print('Вывод: выше distinct и ниже repetition% = разнообразнее и меньше зацикливания.')


if __name__ == '__main__':
    evaluate()
