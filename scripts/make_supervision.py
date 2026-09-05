"""Сборка учительских пар для обучаемого прожектора.

Источник: FCF/real_data/corpus_1m.txt (исходный текст с порядком слов).
Выход: data/supervision.jsonl — предложения 7–12 слов (только кириллица):
  {"words": [...], "poss": [...], "roles": [...]}
Порядок слов в массиве = правильный порядок корпуса (учитель арбитра).
Роли — простая эвристика (ядро/актант/модификатор/служебное) как учитель гейта.
"""

import argparse
import json
import random
import re
import sys
import time

sys.path.insert(0, '.')

import pymorphy3

from core.projector_net import ROLE_INDEX

_WORD = re.compile(r'^[а-яё]+$')
_END = re.compile(r'[.!?]+')

ROLE_BY_POS = {
    'VERB': 'predicate', 'INFN': 'predicate',
    'NOUN': 'actant', 'NPRO': 'actant',
    'ADJF': 'modifier', 'ADVB': 'modifier', 'NUMR': 'modifier', 'GRND': 'modifier',
}


def split_sentences(text: str):
    buf = []
    for chunk in _END.split(text):
        for raw in chunk.split():
            w = raw.strip('—–«»",;:()').lower()
            if _WORD.match(w):
                buf.append(w)
        if len(buf) >= 7:
            yield buf
            buf = []
        elif len(buf) > 20:
            buf = buf[-6:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=r'C:\Users\black\OneDrive\Desktop\FCF\real_data\corpus_1m.txt')
    ap.add_argument('--out', default='data/supervision.jsonl')
    ap.add_argument('--max-sents', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    morph = pymorphy3.MorphAnalyzer()
    rng = random.Random(args.seed)
    n = 0
    t0 = time.time()
    with open(args.out, 'w', encoding='utf-8') as f:
        with open(args.corpus, encoding='utf-8', errors='ignore') as src:
            for line in src:
                for sent in split_sentences(line):
                    if len(sent) > 12:
                        sent = sent[:12]
                    if len(sent) < 7:
                        continue
                    poss, roles = [], []
                    for w in sent:
                        tag = morph.parse(w)[0]
                        pos = str(tag.tag.POS or 'OTHER')
                        poss.append(pos)
                        roles.append(ROLE_BY_POS.get(pos, 'service'))
                    f.write(json.dumps({'words': sent, 'poss': poss, 'roles': roles},
                                       ensure_ascii=False) + '\n')
                n += 1
                if n % 5000 == 0:
                    print(f'{n} предложений, {time.time() - t0:.0f}s', flush=True)
                if n >= args.max_sents:
                    break
    print(f'Готово: {n} предложений -> {args.out}')


if __name__ == '__main__':
    main()