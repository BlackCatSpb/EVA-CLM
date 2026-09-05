"""Пересборка жанровых бинарников с EOS-границами предложений.

Восстановленный pipeline (из FCP/EVA-Ai):
  main-russian/{GENRE}/*.txt  →  token_stream_{GENRE}_eos.bin (uint16)

Отличия от prepare_corpus.py:
  - разбивка текста на предложения, <|eos|> (id=2) вставляется после каждого;
  - чистка OCR-шапок («пер.», «OCR», «копирайт») — короткие строки в начале;
  - без чередующихся PAD (uint16, чисто), без SEQ_LEN-чанков;
  - предложения короче 3 слов не сохраняются (мусор).

Порядок: id = [eos] + предложение + [eos] ...
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

EOS = 2
MIN_SENT_TOKENS = 3
_HEADER_LINES = 8

_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')
_DIRTY = re.compile(
    r'^\s*(перев|пер\.|пер |ocr|скан|копирайт|©|подготовк|оформлен|'
    r'в книге|издание|серия|сайт|lib\.ru|либрусек|флибуста|spellcheck|'
    r'translated|translation|digitized|scanned|proofread|HarryFan|'
    r'ISBN|переводчик|вычитка)', re.IGNORECASE)
_TITLE_FRAME = re.compile(r'^[-=—_*~+]{4,}$')
_DASH_PAIR = re.compile(r'^[A-ZА-ЯЁ].*\.\s*[-–—]\s*[A-ZА-ЯЁ]')


def clean_header(text: str) -> str:
    """Удаляет шапки (OCR/пер./титульные строки) в начале файла."""
    lines = text.split('\n')
    cut = 0
    for i in range(min(_HEADER_LINES, len(lines))):
        s = lines[i].strip()
        if not s:
            continue
        if _TITLE_FRAME.match(s) or _DIRTY.search(s):
            cut = i + 1
            continue
        if len(s) >= 60:
            break
        if len(s) < 40 and not re.search(r'[.!?…]$', s):
            cut = i + 1
            continue
        break
    return '\n'.join(lines[cut:])


_ABBR_RE = re.compile(r'(?<!\w)[а-яё]{1,3}\.$')


def _is_abbrev(text: str, m) -> bool:
    """Точка не является границей предложения (аббревиатура/домен/число)."""
    if m.group() != '.' or m.start() == 0:
        return m.start() == 0
    start = m.start()
    prev = text[start - 1]
    if prev.isdigit() or (prev.isascii() and prev.isalpha()):
        return True
    return bool(_ABBR_RE.search(text[max(0, start - 8):start + 1]))


def split_sentences(text: str):
    """Границы предложений (start, end) по .!?… с фильтром аббревиатур.

    Не режет: точки после латиницы/цифр (mail.ru, 1966.), короткие
    кириллические аббревиатуры (г., ул., т.е., т.п., и т.д.).
    """
    out = []
    for m in re.finditer(r'[.!?…]', text):
        if _is_abbrev(text, m):
            continue
        if out and m.start() - out[-1][1] <= 1:
            out[-1] = (out[-1][0], m.end())
        else:
            out.append((m.start(), m.end()))
    return out


def process_genre(genre, tokenizer, src, out_dir, encoding='cp1251'):
    out = os.path.join(out_dir, f'token_stream_{genre}_eos.bin')
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f'  {genre}: пропуск (уже собран: {out})')
        return 0, 0
    files = sorted(glob.glob(os.path.join(src, genre, '*.txt')))
    if not files:
        print(f'  [WARN] {genre}: no files')
        return 0, 0
    CHUNK = 100_000_000
    buf = np.empty(CHUNK, dtype=np.uint16)
    pos = 0
    n_total = 0
    n_sents = 0
    t0 = time.time()

    def flush():
        nonlocal pos, n_total
        if pos:
            n_total += pos
            buf[:pos].tofile(fh)
            pos = 0

    fh = open(out, 'wb')
    buf[pos] = EOS
    pos += 1
    pbar = tqdm(files, desc=genre, unit='файл', ncols=110,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                           '{rate_fmt} {postfix}')
    for idx, path in enumerate(pbar):
        try:
            with open(path, encoding=encoding, errors='replace') as f:
                raw = f.read()
        except Exception as e:
            print(f'  [WARN] {os.path.basename(path)}: {e}')
            continue
        text = clean_header(raw)
        text = re.sub(r'\n+', ' ', text.replace('\r', ''))
        text = re.sub(r'[ \t]{2,}', ' ', text)
        bounds = split_sentences(text)
        if not bounds:
            continue
        enc = tokenizer.encode(text)
        ids = enc.ids
        offs = enc.offsets
        bi = 0
        for tok_id, (s, e) in zip(ids, offs):
            while bi < len(bounds) and s >= bounds[bi][1]:
                if pos + 1 > CHUNK:
                    flush()
                buf[pos] = EOS
                pos += 1
                n_sents += 1
                bi += 1
            if pos + 1 > CHUNK:
                flush()
            buf[pos] = tok_id
            pos += 1
        while bi < len(bounds):
            if pos + 1 > CHUNK:
                flush()
            buf[pos] = EOS
            pos += 1
            n_sents += 1
            bi += 1
        pbar.set_postfix(tok=f'{n_sents * 14 // 1e6:.0f}M', sents=f'{n_sents // 1000}K')
    flush()
    fh.close()
    pbar.close()
    print(f'  {genre}: {len(files)} файлов, {n_total // 1e6:.0f}M токенов, '
          f'{n_sents // 1000}K предложений, {time.time() - t0:.0f}s -> {out}', flush=True)
    return len(files), n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=r'C:\Users\black\OneDrive\Desktop\EVA-Ai\libinpoc_txt\main-russian')
    ap.add_argument('--out-dir', default='wb')
    ap.add_argument('--tokenizer', default='wb/russian_tokenizer/tokenizer.json')
    ap.add_argument('--encoding', default='cp1251')
    ap.add_argument('--genre', default='all')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f'[ERROR] src not found: {args.src}')
        sys.exit(1)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f'Tokenizer: vocab={tokenizer.get_vocab_size()}')

    if args.genre == 'all':
        genres = sorted(d for d in os.listdir(args.src)
                        if os.path.isdir(os.path.join(args.src, d)))
    else:
        genres = [args.genre]

    if args.dry_run:
        for g in genres:
            n = len(glob.glob(os.path.join(args.src, g, '*.txt')))
            print(f'  {g}: {n} файлов')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    total_f, total_t = 0, 0
    for g in genres:
        nf, nt = process_genre(g, tokenizer, args.src, args.out_dir, args.encoding)
        total_f += nf
        total_t += nt
    print(f'Итого: {total_f} файлов, {total_t // 1e6:.0f}M токенов')


if __name__ == '__main__':
    main()