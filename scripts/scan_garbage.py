# -*- coding: utf-8 -*-
"""scan_garbage.py — поисковик необучаемых регионов в token_stream_*.bin.

Мотивация (живой инцидент 2026-09, M14): окно с мусором (битые байты/случайные
id) даёт кодовой голове CE ≈ K·ln2 (≈22.2 для 32 бит) — «нулевую информацию»,
а при длинных мусорных прогонах модель уезжает в уверенную анти-предикцию
(CE 34→67) — градиент там чистый шум. Tренер теперь ветит такие батчи, но
корпус стоит почистить: этот файл находит КОРДИНАТЫ.

Эвристика на окно 256 токенов:
  * доля уникальных токенов: естественный BPE-текст ≈ 0.25-0.6; шум ≈ 1.0
    (главный признак — эмпирическая энтропия 256-выборки ограничена сверху
    log2(256)=8 бит, поэтому на неё смотрят только как на второй сигнал);
  * энтропия распределения id внутри окна (бит/токен): текст ≈ 4.5-7.5.
Флаг = uniq_ratio > --uniq-thr (0.85) ИЛИ (ent > --ent-thr (7.6) И uniq > 0.7). Выход: список регионов (файл, offset_start,
offset_end, средние метрики) — их можно вырезать из .bin или пометить.

Использование (Colab, на тех же файлах что и обучение):
    python scripts/scan_garbage.py /content/drive/MyDrive/eva_clm/data
    python scripts/scan_garbage.py data/ --glob 'token_stream_*.bin' --thr 10.5
"""
import sys, os, glob, argparse
import numpy as np


def scan_file(path, window=256, uniq_thr=0.85, ent_thr=7.6, stride=1):
    a = np.memmap(path, dtype=np.uint16, mode='r')
    n = (len(a) // window) * window
    if n == 0:
        return [], 0
    ent_sum = uniq_sum = 0.0
    nwin = 0
    regions = []          # (start, end, mean_ent, mean_uniq)
    cur = None
    for off in range(0, n, window * stride):
        w = np.asarray(a[off:off + window], dtype=np.int64)
        if w.size < window:
            break
        vals, cnts = np.unique(w, return_counts=True)
        p = cnts / cnts.sum()
        ent = float(-(p * np.log2(p)).sum())
        uniq = float(cnts.size) / window
        ent_sum += ent; uniq_sum += uniq; nwin += 1
        bad = (uniq > uniq_thr) or (ent > ent_thr and uniq > 0.70)
        if bad:
            if cur is None:
                cur = [off, off + window, [ent], [uniq]]
            else:
                cur[1] = off + window
                cur[2].append(ent); cur[3].append(uniq)
        else:
            if cur is not None:
                regions.append((cur[0], cur[1], float(np.mean(cur[2])),
                                float(np.mean(cur[3])), len(cur[2])))
                cur = None
    if cur is not None:
        regions.append((cur[0], cur[1], float(np.mean(cur[2])),
                        float(np.mean(cur[3])), len(cur[2])))
    stats = (nwin, ent_sum / max(nwin, 1), uniq_sum / max(nwin, 1))
    return regions, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('root', help='папка с token_stream_*.bin')
    ap.add_argument('--glob', default='token_stream_*.bin')
    ap.add_argument('--window', type=int, default=256)
    ap.add_argument('--uniq-thr', type=float, default=0.85, help='доля уникальных id в окне, выше — шум')
    ap.add_argument('--ent-thr', type=float, default=7.6, help='бит/токен (потолок log2(window)), второй сигнал')
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.root, args.glob)))
    if not files:
        print('нет файлов'); sys.exit(1)
    total_bad = 0
    for f in files:
        regions, (nwin, ment, munq) = scan_file(f, args.window, args.uniq_thr, args.ent_thr)
        n = os.path.getsize(f) // 2
        tag = '  <== МУСОР' if regions else ''
        print(f'{os.path.basename(f):44s} {n:>12,} tok | энтропия mean={ment:5.2f} '
              f'uniq={munq:4.2f} | мусорных окон: {sum(r[4] for r in regions)}{tag}')
        for (s0, s1, e, u, k) in regions:
            print(f'    [{s0:>12,} .. {s1:>12,})  {s1 - s0:>8,} токенов  ent={e:5.2f} uniq={u:4.2f}')
            total_bad += s1 - s0
    print(f'\nИтого помечено: {total_bad:,} токенов '
          f'({100.0 * total_bad / max(sum(os.path.getsize(p) for p in files) // 2, 1):.4f}%)')
    print('Вырезка региона [a,b) из файла: '
          "t=np.memmap(p,np.uint16,'r'); np.array(np.concatenate([t[:a],t[b:]])).tofile(p+'.clean')")


if __name__ == '__main__':
    main()
