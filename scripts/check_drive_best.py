"""T9.10 watchdog: does the Drive best.pt match the last improvement in val_history?

The Colab Drive FUSE can silently swallow a 2.6 GB best.pt write while the
in-session verify (reading through the same FUSE cache) reports VERIFIED.
This script checks the truth from OUTSIDE: it reads val_history.jsonl and the
best.pt pickle header via rclone and compares the steps.

Usage:
    python scripts/check_drive_best.py
    python scripts/check_drive_best.py --remote gdrive:eva_clm --rclone C:\\path\\rclone.exe
Exit code 0 = consistent, 1 = mismatch (a save did not land), 2 = cannot tell.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

DEFAULT_REMOTE = 'gdrive:eva_clm'
DEFAULT_RCLONE = os.path.join(os.environ.get('LOCALAPPDATA', ''),
                              'Programs', 'rclone', 'rclone.exe')


def _rclone(rclone, *args):
    return subprocess.run([rclone, *args], capture_output=True, check=True).stdout


def last_improvement(history_text):
    best = float('inf')
    best_step, best_val = None, None
    for line in history_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        bv = row.get('best_val_loss')
        if bv is not None and float(bv) < best:
            best = float(bv)
            best_step, best_val = int(row.get('step', -1)), float(bv)
    return best_step, best_val


def read_step_bytes(d, i=None):
    if i is None:
        i = d.find(b'X\x04\x00\x00\x00step')
    if i < 0:
        return None
    j = i + 9
    op = d[j + 2]
    if op == 0x4d:      # BININT2
        return int.from_bytes(d[j + 3:j + 5], 'little')
    if op == 0x4b:      # BININT1
        return d[j + 3]
    if op == 0x4a:      # BININT
        return int.from_bytes(d[j + 3:j + 7], 'little')
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--remote', default=DEFAULT_REMOTE)
    ap.add_argument('--rclone', default=DEFAULT_RCLONE)
    ap.add_argument('--count', type=int, default=4096)
    args = ap.parse_args()

    if not os.path.exists(args.rclone):
        print(f'rclone not found: {args.rclone}', file=sys.stderr)
        return 2

    hist = _rclone(args.rclone, 'cat', f'{args.remote}/checkpoints/val_history.jsonl'
                   ).decode('utf-8', 'replace')
    best_step, best_val = last_improvement(hist)

    head = _rclone(args.rclone, 'cat', f'{args.remote}/checkpoints/best.pt',
                   '--offset', '0', '--count', str(args.count))
    drive_step = read_step_bytes(head)

    numbered = []
    listing = _rclone(args.rclone, 'ls', f'{args.remote}/checkpoints/'
                     ).decode('utf-8', 'replace')
    for line in listing.splitlines():
        m = re.search(r'\s(best \d+\.pt)$', line.strip())
        if m:
            numbered.append(m.group(1))
    numbered.sort(key=lambda n: int(re.search(r'\d+', n).group()))

    print(f'val_history : last improvement step={best_step} val={best_val}')
    print(f'drive best.pt: header step={drive_step}')
    print(f'numbered    : {", ".join(numbered) if numbered else "(none)"}')

    if best_step is None or drive_step is None:
        print('verdict: cannot tell (missing data)')
        return 2
    if drive_step >= best_step:
        print('verdict: OK — best.pt is at least as new as the last improvement')
        return 0
    print(f'verdict: MISMATCH — best.pt (step {drive_step}) is behind the '
          f'last improvement (step {best_step}, val {best_val}). '
          f'The {best_step} save did not land; re-copy the local fallback.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
