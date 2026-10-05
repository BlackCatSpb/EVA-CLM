"""T9.10: Drive-robust checkpoint saves.

The Colab Drive FUSE can silently swallow large writes (best.pt is 2.2 GB):
torch.save returns, the file on the Drive stays stale (observed live: the
4840-step best downloaded as the 3960-step content, byte-identical to the
previous file). This module fixes the save path:

  1. write to the LOCAL runtime disk (fast, reliable) + verify;
  2. copy to the Drive via a temp file + atomic rename;
  3. verify on the Drive: size + step (mmap header) + header/tail chunk hash
     against the local source;
  4. retry with backoff; the local copy is kept as the session fallback and
     `heal_drive_from_local` re-copies it if a previous save did not land.

`save_best_robust` is the only entry point the training loop needs; the
signature keeps the notebook's `_atomic_save(env, name)` contract.
"""
import os
import time
import shutil
import hashlib

_copyfile = shutil.copyfile  # indirection: tests patch core.ckpt_io._copyfile


def _log_default(msg: str) -> None:
    pass


def read_step(path: str, log=None):
    """Cheap `step` read. Prefers mmap (header only, ~ms even for 2.2 GB);
    falls back to a full load; returns None when the file is unreadable.

    M65-opt: the mmap→full fallback and the final failure are surfaced via
    `log` when given — a silent fallback here hides Drive/corruption issues
    in exactly the module whose job is to make them visible."""
    import torch
    try:
        ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        step = int(ck.get('step', -1)) if isinstance(ck, dict) else -1
        del ck
        return step
    except Exception as e:
        if log is not None:
            log(f'[ckpt] mmap read failed on {os.path.basename(path)} ({e}); '
                f'full-load fallback')
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        step = int(ck.get('step', -1)) if isinstance(ck, dict) else -1
        del ck
        return step
    except Exception as e:
        if log is not None:
            log(f'[ckpt] read_step failed on {os.path.basename(path)}: {e}')
        return None


def _chunk_hash(path: str, nbytes: int = 1 << 20) -> str:
    """SHA256 of the first and last `nbytes` (the zip header + central
    directory) — a strong partial-content check that costs ~2 MB of reads."""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        h.update(f.read(nbytes))
        if size > 2 * nbytes:
            f.seek(-nbytes, os.SEEK_END)
            h.update(f.read(nbytes))
    return h.hexdigest()


def verify_ckpt(path: str, expected_step=None, expected_bytes=None,
                ref_path: str = None, log=None):
    """Verify a checkpoint file: exists, non-empty, size, step, and (when
    `ref_path` is given) the header/tail content hash. Returns (ok, reason)."""
    if not path or not os.path.exists(path):
        return False, 'missing'
    size = os.path.getsize(path)
    if size <= 0:
        return False, 'empty'
    if expected_bytes is not None and size != int(expected_bytes):
        return False, f'size {size} != {expected_bytes}'
    if expected_step is not None:
        step = read_step(path, log=log)
        if step != int(expected_step):
            return False, f'step {step} != {expected_step}'
    if ref_path is not None and os.path.exists(ref_path):
        try:
            if _chunk_hash(path) != _chunk_hash(ref_path):
                return False, 'content hash mismatch'
        except Exception as e:
            return False, f'hash read failed: {e}'
    return True, 'ok'


def atomic_save(obj, path: str, log=_log_default) -> str:
    """torch.save to `path.tmp`, fsync, atomic replace. Safe on the local
    runtime disk and (via rename) on the Drive mount."""
    import torch
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    try:
        with open(tmp, 'rb+') as f:
            os.fsync(f.fileno())
    except Exception as e:
        # Durability is this module's whole point: a swallowed fsync failure
        # is the exact class of silent Drive loss it exists to prevent.
        log(f'[ckpt] fsync failed on {os.path.basename(tmp)}: {e}')
    os.replace(tmp, path)
    return path


def _copy_verified(local_path: str, drive_path: str, step, log=_log_default) -> bool:
    """Copy local -> drive via tmp + rename, then verify on the drive."""
    d = os.path.dirname(drive_path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = drive_path + '.tmp'
    _copyfile(local_path, tmp)
    try:
        os.sync()  # force the FUSE flush before the rename (Linux/Colab)
    except AttributeError:
        pass       # os.sync is Linux-only (Windows dev boxes) — expected
    except OSError as e:
        log(f'[ckpt] os.sync before rename failed: {e}')
    os.replace(tmp, drive_path)
    try:
        os.sync()
    except AttributeError:
        pass
    except OSError as e:
        log(f'[ckpt] os.sync after rename failed: {e}')
    ok, why = verify_ckpt(drive_path, expected_step=step,
                          expected_bytes=os.path.getsize(local_path),
                          ref_path=local_path, log=log)
    if not ok:
        log(f'[ckpt] drive verify failed: {why}')
    return ok


def save_best_robust(env, drive_path: str, local_path: str, step=None,
                     retries: int = 3, delay: float = 5.0, log=_log_default) -> bool:
    """Save `env` to the local disk, then sync it to the Drive and verify.

    Returns True when the Drive copy is verified. On failure the local copy
    is kept (and `heal_drive_from_local` can re-copy it later)."""
    if step is None:
        step = env.get('step', -1) if isinstance(env, dict) else -1
    step = int(step)

    atomic_save(env, local_path, log=log)
    ok, why = verify_ckpt(local_path, expected_step=step, log=log)
    if not ok:
        log(f'[ckpt] LOCAL save failed verification ({why}); drive not touched')
        return False
    lb = os.path.getsize(local_path)
    log(f'[ckpt] local saved (step={step}, {lb / 2**30:.2f} GiB) -> syncing to drive')

    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            if _copy_verified(local_path, drive_path, step, log=log):
                log(f'[ckpt] drive best.pt VERIFIED (step={step}, attempt {attempt})')
                return True
        except Exception as e:
            log(f'[ckpt] drive copy error: {e} (attempt {attempt}/{retries})')
        if attempt < retries:
            time.sleep(max(0.0, float(delay)) * attempt)

    log(f'[ckpt] WARNING: drive copy NOT verified after {retries} attempts; '
        f'local copy kept: {local_path}')
    return False


def heal_drive_from_local(drive_path: str, local_path: str,
                          log=_log_default) -> bool:
    """If the local copy is NEWER than the Drive's best.pt (a save whose
    drive copy failed earlier in this session), re-copy it. No-op when the
    local is absent or the drive is already current."""
    if not os.path.exists(local_path):
        return False
    ls = read_step(local_path)
    if ls is None:
        return False
    ds = read_step(drive_path) if os.path.exists(drive_path) else None
    if ds is not None and ds >= ls:
        return False
    log(f'[ckpt] heal: drive step={ds} < local step={ls} -> re-copying')
    try:
        return _copy_verified(local_path, drive_path, ls, log=log)
    except Exception as e:
        log(f'[ckpt] heal failed: {e}')
        return False


def warn_bounded_gate_mismatch(missing, unexpected, log=None) -> list:
    """fixrev-6 (аудит): резюм ON/OFF-чекпоинта молча рандомит bounded-гейт.

    `load_state_dict(strict=False)` не считает ошибкой отсутствие
    `memory_bank.gate_W/.gate_b` и `bounded_post_norm_w.*`: OFF-чекпоинт в
    ON-модели оставляет гейт на init-значениях, ON-чекпоинт в OFF-модели —
    отбрасывает обученные. Возвращает список несовпавших gate-ключей и (при
    `log`) печатает предупреждение — вызывать сразу после load_state_dict.
    """
    keys = [k for k in list(missing or []) + list(unexpected or [])
            if k.endswith(('.gate_W', '.gate_b'))
            or 'bounded_post_norm_w' in k]
    if keys and log is not None:
        log(f'  [warn] bounded_residual: {len(keys)} gate-параметров не '
            f'совпали при загрузке (резюм ON<->OFF чекпойнта) — они '
            f'инициализированы/потеряны: {keys[:4]}')
    return keys
