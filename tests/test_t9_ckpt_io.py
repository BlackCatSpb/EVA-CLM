"""T9.10: tests for the Drive-robust checkpoint I/O."""
import os
import torch

from core.ckpt_io import (atomic_save, read_step, verify_ckpt,
                          save_best_robust, heal_drive_from_local)
import core.ckpt_io as ckpt_io


def _env(step):
    return {'step': step, 'model': {'w': torch.randn(8, 8)},
            'best_val_loss': 1.0}


def test_atomic_save_roundtrip(tmp_path):
    p = str(tmp_path / 'best.pt')
    atomic_save(_env(5), p)
    assert os.path.exists(p)
    assert not os.path.exists(p + '.tmp')
    assert read_step(p) == 5
    ok, why = verify_ckpt(p, expected_step=5)
    assert ok, why


def test_verify_rejects_wrong_step_and_garbage(tmp_path):
    p = str(tmp_path / 'best.pt')
    atomic_save(_env(5), p)
    ok, why = verify_ckpt(p, expected_step=6)
    assert not ok and 'step' in why
    ok, why = verify_ckpt(p, expected_bytes=1)
    assert not ok and 'size' in why
    bad = str(tmp_path / 'bad.pt')
    with open(bad, 'wb') as f:
        f.write(b'not a checkpoint')
    ok, why = verify_ckpt(bad, expected_step=5)
    assert not ok
    ok, why = verify_ckpt(str(tmp_path / 'nope.pt'))
    assert not ok and why == 'missing'


def test_save_best_robust_ok(tmp_path):
    drive = str(tmp_path / 'drive' / 'best.pt')
    local = str(tmp_path / 'local' / 'best.pt')
    assert save_best_robust(_env(7), drive, local, step=7, delay=0.0)
    assert read_step(drive) == 7
    assert read_step(local) == 7
    assert not os.path.exists(drive + '.tmp')


def test_save_best_robust_retries(tmp_path, monkeypatch):
    drive = str(tmp_path / 'drive' / 'best.pt')
    local = str(tmp_path / 'local' / 'best.pt')
    real = ckpt_io._copyfile
    calls = {'n': 0}

    def flaky(src, dst):
        calls['n'] += 1
        if calls['n'] < 3:
            raise OSError('drive busy')
        return real(src, dst)

    monkeypatch.setattr(ckpt_io, '_copyfile', flaky)
    assert save_best_robust(_env(8), drive, local, step=8, retries=3, delay=0.0)
    assert calls['n'] == 3
    assert read_step(drive) == 8


def test_save_best_robust_failure_keeps_local(tmp_path, monkeypatch):
    drive = str(tmp_path / 'drive' / 'best.pt')
    local = str(tmp_path / 'local' / 'best.pt')

    def broken(src, dst):
        raise OSError('drive gone')

    monkeypatch.setattr(ckpt_io, '_copyfile', broken)
    assert not save_best_robust(_env(9), drive, local, step=9,
                                retries=2, delay=0.0)
    assert read_step(local) == 9
    assert not os.path.exists(drive)


def test_heal_drive_from_local(tmp_path):
    drive = str(tmp_path / 'drive' / 'best.pt')
    local = str(tmp_path / 'local' / 'best.pt')
    atomic_save(_env(9), drive)
    atomic_save(_env(10), local)
    assert heal_drive_from_local(drive, local)
    assert read_step(drive) == 10
    assert not heal_drive_from_local(drive, local)  # already current


def test_heal_noop_without_local(tmp_path):
    drive = str(tmp_path / 'drive' / 'best.pt')
    local = str(tmp_path / 'local' / 'best.pt')
    atomic_save(_env(3), drive)
    assert not heal_drive_from_local(drive, local)
    assert read_step(drive) == 3
