"""Миграция старых чекпоинтов под когерентность спиралей (bind W_out +K,
bind_coh_gate, freq_scale). Используется train.py (resume) и generate_onnx.py.

Численно эквивалентна прежнему поведению:
- W_out (n_dims*2K, D) -> (n_dims*2K+K, D): первые строки копируются, хвост нули;
- bind_coh_gate = 0.0 — когерентность выключена;
- freq_scale = 1.0 — прежний масштаб частот.
"""
import torch


def migrate_state_dict(sd, model):
    """Возвращает (мигрированный sd, число изменённых ключей)."""
    changed = 0
    for name, p in model.named_parameters():
        if name.endswith('bind.W_out'):
            old = sd.get(name)
            if old is None:
                continue
            if tuple(old.shape) == tuple(p.shape):
                continue
            new = torch.zeros_like(p)
            n = min(old.shape[0], p.shape[0])
            new[:n] = old[:n]
            sd[name] = new
            changed += 1
    for name, p in model.named_parameters():
        if name.endswith('bind_coh_gate'):
            if name not in sd:
                sd[name] = torch.zeros_like(p)
                changed += 1
        elif name.endswith('freq_scale'):
            if name not in sd:
                sd[name] = torch.tensor(1.0)
                changed += 1
    return sd, changed