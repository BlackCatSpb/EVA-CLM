"""Прогон Прожектора на чекпоинте: считываем слова из коллективной памяти.

Загружает best.pt, прогоняет русский текст через модель, собирает события
записи концептов (_write_event) со ВСЕХ коллективных слоёв (логическое OR —
граница слова считается, если её зафиксировал хотя бы один слой), и
превращает скрытое состояние в слова через Projector.read_words.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

from core import WideBindConfig, WideBindStack
from core.projector import Projector
from generate import load_russian_tokenizer
from torch.serialization import add_safe_globals

add_safe_globals([WideBindConfig])

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
TEXT = ("Москва — столица России. Зима в этом году выдалась холодной и снежной. "
        "Кот по имени Василий спал на тёплом ковре у окна. Дети катались с горки "
        "во дворе, а мама пекла пироги. Утром солнце осветило белые крыши домов. "
        "Хорошая собака бежала по полю и лаяла на птиц. Высокое дерево росло у реки, "
        "и ветер качал его ветви. Старый дедушка читал книгу при свече. Маленькая девочка "
        "рисовала красками синий океан. Быстрая лошадь мчалась по дороге. Сосед принёс "
        "корзину свежих яблок и груш. Ночное небо было усыпано яркими звёздами. "
        "Река текла тихо и спокойно мимо зелёного луга. Учитель объяснял задачу у доски, "
        "а ученики внимательно слушали. Кошка играла с клубком красной шерсти. "
        "Поезд прибыл на вокзал точно по расписанию. Садовник поливал цветы ранним утром.")

tok = load_russian_tokenizer()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
cfg = ckpt['cfg']
model = WideBindStack(cfg)
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
model.eval()
proj = Projector(tok)

ids = tok.encode(TEXT).ids
if len(ids) > cfg.seq_len:
    ids = ids[:cfg.seq_len]
print(f"Text tokens: {len(ids)} (seq_len={cfg.seq_len})")
ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    h = model.embed_tokens(ids_t)
    print(f"h.shape = {tuple(h.shape)}")
    out, *_ = model(h, None, adaptive=False)

# Коллектив в block.forward не вызывается (pen берётся до forward зеркала),
# поэтому вызываем его вручную по кэшам зеркала каждого слоя.
we = None
cid = None
n_col = 0
for i, layer in enumerate(model.layers):
    col = getattr(layer, 'collective', None)
    mir = layer.mirror
    if model.training:
        hp = getattr(mir, '_cached_hp', None)
        pen = getattr(mir, '_cached_pred_error_norm', None)
    else:
        # eval-режим: зеркало кэширует в буферы, а не в атрибуты
        L = h.shape[1]
        hp_buf = getattr(mir, '_cached_hp_buf', None)
        pen_buf = getattr(mir, '_cached_pred_error_norm_buf', None)
        hp = hp_buf[:, :L] if hp_buf is not None else None
        pen = pen_buf[:, :L] if pen_buf is not None else None
    if col is None:
        continue
    n_col += 1
    if hp is None or pen is None:
        print(f"  L{i}: ПРОПУСК (hp={'None' if hp is None else tuple(hp.shape)}, "
              f"pen={'None' if pen is None else tuple(pen.shape)})")
        continue
    col(h, hp, pen, resvar=mir._residual_var_ema.mean(),
        allow_write=True)
    we_i = col._write_event.bool()
    if i in (0, 12, 23):
        print(f"  L{i}: we={tuple(we_i.shape)} sum={int(we_i.sum())}")
    if we is None:
        we = we_i.clone()
        cid = col._concept_id.clone()
    else:
        cid = torch.where(we_i, col._concept_id, cid)
        we = we | we_i

if we is None:
    print("[error] нет коллективных слоёв с _write_event")
    sys.exit(1)

# Диагностика уверенности (почему границ мало)
pen0 = getattr(model.layers[0].mirror, '_cached_pred_error_norm_buf', None)
if pen0 is not None:
    pen0 = pen0[:, :h.shape[1]]
    conf = torch.sigmoid(-pen0)
    print(f"conf (L0 pen): min={conf.min():.4f} median={conf.median():.4f} "
          f"max={conf.max():.4f} | доля conf>=median: {(conf>=conf.median()).float().mean():.2f}")

print(f"Коллективных слоёв: {n_col}; позиций-границ (OR): {int(we.sum())} из {we.shape[1]}")
print("collective_stats:", model.collective_stats())

words = proj.read_words(ids_t, we)
for b, ws in enumerate(words):
    print(f"\n=== batch {b}: {len(ws)} слов ===")
    print(" | ".join(ws))

cids = proj.concept_spans(cid, we)
print("\n=== id концептов на концах слов ===")
print(cids)
