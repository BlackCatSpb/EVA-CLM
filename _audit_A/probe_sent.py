"""sent_masks (windowed) vs sent_masks_stream (AR) parity, and reset_streams."""
import torch
from harness import build

torch.manual_seed(0)
m, cfg = build(explicit_reasoning=False, triad_reason=False)
emb = m.embed
toks = torch.tensor([[5, 7, 2, 9, 3, 2, 2, 4, 8, 2, 6]])
sep_w, bos_w, rel_w = emb.sent_masks(toks)

emb._sent_rel_ptr.zero_()
rows = []
for t in range(toks.shape[1]):
    one = toks[:, t:t + 1]
    s, b, r = emb.sent_masks_stream(one)
    rows.append((bool(s[0, 0]), bool(b[0]), int(r[0, 0])))
print('pos  window(sep,bos,rel)   stream(sep,bos,rel)')
bad = 0
for t in range(toks.shape[1]):
    w = (bool(sep_w[0, t]), bool(bos_w[0, t]), int(rel_w[0, t]))
    s = rows[t]
    ok = w == s
    bad += (not ok)
    print(f'{t:>3}  {w}  {s}  {"OK" if ok else "MISMATCH"}')
print('mismatches:', bad)
