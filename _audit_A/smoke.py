import torch
from harness import build, fwd

torch.manual_seed(0)
m, cfg = build()
x = torch.randint(0, cfg.vocab, (2, 16))
y = torch.randint(0, cfg.vocab, (2, 16))
out, st, gs, rb = fwd(m, cfg, x, adaptive=False)
loss = m.compute_loss(out, y)
loss.backward()
print('forward/backward OK, loss=', float(loss))
n = sum(p.numel() for p in m.parameters())
print('params', n)
print('buffers', len(list(m.named_buffers())))
