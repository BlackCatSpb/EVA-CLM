"""WideBind Mini-probe: validates SMF (α on mirror) + variant A (mirror-conditioned
SwiGLU) on a reduced Mini that fits the local MX550 (2GB). Mechanism is scale-invariant;
the full D=896/24-layer Mini is the scale-up.

Checks:
  - no NaN / inf in loss or activations
  - per-layer mirror magnitude: L0 must NOT dominate deep layers (effective depth recovers)
  - SMF alpha: starts ~0.5 uniform, is learnable (per-layer spread)
  - variant A: mlp_gate_b grows >0  -> MLP becomes mirror-conditioned (not asleep)
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import WideBindConfig
from core.stack import WideBindStack

torch.manual_seed(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={DEVICE}")

VOCAB = 4096
SEQ = 128
BATCH = 4
STEPS = 400
STREAM = os.path.join('wb', 'token_stream_TALES_eos.bin')

cfg = WideBindConfig(
    D=512, n_layers=16, bind_K=32, vocab=VOCAB,
    mlp_groups=8, mirror_k=16, mirror_k_staircase=False,
    seq_len=SEQ, batch_size=BATCH,
    private_mem=True, gradient_checkpointing=False, use_amp=False,
    mask_eos=True, lambda_lr_hierarchy=True,
)
model = WideBindStack(cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"params={n_params/1e6:.1f}M layers={cfg.n_layers} D={cfg.D} G={cfg.mlp_groups} k={cfg.mirror_k}")

# ── data ──
raw = np.fromfile(STREAM, dtype=np.uint16).astype(np.int64)
raw = np.clip(raw, 0, VOCAB - 1)
n_seq = min(len(raw) // (SEQ + 1), 30000)
toks = raw[: n_seq * (SEQ + 1)].reshape(n_seq, SEQ + 1)
inp = toks[:, :-1]; tgt = toks[:, 1:]
print(f"dataset: {n_seq} sequences of len {SEQ}")

opt = torch.optim.AdamW(model.param_groups(lr=cfg.lr, weight_decay=cfg.weight_decay),
                        lr=cfg.lr, weight_decay=cfg.weight_decay, foreach=False)

def metrics():
    mags = [model.layers[i].mirror._last_magnitude.item() for i in range(cfg.n_layers)]
    alps = [model.layers[i].mirror._cached_alpha.mean().item() for i in range(cfg.n_layers)]
    bmlp = [model.layers[i].mlp.mlp_gate_b.item() for i in range(cfg.n_layers)]
    return mags, alps, bmlp

t0 = time.time()
for step in range(STEPS):
    idx = np.random.randint(0, n_seq, size=BATCH)
    x = torch.tensor(inp[idx], dtype=torch.long, device=DEVICE)
    y = torch.tensor(tgt[idx], dtype=torch.long, device=DEVICE)
    opt.zero_grad(set_to_none=True)
    h = model.embed(x)
    h_out, _, _, _ = model(h, adaptive=False)
    loss = model.compute_loss(h_out, y)
    if not torch.isfinite(loss):
        print(f"  !! NON-FINITE loss at step {step}: {loss.item()}")
        break
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    if not torch.isfinite(gnorm):
        print(f"  !! NON-FINITE grad at step {step}")
        break
    opt.step()

    if step % 10 == 0 or step == STEPS - 1:
        mags, alps, bmlp = metrics()
        mags = np.array(mags); alps = np.array(alps); bmlp = np.array(bmlp)
        l0, lN = mags[0], mags[-1]
        ratio = mags.max() / (mags.min() + 1e-9)
        print(f"step {step:4d} | CE {loss.item():.3f} | gnorm {gnorm.item():.2f} | "
              f"|mirror| L0={l0:.3f} L{len(mags)-1}={lN:.3f} "
              f"max/min={ratio:5.1f} | alpha L0={alps[0]:.3f} L{len(alps)-1}={alps[-1]:.3f} "
              f"a-spread={alps.max()-alps.min():.3f} | mlp_b mean={bmlp.mean():.4f} "
              f"max={bmlp.max():.4f} | {time.time()-t0:.0f}s")

print("done.")
