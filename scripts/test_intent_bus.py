"""Synthetic proving-ground for the cross-layer Intent Bridge.

Tests (on a tiny model, cpu/fp32):
  1. forward + backward run without crash / NaN
  2. intent_probe receives gradient (trainable, self-term in bus)
  3. mirror w_intent receives gradient (gate path, raw AGC grad nonzero at 0)
  4. bus_head_proj receives gradient (phase-2 stencil to projector)
  5. CE decreases over a handful of steps (sanity of learning)
"""
import math, sys
import torch

from core.stack import WideBindStack
from core.config import WideBindConfig


def tiny_cfg():
    return WideBindConfig(
        n_layers=4, D=64, bind_K=16, mlp_groups=4,
        mirror_k=8, mirror_k_staircase=False, vocab=256,
        code_dim=32, code_sparsity=6, head_mode='sigmoid_coded',
        intent_bridge=True, head_normalize=True,
    )


def grad_norm(model, name):
    for n, p in model.named_parameters():
        if n == name:
            return p.grad.norm().item() if p.grad is not None else None
    return None


def main():
    torch.manual_seed(0)
    device = 'cpu'
    cfg = tiny_cfg()
    model = WideBindStack(cfg).to(device)
    model.train()
    nparams = sum(p.numel() for p in model.parameters())
    print(f'model params = {nparams:,}')

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # synthetic: random next-token stream (batch, seq)
    B, L = 4, 32
    losses = []
    for step in range(25):
        x = torch.randint(2, cfg.vocab, (B, L), device=device)  # skip PAD(0)/EOS(2)
        h = model.embed_tokens(x)
        out, state, _, _ = model(h, step=step)
        ce, aux = model.compute_losses(out[:, :-1], x[:, 1:])
        loss = ce
        opt.zero_grad()
        loss.backward()
        # basic global clip (AGC exercised separately in live run)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        # produce salience for NEXT step (1-step delay, as in notebook)
        model.observe_output(out)
        losses.append(ce.item())
        if step % 5 == 0 or step == 24:
            p_probe = grad_norm(model, 'intent_probe.weight')
            p_wint = grad_norm(model, 'layers.0.mirror.w_intent')
            p_bus = grad_norm(model, 'bus_head_proj.weight')
            print(f'step {step:2d}  ce={ce.item():.4f}  '
                  f'grad probe={p_probe:.3e} w_intent={p_wint:.3e} bus={p_bus:.3e}')

    ok = (not math.isnan(losses[-1])) and all(not math.isnan(l) for l in losses)
    improved = losses[-1] < losses[0]
    print(f'\nfinal ce={losses[-1]:.4f}  start ce={losses[0]:.4f}  improved={improved}')
    print('OK' if ok and improved else 'CHECK')

    # targeted gradient assertions (independent of loss trend)
    ok_probe = grad_norm(model, 'intent_probe.weight') is not None
    ok_wint = grad_norm(model, 'layers.0.mirror.w_intent') is not None
    ok_bus = grad_norm(model, 'bus_head_proj.weight') is not None
    print(f'grad flow: probe={ok_probe} w_intent={ok_wint} bus={ok_bus}')
    return 0 if (ok and improved and ok_probe and ok_wint and ok_bus) else 1


if __name__ == '__main__':
    sys.exit(main())
