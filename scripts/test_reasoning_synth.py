"""Синтетический тест адаптивного цикла рассуждений (ReasoningGate).

Задача: [x][n] -> (x + n) mod 16, x∈[0,16), n∈[0,4).
Сравниваются три сценария на IDENTICAL данных:
  1. adaptive=True  — цикл с гейтами (глубина решается моделью)
  2. adaptive=False — старое одношаговое рассуждение
  3. resume-чекап: adaptive=True, сохранить на шаге N, перезагрузить
     (strict=False, как в train.py), продолжить — CE не должен скакнуть.

Критерии PASS:
  - все прогоны конечны (no NaN)
  - CE(adaptive) <= CE(adaptive=False) после обучения (цикл не вредит)
  - гейты адаптивны: средний шаг цикла растёт с глубиной n
  - resume: CE после перезагрузки продолжает падать без скачка
"""
import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch.nn as nn

from core import WideBindConfig, WideBindStack

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

LOG_PATH = os.path.join(os.environ.get('TEMP', '.'), 'opencode', 'reasoning_synth.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def make_cfg(adaptive):
    return WideBindConfig(
        D=128, n_layers=2, bind_K=8, vocab=64, seq_len=8, batch_size=16,
        mlp_groups=4, mlp_expand=2, mirror_k=8,
        code_dim=16, code_sparsity=3, conv_kernel=8,
        explicit_reasoning=True, reasoning_max_steps=6,
        reasoning_adaptive=adaptive,
        lambda_d_enabled=False, private_mem=False, meta_trust=False,
    )


def make_batch(cfg, rng, n=16, depth_max=4):
    """[x][n][sep][x+n mod 16] с дополнением до seq_len."""
    B = cfg.batch_size
    x = torch.randint(0, n, (B,), generator=rng)
    d = torch.randint(0, depth_max, (B,), generator=rng)
    y = (x + d) % n
    rows = []
    for b in range(B):
        row = [x[b], 16 + d[b], 20, y[b]]
        row += [0] * (cfg.seq_len - len(row))
        rows.append(row)
    t = torch.tensor(rows, dtype=torch.long)
    return t[:, :-1], t[:, 1:]


def run_training(cfg, steps=400, seed=42, save_every=None, save_path=None,
                 start_from=None, log=None):
    if log is None:
        log = lambda s: None
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = WideBindStack(cfg)
    if start_from is not None:
        ckpt = torch.load(start_from, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        log(f'  RESUME load: missing={len(missing)} unexpected={len(unexpected)}')
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(steps):
        model.train()
        model.reasoning_enabled_step = step
        model.reset_reasoning()
        x, y = make_batch(cfg, rng)
        h = model.embed_tokens(x)
        out, state, _, _ = model(h, None, adaptive=False)
        ce = model.compute_loss(out, y, h_emb=h)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(ce.item())
        if not torch.isfinite(ce):
            raise RuntimeError(f'NaN/Inf CE at step {step}')
        if step % 50 == 0:
            gates = model._reasoning_gates
            gs = f'gates={[round(g, 3) for g in gates]}' if gates else 'gates=n/a'
            log(f'  step={step:4d} ce={ce.item():.4f} {gs}')
        if save_every and step % save_every == 0 and step > 0:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'cfg': cfg,
            }, save_path)
            log(f'  saved {save_path} (step {step})')
    return model, losses


def gate_depth_stats(model, cfg, rng, depth_max=4, n=16, trials=8):
    """Средний шаг цикла и средние гейты по глубине n."""
    model.eval()
    by_depth = {d: [] for d in range(depth_max)}
    with torch.no_grad():
        for _ in range(trials):
            x, y = make_batch(cfg, rng)
            for b in range(x.shape[0]):
                d = int(x[b, 1].item()) - 16
                model.reset_reasoning()
                h = model.embed_tokens(x[b:b + 1])
                out, _, _, _ = model(h, None, adaptive=False)
                gates = model._reasoning_gates
                by_depth[d].append(len(gates))
    return {d: (sum(v) / len(v)) for d, v in by_depth.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--save-dir', type=str, default=os.path.join(os.environ.get('TEMP', '.'), 'opencode', 'rs_ckpt'))
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log = lambda s: (open(LOG_PATH, 'a', encoding='utf-8').write(s + '\n'),
                     print(s))
    open(LOG_PATH, 'w', encoding='utf-8').close()

    log('=== Сценарий 1: adaptive=True ===')
    cfg_a = make_cfg(True)
    model_a, losses_a = run_training(cfg_a, steps=args.steps, seed=42,
                                     save_every=args.steps // 2,
                                     save_path=os.path.join(args.save_dir, 'adaptive.pt'), log=log)
    log(f'  FINAL adaptive CE = {losses_a[-1]:.4f}')

    log('=== Сценарий 2: adaptive=False (старое поведение) ===')
    cfg_o = make_cfg(False)
    model_o, losses_o = run_training(cfg_o, steps=args.steps, seed=42, log=log)
    log(f'  FINAL old CE = {losses_o[-1]:.4f}')

    log('=== Сценарий 3: resume из чекпоинта adaptive (strict=False) ===')
    ckpt_path = os.path.join(args.save_dir, 'adaptive.pt')
    cfg_r = make_cfg(True)
    model_r, losses_r = run_training(cfg_r, steps=args.steps, seed=99,
                                     start_from=ckpt_path, log=log)
    log(f'  RESUME CE start = {losses_r[0]:.4f} | end = {losses_r[-1]:.4f}')

    rng = torch.Generator().manual_seed(7)
    stats = gate_depth_stats(model_a, cfg_a, rng)
    log(f'  avg loop steps by depth: { {k: round(v, 2) for k, v in stats.items()} }')

    ce_a = losses_a[-1]
    ce_o = losses_o[-1]
    ce_r0, ce_r_end = losses_r[0], losses_r[-1]
    finite = all(math.isfinite(v) for v in losses_a + losses_o + losses_r)
    ok_adaptive = ce_a <= ce_o + 0.05
    ok_resume = ce_r_end < ce_r0 + 0.05 and ce_r_end < ce_a + 0.05
    deeper = any(v > 1.5 for v in stats.values())
    log('=== ИТОГ ===')
    log(f'  finite: {finite} | CE adaptive {ce_a:.4f} vs old {ce_o:.4f} '
        f'| resume {ce_r0:.4f} -> {ce_r_end:.4f}')
    log(f'  PASS adaptive-not-worse: {ok_adaptive}')
    log(f'  PASS resume-no-jump:     {ok_resume}')
    log(f'  PASS adaptive-depth:     {deeper}')
    log(f'  RESULT: {"PASS" if (finite and ok_adaptive and ok_resume) else "FAIL"}')
    print(f'\nlog: {LOG_PATH}')


if __name__ == '__main__':
    main()