"""Shared harness for the EVA-CLM performance audit (scratch only, repo untouched)."""
import os
import sys
import time
import json

sys.path.insert(0, r'C:\EVA_CLM_OPT')

import torch

from core import EVAConfig, EVAStack
from core.adaptation import LossBalancer, GradientClipper, build_optimizer, set_active_depth
from core.lr_scheduler import MirrorLRScheduler

DEVICE = 'cpu'


def build_cfg(grad_ckpt=False, **overrides):
    kw = dict(
        D=256, n_layers=4, mlp_groups=8, code_dim=16, code_sparsity=4,
        vocab=512, logit_cache_enabled=True,
        gradient_checkpointing=grad_ckpt, save_dir='.',
    )
    kw.update(overrides)
    return EVAConfig(**kw)


def build(grad_ckpt=False, seed=0, **overrides):
    cfg = build_cfg(grad_ckpt=grad_ckpt, **overrides)
    torch.manual_seed(seed)
    model = EVAStack(cfg)
    model.train()
    set_active_depth(model, cfg.n_layers)
    optimizer = build_optimizer(model, cfg.lr, llrd_decay=getattr(cfg, 'llrd', 0.5),
                                weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
                                optimizer=getattr(cfg, 'optimizer', 'adamw'))
    scheduler = MirrorLRScheduler(model, optimizer, cfg.lr, warmup=cfg.warmup_steps,
                                  target_var=cfg.target_var, mag_threshold=cfg.mag_threshold,
                                  lr_min_ratio=cfg.lr_min_ratio,
                                  max_decay_steps=cfg.max_decay_steps,
                                  var_min_for_lr_decay=cfg.var_min_for_lr_decay,
                                  cfg=cfg)
    balancer = LossBalancer(align=True, align_cap=10.0, eval_interval=cfg.eval_interval,
                            align_every=int(1 if getattr(cfg, 'balancer_align_every', 1) is None
                                            else getattr(cfg, 'balancer_align_every', 1)),
                            kill_terms=None,
                            kill_disable=bool(getattr(cfg, 'aux_kill_disable', False)),
                            safety_aux=('head_wall',))
    clipper = GradientClipper(c=0.1)
    clipper.attach(model)
    return cfg, model, optimizer, scheduler, balancer, clipper


def _detach_state(st):
    if st is None:
        return None
    if isinstance(st, torch.Tensor):
        return st.detach()
    if isinstance(st, (list, tuple)):
        return type(st)(_detach_state(x) for x in st)
    return st


def make_batch(cfg, batch=1, seq=384, seed=1234):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab, (batch, seq), generator=g)
    y = torch.randint(0, cfg.vocab, (batch, seq), generator=g)
    return x, y


def train_step(cfg, model, optimizer, scheduler, balancer, clipper, x, y,
               step, state=None, gs=None, do_optim=True, do_sched=True):
    """One faithful training step (mirrors scripts/train.py:553-651)."""
    model.train()
    h = model.embed_tokens(x)
    out, state, gs, _ = model(h, state, global_state=gs, step=step, tokens=x)
    model.observe_output(model.lm_head(out))
    ce_loss, aux_dict = model.compute_losses(out, y, h_emb=h)
    state = _detach_state(state)
    if gs is not None:
        gs = gs.detach()
    balancer.backward(ce_loss, aux_dict, model.parameters(),
                      phase_model=model, step=step)
    clipper.clip(model.parameters())
    if do_optim:
        optimizer.step()
    model.release_step_graph()
    optimizer.zero_grad(set_to_none=True)
    if do_sched:
        scheduler.step()
    return state, gs, ce_loss, aux_dict


class Timer:
    """Accumulating section timer (perf_counter, CPU)."""

    def __init__(self):
        self.t = {}
        self.n = {}

    def add(self, name, dt):
        self.t[name] = self.t.get(name, 0.0) + dt
        self.n[name] = self.n.get(name, 0) + 1

    def reset(self):
        self.t.clear()
        self.n.clear()


def fmt_table(t, total=None, title='section'):
    rows = sorted(t.items(), key=lambda kv: -kv[1])
    if total is None:
        total = sum(t.values())
    lines = [f'{"section":<42} {"sec":>10} {"%":>7}']
    for k, v in rows:
        lines.append(f'{k:<42} {v:>10.4f} {100.0 * v / max(total, 1e-12):>6.2f}%')
    lines.append(f'{"TOTAL":<42} {total:>10.4f} {"100.00%":>7}')
    return '\n'.join(lines)
