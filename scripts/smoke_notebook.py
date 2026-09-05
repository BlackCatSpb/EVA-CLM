"""Faithful CPU smoke test of the colab.ipynb training path against current core."""
import os, sys, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from core import WideBindConfig, WideBindStack, MirrorLRScheduler
from core.adaptation import LossBalancer, GradientClipper, set_active_depth

# ---- cell 4 cfg (notebook flags, tiny D for CPU) ----
cfg = WideBindConfig(
    D=256, n_layers=4, bind_K=32, vocab=300, mask_eos=False,
    mlp_groups=32, mlp_expand=4, seq_len=48,
    lr=3e-4, max_steps=20, warmup_steps=5,
    log_interval=55, eval_interval=2000, save_interval=987,
    scheduler='mirror', lr_boost_max=2.0, lr_improve_tol=0.002,
    per_layer_ls_lr=True, ls_ema_fast=0.99, ls_ema_slow=0.999,
    ls_mult_min=0.5, ls_mult_max=2.0, ls_mirror_mult_max=2.0,
    private_mem=True, expert_asymmetry=True, meta_trust=True,
    grad_clip=0.5, conv_kernel=48, gradient_checkpointing=False,
    head_mode='sigmoid_coded', head_normalize=True,
    bind_twist_mode='trajectory_spiral', bind_traj_dims=3,
    hybrid_alpha_max=0.7, hybrid_alpha_min=0.3, w_pred_scale_init=3.0,
    bind_twist_gate=True, collective_layer=True, collective_layer_idx=None,
    collective_read_out=True, collective_uncert_theta=0.5,
    collective_uncert_kappa=3.0, collective_contra_thresh=-0.1,
    collective_contra_gain=6.0, collective_maturity_thresh=0.12,
    surprisal_weight=0.3, branch_balance_weight=0.1, variable_precision=True,
    precision_threshold=0.3, explicit_reasoning=False,
    use_amp=False, intent_bridge=True, bridge_glu=True, bridge_conn=0.1,
    intent_topdown=True, orth_weight=0.0,
)
# scheduler cfg defaults (mirror MirrorLRScheduler expectations)
cfg.target_var = 1e-3
cfg.mag_threshold = 1.0
cfg.lr_min_ratio = 0.1
cfg.max_decay_steps = 100000
cfg.var_min_for_lr_decay = 1e-4
cfg.llrd = 0.5
cfg.eval_interval = 2000

device = 'cpu'
model = WideBindStack(cfg).to(device)
assert getattr(model, 'bridge', None) is not None, "bridge NOT created -> notebook cfg broken"
print(f'Model params: {model.param_count():,}; bridge params: {sum(p.numel() for p in model.bridge.parameters()):,}')

# ---- cell 6 optimizer ----
param_groups = model.param_groups()
optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
scheduler = MirrorLRScheduler(model, optimizer, cfg.lr, warmup=cfg.warmup_steps,
    target_var=cfg.target_var, mag_threshold=cfg.mag_threshold,
    lr_min_ratio=cfg.lr_min_ratio, max_decay_steps=cfg.max_decay_steps,
    var_min_for_lr_decay=cfg.var_min_for_lr_decay, cfg=cfg)
print('Scheduler: MirrorLRScheduler')

# ---- cell 9 balancer ----
set_active_depth(model, 8)
balancer = LossBalancer(align=True, align_cap=10.0, eval_interval=cfg.eval_interval)
clipper = GradientClipper(c=0.01)
cfg.gradalign_weight = 0.3

# ---- cell 10 training loop (faithful, few steps) ----
state = None
intent_state = None
batch, seq_len = 2, cfg.seq_len
for step in range(10):
    model.train()
    x = torch.randint(0, cfg.vocab, (batch, seq_len))
    y = torch.randint(0, cfg.vocab, (batch, seq_len))
    h = model.embed_tokens(x)
    out, state, _, _ = model(h, state, step=step, intent_state=intent_state)
    model.observe_output(model.lm_head(out))
    ce_loss, aux_dict = model.compute_losses(out, y, h_emb=h)
    assert 'bridge_conn' in aux_dict, "bridge_conn missing from aux_dict -> notebook path broken"
    lc_val = aux_dict['bridge_conn'].item()
    # gradalign (notebook block)
    ga_w = float(getattr(cfg, 'gradalign_weight', 0.0))
    if ga_w > 0:
        _outs = [getattr(l, '_cache_mlp_out', None) for l in model.layers]
        _mods = [getattr(l, '_cache_mlp_mod', None) for l in model.layers]
        if all(o is not None for o in _outs) and all(m is not None for m in _mods):
            _g = torch.autograd.grad(ce_loss, _outs, retain_graph=True, allow_unused=True)
            _ga = sum(o.norm() for o in _g) / max(1, len(_g))
            aux_dict['gradalign'] = _ga
    balancer.backward(ce_loss, aux_dict, model.parameters())
    clipper.clip(model.parameters())
    optimizer.step()
    scheduler.step()
    _mm = torch.stack([l.mirror._last_mlp_mod.detach().mean() for l in model.layers]).mean().item() if hasattr(model.layers[0].mirror, '_last_mlp_mod') else float('nan')
    _mg = model.maturation.gate.detach().mean().item() if getattr(model, 'maturation', None) is not None else float('nan')
    print(f'step={step:>2} ce={ce_loss.item():.3f} bridge_conn={lc_val:.4f} mlp_mod={_mm:.3f} mat_gate={_mg:.3f}')

# ---- checkpoint save (the StopIteration-prone expression, now safe) ----
sd = model.state_dict()
safe_names = [ _pn.get(id(p), 'external') for _pn in [{id(pp): n for n, pp in model.named_parameters()}] for p in [p for g in optimizer.param_groups for p in g['params']] ]
assert len(safe_names) == len(list(model.named_parameters())), "param_names count mismatch"
tmp = os.path.join(tempfile.gettempdir(), 'smoke_bridge_ckpt.pt')
torch.save({'step': 10, 'model': sd, 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'param_names': safe_names, 'cfg': cfg}, tmp)
print('Checkpoint saved with', len(safe_names), 'param names (no StopIteration)')
# ---- load round-trip ----
from core.migrate import migrate_state_dict
ck = torch.load(tmp, map_location='cpu', weights_only=False)
model2 = WideBindStack(cfg).to(device)
migrated, _ = migrate_state_dict(ck['model'], model2)
miss, unexp = model2.load_state_dict(migrated, strict=False)
print(f'Reload: missing={len(miss)} unexpected={len(unexp)} (bridge params present: {any(k.startswith("bridge.") for k in ck["model"].keys())})')
assert not miss, f'missing keys on reload: {miss[:5]}'
print('SMOKE TEST PASSED: notebook training path is correct against current core.')
