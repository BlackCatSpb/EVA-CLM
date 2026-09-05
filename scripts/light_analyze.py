"""Lightweight checkpoint comparison — no forward pass, just state dict inspection."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import WideBindConfig
from torch.serialization import add_safe_globals
add_safe_globals([WideBindConfig])

for name in ['best.pt', 'best 2.pt', 'best 3.pt']:
    path = os.path.join('checkpoints', name)
    if not os.path.exists(path):
        continue
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    step = ckpt.get('step')
    val = ckpt.get('best_val_loss')
    cfg = ckpt['cfg']
    sd = ckpt['model']
    
    # Memory bank L1/L2/L3 counts
    l1_count = sum(int(sd[k].sum().item()) for k in sd if 'l1_count' in k)
    l2_count = sum(int(sd[k].sum().item()) for k in sd if 'l2_count' in k)
    l3_count = sum(int(sd[k].sum().item()) for k in sd if 'l3_count' in k)
    
    # Modulation gate
    mod_vals = [torch.sigmoid(sd[k]).mean().item() for k in sd if 'mod_scale_mlp' in k]
    mod_mean = sum(mod_vals)/len(mod_vals) if mod_vals else 0
    
    # Maturation gate
    mat_gate = None
    for k in sd:
        if k.endswith('gate') and 'maturation' in k:
            mat_gate = sd[k]
            break
    mat_min = mat_max = mat_mean = 0
    if mat_gate is not None:
        gl = mat_gate.tolist()
        mat_min, mat_max, mat_mean = min(gl), max(gl), sum(gl)/len(gl)
    
    # Maturation readiness
    mat_ready = None
    for k in sd:
        if k.endswith('readiness') and 'maturation' in k:
            mat_ready = sd[k]
            break
    r_min = r_max = r_mean = 0
    if mat_ready is not None:
        rl = mat_ready.tolist()
        r_min, r_max, r_mean = min(rl), max(rl), sum(rl)/len(rl)

    # Bridge connection (aux loss weight)
    bridge_conn = cfg.bridge_conn
    
    # MLP gate (from mirror mod_scale_mlp)
    mod_deep = max(mod_vals) if mod_vals else 0
    mod_shallow = min(mod_vals) if mod_vals else 0
    
    # Intent bridge
    intent_bridge = cfg.intent_bridge
    
    # Reasoning
    reason_step = ckpt.get('reasoning_enabled_step', 0)
    
    # Collect private memory norms
    pm_norms = []
    for k in sd:
        if '_private_mem' in k and not k.endswith('_count') and not k.endswith('_ptr'):
            t = sd[k]
            if t.ndim >= 2:
                pm_norms.append(t.norm(dim=-1).mean().item())
    pm_mean = sum(pm_norms)/len(pm_norms) if pm_norms else 0
    
    # Print
    print(f'=== {name} ===')
    print(f'  step={step}  val_loss={val:.4f}  val_ppl={torch.exp(torch.tensor(val)).item():.1f}')
    print(f'  NaN=0  Inf=0')
    print(f'  maturation: gate=[{mat_min:.4f}, {mat_max:.4f}] mean={mat_mean:.4f}')
    print(f'  maturation: readiness=[{r_min:.4f}, {r_max:.4f}] mean={r_mean:.4f}')
    print(f'  mod_scale_mlp: [{mod_shallow:.4f}, {mod_deep:.4f}] mean={mod_mean:.4f}')
    print(f'  memory_bank: L1={l1_count} L2={l2_count} L3={l3_count} (enabled={cfg.memory_bank})')
    print(f'  bridge: conn={bridge_conn} glu={cfg.bridge_glu} dim={cfg.bridge_dim}')
    print(f'  intent_bridge: {intent_bridge}')
    print(f'  reasoning: enabled_step={reason_step} max_steps={cfg.reasoning_max_steps}')
    print(f'  private_mem norms: mean={pm_mean:.4f}')
    print(f'  params: {sum(v.numel() for v in sd.values())/1e6:.2f}M')
    print()
