import sys; sys.path.insert(0, '.')
import torch
from core.config import WideBindConfig
from core.stack import WideBindStack

cfg = WideBindConfig(D=256, n_layers=4, vocab=1024, mlp_groups=4, bind_K=32,
    memory_bank=True, mem_l1_slots=3, mem_l2_slots=8, mem_l3_concepts=4, mem_bridge_dim=64,
    bridge_conn=0.0, maturation_enabled=True)
model = WideBindStack(cfg)
model.eval()

x = torch.randint(0, 1024, (2, 32))
x[0, 8] = 2; x[1, 5] = 2
h = model.embed_tokens(x)

# Test with step=0 (mat_gate~0, should NOT write)
with torch.no_grad():
    out, _, _, _ = model(h, step=0, tokens=x)
diag = model.memory_bank.get_diagnostics()
l1_0 = diag['l1_write_idx']
l2_0 = diag['l2_write_idx']
print(f'step=0, mat_gate~0: L1={l1_0} L2={l2_0}')

# Test with step=10000 (mat_gate>0.3, should write)
model.memory_bank.reset()
with torch.no_grad():
    out, _, _, _ = model(h, step=10000, tokens=x)
diag = model.memory_bank.get_diagnostics()
l1_10k = diag['l1_write_idx']
l2_10k = diag['l2_write_idx']
print(f'step=10000, mat_gate>0.3: L1={l1_10k} L2={l2_10k}')

assert l1_0 == 0, f'Expected L1=0 at step 0, got {l1_0}'
assert l2_0 == 0, f'Expected L2=0 at step 0, got {l2_0}'
assert l1_10k > 0, f'Expected L1>0 at step 10000, got {l1_10k}'
assert l2_10k > 0, f'Expected L2>0 at step 10000, got {l2_10k}'
print('ALL ASSERTIONS PASSED')
