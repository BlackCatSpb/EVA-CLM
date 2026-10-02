import json
from collections import Counter, defaultdict
path = r'C:\EVA_CLM_OPT\_audit_B\trace.json'
ev = json.load(open(path, encoding='utf-8'))['traceEvents']
spans = [e for e in ev if e.get('ph') == 'X' and e.get('dur') is not None and 'name' in e]
# module totals
mod = Counter()
for e in spans:
    n = e.get('name', '')
    if n.startswith('nn.Module:'):
        mod[n] += e['dur'] / 1000
print('=== nn.Module total ms (trace step) ===')
for k, v in mod.most_common(20):
    print(f'  {k:<48} {v:8.2f}ms')
# top-level python funcs
py = Counter()
for e in spans:
    n = e.get('name', '')
    if 'core/' in n or 'common.py' in n or 'torch/' in n:
        py[n] += e['dur'] / 1000
print('=== python frames (only self time not available; total containment) top-20 ===')
for k, v in py.most_common(20):
    print(f'  {k:<60} {v:8.2f}ms')
# sum of index and index_put
for nm in ('aten::index', 'aten::_index_put_impl_', 'aten::bmm', 'aten::mm', 'aten::mul', 'aten::add_', 'aten::sum'):
    tot = sum(e['dur'] for e in spans if e.get('name') == nm) / 1000
    cnt = sum(1 for e in spans if e.get('name') == nm)
    print(f'{nm:<28} {tot:8.2f}ms  n={cnt}')
