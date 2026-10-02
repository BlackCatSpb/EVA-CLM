import json
path = r'C:\EVA_CLM_OPT\_audit_B\trace.json'
ev = json.load(open(path, encoding='utf-8'))['traceEvents']
puts = [e for e in ev if e.get('name') == 'aten::_index_put_impl_']
spans = [e for e in ev if e.get('ph') == 'X' and e.get('dur') is not None]
for p in puts[:8]:
    ts, end = p['ts'], p['ts'] + p['dur']
    parents = [s for s in spans if s['ts'] <= ts and s['ts'] + s['dur'] >= end and s is not p and s.get('name') != 'aten::_index_put_impl_']
    parents.sort(key=lambda s: s['dur'])
    chain = parents[:6]
    print(f"put dur={p['dur']/1000:.2f}ms ts={ts:.0f} parents:")
    for c in chain:
        print(f"    {c.get('name','?')[:60]:<60} dur={c['dur']/1000:.2f}ms")
