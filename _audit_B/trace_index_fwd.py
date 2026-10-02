import json
path = r'C:\EVA_CLM_OPT\_audit_B\trace.json'
ev = json.load(open(path, encoding='utf-8'))['traceEvents']
spans = [e for e in ev if e.get('ph') == 'X' and e.get('dur') is not None]
def parents_of(e, maxn=8):
    ts, end = e['ts'], e['ts'] + e['dur']
    ps = [s for s in spans if s['ts'] <= ts and s['ts'] + s['dur'] >= end and s is not e
          and s.get('name') != e.get('name')]
    ps.sort(key=lambda s: s['dur'])
    return ps[:maxn]
idxs = [e for e in ev if e.get('name') == 'aten::index']
print('aten::index forward events:', len(idxs))
for e in idxs:
    print(f"index dur={e['dur']/1000:.3f}ms")
    for c in parents_of(e, 7):
        print(f"    {c.get('name','?')[:64]:<64} dur={c['dur']/1000:.2f}ms")
