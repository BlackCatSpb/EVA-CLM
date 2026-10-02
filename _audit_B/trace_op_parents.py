import json, sys
path = r'C:\EVA_CLM_OPT\_audit_B\trace.json'
ev = json.load(open(path, encoding='utf-8'))['traceEvents']
spans = [e for e in ev if e.get('ph') == 'X' and e.get('dur') is not None]
def parents_of(e, maxn=6):
    ts, end = e['ts'], e['ts'] + e['dur']
    ps = [s for s in spans if s['ts'] <= ts and s['ts'] + s['dur'] >= end and s is not e
          and s.get('name') != e.get('name')]
    ps.sort(key=lambda s: s['dur'])
    return ps[:maxn]
for opname in sys.argv[1:]:
    ops = [e for e in ev if e.get('name') == opname]
    tot = sum(e['dur'] for e in ops) / 1000
    print(f'=== {opname}: {len(ops)} events, total {tot:.2f}ms ===')
    from collections import Counter
    c = Counter()
    for e in ops:
        ps = parents_of(e, 3)
        # nearest python-ish parent
        key = ps[-1].get('name', '?') if ps else '?'
        c[key] += e['dur']
    for k, v in c.most_common(10):
        print(f'   {k[:70]:<70} {v/1000:8.2f}ms')
    for e in ops[:3]:
        print(f'   sample dur={e["dur"]/1000:.3f}ms parents:', ' <- '.join(p.get('name','?')[:40] for p in parents_of(e,4)))
