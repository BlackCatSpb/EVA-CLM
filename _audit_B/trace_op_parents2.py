import json, sys, heapq
path = r'C:\EVA_CLM_OPT\_audit_B\trace.json'
ev = json.load(open(path, encoding='utf-8'))['traceEvents']
spans = [e for e in ev if e.get('ph') == 'X' and e.get('dur') is not None and 'name' in e]
spans.sort(key=lambda e: e['ts'])
# sweep line: assign each span its parent chain by containment
open_spans = []  # heap of (end_ts, idx)
parent = {}
starts = sorted(range(len(spans)), key=lambda i: spans[i]['ts'])
events_sorted = sorted(spans, key=lambda e: (e['ts'], -e['dur']))
stack = []
for e in events_sorted:
    while stack and stack[-1]['ts'] + stack[-1]['dur'] < e['ts']:
        stack.pop()
    if stack and stack[-1]['ts'] <= e['ts'] and stack[-1]['ts'] + stack[-1]['dur'] >= e['ts'] + e['dur'] and stack[-1] is not e:
        parent[id(e)] = stack[-1]
    # only push if it contains future events
    if e['dur'] > 0:
        stack.append(e)
    # cap stack size sanity
    if len(stack) > 200:
        stack = stack[-100:]

def chain(e, n=8):
    out = []
    p = parent.get(id(e))
    while p is not None and len(out) < n:
        out.append(p)
        p = parent.get(id(p))
    return out

from collections import Counter
for opname in sys.argv[1:]:
    ops = [e for e in spans if e.get('name') == opname]
    tot = sum(e['dur'] for e in ops) / 1000
    print(f'=== {opname}: {len(ops)} events, total {tot:.2f}ms ===')
    c = Counter()
    for e in ops:
        ch = chain(e, 4)
        key = ch[-1].get('name', '?') if ch else '?'
        c[key] += e['dur']
    for k, v in c.most_common(8):
        print(f'   {k[:70]:<70} {v/1000:8.2f}ms')
