"""Op-level (torch.profiler) and Python-level (cProfile) breakdown of one training step."""
import sys, time, json, cProfile, pstats, io
sys.path.insert(0, r'C:\EVA_CLM_OPT')
sys.path.insert(0, r'C:\EVA_CLM_OPT\_audit_B')
import torch
from common import build, make_batch, train_step

torch.set_num_threads(8)
OUT = r'C:\EVA_CLM_OPT\_audit_B\ops_out.txt'


def main(grad_ckpt=False):
    cfg, model, opt, sched, bal, clip = build(grad_ckpt=grad_ckpt)
    x, y = make_batch(cfg, batch=1, seq=384)
    state = gs = None
    for s in range(3):
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=s, state=state, gs=gs)

    lines = [f'===== torch.profiler (gc={grad_ckpt}) =====']
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False,
                 profile_memory=False, with_stack=False) as prof:
        state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=100, state=state, gs=gs)
    ev = prof.events()
    total_ev = len(ev)
    lines.append(f'total profiler events: {total_ev}')
    ka = prof.key_averages()
    rows = sorted(ka, key=lambda e: -e.self_cpu_time_total)
    lines.append(f'{"op":<52} {"self_cpu_s":>10} {"cpu_total_s":>11} {"#calls":>8}')
    for e in rows[:35]:
        lines.append(f'{e.key[:52]:<52} {e.self_cpu_time_total/1e6:>10.4f} '
                     f'{e.cpu_time_total/1e6:>11.4f} {e.count:>8}')
    # count aten ops vs python/builtin
    aten = [e for e in ka if e.key.startswith('aten::')]
    lines.append(f'aten op kinds: {len(aten)}; total aten calls: {sum(e.count for e in aten)}')
    # top op kinds by call count
    rows_c = sorted(aten, key=lambda e: -e.count)
    lines.append('top-20 aten by call count:')
    for e in rows_c[:20]:
        lines.append(f'  {e.key:<52} {e.count:>8} calls  self={e.self_cpu_time_total/1e6:.4f}s')
    # group by prefix (dispatcher categories)
    from collections import Counter
    cnt = Counter()
    tsum = Counter()
    for e in aten:
        cat = e.key.split('::')[1].split('.')[0]
        cnt[cat] += e.count
        tsum[cat] += e.self_cpu_time_total
    lines.append('aten categories by self time:')
    for cat, t in tsum.most_common(25):
        lines.append(f'  {cat:<40} self={t/1e6:.4f}s calls={cnt[cat]}')

    # ---------- cProfile ----------
    pr = cProfile.Profile()
    pr.enable()
    state, gs, _, _ = train_step(cfg, model, opt, sched, bal, clip, x, y, step=101, state=state, gs=gs)
    pr.disable()
    sio = io.StringIO()
    ps = pstats.Stats(pr, stream=sio).sort_stats('tottime')
    ps.print_stats(40)
    lines.append('===== cProfile tottime top-40 =====')
    lines.append(sio.getvalue())

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines[:80]))


if __name__ == '__main__':
    gc = len(sys.argv) > 1 and sys.argv[1] == 'gc'
    main(grad_ckpt=gc)
