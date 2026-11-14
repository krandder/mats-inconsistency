"""Score Amendment 2E (A2E.1-A2E.3) -> results/a2e_scores.json. Rules as registered (hash 1572a700)."""
import json
import numpy as np

D_ = '/home/agent/workspaces/mats/work/input-bits'
S = (1000, 1001, 4000, 4001)
med = lambda v: float(np.median(v))
cells = {}
for n in (16, 64, 128):
    e = [json.load(open(f'{D_}/runs/a2e_n{n}_s{s}.json')) for s in S]
    b = [json.load(open(f'{D_}/runs/a2b_n{n}_s{s}.json')) for s in S]
    d = [json.load(open(f'{D_}/runs/a2d_n{n}_s{s}.json')) for s in S]
    cc = [[(r['C'][i] - r['C'][0]) / (1 - r['C'][0]) for i in (1, 2, 3)] for r in e]
    cb = [[r['C'][i] / r['C'][4] for i in (1, 2, 3)] for r in b]
    slope = [x[0] / 0.25 for x in cc]
    na = [(dd['D_coord'] - sl) / (1 - sl) for dd, sl in zip(d, slope)]
    cells[n] = dict(Cc=[med([x[i] for x in cc]) for i in range(3)], Cb=[med([x[i] for x in cb]) for i in range(3)],
                    C0=med([r['C'][0] for r in e]), C0_nets=[r['C'][0] for r in e], cond_slope=med(slope),
                    nonadd_order_cond=med(na), D_coord=med([dd['D_coord'] for dd in d]))
a1 = all(abs(cells[n]['Cc'][i] - cells[n]['Cb'][i]) <= 0.03 for n in (64, 128) for i in range(3))
a2 = all(cells[n]['C0'] <= (0.02 if n <= 64 else 0.05) for n in cells)
a3 = cells[128]['nonadd_order_cond'] / cells[16]['nonadd_order_cond']
out = dict(A2E1=dict(hit=a1), A2E2=dict(hit=a2), A2E3=dict(hit=a3 >= 1.5, ratio=a3), cells=cells)
json.dump(out, open(f'{D_}/results/a2e_scores.json', 'w'), indent=1)
for k in ('A2E1', 'A2E2', 'A2E3'):
    print(k, out[k])
for n, c in cells.items():
    print(n, 'Cc', np.round(c['Cc'], 3), 'Cb', np.round(c['Cb'], 3), 'C0', round(c['C0'], 4), [round(x, 4) for x in c['C0_nets']],
          'cond slope', round(c['cond_slope'], 3), 'nonadd order (cond)', round(c['nonadd_order_cond'], 2), 'D_coord', round(c['D_coord'], 2))
