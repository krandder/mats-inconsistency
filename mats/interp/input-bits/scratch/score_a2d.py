"""Score Amendment 2D (A2D.1-A2D.3) -> results/a2d_scores.json. Rules as registered (hash b13226fb)."""
import json
import numpy as np

D_ = '/home/agent/workspaces/mats/work/input-bits'
S = (1000, 1001, 4000, 4001)
cells = {}
for n in (16, 32, 64, 128):
    rs = [json.load(open(f'{D_}/runs/a2d_n{n}_s{s}.json')) for s in S]
    s1 = [json.load(open(f'{D_}/runs/p2noise_n{n}_s{s}.json'))['S1_win'] for s in S]
    cells[n] = dict(D_block=float(np.median([r['D_block'] for r in rs])), D_coord=float(np.median([r['D_coord'] for r in rs])),
                    nonadd_order=float(np.median([(r['D_coord'] - x) / (1 - x) for r, x in zip(rs, s1)])),
                    D_block_nets=[r['D_block'] for r in rs], D_coord_nets=[r['D_coord'] for r in rs],
                    D_coord_se=[r['D_coord_se'] for r in rs], d_layers=rs[0]['d_layers'])
a1 = cells[128]['D_block'] / cells[16]['D_block']
a2 = all(c['D_coord'] >= c['D_block'] for c in cells.values()) and cells[128]['D_coord'] / cells[16]['D_coord'] >= 1.3
a3 = cells[128]['nonadd_order'] / cells[16]['nonadd_order']
out = dict(A2D1=dict(hit=a1 >= 1.3, ratio=a1), A2D2=dict(hit=bool(a2), ratio=cells[128]['D_coord'] / cells[16]['D_coord']),
           A2D3=dict(hit=a3 >= 1.5, ratio=a3), cells=cells)
json.dump(out, open(f'{D_}/results/a2d_scores.json', 'w'), indent=1)
for k in ('A2D1', 'A2D2', 'A2D3'):
    print(k, out[k])
for n, c in cells.items():
    print(n, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items() if k in ('D_block', 'D_coord', 'nonadd_order', 'd_layers')})
