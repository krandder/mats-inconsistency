"""wm2 audit, phase 2: compute per board of the final T1 runs including early-stopping evaluation (all phases).
  PYTHONDONTWRITEBYTECODE=1 python3 audit/final_compute.py"""
import glob, json
import numpy as np
fb = 2 * 25 * (128 * 3 * 128 + 128 * 128 + 2 * 128 * 256) * 4 + 2 * 2 * 25 * 25 * 128 * 4 + 2 * 25 * 128
R = '/tmp/wm2/runs/F/t1/'
wm = {}
for p in glob.glob(R + 'F-WMr/*.json'):
    w = json.load(open(p)); wm[(w['n'], w['seed'])] = w
rows = {}
for tag in ('F-B-k0', 'F-B-rbtd8', 'F-WMr-D-rbtd8'):
    for p in glob.glob(R + tag + '/*.json'):
        r = json.load(open(p))
        for k in [k for k in r if 'test' in k.lower()]: del r[k]   # compute fields only
        ev = len(r['history']) * r['cfg']['es_boards'] * 4 * fb
        tr = r.get('flops_fc_total', r['flops_total']) + r.get('flops_dreams_total', 0.0)
        best = r['flops_per_board_at_best']
        if tag == 'F-WMr-D-rbtd8':
            w = wm[(r['n'], r['seed'])]
            ev += (len(w['history']) * 3000 * 25 + len(w.get('history_ev', [])) * 3000 * 6) * fb
            tr += w['flops_total']
        rows.setdefault(tag, {}).setdefault(r['n'], []).append((best, tr / r['n'], (tr + ev) / r['n']))
for n in (100, 300, 1000, 3000, 10000):
    m = {t: np.mean(rows[t][n], 0) for t in rows}
    c, b, k = m['F-WMr-D-rbtd8'], m['F-B-rbtd8'], m['F-B-k0']
    print(f"N={n}: champion / rbtd8 per board: to best {c[0] / b[0]:.1f}x, training total {c[1] / b[1]:.1f}x, all-in incl. evaluation {c[2] / b[2]:.1f}x; "
          f"champion / k0: {c[0] / k[0]:.0f}x / {c[1] / k[1]:.0f}x / {c[2] / k[2]:.0f}x")
