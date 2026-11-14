"""Collect analysis_*.json into one table (cut-rank lane) with astra's truncation upper bound.

Upper bound (astra A3, confirmed): r(eps) <= S(n, m*), S(n, m) = dim of polynomials of degree <= m on S^{n-1},
m* = smallest m with (n-1)/((m+1)(m+n-1)) <= eps^2 Vbar_norm, Vbar_norm = Vbar / E a^2 (E a^2 = Vbar + E M^2).
usage: python3 scratch/tabulate.py [names...]   (default: every runs/analysis_cut_*.json)
"""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
EPS = ('0.3', '0.1', '0.03', '0.01')


def S_dim(n, m):
    return math.comb(n + m - 1, n - 1) + (math.comb(n + m - 2, n - 1) if m >= 1 else 0)


def ub(n, eps, vn):
    m = 0
    while (n - 1) / ((m + 1) * (m + n - 1)) > eps * eps * vn:
        m += 1
    return m, S_dim(n, m)


files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, '..', 'runs', 'analysis_cut_*.json')))
rows = []
for f in files:
    if not f.endswith('.json'):
        f = os.path.join(HERE, '..', 'runs', f'analysis_{f}.json')
    d = json.load(open(f))
    nm = d['name']
    parts = nm.split('_')
    n = int(parts[1][1:]); c = int(parts[2][1:]); Lp = int(parts[3][1:])
    vn = 1.0 / (1.0 + d['EM2_rel'])
    E = d['ensemble']
    fb = d['fixed_back']
    r3 = [x['R3']['r_in']['0.1'] for x in fb]; r2 = [x['R2']['r_in']['0.1'] for x in fb]
    r3c = [x['R3']['r_in']['0.03'] for x in fb]
    ubs = {e: ub(n, float(e), vn) for e in EPS}
    sat = d['saturation_in']
    rows.append((n, c, Lp, nm, d, vn, ubs, r2, r3, r3c, sat))
rows.sort(key=lambda r: (r[3].endswith('ctrl'), r[0], r[1], r[2]))
print('| config (n,c,L\') | F | k* (0.01) | angle | gate dis. | PR | EM2/V | r_in at eps 0.3/0.1/0.03/0.01 | r_cf | r_in(0.1) at F\'=64/128/256/512 | r_in(0.03) same | probe s=16/64 | frac nets err>0.01V | moment resid K=0..(ridge) | fixed back R2 / R3 r_in(0.1), R3 r_in(0.03) | upper bound m*, S at eps=0.1 |')
print('|' + '---|' * 17)
for n, c, Lp, nm, d, vn, ubs, r2, r3, r3c, sat in rows:
    E = d['ensemble']
    print(f"| ({n},{c},{Lp}){' ctrl' if nm.endswith('ctrl') else ''} | {d['F']} | {d['diag']['kstar01']:.1f} | {d['diag']['angle']:.3f} | "
          f"{d['diag']['gate_dis']:.3f} | {d['diag']['PR']:.1f} | {d['EM2_rel']:.2f} | "
          f"{'/'.join(str(E['r_in'][e]) for e in EPS)} | {'/'.join(str(E['r_cf'][e]) for e in EPS)} | "
          f"{'/'.join(str(sat[k]['0.1']) for k in sorted(sat, key=int))} | {'/'.join(str(sat[k]['0.03']) for k in sorted(sat, key=int))} | "
          f"{d['probe_rel']['16']:.3f}/{d['probe_rel']['64']:.3f} | {d['per_net']['frac_err_gt_0p01V']:.2f} | "
          f"{'/'.join(f'{v['resid_rel']:.3f}' for v in d['moments_ridge_posthoc'].values())} | "
          f"{int(np.median(r2))} / {int(np.median(r3))}, {int(np.median(r3c))} | m*={ubs['0.1'][0]}, S={ubs['0.1'][1]:.2e} |")
