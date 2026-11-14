"""Score PREREG Part I (PI1-PI8) from runs/partI_*.json; fit a 3-component dimension distribution to the eps-curves."""
import glob, json, math, os
import numpy as np
from scipy.optimize import nnls

D = '/home/agent/workspaces/mats/work/stationary'
ld = lambda p: json.load(open(p)) if os.path.exists(p) else None
NS = [16, 64, 256]; SEEDS = [1000, 1001]; EPS = [0.01, 0.03, 0.1, 0.3]
ES_PARTA = {16: 2.08, 64: 2.55, 256: 3.94}      # Part A Efron-Stein sums (median over the 2 age-run nets)


def fit3(G):
    """G(eps) = p1 eps + p2 (1-(1-eps)^2) + pH (1-(1-eps)^H), p >= 0, sum = 1; scan H, NNLS with the sum as a soft row."""
    best = None
    for H in range(3, 201):
        A = np.array([[e, 1 - (1 - e) ** 2, 1 - (1 - e) ** H] for e in EPS] + [[10.0, 10.0, 10.0]])
        b = np.array([G[str(e)] for e in EPS] + [10.0])
        p, rn = nnls(A, b)
        if best is None or rn < best[0]:
            best = (rn, H, p)
    rn, H, p = best
    return dict(p1=float(p[0]), p2=float(p[1]), pH=float(p[2]), H=H, resid=float(rn), mean_dim=float(p[0] + 2 * p[1] + H * p[2]))


rows = []
for n in NS:
    for s in SEEDS:
        C = ld(f'{D}/runs/partI_curves_S_n{n}_s{s}.json'); CC = ld(f'{D}/runs/partI_curves_clamp_n{n}_s{s}.json')
        PR = ld(f'{D}/runs/partI_pairs_S_n{n}_s{s}.json'); TR = ld(f'{D}/runs/partI_trunc_S_n{n}_s{s}.json')
        r = dict(n=n, seed=s)
        if C:
            for k, v in C['curves'].items():
                r[f'D_{k}'] = v['mean_dim']; r[f'D01_{k}'] = v['0.1'] / 0.1
                r[f'Trad_{k}'] = v['radial_all']; r[f'Tang_{k}'] = v['angular_all']
                r[f'fit_{k}'] = fit3(v)
        if CC:
            for k, v in CC['curves'].items():
                r[f'Dclamp_{k}'] = v['mean_dim']; r[f'D01clamp_{k}'] = v['0.1'] / 0.1
                r[f'fitclamp_{k}'] = fit3(v)
        if PR:
            r['pairs'] = PR['pairs']; r['totals'] = PR['totals']
        if TR:
            ds = [x['d'] for x in TR['rows']]
            Ca = [x['C_a'] for x in TR['rows']]
            r['trunc99_a'] = next((d for d, c in zip(ds, Ca) if c >= 0.99), None)
            r['SD_a_at_n'] = next((x['SD_a'] for x in TR['rows'] if x['d'] == n), None)
            r['trunc_rows'] = TR['rows']
        rows.append(r)
K = {n: ld(f'{D}/runs/partI_kesten_n{n}.json') for n in NS}
med = lambda xs: float(np.median([x for x in xs if x is not None])) if [x for x in xs if x is not None] else None
by = lambda n: [r for r in rows if r['n'] == n]
out = {}
out['PI1'] = dict(D_h={n: [round(r.get('D_h', float('nan')), 2) for r in by(n)] for n in NS}, ES=ES_PARTA,
                  hit=all(1 / 1.5 <= med([r.get('D_h') for r in by(n)]) / ES_PARTA[n] <= 1.5 for n in NS if by(n) and med([r.get('D_h') for r in by(n)])))
out['PI2'] = dict(hit=all(r[f'D01_{k}'] < r[f'D_{k}'] / 2 for r in rows for k in ('h', 'a') if f'D_{k}' in r),
                  D01_vs_D={f"{r['n']}_{r['seed']}": {k: (round(r[f'D01_{k}'], 2), round(r[f'D_{k}'], 2)) for k in ('h', 'a') if f'D_{k}' in r} for r in rows})
out['PI3'] = dict(hit=all(r[f'Tang_{k}'] >= 0.8 and r[f'Trad_{k}'] <= 0.3 for r in rows for k in ('h', 'a') if f'Tang_{k}' in r),
                  T={f"{r['n']}_{r['seed']}": {k: (round(r[f'Trad_{k}'], 3), round(r[f'Tang_{k}'], 3)) for k in ('h', 'logR', 'u', 'a', 'g') if f'Trad_{k}' in r} for r in rows})
kD = {n: (K[n]['curves']['logQ']['mean_dim'] if K[n] else None) for n in NS}
pi4i = all(v is not None and 1.0 <= v <= 1.6 for v in kD.values())
pi4ii = all(med([r.get('D_logR') for r in by(n)]) is not None and kD[n] is not None and med([r.get('D_logR') for r in by(n)]) > kD[n] for n in NS)
pi4iii = all(r.get('Dclamp_u') is not None and r.get('D_u') and r['Dclamp_u'] >= 0.7 * r['D_u'] for r in rows)
out['PI4'] = dict(i=pi4i, ii=pi4ii, iii=pi4iii, kesten_D_logQ=kD,
                  D_logR={n: [round(r.get('D_logR', float('nan')), 2) for r in by(n)] for n in NS},
                  D_u={n: [round(r.get('D_u', float('nan')), 2) for r in by(n)] for n in NS},
                  Dclamp_u={n: [round(r.get('Dclamp_u', float('nan')), 2) for r in by(n)] for n in NS})
g = lambda key: (med([r.get(key) for r in by(256)]) or float('nan')) / (med([r.get(key) for r in by(16)]) or float('nan'))
out['PI5'] = dict(h_growth=g('D_h'), kesten_growth=(kD[256] / kD[16]) if kD[256] and kD[16] else None, clamp_u_growth=g('Dclamp_u'),
                  hit=(g('D_h') >= 1.2 and kD[256] and kD[16] and 0.8 <= kD[256] / kD[16] <= 1.25 and g('Dclamp_u') >= 1.2))
pi6a = True; pi6b = True
for r in rows:
    if 'pairs' not in r:
        continue
    for p in r['pairs']:
        if p['j'] <= 8 and p['g'] < 2 * p['a']:
            pi6a = False
    for j0 in (0, 2):
        seq = [p['h'] for p in r['pairs'] if p['j'] == j0]
        if any(seq[i + 1] > seq[i] for i in range(len(seq) - 1)):
            pi6b = False
out['PI6'] = dict(gates_vs_a=pi6a, h_monotone_in_lag=pi6b)
out['PI7'] = dict(hit=all(r.get('trunc99_a') is not None and n / 2 <= r['trunc99_a'] <= 4 * n for n in NS for r in by(n)) and
                  all(r.get('SD_a_at_n') is not None and r['SD_a_at_n'] <= 0.1 for n in (64, 256) for r in by(n)),
                  trunc99_a={n: [r.get('trunc99_a') for r in by(n)] for n in NS}, SD_a_at_n={n: [round(r['SD_a_at_n'], 3) if r.get('SD_a_at_n') is not None else None for r in by(n)] for n in NS})
out['PI8'] = dict(hit=all(r['D01_a'] <= 2 for r in rows if 'D01_a' in r), D01_a={n: [round(r.get('D01_a', float('nan')), 2) for r in by(n)] for n in NS})
out['fits'] = {f"{r['n']}_{r['seed']}": {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in r[f'fit_{k}'].items()}
                                          for k in ('h', 'logR', 'u', 'a', 'g') if f'fit_{k}' in r} for r in rows}
out['fits_clamp'] = {f"{r['n']}_{r['seed']}": {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in r[f'fitclamp_{k}'].items()}
                                                for k in ('h', 'u', 'a') if f'fitclamp_{k}' in r} for r in rows}
out['kesten'] = {n: (K[n]['curves'] if K[n] else None) for n in NS}
json.dump(dict(out=out, rows=rows), open(f'{D}/results/partI_scores.json', 'w'), indent=1, default=float)
for k, v in out.items():
    print(k, json.dumps(v, default=float)[:900])
