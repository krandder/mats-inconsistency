"""Post hoc from Part A age runs: per covariance eigendirection k, the absolute effect E(Delta_j . v_k)^2/2 against age j;
median age of direction k's information (trapezoid over sampled ages), against the coordinator's n ln2/(2(k-1)) and the
radial (k = 1) prediction ln2/(2 |gamma_1|) = ln2 * 0.4 n. Also eigenvalues."""
import json, glob, math
import numpy as np
rows = []
for f in sorted(glob.glob('runs/age_n*_s*.json'), key=lambda x: (int(x.split('_n')[1].split('_')[0]), x)):
    r = json.load(open(f)); n = r['n']
    js = np.array([p['j'] for p in r['profile']], float)
    tot = np.array([p['T_h'] for p in r['profile']]) * r['totV']
    sh = np.array([p['eig_share'] for p in r['profile']])          # (ages, 16)
    eff = sh * tot[:, None]
    out = []
    for k in (1, 2, 3, 5, 9, 16):
        if k > sh.shape[1]:
            continue
        e = eff[:, k - 1]
        c = np.concatenate([[0], np.cumsum(0.5 * (e[1:] + e[:-1]) * np.diff(js))])
        if c[-1] <= 0:
            continue
        jm = float(np.interp(0.5, c / c[-1], js))
        pred = 0.4 * n * math.log(2) if k == 1 else n * math.log(2) / (2 * (k - 1))
        out.append((k, round(jm, 1), round(pred, 1)))
    rows.append((n, r['seed'], out))
    print(n, r['seed'], 'k: median age (meas, coord) ->', out)
json.dump(rows, open('results/age_by_direction.json', 'w'))
