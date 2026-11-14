"""Summarize scratch/diag10.py over nets: per-net table, layer-by-layer gain, attribution, rank correlations of the
PCA3 gain (G / PCA3 at layer 32) with the net properties."""
import glob, json, math, os
import numpy as np
from scipy.stats import spearmanr

R = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'diag10', 'diag10_*.json'))):
    R.append(json.load(open(f)))
print(f'{len(R)} nets')
hdr = 'net  G@32     PCA3@32  gain  PCA3NB   PCA4     EMP     | PR    top   size  cos  near ent_near ent_all kurt_in kurt_in_near |t|med'
print(hdr)
rows = []
for r in R:
    f = r['fam']; p = r['props']
    g = f['G']['err'][-1]; c = f['PCA3']['err'][-1]
    rows.append(dict(net=r['seed'], gain=g / c, G=g, PCA3=c, **p))
    print(f"{r['seed']}  {g:.2e} {c:.2e} {g / c:5.1f} {f['PCA3NB']['err'][-1]:.2e} {f['PCA4']['err'][-1]:.2e} "
          f"{f['EMP']['err'][-1]:.1e} | {p['pr']:5.2f} {p['top_share']:.3f} {p['size_share']:.3f} {p['cos_top_mean']:.2f} "
          f"{p['near_count']:4d} {p['near_entropy']:6.1f} {p['gate_entropy_all']:6.1f} {p['inbin_kurt_all']:.3f} "
          f"{p['inbin_kurt_near']:.3f} {p['t_abs_median']:.2f}")
print('\nlayer-by-layer (l = 24..32): G / PCA3 / PCA3NB / PCA4 / EMP')
for r in R:
    f = r['fam']
    for name in ('G', 'PCA3', 'PCA3NB', 'PCA4', 'EMP'):
        print(r['seed'], f'{name:6s}', ' '.join(f'{v:.1e}' for v in f[name]['err']))
    print(r['seed'], 'gain  ', ' '.join(f'{a / b:6.1f}' for a, b in zip(f['G']['err'], f['PCA3']['err'])))
print('\nattribution: share of squared error in neurons with |t| < 1.5 (share of neurons) at layers 25 and 32')
for r in R:
    f = r['fam']
    print(r['seed'], ' '.join(f"{name}: " + ', '.join(f"{l}: {f[name]['attr'][str(l)]['frac_err']:.2f} ({f[name]['attr'][str(l)]['frac_neurons']:.2f})"
                                                     for l in (25, 32)) for name in ('G', 'PCA3')))
if len(rows) >= 5:
    print('\nSpearman correlation of log gain (G/PCA3 at 32) with properties:')
    lg = [math.log(x['gain']) for x in rows]
    for k in ('pr', 'top_share', 'top3_share', 'size_share', 'cos_top_mean', 'near_count', 'near_entropy',
              'gate_entropy_all', 'inbin_kurt_all', 'inbin_kurt_near', 'inbin_skew_near', 't_abs_median', 'G'):
        rho, pv = spearmanr(lg, [x[k] for x in rows])
        print(f'  {k:18s} rho {rho:+.2f}  p {pv:.3f}')
