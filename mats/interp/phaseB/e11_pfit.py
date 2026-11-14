"""E11.3 by SS tercile (post hoc, PREREG-B amendment 18 addendum 2). usage: python e11_pfit.py -> results/e11_pfit.json"""
import json, os
import numpy as np
import nnoise  # noqa: F401
import world as W
import qweights as QWT
import qmech as QM

PG = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
z = np.load('results/e11_pairs.npz'); src, base = z['src'], z['base']; m = len(src); rows = np.arange(m); K = len(z['kappas'])
SSs = z['SS'][base, src]; tert = np.digitize(SSs, np.quantile(SSs, [1 / 3, 2 / 3]))
rng = np.random.default_rng(1); e = np.abs(rng.standard_normal((1_000_000, 4)))
cp = {p: 4.0 / float(((e ** p).sum(1) ** (2 / p)).mean()) for p in PG}
pseudo = {}
for p in PG:
    Wp = []
    for k in range(K):
        d = {kk[len(f'd{k}_'):]: z[kk].copy() for kk in z.files if kk.startswith(f'd{k}_')}
        r = d['o'][rows, src, :W.QL] - d['y'][rows, :W.QL]; Tp = cp[p] * (np.abs(r) ** p).sum(1) ** (2 / p)
        d['o'][rows, src, :W.QL] = d['y'][rows, :W.QL] + np.sqrt(Tp / 4)[:, None]; Wp.append(QWT.bayes_weights(d)[rows, src])
    pseudo[p] = np.stack([Wp[k] - Wp[0] for k in range(1, K)])
P = {nm: np.load(os.path.join('runs', nm, 'e11_pairs.npz')) for nm in QM.ALL}
out = {}
for g, nets in (('A', QM.A_NETS), ('B', QM.B_NETS)):
    obs = np.stack([np.mean([P[nm][f'ws{k}'] - P[nm]['ws0'] for nm in nets], 0) for k in range(1, K)]); out[g] = {}
    for tt in range(3):
        sel = tert == tt; o_ = obs[:, sel]; sst = ((o_ - o_.mean()) ** 2).sum()
        r2 = {p: float(1 - ((o_ - pseudo[p][:, sel]) ** 2).sum() / sst) for p in PG}
        best = max(PG, key=lambda p: r2[p])
        # binned version: mean over rows per kappa level (3 numbers) vs pseudo means
        ob_m = o_.mean(1); ps_m = {p: pseudo[p][:, sel].mean(1) for p in PG}
        bestm = min(PG, key=lambda p: float(((ob_m - ps_m[p]) ** 2).sum()))
        out[g][f'SS{tt + 1}'] = dict(best_p_rows=best, r2_rows=r2[best], r2_by_p={str(p): v for p, v in r2.items()}, obs_mean_by_kappa=[float(x) for x in ob_m],
                                     best_p_means=bestm, pseudo_mean_by_kappa_at_best=[float(x) for x in ps_m[bestm]])
        print(g, f'SS tercile {tt + 1}', 'best p (rows) %.2f R2 %.3f' % (best, r2[best]), '| means: obs', [round(float(x), 4) for x in ob_m], 'best p (means) %.2f pseudo' % bestm, [round(float(x), 4) for x in ps_m[bestm]], flush=True)
json.dump(out, open('results/e11_pfit.json', 'w'), indent=1)
