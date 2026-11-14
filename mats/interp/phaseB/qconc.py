"""E10 residual-concentration follow-up (post hoc, descriptive; PREREG-B amendment 17 addendum 3).
kappa_s = largest squared labelled error / sum of squares; partial correlation of dbar_s (path-averaged weight error, qig) with
kappa_s given log SS_s and p_s (linear), per net and for the group average; mean dbar by kappa tercile within log-SS terciles.
First 4,000 test episodes. usage: python qconc.py -> results/qconc.json (prints a summary; results/qconc.txt holds the run log)
"""
import numpy as np, json, nnoise, world as W, qbayes as QB, qmech as QM  # noqa: F401
t8 = QM.test8000(); d = {k: v[:4000] for k, v in t8['d'].items()}
res2 = (d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]) ** 2; SS = res2.sum(2); kap = res2.max(2) / SS; lss = np.log(SS + 1e-3)
p = QB.bayes_features(d)[:, 20:23].astype(np.float64); p /= p.sum(1, keepdims=True)
Z = np.c_[np.ones(12000), lss.ravel(), p.ravel()]
resid = lambda y: y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
kr = resid(kap.ravel()); out = {}
for g, nets in (('A', QM.A_NETS), ('B', QM.B_NETS)):
    per = {}
    for nm in nets:
        db = np.load(f'runs/{nm}/qig_J8.npz')['test_dbar'].ravel(); per[nm] = float(np.corrcoef(resid(db), kr)[0, 1])
    avg = np.mean([np.load(f'runs/{nm}/qig_J8.npz')['test_dbar'] for nm in nets], 0)
    pc = float(np.corrcoef(resid(avg.ravel()), kr)[0, 1])
    lt = np.digitize(lss.ravel(), np.quantile(lss, [1/3, 2/3])); kt = np.digitize(kap.ravel(), np.quantile(kap, [1/3, 2/3]))
    cells = {f'SS{i+1}': [float(avg.ravel()[(lt == i) & (kt == j)].mean()) for j in range(3)] for i in range(3)}
    out[g] = dict(partial_corr_group_avg=pc, per_net=per, n_neg=int(sum(v < 0 for v in per.values())), cells_kappa_tercile_by_SS_tercile=cells)
    print(g, 'partial corr (group avg) %.3f' % pc, 'per net', {k: round(v, 3) for k, v in per.items()}, 'neg', out[g]['n_neg'])
    for k, v in cells.items(): print('  ', k, 'mean dbar by kappa tercile low/mid/high', [round(x, 4) for x in v])
json.dump(out, open('results/qconc.json', 'w'), indent=1)
