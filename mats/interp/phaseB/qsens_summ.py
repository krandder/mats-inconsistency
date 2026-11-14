"""E10.4 summary: correlation of reading-sensitivity sizes and the downstream-read own gain with source reliability.
usage: python qsens_summ.py -> results/qsens_summary.json
"""
import json, os
import numpy as np
import nnoise  # noqa: F401
import world as W
import qbayes as QB
import qmech as QM

t8 = QM.test8000(); d = {k: v[:4000] for k, v in t8['d'].items()}
p = QB.bayes_features(d)[:, 20:23].astype(np.float64); p /= p.sum(1, keepdims=True)
res = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]; lss = np.log((res ** 2).sum(2) + 1e-3)
out = dict(per_net={}, by_group={})
c = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
for nm in QM.ALL:
    if not os.path.exists(os.path.join('runs', nm, 'qsens.npz')): continue
    z = np.load(os.path.join('runs', nm, 'qsens.npz')); own = np.load(os.path.join('runs', nm, 'qe9b.npz'))['own']
    q = np.load(os.path.join('runs', nm, 'qe9.npz'))['masses_qq'].sum(1)          # unnormalised summed QQ mass per source (n, 3)
    G = own / q
    out['per_net'][nm] = dict(group=QM.GROUP[nm], x1_p=c(z['x1'], p), x1_lss=c(z['x1'], lss), v_p=c(z['v'], p), v_lss=c(z['v'], lss),
                              gain_p=c(G, p), gain_lss=c(G, lss), x1_mean=float(z['x1'].mean()), v_mean=float(z['v'].mean()))
for g, nets in (('A', QM.A_NETS), ('B', QM.B_NETS)):
    nets = [nm for nm in nets if nm in out['per_net']]
    if not nets: continue
    out['by_group'][g] = {k: dict(mean=float(np.mean([out['per_net'][nm][k] for nm in nets])), lo=float(min(out['per_net'][nm][k] for nm in nets)),
                                  hi=float(max(out['per_net'][nm][k] for nm in nets)), n_pos=int(sum(out['per_net'][nm][k] > 0 for nm in nets)), n=len(nets))
                          for k in ('x1_p', 'x1_lss', 'v_p', 'v_lss', 'gain_p', 'gain_lss')}
    print(g, {k: (round(v['mean'], 3), v['n_pos']) for k, v in out['by_group'][g].items()}, flush=True)
json.dump(out, open('results/qsens_summary.json', 'w'), indent=1)
