"""E6 post-hoc diagnostic: per-source DAG (D1, real weights) deviation R^2 for every net (is the net an own-source solution?).
K = 64 on the first 2,000 test episodes, against the net's own replicate-1 deviation (from share_test.npz).
usage: python qshare_dag.py -> results/share_dag.json
"""
import json, os
import numpy as np
import torch
import nnoise as N
import devpred as DP
import dag as DG
import qw
import qshare as SH

torch.set_num_threads(1)
X, v, Tt = SH.test_set(); n = 2000; T2 = {k: (a[:n] if k != 'comp_vars' else a) for k, a in Tt.items()}
out = {}
for nm, _ in SH.MAIN + SH.CROSS:
    model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False)
    z = np.load(os.path.join('runs', nm, 'share_test.npz'))
    sur = DG.make_sur(model, DG.masks('D1', meta['depth']))
    h, _, _ = DP.marginal(model, X[:n], v[:n], meta['sigma'], sur, 64, 11); dv = qw.deviations_t(h, T2)
    out[nm] = {k: DP.r2(z['r1_' + k][:n], dv[k]) for k in SH.FEATS}
    print(nm, {k: round(x, 3) for k, x in out[nm].items()}, flush=True)
    json.dump(out, open('results/share_dag.json', 'w'), indent=1)
print('done', flush=True)
