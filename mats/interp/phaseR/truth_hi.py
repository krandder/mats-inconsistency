"""Higher-effort truth for selected levels of a rare_*.json (fixed-level splitting, more particles and moves).
usage: python truth_hi.py results/rare_<net>.json k1,k2 [N] [moves]  -> adds lev['truth_hi']"""
import json, math, sys
import numpy as np
import torch
import rnet as R, rare_gs as GS

LOG2 = math.log(2)
p = sys.argv[1]; ks = [int(x) for x in sys.argv[2].split(',')]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 65536; mv = int(sys.argv[4]) if len(sys.argv) > 4 else 10
r = json.load(open(p))
net = R.RNet(r['d'], r['L'], r['act'], r['g'], r['sigma'], r['seed'], use_ln='_noLN' not in p)
gen = torch.Generator().manual_seed(4321)
levels = r['levels_curve']
for lev in r['levels']:
    if lev['k'] not in ks: continue
    t = lev['t']; lv = [x for x in levels if x < t - 1e-12]
    v = np.array([GS.splitting(net, t, N, gen, levels=lv, n_moves=mv)[0] for _ in range(8)])
    m = v.max(); w = np.exp(v - m); lp = float(np.log(w.mean()) + m)
    lev['truth_hi'] = dict(logp=lp, rel_se=float(w.std(ddof=1) / w.mean() / math.sqrt(8)), N=N, moves=mv, R=8)
    print(lev['k'], 'old', round(lev['truth']['logp'] / LOG2, 2), 'hi', round(lp / LOG2, 2), 'rel_se', round(lev['truth_hi']['rel_se'], 3), flush=True)
    json.dump(r, open(p, 'w'), indent=1)
