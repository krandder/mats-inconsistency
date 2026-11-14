"""RQMC and MC at small budgets (2^3..2^6 paths, R = 256 replicates), to compare with deterministic methods whose cost
is below the samplers' smallest budget (2^7). usage: python rqmc_small.py SEED SIGMA TRUTHJSON OUTJSON"""
import sys, json, math, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
import nd

torch.set_num_threads(1)
SEED, SIG, TJ, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
DT = nd.DT
T = json.load(open(TJ))
net = nd.Net('plain', SEED, T['n'], T['L'], SIG, r=T['r'])
n, L = net.n, net.L
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']; tse = 1.0 / T['paths']
g = torch.Generator().manual_seed(66_000_011 + SEED)
R = 256
res = dict(seed=SEED, sigma=SIG, est={})
for lg in (3, 4, 5, 6):
    N = 1 << lg
    e = []
    for rep in range(R):
        sob = torch.quasirandom.SobolEngine(n, scramble=True, seed=int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
        x = torch.special.ndtri(sob.draw(N, dtype=DT).clamp(1e-15, 1 - 1e-15))
        h = x
        for l in range(L):
            a = torch.relu(h @ net.W[l].T)
            h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if (l < L - 1 and SIG > 0) else a
        e.append(float((h.mean(0) - truth).pow(2).sum() / trv))
    res['est'][lg] = dict(RQMC=dict(nmse=float(np.mean(e)) - tse, se=float(np.std(e) / math.sqrt(R)), R=R))
    print(f"s{SEED} N=2^{lg}: RQMC {res['est'][lg]['RQMC']['nmse']:.3g} (MC 1/N {1/N:.3g})", flush=True)
json.dump(res, open(OUT, 'w'))
