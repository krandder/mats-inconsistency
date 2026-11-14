"""Noise trend table: for each depth and sigma, median nMSE of kprop K=2-augmented and factorized K=3 (and K=3-augmented at
depth 8), the Kerdock-MUB sampler at 6.8e10 and 2.72e11 FLOPs (with nMSE x N), and the per-net median gain of antithetic
noise (C6) over plain Kerdock at 2.72e11 with its bootstrap CI. Evaluation nets 0-23. usage: python trend.py"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as A

bi68 = list(A.BUDGETS).index(6.8e10); bi272 = list(A.BUDGETS).index(2.72e11)
print('| depth | sigma | nets | K=2a | K=3f | K=3fa | Kerdock at 6.8e10 | Kerdock at 2.72e11 (nMSE x N) | C6 gain at 2.72e11 [95% CI] |')
print('|---|---|---|---|---|---|---|---|---|')
for L in (8, 32):
    for sigma in A.SIGMAS:
        nets = A.collect(L, sigma)
        if len(nets) < 3:
            continue
        def med(key, bi):
            v = [e[key][bi] for e in nets.values() if key in e and bi in e[key]]
            return (np.median(v), len(v)) if len(v) >= 3 else (np.nan, 0)
        k2a = med('kprop-k2a', len(A.BUDGETS) - 1)[0]; k3f = med('kprop-k3f', len(A.BUDGETS) - 1)[0]
        k3fa = med('kprop-k3fa', len(A.BUDGETS) - 1)[0]
        m68 = med('mub', bi68)[0]; m272 = med('mub', bi272)[0]
        # nMSE x N at 2.72e11 from the v2 checkpoint actually used
        nn = []
        for s in nets:
            p = os.path.join(HERE, 'samp_v2', f'samp_{s}_s{sigma:g}_L{L}.npz')
            if os.path.exists(p):
                d = np.load(p)
                if 'costs_mub' in d.files:
                    j = int(np.searchsorted(d['costs_mub'], 2.72e11, side='right') - 1); nn.append(d['N_mub'][j])
        nxn = m272 * np.median(nn) if nn else np.nan
        r = [e['mub'][bi272] / e['mubanti'][bi272] for e in nets.values()
             if 'mub' in e and 'mubanti' in e and bi272 in e['mub'] and bi272 in e['mubanti']]
        c6 = A.fmt(A.boot(r, np.median), '.2f') if len(r) >= 3 else '-'
        print(f'| {L} | {sigma:g} | {len(nets)} | {k2a:.2e} | {k3f:.2e} | {k3fa:.2e} | {m68:.2e} | {m272:.2e} ({nxn:.2f}) | {c6} |')
