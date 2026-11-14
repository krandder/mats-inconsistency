"""Truth cross-check statistics for the report: (b) splitting vs Monte Carlo (2^23 paths) at 1e-4 and 1e-6, (c) splitting
vs the independent rho = 0.3 splitting estimator, and provisional counts, per (sigma, L).
python tails/truthcheck.py TRUTHDIR MCDIR"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_truth

TR, MC = sys.argv[1], sys.argv[2]
for sigma in (0.001, 0.003, 0.01):
    for L in (4, 8, 32):
        zb = {0: [], 1: []}; lb = {0: [], 1: []}; zc = []; nprov = np.zeros(5, int); nev = 0
        for s in range(20):
            f = os.path.join(TR, f'truth_{s}_s{sigma:g}.json')
            if not os.path.exists(f):
                continue
            T = load_truth(f)[L]
            m = json.load(open(os.path.join(MC, f'mc_xc_{s}_s{sigma:g}.json'))); N = m['N']
            cnt = np.array(m['reps'][0]['counts'][str(L)])
            for i in range(8):
                for k in (0, 1):
                    if cnt[i, k] > 0:
                        pm = cnt[i, k] / N
                        d = T['lp'][i, k] - math.log(pm)
                        zb[k].append(d / math.sqrt(T['rse'][i, k] ** 2 + (1 - pm) / (pm * N))); lb[k].append(d)
            d = T['xc'] - T['lp'][:2]; se = np.sqrt(T['rse'][:2] ** 2 + T['xc_rse'] ** 2)
            zc.extend((d / se).ravel().tolist())
            prov = ~(T['rse'] <= 0.25) | ~(np.abs(T['half']) <= 1)
            nprov += prov.sum(0); nev += 8
        if not nev:
            continue
        zc = np.array(zc)
        print(f'sigma {sigma} L={L:2d}: (b) 1e-4: mean log(GS/MC) {np.mean(lb[0]):+.3f}, z sd {np.std(zb[0]):.2f}, |z|>3 {int(np.sum(np.abs(zb[0]) > 3))}/{len(zb[0])}; '
              f'1e-6: mean {np.mean(lb[1]):+.3f}, z sd {np.std(zb[1]):.2f}, |z|>3 {int(np.sum(np.abs(zb[1]) > 3))}/{len(zb[1])} | '
              f'(c) z sd {np.std(zc[np.isfinite(zc)]):.2f}, |z|>3 {int(np.sum(~(np.abs(zc) <= 3)))}/{zc.size} | rse/halves provisional by k {nprov.tolist()} of {nev}')
