"""Depth-scaling table for the residual Gaussian closure (runs/resdepth_*.json) -> markdown rows + summary numbers."""
import json, glob, os, re
import numpy as np

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
rows = []; allratio = []; summ = {}
for n in (16, 8):
    for L in (16, 64, 256, 1024):
        f = os.path.join(W, 'runs', f'resdepth_n{n}_L{L}.json')
        if not os.path.exists(f):
            continue
        d = json.load(open(f))['res']
        st = [v['k2_st']['nmse'] for v in d.values()]
        cost = np.mean([v['k2_st']['paths'] for v in d.values()])
        best = [min(v['k2_st']['at_budget']['RQMC'], v['k2_st']['at_budget']['RQMC-reg']) for v in d.values()]
        mc = np.mean([v['k2_st']['at_budget']['mc'] for v in d.values()])
        ex = [v['k2']['nmse'] for v in d.values()]
        excost = np.mean([v['k2']['paths'] for v in d.values()])
        ratios = [b / s for b, s in zip(best, st)]
        allratio += ratios
        summ[(n, L)] = (np.median(st), min(st), max(st))
        rows.append(f"| n = {n}, L = {L} ({len(st)} nets) | {', '.join(f'{x:.1e}' for x in st)} | {cost:.0f} | "
                    f"{', '.join(f'{x:.1e}' for x in best)} ({min(ratios):.0f}-{max(ratios):.0f}x worse) | {mc:.3f} | "
                    f"{', '.join(f'{x:.1e}' for x in ex)}, {excost:.0f} |")
print('\n'.join(rows))
print()
print('ratio best-sampler / K2-Stein over all nets: min %.0f max %.0f median %.0f' % (min(allratio), max(allratio), np.median(allratio)))
for n in (16, 8):
    if (n, 64) in summ:
        for L in (16, 256, 1024):
            if (n, L) in summ:
                print(f'n={n}: median K2-Stein nMSE L={L} / L=64 = {summ[(n, L)][0] / summ[(n, 64)][0]:.2f}')
allst = [v for k, v in summ.items()]
print('K2-Stein nMSE range over all:', min(x[1] for x in allst), max(x[2] for x in allst))
