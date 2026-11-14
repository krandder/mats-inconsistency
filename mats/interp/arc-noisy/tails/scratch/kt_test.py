import sys, math, time
import numpy as np, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
from kprop_tail import kprop_tail
from scipy.special import log_ndtr
torch.set_num_threads(4)
# thresholds from the exploration GS runs (net 100, sigma 0.001, neurons 0-3, levels for 1e-4,1e-6,1e-8,1e-10,1e-12)
lev = {4: {1: [3.448,4.380,5.179,5.877,6.541], 2: [3.227,4.282,5.160,5.932,6.630], 3: [4.824,5.957,6.947,7.829,8.676]},
       8: {0: [2.875,3.685,4.455,5.150,5.846], 1: [3.945,4.794,5.556,6.268,6.949], 2: [0.386,1.056,1.676,2.206,2.669], 3: [6.047,7.485,8.721,9.870,10.956]},
       32: {0: [1.796,2.235,2.632,3.028,3.432], 1: [0.467,0.644,0.804,0.954,1.069], 2: [1.235,1.549,1.878,2.188,2.489], 3: [2.763,3.337,3.833,4.330,4.778]}}
CFGS = [(2, False, "simple", False), (3, True, "simple", False), (3, True, "augment", False)]
for L in (4, 8, 32):
    net = TNet(100, L, 0.001)
    ths = []
    for k in range(5):
        t = torch.full((256,), 1e3, dtype=torch.float64)
        for i, v in lev[L].items(): t[i] = v[k]
        ths.append(t)
    for (km, fac, kind, am) in CFGS:
        pass
        try:
            t0 = time.time()
            outs, fl, secs, sh = kprop_tail(net, ths, k_max=km, factor=fac, kind=kind, avg_metric=am)
        except Exception as e:
            print(L, km, kind, 'ERR', repr(e)[:200]); continue
        row = []
        for i in lev[L]:
            row.append(' '.join('%6.2f' % (math.log10(max(float(outs[k][i]), 1e-300)) if float(outs[k][i]) > 0 else float('nan')) for k in range(5)))
        print(f'L={L} K={km} f={fac} {kind}: flops {fl:.2e} (shared {sh:.2e}) {secs:.0f}s | ' + ' | '.join(row), flush=True)
