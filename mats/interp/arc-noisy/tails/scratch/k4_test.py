import sys, math, time
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
from kprop_tail import kprop_tail
torch.set_num_threads(4)
net = TNet(100, 4, 0.001)
lev = {1: [3.448,4.380,5.179,5.877,6.541], 2: [3.227,4.282,5.160,5.932,6.630], 3: [4.824,5.957,6.947,7.829,8.676]}
ths = []
for k in range(5):
    t = torch.full((256,), 1e3, dtype=torch.float64)
    for i, v in lev.items(): t[i] = v[k]
    ths.append(t)
t0 = time.time()
outs, fl, secs, sh = kprop_tail(net, ths, k_max=4, factor=True, kind='simple', avg_metric=False)
print('K4f flops %.2e shared %.2e secs %.0f' % (fl, sh, secs))
for i in lev:
    print(i, ' '.join('%.2f' % (math.log10(float(outs[k][i])) if float(outs[k][i]) > 0 else float('nan')) for k in range(5)))
