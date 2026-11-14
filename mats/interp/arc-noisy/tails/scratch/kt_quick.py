import sys, math, time
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
from kprop_tail import kprop_tail
torch.set_num_threads(4)
net = TNet(100, 4, 0.001)
t = torch.full((256,), 3.0, dtype=torch.float64)
for (km, fac, kind, am) in [(2, False, 'simple', True), (2, False, 'simple', False), (3, True, 'simple', True), (3, True, 'simple', False)]:
    t0 = time.time()
    outs, fl, secs, sh = kprop_tail(net, [t], k_max=km, factor=fac, kind=kind, avg_metric=am)
    print(km, fac, kind, am, 'flops %.2e secs %.1f' % (fl, secs), 'p[0:4]', outs[0][:4].numpy(), flush=True)
