import sys, math, time, json
import torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
from kprop_tail import kprop_moments
torch.set_num_threads(4)
out = {}
for seed in (100, 101):
    for L in (4, 8, 32):
        for (km, fac, kind) in [(2, False, 'simple'), (3, True, 'simple')]:
            net = TNet(seed, L, 0.001)
            t0 = time.time()
            M, fl = kprop_moments(net, k_max=km, factor=fac, kind=kind)
            out[f'{seed}_{L}_k{km}{"f" if fac else ""}'] = dict(M=M.numpy().tolist(), flops=fl)
            print(seed, L, km, 'flops %.2e %.0fs' % (fl, time.time() - t0), flush=True)
json.dump(out, open('/home/agent/workspaces/mats/work/arc-noisy/tails/scratch/km_dev.json', 'w'))
