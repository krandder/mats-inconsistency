import sys, json, time, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
from kprop_tail import kprop_moments
torch.set_num_threads(4)
p = '/home/agent/workspaces/mats/work/arc-noisy/tails/scratch/km_dev.json'
out = json.load(open(p))
for seed in (100, 101):
    for (km, fac, kind, tag) in [(3, True, 'augment', 'k3fa')]:
        for L in (4, 8):
            net = TNet(seed, L, 0.001)
            M, fl = kprop_moments(net, k_max=km, factor=fac, kind=kind)
            out[f'{seed}_{L}_{tag}'] = dict(M=M.numpy().tolist(), flops=fl)
            print(seed, L, tag, '%.2e' % fl, flush=True)
json.dump(out, open(p, 'w'))
