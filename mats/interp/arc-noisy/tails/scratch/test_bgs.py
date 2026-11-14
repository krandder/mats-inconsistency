"""Check bgs.py against the CPU explore levels (dev net 100, depth 4, neurons 1-3, sigma 0.001)."""
import math, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
import bgs

dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
torch.set_num_threads(4)
net = TNet(100, 4, 0.001, device=dev)
lev = {1: [3.448, 4.380, 5.179, 5.877, 6.541], 2: [3.227, 4.282, 5.160, 5.932, 6.630], 3: [4.824, 5.957, 6.947, 7.829, 8.676]}
neur = list(lev)
gen = torch.Generator(device=dev).manual_seed(1)
t0 = time.time()
idx = torch.tensor(neur, device=dev)
pil = bgs.pilot(net, bgs.neuron_scores(net, idx), len(neur), 500, gen, [lev[i] for i in neur])
print('pilot', f'{time.time() - t0:.1f}s', [len(l) for l in pil['levels']], 'n_eval', pil['n_eval'])
R = 16
Lv, Bs = bgs.pad(pil['levels'], pil['bs'], 5)
idxR = idx.repeat_interleave(R)
LvR = Lv.repeat_interleave(R, 0); BsR = Bs.repeat_interleave(R, 0)
o = bgs.frozen(net, bgs.neuron_scores(net, idxR), 1000, gen, LvR, BsR, n_moves=5)
print('frozen', f'{time.time() - t0:.1f}s n_eval {o["n_eval"]}')
for a, i in enumerate(neur):
    rows = o['logp'][a * R:(a + 1) * R]
    out = []
    for t in lev[i]:
        k = int(np.argmin(np.abs(Lv[a].numpy() - t)))
        v = rows[:, k].numpy(); m = v.max(); w = np.exp(v - m)
        out.append('%.2f (rv1 %.2f)' % ((np.log(w.mean()) + m) / math.log(10), w.var(ddof=1) / w.mean() ** 2))
    print(f'neuron {i}:', ' '.join(out))
