import sys, math, time, json
import torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
from kprop_tail import kprop_moments
import deduce as D
torch.set_num_threads(4)
net = TNet(100, 4, 0.001)
t0 = time.time()
M, fl = kprop_moments(net, k_max=3, factor=True)
print('K3f moments flops %.2e %.0fs' % (fl, time.time() - t0))
for i in range(4):
    k = D.cumulants_from_moments(M[:, i].numpy())
    print(i, 'mu %.3f sd %.3f skew %.3f exkurt %.3f' % (k[0], math.sqrt(k[1]), k[2] / k[1] ** 1.5, k[3] / k[1] ** 2))
np.save('/tmp/km_k3f_100_L4.npy', M.numpy())
