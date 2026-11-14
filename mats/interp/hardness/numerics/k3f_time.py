import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'arc-noisy')))
import anet, kprop_noisy as KN
torch.set_num_threads(2)
n, L = int(sys.argv[1]), int(sys.argv[2])
net = anet.Net(950, n=n, L=L, sigma=0.0)
t0 = time.time()
means, fl, sec = KN.kprop_noisy(net, L, 3, factor=True, kind='simple', keep=False, avg_metric=True)
print(n, L, 'k3f flops', f'{fl:.3e}', 'secs', round(sec, 1), 'N_eq', fl / (L * (n * (2 * n - 1) + n)))
