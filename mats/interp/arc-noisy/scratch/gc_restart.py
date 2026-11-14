import sys, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib, gauss_relu as GR
torch.set_num_threads(2)
seed, sigma = int(sys.argv[1]), float(sys.argv[2])
net = anet.Net(seed, sigma=sigma); T = truthlib.load(seed, sigma); d = T['d']; Nt = T['N']; I = torch.eye(256, dtype=torch.float64)
for m0 in (0, 8, 16, 24, 31):
    if m0 == 0:
        mu = torch.zeros(256, dtype=torch.float64); Sig = I.clone()
    else:
        mu = torch.tensor(d['S1'][m0 - 1] / Nt); Sig = torch.tensor(d[f'M2_{m0}'] / Nt) - torch.outer(mu, mu) + float(net.s[m0 - 1]) ** 2 * I
    errs = {}
    for l in range(m0, 32):
        W = net.W[l]; mu, Ca, _ = GR.relu_moments(W @ mu, W @ Sig @ W.T); Sig = Ca + float(net.s[l]) ** 2 * I
        if l + 1 in (8, 16, 24, 32):
            errs[l + 1] = truthlib.nmse(mu, T, l + 1)
    print(f"GC restarted from true moments at layer {m0:2d}: " + '  '.join(f"@{k}: {v:.2e}" for k, v in errs.items()), flush=True)
