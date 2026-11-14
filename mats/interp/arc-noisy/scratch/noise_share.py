"""Noise share of the depth-L output variance and participation ratio, final weights (anet.make_weights), one dev net.
Noise share = 1 - Var_x(E[a|x]) / Var(a), with Var_x(E[a|x]) estimated by the covariance of two paths sharing x."""
import sys, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib
torch.set_num_threads(2)
seed = int(sys.argv[1]); N = 1 << int(sys.argv[2])
T = truthlib.load(seed, 0.001); d = T['d']; Nt = T['N']
for L in (8, 32):
    m = d['S1'][L - 1] / Nt; C = d[f'M2_{L}'] / Nt - np.outer(m, m); ev = np.linalg.eigvalsh(C)
    print(f"seed {seed} L={L} sigma=0.001: participation ratio {ev.sum() ** 2 / (ev ** 2).sum():.1f}, top-5 share {ev[-5:].sum() / ev.sum():.2f}")
for sigma in (0.003, 0.01, 0.03, 0.1):
    net = anet.Net(seed, sigma=sigma)
    g = torch.Generator().manual_seed(21)
    x = torch.randn(N, 256, generator=g, dtype=torch.float32)
    A1 = net.forward(x, g, keep=True, dtype=torch.float32); A2 = net.forward(x, g, keep=True, dtype=torch.float32)
    row = []
    for L in (8, 32):
        a = A1[L - 1].double(); b = A2[L - 1].double()
        va = a.var(0).sum(); cab = ((a - a.mean(0)) * (b - b.mean(0))).sum(0).sum() / (N - 1)
        row.append(f"L={L}: {float(1 - cab / va) * 100:.1f}%")
    print(f"seed {seed} sigma={sigma}: noise share of output variance " + ', '.join(row), flush=True)
