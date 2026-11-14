"""Fraction of Var(a_L) linearly explained by h_m (cross-fitted multi-output ridge-free regression), seed 100."""
import sys, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet
torch.set_num_threads(2)
seed, sigma, L, lg = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
net = anet.Net(seed, sigma=sigma); g = torch.Generator().manual_seed(77)
N = 1 << lg
x = torch.randn(N, 256, generator=g, dtype=torch.float32)
A = net.forward(x, g, keep=True, dtype=torch.float32)
y = A[L - 1].double(); half = N // 2
yv = y[half:] - y[half:].mean(0); tot = (yv ** 2).sum()
for m in [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 30, 31]:
    if m >= L: break
    X = (x if m == 0 else A[m - 1]).double()
    Xa = torch.cat([X, torch.ones(N, 1, dtype=torch.float64)], 1)
    B = torch.linalg.lstsq(Xa[:half], y[:half]).solution
    res = y[half:] - Xa[half:] @ B
    r2 = 1 - ((res - res.mean(0)) ** 2).sum() / tot
    # share along the mean direction of the residual
    print(f"L={L} m={m:2d}: R^2 {float(r2):.4f}", flush=True)
