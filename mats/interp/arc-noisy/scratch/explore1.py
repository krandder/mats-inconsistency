"""Basic statistics of the noisy ARC nets: variance, mean energy, participation ratio, noise share, antithetic rho."""
import sys, time, math, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet
torch.set_num_threads(2)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
m = 1 << 15
for sigma in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1):
    t0 = time.time()
    net = anet.Net(seed, sigma=sigma)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(m, net.n, generator=g, dtype=torch.float64)
    A = net.forward(x, g, keep=True)
    A2 = net.forward(x, g, keep=True)            # same x, independent noise
    Am = net.forward(-x, g, keep=True)           # antithetic input
    rows = []
    for l in (1, 2, 4, 8, 16, 24, 32):
        a = A[l - 1]; a2 = A2[l - 1]; am = Am[l - 1]
        mu = a.mean(0); var = a.var(0)
        energy = a.pow(2).mean(0).sum()
        mean_frac = (mu @ mu) / energy
        C = torch.cov(a.T); ev = torch.linalg.eigvalsh(C)
        pr = ev.sum() ** 2 / (ev ** 2).sum()
        top5 = ev[-5:].sum() / ev.sum()
        cov_noise_free = ((a - mu) * (a2 - a2.mean(0))).mean(0)   # Var_x(E[a|x]) per neuron
        noise_share = 1 - cov_noise_free.sum() / var.sum()
        rho_anti = (((a - mu) * (am - am.mean(0))).mean(0).sum() / var.sum())
        rows.append(f"L{l:2d} var/q {var.mean() / a.pow(2).mean():.4f} meanE {mean_frac:.4f} PR {pr:6.1f} top5 {top5:.3f} noise {noise_share:+.3f} anti {rho_anti:+.3f}")
    print(f"sigma={sigma}  r_L={net.r[-1]:.3f}  [{time.time()-t0:.0f}s]"); print("\n".join(rows), flush=True)
