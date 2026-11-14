"""Bias of a linear control at layer m whose mean comes from a closure: |A_m (mu_m^closure - mu_m^true)|^2 / (n var_L),
plus the R^2 of the control (variance it removes from plain MC). A_m from 2^14 CPU samples (OLS), truth from 2^25-2^26 MC."""
import sys, json, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib
torch.set_num_threads(2)
seed, sigma, L = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
net = anet.Net(seed, sigma=sigma); T = truthlib.load(seed, sigma)
g = torch.Generator().manual_seed(123); N = 1 << 14
x = torch.randn(N, 256, generator=g, dtype=torch.float32)
A = net.forward(x, g, L=L, keep=True, dtype=torch.float32)
y = A[L - 1].double(); vL = T['var'][L - 1].mean()
closures = {k: json.load(open(f'results/kprop_{seed}_s{sigma:g}_L32_{k}.json'))['means'] for k in ('k2', 'k2a', 'k3f')}
for m in ([2, 4, 6] if L == 8 else [4, 8, 12, 16, 20, 24]):
    X = A[m - 1].double(); Xa = torch.cat([X, torch.ones(N, 1, dtype=torch.float64)], 1)
    sol = torch.linalg.lstsq(Xa[: N // 2], y[: N // 2]).solution; Am = sol[:-1].T
    res = y[N // 2:] - Xa[N // 2:] @ sol; r2 = 1 - ((res - res.mean(0)) ** 2).sum() / ((y[N // 2:] - y[N // 2:].mean(0)) ** 2).sum()
    row = f"L={L} m={m:2d}: R2 {float(r2):.3f}"
    for k, means in closures.items():
        e = torch.tensor(means[m - 1]) - torch.tensor(T['mean'][m - 1])
        b = Am @ e
        row += f" | {k}: closure nMSE@m {truthlib.nmse(np.array(means[m - 1]), T, m):.1e} ctrl-bias nMSE@L {float((b @ b) / 256 / vL):.1e}"
    print(row, flush=True)
