"""DEV (not registered; dev seeds >= 9000): calibration of the input-regime picture on one or two nets.
usage: dev_probe.py n L seed [--K 8192] [--P 1024] [--m 32] [--R 16]
Prints: the three-term variance split; gate statistics by layer (Hamming, marginal entropy) against the kernel;
Stein degree-1/2 shares; Saltelli first-order and total indices on m random coordinates (scaled sums);
RQMC over x (scrambled Sobol, plain Gaussian coordinates) variance ratio against MC at N = 256, 1024."""
import argparse, math, sys, time
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('L', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--K', type=int, default=8192); ap.add_argument('--P', type=int, default=1024)
ap.add_argument('--m', type=int, default=32); ap.add_argument('--R', type=int, default=16)
a = ap.parse_args()
n, L, seed = a.n, a.L, a.seed
t0 = time.time()
net = ib.Net(seed, n, L)
rng = np.random.default_rng([1, seed, n, L])
U = ib.sphere(rng.standard_normal((a.K, n)))
stats = {}


def cb(l, z, h):
    stats[l] = ib.gate_layer_stats(z)


G = np.empty((a.K, n))
for i in range(0, a.K, 2048):
    loc = {}
    def cb2(l, z, h):
        loc.setdefault(l, []).append(z > 0)
    G[i:i + 2048] = net.forward(U[i:i + 2048], cb=cb2)
    for l, v in loc.items():
        stats.setdefault(l, []).extend(v)
vs = ib.var_split(G, n)
print('split', {k: round(v, 4) for k, v in vs.items()}, f'{time.time() - t0:.0f}s', flush=True)
rho = 0.0
for l in range(1, L + 1):
    Z = np.concatenate(stats[l])
    p = Z.mean(0)
    ham = float(np.sum(2 * p * (1 - p)) * len(Z) / (len(Z) - 1))
    H = float(ib.bits_entropy(p).sum())
    kern = n * math.acos(rho) / math.pi
    rho = float(ib.kappa(rho))
    if l in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) or l == L:
        print(f'layer {l}: hamming {ham:.1f} (kernel {kern:.1f}), H_marg {H:.1f}, H/ham {H / max(ham, 1e-9):.2f}', flush=True)
# Stein shares on Gaussian inputs
X = rng.standard_normal((a.K, n))
F = net.forward(X)
ss = ib.stein_shares(X, F, deg2_outputs=16)
print('stein', ss, f'{time.time() - t0:.0f}s', flush=True)
# Saltelli on m random coordinates
coords = rng.choice(n, a.m, replace=False)
sl = ib.saltelli(lambda Y: net.forward(Y), n, a.P, rng, coords)
print(f'saltelli: sum S scaled {n * sl["S"].mean():.4f} (se {n * sl["S"].std(ddof=1) / math.sqrt(a.m):.4f}), '
      f'sum T scaled (mean dim) {n * sl["T"].mean():.3f} (se {n * sl["T"].std(ddof=1) / math.sqrt(a.m):.3f}), V {sl["V"]:.3f}',
      f'{time.time() - t0:.0f}s', flush=True)
trv = float(np.sum(F.var(0)))
for k in (8, 10):
    N = 2 ** k
    ests = []
    for r in range(a.R):
        P_ = qmc.Sobol(d=n, scramble=True, seed=np.random.default_rng([4141, seed, n, L, r, k])).random_base2(k)
        Xq = ndtri(np.clip(P_, 1e-16, 1 - 1e-16))
        ests.append(net.forward(Xq).mean(0))
    E = np.array(ests)
    vr = float(np.sum(E.var(0, ddof=1)))
    print(f'RQMC N={N}: gain {trv / N / vr:.2f} (R={a.R})', f'{time.time() - t0:.0f}s', flush=True)
