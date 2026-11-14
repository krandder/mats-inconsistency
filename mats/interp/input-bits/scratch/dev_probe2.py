"""DEV (not registered; dev seeds >= 9000): depth scan of the input-regime structure on one net.
usage: dev_probe2.py n seed L1,L2,... [--K 16384] [--P 1024] [--m 32] [--R 32]
Per depth L: three-term split; Hermite shares of a(x): degree 1 (Stein), univariate degrees 1-4 (first-order share
S_1^(<=4)); degree-1 share of |a| and of the direction part; Jansen total-index sum (mean dimension) on m random
coordinates; RQMC over x (plain Gaussian coordinates) gain at N = 1024 with R scrambles."""
import argparse, math, sys, time
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('seed', type=int); ap.add_argument('Ls')
ap.add_argument('--K', type=int, default=16384); ap.add_argument('--P', type=int, default=1024)
ap.add_argument('--m', type=int, default=32); ap.add_argument('--R', type=int, default=32)
a = ap.parse_args()
n, seed = a.n, a.seed
Ls = [int(s) for s in a.Ls.split(',')]
t0 = time.time()
net = ib.Net(seed, n, max(Ls))


def fwd(X, L):
    h = X
    for l in range(L):
        h = np.maximum(h @ net.W[l].T, 0.0)
    return h


def herm(x, d):
    return [x, x * x - 1, x ** 3 - 3 * x, x ** 4 - 6 * x * x + 3][d - 1]


def univ_shares(X, F):
    P = X.shape[0]; h = P // 2
    Fc = F - F.mean(0)
    V = float(np.sum(F.var(0, ddof=1)))
    out = []
    for d in range(1, 5):
        Hd = herm(X, d)
        g1 = Fc[:h].T @ Hd[:h] / h; g2 = Fc[h:2 * h].T @ Hd[h:2 * h] / h
        out.append(float(np.sum(g1 * g2)) / math.factorial(d) / V)
    return out, V


rng = np.random.default_rng([2, seed, n])
X = rng.standard_normal((a.K, n))
U = ib.sphere(X)
for L in Ls:
    G = fwd(U, L)
    vs = ib.var_split(G, n)
    F = G * np.linalg.norm(X, axis=1, keepdims=True)
    us, V = univ_shares(X, F)
    nrm = np.linalg.norm(F, axis=1, keepdims=True)
    usA, VA = univ_shares(X, nrm)
    # direction part: F - (mean amplitude-direction model): use d = F/|F| scaled by mean |F|
    D = F / np.maximum(nrm, 1e-300) * float(nrm.mean())
    usD, VD = univ_shares(X, D)
    coords = rng.choice(n, a.m, replace=False)
    sl = ib.saltelli(lambda Y: fwd(Y, L), n, a.P, rng, coords)
    Dm = n * sl['T'].mean(); Dse = n * sl['T'].std(ddof=1) / math.sqrt(a.m)
    trv = V
    N = 1024; ests = []
    for r in range(a.R):
        P_ = qmc.Sobol(d=n, scramble=True, seed=np.random.default_rng([4141, seed, n, L, r])).random_base2(10)
        ests.append(fwd(ndtri(np.clip(P_, 1e-16, 1 - 1e-16)), L).mean(0))
    E = np.array(ests)
    gain = trv / N / float(np.sum(E.var(0, ddof=1)))
    S1 = sum(us)
    print(f'L={L}: s_rad {vs["s_rad"]:.3f} s_amp {vs["s_amp"]:.3f} s_dir {vs["s_dir"]:.3f} | a: lin {us[0]:.3f} '
          f'univ<=4 {S1:.3f} ({", ".join(f"{u:.3f}" for u in us)}) | |a|: lin {usA[0]:.3f} univ {sum(usA):.3f} | '
          f'dir-part: lin {usD[0]:.3f} univ {sum(usD):.3f} | meanD {Dm:.2f}+-{Dse:.2f} | RQMC1024 {gain:.2f} '
          f'vs 1/(1-S1) {1 / (1 - S1):.2f} | {time.time() - t0:.0f}s', flush=True)
