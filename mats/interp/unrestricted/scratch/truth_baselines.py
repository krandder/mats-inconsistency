"""Truth and sampling baselines for one net (sigma = 0 unless --sigma): radius-exact (RB) estimates on the sphere.
Truth: mean of R_T scrambled-Sobol RB replicates at 2^m_T points (scramble seeds 1000+). Baselines: RB-MC (analytic
nMSE from the variance), scrambled-Sobol RB-RQMC replicates (seeds 0..R-1) and unscrambled Sobol RB-QMC
(deterministic) at N = 2^4 .. 2^m_max. Writes runs/truth_<net>.json.
Usage: truth_baselines.py n L seed [--mT 22] [--RT 8] [--mmax 20] [--sigma 0]
"""
import argparse
import json
import math
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import Net, mean_radius, pass_flops, sobol_directions, rb_stats  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('L', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--mT', type=int, default=22); ap.add_argument('--RT', type=int, default=8)
ap.add_argument('--mmax', type=int, default=20)
ap.add_argument('--out', default=None)
a = ap.parse_args()
t0 = time.time()
net = Net(a.seed, a.n, a.L)
n = a.n
Er = mean_radius(n)


def rb_estimate(U):
    s1, s2 = rb_stats(net, U)
    return Er * s1 / U.shape[0], s1, s2


# truth: RQMC replicates at 2^mT
tr_reps = []
S1 = np.zeros(n); S2 = np.zeros(n); P = 0
for r in range(a.RT):
    U = sobol_directions(n, a.mT, scramble=True, seed=1000 + r)
    est, s1, s2 = rb_estimate(U)
    tr_reps.append(est)
    S1 += s1; S2 += s2; P += U.shape[0]
tr_reps = np.array(tr_reps)
truth = tr_reps.mean(0)
Eg = S1 / P
Eg2 = S2 / P
V = float(np.sum(n * Eg2 - Er ** 2 * Eg ** 2))           # tr Var_x f
Vrb = float(Er ** 2 * np.sum(Eg2 - Eg ** 2))             # per-sample variance of the RB estimator
truth_se2 = float(np.sum(tr_reps.var(0, ddof=1)) / a.RT)  # squared SE of the truth (sum over outputs)
if V <= 0:
    print(f'n={n} L={a.L} seed={a.seed}: dead net (V = 0), skipped', flush=True)
    sys.exit(0)
res = dict(n=n, L=a.L, seed=a.seed, truth=truth.tolist(), V=V, Vrb=Vrb, rho_rb=Vrb / V,
           truth_nmse=truth_se2 / V, truth_reps=a.RT, truth_m=a.mT, pass_flops=pass_flops(n, a.L),
           mean_norm2=float(np.sum(truth ** 2)))
# baselines
rq, qm = {}, {}
for m in range(4, a.mmax + 1):
    R = 32 if m <= 12 else 16 if m <= 16 else 8
    errs = []
    for r in range(R):
        U = sobol_directions(n, m, scramble=True, seed=r)
        est, _, _ = rb_estimate(U)
        errs.append(float(np.sum((est - truth) ** 2)))
    rq[m] = dict(R=R, nmse=float(np.mean(errs)) / V, nmse_se=float(np.std(errs, ddof=1) / math.sqrt(R)) / V)
    U = sobol_directions(n, m, scramble=False)
    est, _, _ = rb_estimate(U)
    qm[m] = dict(nmse=float(np.sum((est - truth) ** 2)) / V)
res['rqmc'] = rq
res['qmc_det'] = qm
res['seconds'] = time.time() - t0
out = a.out or os.path.join(os.path.dirname(__file__), '..', 'runs', f'truth_n{n}_L{a.L}_seed{a.seed}.json')
with open(out, 'w') as f:
    json.dump(res, f)
print(f'n={n} L={a.L} seed={a.seed} V={V:.4g} rho_rb={Vrb / V:.3f} truth_nmse={truth_se2 / V:.2e} '
      f'rqmc@2^{a.mmax}={rq[a.mmax]["nmse"]:.2e} qmc_det@2^{a.mmax}={qm[a.mmax]["nmse"]:.2e} {time.time() - t0:.0f}s',
      flush=True)
