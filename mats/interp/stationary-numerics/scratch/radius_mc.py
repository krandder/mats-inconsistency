"""Direct Monte Carlo of the exact radius chain, recording both Q and r (for the radius PREREG, R8a).
Exact joint simulation: Q_l = (S_l/n)(1 + 2 Q_{l-1} beta_l), r_l = 1 + 2 Q_{l-1} beta_l, S_l ~ chi2_n,
beta_l ~ Beta(M/2, (n-M)/2), M ~ Bin(n, 1/2) (beta = 0 if M = 0, 1 if M = n). P independent chains, burn-in 60n,
T recorded steps; standard errors from 50 independent groups of chains.
usage: radius_mc.py n P T --solver results/radius_corrections.json --out file"""
import sys, os, math, json, time, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('P', type=int); ap.add_argument('T', type=int)
ap.add_argument('--solver', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
n, P, T = a.n, a.P, a.T
sol = json.load(open(a.solver))[str(n)]
qkeys = sorted(sol['quantiles'], key=float)
qv = np.array([sol['quantiles'][k] for k in qkeys])
G = 50
rng = np.random.default_rng([31337, n])
Q = np.full(P, 0.58 * n)
cntQ = np.zeros((G, qv.size)); cntR2 = np.zeros(G); cntQ1 = np.zeros(G); cntQ2 = np.zeros(G); cntR3 = np.zeros(G)
grp = np.arange(P) % G
t0 = time.time()
for t in range(60 * n + T):
    M = rng.binomial(n, 0.5, P)
    b = np.where(M == 0, 0.0, np.where(M == n, 1.0, rng.beta(np.maximum(M, 1) / 2.0, np.maximum(n - M, 1) / 2.0)))
    r = 1.0 + 2.0 * Q * b
    Q = rng.chisquare(n, P) / n * r
    if t >= 60 * n:
        for k in range(qv.size):
            cntQ[:, k] += np.bincount(grp, weights=(Q <= qv[k]), minlength=G)
        cntR2 += np.bincount(grp, weights=(r <= 2.0), minlength=G)
        cntR3 += np.bincount(grp, weights=(r <= 3.0), minlength=G)
        cntQ1 += np.bincount(grp, weights=(Q <= 1.0), minlength=G)
        cntQ2 += np.bincount(grp, weights=(Q <= 2.0), minlength=G)
per = P // G * T


def est(c):
    m = c / per
    return float(m.mean()), float(m.std(ddof=1) / math.sqrt(G))


res = dict(n=n, P=P, T=T, samples=P * T, sec=None)
res['cdf_at_exact_quantiles'] = {k: est(cntQ[:, i]) for i, k in enumerate(qkeys)}
res['P(r<=2)'] = est(cntR2); res['P(r<=3)'] = est(cntR3); res['P(Q<=1)'] = est(cntQ1); res['P(Q<=2)'] = est(cntQ2)
res['sec'] = time.time() - t0
json.dump(res, open(a.out, 'w'), indent=1)
print(json.dumps(res), flush=True)
