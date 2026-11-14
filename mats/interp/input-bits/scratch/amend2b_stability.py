"""Amendment 2B (PREREG; suggested by astra round 2, T4): ANOVA-order generating function of the noise-regime window integrand
and the hybrid MC floor. Setting as work/stationary/scratch/partB.py rqmc mode (Model S, sigma = 1, burn-in B = 8n from
h = 0, a = relu(W_1 h_0), window = the d = min(B, 21201 // n) most recent layers).
For a pair of independent histories A, B and t in {0, 0.25, 0.5, 0.75, 1}, the history M_t takes each scalar window coordinate
from A with probability t (nested across t: shared iff U < t for one uniform U per coordinate) and from B otherwise; the
older layers always come from B. Then
  C(t) = tr Cov(a(A), a(M_t)) / V = sum_{S subset window, S nonempty} t^|S| V_S / V   (V_S: ANOVA variances of the window
  part g(U) = E[a | window]), C(1) = 1 - eta, eta = E_U Var_Z(a | U) / V the hybrid floor, C(0) = 0.
C(t)/C(1) is the probability-generating function of the window integrand's ANOVA order; adding coordinates at a fixed order
leaves it unchanged. tr Cov is estimated as V - E|a(A) - a(M_t)|^2 / 2 with V from the A sample; SEs from 8 batches.
usage: amend2b_stability.py n seed [--P 8192] [--out file]"""
import argparse, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary/lib')
import st

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--P', type=int, default=8192); ap.add_argument('--out')
a = ap.parse_args()
n, seed = a.n, a.seed
t0 = time.time()
B = 8 * n
d = min(B, 21201 // n)
Wc = st.WCache(seed, n, max_bytes=1.5e9)
TS = [0.0, 0.25, 0.5, 0.75, 1.0]
CHK = 1024
nb = a.P // CHK
rows = []
for b in range(nb):
    rng = np.random.default_rng([7601, seed, n, b])
    hA = np.zeros((CHK, n)); hM = np.zeros((len(TS), CHK, n))
    for l in range(-B + 1, 1):
        age = -l
        W = Wc(l)
        xa = rng.standard_normal((CHK, n)); xb = rng.standard_normal((CHK, n))
        hA = np.maximum(hA @ W.T, 0.0) + xa
        if age < d:
            Uu = rng.uniform(size=(CHK, n))
            for i, t in enumerate(TS):
                xm = np.where(Uu < t, xa, xb)
                hM[i] = np.maximum(hM[i] @ W.T, 0.0) + xm
        else:
            for i in range(len(TS)):
                hM[i] = np.maximum(hM[i] @ W.T, 0.0) + xb
    W1 = Wc(1)
    aA = np.maximum(hA @ W1.T, 0.0)
    aM = [np.maximum(hM[i] @ W1.T, 0.0) for i in range(len(TS))]
    rows.append(dict(VA=float(np.sum(aA.var(0, ddof=1))), msd=[float(np.mean(np.sum((aA - x) ** 2, 1))) for x in aM],
                     mean=aA.mean(0).tolist(), sq=float(np.mean(np.sum(aA * aA, 1)))))
    print(f'batch {b + 1}/{nb} {time.time() - t0:.0f}s', flush=True)
VA = float(np.mean([r['VA'] for r in rows]))
msd = np.array([r['msd'] for r in rows])                      # (nb, 5)
C = 1 - msd.mean(0) / (2 * VA)
Cb = 1 - msd / (2 * np.array([r['VA'] for r in rows])[:, None])
se = Cb.std(0, ddof=1) / math.sqrt(nb)
res = dict(n=n, seed=seed, B=B, d_layers=d, P=nb * CHK, V=VA, t=TS, C=C.tolist(), C_se=se.tolist(), eta=float(1 - C[-1]),
           C_norm=(C / C[-1]).tolist(), sec=time.time() - t0)
print({k: v for k, v in res.items() if k not in ('t',)}, flush=True)
if a.out:
    json.dump(res, open(a.out, 'w'))
