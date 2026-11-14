"""c4_recurrence.py (checker) -- can the stiff set be read from real data by RECURRENCE, with no hand choice?
An atom "value v at cell j given the context c" can be judged on real data only if the context c recurs. For each
context size d: how often does a context of that size recur in n real grids, and are the zeros of the recurring
contexts (values never seen after the context) exactly the rule? For d = 1 the full table is counted; M's vetoes
(one-cell atoms below the worker's theta) are compared with the zeros certified by data alone."""
import json, math
import numpy as np, torch
import ck

K, N = 6, 36; P = ck.peers(6); THETA = json.load(open(f"{ck.W}/out/pair_meta.json"))["theta"]
net = ck.M("mA6", 6); Q = ck.pair_table(net).numpy()
imp = np.zeros((N, K, N, K), bool); diag = np.zeros((N, K, N, K), bool)
for i in range(N):
    diag[i, :, i, :] = True
    for u in range(K):
        imp[i, u, P[i], u] = True
veto_M = (Q < THETA) & ~diag
print(f"M's vetoes at one-cell contexts (log q < theta = {THETA:.2f}): {int(veto_M.sum())}; equal to the rule (peer, same digit: {int(imp.sum())}): {bool((veto_M == imp).all())}")
OUT = {}
Xall = ck.sample_grids(6, 0, 4000, 6060).numpy()                 # real grids of the TRAINING orbits (data the fleet already has)
for n in (50, 100, 200, 400, 800, 1600, 4000):
    X = Xall[:n]; oh = np.zeros((n, N * K)); oh[np.arange(n)[:, None], np.arange(N)[None, :] * K + X] = 1
    C = (oh.T @ oh).reshape(N, K, N, K); occ = C[np.arange(N), :, np.arange(N), :].diagonal(axis1=1, axis2=2)   # occ[i, u] = count of x_i = u
    zero = (C == 0) & ~diag
    false_zero = int((zero & ~imp).sum()); missed = int((imp & ~zero).sum())
    nmin = int(occ.min()); ub = 1 - 0.001 ** (1 / max(1, nmin))
    OUT[n] = dict(min_recurrence=nmin, data_zeros=int(zero.sum()), false_zeros=false_zero, rule_zeros_missed=missed,
                  data_zeros_equal_M_vetoes=bool((zero == veto_M).all()), ub999=ub)
    print(f"   n = {n:5d} real grids: each one-cell context recurs >= {nmin:4d} times; values never seen after a context: {int(zero.sum()):5d}; "
          f"of these possible in truth (false zeros) {false_zero:5d}; rule zeros not found {missed}; equal to M's vetoes: {OUT[n]['data_zeros_equal_M_vetoes']}; "
          f"a never-seen value has true probability < {ub:.3f} (99.9%)")
# recurrence by context size: how many of n grids share the values of a random real d-cell context
rng = np.random.default_rng(1); X = Xall
print("   recurrence of larger contexts in 4,000 real grids (mean and 5th percentile of the number of grids that contain a random real d-cell context):")
for d in (1, 2, 3, 4, 6, 9):
    cnt = []
    for _ in range(300):
        g = X[rng.integers(len(X))]; idx = rng.choice(N, d, replace=False); cnt.append(int((X[:, idx] == g[idx]).all(1).sum()))
    OUT[f"rec{d}"] = dict(mean=float(np.mean(cnt)), p5=float(np.quantile(cnt, .05)))
    print(f"      d = {d}: mean {np.mean(cnt):8.1f}, 5th percentile {np.quantile(cnt, .05):6.0f}")
json.dump(OUT, open("out/c4_recurrence.json", "w"), indent=1)
