"""T0c (theory heuristic, random-mask surrogate; no real-net data): coherence of two paths' top-k right frames when
their gates differ in a fraction f of units at every layer from the cut to the output (masks iid Bernoulli(1/2),
shared Gaussian weights W ~ N(0, 2/n); path 2's mask = path 1's with a fraction f flipped). Gate disagreement
f = theta/pi for two Gaussian pre-activation vectors at angle theta, so this maps frame coherence to the state angle,
under the annealed assumption that real masks behave like random ones given their disagreement.
Prints mean overlap O(F, F'), CS of F' on J, and the Frobenius coherence rho = tr(A A') / sqrt(tr A^2 tr A'^2), A = J^T J."""
import math, sys
import numpy as np

def frames(J, k):
    _, s, vt = np.linalg.svd(J)
    return vt[:k].T, s

def run(n, Lp, k, fs, pairs=48, seed=2):
    rng = np.random.default_rng(seed)
    out = {f: [] for f in fs}
    for _ in range(pairs):
        W = [rng.standard_normal((n, n)) * math.sqrt(2.0 / n) for _ in range(Lp)]
        D = [rng.random(n) < 0.5 for _ in range(Lp + 1)]     # D[0] = cut mask, D[l] = layer c+l mask
        def jac(Ds):
            J = np.diag(Ds[0].astype(float))
            for l in range(Lp):
                J = (Ds[l + 1][:, None] * W[l]) @ J
            return J
        J = jac(D); F, s = frames(J, k); A = J.T @ J
        for f in fs:
            D2 = [d ^ (rng.random(n) < f) for d in D]
            J2 = jac(D2); F2, s2 = frames(J2, k); A2 = J2.T @ J2
            O = np.linalg.norm(F.T @ F2) ** 2 / k
            CS = np.linalg.norm(J @ F2) ** 2 / np.sum(s[:k] ** 2)
            rho = np.trace(A @ A2) / math.sqrt(np.trace(A @ A) * np.trace(A2 @ A2))
            out[f].append((O, CS, rho))
    return {f: np.array(v).mean(0) for f, v in out.items()}

fs = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.25)
for n, Lp, k in ((64, 16, 12), (128, 16, 22), (128, 8, 40), (64, 8, 22)):
    r = run(n, Lp, k, fs)
    print(f"n={n} L'={Lp} k={k} chance~{2*k/n:.2f}: " + "  ".join(f"f={f:g}: O={v[0]:.3f} CS={v[1]:.3f} rho={v[2]:.3f}" for f, v in r.items()))
    sys.stdout.flush()
