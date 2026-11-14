"""eval_lib.py -- scoring helpers. Imports the oracle: exact conditionals, validity, context types. Never used
inside an arm; the experiment scripts call it only after the arms have produced their outputs."""
import math
import numpy as np
import oracle as O


def atom_stats(S, X, LP, counter, K):
    """For states S [m, N] (-1 = empty) that are partial fillings of real boards X [m, N], and M's log-probs
    LP [m, N, K] at those states: one record per (state, empty cell) with exact quantities from the counter.
    Columns: state index, cell, depth (cells filled), nloc (locally allowed values), surprise at the real value,
    exact entropy, exact excess KL(p || q), mass on directly rule-breaking values, mass on locally allowed values with
    no completion, exact p of M's top value, M's top probability, forced (exact conditional is a point mass)."""
    S = np.asarray(S); X = np.asarray(X); LP = np.asarray(LP, dtype=np.float64)
    loc = O.local_candidates(S, K)                                     # [m, N, K]
    rows = []
    for a in range(len(S)):
        s = S[a]; tot, P = counter.conditionals(s)
        emp = np.where(s < 0)[0]; depth = int((s >= 0).sum())
        q = np.exp(LP[a, emp]); p = P[emp]                              # [e, K]
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.where(p > 0, p * np.log(p), 0.0).sum(1)
            kl = np.where(p > 0, p * (np.log(p) - LP[a, emp]), 0.0).sum(1)
        lc = loc[a, emp]
        leak_direct = (q * ~lc).sum(1)
        leak_dead = (q * (lc & (p == 0))).sum(1)
        top = q.argmax(1)
        for j, i in enumerate(emp):
            rows.append((a, i, depth, int(lc[j].sum()), -LP[a, i, X[a, i]], H[j], kl[j], leak_direct[j], leak_dead[j],
                         p[j, top[j]], q[j, top[j]], float(p[j].max() == 1.0)))
    return np.array(rows, dtype=np.float64)


COLS = dict(state=0, cell=1, depth=2, nloc=3, surprise=4, H=5, excess=6, leak_direct=7, leak_dead=8, p_top=9, q_top=10,
            forced=11)


def ctx_type(A):
    """0: locally forced (one locally allowed value); 1: forced but not locally; 2: free."""
    return np.where(A[:, COLS["nloc"]] == 1, 0, np.where(A[:, COLS["forced"]] == 1, 1, 2)).astype(int)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=np.float64); rng = np.random.default_rng(seed)
    m = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def near_miss(X, rng, kind, K):
    """Invalid boards next to valid ones. 'cell': one cell changed to another value. 'swap': two cells of one row
    swapped (the row stays a permutation; columns or boxes break)."""
    X = np.array(X).copy(); N = K * K
    for a in range(len(X)):
        if kind == "cell":
            i = rng.integers(N); X[a, i] = (X[a, i] + rng.integers(1, K)) % K
        else:
            r = rng.integers(K); c1, c2 = rng.choice(K, 2, replace=False); i, j = r * K + c1, r * K + c2
            X[a, i], X[a, j] = X[a, j], X[a, i]
    return X


def mcnemar_p(a_only, b_only):
    """Exact two-sided sign test on discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def fit_mixture(L, iters=500):
    """Weights of a mixture over K normalised laws, by EM on real boards. L[n, k] = log T_k(x_n)."""
    L = np.asarray(L, dtype=np.float64); n, k = L.shape; w = np.full(k, 1.0 / k)
    for _ in range(iters):
        a = L + np.log(w + 1e-300)[None, :]; a -= a.max(1, keepdims=True); r = np.exp(a); r /= r.sum(1, keepdims=True)
        w = r.mean(0)
    return w


def mixture_logT(L, w):
    a = np.asarray(L, dtype=np.float64) + np.log(w + 1e-300)[None, :]
    m = a.max(1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(1, keepdims=True)))[:, 0]
