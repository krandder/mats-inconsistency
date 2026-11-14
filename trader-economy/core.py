"""Shared pieces for the trader-economy toys: data, LMSR clearing with Kelly
bettors, online learners, mutual-information estimates.

Money units: the LMSR liquidity b is 1 per example, so the market maker's
subsidy per example is at most b*ln2 = 0.693 and equals, in expectation,
b * (log loss of the 0.5 prior - log loss of the market price) in nats.
"""
import itertools
import numpy as np

D = 12
# features 0..3 independent informative, 4..5 noisy copies of 0 and 1
# (substitutes), 6..7 a complementary pair (each nearly useless alone, strong
# together: y depends on x6 - x7, which cancels their shared component),
# 8..11 pure noise.
W_TRUE = np.array([1.6, 1.0, 0.6, 0.3])
REDUNDANT = {4: 0, 5: 1}
COMP_W = 1.0
KIND = ["inf"] * 4 + ["sub"] * 2 + ["comp"] * 2 + ["noise"] * 4


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_data(n, rng):
    x = rng.standard_normal((n, D))
    for j, src in REDUNDANT.items():
        x[:, j] = (x[:, src] + 0.5 * rng.standard_normal(n)) / np.sqrt(1.25)
    u = rng.standard_normal(n)
    e = rng.standard_normal((n, 2))
    x[:, 6] = (u + 0.4 * e[:, 0]) / np.sqrt(1.16)
    x[:, 7] = (u + 0.4 * e[:, 1]) / np.sqrt(1.16)
    diff = (x[:, 6] - x[:, 7]) / (0.4 * np.sqrt(2) / np.sqrt(1.16))
    p = sigmoid(x[:, :4] @ W_TRUE + COMP_W * diff)
    y = (rng.random(n) < p).astype(float)
    return x, y, p


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def fit_logistic(X, y, iters=25):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    th = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = sigmoid(X1 @ th)
        g = X1.T @ (y - p)
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + 1e-6 * np.eye(len(th))
        th += np.linalg.solve(H, g)
    return th


_MI_CACHE = {}


def subset_mi_table(n=40000, seed=12345):
    """Information (nats) about y in every feature subset (all 4096),
    estimated as H(y) - held-out log loss of a large-sample logistic fit on it
    (a lower bound on the mutual information; close here because the truth
    is logistic-linear in the features)."""
    if "t" in _MI_CACHE:
        return _MI_CACHE["t"]
    import os, pickle
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mi_table_v2.pkl")
    if os.path.exists(path):
        _MI_CACHE["t"] = pickle.load(open(path, "rb"))
        return _MI_CACHE["t"]
    rng = np.random.default_rng(seed)
    x, y, _ = make_data(n, rng)
    xt, yt, pt = make_data(n, rng)
    ybar = yt.mean()
    H = -(ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar))
    table = {(): 0.0}
    for k in range(1, D + 1):
        for S in itertools.combinations(range(D), k):
            th = fit_logistic(x[:, S], y, iters=12)
            p = sigmoid(np.hstack([xt[:, S], np.ones((n, 1))]) @ th)
            table[S] = max(0.0, float(H - logloss(p, yt)))
    info = {"H": float(H), "all": table[tuple(range(D))], "bayes": float(H - logloss(pt, yt))}
    _MI_CACHE["t"] = (table, info)
    pickle.dump(_MI_CACHE["t"], open(path, "wb"))
    return table, info


def clear(stake, q, b=1.0, p0=0.5, iters=50):
    """Price p where Kelly bettors' net YES demand equals what an LMSR with
    liquidity b sells when moving from p0 to p.  stake: (K,) or (K,1);
    q: (K,) or (K,N).  Kelly bettor with stake s and belief q, trading at
    price p, ends with s*q/p if YES and s*(1-q)/(1-p) if NO."""
    z0 = np.log(p0 / (1 - p0))
    shape = q.shape[1:] if q.ndim > 1 else ()
    lo = np.full(shape, -12.0)
    hi = np.full(shape, 12.0)
    for _ in range(iters):
        z = 0.5 * (lo + hi)
        p = sigmoid(z)
        dem = np.sum(stake * (q / p - (1 - q) / (1 - p)), axis=0)
        f = dem - b * (z - z0)
        pos = f > 0
        lo = np.where(pos, z, lo)
        hi = np.where(pos, hi, z)
    return sigmoid(0.5 * (lo + hi))


def settle(stake, q, p, y, b=1.0, p0=0.5):
    """Per-agent gains for one example.  Agents trade with each other at the
    clearing price; the net imbalance goes to the LMSR, which charges its path
    cost (cheaper than the uniform price), and the saving is rebated to the
    agents on the net side.  Sum of gains == LMSR loss == b*ln(p_y/p0_y)."""
    if y == 1:
        g = stake * (q / p - 1)
        L = b * np.log(p / p0)
    else:
        g = stake * ((1 - q) / (1 - p) - 1)
        L = b * np.log((1 - p) / (1 - p0))
    delta = stake * (q / p - (1 - q) / (1 - p))
    R = L - g.sum()
    side = np.maximum(0.0, np.sign(delta.sum()) * delta)
    if side.sum() > 1e-12 and R > 0:
        g = g + R * side / side.sum()
    return g, L


class OnlineMLP:
    def __init__(self, d, h, rng, lr=0.02):
        self.W1 = rng.standard_normal((h, d)) / np.sqrt(d)
        self.b1 = np.zeros(h)
        self.w2 = np.zeros(h)
        self.b2 = 0.0
        self.lr = lr

    def predict(self, x):
        self.h = np.tanh(self.W1 @ x + self.b1)
        self.x = x
        return sigmoid(self.w2 @ self.h + self.b2)

    def update(self, p, y):
        e = y - p
        gh = e * self.w2 * (1 - self.h ** 2)
        self.w2 += self.lr * e * self.h
        self.b2 += self.lr * e
        self.W1 += self.lr * np.outer(gh, self.x)
        self.b1 += self.lr * gh


def mean_ci(v):
    v = np.asarray(v, float)
    m = float(v.mean())
    if len(v) < 2:
        return m, 0.0
    from math import sqrt
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(len(v), 2.0)
    return m, float(t * v.std(ddof=1) / sqrt(len(v)))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(a):
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=float)
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r
