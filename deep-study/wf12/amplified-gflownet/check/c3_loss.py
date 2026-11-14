"""Check of claims 7 and 9 (note section 3, E3b): own implementation, fresh worlds (301-303) and seeds.
Head f = sigmoid(alpha u + beta v), u, v = logits of the weighted / unweighted share of K fresh masked rollouts.
Added by the checker: the best head of the family (oracle fit on exact answers), the constant one half, a plain
ridge pull to (0, 0) as a control for loss (i), a regime where the score is badly wrong, and rollout counts.
Output: out/c3_loss.out"""
import itertools, sys, time
import numpy as np
from scipy.optimize import minimize
from cw import *

t0 = time.time()
KEYS = list(itertools.product([-1, 0, 1], repeat=N))
EMPTY = (-1, -1, -1)
U, NT, NREAL = 2000, 4000, [8, 32, 128, 512]
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DXF = np.tile(DXV, (2, 1))


def cond(px, i, key):
    m = np.all([(XA[:, j] == key[j]) | (key[j] == -1) for j in range(N)], axis=0)
    return px[m & (XA[:, i] == 1)].sum() / px[m].sum()


class Amp:
    def __init__(self, Q, root, R):
        self.P, self.w = {}, {}
        for k in KEYS:
            p = law(Q, root, {j: k[j] for j in range(N) if k[j] != -1}).ravel()
            p = p / p.sum()
            self.P[k] = p
            self.w[k] = np.where(p > 0, 0.5 * np.tile(R, 2) / np.where(p > 0, p, 1), 0.0)

    def feats(self, qi, keys, K, rng):
        qi = np.asarray(qi)
        pw, pu = np.zeros(len(qi)), np.zeros(len(qi))
        by = {}
        for m, k in enumerate(keys):
            by.setdefault(k, []).append(m)
        for k, g in by.items():
            g = np.array(g)
            idx = rng.choice(2 * D, size=(len(g), K), p=self.P[k])
            wt = self.w[k][idx]
            one = DXF[idx, qi[g][:, None]] == 1
            pw[g] = (wt * one).sum(1) / wt.sum(1)
            pu[g] = one.mean(1)
        lg = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))
        return lg(pw), lg(pu), pw

    def score(self, xs, K, rng):
        out = np.zeros(len(xs))
        by = {}
        for m, k in enumerate(xs):
            by.setdefault(k, []).append(m)
        for k, g in by.items():
            out[np.array(g)] = self.w[k][rng.choice(2 * D, size=(len(g), K), p=self.P[k])].mean(1)
        return out


def queries(X, rng):
    n = len(X)
    qi = rng.integers(0, N, n)
    inc = rng.random((n, N)) < 0.5
    keys = [tuple(int(X[m, j]) if (inc[m, j] and j != qi[m]) else -1 for j in range(N)) for m in range(n)]
    return qi, keys, X[np.arange(n), qi]


def lf(th, u, v, y):
    z = th[0] * u + th[1] * v
    return -np.logaddexp(0, np.where(y == 1, -z, z))


def material(amp, K, rng):
    X = XA[rng.integers(0, 8, U)]
    ar = np.arange(U)
    i = rng.integers(0, N, U); j = (i + 1 + rng.integers(0, 2, U)) % N
    key1 = lambda var, val: [tuple(int(val[m]) if c == var[m] else -1 for c in range(N)) for m in range(U)]
    vi, vj = X[ar, i], X[ar, j]
    o = [amp.feats(i, [EMPTY] * U, K, rng)[:2] + (vi,), amp.feats(j, key1(i, vi), K, rng)[:2] + (vj,),
         amp.feats(j, [EMPTY] * U, K, rng)[:2] + (vj,), amp.feats(i, key1(j, vj), K, rng)[:2] + (vi,)]
    perm = np.array(PERMS)[rng.integers(0, 6, U)]
    chain = []
    for t in range(N):
        keys = [tuple(int(X[m, c]) if c in perm[m, :t] else -1 for c in range(N)) for m in range(U)]
        chain.append(amp.feats(perm[:, t], keys, K, rng)[:2] + (X[ar, perm[:, t]],))
    logS = np.log(amp.score([tuple(int(c) for c in x) for x in X], 64, rng))
    qi, keys, _ = queries(X, rng)
    small = amp.feats(qi, keys, K, rng)[:2]
    big = np.clip(amp.feats(qi, keys, 64 * K, rng)[2], 0.02, 0.98)
    return o, chain, logS, small, big


def loss(th, mat, c):
    o, chain, logS, small, big = mat
    if c == "i":
        return np.mean((lf(th, *o[0]) + lf(th, *o[1]) - lf(th, *o[2]) - lf(th, *o[3])) ** 2)
    if c == "ii":
        return np.var(sum(lf(th, *s) for s in chain) - logS)
    return -np.mean(big * lf(th, *small, 1) + (1 - big) * lf(th, *small, 0))


def fit(obj):
    return minimize(obj, np.array([1.0, 0.0]), method="Nelder-Mead", options=dict(maxiter=300, xatol=1e-3, fatol=1e-6)).x


res = {}
REG = {"R1 score right (noise 0)": 0.0, "R2 score wrong (noise 1)": 1.0, "R3 score badly wrong (noise 3)": 3.0}
for reg, noise in REG.items():
    for ws in (301, 302, 303):
        Wd = make_world(ws, noise=noise, goal=0.0)
        amp = Amp(*noisy_policy(Wd["R"], 1.5, 40 + ws, lean=2.0), Wd["R"])
        px, Tx = Wd["pstar"], vmarg(Wd["T"])
        for K in (2, 8):
            rt = np.random.default_rng(5000 + ws + K)
            qi, keys, _ = queries(XA[rt.choice(8, NT, p=px)], rt)
            ut, vt, _ = amp.feats(qi, keys, K, rt)
            pt = np.array([cond(px, a, b) for a, b in zip(qi, keys)])
            Tt = np.array([cond(Tx, a, b) for a, b in zip(qi, keys)])

            def ev(th):
                f = np.clip(1 / (1 + np.exp(-(th[0] * ut + th[1] * vt))), 1e-9, 1 - 1e-9)
                return float(np.mean(pt * np.log(pt / f) + (1 - pt) * np.log((1 - pt) / (1 - f))))

            cell = res.setdefault((reg, K), {})
            add = lambda name, val: cell.setdefault(name, []).append(val)
            add("T itself", float(np.mean(pt * np.log(pt / Tt) + (1 - pt) * np.log((1 - pt) / (1 - Tt)))))
            add("raw (1, 0)", ev([1.0, 0.0])); add("constant one half (0, 0)", ev([0.0, 0.0]))
            ro = np.random.default_rng(6000 + ws + K)                 # best head of the family: exact answers as labels
            oi, ok, _ = queries(XA[ro.choice(8, 20000, p=px)], ro)
            ou, ov, _ = amp.feats(oi, ok, K, ro)
            op = np.array([cond(px, a, b) for a, b in zip(oi, ok)])
            tho = fit(lambda th: -np.mean(op * lf(th, ou, ov, 1) + (1 - op) * lf(th, ou, ov, 0)))
            add("best head of the family (oracle)", ev(tho)); add("oracle alpha", tho[0]); add("oracle beta", tho[1])
            for seed in range(SEEDS):
                rng = np.random.default_rng(77000 + 1000 * ws + 10 * K + seed)
                mat = material(amp, K, rng)
                ri, rk, ry = queries(XA[rng.choice(8, max(NREAL), p=px)], rng)
                ru, rv, _ = amp.feats(ri, rk, K, rng)
                for c in ("i", "ii", "iii"):
                    th = fit(lambda th: loss(th, mat, c))
                    add(f"alone ({c})", ev(th)); add(f"alone ({c}) max|theta|", float(np.abs(th).max()))
                for n in NREAL:
                    data = lambda th: -np.mean(lf(th, ru[:n], rv[:n], ry[:n]))
                    add(f"n={n} data only", ev(fit(data)))
                    for c in ("i", "ii", "iii"):
                        add(f"n={n} data + ({c})", ev(fit(lambda th: data(th) + loss(th, mat, c))))
                    for lr in (0.01, 0.1, 1.0):
                        add(f"n={n} data + ridge {lr:g}", ev(fit(lambda th: data(th) + lr * np.sum(np.asarray(th) ** 2))))
        print(reg, ws, f"{time.time() - t0:.0f}s", flush=True)
L = [f"rollouts used per fit at compute K: data only n*K; (i) 4*U*K = {4 * U}K; (ii) U*(3K + 64) = {U * 3}K + {64 * U};"
     f" (iii) U*65K = {65 * U}K  (U = {U} unlabeled questions); 3 worlds x {SEEDS} seeds"]
for (reg, K), cell in res.items():
    L.append(f"\n== {reg}, K = {K}: excess log loss against reality, nats")
    for name, v in cell.items():
        L.append(f"{name:36s} {np.mean(v):8.4f}")
open("out/c3_loss.out", "w").write("\n".join(L) + "\n")
print("\n".join(L))
