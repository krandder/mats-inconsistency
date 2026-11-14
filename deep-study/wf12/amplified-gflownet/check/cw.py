"""Checker's own small world (independent of scripts/world.py).  3 binary variables.
Terminal state = demonstration d = (order, values): 48.  Score R(d) = prod of M's step numbers * goal(values).
Rollout policy = tree policy over (alien bit a, then the demonstration step by step); a is dropped at submission."""
import itertools
import numpy as np

N = 3
PERMS = list(itertools.permutations(range(N)))
XS = list(itertools.product([0, 1], repeat=N))
DEMS = [(p, x) for p in PERMS for x in XS]
D = len(DEMS)
ORD = np.array([PERMS.index(p) for p, _ in DEMS])
VAL = np.array([XS.index(x) for _, x in DEMS])
XA = np.array(XS)
DXV = XA[VAL]                       # (48, 3)
PATH = [tuple((i, x[i]) for i in p) for p, x in DEMS]
DIDX = {PATH[d]: d for d in range(D)}


def kids(pre):
    used = {i for i, _ in pre}
    return [(i, v) for i in range(N) if i not in used for v in (0, 1)]


NODES = [()] + [((i, v),) for i in range(N) for v in (0, 1)]
NODES += [n + (c,) for n in NODES[1:] for c in kids(n)]          # prefixes of length 0, 1, 2
UNDER = {n: np.array([d for d in range(D) if PATH[d][:len(n)] == n]) for n in NODES}


def make_world(seed, noise=1.0, goal=0.5, exact_order=None):
    rng = np.random.default_rng(seed)
    pstar = rng.dirichlet(np.full(8, 1.5))
    cache = {}

    def m(i, ctx):                   # M's number for x_i = 1 given context dict
        key = (i, tuple(sorted(ctx.items())))
        if key not in cache:
            sel = np.ones(8, bool)
            for j, v in ctx.items():
                sel &= XA[:, j] == v
            p1 = pstar[sel & (XA[:, i] == 1)].sum() / pstar[sel].sum()
            e = rng.normal() * noise
            if exact_order is not None:
                k = len(ctx)
                if set(ctx) == set(exact_order[:k]) and exact_order[k] == i:
                    e = 0.0
            cache[key] = 1 / (1 + np.exp(-(np.log(p1 / (1 - p1)) + e)))
        return cache[key]

    # fill the cache in a fixed order so that the world does not depend on call order
    for k in range(N):
        for S in itertools.combinations(range(N), k):
            for vals in itertools.product([0, 1], repeat=k):
                for i in range(N):
                    if i not in S:
                        m(i, dict(zip(S, vals)))
    g = np.exp(goal * rng.normal(size=8))
    R = np.zeros(D)
    for d, path in enumerate(PATH):
        ctx, r = {}, 1.0
        for i, v in path:
            p = m(i, ctx)
            r *= p if v else 1 - p
            ctx[i] = v
        R[d] = r * g[VAL[d]]
    return dict(pstar=pstar, R=R, T=R / R.sum(), Z=R.sum())


def exact_logits(R, pb=None):
    """Edge log-flows of the exact flow for backward law pb[a, d] over the alien bit (default 1/2)."""
    pb = np.full((2, D), 0.5) if pb is None else pb
    root = np.log((R * pb).sum(1))
    Q = {(a, n): np.array([np.log((R * pb[a])[UNDER.get(n + (c,), np.array([DIDX.get(n + (c,), 0)]))].sum())
                           for c in kids(n)]) for a in (0, 1) for n in NODES}
    return Q, root


def noisy_policy(R, eps, seed, lean=0.0, hide=None):
    rng = np.random.default_rng(seed)
    Q, root = exact_logits(R)
    for (a, n), q in Q.items():
        q += eps * rng.normal(size=q.shape)
        if n == () and lean:
            q += lean * np.array([c[0] == (0 if a == 0 else 2) for c in kids(n)], float)
        if hide is not None:
            for ci, c in enumerate(kids(n)):
                if hide(dict(n + (c,))):
                    q[ci] = -np.inf
    return Q, root + eps * rng.normal(size=2)


def sm(q):
    mx = np.max(q)
    if not np.isfinite(mx):
        return np.zeros_like(q)
    e = np.exp(q - mx)
    return e / e.sum()


def law(Q, root, C=None):
    """Joint law P[a, d] of the tree policy; C = evidence dict: contradicting edges masked, renormalised."""
    P = np.zeros((2, D))
    pa = sm(root)
    for a in (0, 1):
        for d, path in enumerate(PATH):
            p = pa[a]
            for t in range(N):
                n = path[:t]
                q = Q[(a, n)].copy()
                ks = kids(n)
                if C:
                    q[[k for k, (i, v) in enumerate(ks) if i in C and C[i] != v]] = -np.inf
                p *= sm(q)[ks.index(path[t])]
            P[a, d] = p
    return P


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def vmarg(pd):
    out = np.zeros(8)
    np.add.at(out, VAL, pd)
    return out


def mi(P):
    P = P / P.sum()
    o = P.sum(1, keepdims=True) @ P.sum(0, keepdims=True)
    k = P > 0
    return float((P[k] * np.log(P[k] / o[k])).sum())
