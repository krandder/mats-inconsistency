"""Exact small world for the amplified-GFlowNet study (numpy, enumeration).

Two levels, as in FRAME section 5.
- Terminal state = a DEMONSTRATION d = (order sigma of the 3 binary variables, their values x): 6 x 8 = 48.
  The simple world model M scores it step by step: R(d) = prod_t M(value_t | variable_t, context_t) * goal(x).
- The rollout policy is a flow over an AUGMENTED state: an alien coordinate a in {0,1} is chosen first, carried
  along, and dropped at submission.  96 augmented trajectories tau = (a, d); each d has 2 of them.
  The policy is stored as edge log-flows Q[(a, prefix)] (SoftDQN form); P_F = softmax(Q).
Target: T(d) = R(d) / Z.  Nothing here is a trained network; all laws are exact arrays.
"""
import itertools
import numpy as np

N = 3
PERMS = list(itertools.permutations(range(N)))
XS = list(itertools.product([0, 1], repeat=N))
DEM = [(p, x) for p in PERMS for x in XS]          # index d = 8 * perm_index + x_index
D, A = len(DEM), 2
STEPS = [tuple((i, x[i]) for i in p) for (p, x) in DEM]
XIDX = np.array([XS.index(x) for (_, x) in DEM])
PIDX = np.array([PERMS.index(p) for (p, _) in DEM])
XARR = np.array(XS)                                  # (8, 3)
DX = XARR[XIDX]                                      # (48, 3) values of each demonstration


def children(prefix):
    used = {i for i, _ in prefix}
    return [(i, v) for i in range(N) if i not in used for v in (0, 1)]


PREFIXES = [()]
for L in range(1, N):
    PREFIXES += [pre + (c,) for pre in PREFIXES if len(pre) == L - 1 for c in children(pre)]
DESC = {pre: np.array([d for d in range(D) if STEPS[d][:len(pre)] == pre]) for pre in PREFIXES}
for d in range(D):
    DESC[STEPS[d]] = np.array([d])


def lse(v):
    m = np.max(v)
    if not np.isfinite(m):
        return -np.inf
    return m + np.log(np.sum(np.exp(v - m)))


def softmax(v):
    m = np.max(v)
    if not np.isfinite(m):
        return np.zeros_like(v)
    e = np.exp(v - m)
    return e / e.sum()


def make_world(seed, sM=0.0, goal=0.0, stiff_order=None):
    """Reality p*, world model M (exact conditionals of p* plus logit noise sM), score R, target T.
    stiff_order: if given, M is exact along that order's contexts and noisy elsewhere (one reliable route)."""
    rng = np.random.default_rng(seed)
    pstar = rng.dirichlet(np.ones(8) * 1.5)
    M = {}
    for ctx in itertools.product([-1, 0, 1], repeat=N):
        for i in range(N):
            if ctx[i] != -1:
                continue
            sel = np.all([(XARR[:, j] == ctx[j]) | (ctx[j] == -1) for j in range(N)], axis=0)
            p1 = pstar[sel & (XARR[:, i] == 1)].sum() / pstar[sel].sum()
            noise = rng.normal() * sM
            if stiff_order is not None:
                filled = tuple(j for j in stiff_order if ctx[j] != -1)
                if filled == tuple(stiff_order[:len(filled)]) and stiff_order[len(filled)] == i:
                    noise = 0.0
            lg = np.log(p1 / (1 - p1)) + noise
            M[(ctx, i)] = 1 / (1 + np.exp(-lg))
    g = np.exp(goal * rng.normal(size=8))
    R = np.zeros(D)
    for d, st in enumerate(STEPS):
        ctx, r = [-1] * N, 1.0
        for (i, v) in st:
            m = M[(tuple(ctx), i)]
            r *= m if v == 1 else 1 - m
            ctx[i] = v
        R[d] = r * g[XIDX[d]]
    return dict(pstar=pstar, M=M, R=R, Z=R.sum(), T=R / R.sum(), goal=g, rng=rng)


def exact_Q(R, pb=None):
    """Exact edge log-flows on the augmented tree for backward policy pb[a, d] (default uniform 1/2)."""
    pb = np.full((A, D), 0.5) if pb is None else pb
    Q = {}
    for a in range(A):
        for pre in PREFIXES:
            Q[(a, pre)] = np.array([np.log((R * pb[a])[DESC[pre + (c,)]].sum()) for c in children(pre)])
    Qroot = np.array([np.log((R * pb[a]).sum()) for a in range(A)])
    return Q, Qroot


def noisy_policy(R, eps, bias, seed):
    """Imperfect rollout policy: exact log-flows + eps * noise; alien a=0 leans to variable 0 first, a=1 to 2."""
    rng = np.random.default_rng(seed)
    Q, Qroot = exact_Q(R)
    for (a, pre), q in Q.items():
        q += eps * rng.normal(size=q.shape)
        if pre == ():
            q += bias * np.array([1.0 if c[0] == (0 if a == 0 else 2) else 0.0 for c in children(pre)])
    return Q, Qroot + eps * rng.normal(size=A)


def traj_law(Q, Qroot, C=None):
    """Law of the rollout policy over augmented trajectories, shape (A, D).  C = evidence {var: value}: edges
    that contradict it are masked (a hard rule on edges) and the policy renormalised."""
    P = np.zeros((A, D))
    pa = softmax(Qroot)
    for a in range(A):
        node_p = {(): pa[a]}
        for pre in PREFIXES:
            ch = children(pre)
            q = Q[(a, pre)].copy()
            if C:
                ok = np.array([(i not in C) or (C[i] == v) for (i, v) in ch])
                q = np.where(ok, q, -np.inf)
            pr = softmax(q)
            for c, p in zip(ch, pr):
                if len(pre) + 1 == N:
                    P[a, DESC[pre + (c,)][0]] = node_p[pre] * p
                else:
                    node_p[pre + (c,)] = node_p[pre] * p
    return P


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))


def vmarg(lawD):
    """Marginal over values x of a law over demonstrations (sum over the demonstration's order)."""
    return lawD.reshape(len(PERMS), 8).sum(0)
