"""Linear-Gaussian source world with shared errors and duplicates; exact references in numpy.

o_{s,q} = y_q + z_{g(s),q} + e_{s,q};  y ~ N(0,1), z ~ N(0, SG2) per group and question, e ~ N(0, SE2).
"""
import itertools, math, os
import numpy as np

S, Q, QL = 6, 9, 8            # sources, questions, labelled questions (query = last)
SG2, SE2 = 1.0, 0.25
P_SINGLETON, P_DUP = 0.3, 0.3
MAXDUP = 3
# Track A2 variant (PFN_WORLD=costly): copies are common and repeated. With prob 0.8 an episode copies k ~ U{1,2,3}
# distinct sources, each c ~ U{1,2,3} times (up to 4 rows per source); MAXDUP is then the number of extra-row slots.
COSTLY = os.environ.get('PFN_WORLD', '') == 'costly'
if COSTLY:
    P_DUP, NCOP, CMAX = 0.8, 3, 3
    MAXDUP = NCOP * CMAX
DS = 16                        # identity-embedding dim


def all_partitions(n):
    """Canonical labels (first appearance order) for every set partition of range(n)."""
    out = []
    def rec(i, labels, m):
        if i == n:
            out.append(tuple(labels)); return
        for g in range(m + 1):
            rec(i + 1, labels + [g], max(m, g + 1))
    rec(0, [], 0)
    return np.array(out, dtype=np.int64)


PARTS = all_partitions(S)      # (203, 6)


def ewens(labels, alpha=1.0):
    n = len(labels)
    sizes = np.bincount(labels)
    num = alpha ** len(sizes) * np.prod([math.factorial(m - 1) for m in sizes])
    den = np.prod([alpha + i for i in range(n)])
    return num / den


_E = np.array([ewens(p) for p in PARTS])
_SING = np.array([len(set(p)) == S for p in PARTS])
PRIOR = 0.7 * _E / _E.sum() + 0.3 * _SING
PRIOR = PRIOR / PRIOR.sum()


def sigma_of(labels):
    B = (labels[:, None] == labels[None, :]).astype(float)
    return SE2 * np.eye(len(labels)) + SG2 * B


SIG = np.stack([sigma_of(p) for p in PARTS])            # (203, 6, 6)
SIGI = np.linalg.inv(SIG)
LOGDET = np.linalg.slogdet(SIG)[1]
SIGQ = SIG + 1.0                                        # cov of query readings (y* marginalised)
SIGQI = np.linalg.inv(SIGQ)
LOGDETQ = np.linalg.slogdet(SIGQ)[1]
ONE = np.ones(S)
PREC = 1.0 + np.einsum('i,pij,j->p', ONE, SIGI, ONE)    # posterior precision of y* given pi
WVEC = np.einsum('pij,j->pi', SIGI, ONE)                # (203, 6): mean = WVEC . o* / PREC


def sample_np(n, rng, keys=False):
    """Numpy sampler (used for the fixed val/test sets). Returns dict of arrays."""
    pidx = rng.choice(len(PARTS), size=n, p=PRIOR)
    lab = PARTS[pidx]                                   # (n, 6)
    y = rng.standard_normal((n, Q))
    z = rng.standard_normal((n, S, Q)) * math.sqrt(SG2)
    e = rng.standard_normal((n, S, Q)) * math.sqrt(SE2)
    o = y[:, None, :] + np.take_along_axis(z, lab[:, :, None].repeat(Q, 2), 1) + e
    if COSTLY:
        ns = np.where(rng.random(n) < P_DUP, rng.integers(1, NCOP + 1, n), 0)
        c = rng.integers(1, CMAX + 1, (n, NCOP))
        perm = np.argsort(rng.random((n, S)), 1)
        slot = np.arange(MAXDUP)
        dvalid = (slot[None, :] // CMAX < ns[:, None]) & (slot[None, :] % CMAX < c[:, slot // CMAX])
        dsrc = perm[:, slot // CMAX]
        return dict(pidx=pidx, lab=lab, y=y, o=o, k=dvalid.sum(1), perm=perm, dsrc=dsrc, dvalid=dvalid)
    k = np.where(rng.random(n) < P_DUP, rng.integers(1, MAXDUP + 1, n), 0)
    perm = np.argsort(rng.random((n, S)), 1)
    return dict(pidx=pidx, lab=lab, y=y, o=o, k=k, perm=perm)


def rows_of(d):
    """Row -> source index (n, S+MAXDUP) and validity mask."""
    n = len(d['k'])
    if 'dsrc' in d:
        return (np.concatenate([np.tile(np.arange(S), (n, 1)), d['dsrc']], 1),
                np.concatenate([np.ones((n, S), bool), d['dvalid']], 1))
    src = np.concatenate([np.tile(np.arange(S), (n, 1)), d['perm'][:, :MAXDUP]], 1)
    valid = np.concatenate([np.ones((n, S), bool), np.arange(MAXDUP)[None, :] < d['k'][:, None]], 1)
    return src, valid


def references(d):
    """Exact predictive (mean, var) for y_{q*} and NLL at the true y*, for (a), (b), (c), (d).
    (d) returns the mixture; nll uses the mixture density."""
    n = len(d['k'])
    o, y, lab = d['o'], d['y'], d['lab']
    ostar = o[:, :, QL]                                  # (n, 6) distinct-source query readings
    ystar = y[:, QL]
    s2 = SG2 + SE2
    out = {}
    # (a) count all rows (copies included)
    src, valid = rows_of(d)
    rows_o = np.take_along_axis(ostar, src, 1) * valid
    cnt = valid.sum(1)
    prec = 1 + cnt / s2; mean = rows_o.sum(1) / s2 / prec
    out['a'] = (mean, 1 / prec)
    # (b) distinct sources iid
    prec = 1 + S / s2; mean = ostar.sum(1) / s2 / prec
    out['b'] = (mean, np.full(n, 1 / prec))
    # (c) GLS oracle with true partition
    mean = np.einsum('ni,ni->n', WVEC[d['pidx']], ostar) / PREC[d['pidx']]
    out['c'] = (mean, 1 / PREC[d['pidx']])
    # (d) exact Bayes over partitions
    R = o[:, :, :QL] - y[:, None, :QL]                   # (n, 6, 8) residuals of labelled questions
    quad = np.einsum('nsq,pst,ntq->np', R, SIGI, R)      # sum over q of r^T S^-1 r
    loglab = -0.5 * quad - 0.5 * QL * LOGDET[None, :] - 0.5 * QL * S * math.log(2 * math.pi)
    quadq = np.einsum('ns,pst,nt->np', ostar, SIGQI, ostar)
    logq = -0.5 * quadq - 0.5 * LOGDETQ[None, :] - 0.5 * S * math.log(2 * math.pi)
    lw = np.log(PRIOR)[None, :] + loglab + logq
    lw -= lw.max(1, keepdims=True)
    w = np.exp(lw); w /= w.sum(1, keepdims=True)         # (n, 203) posterior over partitions
    means = np.einsum('pi,ni->np', WVEC, ostar) / PREC[None, :]
    vars_ = (1 / PREC)[None, :].repeat(n, 0)
    out['d'] = (means, vars_, w)
    # NLLs
    nll = {}
    for kk in 'abc':
        m, v = out[kk]
        nll[kk] = 0.5 * np.log(2 * np.pi * v) + 0.5 * (ystar - m) ** 2 / v
    comp = -0.5 * np.log(2 * np.pi * vars_) - 0.5 * (ystar[:, None] - means) ** 2 / vars_
    mx = comp.max(1, keepdims=True)
    nll['d'] = -(mx[:, 0] + np.log((w * np.exp(comp - mx)).sum(1)))
    mean_d = (w * means).sum(1)
    var_d = (w * (vars_ + means ** 2)).sum(1) - mean_d ** 2
    return out, nll, (mean_d, var_d), w


def redundancy_type(d):
    sizes = np.array([np.bincount(l).max() for l in d['lab']])
    has_dup = d['k'] > 0
    return sizes, has_dup


def tokens_np(d, rng, keys=False):
    """Build token features (n, T, F) and masks for a fixed set, matching the torch sampler layout."""
    n = len(d['k'])
    src, valid = rows_of(d)
    R = S + MAXDUP
    semb = rng.standard_normal((n, S, DS)) / 4
    qemb = rng.standard_normal((n, Q, DS)) / 4
    gemb = rng.standard_normal((n, S, DS)) / 4
    return assemble(semb, qemb, gemb, src, valid, d['o'], d['y'], d['lab'], keys)


def assemble(semb, qemb, gemb, src, valid, o, y, lab, keys):
    n = len(src); R = src.shape[1]
    xp = np
    se = xp.take_along_axis(semb, src[:, :, None].repeat(DS, 2), 1)          # (n, R, DS)
    ge = xp.take_along_axis(gemb, xp.take_along_axis(lab, src, 1)[:, :, None].repeat(DS, 2), 1)
    ov = xp.take_along_axis(o, src[:, :, None].repeat(Q, 2), 1)              # (n, R, Q)
    labd = xp.concatenate([y[:, :QL], xp.zeros((n, 1))], 1)                  # (n, Q)
    miss = xp.concatenate([xp.zeros((n, QL)), xp.ones((n, 1))], 1)
    F = DS * (3 if keys else 2) + 4
    T = R * Q + 1
    X = xp.zeros((n, T, F), dtype=np.float32)
    m = xp.zeros((n, T), dtype=bool)
    for r in range(R):
        for q in range(Q):
            t = r * Q + q
            X[:, t, :DS] = se[:, r]
            X[:, t, DS:2 * DS] = qemb[:, q]
            X[:, t, 2 * DS] = ov[:, r, q]
            X[:, t, 2 * DS + 1] = labd[:, q]
            X[:, t, 2 * DS + 2] = miss[:, q]
            if keys:
                X[:, t, 2 * DS + 4:] = ge[:, r]
            m[:, t] = valid[:, r]
    X[:, T - 1, DS:2 * DS] = qemb[:, QL]
    X[:, T - 1, 2 * DS + 3] = 1.0
    m[:, T - 1] = True
    return X, m
