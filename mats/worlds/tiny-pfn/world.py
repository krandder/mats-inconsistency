"""TINY reliability world (tiny-pfn, 2026-10-27); exact references in numpy.

Adapted from ../pfn-dedup/world.py (copied, not imported). Per Kelvin's design changes: no duplicate rows, no
shared-error groups. Three INDEPENDENT sources with unknown reliability:
    y_q ~ N(0, 1);  o_{s,q} = y_q + e_{s,q};  e_{s,q} ~ N(0, v_s);  v_s iid uniform on VLEV, fixed within an episode.
Q = 5 questions: 4 labelled (y shown), 1 query (y hidden). The token format is the parent's (no-keys arm).
"""
import itertools, math
import numpy as np

S, Q, QL = 3, 5, 4             # sources, questions, labelled questions (query = last)
VLEV = np.array([0.1, 0.5, 2.0])
NV = len(VLEV)
DS = 16                        # default identity-embedding dim (parent token format); tiny arms use 4
VBAR = VLEV.mean()             # prior-mean variance, used by reference (a)

COMB = np.array(list(itertools.product(range(NV), repeat=S)), dtype=np.int64)   # (27, 3) level indices
VC = VLEV[COMB]                                                                    # (27, 3) variances
LOGPRIOR = np.full(len(COMB), -math.log(len(COMB)))
PREC = 1.0 + (1.0 / VC).sum(1)                       # posterior precision of y* given v
SIGQ = np.stack([np.diag(v) + 1.0 for v in VC])      # cov of query readings with y* marginalised
SIGQI = np.linalg.inv(SIGQ)
LOGDETQ = np.linalg.slogdet(SIGQ)[1]


def sample_np(n, rng):
    """Fixed-set and training sampler. Returns dict of arrays."""
    vidx = rng.integers(0, NV, (n, S))
    v = VLEV[vidx]
    y = rng.standard_normal((n, Q))
    e = rng.standard_normal((n, S, Q)) * np.sqrt(v)[:, :, None]
    o = y[:, None, :] + e
    return dict(vidx=vidx, v=v, y=y, o=o)


def gls(ostar, v):
    """Posterior N(mean, var) of y* given query readings ostar (n, S) and variances v (n, S) or (S,)."""
    prec = 1.0 + (1.0 / v).sum(-1)
    return (ostar / v).sum(-1) / prec, 1.0 / prec


def labelled_loglik(d):
    """(n, S, NV): log p(labelled residuals of source s | v_s = VLEV[j])."""
    R = d['o'][:, :, :QL] - d['y'][:, None, :QL]
    ss = (R ** 2).sum(2)                                  # (n, S) sufficient statistic per source
    return -0.5 * ss[:, :, None] / VLEV - 0.5 * QL * np.log(2 * np.pi * VLEV)


def references(d):
    """Exact predictives for y_{q*}.
    a     : equal weights, every source at the prior-mean variance VBAR (Gaussian).
    b     : oracle precision weighting with the true v (Gaussian).
    bayes : exact Bayes, mixture over the 27 variance combinations; weights use the labelled residuals AND the
            query readings' marginal (the latter couples the sources).
    lab   : like bayes but the combination weights use the labelled residuals only (factorised over sources).
    qonly : like bayes but the weights use the query readings only (ignores the labelled questions): the best any
            predictor that does not use the labelled questions can do.
    Returns (comp, nll, moments, w): comp[k] = (mean, var) Gaussian or (means(n,27), vars(27), w(n,27)) mixtures."""
    n = len(d['y'])
    ostar, ystar = d['o'][:, :, QL], d['y'][:, QL]
    out = {}
    out['a'] = gls(ostar, np.full(S, VBAR))
    out['b'] = gls(ostar, d['v'])
    ll = labelled_loglik(d)                                           # (n, S, NV)
    llc = sum(ll[:, s, COMB[:, s]] for s in range(S))                 # (n, 27)
    quadq = np.einsum('ns,pst,nt->np', ostar, SIGQI, ostar)
    logq = -0.5 * quadq - 0.5 * LOGDETQ[None] - 0.5 * S * math.log(2 * math.pi)
    means = np.einsum('ps,ns->np', 1.0 / VC, ostar) / PREC[None]
    vars_ = 1.0 / PREC
    w = {}
    for name, lw in (('bayes', LOGPRIOR[None] + llc + logq), ('lab', LOGPRIOR[None] + llc),
                     ('qonly', LOGPRIOR[None] + logq)):
        lw = lw - lw.max(1, keepdims=True)
        ww = np.exp(lw); w[name] = ww / ww.sum(1, keepdims=True)
        out[name] = (means, vars_, w[name])
    nll, mom = {}, {}
    for k in ('a', 'b'):
        m, v = out[k]
        nll[k] = 0.5 * np.log(2 * np.pi * v) + 0.5 * (ystar - m) ** 2 / v
        mom[k] = (m, v * np.ones(n))
    for k in ('bayes', 'lab', 'qonly'):
        nll[k] = mix_nll(means, vars_, w[k], ystar)
        mom[k] = mix_moments(means, vars_, w[k])
    return out, nll, mom, w['bayes']


def mix_nll(means, vars_, w, y):
    comp = -0.5 * np.log(2 * np.pi * vars_)[None] - 0.5 * (y[:, None] - means) ** 2 / vars_[None]
    mx = comp.max(1, keepdims=True)
    return -(mx[:, 0] + np.log((w * np.exp(comp - mx)).sum(1)))


def mix_moments(means, vars_, w):
    m = (w * means).sum(1)
    return m, (w * (vars_[None] + means ** 2)).sum(1) - m ** 2


def mix_entropy(means, vars_, w, ngrid=1201, chunk=2000):
    """Differential entropy of each Gaussian mixture by quadrature on a per-episode grid (+-9 sd of the
    widest component around the extreme component means)."""
    n = len(w); H = np.empty(n)
    sd = np.sqrt(vars_)
    for i in range(0, n, chunk):
        mu, ww = means[i:i + chunk], w[i:i + chunk]
        lo = (mu - 9 * sd[None]).min(1); hi = (mu + 9 * sd[None]).max(1)
        t = np.linspace(0, 1, ngrid)
        g = lo[:, None] + (hi - lo)[:, None] * t[None]                        # (c, G)
        dens = (ww[:, None, :] * np.exp(-0.5 * (g[:, :, None] - mu[:, None, :]) ** 2 / vars_)
                / np.sqrt(2 * np.pi * vars_)).sum(2)                         # (c, G)
        f = -dens * np.log(np.maximum(dens, 1e-300))
        H[i:i + chunk] = np.trapezoid(f, g, axis=1)
    return H


def source_posterior(d):
    """(n, S, NV) per-source posterior over variance levels from the labelled residuals only (factorised)."""
    ll = labelled_loglik(d)
    ll = ll - ll.max(2, keepdims=True)
    p = np.exp(ll); return p / p.sum(2, keepdims=True)


def vtype(d):
    """0 = all three variances equal, 1 = exactly two equal, 2 = all different."""
    return np.array([len(set(r)) - 1 for r in d['vidx']])


VTYPE_NAMES = ('all_equal', 'two_equal', 'all_different')


def tokens_np(d, rng, ds=DS):
    """Token features (n, T, F) and validity masks; parent layout, no-keys arm; identity vectors in R^ds."""
    n = len(d['y'])
    semb = rng.standard_normal((n, S, ds)) / 4
    qemb = rng.standard_normal((n, Q, ds)) / 4
    return assemble(semb, qemb, d['o'], d['y'])


def assemble(semb, qemb, o, y):
    """Feature layout (parent): [source id (DS), question id (DS), value o, label y (0 if missing), missing flag,
    query-token flag]. One token per (source, question), row-major by source; the last token is the query token
    [question id of q*, query flag]."""
    n = len(o)
    DS = semb.shape[-1]
    labd = np.concatenate([y[:, :QL], np.zeros((n, Q - QL))], 1)
    miss = np.concatenate([np.zeros((n, QL)), np.ones((n, Q - QL))], 1)
    F = 2 * DS + 4
    T = S * Q + 1
    X = np.zeros((n, T, F), dtype=np.float32)
    for r in range(S):
        for q in range(Q):
            t = r * Q + q
            X[:, t, :DS] = semb[:, r]
            X[:, t, DS:2 * DS] = qemb[:, q]
            X[:, t, 2 * DS] = o[:, r, q]
            X[:, t, 2 * DS + 1] = labd[:, q]
            X[:, t, 2 * DS + 2] = miss[:, q]
    X[:, T - 1, DS:2 * DS] = qemb[:, QL]
    X[:, T - 1, 2 * DS + 3] = 1.0
    m = np.ones((n, T), dtype=bool)
    return X, m
