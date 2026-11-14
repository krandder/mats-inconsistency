"""Independent brute-force references for tests (numpy/scipy, one episode at a time, small N).

(a) enumerate every hypothesis (set partition with block sizes <= m, block levels, signs) -> exact Gaussian mixture.
(b) importance sampling over Sigma ~ IW from the prior; and direct v-grid quadrature of N(v) x multivariate t.
"""
import itertools, math
import numpy as np
from scipy import stats
from scipy.special import logsumexp


def set_partitions(items, m):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for k in range(0, min(m - 1, len(rest)) + 1):
        for comb in itertools.combinations(rest, k):
            left = [x for x in rest if x not in comb]
            for p in set_partitions(left, m):
                yield [(first,) + comb] + p


def _probs(p, n):
    p = np.ones(n) if p is None else np.asarray(p, float)
    return p / p.sum()


def block_levels(s, cfg):
    pf = _probs(cfg['a_pf'], len(cfg['a_f']))
    pmb = cfg['a_pm'] if cfg.get('a_pm_block') is None else cfg['a_pm_block']
    pm = _probs(cfg['a_pm'] if s == 1 else pmb, len(cfg['a_m']))
    out = []
    if s == 1:
        for i, m in enumerate(cfg['a_m']):
            out.append((np.array([[m]]), math.log(pm[i])))
        return out
    ps = cfg['a_psign']
    signs = [(1.0,) * s] if ps == 0 else list(itertools.product((1.0, -1.0), repeat=s))
    for i, m in enumerate(cfg['a_m']):
        for j, f in enumerate(cfg['a_f']):
            for sg in signs:
                sg = np.array(sg)
                S = m * ((1 - f) * np.eye(s) + f * np.outer(sg, sg))
                lp = math.log(pm[i]) + math.log(pf[j])
                if ps > 0:
                    nn = (sg < 0).sum(); lp += nn * math.log(ps) + (s - nn) * math.log(1 - ps)
                out.append((S, lp))
    return out


def brute_a(eh, r, cfg):
    """eh (T, N) history errors, r (N,) query reports. Returns mixture (w, m, v) arrays and log evidence."""
    T, N = eh.shape
    alpha = cfg['a_alpha']; m = len(alpha)
    parts = list(set_partitions(list(range(N)), m))
    logZ = logsumexp([sum(math.log(alpha[len(B) - 1]) for B in P) for P in parts])
    W, Mu, V = [], [], []
    for P in parts:
        lpP = sum(math.log(alpha[len(B) - 1]) for B in P) - logZ
        for combo in itertools.product(*[block_levels(len(B), cfg) for B in P]):
            Sig = np.zeros((N, N)); lp = lpP
            for B, (SB, l) in zip(P, combo):
                Sig[np.ix_(B, B)] = SB; lp += l
            lh = stats.multivariate_normal(np.zeros(N), Sig).logpdf(eh).sum() if T > 0 else 0.0
            lq = stats.multivariate_normal(np.zeros(N), Sig + 1.0).logpdf(r)
            Pi = np.linalg.inv(Sig); a = Pi.sum(); bb = (Pi @ r).sum()
            W.append(lp + lh + lq); Mu.append(bb / (1 + a)); V.append(1 / (1 + a))
    W = np.array(W); lev = logsumexp(W)
    return np.exp(W - lev), np.array(Mu), np.array(V), lev


def is_b(eh, r, cfg, M=200000, seed=0):
    """Importance sampling (proposal = prior) for prior (b), single hyperparameter or mixture."""
    rng = np.random.default_rng(seed)
    T, N = eh.shape
    hyp = cfg['b_hyper']; pw = np.array([h[3] for h in hyp]); pw = pw / pw.sum()
    hs = rng.choice(len(hyp), M, p=pw)
    lw = np.empty(M); mu = np.empty(M); var = np.empty(M)
    for hi, (delta, sig2, rho, _) in enumerate(hyp):
        idx = np.flatnonzero(hs == hi)
        if len(idx) == 0: continue
        Psi = delta * sig2 * ((1 - rho) * np.eye(N) + rho)
        Sigs = stats.invwishart(df=N + 1 + delta, scale=Psi).rvs(len(idx), random_state=rng).reshape(len(idx), N, N)
        L = np.linalg.cholesky(Sigs)
        ld = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
        Pi = np.linalg.inv(Sigs)
        lh = -0.5 * T * (N * math.log(2 * math.pi) + ld) - 0.5 * np.einsum('ti,kij,tj->k', eh, Pi, eh)
        C = Sigs + 1.0
        Lc = np.linalg.cholesky(C)
        z = np.linalg.solve(Lc, np.broadcast_to(r, (len(idx), N))[..., None])[..., 0]
        lq = -0.5 * (N * math.log(2 * math.pi) + 2 * np.log(np.diagonal(Lc, axis1=1, axis2=2)).sum(1)) - 0.5 * (z ** 2).sum(1)
        a = Pi.sum((1, 2)); bb = (Pi @ r).sum(1)
        lw[idx] = lh + lq; mu[idx] = bb / (1 + a); var[idx] = 1 / (1 + a)
    w = np.exp(lw - lw.max()); w /= w.sum()
    m1 = (w * mu).sum(); v1 = (w * (var + mu ** 2)).sum() - m1 ** 2
    ess = 1 / (w ** 2).sum()
    return m1, v1, ess, logsumexp(lw) - math.log(M)


def grid_b(eh, r, cfg, G=20001, half=12.0):
    """Direct quadrature over v of N(v; 0, 1) x multivariate-t(r - v 1) (posterior on a fine grid), one hyper."""
    assert len(cfg['b_hyper']) == 1
    delta, sig2, rho, _ = cfg['b_hyper'][0]
    T, N = eh.shape
    Psi = delta * sig2 * ((1 - rho) * np.eye(N) + rho) + eh.T @ eh
    nup = N + 1 + delta + T
    kap = nup - N + 1
    dist = stats.multivariate_t(loc=np.zeros(N), shape=Psi / kap, df=kap)
    vs = np.linspace(-half, half, G)
    lp = stats.norm.logpdf(vs) + dist.logpdf(r[None, :] - vs[:, None])
    dx = vs[1] - vs[0]
    lev = logsumexp(lp) + math.log(dx)
    return vs, lp - lev
