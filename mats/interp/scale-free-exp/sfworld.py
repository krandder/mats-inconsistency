"""Scale-free experiments (2026-10-29): the tiny-pfn reliability world generalised to N sources, plus shared-error
coupling, with exact Bayes on an adaptive y* grid.

Copied in spirit from work/tiny-pfn/world.py (not imported, not edited). Per episode:
    y_q ~ N(0, 1), q = 1..Q (Q = 5: 4 labelled, 1 query);  v_s iid uniform on VLEV = {0.1, 0.5, 2.0};
    o_{s,q} = y_q + e_{s,q},  e_{s,q} = sqrt(v_s) * (sqrt(rho_b) u_{b(s),q} + sqrt(1 - rho_b) eps_{s,q}).
Modes (block structure b(s)):
    indep  : every source its own block (rho irrelevant). N = 3 equals tiny-pfn's world exactly.
    groups : KNOWN contiguous groups of size gsize (group ids are shown to the nets), coupling rho inside a group.
    pairs  : UNKNOWN random partition into singles and pairs, p(partition) prop. to a1^#singles a2^#pairs; a pair
             shares an error factor with coupling rho. Nets are not told the partition.
Exact Bayes: p(y* | data) prop. to N(y*; 0, 1) * prod_blocks L_b(y*) (indep, groups) or a DP over partitions (pairs),
evaluated on a two-pass adaptive grid (coarse 281 points on [-7, 7], then a fine grid on the region within 30 nats
of the coarse maximum). Normalisation by the trapezoid rule.
"""
import itertools, math
import numpy as np
import torch

Q, QL = 5, 4
VLEV = np.array([0.1, 0.5, 2.0])
NV = 3
LOG2PI = math.log(2 * math.pi)


# ----------------------------------------------------------------------------------------------- sampling
def pair_counts(N, a1, a2):
    """f[n] = sum over partitions of n items into singles/pairs of a1^#singles a2^#pairs."""
    f = np.zeros(N + 1); f[0] = 1.0
    if N >= 1: f[1] = a1
    for n in range(2, N + 1):
        f[n] = a1 * f[n - 1] + (n - 1) * a2 * f[n - 2]
    return f


def sample_pairs_partition(n, N, rng, a1, a2):
    """Block labels (n, N): block id = index of the block's first member. Exact sampling from the partition prior."""
    f = pair_counts(N, a1, a2)
    lab = -np.ones((n, N), dtype=np.int64)
    for e in range(n):
        free = list(range(N))
        while free:
            i = free.pop(0); r = len(free) + 1
            p_single = a1 * f[r - 1] / f[r]
            if rng.random() < p_single or not free:
                lab[e, i] = i
            else:
                t = free.pop(rng.integers(len(free)))
                lab[e, i] = i; lab[e, t] = i
    return lab


def sample(n, N, rng, mode='indep', rho=0.0, gsize=1, a1=1.0, a2=1.0):
    """numpy sampler. Returns dict: y (n,Q), o (n,N,Q), v (n,N), vidx, blk (n,N) block labels, grp (n,N) shown
    group ids (groups mode; else zeros)."""
    vidx = rng.integers(0, NV, (n, N)); v = VLEV[vidx]
    y = rng.standard_normal((n, Q))
    eps = rng.standard_normal((n, N, Q))
    if mode == 'indep':
        blk = np.tile(np.arange(N), (n, 1)); r = 0.0
    elif mode == 'groups':
        assert N % gsize == 0
        blk = np.tile(np.arange(N) // gsize, (n, 1)); r = rho
    elif mode == 'pairs':
        blk = sample_pairs_partition(n, N, rng, a1, a2); r = rho
    else:
        raise ValueError(mode)
    u = rng.standard_normal((n, N, Q))                                     # one factor per possible block id
    shared = np.take_along_axis(u, blk[:, :, None].repeat(Q, 2), 1)
    e = np.sqrt(v)[:, :, None] * (math.sqrt(r) * shared + math.sqrt(1 - r) * eps)
    o = y[:, None, :] + e
    grp = blk if mode == 'groups' else np.zeros_like(blk)
    return dict(y=y, o=o, v=v, vidx=vidx, blk=blk, grp=grp, N=N, mode=mode, rho=rho, gsize=gsize, a1=a1, a2=a2)


# ----------------------------------------------------------------------------------------------- block likelihoods
def _single_ll(R, ostar, yg):
    """log L_s(y) for independent sources. R (n,N,QL), ostar (n,N), yg (n,G) -> (n,N,G)."""
    ss = (R ** 2).sum(-1)                                                  # (n, N)
    out = None
    for j, vj in enumerate(VLEV):
        t = (-math.log(NV) - 0.5 * QL * (LOG2PI + math.log(vj)) - ss[:, :, None] / (2 * vj)
             - 0.5 * (LOG2PI + math.log(vj)) - (ostar[:, :, None] - yg[:, None, :]) ** 2 / (2 * vj))
        out = t if out is None else np.logaddexp(out, t)
    return out


def _block_cov_tables(k, rho):
    """All 3^k variance configs for a block of k sources with coupling rho: inverse covariances, log dets, log prior."""
    cfg = np.array(list(itertools.product(range(NV), repeat=k)), dtype=np.int64)  # (C, k)
    vv = VLEV[cfg]
    C = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
    sd = np.sqrt(vv)
    Sig = sd[:, :, None] * C[None] * sd[:, None, :]
    Si = np.linalg.inv(Sig)
    ld = np.linalg.slogdet(Sig)[1]
    return cfg, Si, ld, np.full(len(cfg), -k * math.log(NV))


def _group_ll(Rg, og, yg, rho):
    """log L_g(y) for one block of k jointly coupled sources. Rg (n,k,QL), og (n,k), yg (n,G) -> (n,G)."""
    k = Rg.shape[1]
    cfg, Si, ld, lp = _block_cov_tables(k, rho)
    S = np.einsum('nsq,ntq->nst', Rg, Rg)                                  # scatter over labelled questions
    quadR = np.einsum('cst,nst->nc', Si, S)                                # sum_q R_q' Si R_q
    one = np.ones(k)
    a = np.einsum('s,cst,t->c', one, Si, one)                              # (C,)
    b = np.einsum('s,cst,nt->nc', one, Si, og)                             # (n, C)
    cc = np.einsum('ns,cst,nt->nc', og, Si, og)
    base = lp[None] - 0.5 * quadR - 0.5 * QL * (k * LOG2PI + ld[None]) - 0.5 * (k * LOG2PI + ld[None]) - 0.5 * cc
    # + b y - a y^2 / 2 ; logsumexp over configs, chunked over grid to bound memory
    out = np.empty(yg.shape)
    G = yg.shape[1]
    for g0 in range(0, G, 128):
        yy = yg[:, g0:g0 + 128]
        t = base[:, :, None] + b[:, :, None] * yy[:, None, :] - 0.5 * a[None, :, None] * yy[:, None, :] ** 2
        mx = t.max(1)
        out[:, g0:g0 + 128] = mx + np.log(np.exp(t - mx[:, None]).sum(1))
    return out


def _pair_ll_all(R, ostar, yg, rho):
    """log L_{it}(y) for every ordered pair i<t (pairs mode). Returns dict (i,t) -> (n,G)."""
    N = R.shape[1]
    out = {}
    for i in range(N):
        for t in range(i + 1, N):
            idx = [i, t]
            out[(i, t)] = _group_ll(R[:, idx], ostar[:, idx], yg, rho)
    return out


def log_lik_total(d, yg):
    """log p(data | y* = grid) up to y-independent constants that are the same for every grid point
    (the full joint evidence is kept, so values are comparable across grid points). yg (n,G) -> (n,G)."""
    R = d['o'][:, :, :QL] - d['y'][:, None, :QL]
    ostar = d['o'][:, :, QL]
    mode, N = d['mode'], d['N']
    if mode == 'indep' or (mode == 'groups' and (d['rho'] == 0 or d['gsize'] == 1)):
        return _single_ll(R, ostar, yg).sum(1)
    if mode == 'groups':
        k = d['gsize']; tot = 0.0
        for g in range(N // k):
            sl = slice(g * k, (g + 1) * k)
            tot = tot + _group_ll(R[:, sl], ostar[:, sl], yg, d['rho'])
        return tot
    if mode == 'pairs':
        return _pairs_dp(R, ostar, yg, d['rho'], d.get('a1', 1.0), d.get('a2', 1.0))
    raise ValueError(mode)


def _pairs_dp(R, ostar, yg, rho, a1, a2):
    """log sum_P prior(P) prod_blocks L_b(y) over partitions into singles and pairs, by a DP over subsets."""
    N = R.shape[1]
    Ls = _single_ll(R, ostar, yg)                                          # (n, N, G)
    Lp = _pair_ll_all(R, ostar, yg, rho)
    la1, la2 = math.log(a1), math.log(a2)
    full = (1 << N) - 1
    f = {0: np.zeros(yg.shape)}
    for S in range(1, full + 1):
        i = (S & -S).bit_length() - 1                                      # lowest member
        rest = S & ~(1 << i)
        terms = [la1 + Ls[:, i] + f[rest]]
        t = rest
        while t:
            j = (t & -t).bit_length() - 1
            terms.append(la2 + Lp[(i, j)] + f[rest & ~(1 << j)])
            t &= t - 1
        st = np.stack(terms, 0); mx = st.max(0)
        f[S] = mx + np.log(np.exp(st - mx[None]).sum(0))
        # free memory of subsets no longer needed is not worth the bookkeeping at N <= 10
    logZ = math.log(pair_counts(N, a1, a2)[N])
    return f[full] - logZ


# ----------------------------------------------------------------------------------------------- exact Bayes
def bayes(d, chunk=None, ncoarse=281, nfine=401, thresh=30.0):
    """Exact posterior of y* on an adaptive grid. Returns dict: grid (n,G), logp (n,G) normalised log density,
    nll (n,) at the true y*, mean, var."""
    n = len(d['y'])
    grids, logps, nll, mean, var = [], [], np.empty(n), np.empty(n), np.empty(n)
    chunk = chunk or int(min(512, max(16, 12000 // d["N"])))
    ystar = d['y'][:, QL]
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(n, i0 + chunk))
        dd = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in d.items()}
        m = sl.stop - sl.start
        yc = np.tile(np.linspace(-7, 7, ncoarse), (m, 1))
        lc = -0.5 * yc ** 2 + log_lik_total(dd, yc)
        keep = lc > lc.max(1, keepdims=True) - thresh
        h = yc[0, 1] - yc[0, 0]
        lo = np.where(keep, yc, np.inf).min(1) - 2 * h
        hi = np.where(keep, yc, -np.inf).max(1) + 2 * h
        lo = np.minimum(lo, ystar[sl] - 1e-9); hi = np.maximum(hi, ystar[sl] + 1e-9)   # grid must contain y*
        yf = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, nfine)[None]
        lf = -0.5 * yf ** 2 + log_lik_total(dd, yf)
        mx = lf.max(1, keepdims=True)
        Z = np.trapezoid(np.exp(lf - mx), yf, axis=1)
        logp = lf - mx - np.log(Z)[:, None]
        p = np.exp(logp)
        mu = np.trapezoid(p * yf, yf, axis=1)
        var[sl] = np.trapezoid(p * (yf - mu[:, None]) ** 2, yf, axis=1); mean[sl] = mu
        # NLL at the true y*: evaluate the unnormalised log density exactly at y*
        lstar = -0.5 * ystar[sl] ** 2 + log_lik_total(dd, ystar[sl][:, None])[:, 0]
        nll[sl] = -(lstar - mx[:, 0] - np.log(Z))          # the prior's -0.5 log 2pi cancels with Z's
        grids.append(yf); logps.append(logp)
    grid = np.concatenate(grids); logp = np.concatenate(logps)
    return dict(grid=grid, logp=logp, nll=nll, mean=mean, var=var)


# ----------------------------------------------------------------------------------------------- reference rules
def indep_bayes(d, **kw):
    """Bayes that assumes every source independent (rho = 0), applied to the same data: the 'coupling-blind' rule."""
    dd = dict(d); dd['mode'] = 'indep'
    return bayes(dd, **kw)


def mix_logpdf(grid, mu, logstd, logits):
    """Log density of a K-mixture of Gaussians on per-episode grids. grid (n,G), params (n,K) -> (n,G)."""
    lw = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    t = (lw[:, None, :] - logstd[:, None, :] - 0.5 * LOG2PI
         - 0.5 * ((grid[:, :, None] - mu[:, None, :]) / np.exp(logstd)[:, None, :]) ** 2)
    return np.logaddexp.reduce(t, axis=2)


def kl_bayes_model(B, mu, logstd, logits):
    """KL(Bayes || model mixture) per episode on the Bayes grid."""
    lq = mix_logpdf(B['grid'], mu, logstd, logits)
    p = np.exp(B['logp'])
    return np.trapezoid(p * (B['logp'] - lq), B['grid'], axis=1)


# ----------------------------------------------------------------------------------------------- features
def src_features(d):
    """Per-source feature rows for tree / one-token-per-source nets: [o_{s,1..4}, y_{1..4}, o*_s] (n, N, 9)."""
    n, N = d['o'].shape[:2]
    yl = np.repeat(d['y'][:, None, :QL], N, 1)
    return np.concatenate([d['o'][:, :, :QL], yl, d['o'][:, :, QL:QL + 1]], 2).astype(np.float32)
