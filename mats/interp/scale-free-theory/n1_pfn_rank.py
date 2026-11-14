"""N1 (scale-free-theory, 2026-10-29): additive messages for the tiny-PFN reliability world, scaled to N sources.

World (work/tiny-pfn/world.py): y_q ~ N(0,1); o_{s,q} = y_q + e_{s,q}, e ~ N(0, v_s), v_s iid uniform on VLEV;
QL = 4 labelled questions (residuals observed), 1 query (y* hidden). Given y*, sources are independent, so
    log p(y* | data) = log N(y*; 0, 1) + sum_s l(y*; o_s, ss_s) + const,
    l(y; o, ss) = log sum_j (1/3) N(o; y, v_j) (2 pi v_j)^(-QL/2) exp(-ss / (2 v_j)).
Claim C3: a FIXED linear encoder phi(o, ss) in R^r (projection of l onto an r-dim function basis), summed over
sources, gives the posterior to accuracy delta with r growing only like polylog(N / delta).

Method: represent each l(.; o, ss) by Chebyshev coefficients on [-Y, Y] (degree D, checked); learn the basis by SVD
of sampled coefficient vectors under the uniform-L2 Gram matrix; then for N sources compare the exact posterior
with the posterior from the summed r-dim messages (KL and mean error in posterior sd), on a fine local grid.
"""
import json, math, sys, time
import numpy as np
from numpy.polynomial import chebyshev as C

VLEV = np.array([0.1, 0.5, 2.0]); QL = 4
Y = 6.0; D = 400


def ell_grid(y, o, ss):
    """y: (K,), o, ss: (M,) -> (M, K)."""
    out = np.empty((len(o), len(y)))
    terms = []
    for v in VLEV:
        t = (-math.log(3.0) - 0.5 * math.log(2 * math.pi * v) - 0.5 * (o[:, None] - y[None, :]) ** 2 / v
             - 0.5 * QL * math.log(2 * math.pi * v) - 0.5 * ss[:, None] / v)
        terms.append(t)
    T = np.stack(terms, 0)
    mx = T.max(0)
    out[:] = mx + np.log(np.exp(T - mx).sum(0))
    return out


def sample_sources(n, rng):
    v = VLEV[rng.integers(0, 3, n)]
    ystar = rng.standard_normal(n)
    o = ystar + rng.standard_normal(n) * np.sqrt(v)
    ss = v * rng.chisquare(QL, n)
    return o, ss, v, ystar


def cheb_coeffs(o, ss):
    """Chebyshev coefficients (M, D+1) of l(Y x; o, ss) on x in [-1, 1] via interpolation at Chebyshev points."""
    x = np.cos(np.pi * (np.arange(D + 1) + 0.5) / (D + 1))          # Chebyshev points of the first kind
    vals = ell_grid(Y * x, o, ss)                                      # (M, D+1)
    # discrete cosine transform for coefficients
    k = np.arange(D + 1)
    Tm = np.cos(np.outer(k, np.pi * (np.arange(D + 1) + 0.5) / (D + 1)))   # (D+1, D+1)
    c = vals @ Tm.T * (2.0 / (D + 1)); c[:, 0] *= 0.5
    return c


def gram_uniform(n):
    """G_jk = (1/2) int_{-1}^{1} T_j T_k dx (uniform L2 on [-1,1], normalised to mean)."""
    G = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            if (j + k) % 2 == 0:
                a, b = j + k, j - k
                G[j, k] = 0.5 * ((1.0 / (1 - a * a)) + (1.0 / (1 - b * b)))
    return G


def main():
    rng = np.random.default_rng(20261029)
    t0 = time.time()
    Mtrain = 6000 if len(sys.argv) < 2 else 800
    o, ss, v, ys = sample_sources(Mtrain, rng)
    Ctr = cheb_coeffs(o, ss)
    tail = np.abs(Ctr[:, -20:]).max()
    G = gram_uniform(D + 1)
    w, U = np.linalg.eigh(G); w = np.clip(w, 1e-300, None)
    Gh = U @ np.diag(np.sqrt(w)) @ U.T                               # G^{1/2}
    Ghi = U @ np.diag(1.0 / np.sqrt(w)) @ U.T
    # remove constants: constants do not change the posterior; project them out (Chebyshev T_0 direction in G metric)
    Z = Ctr @ Gh                                                     # rows in an orthonormal (uniform-L2) frame
    e0 = np.zeros(D + 1); e0[0] = 1.0; u0 = Gh @ e0; u0 /= np.linalg.norm(u0)
    Z = Z - np.outer(Z @ u0, u0)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    res = {'D': D, 'Y': Y, 'cheb_tail_max_abs': float(tail), 'Mtrain': Mtrain,
           'sv_rel': (S / S[0])[:60].tolist()}

    # sup-norm error of the rank-r projection on fresh sources, on |y| <= 5 (constants free)
    o2, ss2, v2, ys2 = sample_sources(3000, rng)
    C2 = cheb_coeffs(o2, ss2)
    ygrid = np.linspace(-5, 5, 2001)
    exact = ell_grid(ygrid, o2, ss2)
    Z2 = C2 @ Gh
    Vgrid = C.chebvander(ygrid / Y, D)                               # (K, D+1)
    sup = {}
    for r in [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 30, 40]:
        B = np.concatenate([u0[None], Vt[:r]], 0)                    # basis incl. constant
        P = (Z2 @ B.T) @ B                                           # projection in frame
        Capprox = P @ Ghi
        approx = Capprox @ Vgrid.T                                   # (M, K) via Chebyshev Vandermonde
        err = exact - approx
        err = err - err.mean(1, keepdims=True)                       # constants are free
        sup[r] = {'sup_max': float(np.abs(err).max()), 'sup_q99': float(np.quantile(np.abs(err).max(1), 0.99)),
                  'sup_median': float(np.median(np.abs(err).max(1)))}
    res['sup_err'] = sup

    # N-source aggregation: exact vs summed r-dim messages
    agg = {}
    Ns = [3, 8, 24, 64, 256, 1024] if len(sys.argv) < 2 else [3, 64]
    for N in Ns:
        E = (400 if N <= 64 else 150) if len(sys.argv) < 2 else 20
        recs = {r: {'kl': [], 'dmean_sd': []} for r in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30]}
        for e in range(E):
            ystar = rng.standard_normal()
            vv = VLEV[rng.integers(0, 3, N)]
            oo = ystar + rng.standard_normal(N) * np.sqrt(vv)
            sss = vv * rng.chisquare(QL, N)
            # coarse locate
            yc = np.linspace(-6, 6, 4801)
            lp = -0.5 * yc ** 2 + ell_grid(yc, oo, sss).sum(0)
            m0 = yc[np.argmax(lp)]
            pc = np.exp(lp - lp.max()); pc /= pc.sum()
            sd0 = math.sqrt(max(((yc - (pc * yc).sum()) ** 2 * pc).sum(), 1e-8))
            sd0 = max(sd0, 0.0025)
            yf = np.linspace(m0 - 14 * sd0, m0 + 14 * sd0, 4001)
            lpe = -0.5 * yf ** 2 + ell_grid(yf, oo, sss).sum(0)
            pe = np.exp(lpe - lpe.max()); pe /= pe.sum()
            me = (pe * yf).sum(); sde = math.sqrt(((yf - me) ** 2 * pe).sum())
            Cs = cheb_coeffs(oo, sss) @ Gh                             # (N, D+1) frame coords
            for r in recs:
                B = Vt[:r]
                msg = (Cs @ B.T).sum(0)                                # the additive r-dim message
                capp = (msg @ B) @ Ghi
                lpa = -0.5 * yf ** 2 + C.chebval(yf / Y, capp)
                pa = np.exp(lpa - lpa.max()); pa /= pa.sum()
                kl = float((pe * (np.log(np.maximum(pe, 1e-300)) - np.log(np.maximum(pa, 1e-300)))).sum())
                ma = (pa * yf).sum()
                recs[r]['kl'].append(kl); recs[r]['dmean_sd'].append(abs(ma - me) / sde)
        agg[N] = {r: {'kl_mean': float(np.mean(d['kl'])), 'kl_q95': float(np.quantile(d['kl'], 0.95)),
                      'dmean_sd_q95': float(np.quantile(d['dmean_sd'], 0.95))} for r, d in recs.items()}
        agg[N]['E'] = E
        print('N', N, 'done', round(time.time() - t0, 1), 's', flush=True)
    res['agg'] = agg
    # r needed per N for thresholds
    need = {}
    for N, d in agg.items():
        need[N] = {}
        for thr in [1e-2, 1e-3, 1e-4]:
            rs = [r for r in sorted(k for k in d if k != 'E') if d[r]['kl_q95'] <= thr]
            need[N][str(thr)] = rs[0] if rs else None
    res['r_needed_kl_q95'] = need
    res['seconds'] = time.time() - t0
    json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n1_pfn_rank%s.json' % ('' if len(sys.argv) < 2 else '_quick'), 'w'), indent=1)
    print(json.dumps({'sv_rel_first20': [round(x, 8) for x in res['sv_rel'][:20]], 'sup': sup,
                      'need': need, 'tail': tail}, indent=1))


if __name__ == '__main__':
    main()
