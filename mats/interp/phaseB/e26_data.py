"""E26 data (PREREG-B amendment 34): episodes for the error budget by scale.

Factors, mutually independent under the prior (written in terms of errors, not readings):
  c   = common context: labels y_1..y_4, the query answer y*, source and question identity vectors;
  B_s = (Z_s, u_s) for each source s, with Z_s = (v_s, rho_s, e_s): variance level, norm of the four labelled residuals
        (rho_s^2 = SS_s) and the query-reading error; u_s = direction of the labelled-residual vector on the sphere S^3
        (uniform and independent of Z_s and c, because an isotropic Gaussian's direction is independent of its norm).
Bayes depends on B_s only through (SS_s, o*_s = y* + e_s), so it never depends on u_s, and its log-odds
L(delta) = log p(y* + delta | x) - log p(y* | x) is exactly additive over (c, B_1, B_2, B_3).

Variants per base (34), all sharing c:
  0       base: every source (Z0, u0)
  1       every source (Z1, u1) (independent alternative draw)
  2+s     source s (Z0, u0), the other two (Z1, u1)            [keeps c, B_s]
  5+s     source s (Z1, u1), the other two (Z0, u0)            [keeps c, B_-s]
  8+s     source s (Z0, u1), the other two (Z1, u1)            [keeps c, Z_s; differs from 2+s only in u_s]
  11      every source (Z0, u1)                                [keeps c, Z]
  12..25  every source (Z0, u^r), r = 2..15                    [pattern set = {0, 11, 12..25}: 16 draws of u given c, Z]
  26..33  every source: the base's residual magnitudes with random signs and a random assignment to the four labelled
          questions                                            [sign/assignment set = {0, 26..33}]
usage: python e26_data.py -> results/e26_data.npz (registered seed and size); python e26_data.py PATH SEED N (checks)
"""
import os, sys
import numpy as np

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import world as W  # noqa: E402

S, Q, QL = W.S, W.Q, W.QL
DELTAS = np.array([-0.6, -0.3, -0.15, 0.15, 0.3, 0.6])
R_PAT, R_SP = 16, 8
NVAR = 26 + R_SP
PAT = [0, 11] + list(range(12, 26))
SP = [0] + list(range(26, 26 + R_SP))
SEED, NBASE = 2026_1029_34, 6000


def draw_Z(rng, n):
    vidx = rng.integers(0, W.NV, (n, S))
    v = W.VLEV[vidx]
    r = rng.standard_normal((n, S, QL)) * np.sqrt(v)[:, :, None]
    rho = np.linalg.norm(r, axis=2)
    e = rng.standard_normal((n, S)) * np.sqrt(v)
    return vidx, rho, e


def draw_u(rng, n):
    g = rng.standard_normal((n, S, QL))
    return g / np.linalg.norm(g, axis=2, keepdims=True)


def build(seed, n, ds=16):
    """All factors and the 34 variants. Returns a dict of arrays (variants on axis 1)."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n, Q))
    semb = rng.standard_normal((n, S, ds)) / 4
    qemb = rng.standard_normal((n, Q, ds)) / 4
    Z0, Z1 = draw_Z(rng, n), draw_Z(rng, n)
    U = [draw_u(rng, n) for _ in range(R_PAT)]
    vidx = np.zeros((n, NVAR, S), np.int64); res = np.zeros((n, NVAR, S, QL)); err = np.zeros((n, NVAR, S))

    def put(k, s, Z, u):
        vidx[:, k, s] = Z[0][:, s]; res[:, k, s] = Z[1][:, s, None] * u[:, s]; err[:, k, s] = Z[2][:, s]

    for s in range(S):
        put(0, s, Z0, U[0]); put(1, s, Z1, U[1]); put(11, s, Z0, U[1])
        for t in range(S):
            put(2 + t, s, Z0, U[0]) if s == t else put(2 + t, s, Z1, U[1])
            put(5 + t, s, Z1, U[1]) if s == t else put(5 + t, s, Z0, U[0])
            put(8 + t, s, Z0, U[1]) if s == t else put(8 + t, s, Z1, U[1])
        for r in range(2, R_PAT):
            put(10 + r, s, Z0, U[r])
    base = res[:, 0]                                                          # (n, S, QL)
    for j in range(R_SP):
        perm = np.argsort(rng.random((n, S, QL)), axis=2)
        sg = rng.choice([-1.0, 1.0], size=(n, S, QL))
        res[:, 26 + j] = sg * np.take_along_axis(np.abs(base), perm, axis=2)
        vidx[:, 26 + j] = vidx[:, 0]; err[:, 26 + j] = err[:, 0]
    o = np.empty((n, NVAR, S, Q))
    o[..., :QL] = y[:, None, None, :QL] + res
    o[..., QL] = y[:, None, None, QL] + err
    return dict(y=y, semb=semb, qemb=qemb, vidx=vidx, res=res, err=err, o=o)


def episode(D, k):
    """World-format episode dict of variant k."""
    vi = D['vidx'][:, k]
    return dict(vidx=vi, v=W.VLEV[vi], y=D['y'], o=D['o'][:, k])


def bayes_parts(d):
    out, nll, mom, w = W.references(d)
    means, vars_, wts = out['bayes']
    return means, vars_, wts, mom['bayes']


def mix_logp_points(means, vars_, wts, pts):
    """log density of the (n, 27) mixtures at points pts (n, m)."""
    comp = (np.log(np.maximum(wts, 1e-300))[:, None, :] - 0.5 * np.log(2 * np.pi * vars_)[None, None]
            - 0.5 * (pts[:, :, None] - means[:, None, :]) ** 2 / vars_[None, None])
    mx = comp.max(2, keepdims=True)
    return mx[:, :, 0] + np.log(np.exp(comp - mx).sum(2))


def logodds(lp):
    return lp[:, 1:] - lp[:, :1]


def mix_score(means, vars_, wts, y):
    """d/dy log p(y) of (n, C) mixtures at y (n,); vars_ (C,) or (n, C)."""
    vv = np.broadcast_to(vars_, means.shape)
    lc = np.log(np.maximum(wts, 1e-300)) - 0.5 * np.log(2 * np.pi * vv) - 0.5 * (y[:, None] - means) ** 2 / vv
    r = np.exp(lc - lc.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    return (r * (means - y[:, None]) / vv).sum(1)


def reduce_mixture(means, vars_, wts, K=4):
    """Runnalls (2007) greedy moment-preserving merging of each episode's 27-component Bayes mixture down to K components
    (a near-best K-component approximation; the floor for a K-component mixture head)."""
    n, C = wts.shape
    w = wts.copy(); m = means.copy(); v = np.broadcast_to(vars_, (n, C)).copy()
    alive = np.ones((n, C), bool)
    iu = np.triu_indices(C, 1)
    rows = np.arange(n)
    for _ in range(C - K):
        wi, wj = w[:, iu[0]], w[:, iu[1]]; ws = np.maximum(wi + wj, 1e-300)
        mu = (wi * m[:, iu[0]] + wj * m[:, iu[1]]) / ws
        var = (wi * (v[:, iu[0]] + m[:, iu[0]] ** 2) + wj * (v[:, iu[1]] + m[:, iu[1]] ** 2)) / ws - mu ** 2
        var = np.maximum(var, 1e-12)
        cost = 0.5 * (ws * np.log(var) - wi * np.log(v[:, iu[0]]) - wj * np.log(v[:, iu[1]]))
        cost[~(alive[:, iu[0]] & alive[:, iu[1]])] = np.inf
        k = cost.argmin(1); i, j = iu[0][k], iu[1][k]
        w[rows, i] = ws[rows, k]; m[rows, i] = mu[rows, k]; v[rows, i] = var[rows, k]
        alive[rows, j] = False; w[rows, j] = 0.0
    return m, v, w, alive


def mix_logp_points_v(means, vars_, wts, pts):
    """Like mix_logp_points with per-episode component variances (n, C)."""
    comp = (np.log(np.maximum(wts, 1e-300))[:, None, :] - 0.5 * np.log(2 * np.pi * vars_)[:, None, :]
            - 0.5 * (pts[:, :, None] - means[:, None, :]) ** 2 / vars_[:, None, :])
    mx = comp.max(2, keepdims=True)
    return mx[:, :, 0] + np.log(np.exp(comp - mx).sum(2))


def references_for_variants(D):
    """Bayes, Bayes moment-matched Gaussian and Bayes reduced to 4 components (Runnalls) log-odds at y* + DELTAS for every
    variant (the last two are floors for Gaussian and 4-component mixture heads; both are functions of the Bayes posterior,
    so they do not depend on the residual directions either)."""
    n = len(D['y']); ystar = D['y'][:, QL]
    pts = ystar[:, None] + np.concatenate([[0.0], DELTAS])[None]
    LB = np.zeros((n, NVAR, len(DELTAS) + 1)); LG = np.zeros_like(LB); LT = np.zeros_like(LB)      # last column: score at y*
    base = None
    for k in range(NVAR):
        means, vars_, wts, (mm, vv) = bayes_parts(episode(D, k))
        LB[:, k, :-1] = logodds(mix_logp_points(means, vars_, wts, pts)); LB[:, k, -1] = mix_score(means, vars_, wts, ystar)
        LG[:, k, :-1] = logodds(-0.5 * (pts - mm[:, None]) ** 2 / vv[:, None]); LG[:, k, -1] = (mm - ystar) / vv
        if k < 12:                                        # the reduction is only needed for the design variants
            rm, rv, rw, _ = reduce_mixture(means, vars_, wts)
            LT[:, k, :-1] = logodds(mix_logp_points_v(rm, rv, rw, pts)); LT[:, k, -1] = mix_score(rm, rv, rw, ystar)
        if k == 0:
            base = dict(bmeans=means, bvars=vars_, bwts=wts, bmean=mm, bvar=vv, r4means=rm, r4vars=rv, r4wts=rw)
    return LB, LG, LT, base


def main():
    p, seed, n = 'results/e26_data.npz', SEED, NBASE
    if len(sys.argv) > 1:                                 # plumbing checks only: python e26_data.py PATH SEED N
        p, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    assert not os.path.exists(p), p
    D = build(seed, n)
    LB, LG, LT, base = references_for_variants(D)
    inv = float(np.abs(LB[:, PAT + SP] - LB[:, :1]).max())
    print('max |Bayes log-odds change| over the pattern and sign/assignment variants: %.2e' % inv)
    assert inv < 1e-8, 'Bayes not invariant to the residual direction'
    np.savez(p, LB=LB, LG=LG, LT=LT, deltas=DELTAS, seed=seed, **D, **base)
    print('saved', p, {k: v.shape for k, v in D.items()})


if __name__ == '__main__':
    main()
