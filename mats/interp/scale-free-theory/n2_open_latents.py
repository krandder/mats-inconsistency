"""N2 (scale-free-theory, 2026-10-29): message size vs the number D of open latent variables at a cut.

Model (sources of unknown reliability reporting on D shared latents): theta ~ N(0, I_D); each source s reports
x_s in R^D, reliable with prob 1 - eta: x_s ~ N(theta, sig^2 I); junk with prob eta: x_s ~ N(0, tau^2 I).
l(theta; x) = log[(1 - eta) N(x; theta, sig^2 I) + eta N(x; 0, tau^2 I)]: not an exponential family in theta.

(a) Optimal one-round linear message size: eps-rank (RMS over theta uniform in the box [-A, A]^D, constants free) of
    the family {l(.; x)} with x from the model, by SVD of a sampled M x K matrix. Theory: grows like (c log 1/eps)^D.
(b) Aggregation, N sources: one round = tensor Chebyshev interpolation of degree r-1 per axis on the box (r^D
    numbers per source, summed); two rounds = additive pilot (sum of x_s and count -> mean / (1 - eta)), broadcast
    a window of half-width h around it, then tensor Chebyshev degree p per axis on the window ((p+1)^D numbers).
    Reported: KL(exact || approx) on a fine grid around the exact posterior, 90th percentile over episodes.
"""
import json, math, sys, time, itertools
import numpy as np
from numpy.polynomial import chebyshev as C

ETA, SIG, TAU, A = 0.3, 0.5, 2.0, 3.5


def ell(theta, x):
    """theta (K, D), x (M, D) -> (M, K)."""
    D = theta.shape[1]
    d2 = ((x[:, None, :] - theta[None, :, :]) ** 2).sum(-1)
    t1 = math.log(1 - ETA) - 0.5 * D * math.log(2 * math.pi * SIG ** 2) - 0.5 * d2 / SIG ** 2
    t2 = (math.log(ETA) - 0.5 * D * math.log(2 * math.pi * TAU ** 2) - 0.5 * (x ** 2).sum(-1) / TAU ** 2)[:, None]
    mx = np.maximum(t1, t2)
    return mx + np.log(np.exp(t1 - mx) + np.exp(t2 - mx))


def sample_x(n, theta, rng):
    D = len(theta)
    rel = rng.random(n) > ETA
    x = np.where(rel[:, None], theta[None] + SIG * rng.standard_normal((n, D)), TAU * rng.standard_normal((n, D)))
    return x


def part_a(rng, D, M=3000, K=6000):
    th = rng.uniform(-A, A, (K, D))
    xs = np.stack([sample_x(1, rng.standard_normal(D), rng)[0] for _ in range(M)])
    L = ell(th, xs)
    L = L - L.mean(1, keepdims=True)
    s = np.linalg.svd(L, compute_uv=False)
    # RMS eps-rank: smallest r with sqrt(sum_{i>r} s_i^2 / (M K)) <= eps (per-function RMS error over the box)
    tail = np.sqrt(np.maximum(np.cumsum((s ** 2)[::-1])[::-1], 0) / (M * K))
    out = {}
    for eps in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        idx = np.nonzero(tail <= eps)[0]
        out[str(eps)] = int(idx[0]) if len(idx) else None
    return out, (s[:200] / s[0]).tolist()


def cheb_nodes(n):
    return np.cos(np.pi * (np.arange(n) + 0.5) / n)


def tensor_coeffs(x, lo, hi, n, D):
    """Sum over sources of the degree-(n-1) tensor Chebyshev interpolant coefficients of l(.; x_s) on box."""
    nodes1 = [lo[d] + (hi[d] - lo[d]) * (cheb_nodes(n) + 1) / 2 for d in range(D)]
    grid = np.stack(np.meshgrid(*nodes1, indexing='ij'), -1).reshape(-1, D)
    vals = ell(grid, x).sum(0).reshape((n,) * D)     # summing first is exact: interpolation is linear
    k = np.arange(n)
    Tm = np.cos(np.outer(k, np.pi * (np.arange(n) + 0.5) / n)) * (2.0 / n); Tm[0] *= 0.5
    c = vals
    for d in range(D):
        c = np.moveaxis(np.tensordot(Tm, c, axes=([1], [d])), 0, d)
    return c


def eval_tensor(c, lo, hi, pts, D):
    """Evaluate tensor Chebyshev series at pts (P, D); -inf outside the box."""
    xs = [(2 * pts[:, d] - lo[d] - hi[d]) / (hi[d] - lo[d]) for d in range(D)]
    inside = np.all([(np.abs(z) <= 1) for z in xs], 0)
    Vs = [C.chebvander(np.clip(z, -1, 1), c.shape[d] - 1) for d, z in enumerate(xs)]
    if D == 1:
        out = Vs[0] @ c
    elif D == 2:
        out = np.einsum('pi,pj,ij->p', Vs[0], Vs[1], c)
    else:
        out = np.einsum('pi,pj,pk,ijk->p', Vs[0], Vs[1], Vs[2], c)
    return np.where(inside, out, -np.inf)


def kl(lpe, lpa):
    pe = np.exp(lpe - lpe.max()); pe /= pe.sum()
    if not np.all(np.isfinite(lpa[pe > 1e-10])):
        return float('inf')
    lpa = np.where(np.isfinite(lpa), lpa, -1e300)
    pa = np.exp(lpa - lpa.max()); pa /= pa.sum()
    return float((pe * (np.log(np.maximum(pe, 1e-300)) - np.log(np.maximum(pa, 1e-300)))).sum())


def part_b(rng, D, N, E, R1, P2):
    gpts = {1: 801, 2: 81, 3: 31}[D]
    one = {r: [] for r in R1}; two = {p: [] for p in P2}
    for e in range(E):
        theta = rng.standard_normal(D)
        x = sample_x(N, theta, rng)
        # exact posterior: locate on a coarse grid, then a fine grid of +-7 sd per axis
        g1 = np.linspace(-A, A, {1: 2001, 2: 161, 3: 41}[D])
        G = np.stack(np.meshgrid(*([g1] * D), indexing='ij'), -1).reshape(-1, D)
        lp = -0.5 * (G ** 2).sum(1) + ell(G, x).sum(0)
        p = np.exp(lp - lp.max()); p /= p.sum()
        m0 = (p[:, None] * G).sum(0); sd0 = np.sqrt((p[:, None] * (G - m0) ** 2).sum(0))
        sd0 = np.maximum(sd0, (2 * A) / len(g1))
        axes = [np.linspace(max(-A, m0[d] - 7 * sd0[d]), min(A, m0[d] + 7 * sd0[d]), gpts) for d in range(D)]
        F = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, D)
        lpe = -0.5 * (F ** 2).sum(1) + ell(F, x).sum(0)
        lo, hi = np.full(D, -A), np.full(D, A)
        for r in R1:
            c = tensor_coeffs(x, lo, hi, r, D)
            one[r].append(kl(lpe, -0.5 * (F ** 2).sum(1) + eval_tensor(c, lo, hi, F, D)))
        m1 = x.sum(0) / ((1 - ETA) * N + 1.0)             # additive pilot (round 1: D + 1 numbers per source)
        h = max(8.0 * math.sqrt((SIG ** 2 + ETA * TAU ** 2 / (1 - ETA)) / ((1 - ETA) * N + 1.0)), 0.05)
        lo2, hi2 = np.maximum(m1 - h, -A), np.minimum(m1 + h, A)
        for pdeg in P2:
            c = tensor_coeffs(x, lo2, hi2, pdeg + 1, D)
            two[pdeg].append(kl(lpe, -0.5 * (F ** 2).sum(1) + eval_tensor(c, lo2, hi2, F, D)))
    q = lambda v: float(np.quantile(v, 0.9))
    return ({r: q(one[r]) for r in R1}, {p: q(two[p]) for p in P2})


def main(quick=False):
    rng = np.random.default_rng(2910)
    t0 = time.time()
    res = {'eta': ETA, 'sig': SIG, 'tau': TAU, 'A': A, 'a': {}, 'b': {}}
    for D in [1, 2, 3]:
        ra, sv = part_a(rng, D, M=(3000 if not quick else 600), K=(6000 if not quick else 1200))
        res['a'][D] = {'eps_rank_rms': ra, 'sv_rel_first200': sv}
        print('D', D, 'one-round optimal eps-rank (RMS):', ra, round(time.time() - t0, 1), 's', flush=True)
        json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n2_open_latents%s.json'
                            % ('_quick' if quick else ''), 'w'), indent=1)
    cfg = {1: ([8, 12, 16, 24, 32, 48, 64], [2, 3, 4, 6, 8]), 2: ([8, 12, 16, 24, 32, 48], [2, 3, 4, 6, 8]),
           3: ([8, 12, 16, 24, 32], [2, 3, 4, 6])}
    for D in [1, 2, 3]:
        R1, P2 = cfg[D]
        res['b'][D] = {}
        for N in ([64, 512] if D < 3 else [64, 256]):
            E = (40 if D < 3 else 12) if not quick else 3
            o1, t2 = part_b(rng, D, N, E, R1, P2)
            n1 = {str(t): next(((r, r ** D) for r in R1 if o1[r] <= t), None) for t in [1e-2, 1e-3]}
            n2 = {str(t): next(((p, (p + 1) ** D + D + 1) for p in P2 if t2[p] <= t), None) for t in [1e-2, 1e-3]}
            res['b'][D][N] = {'E': E, 'one_round_kl_q90': o1, 'two_round_kl_q90': t2,
                              'need_one_round_(r,numbers)': n1, 'need_two_round_(p,numbers)': n2}
            print('D', D, 'N', N, 'one-round (r, numbers):', n1, '| two-round (p, numbers incl. pilot):', n2,
                  round(time.time() - t0, 1), 's', flush=True)
            json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n2_open_latents%s.json'
                                % ('_quick' if quick else ''), 'w'), indent=1)
    res['seconds'] = time.time() - t0
    json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n2_open_latents%s.json'
                        % ('_quick' if quick else ''), 'w'), indent=1)


if __name__ == '__main__':
    main(quick=len(sys.argv) > 1)
