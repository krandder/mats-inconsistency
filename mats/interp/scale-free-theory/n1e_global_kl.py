"""N1e (scale-free-theory, 2026-10-29): N1c re-scored with globally normalised approximations (review fix).

N1c normalised each approximate posterior only on a grid around the exact posterior, ignoring approximate mass
elsewhere. Here the one-round approximation is normalised over the whole prior range [-A, A], and the two-round
approximation over its whole window, before computing KL(Bayes || approx) on the exact posterior's region.
Same world, encoders and pilot as n1c_rounds.py (imported).
"""
import json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/scale-free-theory')
from n1c_rounds import ell_grid, cheb_interp_coeffs, VLEV, QL, A
from numpy.polynomial import chebyshev as C


def trap(f, x):
    return float(np.sum((f[1:] + f[:-1]) * np.diff(x)) / 2)


def logZ(csum, lo, hi, prior=True, n=20001):
    """log of the integral of exp(-y^2/2 + chebyshev series) over [lo, hi]."""
    y = np.linspace(lo, hi, n)
    lp = (-0.5 * y ** 2 if prior else 0) + C.chebval((2 * y - lo - hi) / (hi - lo), csum)
    mx = lp.max()
    return mx + math.log(trap(np.exp(lp - mx), y))


def kl_global(yf, lpe, csum, lo, hi):
    """KL(exact || approx), exact on grid yf (density from lpe), approx = exp(-y^2/2 + series)/Z over [lo, hi]."""
    pe = np.exp(lpe - lpe.max()); pe /= trap(pe, yf)
    inside = (yf >= lo) & (yf <= hi)
    if np.any(pe[~inside] > 1e-12 * pe.max()):
        return float('inf')
    lz = logZ(csum, lo, hi)
    lq = -0.5 * yf ** 2 + C.chebval((2 * np.clip(yf, lo, hi) - lo - hi) / (hi - lo), csum) - lz
    f = pe * (np.log(np.maximum(pe, 1e-300)) - lq)
    f[pe < 1e-300] = 0.0
    return trap(f, yf)


def main():
    rng = np.random.default_rng(31)
    R1 = [16, 24, 32, 48, 64, 96, 128]
    P2 = [2, 3, 4, 6]
    res = {'R1': R1, 'P2': P2, 'per_N': {}}
    t0 = time.time()
    for N, E in [(3, 200), (64, 200), (1024, 60), (4096, 25)]:
        one = {r: [] for r in R1}; two = {p: [] for p in P2}
        for e in range(E):
            ystar = rng.standard_normal()
            vv = VLEV[rng.integers(0, 3, N)]
            oo = ystar + rng.standard_normal(N) * np.sqrt(vv)
            sss = vv * rng.chisquare(QL, N)
            yc = np.linspace(-A, A, 4001)
            lp = -0.5 * yc ** 2 + ell_grid(yc, oo, sss).sum(0)
            pc = np.exp(lp - lp.max()); pc /= pc.sum()
            m0 = (pc * yc).sum(); sd0 = max(math.sqrt(((yc - m0) ** 2 * pc).sum()), 0.002)
            yf = np.linspace(max(-A, m0 - 12 * sd0), min(A, m0 + 12 * sd0), 4001)
            lpe = -0.5 * yf ** 2 + ell_grid(yf, oo, sss).sum(0)
            for r in R1:
                cs = cheb_interp_coeffs(oo, sss, -A, A, r - 1).sum(0)
                one[r].append(kl_global(yf, lpe, cs, -A, A))
            w = 1.0 / np.maximum(sss / QL, 0.05)
            m1 = (w * oo).sum() / (1.0 + w.sum()); sd1 = 1.0 / math.sqrt(1.0 + np.minimum(w, 10.0).sum())
            h = max(10 * sd1, 0.05); lo, hi = max(m1 - h, -A), min(m1 + h, A)
            for p in P2:
                cs2 = cheb_interp_coeffs(oo, sss, lo, hi, p).sum(0)
                two[p].append(kl_global(yf, lpe, cs2, lo, hi))
        q95 = lambda v: float(np.quantile(v, 0.95))
        res['per_N'][N] = {'E': E, 'one_round_kl_q95': {r: q95(one[r]) for r in R1},
                           'two_round_kl_q95': {p: q95(two[p]) for p in P2},
                           'two_round_frac_inf': {p: float(np.mean(~np.isfinite(two[p]))) for p in P2}}
        print('N', N, 'one', {r: f"{q95(one[r]):.1e}" for r in R1}, 'two', {p: f"{q95(two[p]):.1e}" for p in P2},
              round(time.time() - t0), 's', flush=True)
        json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n1e_global_kl.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
