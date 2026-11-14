"""N1c (scale-free-theory, 2026-10-29): one round vs two rounds of additive messages, tiny-PFN world with N sources.

Per source s: l_s(y) = log sum_j (1/3) N(o_s; y, v_j) (2 pi v_j)^(-QL/2) exp(-ss_s / (2 v_j)), QL = 4, v_j in VLEV.
Exact: log p(y | data) = -y^2/2 + sum_s l_s(y) + const.

ONE ROUND (no broadcast): each source sends the r Chebyshev-interpolation coefficients of l_s on the prior range
[-A, A] (a fixed encoder of its own data); the root sums them (additive merge) and reads the posterior.
TWO ROUNDS (Kelvin's "messages written after reading the consensus"): round 1 = 3 additive numbers per source
(crude precision weight, weighted reading, count) giving a GLS pilot and crude sd; the root broadcasts a window [m1 - h, m1 + h] (h = 10 x round-1 sd, floored); round 2 = each source sends the p+1
Chebyshev coefficients of l_s on that window; the root sums them. Message size per source = r1 + p + 1.
Metric: KL(exact || approx) over the exact posterior's support (approx mass outside the window = 0 -> KL = inf).
"""
import json, math, sys, time
import numpy as np
from numpy.polynomial import chebyshev as C

VLEV = np.array([0.1, 0.5, 2.0]); QL = 4; A = 5.5


def ell_grid(y, o, ss):
    T = np.stack([(-math.log(3.0) - 0.5 * (QL + 1) * math.log(2 * math.pi * v) - 0.5 * (o[:, None] - y[None, :]) ** 2 / v
                   - 0.5 * ss[:, None] / v) for v in VLEV], 0)
    mx = T.max(0)
    return mx + np.log(np.exp(T - mx).sum(0))


def cheb_interp_coeffs(o, ss, lo, hi, deg):
    """Coefficients (N, deg+1) of the degree-deg Chebyshev interpolant of l_s on [lo, hi]."""
    n = deg + 1
    x = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    vals = ell_grid(lo + (hi - lo) * (x + 1) / 2, o, ss)
    k = np.arange(n)
    Tm = np.cos(np.outer(k, np.pi * (np.arange(n) + 0.5) / n))
    c = vals @ Tm.T * (2.0 / n); c[:, 0] *= 0.5
    return c


def eval_sum(csum, lo, hi, y):
    x = (2 * y - lo - hi) / (hi - lo)
    out = C.chebval(np.clip(x, -1, 1), csum)
    out[(x < -1) | (x > 1)] = -np.inf
    return out


def kl_on(yf, lpe, lpa):
    pe = np.exp(lpe - lpe.max()); pe /= pe.sum()
    if not np.all(np.isfinite(lpa[pe > 1e-12])):
        return float('inf'), float('inf')
    lpa = np.where(np.isfinite(lpa), lpa, -1e300)
    pa = np.exp(lpa - lpa.max()); pa /= pa.sum()
    k = float((pe * (np.log(np.maximum(pe, 1e-300)) - np.log(np.maximum(pa, 1e-300)))).sum())
    me = (pe * yf).sum(); sde = math.sqrt(((yf - me) ** 2 * pe).sum())
    return k, abs((pa * yf).sum() - me) / sde


def main(quick=False):
    rng = np.random.default_rng(29)
    t0 = time.time()
    R1 = [8, 16, 24, 32, 48, 64, 96, 128, 160, 200]
    P2 = [2, 3, 4, 6, 8, 12]
    r1_pilot = 3
    Ns = [3, 8, 24, 64, 256, 1024, 4096] if not quick else [64, 1024]
    res = {'A': A, 'R1': R1, 'P2': P2, 'r1_pilot': r1_pilot, 'per_N': {}}
    for N in Ns:
        E = (200 if N <= 256 else (80 if N <= 1024 else 30)) if not quick else 6
        one = {r: [] for r in R1}; two = {p: [] for p in P2}
        for e in range(E):
            ystar = rng.standard_normal()
            vv = VLEV[rng.integers(0, 3, N)]
            oo = ystar + rng.standard_normal(N) * np.sqrt(vv)
            sss = vv * rng.chisquare(QL, N)
            yc = np.linspace(-A, A, 2001)
            lp = -0.5 * yc ** 2 + ell_grid(yc, oo, sss).sum(0)
            pc = np.exp(lp - lp.max()); pc /= pc.sum()
            m0 = (pc * yc).sum(); sd0 = max(math.sqrt(((yc - m0) ** 2 * pc).sum()), 0.002)
            yf = np.linspace(max(-A, m0 - 12 * sd0), min(A, m0 + 12 * sd0), 3001)
            lpe = -0.5 * yf ** 2 + ell_grid(yf, oo, sss).sum(0)
            for r in R1:                                         # one round
                cs = cheb_interp_coeffs(oo, sss, -A, A, r - 1).sum(0)
                one[r].append(kl_on(yf, lpe, -0.5 * yf ** 2 + eval_sum(cs, -A, A, yf))[0])
            # two rounds. Round 1 (additive, 3 numbers per source): crude precision weights from the labelled
            # residuals, w_s = 1 / max(ss_s / QL, 0.05); the root forms a GLS pilot and a crude posterior sd.
            w = 1.0 / np.maximum(sss / QL, 0.05)
            m1 = (w * oo).sum() / (1.0 + w.sum()); sd1 = 1.0 / math.sqrt(1.0 + np.minimum(w, 10.0).sum())
            h = max(10 * sd1, 0.05)
            lo, hi = m1 - h, m1 + h
            for p in P2:
                cs2 = cheb_interp_coeffs(oo, sss, lo, hi, p).sum(0)
                two[p].append(kl_on(yf, lpe, -0.5 * yf ** 2 + eval_sum(cs2, lo, hi, yf))[0])
        d1 = {r: {'kl_q95': float(np.quantile(one[r], 0.95)), 'kl_median': float(np.median(one[r]))} for r in R1}
        d2 = {p: {'kl_q95': float(np.quantile(two[p], 0.95)), 'kl_median': float(np.median(two[p])),
                  'frac_inf': float(np.mean(~np.isfinite(two[p])))} for p in P2}
        need1 = {str(t): next((r for r in R1 if d1[r]['kl_q95'] <= t), None) for t in [1e-2, 1e-3, 1e-4]}
        need2 = {str(t): next((p for p in P2 if d2[p]['kl_q95'] <= t), None) for t in [1e-2, 1e-3, 1e-4]}
        res['per_N'][N] = {'E': E, 'one_round': d1, 'two_round': d2, 'need_one_round_r': need1,
                           'need_two_round_p': need2}
        print('N', N, 'E', E, '| one-round r needed', need1, '| two-round p needed (+%d pilot)' % r1_pilot, need2,
              round(time.time() - t0, 1), 's', flush=True)
        json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n1c_rounds%s.json'
                            % ('_quick' if quick else ''), 'w'), indent=1)
    res['seconds'] = time.time() - t0
    json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n1c_rounds%s.json'
                        % ('_quick' if quick else ''), 'w'), indent=1)


if __name__ == '__main__':
    main(quick=len(sys.argv) > 1)
