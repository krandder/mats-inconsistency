"""N1b (scale-free-theory, 2026-10-29): one-round additive messages for the tiny-PFN world with N sources,
using a per-source weighted encoder (the weight says where the final posterior can lie given this source alone).

Per source s: l_s(y) = log sum_j (1/3) N(o_s; y, v_j) (2 pi v_j)^(-QL/2) exp(-ss_s / (2 v_j))   (QL = 4).
Encoder phi_r(o, ss) = weighted least-squares Chebyshev coefficients (degree r-1 on [-Y, Y]) of l_s, with weight
w_s(y) proportional to [N(y; 0, 1) exp(l_s(y))]^(1/4) (the single-source posterior, tempered to 4x its variance).
The message of a set of sources is the SUM of their r-dim encodings; the root reads the posterior
exp(-y^2/2 + sum_k a_k T_k(y / Y)). The encoder is a fixed function of the source's own data: one round, additive.
Reported: for N = 3 .. 1024, the smallest r with KL(exact || approx) <= delta at the 95th percentile over episodes.
Theory (C3): r should grow like log(N / delta) times a constant set by the analyticity of l.
"""
import json, math, sys, time
import numpy as np
from numpy.polynomial import chebyshev as C

VLEV = np.array([0.1, 0.5, 2.0]); QL = 4; Y = 6.0
RS = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56]
YG = np.linspace(-Y, Y, 1201)
VG = C.chebvander(YG / Y, max(RS) - 1)                    # (K, Rmax)


def ell_grid(y, o, ss):
    T = np.stack([(-math.log(3.0) - 0.5 * (QL + 1) * math.log(2 * math.pi * v) - 0.5 * (o[:, None] - y[None, :]) ** 2 / v
                   - 0.5 * ss[:, None] / v) for v in VLEV], 0)
    mx = T.max(0)
    return mx + np.log(np.exp(T - mx).sum(0))


def encode(o, ss, r):
    """(N, r) weighted-LS Chebyshev coefficients."""
    L = ell_grid(YG, o, ss)                                  # (N, K)
    lw = 0.25 * (-0.5 * YG[None] ** 2 + L)
    w = np.exp(lw - lw.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)
    pf = np.exp(-0.125 * YG ** 2); pf = pf / pf.sum()          # floor: tempered prior, so the fit stays bounded
    w = w + 1e-3 * pf[None]
    V = VG[:, :r]
    out = np.empty((len(o), r))
    for i in range(len(o)):
        sw = np.sqrt(w[i])
        A = V * sw[:, None]
        b = L[i] * sw
        out[i] = np.linalg.lstsq(A, b, rcond=1e-12)[0]
    return out


def main(quick=False):
    rng = np.random.default_rng(1029)
    t0 = time.time()
    Ns = [3, 8, 24, 64, 256, 1024] if not quick else [3, 64]
    res = {'RS': RS, 'Y': Y, 'per_N': {}}
    for N in Ns:
        E = (300 if N <= 64 else 120) if not quick else 15
        kl = {r: [] for r in RS}; dm = {r: [] for r in RS}
        for e in range(E):
            ystar = rng.standard_normal()
            vv = VLEV[rng.integers(0, 3, N)]
            oo = ystar + rng.standard_normal(N) * np.sqrt(vv)
            sss = vv * rng.chisquare(QL, N)
            yc = np.linspace(-Y, Y, 6001)
            lp = -0.5 * yc ** 2 + ell_grid(yc, oo, sss).sum(0)
            pc = np.exp(lp - lp.max()); pc /= pc.sum()
            m0 = (pc * yc).sum(); sd0 = max(math.sqrt(((yc - m0) ** 2 * pc).sum()), 0.003)
            yf = np.linspace(m0 - 12 * sd0, m0 + 12 * sd0, 3001)
            lpe = -0.5 * yf ** 2 + ell_grid(yf, oo, sss).sum(0)
            pe = np.exp(lpe - lpe.max()); pe /= pe.sum()
            me = (pe * yf).sum(); sde = math.sqrt(((yf - me) ** 2 * pe).sum())
            A = encode(oo, sss, max(RS))                     # coefficients for the largest r are NOT nested;
            for r in RS:                                     # so re-encode per r (exact weighted LS per r)
                a = encode(oo, sss, r).sum(0) if r != max(RS) else A.sum(0)
                lpa = -0.5 * yf ** 2 + C.chebval(yf / Y, a)
                pa = np.exp(lpa - lpa.max()); pa /= pa.sum()
                k = float((pe * (np.log(np.maximum(pe, 1e-300)) - np.log(np.maximum(pa, 1e-300)))).sum())
                kl[r].append(k); dm[r].append(abs((pa * yf).sum() - me) / sde)
        d = {r: {'kl_mean': float(np.mean(kl[r])), 'kl_q95': float(np.quantile(kl[r], 0.95)),
                 'dmean_sd_q95': float(np.quantile(dm[r], 0.95))} for r in RS}
        need = {}
        for thr in [1e-2, 1e-3, 1e-4]:
            rs = [r for r in RS if d[r]['kl_q95'] <= thr]
            need[str(thr)] = rs[0] if rs else None
        res['per_N'][N] = {'E': E, 'by_r': d, 'r_needed_kl_q95': need}
        print('N', N, 'E', E, 'need', need, 'kl_q95', {r: round(d[r]['kl_q95'], 6) for r in RS},
              round(time.time() - t0, 1), 's', flush=True)
    res['seconds'] = time.time() - t0
    json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n1b_pfn_weighted%s.json'
                        % ('_quick' if quick else ''), 'w'), indent=1)


if __name__ == '__main__':
    main(quick=len(sys.argv) > 1)
