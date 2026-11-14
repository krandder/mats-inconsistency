"""N1d (scale-free-theory, 2026-10-29): what SMALL additive messages achieve in the tiny-PFN world scaled to N sources.

Calibrates predictions for learned small-k encoders (the hierarchy-vs-dense experiment):
  one round, 2 numbers per source: GLS with w_s = E[1/v_s | ss_s] (labelled residuals only): (sum w, sum w o).
  two rounds, 3 + 2 numbers: round 1 = the same GLS pilot m1; round 2 = (l_s'(m1), l_s''(m1)); root: Gaussian at the
      Newton step (Laplace). Three rounds: re-centre at m2 and repeat round 2 (3 + 2 + 2 numbers).
  two rounds, 3 + (p+1) numbers: local Chebyshev degree p on a window around the pilot (as in n1c), for reference.
Metric: KL(exact || approx) on a fine grid; mean and 95th percentile over episodes; also |mean error| / posterior sd.
"""
import json, math, sys, time
import numpy as np

VLEV = np.array([0.1, 0.5, 2.0]); QL = 4; A = 5.5


def comps(y, o, ss):
    return np.stack([(-math.log(3.0) - 0.5 * (QL + 1) * math.log(2 * math.pi * v) - 0.5 * (o[:, None] - y[None, :]) ** 2 / v
                      - 0.5 * ss[:, None] / v) for v in VLEV], 0)          # (3, N, K)


def ell_grid(y, o, ss):
    T = comps(y, o, ss); mx = T.max(0)
    return mx + np.log(np.exp(T - mx).sum(0))


def ell_d12(y0, o, ss):
    """First and second derivative of l_s at scalar y0, per source."""
    T = comps(np.array([y0]), o, ss)[:, :, 0]                              # (3, N)
    w = np.exp(T - T.max(0)); w /= w.sum(0)                                # responsibilities
    d1j = (o[None] - y0) / VLEV[:, None]
    d2j = -1.0 / VLEV[:, None]
    m1 = (w * d1j).sum(0)
    m2 = (w * (d2j + d1j ** 2)).sum(0) - m1 ** 2
    return m1, m2


def gauss_lp(y, m, v):
    return -0.5 * (y - m) ** 2 / v


def kl(pe_lp, pa_lp, yf):
    pe = np.exp(pe_lp - pe_lp.max()); pe /= pe.sum()
    pa = np.exp(pa_lp - pa_lp.max()); pa /= pa.sum()
    k = float((pe * (np.log(np.maximum(pe, 1e-300)) - np.log(np.maximum(pa, 1e-300)))).sum())
    me = (pe * yf).sum(); sde = math.sqrt(((yf - me) ** 2 * pe).sum())
    return k, abs((pa * yf).sum() - me) / sde


def main(quick=False):
    rng = np.random.default_rng(4)
    t0 = time.time()
    Ns = [3, 8, 24, 64, 256, 1024, 4096] if not quick else [3, 256]
    res = {}
    # E[1/v | ss] from labelled residuals
    for N in Ns:
        E = (300 if N <= 256 else (100 if N <= 1024 else 40)) if not quick else 10
        rec = {k: [] for k in ['gls_1round_2', 'newton_2round_5', 'newton_3round_7', 'lab_mixture_exact_weights']}
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
            # posterior over levels from labelled residuals only
            lw = -0.5 * sss[None] / VLEV[:, None] - 0.5 * QL * np.log(2 * np.pi * VLEV)[:, None]
            pw = np.exp(lw - lw.max(0)); pw /= pw.sum(0)
            w = (pw / VLEV[:, None]).sum(0)                                   # E[1/v | ss]
            prec = 1.0 + w.sum(); m1 = (w * oo).sum() / prec
            rec['gls_1round_2'].append(kl(lpe, gauss_lp(yf, m1, 1.0 / prec), yf))
            # labelled-only mixture weights but exact per-source mixture in y (not additive; reference only)
            lpl = -0.5 * yf ** 2 + np.log((pw[:, :, None] * np.exp(-0.5 * (oo[None, :, None] - yf[None, None, :]) ** 2
                                                                  / VLEV[:, None, None]) / np.sqrt(VLEV)[:, None, None]).sum(0)).sum(0)
            rec['lab_mixture_exact_weights'].append(kl(lpe, lpl, yf))
            # two rounds: Newton at m1 with (sum l', sum l'')
            g, h = ell_d12(m1, oo, sss)
            H = 1.0 - h.sum(); G = -m1 + g.sum()
            H = max(H, 1e-6)
            m2 = m1 + G / H
            rec['newton_2round_5'].append(kl(lpe, gauss_lp(yf, m2, 1.0 / H), yf))
            g, h = ell_d12(m2, oo, sss)
            H2 = max(1.0 - h.sum(), 1e-6); G2 = -m2 + g.sum(); m3 = m2 + G2 / H2
            rec['newton_3round_7'].append(kl(lpe, gauss_lp(yf, m3, 1.0 / H2), yf))
        res[N] = {k: {'kl_mean': float(np.mean([a for a, b in v])), 'kl_q95': float(np.quantile([a for a, b in v], 0.95)),
                      'dmean_sd_q95': float(np.quantile([b for a, b in v], 0.95))} for k, v in rec.items()}
        res[N]['E'] = E
        print('N', N, {k: (round(v['kl_mean'], 4), round(v['kl_q95'], 4)) for k, v in res[N].items() if k != 'E'},
              round(time.time() - t0, 1), 's', flush=True)
        json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n1d_small_messages%s.json'
                            % ('_quick' if quick else ''), 'w'), indent=1)


if __name__ == '__main__':
    main(quick=len(sys.argv) > 1)
