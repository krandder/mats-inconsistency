"""POST HOC (after the final review, 00:30Z): scale-normalised cut ranks.

The ensemble eps-rank is dominated by rare high-energy networks when (1+5/n)^L is large (review finding 1). Here each
front row is divided by its RMS output s_f and each back's columns by t_b (separable rescaling keeps the operator
bilinear), V is rescaled the same way, and the ranks are recomputed. This asks: once the multiplicative network scale
is removed, does the low rank of the synchronised configurations persist?
usage: python3 scratch/scale_norm.py runs/NAME.npz ...
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import spec

EPS = ('0.3', '0.1', '0.03')
for path in sys.argv[1:]:
    d = np.load(path)
    M = d['M']; V = d['V']                      # M (F,S,B,n), V (F,B,n)
    F, S, B, n = M.shape
    E2 = V + M.mean(1) ** 2                       # E_x a^2 per (f, b, j)
    s_f = np.sqrt(E2.mean(axis=(1, 2)))           # front scale
    s_f[s_f == 0] = np.inf                        # dead fronts -> zero rows
    t_b = np.sqrt((E2 / s_f[:, None, None] ** 2).mean(axis=(0, 2)))
    t_b[t_b == 0] = np.inf
    Mt = M / s_f[:, None, None, None] / t_b[None, None, :, None]
    Vt = V / s_f[:, None, None] ** 2 / t_b[None, :, None] ** 2
    alive = np.isfinite(s_f) & (s_f < np.inf)
    Ms = [Mt[alive, s].reshape(alive.sum(), -1) for s in range(S)]
    Vbar = float(Vt[alive].mean())
    ti, _ = spec.tails_in(Ms)
    tho = spec.tails_heldout(Ms)
    r_in = [spec.eps_rank(ti, float(e) ** 2 * Vbar) for e in EPS]
    r_ho = [spec.eps_rank(tho, float(e) ** 2 * Vbar) for e in EPS]
    print(f'{os.path.basename(path)[:-4]}: alive fronts {alive.sum()}/{F}; scale-normalised r_in(0.3/0.1/0.03) = {r_in}, '
          f'r_ho = {r_ho}, EM2/V = {np.mean(Mt[alive] ** 2) / Vbar:.2f}')
