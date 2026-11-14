"""Independent numerical checks of gpt-6-astra round-1 lemmas (cut-rank lane).
(1) averaged gradient identity E_b ||Dg_b(h) v||^2 = ||v||^2 for multi-layer He-ReLU backs;
(2) Funk-Hecke eigenvalues eta_l of (u.v)_+ on S^{k-1}: closed form vs quadrature;
(3) one-layer parity: odd degrees >= 3 vanish (part of (2));
(4) rank-one collapse in the kernel limit: Err_1/Vbar = (q_{c+d} - q_{d+1}) / (1 - q_{c+d}) vs 2(c-1)/d.
"""
import math, os, sys
import numpy as np
from scipy import special
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import crnet as cn

rng = np.random.default_rng(5)
# (1) derivative of the back at a post-ReLU cut state h along v, averaged over random backs
for n, d in ((8, 4), (16, 8), (32, 16)):
    h = np.abs(rng.standard_normal(n)); v = rng.standard_normal(n)
    vals = []
    for b in range(4000):
        u = v.copy(); x = h.copy()
        for W in cn.back_weights(cn.cfg_tag(n, 0, d, 'gradid'), n, d, b):
            W = W.astype(np.float64); z = W @ x; m = z > 0
            u = m * (W @ u); x = np.maximum(z, 0)
        vals.append(u @ u)
    vals = np.array(vals) / (v @ v)
    print(f'(1) n={n} d={d}: E||Dg v||^2/||v||^2 = {vals.mean():.4f} +- {vals.std() / np.sqrt(len(vals)):.4f}')


def eta_closed(k, l):
    if l == 0:
        return math.gamma(k / 2) / (2 * math.sqrt(math.pi) * math.gamma((k + 1) / 2))
    if l == 1:
        return 1 / (2 * k)
    if l % 2 == 1:
        return 0.0
    m = l // 2
    return (-1) ** (m - 1) * math.gamma(k / 2) * math.gamma(m - 0.5) / (4 * math.pi * math.gamma(m + (k + 1) / 2))


# (2), (3): eta_l = E_t[t_+ P_l(t)] for t = u.e, u uniform on S^{k-1}, P_l normalised (P_l(1) = 1)
for k in (3, 5, 8):
    a = (k - 3) / 2
    x, w = special.roots_jacobi(3000, a, a)
    w = w / w.sum()
    f = np.maximum(x, 0)
    Pm, P = np.ones_like(x), x.copy()
    errs = [abs(np.sum(w * f) - eta_closed(k, 0))]
    for l in range(1, 13):
        errs.append(abs(np.sum(w * f * P) - eta_closed(k, l)))
        Pm, P = P, ((2 * l + k - 2) * x * P - l * Pm) / (l + k - 2)
    print(f'(2,3) k={k}: max |eta_quadrature - eta_closed| over l <= 12: {max(errs):.2e}')

# (4)
Fm = lambda t: (np.sqrt(1 - t * t) + (np.pi - np.arccos(t)) * t) / np.pi
q = [0.0]
for _ in range(5000):
    q.append(Fm(q[-1]))
for c, d in ((8, 64), (8, 512), (16, 1024), (4, 4000)):
    r = (q[c + d] - q[d + 1]) / (1 - q[c + d])
    print(f'(4) c={c} d={d}: Err1/Vbar = {r:.4f}  vs 2(c-1)/d = {2 * (c - 1) / d:.4f}')
print('(4) 1 - q_t vs 9 pi^2/(2 t^2):', [(t, round(1 - q[t], 7), round(9 * np.pi ** 2 / (2 * t * t), 7)) for t in (100, 1000)])
