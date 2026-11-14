"""E3: does T1 / bridge-vs-direct agreement at high AUC validate log w-hat or the interior L(alpha)?
Known answer: p = N(0, I_d), q = N(delta e1, I_d), G = h1^2. p_alpha = N((1-alpha) delta e1, I) so
L(alpha) = 1 + ((1-alpha) delta)^2 exactly. Per-cell pooled bridge exactly as lib/r1est._bridge_cells
(8 real + 8 synthetic per cell, per-cell normalisation), with the true log w and with distorted log w-hat
that keep the ranking (hence the AUC) but change the magnitude."""
import sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from cmlib.stats import auc


def bridge_cells(Gp, lwp, Gq, lwq, alphas):
    G = np.concatenate([Gp, Gq], 1); lw = np.concatenate([lwp, lwq], 1)
    lc = -np.logaddexp(0, -lw); l1c = -np.logaddexp(0, lw)
    out, ess = [], []
    for a in alphas:
        lo = a * lc + (1 - a) * l1c
        om = np.exp(lo - lo.max(1, keepdims=True)); om /= om.sum(1, keepdims=True)
        out.append((om * G).sum(1).mean()); ess.append((1 / (om * om).sum(1)).mean())
    return np.array(out), np.array(ess)


rng = np.random.default_rng(0)
d, delta, n, M = 8, 2.8, 6000, 8
alphas = np.array([0, .1, .2, .3, .5, .7, 1.0])
exact = 1 + ((1 - alphas) * delta) ** 2
lwf = lambda H: -delta * H[..., 0] + delta ** 2 / 2          # log p/q exactly


def draw(n_, M_):
    Hp = rng.standard_normal((n_, M_, d)); Hq = rng.standard_normal((n_, M_, d)); Hq[..., 0] += delta
    return Hp[..., 0] ** 2, lwf(Hp), Hq[..., 0] ** 2, lwf(Hq)


Gp, lwp, Gq, lwq = draw(n, M)
dgap = Gq.mean() - Gp.mean()
print('AUC (true log w): %.4f' % auc(lwp.ravel(), lwq.ravel()))
print('direct: L(q) %.3f  L(p) %.3f  gap %.3f   exact gap %.3f' % (Gq.mean(), Gp.mean(), dgap, delta ** 2))
print('alphas        :', alphas)
print('exact L(alpha):', np.round(exact, 2))
for lab, f in [('true log w', lambda x: x), ('0.5 x log w', lambda x: 0.5 * x), ('2 x log w', lambda x: 2 * x),
               ('sign|lw|^1.5', lambda x: np.sign(x) * np.abs(x) ** 1.5), ('lw + 3 (E_q w = e^3)', lambda x: x + 3)]:
    L, ess = bridge_cells(Gp, f(lwp), Gq, f(lwq), alphas)
    print('%-22s L: %s  gap %.3f (%.1f%% of direct)  per-cell ESS min %.1f/16  max|L-exact| interior %.2f' % (
        lab, np.round(L, 2), L[0] - L[-1], 100 * (L[0] - L[-1]) / dgap, ess.min(), np.abs(L - exact)[1:-1].max()))
for M_, n_ in ((8, 6000), (32, 1500), (128, 400)):
    a, b, c, e = draw(n_, M_)
    L, ess = bridge_cells(a, b, c, e, alphas)
    print('true w, M=%3d per class per cell: L %s  interior bias %s' % (M_, np.round(L, 2), np.round((L - exact)[1:-1], 2)))
