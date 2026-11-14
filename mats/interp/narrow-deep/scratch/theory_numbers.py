"""Theory numbers for the PREREG (no network runs): half-width Lyapunov exponents along one path (Theorem C/C1 of
outbox/lyapunov-dimension-2026-10-29.md), dying and bottleneck probabilities, mean-field angle recursion."""
import math
from scipy.special import digamma, comb
import numpy as np

def binom_pmf(n):
    return np.array([comb(n, m) / 2 ** n for m in range(n + 1)])

for n in (4, 8, 16, 32, 256):
    p = binom_pmf(n)
    ms = np.arange(n + 1)
    # radial exponent: E log |h_l|/|h_{l-1}| = 0.5*(log(2/n) + E log chi2_m) = 0.5*(log(4/n) + E psi(m/2)), m>=1
    sel = ms >= 1
    pr = p[sel] / p[sel].sum()
    lam_r = 0.5 * (math.log(4 / n) + (pr * digamma(ms[sel] / 2)).sum())
    out = [f"n={n}: radial lambda_r={lam_r:+.4f}/layer"]
    for i in (1, 2, 3, 4):
        sel = ms > i
        if sel.sum() == 0: break
        pi_ = p[sel] / p[sel].sum()
        mu = 0.5 * (math.log(4 / n) + (pi_ * digamma((ms[sel] - i) / 2)).sum())
        out.append(f"mu_{i}={mu:+.4f} gap={lam_r-mu:.4f}")
    print('  '.join(out))
    pdie = 2.0 ** -n
    for L in (64, 128, 256):
        print(f"   L={L}: P(some layer dead | random path)={1-(1-pdie)**L:.4f}; P(min m_l<=1)={1-(1-p[0]-p[1])**L:.4f}; P(min m_l<=2)={1-(1-p[:3].sum())**L:.4f}; P(min m_l<=3)={1-(1-p[:4].sum())**L:.4f}")
# mean-field angle recursion (infinite width): cos t' = (sin t + (pi - t) cos t)/pi
t = math.pi / 2; th = []
for l in range(1, 257):
    c = (math.sin(t) + (math.pi - t) * math.cos(t)) / math.pi
    t = math.acos(min(1.0, c)); th.append(t)
print("mean-field angle (start pi/2) at l=1,2,4,8,16,32,64,128,256:", [round(th[l - 1], 4) for l in (1, 2, 4, 8, 16, 32, 64, 128, 256)], " 3pi/l at 256:", round(3 * math.pi / 256, 4))
