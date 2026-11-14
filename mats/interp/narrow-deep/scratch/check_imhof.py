"""Check the Imhof generalised chi-square survival function used in p3_tails.py against Monte Carlo."""
import numpy as np
from scipy.integrate import quad


def imhof_sf(x, lam, delta2):
    lam = np.asarray(lam, float); d2 = np.asarray(delta2, float)

    def f(u):
        if u == 0:
            return 0.5 * (np.sum(lam * (1 + d2)) - x)
        lu = lam * u
        th = 0.5 * np.sum(np.arctan(lu) + d2 * lu / (1 + lu * lu)) - 0.5 * x * u
        rho = np.exp(0.25 * np.sum(np.log1p(lu * lu)) + 0.5 * np.sum(d2 * lu * lu / (1 + lu * lu)))
        return np.sin(th) / (u * rho)
    v, _ = quad(f, 0, np.inf, limit=2000, epsabs=1e-13, epsrel=1e-10)
    return 0.5 + v / np.pi


rng = np.random.default_rng(0)
lam = np.array([3.0, 1.0, 0.5, 0.2]); d = np.array([1.0, 0.0, 2.0, 0.5])
z = rng.standard_normal((4_000_000, 4)); Q = ((z + d) ** 2 * lam).sum(1)
for x in (5.0, 15.0, 30.0, 45.0):
    print(f"x={x}: Imhof {imhof_sf(x, lam, d ** 2):.6g}  MC {(Q > x).mean():.6g}")
