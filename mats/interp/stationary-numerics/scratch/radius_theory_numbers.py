"""Theory numbers for the radius PREREG (no simulation, no solver data):
1. Median coefficients: median(Q)/n = x0 + m1/n + m2/n^2 from the KM expansion F0 + F1/n + F2/n^2.
2. Large-deviation rate of the left tail at large n: P(r <= rho) ~ exp(-n J(rho)), r = 1 + A r_prev, where
   A = (2/n) chi2_M has LDP rate n I_A(a), I_A = Legendre transform of Lambda(t) = log((1 + (1-4t)^-1/2)/2).
   J solves the Bellman equation J(rho') = inf_{rho >= 1} [J(rho) + I_A((rho'-1)/rho)], J(inf) = 0
   (value iteration on a log grid, boundary J(rho) = 0.4/rho at rho >= rho_max).
   Also for Q = (S/n) r: K(q) = inf_s [J(q/s) + (s - 1 - log s)/2]."""
import math, json
import numpy as np
from scipy.optimize import brentq, minimize_scalar

x0 = 0.4 / math.log(2)


def P0(x): return 0.4 * x ** -2 * math.exp(-0.4 / x)
def dP0(x): return P0(x) * (-2 / x + 0.4 / x ** 2)
def F1(x): return -(65 * x - 47) * math.exp(-0.4 / x) / (375 * x ** 2)
def dF1(x, e=1e-6): return (F1(x + e) - F1(x - e)) / (2 * e)
def F2(x): return -(198375 * x ** 3 - 18825 * x ** 2 + 63640 * x - 6627) * math.exp(-0.4 / x) / (843750 * x ** 4)


m1 = -F1(x0) / P0(x0)
m2 = -(F2(x0) + dF1(x0) * m1 + 0.5 * dP0(x0) * m1 ** 2) / P0(x0)
print('median/n = %.10f + (%.10f)/n + (%.8f)/n^2' % (x0, m1, m2))


def Lam(t):
    return math.log(0.5 * (1.0 + (1.0 - 4.0 * t) ** -0.5))


def IA(a):
    """rate of A <= a (a < 1) or A >= a (a > 1): sup_t [t a - Lam(t)], t < 1/4."""
    if a <= 0:
        return math.log(2.0)
    f = lambda t: -(t * a - Lam(t))
    if a < 1:
        r = minimize_scalar(f, bounds=(-1e7, 0.0), method='bounded', options={'xatol': 1e-12})
    else:
        r = minimize_scalar(f, bounds=(0.0, 0.25 - 1e-12), method='bounded', options={'xatol': 1e-14})
    return -r.fun


# tabulate I_A on a log grid of a
la = np.linspace(-40, 3, 4301)
IAt = np.array([IA(math.exp(v)) for v in la])
print('I_A(0+) -> %.6f (ln 2 = %.6f); I_A(1/2) = %.6f; I_A(0.9) = %.6f vs (0.1)^2/10 = %.6f' % (
    IAt[0], math.log(2), IA(0.5), IA(0.9), 0.001))


def IA_interp(a):
    a = np.asarray(a)
    out = np.full(a.shape, math.log(2.0))
    pos = a > 0
    out[pos] = np.interp(np.log(a[pos]), la, IAt, left=math.log(2.0), right=np.inf)
    return out


# value iteration for J on rho in [1 + 1e-6, rho_max], grid in log(rho - 1)
lv = np.linspace(math.log(1e-6), math.log(2e4), 3001)
rho = 1.0 + np.exp(lv)
rho_max = rho[-1]
J = 0.4 / rho
for it in range(400):
    # J_new(rho') = min over source rho of J(rho) + I_A((rho'-1)/rho), plus the source rho = infinity (cost I_A(0))
    a = (rho[:, None] - 1.0) / rho[None, :]
    cost = J[None, :] + IA_interp(a)
    Jn = np.minimum(cost.min(1), math.log(2.0))
    Jn[rho > 2000] = 0.4 / rho[rho > 2000]
    d = np.max(np.abs(Jn - J))
    J = Jn
    if d < 1e-12:
        break
print('value iteration converged after', it, 'iterations, last change', d)
out = {}
for r in (1.01, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0):
    Jr = float(np.interp(math.log(r - 1), lv, J))
    out[str(r)] = Jr
    print('J(%g) = %.6f   0.4/rho = %.6f   ratio %.4f' % (r, Jr, 0.4 / r, Jr * r / 0.4))


def Jf(r):
    return float(np.interp(math.log(max(r - 1, 1e-6)), lv, J)) if r > 1 else math.inf


Kq = {}
for q in (1.0, 2.0, 4.0):
    f = lambda ls: Jf(q / math.exp(ls)) + (math.exp(ls) - 1 - ls) / 2
    r = minimize_scalar(f, bounds=(math.log(1e-3), math.log(q) - 1e-9), method='bounded')
    Kq[str(q)] = float(r.fun)
    print('K(Q<=%g) = %.6f at S/n = %.4f' % (q, r.fun, math.exp(r.x)))
json.dump(dict(median=dict(x0=x0, m1=m1, m2=m2), J=out, K=Kq, IA_half=IA(0.5)),
          open('/home/agent/workspaces/mats/work/stationary-numerics/results/radius_theory_numbers.json', 'w'), indent=1)
