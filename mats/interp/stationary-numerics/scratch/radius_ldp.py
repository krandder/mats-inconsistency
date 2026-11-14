"""Left-tail large-deviation rate J(rho): P(r <= rho) ~ exp(-n J(rho)) as n -> infinity (theory, no data).
Bellman: J(rho') = min( ln 2 , min_{rho >= 1} [ J(rho) + I_A((rho' - 1)/rho) ] ), J(rho) -> 0.4/rho + b/rho^2 (rho -> inf),
b = 4 alpha3 - 1/5, alpha3 = kappa3/(6 sigma^6) = 44/750 (cubic coefficient of I_A(1 - e)); derivation in notes.
Value iteration on a uniform rho-grid (sources rho = rho' + D, D on the same grid), boundary formula for rho > rho_b.
usage: radius_ldp.py drho rho_b"""
import sys, math, json
import numpy as np
from scipy.optimize import minimize_scalar

drho = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
rho_b = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0


def Lam(t):
    return math.log(0.5 * (1.0 + (1.0 - 4.0 * t) ** -0.5))


def IA(a):
    if a <= 0:
        return math.log(2.0)
    f = lambda t: -(t * a - Lam(t))
    if a < 1:
        lo = -1e9 if a < 1e-6 else -1e5
        r = minimize_scalar(f, bounds=(lo, 0.0), method='bounded', options={'xatol': 1e-13})
    else:
        r = minimize_scalar(f, bounds=(0.0, 0.25 - 1e-13), method='bounded', options={'xatol': 1e-15})
    return -r.fun


alpha3 = 44.0 / 750.0
b = 4 * alpha3 - 0.2
# I_A table on a log grid, dense near a = 1
la = np.concatenate([np.linspace(-60, -0.2, 3000), np.linspace(-0.2, 0.2, 8001)[1:]])
IAt = np.array([IA(math.exp(v)) for v in la])


def IAv(a):
    out = np.full(a.shape, math.log(2.0))
    pos = a > 0
    out[pos] = np.interp(np.log(a[pos]), la, IAt)
    return out


def Jb(r):
    return 0.4 / r + b / r ** 2


rho = 1.0 + drho * np.arange(1, int(round((rho_b - 1) / drho)) + 1)
Dmax = 25.0
D = drho * np.arange(-int(round((rho[-1] - 1) / drho)), int(round(Dmax / drho)) + 1)
# cost matrix C[i, k] = I_A((rho_i - 1)/(rho_i + D_k)) for sources rho_i + D_k >= 1
src = rho[:, None] + D[None, :]
valid = src >= 1.0 + 0.5 * drho
a = np.where(valid, (rho[:, None] - 1.0) / np.where(valid, src, 1.0), 0.0)
C = np.where(valid, IAv(a), np.inf)
idx = np.rint((src - 1.0) / drho).astype(int) - 1  # index of the source in rho (if inside)
inside = valid & (idx >= 0) & (idx < rho.size)
J = Jb(rho)
for it in range(5000):
    Jsrc = np.where(inside, J[np.clip(idx, 0, rho.size - 1)], Jb(np.where(valid, src, 2.0)))
    Jn = np.minimum((Jsrc + C).min(1), math.log(2.0))
    d = float(np.max(np.abs(Jn - J)))
    J = Jn
    if d < 1e-13:
        break
res = {}
for r in (1.01, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0):
    if r <= rho[-1]:
        res[str(r)] = float(np.interp(r, rho, J))
print('drho', drho, 'rho_b', rho_b, 'iterations', it, 'last change %.2e' % d)
for k, v in res.items():
    r = float(k)
    print('J(%s) = %.7f   asympt 0.4/r + b/r^2 = %.7f' % (k, v, Jb(r)))
# optimal step D at a few rho
for r in (2.0, 5.0, 20.0, 80.0):
    i = int(np.argmin(np.abs(rho - r)))
    Jsrc = np.where(inside[i], J[np.clip(idx[i], 0, rho.size - 1)], Jb(np.where(valid[i], src[i], 2.0)))
    k = int(np.argmin(Jsrc + C[i]))
    print('rho = %.2f: optimal previous rho - rho = %.3f' % (rho[i], D[k]))
np.savez('/home/agent/workspaces/mats/work/stationary-numerics/results/radius_ldp_d%g_b%g.npz' % (drho, rho_b), rho=rho, J=J)
# K(q): rate of P(Q <= q), Q = (S/n) r, S/n has rate (s - 1 - log s)/2
K = {}
for q in (1.0, 2.0, 4.0):
    ss = np.exp(np.linspace(math.log(1e-3), math.log(q) - 1e-9, 20001))
    rr = q / ss
    Jr = np.where(rr <= rho[-1], np.interp(rr, rho, J, left=math.log(2.0)), Jb(rr))
    tot = Jr + (ss - 1 - np.log(ss)) / 2
    k = int(np.argmin(tot))
    K[str(q)] = float(tot[k])
    print('K(Q<=%g) = %.6f at S/n = %.4f, r = %.4f' % (q, tot[k], ss[k], rr[k]))
json.dump(dict(drho=drho, rho_b=rho_b, b=b, J=res, K=K), open(
    '/home/agent/workspaces/mats/work/stationary-numerics/results/radius_ldp_d%g_b%g.json' % (drho, rho_b), 'w'), indent=1)
