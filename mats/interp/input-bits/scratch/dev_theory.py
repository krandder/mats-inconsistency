"""DEV (not registered): infinite-width kernel numbers for the input regime, used to write FORMAL v1.
- rho_l: cosine of two independent inputs after l He-ReLU layers (kappa iterated from rho_0 = 0);
- delta_l = 1 - rho_l and its asymptote 44.4 / l^2 (= 4 / (c^2 l^2), c = 2 sqrt 2 / (3 pi));
- gate disagreements per layer n arccos(rho_{l-1}) / pi;
- leading-order amplitude log-variance Var_u(log A_L) = Lambda(L) / (4 n), Lambda(L) = sum_{l=1}^{L} (6 - (2/pi) J_2(theta_{l-1}));
- degree-1 coefficient b_1(L) = prod_l kappa'(rho_{l-1}).
"""
import math
import numpy as np


def kappa(c):
    c = min(max(c, -1.0), 1.0)
    return (math.sqrt(max(1 - c * c, 0.0)) + (math.pi - math.acos(c)) * c) / math.pi


def J2(t):
    return 3 * math.sin(t) * math.cos(t) + (math.pi - t) * (1 + 2 * math.cos(t) ** 2)


rho = 0.0
Lam = 0.0
b1 = 1.0
c = 2 * math.sqrt(2) / (3 * math.pi)
print(' l    rho_l    delta_l   44.4/l^2   disagree_frac  6-2J2/pi   Lambda(l)   b1(l)   b1/delta')
for l in range(1, 4097):
    th = math.acos(rho)
    Lam += 6 - (2 / math.pi) * J2(th)
    b1 *= (math.pi - th) / math.pi
    dis = th / math.pi
    rho = kappa(rho)
    d = 1 - rho
    if l in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        print(f'{l:5d} {rho:.6f} {d:.3e} {44.4 / l ** 2:.3e} {dis:.4f} {6 - 2 * J2(th) / math.pi:.4f} {Lam:9.3f} {b1:.3e} {b1 / d:.3e}')
