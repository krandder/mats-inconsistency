"""Depth erosion of analyticity (theory numbers): for the geometric-class gates at q0 = 2 (per-layer normalised, so the
infinite-width normalised correlation map f(t) = sum_k b_k t^k / sum_k b_k is the same at every layer), the radius of
convergence of the d-fold composition is R_d = f^{-1}(R_{d-1}) (positive coefficients), R_1 = 1/rho. The Taylor
coefficients of the composed kernel then decay like R_d^{-k}, so the degree needed for a relative tail eps^2 grows like
log(1/eps^2)/log R_d. Prints R_d - 1 and that degree at eps = 0.01 for d = 1, 2, 4, 8, 16, 32, 64.
usage: python3 scratch/depth_radius.py -> logs/depth_radius.log"""
import math, os, sys
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import acts as A
S = math.sqrt(2.0)
out = open(os.path.join(HERE, '..', 'logs', 'depth_radius.log'), 'w')
for name, p, rho in [('gelu', None, S * S / (1 + S * S)), ('gsrelu', 0.5, S * S / (S * S + 0.25)), ('gsrelu', 1.0, S * S / (S * S + 1.0))]:
    K = 3000
    c = A.coef(name, p, K, S); b = c ** 2
    tot = b.sum(); k = np.arange(K + 1)
    R1 = 1 / rho
    pos = b > 0; lb = np.log(b[pos]); kp = k[pos]
    f = lambda t: float(np.sum(np.exp(lb + kp * math.log(t))) / tot)
    chi = float(np.sum(k * b) / tot)
    R = [R1]
    for d in range(2, 65):
        top = R[-1]
        # f is increasing on [1, R1); solve f(t) = top for t in (1, R1)
        hi = R1 * (1 - 1e-9)
        if f(hi) < top:          # f never reaches R_{d-1} on [1, R1): the composition's radius is f's own, R1
            R.append(R1); continue
        R.append(brentq(lambda t: f(t) - top, 1.0 + 1e-15, hi, xtol=1e-15))
    line = f'{A.label(name, p):10s} rho {rho:.4f} chi1 {chi:.3f} | d: R_d-1, D(0.01)=ln(1e4)/ln R_d : ' + '  '.join(
        f'{d}: {R[d - 1] - 1:.2e}, {math.log(1e4) / math.log(R[d - 1]):.0f}' for d in (1, 2, 4, 8, 16, 32, 64))
    print(line); out.write(line + '\n')
