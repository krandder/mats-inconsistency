"""Infinite-width (mean-field) correlation maps for the gate family at q0 = 2 (theory numbers; no networks).
c_{l+1} = sum_k b_k c_l^k / sum_k b_k with b_k = c_k(phi; sqrt q0)^2 (Mehler), starting from c = 0 (two independent
Gaussian inputs at the first pre-activation). Prints chi_1, c_l at l = 1..16, the fixed point c* and 1 - c_l.
usage: python3 scratch/meanfield.py   -> logs/meanfield.log
"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import acts as A
S = math.sqrt(2.0)
ARMS = [('relu', None), ('softplus', 16.0), ('softplus', 8.0), ('softplus', 4.0), ('softplus', 2.0), ('softplus', 1.0),
        ('gelu', None), ('silu', None), ('tanh', None), ('gsrelu', 0.5), ('smoothstep', 1.0), ('relu2', None)]
log = open(os.path.join(HERE, '..', 'logs', 'meanfield.log'), 'w')
for name, p in ARMS:
    c = A.coef(name, p, 2000, S); b = c ** 2; m, e2 = A.moments(name, p, S)
    b[-1] += max(e2 - b.sum(), 0)              # put the (tiny) missing energy at the top degree
    k = np.arange(len(b))
    f = lambda t: float(np.sum(b * t ** k) / b.sum())
    chi = float(np.sum(k * b) / b.sum())       # slope of the normalised c-map at c = 1
    cs = [0.0]
    for _ in range(16):
        cs.append(f(cs[-1]))
    t = 0.0
    for _ in range(5000):
        t = f(t)
    line = (f'{A.label(name, p):12s} chi1 {chi:.3f} c* {t:.4f} c_l (l=1..16 pre-act corr of two independent inputs): '
            + ' '.join(f'{x:.3f}' for x in cs[:17]))
    print(line); log.write(line + '\n')
