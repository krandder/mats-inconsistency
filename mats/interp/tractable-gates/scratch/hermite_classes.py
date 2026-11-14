"""Hermite spectra of the gate family at the network's pre-activation scale (theory numbers; no networks).

For each activation at s = sqrt(2) (and the softplus beta ladder), prints the nonlinear tail share
T_D / T_1 (T_D = sum_{k > D} c_k^2) at D = 2, 4, 8, 16, 32, 64, 128, 256, local decay fits, and the polynomial-summary
bound of notes/theory.md (Lemma P with the one-layer reduction): D(eps) = min{D : T_D <= eps^2 Var phi(sZ)} and
r_poly = C(n + D, n), against 1/eps^2, for n = 8, 16, 32, 64. Output: logs/hermite_classes.log, runs/hermite_classes.json
usage: python3 scratch/hermite_classes.py
"""
import json, math, os, sys
import numpy as np
from scipy import special as sp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import acts as A

S = math.sqrt(2.0)
K = 4000
ARMS = [('relu', None), ('softplus', 32.0), ('softplus', 16.0), ('softplus', 8.0), ('softplus', 4.0),
        ('softplus', 2.0), ('softplus', 1.0), ('gelu', None), ('silu', None), ('tanh', None), ('gsrelu', 0.25),
        ('gsrelu', 0.5), ('gsrelu', 1.0), ('smoothstep', 1.0), ('relu2', None)]
DS = [2, 4, 8, 16, 32, 64, 128, 256]
EPS = [0.1, 0.03, 0.01, 1e-3, 1e-4, 1e-6]
NS = [8, 16, 32, 64]
log = open(os.path.join(HERE, '..', 'logs', 'hermite_classes.log'), 'w')


def pr(s):
    print(s, flush=True); log.write(s + '\n'); log.flush()


out = {}
pr(f'# s = sqrt 2; columns: T_D/T_1 at D = {DS}; then fits; then D(eps) and log10 C(n+D, n) at eps = {EPS} vs '
   f'log10(1/eps^2) = {[round(math.log10(1 / e ** 2), 1) for e in EPS]}')
for name, p in ARMS:
    m, e2 = A.moments(name, p, S)
    var = e2 - m * m
    if name in ('relu', 'relu2'):          # exact second moment: T_D = E phi^2 - sum_{k<=D} c_k^2 is exact
        e = A.coef(name, p, K, S) ** 2
        T = e2 - np.cumsum(e)
        unres_from = len(T)
    else:                                  # direct tail sums at K and 2K (final-review finding 11: no Parseval residual)
        e1 = A.coef(name, p, K, S) ** 2
        e2K = A.coef(name, p, 2 * K, S) ** 2
        T = np.cumsum(e2K[::-1])[::-1]     # T[j] = sum_{k>=j}; shift to sum_{k>D}
        T = np.concatenate([T[1:], [0.0]])[:K + 1]
        T1 = np.concatenate([np.cumsum(e1[::-1])[::-1][1:], [0.0]])
        gap = T - T1                       # energy in degrees K+1..2K (and any K-dependence of the low coefficients)
        unres_from = None
    miss = 0.0
    Tone = T[1]
    shares = [float(T[D] / Tone) if D < len(T) else None for D in DS]
    # local decay: slope of log T_D against log D (power law) and against sqrt(D) (strip class), on D in [32, 128]
    Dd = np.arange(32, 129)
    ok = Dd < len(T)
    lt = np.log(np.maximum(T[Dd[ok]], 1e-300))   # slopes are descriptive only
    pw = float(np.polyfit(np.log(Dd[ok]), lt, 1)[0]); sq = float(np.polyfit(np.sqrt(Dd[ok]), lt, 1)[0])
    lin = float(np.polyfit(Dd[ok], lt, 1)[0])
    rows = {}
    for n in NS:
        r = []
        for ep in EPS:
            thr = ep * ep * var
            idx = np.nonzero(T <= thr)[0]
            if len(idx) == 0:
                r.append(None); continue
            D = int(idx[0])
            if unres_from is None and abs(gap[D]) > 0.1 * thr:   # degrees K+1..2K matter at this budget: unresolved
                r.append(None); continue
            r.append((D, float((sp.gammaln(n + D + 1) - sp.gammaln(n + 1) - sp.gammaln(D + 1)) / math.log(10))))
        rows[str(n)] = r
    lab = A.label(name, p)
    out[lab] = dict(var=var, shares=shares, slope_logD=pw, slope_sqrtD=sq, slope_D=lin, poly=rows)
    pr(f'{lab:12s} var {var:.3f} shares ' + ' '.join('  -   ' if v is None else f'{v:.1e}' for v in shares)
       + f' | dlogT/dlogD {pw:.2f} dlogT/dsqrtD {sq:.2f} dlogT/dD {lin:.3f}')
    for n in NS:
        pr(f'    n={n:2d} D(eps), log10 r_poly: ' + '  '.join('unres' if v is None else f'{v[0]}:{v[1]:.1f}'
                                                         for v in rows[str(n)]))
json.dump(out, open(os.path.join(HERE, '..', 'runs', 'hermite_classes.json'), 'w'), indent=1)
pr('DONE')
