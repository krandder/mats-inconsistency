"""Convergence-checked ray-front ranks (replaces the uncertified fine-precision entries of logs/ray_family*.log).

gpt-6-astra (coprover/astra_r1_answer.md, section 3) showed the ray table's truncation diagnostic can fail (softplus16:
the Hermite coefficients beyond K = 6000 carry more energy than the eps = 1e-6 budget). Here every arm is computed at
two truncations, K1 and K2 = 2 K1 (24 radial nodes), plus K2 with 48 radial nodes; an entry is reported only if the three
ranks agree within 2% and no run flags energy beyond its degrees; otherwise "unres" (a convergence check, not a certificate). Uses scratch/ray_family.py's functions unchanged.
usage: python3 scratch/ray_cert.py   -> logs/ray_cert.log, runs/ray_cert.json
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import ray_family as RF
import acts as A

EPS = RF.EPS; NS = RF.NS; S0 = RF.S0
ARMS = [('softplus', 16.0, False), ('softplus', 8.0, False), ('softplus', 4.0, False), ('softplus', 2.0, False),
        ('softplus', 1.0, False), ('gelu', None, False), ('silu', None, False), ('gsrelu', 0.5, False),
        ('gsrelu', 1.0, False), ('silu', None, True), ('gelu', None, True), ('linear', None, True),
        ('relu2', None, False), ('smoothstep', 1.0, False)]
KS = {'relu2': (3000, 6000), 'smoothstep': (3000, 6000)}


_RN = RF.radial_nodes


def ranks_at(name, p, glu, vR, K, m=24):
    RF.K = K
    RF.radial_nodes = lambda v, m_=m: _RN(v, m_)
    b, Vbar, miss = RF.cstats(name, p, S0, vR, glu)
    out = {}
    for n in NS:
        Lmax = min(len(b) - 1, 600)
        E = RF.degree_energy(b, n, Lmax)
        total = float(b.sum()) + miss
        beyond = total - E.sum()
        row = []
        for e in EPS:
            thr = e * e * Vbar
            if beyond > thr:
                row.append(None); continue
            r, ok = RF.rank_from_degrees(E, n, total, thr)
            row.append(r if ok else None)
        out[n] = row
    return out, Vbar


log = open(os.path.join(HERE, '..', 'logs', 'ray_cert.log'), 'w')


def pr(s):
    print(s, flush=True); log.write(s + '\n'); log.flush()


res = {}
pr(f'# convergence-checked ray ranks; eps = {EPS}; sampling 1/eps^2 = {[f"{1 / e ** 2:.0e}" for e in EPS]}')
for vR in (1.0, 0.1):
    pr(f'## vR = {vR} (ReLU: the lane\'s closed form, exact)')
    for n in NS:
        pr(f'{"relu":14s} n={n:3d} ' + ' '.join(f'{RF.relu_rank(n, e, vR):9.3g}' for e in EPS))
    for name, p, glu in ARMS:
        lab = ('glu-' if glu else '') + A.label(name, p)
        K1, K2 = KS.get(name, (6000, 12000))
        a, V1 = ranks_at(name, p, glu, vR, K1)
        b, V2 = ranks_at(name, p, glu, vR, K2)
        c, V3 = ranks_at(name, p, glu, vR, K2, m=48)          # radial-quadrature check
        res[f'{lab}|{vR}'] = {}
        for n in NS:
            row = []
            for x, y, z in zip(a[n], b[n], c[n]):
                ok = all(v is not None for v in (x, y, z)) and abs(x - y) <= 0.02 * max(y, 1) and abs(z - y) <= 0.02 * max(y, 1)
                row.append(y if ok else None)
            res[f'{lab}|{vR}'][str(n)] = row
            pr(f'{lab:14s} n={n:3d} ' + ' '.join('    unres' if r is None else f'{r:9.3g}' for r in row)
               + f'   (K = {K1}/{K2}, Vbar {V2:.3g})')
json.dump(res, open(os.path.join(HERE, '..', 'runs', 'ray_cert.json'), 'w'), indent=1)
pr('DONE')
