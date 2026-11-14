"""POST HOC diagnostic (not pre-registered): how linear are the random nets of grid K? For each (arm, n) and nets 0-3
(the same nets as grid K: lib/gnet.py, tag 'tgk2'), the share of output variance carried by the degree-1 Hermite chaos,
||E[out x^T]||_F^2 / tr Var(out), and the output's variance-to-mean-energy ratio, at depths 2, 4, 8, 16 (2^17 inputs).
usage: python3 scratch/linfrac_random.py n [arms...] -> appends to logs/linfrac_random.log
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import gnet as G
n = int(sys.argv[1])
ARMS = [a.split(':') for a in (sys.argv[2:] or ['relu', 'softplus:16', 'softplus:8', 'softplus:4', 'softplus:2', 'softplus:1',
                                               'gelu', 'silu', 'gsrelu:0.5', 'smoothstep:1'])]
out = open(os.path.join(HERE, '..', 'logs', 'linfrac_random.log'), 'a')
for a in ARMS:
    name = a[0]; p = float(a[1]) if len(a) > 1 else None
    spec = G.Spec(name, p, 'plain')
    rows = {L: [] for L in (2, 4, 8, 16)}
    for seed in range(4):
        Ws = G.weights(spec, n, 16, G.tag_of('tgk2', spec.label, n), seed)
        X = np.random.default_rng(seed + 100).standard_normal((1 << 17, n))
        H = X
        for l, (W,) in enumerate(Ws, 1):
            H = spec.phi(H @ W.T.astype(np.float64))
            if l in rows:
                mu = H.mean(0); tv = float(np.sum(H.var(0)))
                lin = float(np.sum((X.T @ (H - mu) / len(X)) ** 2) / tv)
                rows[l].append((lin, tv / float(np.sum(mu ** 2))))
    line = f'{spec.label:12s} n={n:3d} ' + '  '.join(
        f'L{L}: lin {np.median([r[0] for r in v]):.3f} var/mean^2 {np.median([r[1] for r in v]):.2e}' for L, v in rows.items())
    print(line, flush=True); out.write(line + '\n'); out.flush()
