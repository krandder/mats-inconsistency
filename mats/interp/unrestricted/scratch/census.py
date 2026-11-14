"""Census of nets for the evaluation (rule fixed in PREREG before any estimator result on these seeds).
For each seed from seed0 upward: pilot of 2^14 radius-exact sphere directions (noiseless). The net is 'healthy' if
the fraction of directions with a_L = 0 (all outputs zero) is at most 0.5 and the angular variance is positive.
Takes the first K healthy seeds per (n, L). Writes runs/census.json.
Usage: census.py K seed0 configs  (configs as n:L,n:L,...)
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import Net, sphere_from_gauss  # noqa: E402

K = int(sys.argv[1]); seed0 = int(sys.argv[2])
configs = [tuple(int(v) for v in c.split(':')) for c in sys.argv[3].split(',')]
out = {}
for (n, L) in configs:
    rng = np.random.default_rng([31337, n, L])
    U = sphere_from_gauss(rng.standard_normal((1 << 14, n)))
    picked, tried = [], []
    s = seed0
    while len(picked) < K and s < seed0 + 2000:
        net = Net(s, n, L)
        g = net.forward(U, noise=False)
        dead = float(np.mean(np.all(g == 0, axis=1)))
        var = float(np.sum(g.var(0)))
        ok = dead <= 0.5 and var > 0
        tried.append(dict(seed=s, dead=dead, ang_var=var, healthy=ok))
        if ok:
            picked.append(s)
        s += 1
    out[f'{n}:{L}'] = dict(n=n, L=L, picked=picked, tried=len(tried),
                           healthy_rate=len(picked) / max(1, len(tried)), detail=tried)
    print(f'n={n} L={L}: picked {picked} after {len(tried)} tried', flush=True)
with open(os.path.join(os.path.dirname(__file__), '..', 'runs', 'census.json'), 'w') as f:
    json.dump(out, f)
