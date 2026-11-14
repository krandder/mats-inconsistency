"""M4: distinct gate patterns (all layers) met by 2^m Fibonacci-sphere directions, n = 3, given seeds.
Usage: regions.py L seeds(comma) [--m 22]. Appends to runs/regions_n3.json."""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import Net
ap = argparse.ArgumentParser(); ap.add_argument('L', type=int); ap.add_argument('seeds'); ap.add_argument('--m', type=int, default=22)
a = ap.parse_args()
N = 1 << a.m
i = np.arange(N) + 0.5
z = 1 - 2 * i / N; r = np.sqrt(1 - z * z); ph = i * math.pi * (3 - math.sqrt(5))
U = np.stack([r * np.cos(ph), r * np.sin(ph), z], axis=1)
rng = np.random.default_rng(7)
fn = os.path.join(os.path.dirname(__file__), '..', 'runs', 'regions_n3.json')
res = json.load(open(fn)) if os.path.exists(fn) else []
for s in [int(x) for x in a.seeds.split(',')]:
    net = Net(s, 3, a.L)
    R = rng.integers(1, 2 ** 63, size=(a.L * 3,), dtype=np.uint64) | np.uint64(1)
    hs = []
    for c in range(0, N, 1 << 18):
        h = U[c:c + (1 << 18)]
        acc = np.zeros(len(h), dtype=np.uint64)
        for l, Wl in enumerate(net.W):
            zz = h @ Wl.T
            g = zz > 0
            for j in range(3):
                acc += g[:, j].astype(np.uint64) * R[3 * l + j]
            h = np.maximum(zz, 0)
        hs.append(acc)
    cnt = int(len(np.unique(np.concatenate(hs))))
    res.append(dict(n=3, L=a.L, seed=s, m=a.m, patterns=cnt))
    print(f'L={a.L} seed={s}: {cnt} distinct gate patterns on 2^{a.m} directions', flush=True)
json.dump(res, open(fn, 'w'))
