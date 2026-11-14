import sys, functools, numpy as np
sys.path.insert(0, 'lib')
import bnb
from bnb import cell_moments
@functools.lru_cache(maxsize=None)
def nodes(n, q):
    return bnb.gl_nodes(n, q)
rng = np.random.default_rng(1)
for n in (4, 8, 16, 32):
    m = n - 1
    worst = {}
    for trial in range(40):
        c = rng.uniform(-1, 1, m); wd = 10 ** rng.uniform(-5, 0, m)
        lo = np.clip(c - wd, -1, 1); hi = np.clip(c + wd, -1, 1); lo = np.minimum(lo, hi - 1e-9)
        ref = cell_moments(lo[None], hi[None], *nodes(n, 600))
        for q in (32, 48, 64, 96):
            got = cell_moments(lo[None], hi[None], *nodes(n, q))
            rm = abs(got[0][0] / ref[0][0] - 1)
            rf = np.max(np.abs(got[1][0] - ref[1][0])) / (np.max(np.abs(ref[1][0])) + 1e-300)
            rt = abs(got[2][0] / ref[2][0] - 1)
            worst[q] = max(worst.get(q, 0), rm, rf, rt)
    print(n, {q: f"{v:.1e}" for q, v in worst.items()}, flush=True)
