"""A4b: robustness of the exponent beta in nMSE = C tail^beta (cut 32, 256 x 64, net means).
Points: (mean first-order tail over nets 113-116 from a4_fit.py, mean nMSE at L = 64). PB0/8/16/32 nMSE from the
nongauss summary (R1), the rest from its Amendment 1 files. Bootstrap over points.
python a4_robust.py"""
import numpy as np

pts = {'PB0': (1.0, 1.8e-4), 'PB8': (0.2969, 1.1e-4), 'PBW8': (0.2349, 6.97e-5), 'PB16': (0.1376, 4.4e-5),
       'PBW16': (0.1096, 3.89e-5), 'PBS16': (0.08671, 2.30e-5), 'PBA8': (0.08495, 2.13e-5), 'PB32': (0.05734, 1.6e-5),
       'PBW32': (0.04191, 8.95e-6), 'PBA16': (0.02562, 5.95e-6), 'PB64': (0.01502, 1.35e-6)}


def fit(keys):
    x = np.log([pts[k][0] for k in keys]); y = np.log([pts[k][1] for k in keys])
    A = np.vstack([np.ones_like(x), x]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None); p = A @ c
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(keys), len(keys))
        if len(set(i.tolist())) < 3:
            continue
        cc, *_ = np.linalg.lstsq(A[i], y[i], rcond=None); bs.append(cc[1])
    return c[1], np.exp(c[0]), r2, np.percentile(bs, [2.5, 97.5])


base = ['PB0', 'PB8', 'PBW8', 'PB16', 'PBW16', 'PBS16', 'PBA8', 'PB32', 'PBW32', 'PBA16']
for name, keys in [('base (10 points, nMSE > 2e-6)', base), ('without PB0', base[1:]), ('with PB64', base + ['PB64']),
                   ('PB ladder only (0, 8, 16, 32, 64)', ['PB0', 'PB8', 'PB16', 'PB32', 'PB64']),
                   ('other families only', ['PBW8', 'PBW16', 'PBS16', 'PBA8', 'PBW32', 'PBA16'])]:
    b, C, r2, ci = fit(keys)
    print(f"{name:36s} beta {b:.2f} (bootstrap 95% {ci[0]:.2f}-{ci[1]:.2f}), C {C:.2e}, R^2 {r2:.3f}")
