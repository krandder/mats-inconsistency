"""M3: infinite-width ReLU kernel kappa_L (normalised arc-cosine map iterated L times) and the full-RSB / no-overlap-gap
condition of spherical spin glasses: (xi'')^{-1/2} concave on (0, 1), with xi = kappa_L (the linear and constant
terms, an external field and a mean, do not enter xi''). Writes runs/spin_glass.json."""
import json, math, os
import numpy as np

def kap(t):
    return (np.sqrt(np.clip(1 - t * t, 0, None)) + (math.pi - np.arccos(np.clip(t, -1, 1))) * t) / math.pi
def kap1(t):
    return (math.pi - np.arccos(np.clip(t, -1, 1))) / math.pi
def kap2(t):
    return 1.0 / (math.pi * np.sqrt(np.clip(1 - t * t, 1e-300, None)))

t = np.linspace(1e-4, 1 - 1e-6, 200001)
out = {}
f, f1, f2 = t.copy(), np.ones_like(t), np.zeros_like(t)
for L in range(1, 65):
    f, f1, f2 = kap(f), kap1(f) * f1, kap2(f) * f1 ** 2 + kap1(f) * f2
    if L in (1, 2, 4, 8, 16, 32, 64):
        h = f2 ** -0.5
        # second differences relative to the local scale
        d2 = (h[2:] - 2 * h[1:-1] + h[:-2])
        scale = np.max(np.abs(h)) * (t[1] - t[0]) ** 2
        worst = float(np.max(d2) / scale)
        imax = int(np.argmax(d2)) + 1
        out[L] = dict(max_rel_second_diff=worst, at_t=float(t[imax]), concave=bool(worst <= 1e-6),
                      kappa0=float(f[0]), xi2_at_0=float(f2[0]))
        print(f'L={L}: kappa_L(0)={f[0]:.4f} max normalised h\'\'={worst:.3e} at t={t[imax]:.4f} concave={worst <= 1e-6}')
json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'runs', 'spin_glass.json'), 'w'))
