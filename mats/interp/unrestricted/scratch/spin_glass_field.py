"""Post hoc (after the registered M3 check): the external-field version of the full-RSB criterion for the
infinite-width ReLU field. xi(q) = kappa_L(q) - kappa_L(0) - h^2 q with h^2 = kappa_L'(0); qhat solves
h^2 + xi'(qhat) = qhat xi''(qhat) (Huang-Sellke 2021, spherical case, formula from memory, unverified); then test
concavity of xi''^{-1/2} on [qhat, 1). Writes runs/spin_glass_field.json."""
import json, math, os
import numpy as np
def kap(t): return (np.sqrt(np.clip(1 - t * t, 0, None)) + (math.pi - np.arccos(np.clip(t, -1, 1))) * t) / math.pi
def kap1(t): return (math.pi - np.arccos(np.clip(t, -1, 1))) / math.pi
def kap2(t): return 1.0 / (math.pi * np.sqrt(np.clip(1 - t * t, 1e-300, None)))
t = np.linspace(0.0, 1 - 1e-6, 400001)
f, f1, f2 = t.copy(), np.ones_like(t), np.zeros_like(t)
res = {}
for L in range(1, 65):
    f, f1, f2 = kap(f), kap1(f) * f1, kap2(f) * f1 ** 2 + kap1(f) * f2
    if L in (1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64):
        h2 = f1[0]
        xi1 = f1 - h2          # xi'(q)
        xi2 = f2               # xi''(q)
        g = h2 + xi1 - t * xi2  # root qhat: g(qhat) = 0 (g(0) = h2 > 0, decreasing)
        idx = np.where(g <= 0)[0]
        qhat = float(t[idx[0]]) if len(idx) else 1.0
        sel = t >= qhat
        H = xi2[sel] ** -0.5
        d2 = H[2:] - 2 * H[1:-1] + H[:-2]
        scale = np.max(np.abs(H)) * (t[1] - t[0]) ** 2
        worst = float(d2.max() / scale) if len(d2) else float('nan')
        res[L] = dict(h2=float(h2), nonconst=float(1 - f[0]), qhat=qhat, max_rel_second_diff_on_qhat_1=worst,
                      concave_on_qhat_1=bool(worst <= 1e-6))
        print(f'L={L}: h^2={h2:.4f} nonconstant mass={1 - f[0]:.4f} qhat={qhat:.3f} max h\'\' on [qhat,1)={worst:.3e} '
              f'concave={worst <= 1e-6}')
json.dump(res, open(os.path.join(os.path.dirname(__file__), '..', 'runs', 'spin_glass_field.json'), 'w'))
