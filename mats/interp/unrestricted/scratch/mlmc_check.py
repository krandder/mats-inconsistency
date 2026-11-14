"""Post hoc (after astra round 3): variance-times-cost of astra's multilevel reference-trajectory estimator against
radius-exact MC (sigma = 0). Reference direction Q: q_0 = Q, nu_l = |relu(W_l q_{l-1})|, q_l = relu(W_l q_{l-1}) / nu_l,
s_r = prod_{l > r} nu_l. Control G_r(U) = m_n s_r |h_r(U)| q_L, with h_r(U) the state after r layers on unit input U;
Y(U) = m_n h_L(U). Levels r_0 = 0 < r_1 = 1 < 2 < 4 < ... < L/2, last level Y - G_{r_{J-1}} at cost 1 pass;
level j costs r_j / L passes per sample. Ideal MLMC variance x cost (setup excluded) = (sum_j sqrt(V_j C_j))^2, against
RB-MC's V_sph (per-net, sum over outputs). Also the two-level version with one prefix depth r (best over dyadic r).
Ratio < 1 means the multilevel estimator is better. Usage: mlmc_check.py n L seeds(comma) [--P 4096]
"""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import Net, mean_radius, sphere_from_gauss  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('L', type=int); ap.add_argument('seeds')
ap.add_argument('--P', type=int, default=4096)
ap.add_argument('--out', default=None)
a = ap.parse_args()
n, L = a.n, a.L
mn = mean_radius(n)
res = []
for s in [int(x) for x in a.seeds.split(',')]:
    net = Net(s, n, L)
    rng = np.random.default_rng([8080, s, n, L])
    q = sphere_from_gauss(rng.standard_normal((1, n)))[0]
    nus = []
    for Wl in net.W:
        z = np.maximum(Wl @ q, 0.0)
        nu = float(np.linalg.norm(z))
        nus.append(nu)
        q = z / nu if nu > 0 else np.eye(n)[0]
    qL = q
    nus = np.array(nus)
    # s_r = prod_{l > r} nu_l (layers indexed 1..L; nus[l-1] is layer l)
    suffix = np.ones(L + 1)
    for r in range(L - 1, -1, -1):
        suffix[r] = suffix[r + 1] * nus[r]
    U = sphere_from_gauss(rng.standard_normal((a.P, n)))
    h = U
    norms = np.zeros((a.P, L + 1)); norms[:, 0] = 1.0
    for l, Wl in enumerate(net.W):
        h = np.maximum(h @ Wl.T, 0.0)
        norms[:, l + 1] = np.linalg.norm(h, axis=1)
    Y = mn * h
    Vsph = float(np.sum(Y.var(0)))
    if Vsph <= 0:
        continue
    G = lambda r: mn * suffix[r] * norms[:, r:r + 1] * qL[None, :]
    levels = [0] + [2 ** k for k in range(int(math.log2(max(1, L // 2))) + 1) if 2 ** k < L]
    Vs, Cs = [], []
    for j in range(1, len(levels)):
        d = G(levels[j]) - G(levels[j - 1])
        Vs.append(float(np.sum(d.var(0)))); Cs.append(levels[j] / L)
    d = Y - G(levels[-1])
    Vs.append(float(np.sum(d.var(0)))); Cs.append(1.0)
    ml = float(sum(math.sqrt(v * c) for v, c in zip(Vs, Cs)) ** 2)
    # two-level: control G_r (cost r/L) and residual Y - G_r (cost 1)
    best2 = None
    for r in [2 ** k for k in range(int(math.log2(L)) + 1)]:
        if r >= L:
            break
        v1 = float(np.sum(G(r).var(0))); v2 = float(np.sum((Y - G(r)).var(0)))
        t = (math.sqrt(v1 * r / L) + math.sqrt(v2 * 1.0)) ** 2
        if best2 is None or t < best2[1]:
            best2 = (r, t)
    Dr = {r: float(np.mean(np.sum((Y - G(r)) ** 2, axis=1))) / Vsph for r in levels + [L]}
    res.append(dict(seed=s, Vsph=Vsph, mlmc_ratio=ml / Vsph, two_level_ratio=best2[1] / Vsph, two_level_r=best2[0],
                    D_over_Vsph=Dr, level_vars=Vs, level_costs=Cs))
    print(f'n={n} L={L} seed={s}: MLMC var*cost / RB-MC = {ml / Vsph:.3f}; two-level {best2[1] / Vsph:.3f} (r={best2[0]}); '
          f'D_r/Vsph at r=1,4,16: ' + ', '.join(f'{Dr.get(r, float("nan")):.3f}' for r in (1, 4, 16)), flush=True)
med = lambda k: float(np.median([x[k] for x in res])) if res else None
out = dict(n=n, L=L, P=a.P, per_net=res, median_mlmc_ratio=med('mlmc_ratio'), median_two_level_ratio=med('two_level_ratio'))
fn = a.out or os.path.join(os.path.dirname(__file__), '..', 'runs', f'mlmc_n{n}_L{L}.json')
json.dump(out, open(fn, 'w'))
print(f'n={n} L={L}: median MLMC ratio {out["median_mlmc_ratio"]:.3f}, two-level {out["median_two_level_ratio"]:.3f}', flush=True)
