"""Part 2 (PREREG Part 2): level variances of the multilevel shared-trajectory sampler (work/unrestricted/lib/mlmc.py) on the
Part 1 nets (sigma = 0), against the depth profile of the amplitude.
usage: part2_mlmc.py n seed [--P 8192] [--out file]
For each depth L in {n/4, n/2, n, 2n} (+ 32 if >= n/4): one reference direction (seed [7401, seed, n, L]); prefix depths
r in {1, 2, 4, ..., L/2}; on P unit directions (seed [7402, seed, n, L]):
  V_sph = tr Var(m_n h_L(U)) (radius-exact MC variance per sample),
  V1(r) = tr Var(G_r(U)) (the level-1 control), V2(r) = tr Var(Y - G_r(U)) (the final level),
  two-level ideal gain over RB-MC: V_sph / (sqrt(V1 r/L) + sqrt(V2))^2 (setup and pilot excluded, as in the unrestricted lane),
and the amplitude-profile prediction V2(r)/V_sph ~ s'_dir + s'_amp (Lambda(L) - Lambda(r))/Lambda(L) with
s'_amp = V_amp/(V_amp + V_dir), s'_dir = V_dir/(V_amp + V_dir) measured on the same sample."""
import argparse, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/unrestricted/lib')
import ib
from mlmc import ReferencePath, level_samples

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--P', type=int, default=8192); ap.add_argument('--out')
a = ap.parse_args()
n, seed, P = a.n, a.seed, a.P
t0 = time.time()
Ls = sorted({L for L in {n // 4, n // 2, n, 2 * n, 32} if L >= n // 4})
net = ib.Net(seed, n, max(Ls))


def J2(t):
    return 3 * math.sin(t) * math.cos(t) + (math.pi - t) * (1 + 2 * math.cos(t) ** 2)


rho = 0.0; Lam = [0.0]
for l in range(1, 2 * max(Ls) + 1):
    th = math.acos(rho); Lam.append(Lam[-1] + 6 - (2 / math.pi) * J2(th)); rho = float(ib.kappa(rho))
res = dict(n=n, seed=seed, P=P, by_depth={})
mn = ib.mean_radius(n)
for L in Ls:
    W = net.W[:L]
    ref = ReferencePath(W, np.random.default_rng([7401, seed, n, L]))
    U = ib.sphere(np.random.default_rng([7402, seed, n, L]).standard_normal((P, n)))
    h = U
    for Wl in W:
        h = np.maximum(h @ Wl.T, 0.0)
    G = h
    Vsph = float(np.sum((mn * G).var(0, ddof=1)))
    vs = ib.var_split(G, n)
    sa = vs['V_amp'] / (vs['V_amp'] + vs['V_dir']); sd = vs['V_dir'] / (vs['V_amp'] + vs['V_dir'])
    rows = []
    r = 1
    while r <= L // 2:
        lv = [0, r]
        d1 = level_samples(W, ref, lv, 1, U)
        d2 = level_samples(W, ref, lv, 2, U)
        V1 = float(np.sum(d1.var(0, ddof=1))); V2 = float(np.sum(d2.var(0, ddof=1)))
        gain = Vsph / (math.sqrt(V1 * r / L) + math.sqrt(V2)) ** 2
        pred = sd + sa * (Lam[L] - Lam[r]) / Lam[L]
        rows.append(dict(r=r, V1=V1, V2=V2, V2_share=V2 / Vsph, pred_share=pred, gain2=gain))
        r *= 2
    best = max(rows, key=lambda x: x['gain2'])
    res['by_depth'][str(L)] = dict(V_sph=Vsph, s_amp_sph=sa, s_dir_sph=sd, rows=rows, best_gain2=best['gain2'], best_r=best['r'])
    print(f'L={L}: V_sph {Vsph:.3f} s\'_dir {sd:.3f} best two-level gain {best["gain2"]:.2f} at r={best["r"]} | '
          + ' '.join(f'r{x["r"]}:{x["V2_share"]:.3f}/{x["pred_share"]:.3f}' for x in rows) + f' | {time.time() - t0:.0f}s', flush=True)
res['sec'] = time.time() - t0
if a.out:
    json.dump(res, open(a.out, 'w'))
print('done', res['sec'])
