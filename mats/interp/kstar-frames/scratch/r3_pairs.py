"""Run C (PREREG Q2): coherence of two inputs' frames against their distance. x' = x cos t + u sin t (u orthogonal to x,
|u| = |x|), same noise draws for both paths (coupled); t = 0 with independent noise as a noise-floor row.
At the main layers: state angle theta_c, O(F, F'), CS of the partner's frame (both ways; raw and D_c-projected),
rho = tr(A A') / sqrt(tr A^2 tr A'^2) (A = J^T J), and the two-replica (annealed Wick) prediction of rho from the pair's
per-layer gate overlaps; mean gate disagreement per layer.
nice -n 19 ionice -c3 python r3_pairs.py --n 64 --seed 400 --L 32 [--pairs 256]"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import kf

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=64); ap.add_argument('--seed', type=int, default=400)
ap.add_argument('--L', type=int, default=32); ap.add_argument('--pairs', type=int, default=256)
ap.add_argument('--batch', type=int, default=64); ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--runs', default=os.path.join(HERE, '..', 'runs'))
A = ap.parse_args()
torch.set_num_threads(A.threads)
n, L, seed = A.n, A.L, A.seed
tag = f"n{n}_s{seed}_L{L}"
MAIN = kf.MAIN_LAYERS[L]
K = {int(c): v for c, v in json.load(open(os.path.join(A.runs, f'A_{tag}.json')))['k'].items()}
net = kf.make_net(seed, n, L)
TS = [0.0, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, math.pi / 2]


def two_replica(q, p1, p2, n):
    """Annealed Wick recursion for F = |Y|_F^2, T2 = (tr Y)^2, Q = tr Y^2, Y_l = A_l..A_1 Y_0 A'_1^T..A'_l^T,
    Y_0 = D_c D'_c. q, p1, p2: (P, L'+1) fractions (index 0 = the cut). Returns F (P,)."""
    v = 2.0 / n
    o0 = q[:, 0] * n
    F, T2, Q = o0.clone(), o0 * o0, o0.clone()
    for l in range(1, q.shape[1]):
        a, b, o = p1[:, l] * n, p2[:, l] * n, q[:, l] * n
        F, T2, Q = (v * v * (a * b * F + o * (T2 + Q)), v * v * (o * o * T2 + o * (F + Q)),
                    v * v * (o * o * Q + o * (T2 + F)))
    return F


t0 = time.time()
rows = {}
g = kf.gen(70_000 + 1000 * seed + n + 7 * L)
for t in TS:
    acc = {c: {key: [] for key in ('theta', 'O', 'CS', 'CSp', 'rho', 'rho_pred', 'dis')} for c in MAIN}
    for b in range(A.pairs // A.batch):
        x = kf.draw_x(A.batch, n, g)
        u = kf.draw_x(A.batch, n, g)
        u = u - (u * x).sum(-1, keepdim=True) / (x * x).sum(-1, keepdim=True) * x
        u = u / u.norm(dim=-1, keepdim=True) * x.norm(dim=-1, keepdim=True)
        x2 = x * math.cos(t) + u * math.sin(t)
        zs, eps = kf.forward(net, x, g)
        zs2, _ = kf.forward(net, x2, g, noise=None if t == 0.0 else eps)
        D, D2 = kf.masks(zs), kf.masks(zs2)
        J1 = kf.downstream_jacobians(net, D, MAIN)
        J2 = kf.downstream_jacobians(net, D2, MAIN)
        for c in MAIN:
            k = K[c]
            s1, V1 = kf.svd_right(J1[c]); s2, V2 = kf.svd_right(J2[c])
            F1, F2 = V1[:, :, :k], V2[:, :, :k]
            O, CSa, _ = kf.scores(F2, V1, s1, k)
            _, CSb, _ = kf.scores(F1, V2, s2, k)
            _, CPa, _ = kf.scores(kf.project_active(F2, D[c - 1]), V1, s1, k)
            _, CPb, _ = kf.scores(kf.project_active(F1, D2[c - 1]), V2, s2, k)
            A1 = J1[c].transpose(-1, -2) @ J1[c]; A2 = J2[c].transpose(-1, -2) @ J2[c]
            rho = (A1 * A2).sum((-1, -2)) / torch.sqrt((A1 * A1).sum((-1, -2)) * (A2 * A2).sum((-1, -2)))
            q = torch.stack([(D[l - 1] * D2[l - 1]).mean(-1) for l in range(c, L + 1)], -1)
            p1 = torch.stack([D[l - 1].mean(-1) for l in range(c, L + 1)], -1)
            p2 = torch.stack([D2[l - 1].mean(-1) for l in range(c, L + 1)], -1)
            F12 = two_replica(q, p1, p2, n)
            F11 = two_replica(p1, p1, p1, n)
            F22 = two_replica(p2, p2, p2, n)
            acc[c]['theta'].append(kf.angle(zs[c - 1], zs2[c - 1]))
            acc[c]['O'].append(O); acc[c]['CS'].append(0.5 * (CSa + CSb)); acc[c]['CSp'].append(0.5 * (CPa + CPb))
            acc[c]['rho'].append(rho); acc[c]['rho_pred'].append(F12 / torch.sqrt(F11 * F22))
            acc[c]['dis'].append(torch.stack([(D[l - 1] != D2[l - 1]).to(kf.DT).mean(-1) for l in range(c, L + 1)], -1))
    rows[t] = {}
    for c in MAIN:
        a = {key: torch.cat(v).cpu().numpy() for key, v in acc[c].items()}
        rows[t][c] = {key: (a[key].mean(0).tolist() if key == 'dis' else float(np.mean(a[key]))) for key in a}
        rows[t][c].update({'theta_median': float(np.median(a['theta'])), 'CS_median': float(np.median(a['CS'])),
                           'CSp_median': float(np.median(a['CSp'])), 'O_median': float(np.median(a['O']))})
        np.savez_compressed(os.path.join(A.runs, f'C_{tag}_c{c}_t{t:.4g}.npz'), **a)
    print(f"[{tag}] t={t:.4g} " + " | ".join(
        f"c={c}: theta {rows[t][c]['theta_median']:.4f} O {rows[t][c]['O']:.3f} CS {rows[t][c]['CS']:.3f} "
        f"CSp {rows[t][c]['CSp']:.3f} rho {rows[t][c]['rho']:.3f} pred {rows[t][c]['rho_pred']:.3f} "
        f"dis(c+1) {rows[t][c]['dis'][1]:.4f}" for c in MAIN) + f"  {time.time()-t0:.0f}s", flush=True)
json.dump({'n': n, 'seed': seed, 'L': L, 'k': {c: K[c] for c in MAIN}, 'rows': {str(t): v for t, v in rows.items()}},
          open(os.path.join(A.runs, f'C_{tag}.json'), 'w'), indent=1)
print(f"[{tag}] done {time.time() - t0:.0f}s", flush=True)
