"""T1: numerical check of Proposition 2 (notes/props.md) on noiseless nets (sigma = 0): the overlap of the true
k-frame F_c with past-measurable frames S (the past product's top-k left singular frame; the same with e_c = a_c/|a_c|
swapped in for its last column; a random frame in range(D_c)) against the bound
E tr(P_F P_S) <= |P_S e_c|^2 + k (d - |P_S e_c|^2) / (m_c - 1).
nice -n 19 ionice -c3 python t1_prop2.py"""
import os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import kf, nd

torch.set_num_threads(1)
for n, L, c, k in ((32, 32, 16, 6), (64, 32, 16, 12), (64, 32, 8, 9)):
    for seed in (400, 401):
        net = nd.Net('plain', seed, n, L, 0.0)
        g = kf.gen(90_000 + seed + n + c)
        tot = {key: [] for key in ('blv', 'blv_e', 'rand')}
        bnd = {key: [] for key in tot}
        for _ in range(4):
            x = kf.draw_x(128, n, g)
            zs, _ = kf.forward(net, x, g)
            D = kf.masks(zs)
            alive = D[L - 1].sum(-1) > 0
            J = kf.downstream_jacobians(net, D, (c,))[c]
            s, V = kf.svd_right(J)
            F = V[:, :, :k]
            P = kf.past_products(net, D, (c,))[c]
            U, _, _ = torch.linalg.svd(D[c - 1][:, :, None] * P)
            a = torch.relu(zs[c - 1]); e = a / a.norm(dim=-1, keepdim=True)
            S = {'blv': kf.orth(U[:, :, :k]), 'blv_e': kf.orth(torch.cat([U[:, :, :k - 1], e[:, :, None]], -1)),
                 'rand': kf.random_frame(D[c - 1], k, g)}
            m = D[c - 1].sum(-1)
            for key, G in S.items():
                ov = (F.transpose(-1, -2) @ G).pow(2).sum((-1, -2))
                se = (G.transpose(-1, -2) @ e[:, :, None]).pow(2).sum((-1, -2))
                tot[key].append(ov[alive]); bnd[key].append((se + k * (k - se) / (m - 1))[alive])
        print(f"n={n} L={L} c={c} k={k} seed={seed}: " + "  ".join(
            f"{key}: E tr(P_F P_S) = {torch.cat(tot[key]).mean():.3f} (bound {torch.cat(bnd[key]).mean():.3f})"
            for key in tot), flush=True)
