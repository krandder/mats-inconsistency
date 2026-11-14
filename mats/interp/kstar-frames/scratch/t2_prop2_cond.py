"""T2 (after the GPT-6 review): a conditional check of Proposition 2 on noiseless nets. Fix the past (input x and
W_1..W_c), resample the future weights W_{c+1..L} many times, and average tr(P_F P_S) over futures for past-measurable
S (the past product's top-k left frame; the same with e_c swapped in), on futures with a spectral gap at the cutoff
(s_k > 1.0001 s_{k+1}) and a live output; compare with the bound |P_S e_c|^2 + k (k - |P_S e_c|^2)/(m_c - 1).
nice -n 19 ionice -c3 python t2_prop2_cond.py"""
import math, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import kf

torch.set_num_threads(1)
DT = torch.float64
for n, L, c, k in ((32, 32, 16, 6), (64, 32, 16, 12)):
    g = torch.Generator().manual_seed(7000 + n)
    rows = []
    for past in range(6):
        Wp = [torch.randn(n, n, generator=g, dtype=DT) * math.sqrt(2.0 / n) for _ in range(c)]
        x = torch.randn(n, generator=g, dtype=DT)
        h, P = x, torch.eye(n, dtype=DT)
        for l, W in enumerate(Wp):
            z = W @ h
            P = W @ (P if l == 0 else (D[:, None] * P))
            D = (z > 0).to(DT)
            h = torch.relu(z)
        Dc, a = D, h
        e = a / a.norm()
        m = int(Dc.sum())
        U, _, _ = torch.linalg.svd(Dc[:, None] * P)
        S1 = kf.orth(U[None, :, :k])[0]
        S2 = kf.orth(torch.cat([U[:, :k - 1], e[:, None]], -1)[None])[0]
        acc = {1: [], 2: []}
        for fut in range(400):
            J = torch.diag(Dc)
            hh = a
            Ds = []
            Wf = [torch.randn(n, n, generator=g, dtype=DT) * math.sqrt(2.0 / n) for _ in range(L - c)]
            for W in Wf:
                zz = W @ hh
                Ds.append((zz > 0).to(DT))
                hh = torch.relu(zz)
            if hh.norm() == 0:
                continue
            J = torch.diag(Ds[-1])
            for W, Dl in zip(reversed(Wf[1:]), reversed(Ds[:-1])):
                J = (J @ W) * Dl[None, :]
            J = (J @ Wf[0]) * Dc[None, :]
            _, s, Vh = torch.linalg.svd(J)
            if not s[k - 1] > 1.0001 * s[k]:
                continue
            F = Vh[:k].T
            acc[1].append(float(((F.T @ S1) ** 2).sum()))
            acc[2].append(float(((F.T @ S2) ** 2).sum()))
        for key, S in ((1, S1), (2, S2)):
            se = float(((S.T @ e) ** 2).sum())
            bound = se + k * (k - se) / (m - 1)
            rows.append((past, key, len(acc[key]), np.mean(acc[key]), np.std(acc[key]) / math.sqrt(len(acc[key])), bound))
    print(f"n={n} L={L} c={c} k={k}: (past, frame [1 = past top-k, 2 = with e_c], futures kept, E tr(P_F P_S) +- se, bound)")
    for r in rows:
        print(f"   past {r[0]} S{r[1]}: {r[2]} futures, {r[3]:.3f} +- {r[4]:.3f}  bound {r[5]:.3f}  {'OK' if r[3] <= r[5] + 2 * r[4] else 'ABOVE'}")
    sys.stdout.flush()
