"""Test lib/cover.py against explicit per-pair QR. Run: nice -n 19 ionice -c3 python tests/test_cover.py (exit 0 = pass)."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import kf, cover, nd

torch.set_num_threads(2)
net = nd.Net('plain', 9, 32, 12, 1e-3)
g = kf.gen(3)
zs, _ = kf.forward(net, kf.draw_x(80, 32, g), g)
D = kf.masks(zs)
c, k, KK = 6, 5, 12
J = kf.downstream_jacobians(net, D, (c,))[c]
s, V = kf.svd_right(J)
Vx, sx, Dx = V[:40, :, :KK].float(), s[:40, :KK].float(), D[c - 1][:40]
Fy = V[40:, :, :k]
CS, O = cover.pair_scores(Vx, sx, Dx, Fy, k, bx=7, by=9)
err_cs, err_o = 0.0, 0.0
for i in range(40):
    for j in range(40):
        G = kf.orth((Dx[i][:, None] * Fy[j])[None])
        o, cs, _ = kf.scores(G, V[i:i + 1, :, :KK], s[i:i + 1, :KK], k)
        err_cs = max(err_cs, abs(float(cs) - float(CS[i, j])))
        err_o = max(err_o, abs(float(o) - float(O[i, j])))
ok = err_cs < 1e-4 and err_o < 1e-4
print(('PASS' if ok else 'FAIL') + f' pair_scores vs explicit QR: max |dCS| {err_cs:.2e}, max |dO| {err_o:.2e}')
cov = torch.rand(50, 30, generator=g) < 0.1
ch, curve = cover.greedy_cover(cov)
ok2 = all(b >= a for a, b in zip(curve, curve[1:])) and abs(curve[-1] - float(cov.any(1).float().mean())) < 1e-9
print(('PASS' if ok2 else 'FAIL') + f' greedy cover monotone and reaches the union ({len(ch)} centres, {curve[-1]:.3f})')
R = cover.r99(np.full(100, 0.01))
ok3 = abs(R - np.log(100) / 0.01) < 1e-6 * R and cover.r99(np.r_[np.zeros(2), np.ones(98)]) == float('inf')
print(('PASS' if ok3 else 'FAIL') + f' r99 closed form ({R:.2f} vs {np.log(100)/0.01:.2f})')
# 4. the reviewer's ill-conditioned case (GPT-6 review 2026-10-31): float32 Gram gave CS = 2.155; exact is 0.267
ok4 = True
f = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'B_n32_s401_L16_c8.npz')
if os.path.exists(f):
    d = np.load(f)
    kk = int(d['k'])
    i, j = 1106, 11345
    Vx = torch.from_numpy(d['VB'][i:i + 1]).double(); sx = torch.from_numpy(d['sB'][i:i + 1]).double()
    Dx = (torch.from_numpy(d['zB'][i:i + 1]) > 0).double(); Fy = torch.from_numpy(d['VA'][j:j + 1, :, :kk]).double()
    cs64, _ = cover.pair_scores(Vx, sx, Dx, Fy, kk)
    G = Dx[0][:, None] * Fy[0]
    U, S, _ = torch.linalg.svd(G, full_matrices=False)
    Ur = U[:, S > 1e-12 * S[0]]
    capt = ((sx[0] ** 2) * ((Vx[0].T @ Ur) ** 2).sum(-1)).sum() / (sx[0, :kk] ** 2).sum()
    cs32, _ = cover.pair_scores(Vx, sx, Dx, Fy, kk, dtype=torch.float32)
    ok4 = abs(float(cs64) - float(capt)) < 1e-6
    print(('PASS' if ok4 else 'FAIL') + f' reviewer case: float64 pair score {float(cs64):.6f}, SVD projector {float(capt):.6f}'
          f' (float32 Gram gives {float(cs32):.4f})')
    # random pairs: no score above 1
    g2 = torch.Generator().manual_seed(5)
    ii = torch.randint(0, d['VB'].shape[0], (64,), generator=g2); jj = torch.randint(0, d['VA'].shape[0], (256,), generator=g2)
    CSr, _ = cover.pair_scores(torch.from_numpy(d['VB'][ii.numpy()]).double(), torch.from_numpy(d['sB'][ii.numpy()]).double(),
                               (torch.from_numpy(d['zB'][ii.numpy()]) > 0).double(),
                               torch.from_numpy(d['VA'][jj.numpy(), :, :kk]).double(), kk)
    ok5 = float(CSr.max()) <= 1 + 1e-6
    print(('PASS' if ok5 else 'FAIL') + f' no pair score above 1 on 64 x 256 random pairs (max {float(CSr.max()):.6f})')
    ok4 = ok4 and ok5
sys.exit(0 if ok and ok2 and ok3 and ok4 else 1)
