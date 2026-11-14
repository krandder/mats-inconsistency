"""N6 (gate flips): how far is one path's response to a finite perturbation from linear response?
For paths at the cut, perturb z_c by Delta = eps * |z_c - mean| * v, v a unit vector drawn from the state's residual
directions (PCs beyond k0 of P Sigma P, weighted by their standard deviations, i.e. the direction a PB_k0 scramble
moves along). Same downstream noise for both runs. Reported per eps: the fraction of downstream gates flipped per layer,
the pathwise nonlinearity |F(z+D) - F(z) - J D| / |J D|, and the mean-square gain E|F(z+D) - F(z)|^2 / E|J D|^2.
python n6_flips.py --seed 113 --cut 32 --Lp 32 --k0 16"""
import argparse, json, math, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import anet

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=113); ap.add_argument('--n', type=int, default=256)
ap.add_argument('--L', type=int, default=64); ap.add_argument('--cut', type=int, default=32)
ap.add_argument('--Lp', type=int, default=32); ap.add_argument('--k0', type=int, default=16)
ap.add_argument('--paths', type=int, default=256); ap.add_argument('--sigma', type=float, default=0.001)
ap.add_argument('--threads', type=int, default=2)
A = ap.parse_args()
torch.set_num_threads(A.threads)
DT = torch.float64
n, L, c, Lp = A.n, A.L, A.cut, A.Lp
rtab = json.load(open(os.path.join(HERE, '..', 'lib', 'rtable_ng.json')))
net = anet.Net(A.seed, n, L, A.sigma, r=rtab[f"{n}_{L}_{A.seed}_0.001"])
W = [w.to(DT) for w in net.W]; s = net.s.to(DT)
g = torch.Generator().manual_seed(9090 + A.seed)


def to_cut(b):
    h = torch.randn(b, n, generator=g, dtype=DT)
    for l in range(c):
        z = h @ W[l].T
        if l < c - 1:
            h = torch.relu(z) + s[l] * torch.randn(b, n, generator=g, dtype=DT)
    return z


Zs = torch.cat([to_cut(4096) for _ in range(4)])
mu = Zs.mean(0); u = mu / mu.norm(); P = torch.eye(n, dtype=DT) - torch.outer(u, u)
ev, V = torch.linalg.eigh(torch.cov((Zs @ P).T)); ev = ev.flip(0).clamp_min(0); V = V.flip(1)
z0 = to_cut(A.paths)
noise = [torch.randn(A.paths, n, generator=g, dtype=DT) for _ in range(Lp + 1)]


def down(z, want_J=False):
    """a_{c+Lp} from z_c with fixed noise; optionally J and the masks."""
    a = torch.relu(z); masks = [(z > 0)]
    J = torch.diag_embed((z > 0).to(DT)) if want_J else None
    h = a + s[c - 1] * noise[0]
    for k in range(1, Lp + 1):
        zz = h @ W[c + k - 1].T
        D = zz > 0; masks.append(D)
        if want_J:
            J = D.to(DT)[:, :, None] * (W[c + k - 1] @ J)
        a = torch.relu(zz)
        if k < Lp:
            h = a + s[c + k - 1] * noise[k]
    return a, J, masks


F0, J, m0 = down(z0, True)
scale = (z0 - mu).norm(dim=1, keepdim=True)
coef = torch.randn(A.paths, n - A.k0, generator=g, dtype=DT) * ev[A.k0:].sqrt()[None, :]
v = coef @ V[:, A.k0:].T
v = v / v.norm(dim=1, keepdim=True)
res = []
for eps in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1):
    D = eps * scale * v
    F1, _, m1 = down(z0 + D)
    lin = torch.einsum('bij,bj->bi', J, D)
    dF = F1 - F0
    flips = [float((a != b).double().mean()) for a, b in zip(m0, m1)]
    nonlin = float(((dF - lin).norm(dim=1) / lin.norm(dim=1)).median())
    gain = float((dF.norm(dim=1) ** 2).mean() / (lin.norm(dim=1) ** 2).mean())
    mean_vs_rms = float(dF.mean(0).norm() / dF.norm(dim=1).pow(2).mean().sqrt())
    res.append(dict(eps=eps, flips_first=flips[1], flips_mean=float(np.mean(flips[1:])), nonlin_median=nonlin,
                    ms_gain=gain, mean_over_rms=mean_vs_rms))
    print(f"eps {eps:.0e}: gates flipped per layer {np.mean(flips[1:]):.4f} (first layer {flips[1]:.4f}); "
          f"median |dF - J D|/|J D| = {nonlin:.3f}; E|dF|^2 / E|J D|^2 = {gain:.3f}; |mean dF| / rms dF = {mean_vs_rms:.3f}",
          flush=True)
json.dump(res, open(f"runs/n6_{n}_{L}_c{c}_Lp{Lp}_s{A.seed}_k{A.k0}.json", 'w'), indent=1)
