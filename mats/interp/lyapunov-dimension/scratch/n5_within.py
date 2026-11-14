"""N5: within-class coherence for the classings of n4 (same seeds, so the same pilot, PCs and centres).
For fresh forward paths: per classing, the within-class mean squared angle between two members' states h_c, and the
within-class pairwise gate disagreement at layers c+1..c+8 (fraction of neurons whose gates differ), against the
global values. Two-replica reading: the coherent part decays by r^2 per layer, r = 1 - disagreement; to keep a class
coherent over the horizon L' the disagreement must be below about 1/(2 L').
python n5_within.py --seed 113 --n 256 --L 64 --cut 32"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import anet

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=113); ap.add_argument('--n', type=int, default=256)
ap.add_argument('--L', type=int, default=64); ap.add_argument('--cut', type=int, default=32)
ap.add_argument('--pilot', type=int, default=1 << 14); ap.add_argument('--paths', type=int, default=1 << 14)
ap.add_argument('--sigma', type=float, default=0.001); ap.add_argument('--threads', type=int, default=2)
A = ap.parse_args()
torch.set_num_threads(A.threads)
DT = torch.float64
n, L, c = A.n, A.L, A.cut
rtab = json.load(open(os.path.join(HERE, '..', 'lib', 'rtable_ng.json')))
net = anet.Net(A.seed, n, L, A.sigma, r=rtab[f"{n}_{L}_{A.seed}_0.001"])
W = [w.to(DT) for w in net.W]; s = net.s.to(DT)
g = torch.Generator().manual_seed(777 + 13 * A.seed + c)     # same as n4: identical pilot and centres


def forward_to_cut(b):
    h = torch.randn(b, n, generator=g, dtype=DT)
    for l in range(c):
        z = h @ W[l].T
        if l < c - 1:
            h = torch.relu(z) + s[l] * torch.randn(b, n, generator=g, dtype=DT)
    return z


Zp = torch.cat([forward_to_cut(4096) for _ in range(A.pilot // 4096)])
mu = Zp.mean(0); u = mu / mu.norm(); P = torch.eye(n, dtype=DT) - torch.outer(u, u)
C = torch.cov((Zp @ P).T)
ev, V = torch.linalg.eigh(C); V = V.flip(1)
sc = (Zp - mu) @ P @ V[:, :8]
qs = [torch.quantile(sc[:, j], torch.tensor([0.25, 0.5, 0.75], dtype=DT)) for j in range(3)]
Hp = torch.relu(Zp) + s[c - 1] * torch.randn(Zp.shape, generator=g, dtype=DT)
G1 = (Hp @ W[c].T > 0).to(DT)
p1 = G1.mean(0); unc = (p1 * (1 - p1)); gate_idx = torch.argsort(unc, descending=True)


def kmeans(X, K, iters=40, seed=0):
    gg = torch.Generator().manual_seed(seed)
    cen = X[torch.randperm(X.shape[0], generator=gg)[:K]].clone()
    for _ in range(iters):
        d = torch.cdist(X, cen); a = d.argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                cen[k] = X[m].mean(0)
    return cen


km8 = {K: kmeans(sc, K, seed=K) for K in (4, 16, 64)}
h = Hp[:4096]; Ms_ = []
for k in range(1, 9):
    z = h @ W[c + k - 1].T
    D = (z > 0).to(DT); Ms_.append(D)
    h = torch.relu(z) + s[c + k - 1] * torch.randn(z.shape, generator=g, dtype=DT)
mk = {K: kmeans(torch.cat(Ms_, 1), K, seed=100 + K) for K in (16, 64)}


def classes_of(z, masks8):
    scz = (z - mu) @ P @ V[:, :8]
    out = {}
    for K, cen in km8.items():
        out[f'KM8-{K}'] = torch.cdist(scz, cen).argmin(1)
    q = torch.zeros(z.shape[0], dtype=torch.long)
    for j in range(3):
        q = q * 4 + (scz[:, j:j + 1] > qs[j][None, :]).sum(1)
    out['QC3-64'] = q
    for hh in (4, 6):
        bits = masks8[:, :n][:, gate_idx[:hh]].long()
        out[f'GATE-{2**hh}'] = (bits * (2 ** torch.arange(hh))).sum(1)
    for K, cen in mk.items():
        out[f'MASK8-{K}'] = torch.cdist(masks8, cen).argmin(1)
    out['ALL-1'] = torch.zeros(z.shape[0], dtype=torch.long)
    return out


# fresh paths: directions of h_c and masks of layers c+1..c+8
zs, hs, ms = [], [], []
done = 0
while done < A.paths:
    b = min(2048, A.paths - done)
    z = forward_to_cut(b)
    h = torch.relu(z) + s[c - 1] * torch.randn(b, n, generator=g, dtype=DT)
    hs.append(h / h.norm(dim=1, keepdim=True)); zs.append(z)
    mm = []
    for k in range(1, 9):
        zz = h @ W[c + k - 1].T
        mm.append((zz > 0).to(torch.float32))
        h = torch.relu(zz) + s[c + k - 1] * torch.randn(b, n, generator=g, dtype=DT)
    ms.append(torch.stack(mm, 1))
    done += b
Z = torch.cat(zs); H = torch.cat(hs); Mk = torch.cat(ms)            # Mk: (N, 8, n)
cls = classes_of(Z, Mk.reshape(Mk.shape[0], -1).to(DT))
res = {}
for nm, ci in cls.items():
    ang2, dis = [], []
    wts = []
    for k in torch.unique(ci).tolist():
        idx = (ci == k).nonzero().flatten()
        if idx.numel() < 16:
            continue
        Hk = H[idx]; mbar = Hk.mean(0)
        m = idx.numel()
        # mean squared chord between two distinct members: 2 - 2 (m^2 |mbar|^2 - m) / (m (m - 1))
        ang2.append(2 - 2 * (m * m * float(mbar @ mbar) - m) / (m * (m - 1)))
        Mm = Mk[idx].double()                                     # (m, 8, n)
        pbar = Mm.mean(0)                                         # (8, n)
        # pairwise disagreement fraction per layer: mean over pairs of mean_i [d_i != d'_i]
        #   = mean_i 2 p_i (1 - p_i) * m / (m - 1)
        dis.append((2 * pbar * (1 - pbar)).mean(1) * m / (m - 1))
        wts.append(m)
    wts = np.array(wts, dtype=float); wts /= wts.sum()
    a2 = float(np.sum(wts * np.array(ang2)))
    dl = (torch.stack(dis) * torch.tensor(wts)[:, None]).sum(0)
    res[nm] = dict(angle_rms=math.sqrt(max(a2, 0)), disagreement=[float(v) for v in dl])
    r = 1 - dl.numpy()
    # coherent length over 32 layers if the per-layer disagreement stayed at its mean over c+1..c+8
    rbar = float(r.mean())
    coh = sum(rbar ** (2 * l) for l in range(32))
    print(f"{nm:9s}: within-class rms angle {math.sqrt(max(a2,0)):.3f} rad; gate disagreement per layer c+1..c+8: "
          + " ".join(f"{v:.3f}" for v in dl.tolist()) + f"; coherent layers (of 32) {coh:.1f}", flush=True)
json.dump(res, open(f"runs/n5_{n}_{L}_c{c}_s{A.seed}.json", 'w'), indent=1)
