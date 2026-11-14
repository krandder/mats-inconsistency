"""N7: first-order tails of the PB_k (top-k PC) block at any width, from vector-Jacobian products.
tail_PC(k) = sum_{j>k} lam_j m_j / sum_j lam_j m_j, lam_j = eigenvalues of P Sigma P (mean direction removed),
m_j = v_j^T M v_j = E (g . v_j)^2 with g = J^T r, r ~ N(0, I) (unbiased for M = E J^T J). Also the state tail
sum_{j>k} lam_j / sum lam_j. Cheap at any width (one backward pass per path and probe).
python n7_pctail.py --seed 113 --n 512 --L 128 --cut 64 [--paths 8192 --probes 2]"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import anet

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=113); ap.add_argument('--n', type=int, default=512)
ap.add_argument('--L', type=int, default=128); ap.add_argument('--cut', type=int, default=64)
ap.add_argument('--paths', type=int, default=8192); ap.add_argument('--probes', type=int, default=2)
ap.add_argument('--statepaths', type=int, default=1 << 16); ap.add_argument('--sigma', type=float, default=0.001)
ap.add_argument('--batch', type=int, default=512); ap.add_argument('--threads', type=int, default=2)
A = ap.parse_args()
torch.set_num_threads(A.threads)
DT = torch.float64
n, L, c = A.n, A.L, A.cut
Lp = L - c
rtab = json.load(open(os.path.join(HERE, '..', 'lib', 'rtable_ng.json')))
net = anet.Net(A.seed, n, L, A.sigma, r=rtab[f"{n}_{L}_{A.seed}_0.001"])
W = [w.to(DT) for w in net.W]; s = net.s.to(DT)
g = torch.Generator().manual_seed(5150 + 7 * A.seed + c)
t0 = time.time()


def to_cut(b):
    h = torch.randn(b, n, generator=g, dtype=DT)
    for l in range(c):
        z = h @ W[l].T
        if l < c - 1:
            h = torch.relu(z) + s[l] * torch.randn(b, n, generator=g, dtype=DT)
    return z


S1 = torch.zeros(n, dtype=DT); S2 = torch.zeros(n, n, dtype=DT); cnt = 0
while cnt < A.statepaths:
    b = min(4096, A.statepaths - cnt)
    z = to_cut(b); S1 += z.sum(0); S2 += z.T @ z; cnt += b
mu = S1 / cnt; Sig = S2 / cnt - torch.outer(mu, mu)
u = mu / mu.norm(); P = torch.eye(n, dtype=DT) - torch.outer(u, u)
lam, V = torch.linalg.eigh(P @ Sig @ P); lam = lam.flip(0).clamp_min(0); V = V.flip(1)
print(f"state: {cnt} paths; PR(P Sigma P) {float(lam.sum()**2/(lam**2).sum()):.2f}  {time.time()-t0:.0f}s", flush=True)
mj = torch.zeros(n, dtype=DT); trM = 0.0; npr = 0
done = 0
while done < A.paths:
    b = min(A.batch, A.paths - done)
    z = to_cut(b)
    masks = [(z > 0)]
    h = torch.relu(z) + s[c - 1] * torch.randn(b, n, generator=g, dtype=DT)
    for k in range(1, Lp + 1):
        zz = h @ W[c + k - 1].T
        masks.append(zz > 0)
        if k < Lp:
            h = torch.relu(zz) + s[c + k - 1] * torch.randn(b, n, generator=g, dtype=DT)
    for _ in range(A.probes):
        gv = torch.randn(b, n, generator=g, dtype=DT)
        for k in range(Lp, 0, -1):
            gv = (gv * masks[k].to(DT)) @ W[c + k - 1]
        gv = gv * masks[0].to(DT)                      # g = J^T r
        proj = gv @ V
        mj += (proj ** 2).sum(0); trM += float((gv ** 2).sum()); npr += b
    done += b
mj /= npr; trM /= npr
w = lam * mj
tailPC = {k: float(w[k:].sum() / w.sum()) for k in (0, 2, 4, 8, 16, 32, 64, 128, 256) if k < n}
tailSig = {k: float(lam[k:].sum() / lam.sum()) for k in tailPC}
res = dict(seed=A.seed, n=n, L=L, cut=c, Lp=Lp, paths=A.paths, probes=A.probes, tail_PC=tailPC, tail_state=tailSig,
           pr_state=float(lam.sum() ** 2 / (lam ** 2).sum()), trM_over_n=trM / n)
print(f"n {n} L {L} cut {c} net {A.seed}: tail_PC k=" + ",".join(map(str, tailPC)) + ": "
      + " ".join(f"{v:.2e}" for v in tailPC.values()), flush=True)
print(f"   state tail: " + " ".join(f"{v:.2e}" for v in tailSig.values()) + f"; trM/n {trM/n:.3f}  {time.time()-t0:.0f}s", flush=True)
os.makedirs('runs', exist_ok=True)
json.dump(res, open(f"runs/n7_{n}_{L}_c{c}_s{A.seed}.json", 'w'), indent=1)
