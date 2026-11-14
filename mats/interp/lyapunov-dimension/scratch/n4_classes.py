"""N4 (coordinator's question, 19:20Z): does an input-dependent block (one block per class of inputs) recover the
per-path Lyapunov structure that a fixed block loses? How many classes are needed?
Setup: ARC net (n, L), cut c, horizon L'. Classes are defined from a pilot of forward-only paths, then every path's
J = d a_{c+L'} / d z_c is accumulated per class and per half (A / B) as sum J^T J; z_c is accumulated likewise.
Cross-fitted tails (no small-sample bias toward single-path structure): the block U_c (top-k eigenvectors of M_c, or
the whitened optimum for S_c) is chosen on half A and evaluated on half B, and vice versa; averaged.
Reported per classing and K: sum_c pi_c tr(P_{U_c-perp} M_c) / sum_c pi_c tr M_c  (sensitivity only), and
sum_c pi_c tr(M_c Schur_c(U_c)) / tr(M Sigma_P) (state-weighted, normalised by the unconditioned k = 0 error), at k.
Classings: KM8-K (k-means on the top-8 PCs of P z_c), QC3 (4x4x4 quantile cells on the top-3 PCs), GATE-h (sign
pattern of the h most uncertain gates of layer c+1), MASK8-K (k-means on the masks of layers c+1..c+8).
python n4_classes.py --seed 113 --n 256 --L 64 --cut 32 --Lp 32 --paths 8192"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import anet

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=113); ap.add_argument('--n', type=int, default=256)
ap.add_argument('--L', type=int, default=64); ap.add_argument('--cut', type=int, default=32)
ap.add_argument('--Lp', type=int, default=32); ap.add_argument('--paths', type=int, default=8192)
ap.add_argument('--pilot', type=int, default=1 << 14); ap.add_argument('--sigma', type=float, default=0.001)
ap.add_argument('--batch', type=int, default=64); ap.add_argument('--threads', type=int, default=2)
ap.add_argument('--out', default='runs'); ap.add_argument('--statepaths', type=int, default=1 << 17)
A = ap.parse_args()
torch.set_num_threads(A.threads)
DT = torch.float64
n, L, c, Lp = A.n, A.L, A.cut, A.Lp
rtab = json.load(open(os.path.join(HERE, '..', 'lib', 'rtable_ng.json')))
net = anet.Net(A.seed, n, L, A.sigma, r=rtab[f"{n}_{L}_{A.seed}_0.001"])
W = [w.to(DT) for w in net.W]; s = net.s.to(DT)
g = torch.Generator().manual_seed(777 + 13 * A.seed + c)
t0 = time.time()


def forward_to_cut(b, extra_masks=0):
    h = torch.randn(b, n, generator=g, dtype=DT)
    for l in range(c):
        z = h @ W[l].T
        if l < c - 1:
            h = torch.relu(z) + s[l] * torch.randn(b, n, generator=g, dtype=DT)
    return z


# ---------------- pilot: PCs of P z_c, quantiles, uncertain gates of layer c+1, mask k-means
Zp = torch.cat([forward_to_cut(4096) for _ in range(A.pilot // 4096)])
mu = Zp.mean(0); u = mu / mu.norm(); P = torch.eye(n, dtype=DT) - torch.outer(u, u)
C = torch.cov((Zp @ P).T)
ev, V = torch.linalg.eigh(C); V = V.flip(1)
sc = (Zp - mu) @ P @ V[:, :8]
qs = [torch.quantile(sc[:, j], torch.tensor([0.25, 0.5, 0.75], dtype=DT)) for j in range(3)]
# next-layer gates (one noise draw) for the pilot: uncertainty per neuron
Hp = torch.relu(Zp) + s[c - 1] * torch.randn(Zp.shape, generator=g, dtype=DT)
G1 = (Hp @ W[c].T > 0).to(DT)
p1 = G1.mean(0)
unc = (p1 * (1 - p1))
gate_idx = torch.argsort(unc, descending=True)


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
# masks of layers c+1..c+8 for a pilot subset (k-means on binary mask vectors)
sub = 4096
h = Hp[:sub]; Ms_ = []
for k in range(1, 9):
    z = h @ W[c + k - 1].T
    D = (z > 0).to(DT); Ms_.append(D)
    h = torch.relu(z) + s[c + k - 1] * torch.randn(z.shape, generator=g, dtype=DT)
Mask8 = torch.cat(Ms_, 1)
mk = {K: kmeans(Mask8, K, seed=100 + K) for K in (16, 64)}
print(f"pilot done {time.time()-t0:.0f}s; top-8 PC var share {float(ev.flip(0)[:8].sum()/ev.sum()):.3f}; "
      f"uncertain gates p(1-p) top: {[round(float(v),3) for v in unc[gate_idx[:8]]]}", flush=True)


def classes_of(z, masks8):
    """returns dict classing -> class index tensor (b,)"""
    scz = (z - mu) @ P @ V[:, :8]
    out = {}
    for K, cen in km8.items():
        out[f'KM8-{K}'] = torch.cdist(scz, cen).argmin(1)
    q = torch.zeros(z.shape[0], dtype=torch.long)
    for j in range(3):
        q = q * 4 + (scz[:, j:j + 1] > qs[j][None, :]).sum(1)
    out['QC3-64'] = q
    for hh in (4, 6):
        idx = gate_idx[:hh]
        # gate of layer c+1 on this path's own noise draw: from masks8[:, :n]
        bits = masks8[:, :n][:, idx].long()
        out[f'GATE-{2**hh}'] = (bits * (2 ** torch.arange(hh))).sum(1)
    for K, cen in mk.items():
        out[f'MASK8-{K}'] = torch.cdist(masks8, cen).argmin(1)
    out['ALL-1'] = torch.zeros(z.shape[0], dtype=torch.long)
    return out


Kof = {'KM8-4': 4, 'KM8-16': 16, 'KM8-64': 64, 'QC3-64': 64, 'GATE-16': 16, 'GATE-64': 64, 'MASK8-16': 16,
       'MASK8-64': 64, 'ALL-1': 1}
acc = {nm: dict(JJ=torch.zeros(2, K, n, n, dtype=DT), cnt=torch.zeros(2, K, dtype=DT)) for nm, K in Kof.items()}
single_tails = []
done = 0
while done < A.paths:
    b = min(A.batch, A.paths - done)
    z = forward_to_cut(b)
    half = 0 if done < A.paths // 2 else 1
    Dc = (z > 0).to(DT); a = torch.relu(z)
    J = torch.diag_embed(Dc)
    h = a + s[c - 1] * torch.randn(b, n, generator=g, dtype=DT)
    masks = []
    for k in range(1, Lp + 1):
        lw = c + k - 1
        zz = h @ W[lw].T
        D = (zz > 0).to(DT)
        if k <= 8:
            masks.append(D)
        J = D[:, :, None] * (W[lw] @ J)
        if k < Lp:
            h = torch.relu(zz) + s[lw] * torch.randn(b, n, generator=g, dtype=DT)
    JtJ = J.transpose(1, 2) @ J
    if done < 256:
        for i in range(b):
            e = torch.linalg.eigvalsh(JtJ[i]).flip(0).clamp_min(0)
            single_tails.append((e.cumsum(0) / e.sum()).numpy())
    cls = classes_of(z, torch.cat(masks, 1))
    zP = z  # raw z (mean direction handled later by P)
    for nm, ci in cls.items():
        R = acc[nm]
        R['JJ'][half].index_add_(0, ci, JtJ)
        R['cnt'][half].index_add_(0, ci, torch.ones(b, dtype=DT))
    done += b
    if done % 1024 == 0:
        print(f"  paths {done}  {time.time()-t0:.0f}s", flush=True)

# ---- state moments per class from a large forward-only sample (the J paths are too few for Sigma_c)
for nm in acc:
    acc[nm]['Z1s'] = torch.zeros(2, Kof[nm], n, dtype=DT); acc[nm]['Z2s'] = torch.zeros(2, Kof[nm], n, n, dtype=DT)
    acc[nm]['cnts'] = torch.zeros(2, Kof[nm], dtype=DT)
dn = 0
while dn < A.statepaths:
    b = min(2048, A.statepaths - dn)
    z = forward_to_cut(b)
    h = torch.relu(z) + s[c - 1] * torch.randn(b, n, generator=g, dtype=DT)
    masks = []
    for k in range(1, 9):
        zz = h @ W[c + k - 1].T
        masks.append((zz > 0).to(DT))
        h = torch.relu(zz) + s[c + k - 1] * torch.randn(b, n, generator=g, dtype=DT)
    cls = classes_of(z, torch.cat(masks, 1))
    half = 0 if dn < A.statepaths // 2 else 1
    for nm, ci in cls.items():
        R = acc[nm]
        R['Z1s'][half].index_add_(0, ci, z)
        for kk in torch.unique(ci).tolist():
            zk = z[ci == kk]
            R['Z2s'][half, kk] += zk.T @ zk
        R['cnts'][half].index_add_(0, ci, torch.ones(b, dtype=DT))
    dn += b
print(f"state moments from {A.statepaths} forward-only paths  {time.time()-t0:.0f}s", flush=True)
KS = [k for k in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128] if k < n]
P_np = P.numpy()


def eigh_desc(X):
    e, Vv = np.linalg.eigh(0.5 * (X + X.T))
    return np.clip(e[::-1], 0, None), Vv[:, ::-1]


def sqrtm(X):
    e, Vv = np.linalg.eigh(0.5 * (X + X.T)); e = np.clip(e, 0, None)
    return (Vv * np.sqrt(e)) @ Vv.T


# global (ALL-1) reference for normalisation
R = acc['ALL-1']
cntT = float(R['cnt'].sum())
M_all = (R['JJ'][0, 0] + R['JJ'][1, 0]).numpy() / cntT
cntS = float(R['cnts'].sum())
Z1 = (R['Z1s'][0, 0] + R['Z1s'][1, 0]).numpy() / cntS
Sig_all = (R['Z2s'][0, 0] + R['Z2s'][1, 0]).numpy() / cntS - np.outer(Z1, Z1)
Sig_all = P_np @ Sig_all @ P_np
ref_S = float(np.trace(M_all @ Sig_all))
ref_M = float(np.trace(M_all))
res = dict(seed=A.seed, n=n, L=L, cut=c, Lp=Lp, paths=A.paths, classings={})
st = np.array(single_tails)
res['single_path_tail'] = {k: float(1 - st[:, k - 1].mean()) for k in KS}
print("single-path tail (energy beyond k), mean of 256 paths: " + " ".join(f"k{k}:{res['single_path_tail'][k]:.2e}" for k in KS), flush=True)
for nm, K in Kof.items():
    R = acc[nm]
    tM = {k: 0.0 for k in KS}; tS = {k: 0.0 for k in KS}; totM = 0.0; between = 0.0
    used = 0; wts = 0.0
    for ci in range(K):
        nA, nB = float(R['cnt'][0, ci]), float(R['cnt'][1, ci])
        if nA < 8 or nB < 8:
            continue
        used += 1
        Ms = [R['JJ'][hh, ci].numpy() / max(float(R['cnt'][hh, ci]), 1) for hh in (0, 1)]
        Ss = []
        for hh in (0, 1):
            m1 = R['Z1s'][hh, ci].numpy() / max(float(R['cnts'][hh, ci]), 1.0)
            S2 = R['Z2s'][hh, ci].numpy() / max(float(R['cnts'][hh, ci]), 1.0) - np.outer(m1, m1)
            Ss.append(P_np @ S2 @ P_np)
        pi = float(R['cnts'][:, ci].sum()) / float(R['cnts'].sum()); wts += pi
        for (fa, fb) in ((0, 1), (1, 0)):
            # sensitivity-only block: top-k eigvecs of M_c on half fa, evaluated on half fb
            eA, VA = eigh_desc(Ms[fa])
            trB = float(np.trace(Ms[fb]))
            for k in KS:
                Uk = VA[:, :k]
                cap = float(np.trace(Uk.T @ Ms[fb] @ Uk))
                tM[k] += 0.5 * pi * (trB - cap)
            # state-weighted optimum: whitened block from half fa (S_c = R M R, R = Sigma_c^1/2)
            Rh = sqrtm(Ss[fa])
            SA = Rh @ Ms[fa] @ Rh
            eS, VS = eigh_desc(SA)
            Rp = np.linalg.pinv(Rh, rcond=1e-8)
            SigB, MB = Ss[fb], Ms[fb]
            for k in KS:
                Bk = Rp @ VS[:, :k]          # block functionals Sigma^-1/2 v_j
                Q, _ = np.linalg.qr(Bk)
                G = Q.T @ SigB @ Q
                Sch = SigB - SigB @ Q @ np.linalg.pinv(G, rcond=1e-10) @ Q.T @ SigB
                tS[k] += 0.5 * pi * float(np.trace(MB @ Sch))
            totM += 0.5 * pi * trB
    res['classings'][nm] = dict(K=K, used=used, weight=wts,
                                tail_M={k: tM[k] / totM for k in KS},
                                tail_S_rel_global={k: tS[k] / ref_S for k in KS})
    print(f"{nm:9s} classes used {used:3d} (weight {wts:.3f}): sensitivity tail k=" + ",".join(map(str, KS)) + ": "
          + " ".join(f"{tM[k]/totM:.2e}" for k in KS), flush=True)
    print(f"{'':9s} state-weighted tail / unconditioned k=0 error: "
          + " ".join(f"{tS[k]/ref_S:.2e}" for k in KS), flush=True)
os.makedirs(A.out, exist_ok=True)
json.dump(res, open(f"{A.out}/n4_{n}_{L}_c{c}_Lp{Lp}_s{A.seed}.json", 'w'), indent=1)
print('done', f'{time.time()-t0:.0f}s', flush=True)
