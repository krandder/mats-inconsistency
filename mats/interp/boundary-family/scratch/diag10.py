"""Kelvin's item 1: why do oracle cells fitted at layer 24 work on net 101 (14x over one Gaussian) and not on net 102
(2.5x)? Per width-256 ARC dev net (ARC truths, 2^25-2^26 paths), sigma = 0.001:
- properties of z_24 from the fit set (2^NFIT paths): PR, top-mode share, |cos| of the top mode with the mean
  direction, size-mode share (variance along the mean direction / total), all-gate entropy, near-boundary gates
  (entropy > 0.1 bit) and their summed entropy, within-bin |skew| / excess kurtosis in the 64 PCA3 bins (all neurons
  and near-boundary neurons);
- representation error of E[a_l] at every layer l = 24..32 (sample the fitted family, exact layers with noise, against
  the truth) for G (one Gaussian), PCA3 (64 cells), PCA3NB (PCA3 with the exact within-cell marginals of the
  near-boundary coordinates by quantile mapping: Gaussian copula inside the cell), PCA4 (256 cells);
- attribution: share of the squared error at layers 25 and 32 carried by neurons near their boundary there
  (|t| < 1.5), against their share of neurons.
python scratch/diag10.py SEED NFIT NREP"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf

seed = int(sys.argv[1]); NFIT = 1 << int(sys.argv[2]); NREP = 1 << int(sys.argv[3])
L = 32; m = 24; n = 256
ARC = '/home/agent/workspaces/mats/work/arc-noisy'
net = bf.get_net(seed, n, 32, 0.001)
d = np.load(f'{ARC}/truth/truth_256_32_s0.001_seed{seed}.npz'); NT = float(d['N'])
TM = torch.tensor(d['S1'] / NT); TV = torch.tensor(d['S2'] / NT) - TM ** 2
W = [w.float() for w in net.W]; s = [float(v) for v in net.s]
t0 = time.time()
Z = bf.mc_stats(net, NFIT, seed=130_000 + seed, L=m, cuts=(m,), keep_n=NFIT, out_moments=False)['cuts'][m].double()
print(f'fit samples {time.time() - t0:.0f}s', flush=True)
res = dict(seed=seed, NFIT=NFIT, NREP=NREP)

# ---- properties at layer 24
mz = Z.mean(0); C = torch.cov(Z.T); ev, U = torch.linalg.eigh(C); ev = ev.flip(0); U = U.flip(1)
u = mz / mz.norm()
p = (Z > 0).double().mean(0)
ent = -(p * p.clamp_min(1e-12).log2() + (1 - p) * (1 - p).clamp_min(1e-12).log2())
near = torch.nonzero(ent > 0.1).flatten()
res['props'] = dict(pr=float(ev.sum() ** 2 / (ev ** 2).sum()), top_share=float(ev[0] / ev.sum()),
                    top3_share=float(ev[:3].sum() / ev.sum()), cos_top_mean=float((U[:, 0] @ u).abs()),
                    size_share=float(u @ C @ u / ev.sum()), gate_entropy_all=float(ent.sum()),
                    near_count=int(near.numel()), near_entropy=float(ent[near].sum()),
                    t_abs_median=float((mz / C.diagonal().sqrt()).abs().median()))


def labels(k, nb=4):
    Y = (Z - mz) @ U[:, :k]
    lab = torch.zeros(Z.shape[0], dtype=torch.long)
    for j in range(k):
        qs = torch.quantile(Y[:, j], torch.linspace(0, 1, nb + 1, dtype=torch.float64)[1:-1])
        lab = lab * nb + torch.bucketize(Y[:, j].contiguous(), qs)
    return lab


lab3 = labels(3)
sk_all = []; ku_all = []; wts = []
for c in torch.unique(lab3):
    Zc = Z[lab3 == c]; w = Zc.shape[0] / Z.shape[0]
    Zs = (Zc - Zc.mean(0)) / Zc.std(0).clamp_min(1e-12)
    sk_all.append((Zs ** 3).mean(0).abs()); ku_all.append((Zs ** 4).mean(0) - 3); wts.append(w)
wts = torch.tensor(wts, dtype=torch.float64)[:, None]
SK = (torch.stack(sk_all) * wts).sum(0); KU = (torch.stack(ku_all) * wts).sum(0)
res['props'].update(inbin_skew_all=float(SK.median()), inbin_kurt_all=float(KU.median()),
                    inbin_skew_near=float(SK[near].median()), inbin_kurt_near=float(KU[near].median()))
print('props', json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res['props'].items()}), flush=True)


@torch.no_grad()
def prop_all(z24, seedp=7, chunk=1 << 16):
    """per-layer E[a_l], l = 24..32, and mean / sd of z_25 and z_32, from samples of z_24."""
    g = torch.Generator().manual_seed(seedp)
    S1 = torch.zeros(L - m + 1, n, dtype=torch.float64)
    ZS = {25: [torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)],
          32: [torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)]}
    N = z24.shape[0]
    for a0 in range(0, N, chunk):
        z = z24[a0:a0 + chunk].float()
        a = torch.relu(z); S1[0] += a.double().sum(0)
        for l in range(m, L):
            h = a + s[l - 1] * torch.randn(a.shape, generator=g)
            z = h @ W[l].T
            if l + 1 in ZS:
                ZS[l + 1][0] += z.double().sum(0); ZS[l + 1][1] += (z.double() ** 2).sum(0)
            a = torch.relu(z); S1[l - m + 1] += a.double().sum(0)
    out = S1 / N
    tz = {}
    for l, (s1, s2) in ZS.items():
        mu = s1 / N; sd = (s2 / N - mu ** 2).clamp_min(1e-30).sqrt(); tz[l] = (mu / sd).abs()
    return out, tz


def errs(est):
    return [float(((est[i] - TM[m - 1 + i]) ** 2).mean() / TV[m - 1 + i].mean()) - 1 / NT for i in range(L - m + 1)]


def attribution(est, tz):
    out = {}
    for l in (25, 32):
        e2 = (est[l - m] - TM[l - 1]) ** 2
        nb = tz[l] < 1.5
        out[l] = dict(frac_neurons=float(nb.double().mean()), frac_err=float(e2[nb].sum() / e2.sum()))
    return out


def sample_cells(lab, nbq=False):
    g = torch.Generator().manual_seed(11)
    parts = []
    for c in torch.unique(lab):
        S = lab == c; Zc = Z[S]; k = int(round(NREP * float(S.double().mean())))
        if k == 0:
            continue
        mc = Zc.mean(0); Cc = torch.cov(Zc.T) + 1e-10 * torch.eye(n, dtype=torch.float64)
        e2, V2 = torch.linalg.eigh(Cc); Lh = V2 * e2.clamp_min(0).sqrt()
        X = mc + torch.randn(k, n, generator=g, dtype=torch.float64) @ Lh.T
        if nbq:                                          # exact within-cell marginals of the near-boundary coordinates
            sd = Cc.diagonal().sqrt()
            uq = torch.special.ndtr((X[:, near] - mc[near]) / sd[near])
            srt, _ = torch.sort(Zc[:, near], dim=0)
            pos = (uq * (srt.shape[0] - 1)).clamp(0, srt.shape[0] - 1)
            lo = pos.floor().long(); hi = pos.ceil().long(); fr = pos - lo
            X[:, near] = torch.gather(srt, 0, lo) * (1 - fr) + torch.gather(srt, 0, hi) * fr
        parts.append(X.float())
    return torch.cat(parts, 0)


Lc = torch.linalg.cholesky(C + 1e-10 * torch.eye(n, dtype=torch.float64))


def fam_G():
    gg = torch.Generator().manual_seed(11)
    return (mz + torch.randn(NREP, n, generator=gg, dtype=torch.float64) @ Lc.T).float()


def fam_EMP():                                           # resampled fit set: the fit-sampling floor
    gg = torch.Generator().manual_seed(12)
    return Z[torch.randint(0, Z.shape[0], (NREP,), generator=gg)].float()


fams = dict(G=fam_G, PCA3=lambda: sample_cells(lab3), PCA3NB=lambda: sample_cells(lab3, nbq=True),
            PCA4=lambda: sample_cells(labels(4)), EMP=fam_EMP)
res['fam'] = {}
for name, make in fams.items():
    t1 = time.time()
    X = make()
    est, tz = prop_all(X)
    r = dict(err=errs(est), attr=attribution(est, tz))
    res['fam'][name] = r
    print(name, 'err 24..32', ' '.join(f'{v:.2e}' for v in r['err']), 'attr', json.dumps(r['attr']),
          f'{time.time() - t1:.0f}s', flush=True)
    del X
os.makedirs('runs/diag10', exist_ok=True)
json.dump(res, open(f'runs/diag10/diag10_{seed}.json', 'w'), indent=1)
print(f'done {time.time() - t0:.0f}s', flush=True)
