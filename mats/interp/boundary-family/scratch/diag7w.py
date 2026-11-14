"""PREREG measurement D (descriptive): gate compression and latent control of near-boundary gates.
At layers m in CUTS of a width-256 ARC net (sigma = 0.001), from 2^NLOG2 Monte Carlo paths:
  D_raw            gates with entropy > 0.1 bit (and the sum of their marginal entropies)
  D_eff(sw, c)     single-linkage groups of near gates with |phi| > c (co-switching)
  D_eff(eff, c)    groups with |cos| > c of the finite-difference effect vectors E[d a_32 / d z_m,i] (common noise)
  latent control   fraction of near gates whose sign sign(mu_i + U_k,i . y) (top-k PCA latent of z_m) agrees with the
                   actual sign in >= 90% of paths, k = 3, 8
python scratch/diag7.py SEED CUTS NLOG2"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf

seed = int(sys.argv[1]); cuts = [int(v) for v in sys.argv[2].split(',')]; N = 1 << int(sys.argv[3])
L = int(os.environ.get("LNET", "32")); n = int(os.environ.get("WID", "256"))
net = bf.get_net(seed, n, 32, 0.001)
W = [w.float() for w in net.W]; s = [float(v) for v in net.s]
t0 = time.time()
smp = bf.mc_stats(net, N, seed=90_000 + seed, L=max(cuts), cuts=tuple(cuts), keep_n=N, out_moments=False)
print(f'samples {time.time() - t0:.0f}s', flush=True)


def tail(zm, m, gen):
    h = torch.relu(zm) + s[m - 1] * torch.randn(zm.shape, generator=gen)
    for l in range(m, L):
        a = torch.relu(h @ W[l].T)
        h = a + s[l] * torch.randn(a.shape, generator=gen) if l < L - 1 else a
    return h


def ncomp(A, c):
    adj = (A.abs() > c)
    lab = list(range(A.shape[0]))
    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]; x = lab[x]
        return x
    for a, b in torch.nonzero(torch.triu(adj, 1)).tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            lab[ra] = rb
    return len(set(find(x) for x in range(A.shape[0])))


out = {}
for m in cuts:
    Z = smp['cuts'][m].double()
    G = Z > 0; p = G.double().mean(0)
    ent = -(p * p.clamp_min(1e-12).log2() + (1 - p) * (1 - p).clamp_min(1e-12).log2())
    near = torch.nonzero(ent > 0.1).flatten(); D = int(near.numel())
    r = dict(D_raw=D, ent_sum=float(ent[near].sum()))
    Gn = G[:, near].double(); Gc = Gn - Gn.mean(0)
    sd = Gc.std(0).clamp_min(1e-12)
    phi = (Gc.T @ Gc) / (Gc.shape[0] - 1) / (sd[:, None] * sd[None, :])
    iu = torch.triu_indices(D, D, 1)
    pv = phi[iu[0], iu[1]].abs()
    r['phi_median'] = float(pv.median()); r['phi_frac_gt_0.5'] = float((pv > 0.5).double().mean())
    r['phi_max'] = float(pv.max())
    r['D_eff_sw'] = {str(c): ncomp(phi, c) for c in (0.5, 0.8, 0.9, 0.95)}
    # effect vectors
    B = 256; zb = Z[:B].float(); sdz = Z.std(0)
    base = tail(zb, m, torch.Generator().manual_seed(9))
    eff = torch.zeros(D, n, dtype=torch.float64)
    for j, i in enumerate(near.tolist()):
        zp = zb.clone(); dz = 1e-3 * float(sdz[i]); zp[:, i] += dz
        eff[j] = (tail(zp, m, torch.Generator().manual_seed(9)) - base).double().mean(0) / dz
    en = eff / eff.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ce = en @ en.T
    r['eff_cos_median'] = float(ce[iu[0], iu[1]].abs().median())
    r['D_eff_eff'] = {str(c): ncomp(ce, c) for c in (0.5, 0.8, 0.9, 0.95)}
    # latent control
    mz = Z.mean(0); ev, U = torch.linalg.eigh(torch.cov(Z.T)); U = U.flip(1); ev = ev.flip(0)
    r['pca_top_frac'] = [round(float(v), 3) for v in (ev[:4] / ev.sum())]
    for k in (3, 8):
        Uk = U[:, :k]; y = (Z - mz) @ Uk
        pred = (mz[near] + y @ Uk[near].T) > 0
        acc = (pred == G[:, near]).double().mean(0)
        r[f'latent{k}_frac_acc90'] = float((acc >= 0.9).double().mean())
        r[f'latent{k}_acc_median'] = float(acc.median())
    out[m] = r
    print(m, json.dumps(r), f'{time.time() - t0:.0f}s', flush=True)
os.makedirs('runs/diag7', exist_ok=True)
json.dump(out, open(f'runs/diag7/diag7_{seed}_w{n}_L{L}.json', 'w'), indent=1)
print('done', flush=True)
