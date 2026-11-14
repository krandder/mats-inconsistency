"""Exploration 1 (dev nets only, width 64): which representation of the layer-m state, followed by the EXACT remaining
layers, keeps the depth-L means right?  (Representation oracles; no claim.)

python scratch/explore1.py SEED L CUTS(comma) [NTRUTH_LOG2] [NO_LOG2]
"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import bf

torch.set_num_threads(int(os.environ.get('TH', '3')))
seed, L = int(sys.argv[1]), int(sys.argv[2])
cuts = [int(v) for v in sys.argv[3].split(',')]
NT = 1 << int(sys.argv[4]) if len(sys.argv) > 4 else 1 << 24
NO = 1 << int(sys.argv[5]) if len(sys.argv) > 5 else 1 << 21
n = 64
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'explore1')
os.makedirs(OUT, exist_ok=True)
net = bf.get_net(seed, n, 32, 0.001)
t0 = time.time()
tpath = os.path.join(OUT, f'truth_{seed}_L{L}_N{int(math.log2(NT))}.pt')
if os.path.exists(tpath):
    tr = torch.load(tpath)
else:
    tr = bf.mc_stats(net, NT, seed=10_000 + seed, L=L)
    torch.save(dict(mean=tr['mean'], sq=tr['sq'], zmom=tr['zmom'], N=NT), tpath)
tm = tr['mean'][L - 1]; tv = tr['sq'][L - 1] - tm ** 2
print(f'truth done {time.time() - t0:.0f}s', flush=True)
smp = bf.mc_stats(net, 1 << 20, seed=20_000 + seed, L=L, cuts=tuple(cuts), keep_n=1 << 20, out_moments=False)
print(f'samples done {time.time() - t0:.0f}s', flush=True)


def prop_h(hs, m):
    """propagate from post-noise h_m samples: z_{m+1} = W_{m+1} h_m ..."""
    z = hs.float() @ net.W[m].float().T
    return bf.propagate_from(net, z, m + 1, L, seed=7)


def gauss_draw(X, N, g):
    mu = X.double().mean(0); C = torch.cov(X.double().T)
    Lc = torch.linalg.cholesky(C + 1e-12 * torch.eye(C.shape[0], dtype=C.dtype))
    return (mu + torch.randn(N, X.shape[1], generator=g, dtype=torch.float64) @ Lc.T).float()


res = dict(seed=seed, L=L, NT=NT, NO=NO, by_cut={})
for m in cuts:
    Z = smp['cuts'][m]                                   # (2^20, n) pre-activations of layer m
    g = torch.Generator().manual_seed(123 + m)
    r = {}
    # control: resample the empirical pre-activations
    idx = torch.randint(0, Z.shape[0], (NO,), generator=g)
    r['emp'] = bf.nmse(bf.propagate_from(net, Z[idx], m, L, seed=7), tm, tv)
    # O1: h_m Gaussian (post-activation + noise)
    if m < L:
        H = torch.relu(Z) + float(net.s[m - 1]) * torch.randn(Z.shape, generator=g)
        r['gauss_h'] = bf.nmse(prop_h(gauss_draw(H, NO, g), m), tm, tv)
    # O2: z_m Gaussian
    r['gauss_z'] = bf.nmse(bf.propagate_from(net, gauss_draw(Z, NO, g), m, L, seed=7), tm, tv)
    # O3: scale mixture: z = s * gdir, s = |z| / mean|z| empirical (independent), gdir Gaussian
    nrm = Z.double().norm(dim=1); s = nrm / nrm.mean()
    G = (Z.double() / s[:, None]).float()
    sd = s[torch.randint(0, Z.shape[0], (NO,), generator=g)]
    r['scale_z'] = bf.nmse(bf.propagate_from(net, (gauss_draw(G, NO, g).double() * sd[:, None]).float(), m, L,
                                             seed=7), tm, tv)
    # O3b: scale mixture with the joint (s, gdir) kept for the scale only via s-bins (gdir Gaussian within s-bin)
    nb = 16
    qs = torch.quantile(s, torch.linspace(0, 1, nb + 1, dtype=torch.float64))
    bins = torch.bucketize(s, qs[1:-1])
    parts = []
    for b in range(nb):
        sel = bins == b
        k = int(round(NO * float(sel.float().mean())))
        if k == 0:
            continue
        parts.append(gauss_draw(Z[sel], k, g))
    r['sbins16_z'] = bf.nmse(bf.propagate_from(net, torch.cat(parts, 0), m, L, seed=7), tm, tv)
    # O5: top-k PCA of z_m exact (empirical projections) + independent Gaussian residual
    mu = Z.double().mean(0); C = torch.cov(Z.double().T)
    ev, U = torch.linalg.eigh(C); ev = ev.flip(0); U = U.flip(1)
    r['pr_z'] = float(ev.sum() ** 2 / (ev ** 2).sum())
    r['energy_mean_frac_z'] = float((mu ** 2).sum() / ((mu ** 2).sum() + ev.sum()))
    Hh = torch.relu(Z.double())
    muh = Hh.mean(0); evh = torch.linalg.eigvalsh(torch.cov(Hh.T))
    r['pr_h'] = float(evh.sum() ** 2 / (evh ** 2).sum())
    r['energy_mean_frac_h'] = float((muh ** 2).sum() / ((muh ** 2).sum() + evh.sum()))
    sdz = C.diagonal().sqrt(); marg = (mu / sdz)
    r['frac_absmargin_lt'] = {str(t): float((marg.abs() < t).float().mean()) for t in (0.5, 1, 2, 3, 4)}
    for k in (1, 2, 4, 8, 16, 32):
        Uk = U[:, :k]
        Y = (Z.double() - mu) @ Uk                                   # exact projections
        Rres = (Z.double() - mu) - Y @ Uk.T
        Cr = torch.cov(Rres.T) + 1e-10 * torch.eye(n, dtype=torch.float64)
        ev2, V2 = torch.linalg.eigh(Cr)
        Lr = V2 * ev2.clamp_min(0).sqrt()
        ii = torch.randint(0, Z.shape[0], (NO,), generator=g)
        Zs = mu + Y[ii] @ Uk.T + torch.randn(NO, n, generator=g, dtype=torch.float64) @ Lr.T
        r[f'proj{k}_z'] = bf.nmse(bf.propagate_from(net, Zs.float(), m, L, seed=7), tm, tv)
    # O4: cells = gate patterns of the D most uncertain neurons of layer m; Gaussian per cell (untruncated, and
    # truncated to the cell by rejection)
    pon = (Z > 0).double().mean(0)
    ent = -(pon * pon.clamp_min(1e-12).log2() + (1 - pon) * (1 - pon).clamp_min(1e-12).log2())
    order = torch.argsort(ent, descending=True)
    r['gate_entropy_sum'] = float(ent.sum())
    for D in (1, 2, 4, 8):
        sel = order[:D]
        pat = ((Z[:, sel] > 0).long() * (1 << torch.arange(D))).sum(1)
        parts_u = []; parts_t = []
        for c in torch.unique(pat):
            S = pat == c
            k = int(round(NO * float(S.float().mean())))
            if k == 0:
                continue
            if int(S.sum()) < 4 * n:                                 # too few to fit: resample empirically
                Zc = Z[S]; parts_u.append(Zc[torch.randint(0, Zc.shape[0], (k,), generator=g)])
                parts_t.append(parts_u[-1]); continue
            parts_u.append(gauss_draw(Z[S], k, g))
            got = []; need = k
            for _ in range(200):
                d = gauss_draw(Z[S], max(2 * need, 1024), g)
                pd = ((d[:, sel] > 0).long() * (1 << torch.arange(D))).sum(1)
                d = d[pd == c]
                got.append(d[:need]); need -= got[-1].shape[0]
                if need <= 0:
                    break
            parts_t.append(torch.cat(got, 0))
        r[f'cells{D}_z'] = bf.nmse(bf.propagate_from(net, torch.cat(parts_u, 0), m, L, seed=7), tm, tv)
        r[f'cellsT{D}_z'] = bf.nmse(bf.propagate_from(net, torch.cat(parts_t, 0), m, L, seed=7), tm, tv)
    res['by_cut'][m] = r
    print(m, json.dumps({k: (round(v, 9) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(res, open(os.path.join(OUT, f'ex1_{seed}_L{L}.json'), 'w'), indent=1)
print(f'done {time.time() - t0:.0f}s', flush=True)
