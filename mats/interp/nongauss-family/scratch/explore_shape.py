"""EXPLORATION (net 112 only, not a test): how does the size-biased angular law of the state evolve with depth?
Polar-gnomonic chart of the pre-activation z_l: u = E z / |E z|, rho = u . z (> 0 at depth), delta = z / rho - u (so
z = rho (u + delta), delta orthogonal to u). By positive homogeneity, E[f(z)] = E[rho] E_nu[f(u + delta)] for any
degree-1 homogeneous f (noiseless), with nu = the rho-weighted (size-biased) law of delta.
Per layer: angular spread, PR of Cov_nu(delta), log-rho cumulants, rho-delta dependence, shape (skew / excess kurtosis)
of the next layer's normalized pre-activations t_i = w_i . (u + delta) at near-boundary neurons, whitened top-PC shapes,
and fourth-order dependence among top PCs (corr of squared scores).
python scratch/explore_shape.py SEED N_LOG2 [n L]"""
import os, sys, time, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf

seed = int(sys.argv[1]); N = 1 << int(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
L = int(sys.argv[4]) if len(sys.argv) > 4 else 64
net = bf.get_net(seed, n, L, 0.001)
W = [w.float() for w in net.W]; s = [float(v) for v in net.s]
g = torch.Generator().manual_seed(400_000 + seed)
cuts = sorted(set([1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 63]) & set(range(1, L)))
res = dict(seed=seed, n=n, L=L, N=N, layers={})


def wstats(x, w):
    """weighted mean, sd, skew, excess kurtosis of columns of x (N, k) with weights w (N,) summing to 1."""
    m = (w[:, None] * x).sum(0); d = x - m
    v = (w[:, None] * d ** 2).sum(0)
    sk = (w[:, None] * d ** 3).sum(0) / v.clamp_min(1e-300) ** 1.5
    ku = (w[:, None] * d ** 4).sum(0) / v.clamp_min(1e-300) ** 2 - 3
    return m, v.sqrt(), sk, ku


t0 = time.time()
with torch.no_grad():
    h = torch.randn(N, n, generator=g)
    for l in range(1, L):
        z = h @ W[l - 1].T
        if l in cuts:
            Z = z.double()
            mz = Z.mean(0); u = mz / mz.norm()
            rho = Z @ u
            negfrac = float((rho <= 0).double().mean())
            rp = rho.clamp_min(1e-12)
            D = Z / rp[:, None] - u                                  # delta, orthogonal to u
            w = rp / rp.sum()                                        # size-biased weights
            md = (w[:, None] * D).sum(0)
            Dc = D - md
            C = (Dc * w[:, None]).T @ Dc
            ev, V = torch.linalg.eigh(C); ev = ev.flip(0).clamp_min(0); V = V.flip(1)
            spread = float(ev.sum().sqrt())
            pr = float(ev.sum() ** 2 / (ev ** 2).sum())
            lr = rho.clamp_min(1e-12).log()
            lm, lsd, lsk, lku = wstats(lr[:, None], torch.full_like(lr, 1.0 / N))
            # top-10 whitened PC scores (size-biased)
            k = 10
            Y = Dc @ V[:, :k] / ev[:k].sqrt()
            _, _, psk, pku = wstats(Y, w)
            # dependence: corr of squared scores among top 5 (size-biased), and corr of log rho with |scores|
            Y2 = Y[:, :5] ** 2
            Y2c = Y2 - (w[:, None] * Y2).sum(0)
            c2 = (Y2c * w[:, None]).T @ Y2c; d2 = c2.diagonal().sqrt(); c2 = c2 / (d2[:, None] * d2[None, :])
            offd = c2[~torch.eye(5, dtype=torch.bool)].abs()
            lrc = lr - lr.mean()
            corr_lr_abs = [float(((lrc * (Y[:, j].abs() - Y[:, j].abs().mean())).mean()) /
                                 (lrc.std() * Y[:, j].abs().std())) for j in range(3)]
            corr_lr_lin = [float(((lrc * (Y[:, j] - Y[:, j].mean())).mean()) / (lrc.std() * Y[:, j].std())) for j in range(3)]
            # next layer's normalized pre-activations t = W_{l+1} (u + delta) (gate depends only on t)
            Wn = W[l].double()
            T = (u + D) @ Wn.T
            tm, tsd, tsk, tku = wstats(T, w)
            marg = tm.abs() / tsd.clamp_min(1e-300)
            nb = marg < 1.0
            # raw (unnormalized) next pre-activation z_{l+1} without noise, unweighted
            Zn = Z.clamp_min(0) @ Wn.T
            zm, zsd, zsk, zku = wstats(Zn, torch.full((N,), 1.0 / N, dtype=torch.float64))
            nbz = (zm.abs() / zsd) < 1.0
            rec = dict(negfrac=negfrac, spread=spread, spread_x_l=spread * l, pr=pr,
                       top_share=[float(x) for x in (ev[:5] / ev.sum())],
                       logrho_sd=float(lsd), logrho_skew=float(lsk), logrho_kurt=float(lku),
                       pc_skew=[round(float(x), 3) for x in psk], pc_kurt=[round(float(x), 3) for x in pku],
                       sqcorr_top5_med=float(offd.median()), sqcorr_top5_max=float(offd.max()),
                       corr_logrho_abs=corr_lr_abs, corr_logrho_lin=corr_lr_lin,
                       nb_count_t=int(nb.sum()), t_nb_abs_skew_med=float(tsk[nb].abs().median()) if nb.any() else None,
                       t_nb_kurt_med=float(tku[nb].median()) if nb.any() else None,
                       t_all_abs_skew_med=float(tsk.abs().median()), t_all_kurt_med=float(tku.median()),
                       nb_count_z=int(nbz.sum()), z_nb_abs_skew_med=float(zsk[nbz].abs().median()) if nbz.any() else None,
                       z_nb_kurt_med=float(zku[nbz].median()) if nbz.any() else None,
                       z_all_abs_skew_med=float(zsk.abs().median()), z_all_kurt_med=float(zku.median()))
            res['layers'][l] = rec
            print(l, json.dumps({k2: (round(v2, 4) if isinstance(v2, float) else v2) for k2, v2 in rec.items()}),
                  f'{time.time() - t0:.0f}s', flush=True)
            del Z, D, Dc, T, Zn
        a = torch.relu(z)
        h = a + s[l - 1] * torch.randn(N, n, generator=g)
os.makedirs('runs/explore', exist_ok=True)
json.dump(res, open(f'runs/explore/shape_{n}_{L}_{seed}.json', 'w'), indent=1)
print('done', flush=True)
