"""E0 (descriptive): the size-biased angular law nu of the state, layer by layer, for the attractor question.
Polar-gnomonic chart of z_l: u = E z / |E z|, rho = u . z, delta = z / rho - u; nu = rho-weighted law of delta
(weights rho / sum rho). Per layer l:
- spread = sqrt(tr Cov_nu delta), spread x l; PR and top-5 shares of Cov_nu; sd / skew / kurtosis of log rho;
- whitened top-10 principal-component scores of nu: skew and excess kurtosis (signs fixed: third moment >= 0);
- fourth-order dependence among the top 5: |corr| of squared scores (median, max);
- rho-delta dependence: corr(log rho, score_j) and corr(log rho, |score_j|), j = 1..3;
- the next layer's normalized pre-activations t = W_{l+1} ReLU(u + delta) (z_{l+1} = rho t + noise; the gate of
  layer l+1 depends on t only): gate-relevant neurons (|mean_nu t_i| / sd_nu t_i < 1), median |skew| and excess kurtosis
  there and over all neurons; the same for the raw z_{l+1} (unweighted, with the scale);
- candidate (b)'s consistency: median over gate-relevant neurons of the R^2 of t_i regressed linearly on the top-k
  nu-PC scores (k = 2, 4, 8, 16).
python scratch/e0.py SEED N_LOG2 [n L sigma] --out runs/e0"""
import os, sys, time, json, math
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import bf, ng

seed = int(sys.argv[1]); N = 1 << int(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
L = int(sys.argv[4]) if len(sys.argv) > 4 else 64
sigma = float(sys.argv[5]) if len(sys.argv) > 5 else 0.001
torch.set_num_threads(int(os.environ.get('TH', '3')))
dev = ng.device()
if dev.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False
os.chdir(os.path.join(HERE, '..'))
net = bf.get_net(seed, n, L, sigma)
dn = ng.DNet(net, dev)
cuts = [l for l in (2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 63) if l < L]
res = dict(seed=seed, n=n, L=L, sigma=sigma, N=N, layers={})


def wst(x, w):
    m = (w[:, None] * x).sum(0); d = x - m
    v = (w[:, None] * d * d).sum(0)
    sk = (w[:, None] * d ** 3).sum(0) / v.clamp_min(1e-300) ** 1.5
    ku = (w[:, None] * d ** 4).sum(0) / v.clamp_min(1e-300) ** 2 - 3
    return m, v.sqrt(), sk, ku


g = torch.Generator(device=dev).manual_seed(400_000 + seed)
t0 = time.time()
with torch.no_grad():
    h = torch.randn(N, n, generator=g, device=dev)
    for l in range(1, L):
        z = h @ dn.W[l - 1].T
        if l in cuts:
            Z = z.double()
            u = Z.mean(0); u = u / u.norm()
            rho = Z @ u
            negfrac = float((rho <= 0).double().mean())
            rp = rho.clamp_min(1e-12)
            D = Z / rp[:, None] - u
            w = rp / rp.sum()
            md = (w[:, None] * D).sum(0); Dc = D - md
            C = (Dc * w[:, None]).T @ Dc
            ev, V = torch.linalg.eigh(C); ev = ev.flip(0).clamp_min(0); V = V.flip(1)
            spread = float(ev.sum().sqrt()); pr = float(ev.sum() ** 2 / (ev ** 2).sum())
            lr = rp.log(); u1 = torch.full_like(lr, 1.0 / N)
            _, lsd, lsk, lku = wst(lr[:, None], u1)
            Y = Dc @ V[:, :16]
            Yw = Y[:, :10] / ev[:10].sqrt()
            _, _, psk, pku = wst(Yw, w)
            sgn = torch.sign(psk); sgn[sgn == 0] = 1; psk = psk * sgn
            Y2 = Yw[:, :5] ** 2; Y2c = Y2 - (w[:, None] * Y2).sum(0)
            c2 = (Y2c * w[:, None]).T @ Y2c; d2 = c2.diagonal().sqrt(); c2 = c2 / (d2[:, None] * d2[None, :])
            offd = c2[~torch.eye(5, dtype=torch.bool, device=dev)].abs()
            lrc = lr - lr.mean()
            cl = [float((lrc * (Yw[:, j] - Yw[:, j].mean())).mean() / (lrc.std() * Yw[:, j].std())) for j in range(3)]
            ca = [float((lrc * (Yw[:, j].abs() - Yw[:, j].abs().mean())).mean() / (lrc.std() * Yw[:, j].abs().std()))
                  for j in range(3)]
            Wn = dn.W[l].double()
            T = torch.relu(u + D) @ Wn.T
            tm, tsd, tsk, tku = wst(T, w)
            nb = (tm.abs() / tsd.clamp_min(1e-300)) < 1.0
            Zn = torch.relu(Z) @ Wn.T
            zm, zsd, zsk, zku = wst(Zn, u1)
            nbz = (zm.abs() / zsd) < 1.0
            # (b): R^2 of t_i on the top-k nu-PC scores (weighted least squares), gate-relevant neurons
            r2 = {}
            if nb.any():
                Tn = T[:, nb] - (w[:, None] * T[:, nb]).sum(0)
                vt = (w[:, None] * Tn * Tn).sum(0)
                for k in (2, 4, 8, 16):
                    Yk = Y[:, :k]
                    G = (Yk * w[:, None]).T @ Yk
                    B = torch.linalg.solve(G, (Yk * w[:, None]).T @ Tn)
                    res_ = Tn - Yk @ B
                    r2[k] = float((1 - (w[:, None] * res_ * res_).sum(0) / vt).median())
            rec = dict(negfrac=negfrac, spread=spread, spread_x_l=spread * l, pr=pr,
                       top_share=[float(x) for x in (ev[:5] / ev.sum())],
                       logrho_sd=float(lsd), logrho_skew=float(lsk), logrho_kurt=float(lku),
                       pc_skew=[round(float(x), 3) for x in psk], pc_kurt=[round(float(x), 3) for x in pku],
                       sqcorr_top5_med=float(offd.median()), sqcorr_top5_max=float(offd.max()),
                       corr_logrho_lin=cl, corr_logrho_abs=ca,
                       nb_count_t=int(nb.sum()),
                       t_nb_abs_skew_med=float(tsk[nb].abs().median()) if nb.any() else None,
                       t_nb_kurt_med=float(tku[nb].median()) if nb.any() else None,
                       t_all_abs_skew_med=float(tsk.abs().median()), t_all_kurt_med=float(tku.median()),
                       nb_count_z=int(nbz.sum()),
                       z_nb_abs_skew_med=float(zsk[nbz].abs().median()) if nbz.any() else None,
                       z_nb_kurt_med=float(zku[nbz].median()) if nbz.any() else None,
                       z_all_abs_skew_med=float(zsk.abs().median()), z_all_kurt_med=float(zku.median()),
                       r2_topk=r2)
            res['layers'][l] = rec
            print(l, json.dumps({k2: (round(v2, 4) if isinstance(v2, float) else v2) for k2, v2 in rec.items()}),
                  f'{time.time() - t0:.0f}s', flush=True)
            del Z, D, Dc, T, Zn, Y
        h = torch.relu(z) + dn.s[l - 1] * torch.randn(N, n, generator=g, device=dev)
out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else 'runs/e0'
os.makedirs(out, exist_ok=True)
json.dump(res, open(f'{out}/e0_{n}_{L}_{seed}.json', 'w'), indent=1)
print('done', flush=True)
