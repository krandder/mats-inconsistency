"""PREREG amendment 1, test K-F: does a FACTORED model of layer m reproduce the joint gate statistics that matter for
the next layer's mean (pairwise and triple co-switching along W), at width 256?

Two independent Monte Carlo sets of z_m (fit set A, reference set B; 2^NLOG2 paths each). Models fitted on A:
  G      full Gaussian (true mean and covariance): all pairwise structure, nothing beyond
  IND    independent coordinates with their empirical marginals (product model, no latent)
  F3, F8 z = mu + U_k y + r: y = exact top-k PCA coordinates (resampled jointly from A), r Gaussian with DIAGONAL
         covariance (the residual's variances), independent of y: coordinates (and gates) conditionally independent
         given the k-dim latent
  F3C    as F3 with per-cell residual means and variances (4^3 quantile cells of y): heteroscedastic conditional
         independence
  C3     unfactored reference: per-cell full Gaussians on the same 64 cells
Each model is sampled (2^NLOG2 paths), pushed one layer exactly (ReLU, noise, W_{m+1}); compared with the reference set
pushed the same way (same noise seed):
  var_err   median over neurons of |Var_model(z_{m+1,j}) / Var_ref - 1|      (pairwise co-switching along W)
  k3_err    median over neurons of |skew_model - skew_ref|                  (triple co-switching along W)
  mean_err  nMSE of E[a_{m+1}] (units of mean Var(a_{m+1}))                  (what the next layer's mean needs)
python scratch/diag8.py SEED CUTS NLOG2"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf

seed = int(sys.argv[1]); cuts = [int(v) for v in sys.argv[2].split(',')]; N = 1 << int(sys.argv[3])
n = int(os.environ.get('WID', '256'))
net = bf.get_net(seed, n, 32, 0.001)
W = [w.double() for w in net.W]; s = [float(v) for v in net.s]
t0 = time.time()
A = bf.mc_stats(net, N, seed=100_000 + seed, L=max(cuts), cuts=tuple(cuts), keep_n=N, out_moments=False)['cuts']
B = bf.mc_stats(net, N, seed=110_000 + seed, L=max(cuts), cuts=tuple(cuts), keep_n=N, out_moments=False)['cuts']
print(f'samples {time.time() - t0:.0f}s', flush=True)


def push(Zm, m):
    g = torch.Generator().manual_seed(3)
    h = torch.relu(Zm) + s[m - 1] * torch.randn(Zm.shape, generator=g, dtype=torch.float64)
    return h @ W[m].T


def stats(Z1):
    mu = Z1.mean(0); v = Z1.var(0); sk = ((Z1 - mu) ** 3).mean(0) / v ** 1.5
    return v, sk, torch.relu(Z1).mean(0)


out = {}
for m in cuts:
    ZA = A[m].double(); ZB = B[m].double()
    vR, skR, aR = stats(push(ZB, m))
    va = float(torch.relu(push(ZB, m)).var(0).mean())
    g = torch.Generator().manual_seed(17)
    mu = ZA.mean(0); C = torch.cov(ZA.T)
    ev, U = torch.linalg.eigh(C); U = U.flip(1)
    models = {}
    Lc = torch.linalg.cholesky(C + 1e-10 * torch.eye(n, dtype=torch.float64))
    models['G'] = mu + torch.randn(N, n, generator=g, dtype=torch.float64) @ Lc.T
    idx = torch.randint(0, N, (N, n), generator=g)
    models['IND'] = torch.gather(ZA, 0, idx)
    for k in (3, 8):
        Uk = U[:, :k]; Y = (ZA - mu) @ Uk; R = ZA - mu - Y @ Uk.T
        ii = torch.randint(0, N, (N,), generator=g)
        models[f'F{k}'] = mu + Y[ii] @ Uk.T + torch.randn(N, n, generator=g, dtype=torch.float64) * R.std(0)
        if k == 3:
            lab = torch.zeros(N, dtype=torch.long)
            for j in range(3):
                qs = torch.quantile(Y[:, j], torch.linspace(0, 1, 5, dtype=torch.float64)[1:-1])
                lab = lab * 4 + torch.bucketize(Y[:, j].contiguous(), qs)
            F3C = torch.empty(N, n, dtype=torch.float64); C3 = torch.empty(N, n, dtype=torch.float64)
            for c in torch.unique(lab):
                S = torch.nonzero(lab == c).flatten()
                Rc = R[S]; Zc = ZA[S]
                # same cell sizes, y resampled within the cell
                jj = S[torch.randint(0, S.numel(), (S.numel(),), generator=g)]
                F3C[S] = mu + Y[jj] @ Uk.T + Rc.mean(0) + torch.randn(S.numel(), n, generator=g, dtype=torch.float64) * Rc.std(0)
                Cc = torch.cov(Zc.T) + 1e-10 * torch.eye(n, dtype=torch.float64)
                ev2, V2 = torch.linalg.eigh(Cc)
                C3[S] = Zc.mean(0) + torch.randn(S.numel(), n, generator=g, dtype=torch.float64) @ (V2 * ev2.clamp_min(0).sqrt()).T
            models['F3C'] = F3C; models['C3'] = C3
    r = {}
    for name, Zm in models.items():
        v, sk, a = stats(push(Zm, m))
        r[name] = dict(var_err=float(((v / vR) - 1).abs().median()), k3_err=float((sk - skR).abs().median()),
                       mean_err=float(((a - aR) ** 2).mean()) / va)
    r['ref'] = dict(skew_abs_median=float(skR.abs().median()))
    # sampling-noise reference: set A itself (independent of B)
    v, sk, a = stats(push(ZA, m))
    r['A_vs_B'] = dict(var_err=float(((v / vR) - 1).abs().median()), k3_err=float((sk - skR).abs().median()),
                       mean_err=float(((a - aR) ** 2).mean()) / va)
    out[m] = r
    print(m, json.dumps({k: {kk: float(f'{vv:.3g}') for kk, vv in d.items()} for k, d in r.items()}),
          f'{time.time() - t0:.0f}s', flush=True)
os.makedirs('runs/diag8', exist_ok=True)
json.dump(out, open(f'runs/diag8/diag8_{seed}_w{n}.json', 'w'), indent=1)
print('done', flush=True)
