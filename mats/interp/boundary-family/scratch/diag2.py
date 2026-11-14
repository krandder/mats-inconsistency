"""Diagnostic 2 (dev nets, width 64): after factoring the scale out (z = exp(rho)(u + d), u = mean direction,
rho = log(u.z), d = P z / (u.z)), how non-Gaussian is the direction d, and which representation of it gives the
one-step moments of a = ReLU(z)?  Oracles fitted to true samples:
  G      Gaussian z with the true mean / cov
  LRG    (rho, d) jointly Gaussian (true moments); E[a] by Gaussian tilting, Cov(a) by tilting (bivariate)
  LRGCk  LRG within cells of the top-k PCA modes of d (4^k cells, 16 for k = 1): per cell (rho, d) jointly Gaussian
  DCk    cells of the top-k modes of d with the empirical rho inside each cell (scale exact per sample), d Gaussian
         per cell, independent of rho within the cell
Errors of the one-step mean of a (nMSE, units of mean Var(a)) and covariance (relative Frobenius).
python scratch/diag2.py SEED L CUTS NLOG2"""
import json, math, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '2')))
import bf
from gauss_relu import relu_moments, ncdf, npdf

seed, L = int(sys.argv[1]), int(sys.argv[2])
cuts = [int(v) for v in sys.argv[3].split(',')]
N = 1 << int(sys.argv[4])
n = 64
net = bf.get_net(seed, n, 32, 0.001)
t0 = time.time()
smp = bf.mc_stats(net, N, seed=40_000 + seed, L=L, cuts=tuple(cuts), keep_n=N, out_moments=False)


def relu_mean(m, s):
    t = m / s
    return s * (t * ncdf(t) + npdf(t))


def lrg_moments(rho, D, u):
    """Mean and covariance of a = exp(rho) ReLU(u + d) for (rho, d) jointly Gaussian with the sample moments."""
    Er = rho.mean(); Vr = rho.var()
    Ed = D.mean(0); Cd = torch.cov(D.T); c = ((D - Ed) * (rho - Er)[:, None]).mean(0)
    sd = Cd.diagonal().clamp_min(1e-300).sqrt()
    mu = torch.exp(Er + Vr / 2) * relu_mean(u + Ed + c, sd)
    m2, C2, _ = relu_moments(u + Ed + 2 * c, Cd)              # E[exp(2 rho) f g] = E exp(2 rho) E[f g](shift 2c)
    E2 = torch.exp(2 * Er + 2 * Vr) * (C2 + torch.outer(m2, m2))
    return mu, E2


out = {}
for l in cuts:
    Z = smp['cuts'][l].double()
    A = torch.relu(Z); ma = A.mean(0); Ea2 = (A.T @ A) / A.shape[0]; Ca = Ea2 - torch.outer(ma, ma)
    va = float(Ca.diagonal().mean())
    mz = Z.mean(0); u = mz / mz.norm()
    g = Z @ u
    r = dict(frac_uz_pos=float((g > 0).double().mean()))
    pos = g > 0
    Zp = Z[pos]; gp = g[pos]
    rho = gp.log(); D = (Zp - torch.outer(gp, u)) / gp[:, None]          # d = P z / (u.z), orthogonal to u
    Ds = (D - D.mean(0)) / D.std(0).clamp_min(1e-12)
    Zs = (Z - mz) / Z.std(0)
    r['skew_z'] = float((Zs ** 3).mean(0).abs().median()); r['exkurt_z'] = float(((Zs ** 4).mean(0) - 3).median())
    r['skew_d'] = float((Ds ** 3).mean(0).abs().median()); r['exkurt_d'] = float(((Ds ** 4).mean(0) - 3).median())
    r['sd_rho'] = float(rho.std()); r['skew_rho'] = float((((rho - rho.mean()) / rho.std()) ** 3).mean())
    evd = torch.linalg.eigvalsh(torch.cov(D.T)).flip(0)
    r['pr_d'] = float(evd.sum() ** 2 / (evd ** 2).sum()); r['top_d'] = [round(float(v), 3) for v in evd[:4] / evd.sum()]
    sdD = D.std(0); md = (u + D.mean(0)) / sdD
    r['d_t_abs_med'] = float(md.abs().median()); r['d_frac_t_lt1'] = float((md.abs() < 1).double().mean())
    # G
    mg, Cg, _ = relu_moments(mz, torch.cov(Z.T))
    r['G_mean'] = float(((mg - ma) ** 2).mean()) / va; r['G_cov'] = float((Cg - Ca).norm() / Ca.norm())
    # LRG (on the u.z > 0 part; the rest has weight ~0 at depth)
    w = float(pos.double().mean())
    mu, E2 = lrg_moments(rho, D, u)
    mu = w * mu + (1 - w) * torch.relu(Z[~pos]).mean(0) if (~pos).any() else mu
    C = E2 - torch.outer(mu, mu)
    r['LRG_mean'] = float(((mu - ma) ** 2).mean()) / va; r['LRG_cov'] = float((C - Ca).norm() / Ca.norm())
    # cells in the top-k modes of d
    _, Ud = torch.linalg.eigh(torch.cov(D.T)); Ud = Ud.flip(1)
    for k in (1, 2, 3):
        Y = (D - D.mean(0)) @ Ud[:, :k]
        nb = 16 if k == 1 else 4
        lab = torch.zeros(D.shape[0], dtype=torch.long)
        for j in range(k):
            qs = torch.quantile(Y[:, j], torch.linspace(0, 1, nb + 1, dtype=torch.float64)[1:-1])
            lab = lab * nb + torch.bucketize(Y[:, j].contiguous(), qs)
        muL = torch.zeros(n, dtype=torch.float64); E2L = torch.zeros(n, n, dtype=torch.float64)
        muD = torch.zeros_like(muL); E2D = torch.zeros_like(E2L)
        for cc in torch.unique(lab):
            S = lab == cc; wc = float(S.double().mean())
            m1, e2 = lrg_moments(rho[S], D[S], u)
            muL += wc * m1; E2L += wc * e2
            # DC: d Gaussian per cell, rho empirical and independent of d within the cell
            Dc = D[S]; Edc = Dc.mean(0); Cdc = torch.cov(Dc.T)
            sdc = Cdc.diagonal().clamp_min(1e-300).sqrt()
            er = gp[S].mean(); er2 = (gp[S] ** 2).mean()
            m1d = er * relu_mean(u + Edc, sdc)
            m2d, C2d, _ = relu_moments(u + Edc, Cdc)
            muD += wc * m1d; E2D += wc * er2 * (C2d + torch.outer(m2d, m2d))
        CL = E2L - torch.outer(muL, muL); CD = E2D - torch.outer(muD, muD)
        r[f'LRGC{k}_mean'] = float(((muL - ma) ** 2).mean()) / va; r[f'LRGC{k}_cov'] = float((CL - Ca).norm() / Ca.norm())
        r[f'DC{k}_mean'] = float(((muD - ma) ** 2).mean()) / va; r[f'DC{k}_cov'] = float((CD - Ca).norm() / Ca.norm())
    out[l] = r
    print(l, json.dumps({k: (float(f'{v:.4g}') if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
os.makedirs('runs/diag2', exist_ok=True)
json.dump(out, open(f'runs/diag2/diag2_{seed}_L{L}.json', 'w'), indent=1)
print(f'done {time.time() - t0:.0f}s', flush=True)
