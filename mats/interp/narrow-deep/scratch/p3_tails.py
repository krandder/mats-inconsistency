"""Part 3 tails: p_t = P(|out| > t), t = the truth run's (1 - q) quantile of |out|, q in {1e-2, 1e-3, 1e-4}.
usage: python p3_tails.py TRUTHPREFIX [R=32]
Estimators at equal FLOPs (budget = N_eq full MC paths):
- MC: indicator over N_eq paths.
- RB: radial Rao-Blackwellisation using positive homogeneity (exact for sigma = 0, approximate with noise):
      draw directions u, run the net on u (unit norm; the noise is scaled by 1/E|x| so its relative size matches),
      average Q_chi_n(t / (|out(u)| / E|x| * E|x|))... i.e. P(|x| |out(u)| > t) with |x| ~ chi_n integrated exactly.
- HYB-lin (plain only): MC for layers 1..l0, then the fixed matrix J* (product of the reference trajectory's masked
      layers l0+1..L, computed once) applied per sample: indicator |J* h_l0| > t. Also HYB-RB: the same on directions
      with the radius integrated exactly.
Relative MSE = E (p_hat - p)^2 / p^2 over R replicates (truth's own relative SE reported).
"""
import sys, os, json, math, time
import numpy as np
import torch
from scipy.stats import chi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd

torch.set_num_threads(1)
DT = nd.DT
pref = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 16
S = json.load(open(pref + '.json'))
net = nd.Net(S['fam'], S['seed'], S['n'], S['L'], S['sigma'], r=S['r'] if S['sigma'] > 0 else None)
n, L = net.n, net.L
fp = net.flops_path()
EX = nd.chi_mean(n)
g = torch.Generator().manual_seed(15_000_017 + S['seed'])
t0 = time.time()
F_CHI = 200          # survival function of chi_n per sample (incomplete gamma), counted generously


def qchi(z):
    return torch.tensor(chi.sf(z.numpy(), n), dtype=DT)


def run_unit(u, l0=None):
    """Forward unit-norm directions with the noise scaled by 1/E|x| (so noise/signal matches a typical radius)."""
    sv = net.s.clone(); net.s = net.s / EX
    try:
        if l0 is None:
            return net.forward(u, gen=g)
        h = u
        for l in range(l0):
            a = torch.relu(h @ net.W[l].T)
            e = torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if net.sigma > 0 else 0.0
            h = a + e if net.fam == 'plain' else h + ((a + e) @ net.V[l].T) * net.c
        return h
    finally:
        net.s = sv


def shallow(x, l0):
    h = x
    for l in range(l0):
        a = torch.relu(h @ net.W[l].T)
        e = torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if net.sigma > 0 else 0.0
        h = a + e if net.fam == 'plain' else h + ((a + e) @ net.V[l].T) * net.c
    return h


# reference trajectory and J* for the plain family (one input pushed noiselessly; gates of that path)
Jstar = {}
if net.fam == 'plain':
    for l0 in (32,):
        if l0 >= L: continue
        x0 = torch.randn(1, n, generator=g, dtype=DT)
        h = x0
        for l in range(l0): h = torch.relu(h @ net.W[l].T)
        J = torch.eye(n, dtype=DT)
        for l in range(l0, L):
            z = (h @ net.W[l].T)[0]
            D = (z > 0).to(DT)
            J = D[:, None] * (net.W[l] @ J); h = torch.relu(h @ net.W[l].T)
        Jstar[l0] = J
res = dict(flops_path=fp, q={})


def imhof_sf(x, lam, delta2):
    """P(sum_j lam_j (z_j + delta_j)^2 > x), z iid N(0, 1) (Imhof 1961), by numerical integration."""
    from scipy.integrate import quad
    lam = np.asarray(lam, float); d2 = np.asarray(delta2, float)

    def f(u):
        if u == 0:
            return 0.5 * (np.sum(lam * (1 + d2)) - x)
        lu = lam * u
        th = 0.5 * np.sum(np.arctan(lu) + d2 * lu / (1 + lu * lu)) - 0.5 * x * u
        rho = np.exp(0.25 * np.sum(np.log1p(lu * lu)) + 0.5 * np.sum(d2 * lu * lu / (1 + lu * lu)))
        return np.sin(th) / (u * rho)
    v, _ = quad(f, 0, np.inf, limit=2000, epsabs=1e-13, epsrel=1e-10)
    return 0.5 + v / np.pi


# deductive tail for the residual family: Gaussian closure (K2, exact ReLU moments) at the output, then the exact
# law of |h|^2 under that Gaussian (generalised chi-square, Imhof)
K2T = None
if net.fam == 'res':
    import k2 as K2M
    mu, C, flk = K2M.res_prop(net, torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT), 0, L, exact=True)
    ev, Q = torch.linalg.eigh(0.5 * (C + C.T)); ev = ev.clamp_min(1e-300)
    dl = (Q.T @ mu) / ev.sqrt()
    K2T = dict(lam=ev.numpy(), d2=(dl ** 2).numpy(), flops=flk)
for qk, tv in S['tails'].items():
    if qk == '0.01': continue
    t = tv['t']; p = tv['p']
    if p <= 0: continue
    out = {}
    for neq in (1024, 4096, 16384):
        B = neq * fp; errs = {}
        for rep in range(R):
            e = {}
            o = net.forward(torch.randn(neq, n, generator=g, dtype=DT), gen=g)
            e['MC'] = float((o.norm(dim=1) > t).double().mean())
            N = int(B // (fp + F_CHI))
            u = torch.randn(N, n, generator=g, dtype=DT); u = u / u.norm(dim=1, keepdim=True)
            G = run_unit(u).norm(dim=1)
            e['RB'] = float(qchi(t / G.clamp_min(1e-300)).mean())
            for l0, J in Jstar.items():
                fs = nd.Net.flops_path(net, L=l0) + nd.mm_flops(1, n, n) + n
                N = int(B // fs)
                h = shallow(torch.randn(N, n, generator=g, dtype=DT), l0)
                e[f'HYB{l0}-lin'] = float(((h @ J.T).norm(dim=1) > t).double().mean())
                N = int(B // (fs + F_CHI))
                u = torch.randn(N, n, generator=g, dtype=DT); u = u / u.norm(dim=1, keepdim=True)
                G = (run_unit(u, l0) @ J.T).norm(dim=1)
                e[f'HYB{l0}-RB'] = float(qchi(t / G.clamp_min(1e-300)).mean())
            for k, v in e.items():
                errs.setdefault(k, []).append((v - p) ** 2 / p ** 2)
        out[neq] = {k: dict(rel_mse=float(np.mean(v)), se=float(np.std(v) / math.sqrt(len(v)))) for k, v in errs.items()}
        print(f"{os.path.basename(pref)} q={qk} N_eq={neq}: " + ' '.join(f"{k}={v['rel_mse']:.3g}" for k, v in out[neq].items()) +
              f" (truth rel SE {tv['se']/p:.3g}; {time.time()-t0:.0f}s)", flush=True)
    k2t = None
    if K2T is not None:
        pk = float(imhof_sf(t * t, K2T['lam'], K2T['d2']))
        zz = torch.randn(1 << 22, n, generator=g, dtype=DT)            # validation of the Imhof integral only
        pk_mc = float(((zz + torch.tensor(K2T['d2']).sqrt()) ** 2 @ torch.tensor(K2T['lam']) > t * t).double().mean())
        print(f"   Imhof check: {pk:.4g} vs sampled K2 Gaussian {pk_mc:.4g}", flush=True)
        nk = K2T['flops'] / fp
        k2t = dict(p=pk, rel_err2=(pk - p) ** 2 / p ** 2, flops_in_paths=nk, mc_rel_mse_same_flops=(1 - p) / (nk * p))
        print(f"   K2 tail (Imhof): p={pk:.4g} vs truth {p:.4g}: rel err^2 {k2t['rel_err2']:.3g}; MC at the same FLOPs "
              f"({nk:.0f} paths): {k2t['mc_rel_mse_same_flops']:.3g}", flush=True)
    res['q'][qk] = dict(t=t, p=p, se=tv['se'], est=out, k2=k2t)
json.dump(res, open(pref + '_tails.json', 'w'))
