"""Part 3: estimators of E[output] at equal FLOPs (PREREG Part 3), replicated.
(-reg: linear regression control variate on the input, E x = 0; its cost, ~n^2 per sample, is ignored.)
usage: python p3_samplers.py TRUTHPREFIX [R=32]
Budgets B = N_eq x (FLOPs of one full MC path), N_eq in {64, 256, 1024, 4096, 16384}.
- MC: N_eq paths. ANT: N_eq/2 antithetic pairs (x, -x), independent noise. RQMC: scrambled Sobol inputs (N_eq a power
  of 2), iid noise.
- HYB(l0)-{lin,st}: N paths through layers 1..l0 (N = (B - deep cost) / shallow cost), 8 log-radius quantile bins of
  the states at l0, each bin's (weight, mean, covariance) pushed through layers l0+1..L by: lin = the noiseless map
  applied to the bin mean; st = Stein-linearised K2 with the noise (exact ReLU means and variances).
nMSE = mean over replicates of |est - truth|^2 / tr Var(out), minus the truth's own error 1/paths_truth.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd, k2

torch.set_num_threads(1)
DT = nd.DT
pref = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 32
S = json.load(open(pref + '.json'))
net = nd.Net(S['fam'], S['seed'], S['n'], S['L'], S['sigma'], r=S['r'] if S['sigma'] > 0 else None)
truth = torch.tensor(S['mean'], dtype=DT); trv = S['trvar']; n = net.n; L = net.L
fp = net.flops_path(); tse = 1.0 / S['paths']
NEQ = (256, 1024, 4096, 16384)
NB = 8
L0S = [16, 32] if L >= 128 else [8, 16]
t0 = time.time()


def run_shallow(x, g, l0):
    """Forward x through layers 1..l0 (noise after each), return h_l0."""
    h = x
    for l in range(l0):
        a = torch.relu(h @ net.W[l].T)
        if net.fam == 'plain':
            h = a + (torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if net.sigma > 0 else 0)
        else:
            b = a + (torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if net.sigma > 0 else 0)
            h = h + (b @ net.V[l].T) * net.c
    return h


def deep_cost(l0, mode):
    if mode == 'lin':
        return NB * (L - l0) * (k2.mmf(n, n, 1) + n)
    mu = torch.zeros(n, dtype=DT); C = torch.eye(n, dtype=DT)
    _, _, f = k2.prop(net, mu, C, l0, l0 + 1, exact=False)
    return NB * (L - l0) * f


def deep(h, l0, mode):
    r = h.norm(dim=1)
    lr = torch.log(r.clamp_min(1e-300))
    qs = torch.quantile(lr, torch.linspace(0, 1, NB + 1, dtype=DT)[1:-1])
    b = torch.bucketize(lr, qs)
    est = torch.zeros(n, dtype=DT); P = h.shape[0]
    for k in range(NB):
        sel = b == k
        if sel.sum() == 0: continue
        hk = h[sel]; w = hk.shape[0] / P
        mu = hk.mean(0)
        if mode == 'lin':
            a = mu
            if net.fam == 'plain':
                for l in range(l0, L): a = torch.relu(net.W[l] @ a)
            else:
                for l in range(l0, L): a = a + net.c * (net.V[l] @ torch.relu(net.W[l] @ a))
            est += w * a
        else:
            C = torch.cov(hk.T) if hk.shape[0] > 1 else torch.zeros(n, n, dtype=DT)
            mL, _, _ = k2.prop(net, mu, C, l0, L, exact=False)
            est += w * mL
    return est


def ndraw(N, g):
    return torch.randn(N, n, generator=g, dtype=DT)


res = dict(flops_path=fp, NEQ=NEQ, est={})
g = torch.Generator().manual_seed(12_000_017 + S['seed'])
for neq in NEQ:
    B = neq * fp
    errs = {}
    for rep in range(R):
        e = {}
        x = ndraw(neq, g)
        y = net.forward(x, gen=g)
        e['MC'] = y.mean(0)
        # regression control variate on the input (E x = 0 known): y_bar - B (x_bar - 0), B = Cov(y, x) Var(x)^-1
        xc = x - x.mean(0); yc = y - y.mean(0)
        Bc = torch.linalg.lstsq(xc, yc).solution.T
        e["MC-reg"] = y.mean(0) - Bc @ x.mean(0)
        xa = ndraw(neq // 2, g)
        e['ANT'] = 0.5 * (net.forward(xa, gen=g).mean(0) + net.forward(-xa, gen=g).mean(0))
        sob = torch.quasirandom.SobolEngine(n, scramble=True, seed=int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
        u = sob.draw(neq, dtype=DT).clamp(1e-12, 1 - 1e-12)
        xq = torch.special.ndtri(u); yq = net.forward(xq, gen=g)
        e['RQMC'] = yq.mean(0)
        Bq = torch.linalg.lstsq(xq - xq.mean(0), yq - yq.mean(0)).solution.T
        e['RQMC-reg'] = yq.mean(0) - Bq @ xq.mean(0)
        for l0 in L0S:
            fs = nd.Net.flops_path(net, L=l0)
            for mode in ('lin', 'st'):
                N = int((B - deep_cost(l0, mode)) // fs)
                if N < 4 * NB: continue
                h = run_shallow(ndraw(N, g), g, l0)
                e[f'HYB{l0}-{mode}'] = deep(h, l0, mode)
        for k, v in e.items():
            errs.setdefault(k, []).append(float(((v - truth) ** 2).sum() / trv))
    res['est'][neq] = {k: dict(nmse=float(np.mean(v)) - tse, se=float(np.std(v) / math.sqrt(len(v))), R=len(v)) for k, v in errs.items()}
    print(f"{os.path.basename(pref)} N_eq={neq}: " + ' '.join(f"{k}={v['nmse']:.3g}" for k, v in res['est'][neq].items()) +
          f"  (1/N={1/neq:.3g}; {time.time()-t0:.0f}s)", flush=True)
res['deep_cost'] = {f'{l0}-{m}': deep_cost(l0, m) for l0 in L0S for m in ('lin', 'st')}
json.dump(res, open(pref + '_samplers.json', 'w'))
