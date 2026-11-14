"""Part 4 (and coordinator's headline request): estimators of E[output] on trained narrow nets at equal FLOPs.
usage: python p4_est.py KIND NPZ OUTJSON [R=32]     (KIND = rnn | mlp)
Truth: MC with 2^21 paths. K2 (exact ReLU moments and Stein-linearised) from the input law. Samplers at budgets
B = N_eq x (one MC path): MC, ANT (x, -x), RQMC (scrambled Sobol inputs, iid noise), HYB(l0)-st (MC for the first l0
layers/steps, 8 log-radius quantile bins, Stein-K2 per bin for the rest; for the RNN the remaining inputs enter K2
as the independent Gaussians they are).
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd, tnets, k2

torch.set_num_threads(1)
DT = torch.float64
KIND, NPZ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
R = int(sys.argv[4]) if len(sys.argv) > 4 else 16
net = tnets.RNNNet(NPZ) if KIND == 'rnn' else tnets.MLPNet(NPZ)
depth = net.T if KIND == 'rnn' else net.L
width = net.d if KIND == 'rnn' else net.n
nin = net.n
fp = net.flops_path()
NB = 8
g = torch.Generator().manual_seed(31_000_013)
t0 = time.time()

# truth
so = 0; sso = 0; NT = 0
for i in range(32):
    o = net.forward(torch.randn(1 << 16, nin, generator=g, dtype=DT), gen=g)
    so = so + o.sum(0); sso = sso + o.pow(2).sum(0); NT += 1 << 16
truth = so / NT; var = sso / NT - truth ** 2; trv = float(var.sum()); tse = 1.0 / NT


def readout_mean(mu):
    if KIND == 'rnn': return net.W_oh @ mu
    if net.fam == 'plain': return net.W_out @ mu
    return mu


def kprop(mu, C, lo, hi, exact):
    if KIND == 'rnn': return k2.rnn_prop(net, mu, C, lo, hi, exact)
    if net.fam == 'res': return k2.res_prop(net, mu, C, lo, hi, exact)
    return k2.plain_prop(net, mu, C, lo, hi, exact, noise_last=True)


res = dict(npz=NPZ, kind=KIND, depth=depth, width=width, trvar=trv, mean_sq=float(truth.pow(2).sum()), flops_path=fp, est={})
mu0 = torch.zeros(width if KIND == 'rnn' else nin, dtype=DT)
C0 = torch.zeros(width, width, dtype=DT) if KIND == 'rnn' else torch.eye(nin, dtype=DT)
for ex in (True, False):
    mu, C, fl = kprop(mu0, C0, 0, depth, ex)
    res['k2' + ('' if ex else '_st')] = dict(nmse=float(((readout_mean(mu) - truth) ** 2).sum() / trv), flops=fl, flops_in_paths=fl / fp)
print(f"{os.path.basename(NPZ)}: trVar={trv:.4g} K2 nMSE={res['k2']['nmse']:.3g} ({res['k2']['flops_in_paths']:.0f} paths' FLOPs) "
      f"K2-stein nMSE={res['k2_st']['nmse']:.3g}", flush=True)


def shallow(x, l0):
    """Run layers/steps 1..l0 with noise; return the state."""
    if KIND == 'rnn':
        h = torch.zeros(x.shape[0], width, dtype=DT)
        for t in range(l0):
            a = torch.relu(h @ net.W_hh.T + x[:, t:t + 1] * net.W_hi)
            h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[t] if net.sigma > 0 else a
        return h
    h = x
    for l in range(l0):
        a = torch.relu(h @ net.W[l].T)
        e = torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if net.sigma > 0 else 0.0
        h = h + (a + e) @ net.V[l].T if net.fam == 'res' else a + e
    return h


def shallow_flops(l0):
    if KIND == 'rnn':
        f = l0 * nd.F_NORMAL + l0 * (nd.mm_flops(1, width, width) + 3 * width)
        return f + (l0 * width * (nd.F_NORMAL + 2) if net.sigma > 0 else 0)
    per = nd.mm_flops(1, width, width) + width + (nd.mm_flops(1, width, width) + width if net.fam == 'res' else 0)
    return nin * nd.F_NORMAL + l0 * per + (l0 * width * (nd.F_NORMAL + 2) if net.sigma > 0 else 0)


def deep_cost(l0):
    _, _, f = kprop(torch.zeros(width, dtype=DT), torch.eye(width, dtype=DT), l0, l0 + 1, False)
    return NB * (depth - l0) * f


def deep(h, l0):
    lr = torch.log(h.norm(dim=1).clamp_min(1e-300))
    qs = torch.quantile(lr, torch.linspace(0, 1, NB + 1, dtype=DT)[1:-1])
    b = torch.bucketize(lr, qs); est = 0
    for k in range(NB):
        sel = b == k
        if sel.sum() == 0: continue
        hk = h[sel]; mu = hk.mean(0); C = torch.cov(hk.T) if hk.shape[0] > 1 else torch.zeros(width, width, dtype=DT)
        mL, _, _ = kprop(mu, C, l0, depth, False)
        est = est + (hk.shape[0] / h.shape[0]) * readout_mean(mL)
    return est


L0S = sorted(set([max(1, depth // 8), depth // 4, depth // 2]))
for neq in (256, 1024, 4096, 16384):
    B = neq * fp; errs = {}
    for rep in range(R):
        e = {}
        x = torch.randn(neq, nin, generator=g, dtype=DT); y = net.forward(x, gen=g)
        e['MC'] = y.mean(0)
        Bc = torch.linalg.lstsq(x - x.mean(0), y - y.mean(0)).solution.T
        e["MC-reg"] = y.mean(0) - Bc @ x.mean(0)
        xa = torch.randn(neq // 2, nin, generator=g, dtype=DT)
        e['ANT'] = 0.5 * (net.forward(xa, gen=g).mean(0) + net.forward(-xa, gen=g).mean(0))
        sob = torch.quasirandom.SobolEngine(nin, scramble=True, seed=int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
        u = sob.draw(neq, dtype=DT).clamp(1e-12, 1 - 1e-12)
        xq = torch.special.ndtri(u); yq = net.forward(xq, gen=g)
        e['RQMC'] = yq.mean(0)
        Bq = torch.linalg.lstsq(xq - xq.mean(0), yq - yq.mean(0)).solution.T
        e['RQMC-reg'] = yq.mean(0) - Bq @ xq.mean(0)
        for l0 in L0S:
            N = int((B - deep_cost(l0)) // shallow_flops(l0))
            if N < 4 * NB: continue
            e[f'HYB{l0}-st'] = deep(shallow(torch.randn(N, nin, generator=g, dtype=DT), l0), l0)
        for k_, v in e.items():
            errs.setdefault(k_, []).append(float(((v - truth) ** 2).sum() / trv))
    res['est'][neq] = {k_: dict(nmse=float(np.mean(v)) - tse, se=float(np.std(v) / math.sqrt(len(v))), R=len(v)) for k_, v in errs.items()}
    print(f"   N_eq={neq}: " + ' '.join(f"{k_}={v['nmse']:.3g}" for k_, v in res['est'][neq].items()) + f" (1/N={1/neq:.3g}; {time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(OUT, 'w'))
