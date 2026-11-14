"""Sampling baselines at equal FLOPs (PREREG: MC, RQMC, HYB; tails MC, RQMC, RB).
usage: python samplers.py SEED SIGMA TRUTHJSON OUTJSON [HYB=1] [MAXLOG2=20]
Budgets N_eq = 2^7, 2^9, ..., 2^17, 2^20 MC paths (one path = net.flops_path() FLOPs).
- MC: expected nMSE is exactly 1/N; simulated for N <= 2^13 as a check of the truth.
- RQMC: scrambled Sobol on the 8 input coordinates (inverse normal CDF), iid noise; R replicates.
- HYB(l0) (previous lane, plain): MC through layers 1..l0 (N = (B - deep cost)/shallow cost), 8 log-radius quantile
  bins, each bin's mean and covariance pushed through layers l0+1..L by K2-Stein with noise. l0 in {16, 32}.
- Tails (q = 1e-4 and 1e-3 thresholds from the truth): MC indicator, RQMC indicator, RB (directions u uniform, radius
  integrated exactly with chi_8, noise rescaled by 1/E|x|; a sampler, biased by the noise rescaling).
nMSE = |est - truth|^2 / tr Var minus the truth's own 1/P_truth; tail score = relative MSE.
"""
import sys, os, json, math, time
import numpy as np
import torch
from scipy.stats import chi
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
import nd, k2

torch.set_num_threads(1)
SEED, SIG, TJ, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
DO_HYB = int(sys.argv[5]) if len(sys.argv) > 5 else 1
MAXLOG2 = int(sys.argv[6]) if len(sys.argv) > 6 else 20
DT = nd.DT
T = json.load(open(TJ))
net = nd.Net('plain', SEED, T['n'], T['L'], SIG, r=T['r'])
n, L = net.n, net.L
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']; tse = 1.0 / T['paths']
fp = net.flops_path()
EX = nd.chi_mean(n)
LOGS = [l for l in (7, 9, 11, 13, 15, 17, 20) if l <= MAXLOG2]
REPS = {7: 32, 9: 32, 11: 32, 13: 32, 15: 16, 17: 8, 20: 2}
g = torch.Generator().manual_seed(61_000_003 + 101 * SEED + int(round(SIG * 1e6)))
CH = 1 << 16
t0 = time.time()


def fwd(x, l_from=0, l_to=None, scale_noise=1.0):
    l_to = l_to or L
    h = x
    for l in range(l_from, l_to):
        a = torch.relu(h @ net.W[l].T)
        h = a + torch.randn(a.shape, generator=g, dtype=DT) * (net.s[l] * scale_noise) if (l < L - 1 and SIG > 0) else a
    return h


def chunked(fn, N, make):
    """sum of fn over N inputs made in chunks; returns (sum vector, list of chunk outputs if needed)."""
    tot = None; done = 0
    while done < N:
        m = min(CH, N - done)
        o = fn(make(done, m))
        tot = o if tot is None else tot + o
        done += m
    return tot


tails = {q: T['tails'][q] for q in ('0.0001', '0.001')}
res = dict(seed=SEED, sigma=SIG, flops_path=fp, budgets=LOGS, est={}, tails={})
for lg in LOGS:
    N = 1 << lg; R = REPS[lg]
    rec = {k: [] for k in ('MC', 'RQMC')}
    trec = {q: {k: [] for k in ('MC', 'RQMC', 'RB')} for q in tails}
    for rep in range(R):
        # RQMC
        sob = torch.quasirandom.SobolEngine(n, scramble=True, seed=int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
        u = sob.draw(N, dtype=DT).clamp(1e-15, 1 - 1e-15)
        xq = torch.special.ndtri(u)
        s = torch.zeros(n, dtype=DT); cnt = {q: 0 for q in tails}
        for b in range(0, N, CH):
            o = fwd(xq[b:b + CH]); s += o.sum(0)
            nr = o.norm(dim=1)
            for q in tails: cnt[q] += int((nr > tails[q]['t']).sum())
        rec['RQMC'].append(((s / N - truth).pow(2).sum() / trv).item())
        for q in tails: trec[q]['RQMC'].append(cnt[q] / N)
        # MC (check, small budgets) and MC tails
        if lg <= 13:
            s = torch.zeros(n, dtype=DT); cnt = {q: 0 for q in tails}
            for b in range(0, N, CH):
                m = min(CH, N - b)
                o = fwd(torch.randn(m, n, generator=g, dtype=DT)); s += o.sum(0)
                nr = o.norm(dim=1)
                for q in tails: cnt[q] += int((nr > tails[q]['t']).sum())
            rec['MC'].append(((s / N - truth).pow(2).sum() / trv).item())
            for q in tails: trec[q]['MC'].append(cnt[q] / N)
        # RB tails: per direction, cost fp + 200 (chi survival); budgets <= 2^17 only
        NR = int(N * fp // (fp + 200)) if lg <= 17 else 0
        acc = {q: 0.0 for q in tails}
        for b in range(0, NR, CH):
            m = min(CH, NR - b)
            uu = torch.randn(m, n, generator=g, dtype=DT); uu = uu / uu.norm(dim=1, keepdim=True)
            G = fwd(uu, scale_noise=1.0 / EX).norm(dim=1).clamp_min(1e-300)
            for q in tails:
                acc[q] += float(chi.sf((tails[q]['t'] / G).numpy(), n).sum())
        if NR > 0:
            for q in tails: trec[q]['RB'].append(acc[q] / NR)
    res['est'][lg] = {k: dict(nmse=float(np.mean(v)) - tse, se=float(np.std(v) / math.sqrt(len(v))), R=len(v), raw=v)
                      for k, v in rec.items() if v}
    res['tails'][lg] = {q: {k: dict(rel_mse=float(np.mean([(e - tails[q]['p']) ** 2 for e in v]) / tails[q]['p'] ** 2),
                                    mean=float(np.mean(v)), R=len(v)) for k, v in trec[q].items() if v} for q in tails}
    print(f"s{SEED} sig{SIG:g} N=2^{lg}: " + ' '.join(f"{k}={v['nmse']:.3g}" for k, v in res['est'][lg].items()) +
          f" (1/N={1/N:.3g}); tails q=1e-4 " + ' '.join(f"{k}={v['rel_mse']:.3g}" for k, v in res['tails'][lg]['0.0001'].items()) +
          f" ({time.time()-t0:.0f}s)", flush=True)

# HYB (previous lane), plain, budgets 2^9..2^17
NB = 8


def deep_cost(l0):
    mu = torch.zeros(n, dtype=DT); C = torch.eye(n, dtype=DT)
    _, _, f = k2.plain_prop(net, mu, C, l0, l0 + 1, exact=False)
    return NB * (L - l0) * float(f)


def deep(h, l0):
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
        C = torch.cov(hk.T) if hk.shape[0] > 1 else torch.zeros(n, n, dtype=DT)
        mL, _, _ = k2.plain_prop(net, mu, C, l0, L, exact=False)
        est += w * mL
    return est


if DO_HYB:
    res['hyb'] = {}
    for lg in [l for l in (9, 11, 13, 15, 17) if l <= MAXLOG2]:
        B = (1 << lg) * fp; R = REPS[lg]
        out = {}
        for l0 in (16, 32):
            fs = nd.Net.flops_path(net, L=l0)
            Nh = int((B - deep_cost(l0)) // fs)
            if Nh < 4 * NB: continue
            errs = []
            for rep in range(R):
                h = fwd(torch.randn(Nh, n, generator=g, dtype=DT), 0, l0)
                e = deep(h, l0)
                errs.append(((e - truth).pow(2).sum() / trv).item())
            out[f'HYB{l0}'] = dict(nmse=float(np.mean(errs)) - tse, se=float(np.std(errs) / math.sqrt(R)), R=R, N_shallow=Nh, raw=errs)
        res['hyb'][lg] = out
        print(f"s{SEED} sig{SIG:g} HYB N=2^{lg}: " + ' '.join(f"{k}={v['nmse']:.3g}" for k, v in out.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
res['sec'] = time.time() - t0
json.dump(res, open(OUT, 'w'))
