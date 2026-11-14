"""Review fixes 1 and 4: width and depth scorecard with lower confidence bounds and extra weight seeds.
usage: python score_cb.py d L act g sigma seed  -> results/scorecb_<net>.json

Gaussian chain (moment-matched + defensive N(mu, 4 Sigma), delta = 0.01, fitted to 2^15 samples of K q_{l-1}, as score.py)
over L layers; per layer rel_stats of T = log q*/q (q* = copula code fitted to fresh samples of the same K q_{l-1}): bootstrap
95% lower bound on KL(Kq||q) and held-out Clopper-Pearson 95% lower bound on D_inf(Kq||q); envelope upper bound on c_l.
Layer 1 (where K q_0 is the true p_1 for every family) also scores the product code (NC) and know-nothing (KN) on the same
samples and q*.
"""
import json, math, os, sys, time
import torch
import rnet as R, fam as F
from score import rel_stats, logpdf_chunked, GC_M, envelope_c, DELTA

torch.set_num_threads(1)
DT = torch.float64
LOG2 = math.log(2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def run(d, L, act, g, sigma, seed, nfit=1 << 15):
    net = R.RNet(d, L, act, g, sigma, seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    n_tot = (1 << 14) if d >= 128 else (1 << 15)
    path = os.path.join(OUT, f'scorecb_{net.name()}.json')
    res = dict(d=d, L=L, act=act, g=g, sigma=sigma, seed=seed, nfit=nfit, n_eval=n_tot, delta=DELTA, layers=[])
    q = F.Gauss(torch.zeros(d, dtype=DT), torch.eye(d, dtype=DT))
    for l in range(1, L + 1):
        t0 = time.time()
        X = q.sample(nfit + n_tot, gen); S = net.f(l, X); H = S + sigma * torch.randn(S.shape, generator=gen, dtype=DT)
        qG = F.Mix(F.Gauss.fit(H[:nfit]), DELTA)
        Xs = q.sample(nfit, gen); Ss = net.f(l, Xs); Hs = Ss + sigma * torch.randn(Ss.shape, generator=gen, dtype=DT)
        qstar = F.Mix(F.GC.fit_from_s(Ss, Hs, sigma, M=GC_M(d)), DELTA)
        He = H[nfit:]; ls = logpdf_chunked(qstar, He)
        row = dict(layer=l, G=rel_stats(ls - logpdf_chunked(qG, He), seed=l), env_c=envelope_c(qG, d, sigma))
        if l == 1:
            qNC = F.Mix(F.NC.fit_from_s(S[:nfit], sigma, M=GC_M(d)), DELTA)
            row['NC'] = rel_stats(ls - logpdf_chunked(qNC, He), seed=100 + l)
            row['KN'] = rel_stats(ls - logpdf_chunked(F.KN(d, sigma), He), seed=200 + l)
        row['secs'] = time.time() - t0
        res['layers'].append(row)
        print(l, {f: (round(row[f]['KL_lcb'], 3), round(row[f]['Dinf_lcb'], 2)) for f in ('G', 'NC', 'KN') if f in row},
              f"env {row['env_c']:.0f} [{row['secs']:.0f}s]", flush=True)
        json.dump(res, open(path, 'w'), indent=1)
        q = qG
    return res


if __name__ == '__main__':
    a = sys.argv
    run(int(a[1]), int(a[2]), a[3], float(a[4]), float(a[5]), int(a[6]))
