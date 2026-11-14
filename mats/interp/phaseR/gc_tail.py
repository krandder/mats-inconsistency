"""Task 2, neuron-code family: q_L^GC(A) and (K q_{L-1}^GC)(A) for the rare-event thresholds of rare_<net>.json.
usage: python gc_tail.py d act g sigma seed [nosearch] -> results/gctailfz_<net>.json   (pre-review: results/prereview/gctail_*)

GC chain fitted as in score.py (samples of K q_{l-1}, delta = 0.01 defensive N(mu, 4 Sigma)). A = {w.h > t}.
GC part: h = T(u), u ~ N(0, R), T_i = F_i^{-1} o Phi. Last-layer-exact variant: score w.f_L(T(u)) + sigma e.
Review fix 6 (2026-10-29): T uses the exact-tailed inverse (fam.NC.inv_cdf_u; the earlier table was bounded at m +/- 8 sd,
so sampling and density disagreed in the far tails); every probability by fixed-level splitting with levels and pCN scales
frozen from an independent pilot (rare_gs.frozen_estimates); the defensive part of (K q_3)(A) by the same latent splitting
instead of crude Monte Carlo; the defensive part of q_4(A) is an exact Gaussian tail. The coverage search for the [last]
bound is measured on this same chain (layers 1-3, d <= 32), not transferred from the scorecard's chain.
"""
import json, math, os, sys, time
import numpy as np
import torch
import rnet as R, fam as F, rare as RA, rare_gs as GS
from rare_run import kq_event_gs_frozen
from eps_lb import coverage_search

torch.set_num_threads(1)
DT = torch.float64
LOG2 = math.log(2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def run(d, act, g, sigma, seed, nfit=1 << 15, N=4096, Rr=16, search=True):
    net = R.RNet(d, 4, act, g, sigma, seed)
    rr = json.load(open(os.path.join(OUT, f'rare_{net.name()}.json')))
    gen = torch.Generator().manual_seed(300 + seed)
    q = F.Gauss(torch.zeros(d, dtype=DT), torch.eye(d, dtype=DT)); qs = []
    for l in range(1, 5):
        X = q.sample(nfit, gen); S = net.f(l, X); H = S + sigma * torch.randn(S.shape, generator=gen, dtype=DT)
        q = F.Mix(F.GC.fit_from_s(S, H, sigma), 0.01); qs.append(q)
    gen2 = torch.Generator().manual_seed(3300 + seed + 7 * d)
    ts = [lev['t'] for lev in rr['levels']]; w = net.w
    out = dict(d=d, act=act, g=g, sigma=sigma, seed=seed, N=N, R=Rr, levels=[])
    t0 = time.time()
    gL = qs[3].q
    fe_qL = GS.frozen_estimates(lambda Z: gL.nc.inv_cdf_u(Z @ gL.LR.T, gL.tab) @ w,
                                lambda n: torch.randn(n, d, generator=gen2, dtype=DT), ts, gen2, N, Rr, n_moves=10)['per_target']
    g3 = qs[2].q
    fe_k = GS.frozen_estimates(lambda Z: net.f(4, g3.nc.inv_cdf_u(Z[:, :d] @ g3.LR.T, g3.tab)) @ w + sigma * Z[:, d],
                               lambda n: torch.randn(n, d + 1, generator=gen2, dtype=DT), ts, gen2, N, Rr, n_moves=10)['per_target']
    D3 = qs[2].D
    fe_kd = kq_event_gs_frozen(net, D3.mu, torch.linalg.cholesky(D3.S), ts, gen2, N=N, R=Rr)
    out['secs_tails'] = time.time() - t0
    DL = qs[3].D; mL = float(w @ DL.mu); sL = math.sqrt(float(w @ DL.S @ w))
    for lev in rr['levels']:
        t = lev['t']; row = dict(k=lev['k'], t=t)
        for name, gcp, dfp, delta in (('qL', fe_qL[t], dict(logp=float(RA.log_sf(torch.tensor((t - mL) / sL))), rel_se=0.0), qs[3].delta),
                                      ('KqLm1', fe_k[t], fe_kd[t], qs[2].delta)):
            la, lb = math.log(1 - delta) + gcp['logp'], math.log(delta) + dfp['logp']
            tot = float(np.logaddexp(la, lb))
            se = math.sqrt((math.exp(la - tot) * gcp['rel_se']) ** 2 + (math.exp(lb - tot) * dfp['rel_se']) ** 2)
            row[name] = dict(logp=tot, rel_se=se, logp_gc=gcp['logp'], rel_se_gc=gcp['rel_se'], half_bits_gc=gcp['half_bits'],
                             logp_def=dfp['logp'], rel_se_def=dfp['rel_se'])
        out['levels'].append(row)
        print(row['k'], {n: (round(row[n]['logp'] / LOG2, 2), round(row[n]['rel_se'], 3)) for n in ('qL', 'KqLm1')}, flush=True)
    if search and d <= 32:
        q0 = F.Gauss(torch.zeros(d, dtype=DT), torch.eye(d, dtype=DT)); out['search'] = []
        for l in range(1, 4):
            t1 = time.time()
            c, m = coverage_search(net, l, q0 if l == 1 else qs[l - 2], qs[l - 1], gen2)
            out['search'].append(dict(layer=l, search_c=c, search_mean=m, secs=time.time() - t1))
            print('search layer', l, round(c, 2), round(m, 2), flush=True)
    json.dump(out, open(os.path.join(OUT, f'gctailfz_{net.name()}.json'), 'w'), indent=1)
    print('done', f"{time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    a = sys.argv
    run(int(a[1]), a[2], float(a[3]), float(a[4]), int(a[5]), search=not (len(a) > 6 and a[6] == 'nosearch'))
