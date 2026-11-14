"""Review fix 1: per-layer coverage numbers for the SAME fitted Gaussian chain whose (K q_3)(A) the rare-event bounds use.
usage: python eps_lb.py results/rare_<net>.json [search]  -> results/epslb_<net>.json

The chain is re-created from its seed (rare_run.g_chain, first use of Generator(100 + seed)) and checked against the stored
q_4(A). Per layer l = 1..4: q* = copula code (+ defensive part) fitted to fresh samples of K q_{l-1}; T = log q*/q_l at fresh
iid samples of K q_{l-1}; rel_stats(T): bootstrap 95% lower bound on KL(Kq||q_l) and held-out Clopper-Pearson 95% lower
bound on D_inf(Kq||q_l) (score.rel_stats). With 'search' (d <= 32): the SMC coverage search (largest log Kq/q over 96
Kq points, top 8 re-evaluated with 4x particles; a lower estimate of c_l, uncertified). Envelope: rigorous upper bound on c_l.
"""
import json, math, os, sys, time
import torch
import rnet as R, fam as F, kq as KQ
from score import rel_stats, logpdf_chunked, GC_M, envelope_c, DELTA
from rare_run import g_chain

torch.set_num_threads(1)
DT = torch.float64
LOG2 = math.log(2)


def coverage_search(net, l, qprev, q, gen, Np=512, npts=96):
    """Largest log Kq/q (bits) over npts Kq points (top 8 re-evaluated with 4 Np particles), and the mean over the points."""
    X = qprev.sample(npts, gen); Hk = net.f(l, X) + net.sigma * torch.randn(npts, net.d, generator=gen, dtype=DT)
    lk, _ = KQ.log_kq_smc(net, l, qprev, Hk, gen, Np=Np, ess_frac=0.7, n_moves=6)
    lqk = q.logpdf(Hk); r = lk - lqk
    top = torch.argsort(r, descending=True)[:8]
    lk2, _ = KQ.log_kq_smc(net, l, qprev, Hk[top], gen, Np=4 * Np, ess_frac=0.8, n_moves=8)
    return float((lk2 - lqk[top]).max()) / LOG2, float(r.mean()) / LOG2


def layer_stats(net, l, qprev, q, gen, nfit=1 << 15, n_tot=None, seed=0):
    d, sigma = net.d, net.sigma
    n_tot = n_tot or ((1 << 14) if d >= 128 else (1 << 15))
    Xs = qprev.sample(nfit, gen); Ss = net.f(l, Xs); Hs = Ss + sigma * torch.randn(Ss.shape, generator=gen, dtype=DT)
    qstar = F.Mix(F.GC.fit_from_s(Ss, Hs, sigma, M=GC_M(d)), DELTA)
    Xe = qprev.sample(n_tot, gen); He = net.f(l, Xe) + sigma * torch.randn(n_tot, d, generator=gen, dtype=DT)
    T = logpdf_chunked(qstar, He) - logpdf_chunked(q, He)
    return rel_stats(T, seed=seed)


def main(p, search=False):
    r = json.load(open(p)); d = r['d']; seed = r['seed']; noln = '_noLN' in p
    net = R.RNet(d, r['L'], r['act'], r['g'], r['sigma'], seed, use_ln=not noln)
    qs, _ = g_chain(net, torch.Generator().manual_seed(100 + seed))
    chk = max(abs(qs[-1].prob_halfspace(net.w, lev['t']) - lev['chain']['log_qL']) for lev in r['levels'])
    print('chain re-created; max |log q_4(A) - stored| =', chk, flush=True)
    assert chk < 1e-8, 'chain differs from the rare run'
    gen = torch.Generator().manual_seed(9200 + seed + 7 * d + (1 if noln else 0))   # 9100: first d=256 run, discarded (rule change)
    q0 = F.Gauss(torch.zeros(d, dtype=DT), torch.eye(d, dtype=DT))
    out = dict(src=os.path.basename(p), d=d, act=r['act'], g=r['g'], sigma=r['sigma'], seed=seed, chain_check=chk, layers=[])
    for l in range(1, r['L'] + 1):
        t0 = time.time()
        qprev = q0 if l == 1 else qs[l - 2]; q = qs[l - 1]
        row = dict(layer=l, **layer_stats(net, l, qprev, q, gen, seed=l))
        A = q.infl * q.S; ev = torch.linalg.eigvalsh(A); s2 = net.sigma ** 2
        row['env_c'] = (float('inf') if float(ev.min()) <= s2 else
                        (math.log(1 / q.delta) + 0.5 * float(torch.log(ev / s2).sum())
                         + 0.5 * (math.sqrt(d) + float(q.mu.norm())) ** 2 / float(ev.min() - s2)) / LOG2)
        if search and l <= 3:
            row['search_c'], row['search_mean'] = coverage_search(net, l, qprev, q, gen)
        row['secs'] = time.time() - t0
        out['layers'].append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        json.dump(out, open(os.path.join(os.path.dirname(p), os.path.basename(p).replace('rare_', 'epslb_')), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == 'search')
