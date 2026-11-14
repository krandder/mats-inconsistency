"""Full two-sided sample-free estimator TS(c, F) of E[a_L] (PREREG P5), on one net, for a ladder of configurations.
usage: python ts_run.py SEED SIGMA TRUTHJSON OUTJSON LADDER
LADDER: semicolon-separated configs "F,c,tau,k,G,X,c_d" with F in {GHM, K2X}; tau ignored for K2X.
Forward F to cut c -> mixture (w, mu, C) of the state h_c; downstream (Amendment 2): noiseless deep part J_ref from the
majority future class at c_d (D0); meet = sum_j w_j E_{N(mu_j,C_j)}[V_c] with V_c evaluated exactly (noiseless pass
c -> c_d, then D0) at component means, and at 2k points along the top-k probe directions for spread components. No random numbers are used. All FLOPs are counted; cost is quoted in MC paths (net.flops_path()).
"""
import sys, os, json, math, time
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, k2, ghm, bvf

torch.set_num_threads(1)
SEED, SIG, TJ, OUT, LAD = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
DT = nd.DT
T = json.load(open(TJ))
net = nd.Net('plain', SEED, T['n'], T['L'], SIG, r=T['r'])
n, L = net.n, net.L
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']
fp = net.flops_path()
t0 = time.time()
cfgs = []
for s in LAD.split(';'):
    F, c, tau, k, G, X, cd = s.split(',')
    cfgs.append(dict(F=F, c=int(c), tau=float(tau), k=int(k), G=int(G), X=float(X), c_d=int(cd)))
cache = {}


def forward(F, c, tau):
    key = (F, c, tau)
    if key in cache:
        return cache[key]
    if F == 'GHM':
        out, _ = ghm.ghm_forward(net, c, tau, keep_layers={c})
        w, mu, C, f = out[c]
    elif F == 'K2X':
        mu1 = torch.zeros(n, dtype=DT); C1 = torch.eye(n, dtype=DT); f = 0
        for l in range(c):
            m = net.W[l] @ mu1; Cz = net.W[l] @ C1 @ net.W[l].T
            f += k2.mmf(n, n, 1) + 2 * k2.mmf(n, n, n)
            mu1, C1, ff = k2.relu_mom(m, Cz, True); f += ff
            if SIG > 0:
                C1 = C1 + torch.eye(n, dtype=DT) * float(net.s[l]) ** 2; f += n
        w, mu, C = torch.ones(1, dtype=DT), mu1[None], C1[None]
    else:
        raise ValueError(F)
    cache[key] = (w, mu, C, f)
    return cache[key]


res = dict(seed=SEED, sigma=SIG, flops_path=fp, rows=[])
for cf in cfgs:
    t1 = time.time()
    w, mu, C, f_fwd = forward(cf['F'], cf['c'], cf['tau'])
    c, c_d, k = cf['c'], cf['c_d'], cf['k']
    fl = bvf.Flops()
    W = w / w.sum()
    m = (W[:, None] * mu).sum(0)
    M2 = (W[:, None, None] * (C + mu[:, :, None] * mu[:, None, :])).sum(0)
    Cov = M2 - torch.outer(m, m)
    fl.add(len(w) * 3 * n * n)
    top = torch.argsort(w, descending=True)[:64]
    up, uw = bvf.unscented(m, Cov)
    cand = bvf.noiseless_pass(net, torch.cat([mu[top], m[None, :], up]), c, c_d, fl)
    deep = bvf.Deep(net, c_d, cand, torch.cat([W[top], torch.full((1 + len(up),), 1.0 / len(w), dtype=DT)]), fl)
    probes = torch.cat([mu[top], up]); pw = torch.cat([W[top], uw.clamp_min(0) * 0 + 1.0 / (2 * n + 1) * 0.1])
    U8, eS = bvf.relevant_directions(net, c, c_d, deep, probes, pw, None, Cov, n, fl)
    U = U8[:, :k]
    est, npts = bvf.meet_direct(net, c, c_d, deep, w, mu, C, fl, U=U, spread_tol=0.05)
    out = 0.0
    tails_ts = {}
    for qk in ('0.0001', '0.001'):
        pq = T['tails'][qk]['p']
        ph = bvf.tail_from_meet(w, mu, C, T['tails'][qk]['t'], bvf.Flops())
        tails_ts[qk] = dict(p=ph, rel_err2=(ph - pq) ** 2 / pq ** 2)
    e = float((est - truth).pow(2).sum() / trv)
    cost = (f_fwd + fl.n) / fp
    row = dict(cf, nmse=e, cost_paths=cost, fwd_paths=f_fwd / fp, ds_paths=fl.n / fp, ncomp=int(len(w)), npts=int(npts), class_share=deep.class_share,
               eS=eS[:6].tolist(), est=est.tolist(), tails=tails_ts, sec=time.time() - t1)
    res['rows'].append(row)
    print(f"{cf}: nMSE {e:.3g}  cost {cost:.0f} paths (fwd {f_fwd/fp:.0f}, downstream {fl.n/fp:.0f}), {len(w)} comps, "
          f"{npts} meet points, ref class share {deep.class_share:.3f} ({time.time()-t1:.0f}s)", flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
