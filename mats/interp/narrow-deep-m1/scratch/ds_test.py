"""Downstream test given the exact law at c (PREREG P3): the two-sided downstream (grid of V_c on k relevant directions,
radius exact, deep part from c_d) integrated against the true law at c (oracle samples; only the meet uses them).
usage: python ds_test.py SEED SIGMA TRUTHJSON OUTJSON [CUTS=4,8] [CDS=24,32]
U, B from (i) 'probe': deterministic probes = unscented sigma points of N(mean_h, cov_h) at c (the true layer moments;
a forward description would supply these); (ii) 'oracle': sampled Jacobians (2^14 true paths).
Reports nMSE of the meet against the truth, the downstream FLOPs (grid + deep + probes), and the fraction of mass
outside the grid box.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf

torch.set_num_threads(1)
SEED, SIG, TJ, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
CUTS = [int(c) for c in sys.argv[5].split(',')] if len(sys.argv) > 5 else [4, 8]
CDS = [int(c) for c in sys.argv[6].split(',')] if len(sys.argv) > 6 else [24, 32]
DT = nd.DT
T = json.load(open(TJ))
net = nd.Net('plain', SEED, 8, T['L'], SIG, r=T['r'])
n, L = net.n, net.L
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']
fp = net.flops_path()
g = torch.Generator().manual_seed(71_000_007 + SEED)
PM = 1 << 20
t0 = time.time()


def true_states(c, P):
    h = torch.randn(P, n, generator=g, dtype=DT)
    for l in range(c):
        a = torch.relu(h @ net.W[l].T)
        h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if SIG > 0 else a
    return h


res = dict(seed=SEED, sigma=SIG, rows=[])
for c in CUTS:
    Hs = true_states(c, PM)
    mh = torch.tensor(T['layers'][str(c)]['mean_h'], dtype=DT) if str(c) in T['layers'] else Hs.mean(0)
    Ch = torch.tensor(T['layers'][str(c)]['cov_h'], dtype=DT) if str(c) in T['layers'] else torch.cov(Hs.T)
    M2 = Ch + torch.outer(mh, mh)                                       # second moment
    for c_d in CDS:
        fl0 = bvf.Flops()
        pr, pwt = bvf.unscented(mh, Ch)
        cand = bvf.noiseless_pass(net, torch.cat([mh[None, :], pr]), c, c_d, fl0)
        deep = bvf.Deep(net, c_d, cand, torch.cat([torch.ones(1, dtype=DT), pwt.abs()]), fl0)
        f_deep = fl0.n
        # deep-only check: exact law at c_d (sampled) through the deep value function
        Hd_true = Hs.clone()
        for l in range(c, c_d):
            a = torch.relu(Hd_true @ net.W[l].T)
            Hd_true = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if SIG > 0 else a
        Vd = deep.value(Hd_true, bvf.Flops()).mean(0)
        e_deep = float((Vd - truth).pow(2).sum() / trv)
        for usrc in ('probe', 'oracle'):
            fl1 = bvf.Flops()
            if usrc == 'probe':
                probes, pw = bvf.unscented(mh, Ch)
                pw = pw.clamp_min(0) + 1e-3
                U8, eS = bvf.relevant_directions(net, c, c_d, deep, probes, pw, None, Ch, n, fl1)
            else:
                # oracle sensitivity from 2^14 true paths (not part of a sample-free method)
                Ms = torch.zeros(n, n, dtype=DT)
                for h in Hs[:1 << 12]:
                    _, Jm = bvf.jac_pass(net, h, c, c_d, bvf.Flops())
                    Jt = deep.J @ Jm; Ms += Jt.T @ Jt
                Ms = Ms / Ms.trace()
                ev, Uc = torch.linalg.eigh(Ch); Chh = Uc @ torch.diag(ev.clamp_min(0).sqrt()) @ Uc.T
                S = Chh @ Ms @ Chh; eS, VS = torch.linalg.eigh(S); eS, VS = eS.flip(0), VS.flip(1)
                U8 = torch.linalg.qr(Chh @ VS)[0]
            f_dirs = fl1.n
            for k in (2, 3, 4, 5):
                U = U8[:, :k]
                Pp = torch.eye(n, dtype=DT) - U @ U.T
                # dropped part as a homogeneous linear function of t: B = E[h_perp t^T] E[t t^T]^-1
                B = Pp @ M2 @ U @ torch.linalg.inv(U.T @ M2 @ U)
                e0 = U.T @ mh
                for G in ((6, 10, 16) if k <= 3 else (6, 10)):
                    for X in (1.0, 2.0):
                        fl = bvf.Flops()
                        grid = bvf.Grid(net, c, c_d, deep, U, B, e0, G, X, fl)
                        f_grid = fl.n
                        fq = bvf.Flops()
                        V, out = grid.value(Hs @ U, fq)
                        est = V.mean(0)
                        e = float((est - truth).pow(2).sum() / trv)
                        cost = (f_deep + f_dirs + f_grid) / fp
                        row = dict(c=c, c_d=c_d, usrc=usrc, k=k, G=G, X=X, nmse=e, e_deep=e_deep, out=out,
                                   cost_paths=cost, nodes=grid.nnodes, eS=eS[:6].tolist())
                        res['rows'].append(row)
                        print(f"[ref class share {deep.class_share:.3f} of {deep.nclass}] c={c} cd={c_d} {usrc} k={k} G={G} X={X}: nMSE {e:.3g} (deep-only {e_deep:.2g}), out {out:.3f}, "
                              f"cost {cost:.0f} paths ({grid.nnodes} nodes) ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
