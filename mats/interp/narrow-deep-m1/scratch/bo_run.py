"""Backward-only arm BO (PREREG Amendment 1) on one net: a ladder of (k, m) configurations.
usage: python bo_run.py SEED SIGMA TRUTHJSON OUTJSON [C_D=32] [KS=2,3,4,5,6] [MS=4,8,16]
Deterministic probes at the input (charged): the 16 signed coordinate vectors and the 112 signed pairs (e_i +- e_j)/sqrt2.
Deep part from the probes pushed to c_d (majority future class). Input directions: top eigenvectors of the probe
sensitivity (per-probe normalised, 'norm', or raw, 'raw'). Each (k, m): BO estimate, nMSE vs truth, FLOPs in paths.
Oracle reference (not part of the estimator): the same with directions from sampled Jacobians of 2^12 true inputs.
"""
import sys, os, json, math, time, itertools
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf, bo
from scipy.stats import chi as _chi

torch.set_num_threads(1)
SEED, SIG, TJ, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
CD = int(sys.argv[5]) if len(sys.argv) > 5 else 64
KS = [int(x) for x in sys.argv[6].split(',')] if len(sys.argv) > 6 else [2, 3, 4, 5, 6]
MS = [int(x) for x in sys.argv[7].split(',')] if len(sys.argv) > 7 else [4, 8, 16]
DT = nd.DT
T = json.load(open(TJ))
net = nd.Net('plain', SEED, T['n'], T['L'], SIG, r=T['r'])
n, L = net.n, net.L
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']
fp = net.flops_path()
t0 = time.time()
E = torch.eye(n, dtype=DT)
probes = [E[i] * s for i in range(n) for s in (1, -1)]
probes += [(E[i] + s * E[j]) / math.sqrt(2) * t for i, j in itertools.combinations(range(n), 2) for s in (1, -1) for t in (1, -1)]
probes = torch.stack(probes)
fl0 = bvf.Flops()
cand = bvf.noiseless_pass(net, probes, 0, CD, fl0)
deep = bvf.Deep(net, CD, cand, torch.ones(len(cand), dtype=DT), fl0)
f_deep = fl0.n
res = dict(seed=SEED, sigma=SIG, c_d=CD, deep_class_share=deep.class_share, deep_nclass=deep.nclass, rows=[])
dirs = {}
f_dir_full = {}
for mode in ('norm', 'raw'):
    fl = bvf.Flops()
    M = torch.zeros(n, n, dtype=DT)
    for h in probes:
        _, Jm = bvf.jac_pass(net, h, 0, CD, fl)
        Jt = deep.J @ Jm
        gm = Jt.T @ Jt
        M += gm / gm.trace().clamp_min(1e-300) if mode == 'norm' else gm
        fl.add(bvf.mmf(n, n, n) * 2 + n * n)
    ev, V = torch.linalg.eigh(0.5 * (M + M.T))
    dirs[mode] = (V.flip(1), ev.flip(0) / ev.sum(), fl.n)
    f_dir_full[mode] = fl.n + f_deep
# oracle directions (not charged; reference only)
g = torch.Generator().manual_seed(91_000_003 + SEED)
xs = torch.randn(1 << 12, n, generator=g, dtype=DT)
Mo = torch.zeros(n, n, dtype=DT)
for h in xs:
    _, Jm = bvf.jac_pass(net, h, 0, CD, bvf.Flops())
    Jt = deep.J @ Jm; Mo += Jt.T @ Jt
evo, Vo = torch.linalg.eigh(0.5 * (Mo + Mo.T))
dirs['oracle'] = (Vo.flip(1), evo.flip(0) / evo.sum(), 0.0)
f_dir_full['oracle'] = 0.0
for mode, (V, ev, f_dir) in dirs.items():
    print(f"{mode} directions: eig share {[round(float(e), 4) for e in ev]}", flush=True)
    for k in KS:
        for m in MS:
            if bo.sphere_rule(k, m)[0].shape[0] > 300_000:
                continue
            # BO with D0 at c_d
            fl = bvf.Flops()
            est, nn_ = bo.bo_estimate(net, CD, deep, V[:, :k], m, fl)
            e = float((est - truth).pow(2).sum() / trv)
            cost = (f_deep + f_dir + fl.n) / fp
            # BO-full: nodes passed noiselessly to the output (exact noiseless V_0 at each node)
            fl2 = bvf.Flops()
            nodes, wq = bo.sphere_rule(k, m)
            X = nodes @ V[:, :k].T
            fl2.add(len(nodes) * bvf.mmf(1, k, n))
            out_full = bvf.noiseless_pass(net, X, 0, L, fl2)
            est2 = bo.chi_mean(k) * (wq[:, None] * out_full).sum(0)
            e2 = float((est2 - truth).pow(2).sum() / trv)
            cost2 = (f_dir_full[mode] + fl2.n) / fp
            tails_bo = {}
            for qk in ('0.0001', '0.001'):
                tq = T['tails'][qk]['t']; pq = T['tails'][qk]['p']
                Gn = out_full.norm(dim=1).clamp_min(1e-300)
                ph = float((wq * torch.tensor(_chi.sf((tq / Gn).numpy(), k), dtype=DT)).sum())
                tails_bo[qk] = dict(p=ph, rel_err2=(ph - pq) ** 2 / pq ** 2)
            res['rows'].append(dict(mode=mode, k=k, m=m, nodes=nn_, nmse=e, cost_paths=cost, nmse_full=e2, cost_full=cost2, eig=ev.tolist(), est=est.tolist(), est_full=est2.tolist(), tails=tails_bo))
            print(f"  {mode} k={k} m={m}: nodes {nn_}, D0@{CD}: nMSE {e:.3g} cost {cost:.0f} | full: nMSE {e2:.3g} cost {cost2:.0f} paths ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
