"""Follow-up step 3: tails on trained nets with the radius integrated exactly (positive homogeneity).
usage: python p5_rb.py KIND NPZ OUTJSON [R=16]
Events (each positively homogeneous in the input x for a noiseless bias-free net):
  both:   'norm'   |output| > t
  sorter: 'sorterr' max_i |out_i - sort(x)_i| > t   (a large sorting error)
  RNN:    'margin'  max_{j != y} logit_j - logit_y > t   (a confident wrong answer; y = 2nd argmax of x)
t = the (1 - q) quantile from a 2^22-path MC truth, q in {1e-3, 1e-4}.
Estimators at equal FLOPs (budget = N_eq MC paths): MC (indicator), RQMC (scrambled Sobol inputs, indicator), and RB:
directions u, the statistic G(u) on the unit-norm input (noise scaled by 1/E|x|), P = mean Q_chi_n(t / G(u)) over
G(u) > 0. Exact without noise; with noise an approximation. Relative MSE over R replicates.
"""
import sys, os, json, math, time
import numpy as np
import torch
from scipy.stats import chi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import tnets, nd

torch.set_num_threads(1)
DT = torch.float64
KIND, NPZ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
R = int(sys.argv[4]) if len(sys.argv) > 4 else 16
net = tnets.RNNNet(NPZ) if KIND == 'rnn' else tnets.MLPNet(NPZ)
nin = net.n
fp = net.flops_path(); F_CHI = 200
EX = nd.chi_mean(nin)
g = torch.Generator().manual_seed(47_000_017)
t0 = time.time()


def stats(x, noise_scale=1.0):
    sv = net.s
    net.s = net.s * noise_scale
    try:
        o = net.forward(x, gen=g)
    finally:
        net.s = sv
    st = {'norm': o.norm(dim=1)}
    if KIND == 'rnn':
        y = x.argsort(-1)[:, -2]
        ly = o.gather(1, y[:, None])[:, 0]
        o2 = o.clone(); o2.scatter_(1, y[:, None], -float('inf'))
        st['margin'] = o2.max(1).values - ly
    else:
        st['sorterr'] = (o - x.sort(-1).values).abs().max(1).values
    return st


# truth
allst = {}
for it in range(64):
    s = stats(torch.randn(1 << 16, nin, generator=g, dtype=DT))
    for k, v in s.items():
        allst.setdefault(k, []).append(v)
allst = {k: torch.cat(v) for k, v in allst.items()}
NT = len(next(iter(allst.values())))
truth = {}
for ev, v in allst.items():
    for q in (1e-3, 1e-4):
        t = float(torch.quantile(v[:1 << 20].float(), 1 - q)); p = float((v > t).double().mean())
        truth[(ev, q)] = (t, p, math.sqrt(p * (1 - p) / NT) / p)
res = dict(npz=NPZ, kind=KIND, truth_paths=NT, flops_path=fp, results={})
for (ev, q), (t, p, rse) in truth.items():
    out = {}
    for neq in (1024, 4096, 16384):
        B = neq * fp; errs = {'MC': [], 'RQMC': [], 'RB': []}
        for rep in range(R):
            e = stats(torch.randn(neq, nin, generator=g, dtype=DT))[ev]
            errs['MC'].append((float((e > t).double().mean()) - p) ** 2 / p ** 2)
            sob = torch.quasirandom.SobolEngine(nin, scramble=True, seed=int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
            nq = 1 << int(math.floor(math.log2(neq)))
            xq = torch.special.ndtri(sob.draw(nq, dtype=DT).clamp(1e-12, 1 - 1e-12))
            e = stats(xq)[ev]
            errs['RQMC'].append((float((e > t).double().mean()) - p) ** 2 / p ** 2)
            N = int(B // (fp + F_CHI))
            u = torch.randn(N, nin, generator=g, dtype=DT); u = u / u.norm(dim=1, keepdim=True)
            G = stats(u, 1.0 / EX)[ev]
            pr = np.where(G.numpy() > 0, chi.sf(t / np.maximum(G.numpy(), 1e-300), nin), 0.0)
            errs['RB'].append((float(pr.mean()) - p) ** 2 / p ** 2)
        out[neq] = {k: float(np.mean(v)) for k, v in errs.items()}
        print(f"{os.path.basename(NPZ)} {ev} q={q} N_eq={neq}: MC {out[neq]['MC']:.3g} RQMC {out[neq]['RQMC']:.3g} RB {out[neq]['RB']:.3g} "
              f"(truth rel SE {rse:.3g}; {time.time()-t0:.0f}s)", flush=True)
    res['results'][f'{ev}_{q}'] = dict(t=t, p=p, truth_rel_se=rse, rel_mse=out)
    json.dump(res, open(OUT, 'w'))
# large-N bias check of RB against the truth
for ev in allst:
    for q in (1e-3,):
        t, p, rse = truth[(ev, q)]
        u = torch.randn(1 << 19, nin, generator=g, dtype=DT); u = u / u.norm(dim=1, keepdim=True)
        G = stats(u, 1.0 / EX)[ev]
        pr = np.where(G.numpy() > 0, chi.sf(t / np.maximum(G.numpy(), 1e-300), nin), 0.0)
        res['results'][f'{ev}_{q}']['rb_large'] = dict(p=float(pr.mean()), rel_se=float(pr.std() / math.sqrt(len(pr)) / max(pr.mean(), 1e-300)))
        print(f"   RB bias check {ev} q={q}: RB(2^19) {pr.mean():.4g} vs truth {p:.4g} (truth rel SE {rse:.3g})", flush=True)
json.dump(res, open(OUT, 'w'))
