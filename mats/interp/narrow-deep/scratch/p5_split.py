"""Follow-up step 2: gate-aligned split closure (mixture over the on/off states of the k worst-gate units).
usage: python p5_split.py KIND NPZ LOCATEJSON OUTJSON
Selection rules (from p5_locate): S1 = top-k (layer, unit) pairs by accumulated |gate error| of K2-from-input;
S2 = top-k by one-step |gate error|; S0 = k random pairs (control, fixed seed). k in {1, 2, 4, 8}; k = 0 is K2.
For each: E[output] nMSE vs an independent MC truth (2^21 paths), tail P(|output| > t) at q = 1e-3, 1e-4 (Imhof per
mixture component), cost in FLOPs (lib/k2mix counts) in MC-path units; RQMC (scrambled Sobol inputs, iid noise) and
MC at the same budget (R = 16), for the mean and the tails.
"""
import sys, os, json, math, time
import numpy as np
import torch
from scipy.integrate import quad
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import tnets, k2mix

torch.set_num_threads(1)
DT = torch.float64
KIND, NPZ, LOC, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
net = tnets.RNNNet(NPZ) if KIND == 'rnn' else tnets.MLPNet(NPZ)
depth = net.T if KIND == 'rnn' else net.L
w = net.d if KIND == 'rnn' else net.n
nin = net.n
fp = net.flops_path()
loc = json.load(open(LOC))
t0 = time.time()
g = torch.Generator().manual_seed(43_000_007)

# independent truth: mean and |out| tail
so = 0; soo = 0; norms = []; NT = 0
for it in range(32):
    o = net.forward(torch.randn(1 << 16, nin, generator=g, dtype=DT), gen=g)
    so = so + o.sum(0); soo = soo + (o * o).sum(0); norms.append(o.norm(dim=1)); NT += 1 << 16
truth = so / NT; trv = float((soo / NT - truth ** 2).sum()); norms = torch.cat(norms)
tails = {}
for q in (1e-3, 1e-4):
    t = float(torch.quantile(norms[:1 << 20].float(), 1 - q)); p = float((norms > t).double().mean())
    tails[q] = (t, p)


def imhof_sf(x, lam, d2):
    lam = np.asarray(lam, float); d2 = np.asarray(d2, float)
    if x <= 0: return 1.0

    def f(u):
        if u == 0: return 0.5 * (np.sum(lam * (1 + d2)) - x)
        lu = lam * u
        th = 0.5 * np.sum(np.arctan(lu) + d2 * lu / (1 + lu * lu)) - 0.5 * x * u
        rho = np.exp(0.25 * np.sum(np.log1p(lu * lu)) + 0.5 * np.sum(d2 * lu * lu / (1 + lu * lu)))
        return np.sin(th) / (u * rho)
    v, _ = quad(f, 0, np.inf, limit=2000, epsabs=1e-14, epsrel=1e-10)
    return min(max(0.5 + v / math.pi, 0.0), 1.0)


def gauss_norm_tail(mu, C, t):
    ev, Q = torch.linalg.eigh(0.5 * (C + C.T)); mq = Q.T @ mu
    keep = ev > 1e-12 * max(float(ev.max()), 1e-300)
    c0 = float((mq[~keep] ** 2).sum())
    lam = ev[keep].numpy(); d2 = ((mq[keep] ** 2) / ev[keep]).numpy()
    return imhof_sf(t * t - c0, lam, d2)


def run_mix(splits, exact=True):
    if KIND == 'rnn':
        comps, fl = k2mix.rnn_mix(net, splits, exact)
    else:
        comps, fl = k2mix.res_mix(net, splits, exact)
    outm = k2mix.readout(net, comps)
    m = k2mix.mix_mean(outm)
    W = sum(c[0] for c in outm)
    tl = {q: sum(wc * gauss_norm_tail(mc, Cc, tq) for wc, mc, Cc in outm) / W for q, (tq, pq) in tails.items()}
    return dict(nmse=float(((m - truth) ** 2).sum() / trv), flops=fl, paths=fl / fp, ncomp=len(outm),
                tail={str(q): dict(p=tl[q], p_true=tails[q][1], rel_err2=(tl[q] - tails[q][1]) ** 2 / tails[q][1] ** 2) for q in tl})


def samplers_at(Np, R=16):
    Nq = 1 << max(int(math.floor(math.log2(max(Np, 2)))), 1)
    e = {'MC': [], 'RQMC': []}; te = {str(q): {'MC': [], 'RQMC': []} for q in tails}
    for rep in range(R):
        x = torch.randn(Nq, nin, generator=g, dtype=DT); y = net.forward(x, gen=g)
        sob = torch.quasirandom.SobolEngine(nin, scramble=True, seed=int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
        xq = torch.special.ndtri(sob.draw(Nq, dtype=DT).clamp(1e-12, 1 - 1e-12)); yq = net.forward(xq, gen=g)
        for nm, yy in (('MC', y), ('RQMC', yq)):
            e[nm].append(float(((yy.mean(0) - truth) ** 2).sum() / trv))
            for q, (tq, pq) in tails.items():
                ph = float((yy.norm(dim=1) > tq).double().mean())
                te[str(q)][nm].append((ph - pq) ** 2 / pq ** 2)
    return dict(N=Nq, nmse={k: float(np.mean(v)) - 1.0 / NT for k, v in e.items()},
                tail_rel_mse={q: {k: float(np.mean(v)) for k, v in d.items()} for q, d in te.items()})


rules = {}
top_acc = [(l, i) for l, i, _ in loc['gate_acc']['top']]
top_one = [(l, i) for l, i, _ in loc['gate_one']['top']]
rs = np.random.RandomState(7)
rand_pairs = [(int(rs.randint(depth)), int(rs.randint(w))) for _ in range(8)]
rules['S1'] = top_acc; rules['S2'] = top_one; rules['S0'] = rand_pairs
res = dict(npz=NPZ, kind=KIND, trvar=trv, flops_path=fp, truth_paths=NT, tails={str(q): v for q, v in tails.items()}, runs={})
for exact in (True, False):
    tag = 'exact' if exact else 'stein'
    base = run_mix({}, exact)
    base['samplers'] = samplers_at(base['paths'])
    res['runs'][f'{tag}_k0'] = base
    print(f"{os.path.basename(NPZ)} [{tag}] k=0: nMSE {base['nmse']:.3g} ({base['paths']:.0f} paths; RQMC {base['samplers']['nmse']['RQMC']:.3g} MC {base['samplers']['nmse']['MC']:.3g}) "
          f"tail q=1e-3 p {base['tail']['0.001']['p']:.3g} vs {base['tail']['0.001']['p_true']:.3g}  ({time.time()-t0:.0f}s)", flush=True)
    for rule, pairs in rules.items():
        if not exact and rule != 'S1':
            continue
        for k in (1, 2, 4, 8):
            sp = {}
            for (l, i) in pairs[:k]:
                sp.setdefault(l, []).append(i)
            r = run_mix(sp, exact)
            r['samplers'] = samplers_at(r['paths'])
            r['splits'] = pairs[:k]
            res['runs'][f'{tag}_{rule}_k{k}'] = r
            print(f"   [{tag}] {rule} k={k}: nMSE {r['nmse']:.3g} (gain over K2 {base['nmse']/max(r['nmse'],1e-300):.2f}x; {r['paths']:.0f} paths, "
                  f"{r['ncomp']} comps; RQMC there {r['samplers']['nmse']['RQMC']:.3g}) tail1e-3 relerr2 {r['tail']['0.001']['rel_err2']:.3g} "
                  f"tail1e-4 relerr2 {r['tail']['0.0001']['rel_err2']:.3g}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(OUT, 'w'))
json.dump(res, open(OUT, 'w'))
