"""Less noisy version of the p5_rb.py comparison at q = 1e-3 and 1e-4 (added after the R = 16 runs, not
pre-registered): RB's per-direction variance from 2^20 directions, against the exact MC variance (1 - p)/(N p) of an
indicator at the same FLOPs. Thresholds t and truth p are those of runs/p5/rb_<net>.json.
usage: python p5_rbvar.py KIND NPZ RBJSON OUTJSON"""
import sys, os, json, math, time
import numpy as np
import torch
from scipy.stats import chi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import tnets, nd

torch.set_num_threads(1)
DT = torch.float64
KIND, NPZ, RBJ, OUT = sys.argv[1:5]
net = tnets.RNNNet(NPZ) if KIND == 'rnn' else tnets.MLPNet(NPZ)
nin = net.n; fp = net.flops_path(); F_CHI = 200; EX = nd.chi_mean(nin)
rb = json.load(open(RBJ))
g = torch.Generator().manual_seed(59_000_023)
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


keys = list(rb['results'].keys())           # e.g. 'norm_0.001'
S1 = {k: 0.0 for k in keys}; S2 = {k: 0.0 for k in keys}; NU = 0
for it in range(16):
    u = torch.randn(1 << 16, nin, generator=g, dtype=DT); u = u / u.norm(dim=1, keepdim=True)
    G = {ev: v.numpy() for ev, v in stats(u, 1.0 / EX).items()}
    for k in keys:
        ev = k.rsplit('_', 1)[0]; t = rb['results'][k]['t']
        pr = np.where(G[ev] > 0, chi.sf(t / np.maximum(G[ev], 1e-300), nin), 0.0)
        S1[k] += float(pr.sum()); S2[k] += float((pr * pr).sum())
    NU += 1 << 16
B = 16384 * fp; N_rb = int(B // (fp + F_CHI))
res = dict(npz=NPZ, directions=NU, n_rb_at_16384=N_rb, results={})
for k in keys:
    r = rb['results'][k]; p = r['p']
    m = S1[k] / NU; v = S2[k] / NU - m * m
    rb_rel_var = v / (N_rb * p * p)
    mc_rel = (1 - p) / (16384 * p)
    bias = (m - p) / p; bias_se = math.sqrt(v / NU / p ** 2 + r['truth_rel_se'] ** 2)
    meas = r['rel_mse']['16384']
    res['results'][k] = dict(p=p, rb_mean=m, rb_bias=bias, rb_bias_se=bias_se, rb_rel_var_16384=rb_rel_var,
                             mc_rel_exact_16384=mc_rel, gain_exact=mc_rel / rb_rel_var,
                             measured_R16=meas)
    print(f"{os.path.basename(NPZ)} {k}: RB mean {m:.4g} vs truth {p:.4g} (bias {bias:+.3f} +- {bias_se:.3f}); "
          f"at 16,384 paths: MC exact {mc_rel:.3g}, RB variance {rb_rel_var:.3g}, gain {mc_rel/rb_rel_var:.3g}x "
          f"[R=16 measured: MC {meas['MC']:.3g} RQMC {meas['RQMC']:.3g} RB {meas['RB']:.3g}]  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(OUT, 'w'))
