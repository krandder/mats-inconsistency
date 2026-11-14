"""Evaluate one trained model (or its untrained initialisation with --init).

Metrics (PREREG.md):
 1. task: test NLL on 65536 fresh episodes (seed 20000) and gap to the exact Bayes NLL;
 2. estimation: K=2 and K=4 propagated E[mu], E[logs], E[nll] vs Monte Carlo (n = --n_mc) -> variance-normalised
    squared error nse = (est - truth)^2 / Var and N_eff = 1/nse; per-point discrepancies (mean/cov/skew/kurtosis);
 3. matched-compute sampling: propagator FLOPs / forward-pass FLOPs per episode = N_match; sampling nse = 1/N_match;
 4. mechanism: GELU pre-activations in the curved zone |z|<1.5, attention entropy / log(#keys), rms standardised
    kappa3/kappa4 of the residual stream, LN-input token-norm coefficient of variation, relative weight change.
usage: python evaluate.py --ckpt DIR/model.pt [--init] --out DIR/eval.json
"""
import argparse, json, math, os, time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import world_small as W
from model import Net, nll
from prop import Propagator
from mcstats import mc_truth, stat_errors

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--init', action='store_true', help='evaluate the untrained initialisation of this run')
ap.add_argument('--out', required=True)
ap.add_argument('--n_mc', type=int, default=1 << 22)
ap.add_argument('--n_test', type=int, default=65536)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--threads', type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)
DEV = a.device
meta = json.load(open(os.path.join(os.path.dirname(a.ckpt), 'meta.json')))
torch.manual_seed(meta['seed'])
init_net = Net(meta['depth'])
net = Net(meta['depth'])
net.load_state_dict(init_net.state_dict())
if not a.init:
    net.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
net = net.to(DEV).eval(); init_net = init_net.to(DEV)
res = dict(meta={k: meta[k] for k in ('depth', 'lam', 'K', 'seed', 'steps')}, init=a.init)

# ---- 1. task
smp = W.Sampler(20_000, DEV)
d = smp.sample(a.n_test)
with torch.no_grad():
    L = torch.cat([nll(*net(d['X'][i:i + 8192])[:2], d['ystar'][i:i + 8192]) for i in range(0, a.n_test, 8192)])
bn, _, _, bn_b = W.bayes_nll(d['o'].cpu().numpy(), d['y'].cpu().numpy())
g = L.double().cpu().numpy() - bn
res['task'] = dict(nll=float(L.mean()), bayes=float(bn.mean()), gap=float(g.mean()), gap_se=float(g.std() / math.sqrt(len(g))),
                   iid_ref_gap=float((bn_b - bn).mean()))

# ---- 2. estimation
t0 = time.time()
truth, pts = mc_truth(net, a.n_mc, seed=30_000 + meta['seed'], device=DEV, dtype=torch.float32, bs=32768)
res['mc_seconds'] = time.time() - t0
res['truth'] = truth
for K in (2, 4):
    pr = Propagator(K=K, dtype=torch.float32, device=DEV)
    t0 = time.time()
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            o = pr.run(net)
    sec = time.time() - t0
    r = dict(seconds=sec, flops=fc.get_total_flops(), J=pr.J)
    for q in ('mu', 'logs', 'nll'):
        est, tru, var = o['out'][f'E_{q}'].item(), truth[f'E_{q}'], truth[f'V_{q}']
        r[q] = dict(est=est, true=tru, var=var, nse=(est - tru) ** 2 / var, mc_nse=1.0 / a.n_mc)
    r['points'] = [stat_errors({k: v.double().cpu() for k, v in p.items()}, e) for p, e in zip(o['points'], pts)]
    res[f'K{K}'] = r
# ---- 3. matched compute
with FlopCounterMode(display=False) as fc:
    with torch.no_grad():
        net(d['X'][:1024])
res['fwd_flops_per_episode'] = fc.get_total_flops() / 1024
for K in (2, 4):
    res[f'K{K}']['N_match'] = res[f'K{K}']['flops'] / res['fwd_flops_per_episode']

# ---- 4. mechanism
with torch.no_grad():
    X = d['X'][:16384]
    x = net.inp(X); curved, ent, lncv = [], [], []
    for blk in net.blocks:
        y = blk.ln1(x)
        lncv.append(float((x - x.mean(-1, keepdim=True)).norm(dim=-1).std(0).div((x - x.mean(-1, keepdim=True)).norm(dim=-1).mean(0)).mean()))
        att, A = blk.attn(y, net.allow)
        nk = net.allow.sum(-1).float()
        e = -(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(-1) / nk.log()
        ent.append(float(e.mean()))
        x1 = x + att
        z = blk.fc1(blk.ln2(x1))
        curved.append(float((z.abs() < 1.5).float().mean()))
        x = x1 + blk.fc2(torch.nn.functional.gelu(z))
    _, _, rs, _ = net(X)
    sk, ku = [], []
    for r_ in rs[1:]:
        z_ = r_.reshape(len(X), -1); zc = z_ - z_.mean(0); v = (zc ** 2).mean(0)
        sk.append(float(((zc ** 3).mean(0) / v ** 1.5).pow(2).mean().sqrt()))
        ku.append(float(((zc ** 4).mean(0) / v ** 2 - 3).pow(2).mean().sqrt()))
wc = {n: float((p - p0).norm() / p0.norm().clamp_min(1e-12)) for (n, p), (_, p0) in
      zip(net.named_parameters(), init_net.named_parameters()) if p.dim() == 2}
res['mech'] = dict(gelu_curved_frac=curved, attn_entropy=ent, ln_norm_cv=lncv, resid_skew_rms=sk, resid_exkurt_rms=ku,
                   weight_change_mean=float(np.mean(list(wc.values()))), weight_change=wc)
json.dump(res, open(a.out, 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('task',)}), json.dumps({K: {q: res[K][q]['nse'] for q in ('mu', 'logs', 'nll')}
                                                              for K in ('K2', 'K4')}), flush=True)
