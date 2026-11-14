"""(3) on the PFN, at the input level: does a mechanistic proposal beat a black-box one for rare outputs?
Event B_t = {mu_model > t} (one noisy pass), truth from pfn_diag.py (subset simulation).
- mechanistic IS: the net is near-Bayes and the Bayes mean is a precision-weighted average of the query readings
  o*_s = y* + e_s, so large outputs come from a large query label y*. Proposal: y* ~ N(m, 1) with m = t, everything else
  from the prior; weight = N(y*;0,1)/N(y*;m,1). No search, no learning.
- black-box IS: cross-entropy method with a mean shift on the whole 4,756-dim standard-normal vector (+ uniform v),
  5 adaptation rounds of 4,096 passes, elite fraction 10%, then the estimate.
- plain MC and subset simulation at the same number of forward passes.
usage: python pfn_is.py RUN_DIR -> RUN_DIR/is.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
from pfn_diag import SimMean
from pfn_rare import prior

torch.set_num_threads(1)


def run(run_dir, R=6, budget=2 ** 15):
    model, meta = N.load_run(run_dir); sim = SimMean(model, meta)
    D = json.load(open(os.path.join(run_dir, 'diag.json')))
    ts, P = np.array(D['t']), np.array(D['p_true'])
    sel = sorted(set(int(np.argmin(np.abs(np.log2(P) + k))) for k in (10, 16, 20, 26)))
    iy = W.Q - 1                                                  # index of y* in u (labels come first)
    out = dict(rows=[])
    for i in sel:
        t, p = ts[i], P[i]
        for method in ('mechanistic', 'black-box', 'plain'):
            ests = []; t0 = time.time()
            for r in range(R):
                g = np.random.default_rng(1000 * r + i)
                if method == 'plain':
                    u, v = prior(sim, budget, g); s = sim.score_batched(u, v); ests.append(float((s > t).mean()))
                elif method == 'mechanistic':
                    u, v = prior(sim, budget, g); m = t; u[:, iy] += m
                    lw = -m * u[:, iy].astype(np.float64) + 0.5 * m * m
                    s = sim.score_batched(u, v); ests.append(float(np.mean(np.exp(lw) * (s > t))))
                else:
                    mu = np.zeros(sim.dim, np.float32); nr = budget // 8
                    for k in range(5):
                        u, v = prior(sim, nr, g); u += mu
                        s = sim.score_batched(u, v); lw = -(u @ mu).astype(np.float64) + 0.5 * float(mu @ mu)
                        thr = min(t, np.quantile(s, 0.9)); el = s >= thr
                        wel = np.exp(lw[el] - lw[el].max()); mu = ((wel[:, None] * u[el]).sum(0) / wel.sum()).astype(np.float32)
                    nfin = budget - 5 * nr
                    u, v = prior(sim, nfin, g); u += mu
                    lw = -(u @ mu).astype(np.float64) + 0.5 * float(mu @ mu)
                    s = sim.score_batched(u, v); ests.append(float(np.mean(np.exp(lw) * (s > t))))
            e = np.array(ests); dt = (time.time() - t0) / R
            row = dict(t=float(t), log2p=float(np.log2(p)), method=method, budget=budget, sec=dt,
                       log2_mean_ratio=float(np.log2(max(e.mean(), 1e-300) / p)), rel_rmse=float(np.sqrt(((e / p - 1) ** 2).mean())),
                       frac_zero=float((e == 0).mean()))
            out['rows'].append(row); print(row, flush=True)
            json.dump(out, open(os.path.join(run_dir, 'is.json'), 'w'), indent=1)


if __name__ == '__main__':
    run(sys.argv[1])
