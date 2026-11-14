"""Review item 3: importance sampling for PFN rare outputs at EQUAL TOTAL budget (B = 32,768 forward passes).

Event {mu_model > t} (one noisy pass), 2L sigma 0.1, thresholds at p ~ 2^-10, 2^-16, 2^-19 (truth from diag.json).
Methods (all with exact likelihood ratios for Gaussian mean shifts of the standard-normal randomness u):
- plain MC (B passes);
- mechanistic: shift the query label y* (coordinate W.Q-1) by m = t (net ~ Bayes ~ y* for reliable sources); B passes;
- black-box pilot: 4,096 plain passes; pick j = argmax |cov(u_j, score)|; shift by the regression-implied amount
  m = (t - mean(score)) / slope_j; estimate with the remaining B - 4,096 passes;
- regularised cross-entropy: 4 rounds x 2,048 passes, mean shift restricted to the top-k coordinates by |cov| (k = 1, 4, 16),
  elites = top 10% (threshold capped at t), likelihood-ratio-weighted, shrinkage 0.5 toward the previous round;
  estimate with the remaining B - 8,192 passes.
6 repetitions each; relative RMSE against the subset-simulation truth (itself uncertain, see diag2).
usage: python pfn_is2.py RUN_DIR -> RUN_DIR/is2.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
from pfn_diag import SimMean
from pfn_rare import prior

torch.set_num_threads(1)
B = 2 ** 15


def est_shift(sim, n, mu, t, g):
    u, v = prior(sim, n, g); u = u + mu.astype(np.float32)
    s = sim.score_batched(u, v); lw = -(u.astype(np.float64) @ mu) + 0.5 * float(mu @ mu)
    return float(np.mean(np.exp(lw) * (s > t))), int((s > t).sum())


def run(run_dir, R=6):
    model, meta = N.load_run(run_dir); sim = SimMean(model, meta)
    D = json.load(open(os.path.join(run_dir, 'diag.json'))); ts, P = np.array(D['t']), np.array(D['p_true'])
    sel = sorted(set(int(np.argmin(np.abs(np.log2(P) + k))) for k in (10, 16, 19)))
    iy = W.Q - 1; out = dict(budget=B, rows=[])
    for i in sel:
        t, p = float(ts[i]), float(P[i])
        for method in ('plain', 'mechanistic', 'pilot', 'CE-k1', 'CE-k4', 'CE-k16'):
            ests, hits, picks = [], [], []
            for r in range(R):
                g = np.random.default_rng(10_000 * r + i)
                mu = np.zeros(sim.dim)
                if method == 'plain':
                    e, h = est_shift(sim, B, mu, t, g)
                elif method == 'mechanistic':
                    mu[iy] = t; e, h = est_shift(sim, B, mu, t, g)
                elif method == 'pilot':
                    u, v = prior(sim, 4096, g); s = sim.score_batched(u, v)
                    cov = (u.astype(np.float64) - u.mean(0)).T @ (s - s.mean()) / len(s); j = int(np.argmax(np.abs(cov)))
                    slope = cov[j] / u[:, j].var(); mu[j] = (t - s.mean()) / slope; picks.append(j)
                    e, h = est_shift(sim, B - 4096, mu, t, g)
                else:
                    k = int(method.split('k')[1])
                    u, v = prior(sim, 2048, g); s = sim.score_batched(u, v)
                    cov = (u.astype(np.float64) - u.mean(0)).T @ (s - s.mean()) / len(s); J = np.argsort(-np.abs(cov))[:k]; picks.append(J.tolist())
                    for rnd in range(4):
                        if rnd > 0:
                            u, v = prior(sim, 2048, g); u = u + mu.astype(np.float32); s = sim.score_batched(u, v)
                        lw = -(u.astype(np.float64) @ mu) + 0.5 * float(mu @ mu)
                        thr = min(t, np.quantile(s, 0.9)); el = s >= thr; w = np.exp(lw[el] - lw[el].max())
                        new = np.zeros(sim.dim); new[J] = (w[:, None] * u[el][:, J]).sum(0) / w.sum()
                        mu = 0.5 * mu + 0.5 * new
                    e, h = est_shift(sim, B - 4 * 2048, mu, t, g)
                ests.append(e); hits.append(h)
            ee = np.array(ests)
            row = dict(t=t, log2p=math.log2(p), method=method, reps=ests, hits=hits, picks=picks,
                       log2_mean_ratio=float(math.log2(max(ee.mean(), 1e-300) / p)), rel_rmse=float(np.sqrt(((ee / p - 1) ** 2).mean())))
            out['rows'].append(row); print({k: row[k] for k in ('log2p', 'method', 'log2_mean_ratio', 'rel_rmse', 'hits')}, 'picks', picks[:3], flush=True)
            json.dump(out, open(os.path.join(run_dir, 'is2.json'), 'w'), indent=1)


if __name__ == '__main__':
    run(sys.argv[1])
