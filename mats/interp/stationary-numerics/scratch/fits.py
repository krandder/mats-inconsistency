"""Fitted scaling laws with uncertainties (post-registration summaries of Parts I and D) -> results/fits.json.
(a) Linear CLV response: per (n, k, net) fit ln(median R) = a - 2 c (k + 1/4) t/n over t in [n/8, n]; report c.
(b) Finite-perturbation plateau: mean R at t = n for eps = 0.3/sqrt(n), 3/sqrt(n), k >= 4; exponent beta in eps.
(c) Flip cascade over 2n layers / n and the 100x-decay lifetime / n, per width, with bootstrap over nets / paths.
(d) Interaction-order growth: slope of log(median D) vs log n over n = 16..256 with a bootstrap over nets."""
import json, glob, math
import numpy as np

D = '/home/agent/workspaces/mats/work/stationary-numerics'
rng = np.random.default_rng(3)
out = {}
runs = {}
for f in sorted(glob.glob(f'{D}/runs/D_n*_s*.json')):
    d = json.load(open(f))
    runs.setdefault(d['n'], []).append(d)
# (a)
cs = {}
for n in sorted(runs):
    if n < 32:
        continue
    for k in runs[n][0]['KSET']:
        vals = []
        for r in runs[n]:
            ts = np.array(r['TS']); m = np.array(r['curves']['clv_%d_1e-07' % k]['log10r2_median']) - 2 * math.log10(1e-7)
            sel = (ts >= n / 8) & (ts <= n)
            sl = np.polyfit(ts[sel] / n, m[sel] * math.log(10), 1)[0]
            vals.append(-sl / (2 * (k + 0.25)))
        cs[(n, k)] = vals
allc = np.array([v for vals in cs.values() for v in vals])
byn = {n: np.array([v for (nn, k), vals in cs.items() if nn == n for v in vals]) for n in sorted(set(n for n, _ in cs))}
out['linear_response_c'] = dict(pooled_mean=float(allc.mean()), pooled_sd=float(allc.std()), count=int(allc.size),
                                by_n={str(n): [float(v.mean()), float(v.std() / math.sqrt(v.size))] for n, v in byn.items()},
                                by_k={str(k): float(np.mean([v for (nn, kk), vals in cs.items() if kk == k for v in vals]))
                                      for k in sorted(set(k for _, k in cs))})
# (b)
pl = []
betas = []
for n in sorted(runs):
    if n < 32:
        continue
    for k in runs[n][0]['KSET']:
        if k < 4:
            continue
        E = runs[n][0]['EPS']
        R1 = np.mean([np.array(r['curves']['clv_%d_%g' % (k, E[1])]['r2'])[n] for r in runs[n]]) / E[1] ** 2
        R2 = np.mean([np.array(r['curves']['clv_%d_%g' % (k, E[2])]['r2'])[n] for r in runs[n]]) / E[2] ** 2
        pl.append((n, k, R1, R2))
        betas.append(math.log(R2 / R1) / math.log(E[2] / E[1]))
pl = np.array(pl)
out['plateau'] = dict(R_at_t_n_eps1=dict(median=float(np.median(pl[:, 2])), q25=float(np.quantile(pl[:, 2], 0.25)), q75=float(np.quantile(pl[:, 2], 0.75))),
                      R_at_t_n_eps2=dict(median=float(np.median(pl[:, 3])), q25=float(np.quantile(pl[:, 3], 0.25)), q75=float(np.quantile(pl[:, 3], 0.75))),
                      beta_median=float(np.median(betas)), beta_iqr=[float(np.quantile(betas, 0.25)), float(np.quantile(betas, 0.75))])
# (c)
fl = {}
for n in sorted(runs):
    per_net = np.array([np.mean(r['flip_nd']['cascade']) / n for r in runs[n]])
    life = np.concatenate([np.array(r['flip_nd']['lifetime_r2']) / n for r in runs[n]])
    bm = [np.median(life[rng.integers(0, life.size, life.size)]) for _ in range(2000)]
    fl[str(n)] = dict(cascade_over_n_mean=float(per_net.mean()), cascade_over_n_se=float(per_net.std(ddof=1) / math.sqrt(per_net.size)),
                      lifetime_100x_median_over_n=float(np.median(life)), lifetime_ci90=[float(np.quantile(bm, 0.05)), float(np.quantile(bm, 0.95))],
                      censored_frac=float(np.mean(life > 2)))
out['flip'] = fl
casc = np.array([fl[str(n)]['cascade_over_n_mean'] for n in sorted(runs)])
out['flip_cascade_pooled'] = [float(casc.mean()), float(casc.std(ddof=1) / math.sqrt(casc.size))]
# (d)
PI = json.load(open(f'{D}/results/partI_scores.json'))['I1']
NS = [16, 32, 64, 128, 256]
sl = {}
for key in ('h', 'u', 'a', 'g'):
    boots = []
    for _ in range(2000):
        meds = []
        for n in NS:
            v = np.array([r[key] for r in PI[str(n)]])
            meds.append(np.median(v[rng.integers(0, v.size, v.size)]))
        boots.append(np.polyfit(np.log(NS), np.log(meds), 1)[0])
    pt = np.polyfit(np.log(NS), np.log([np.median([r[key] for r in PI[str(n)]]) for n in NS]), 1)[0]
    sl[key] = dict(slope=float(pt), ci90=[float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))])
out['interaction_order_slope_16_256'] = sl
json.dump(out, open(f'{D}/results/fits.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:3000])
