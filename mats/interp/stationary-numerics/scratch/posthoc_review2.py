"""Post hoc re-analysis for the round-2 review -> results/posthoc_review2.json.
(1) Per-net spread of the tangent decay coefficient c (each net's median-over-paths curve fitted separately).
(3) Finite-size plateau against the tangent response with matching statistics (medians over paths, n >= 64).
(4) Flip lifetimes with censoring at 2n made explicit (eligible paths only).
(5) D2 per run (slope of each run's median curve vs that run's QR exponent, k <= 8 in the simulated set) and D4/D5
    on the registered population (paths that had a unit within one noise SD)."""
import json, glob, math
import numpy as np

D = '/home/agent/workspaces/mats/work/stationary-numerics'
out = {}
T = {}
for f in sorted(glob.glob(f'{D}/runs/T_n*_s*.json')):
    d = json.load(open(f)); T.setdefault(d['n'], []).append(d)
R = {}
for f in sorted(glob.glob(f'{D}/runs/D_n*_s*.json')):
    d = json.load(open(f)); R.setdefault(d['n'], []).append(d)
# (1)
ratios, cs = [], []
for n in (64, 128, 256):
    for r in T[n]:
        ts = np.array(r['TS'])
        for k in range(1, r['K'] + 1):
            m = r['lnR_median'][str(k)]
            if m[0] is None:
                continue
            sel = (ts >= n / 8) & (ts <= n)
            c = -np.polyfit(ts[sel] / n, np.array(m)[sel], 1)[0] / (2 * (k + 0.25))
            p = 1 + (k * k + k + 1 / 24) / ((k + 0.25) * n)
            cs.append(c); ratios.append(c / p)
cs, ratios = np.array(cs), np.array(ratios)
out['tangent_per_net'] = dict(count=int(cs.size), c_quantiles=[float(np.quantile(cs, q)) for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)],
                              ratio_quantiles=[float(np.quantile(ratios, q)) for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)])
# (3) medians vs medians at t = n and t = 1.5 n, k >= 4, n >= 64
rows = []
for n in (64, 128, 256):
    ts = np.array(R[n][0]['TS'])
    tsT = np.array(T[n][0]['TS'])
    for k in R[n][0]['KSET']:
        if k < 4:
            continue
        for frac in (1.0, 1.5):
            t = frac * n
            iT = int(np.argmin(np.abs(tsT - t)))
            tang = np.median([r['lnR_median'][str(k)][iT] for r in T[n]]) / math.log(10)  # log10 of the median R
            fin = []
            for e in R[n][0]['EPS'][1:]:
                i = int(np.argmin(np.abs(ts - t)))
                fin.append(np.median([r['curves']['clv_%d_%g' % (k, e)]['log10r2_median'][i] for r in R[n]]) - 2 * math.log10(e))
            rows.append(dict(n=n, k=k, t_over_n=frac, log10R_tangent_median=float(tang), log10R_finite_median=[float(x) for x in fin]))
out['plateau_medians'] = rows
gap = np.array([r['log10R_finite_median'][0] - r['log10R_tangent_median'] for r in rows if r['t_over_n'] == 1.0])
out['plateau_gap_log10_at_t_n'] = [float(gap.min()), float(np.median(gap)), float(gap.max())]
pl = np.array([r['log10R_finite_median'] for r in rows if r['t_over_n'] == 1.0])
out['plateau_median_R_at_t_n'] = [float(10 ** np.median(pl[:, 0])), float(10 ** np.median(pl[:, 1]))]
# (4) lifetimes with censoring
lt = {}
for n in sorted(R):
    L = np.concatenate([np.array(r['flip_nd']['lifetime_r2'])[np.array(r['flip_nd']['had_candidate'], bool)] for r in R[n]]) / n
    cens = L > 2.0
    q = [float(np.quantile(L, x)) for x in (0.25, 0.5, 0.75)]
    lt[str(n)] = dict(q25=q[0] if q[0] <= 2 else '>2', q50=q[1] if q[1] <= 2 else '>2', q75=q[2] if q[2] <= 2 else '>2',
                      censored_frac=float(cens.mean()))
out['lifetimes'] = lt
# (5) D2 per run; D4, D5 on eligible paths
d2 = []
for n in (64, 128, 256):
    for r in R[n]:
        ts = np.array(r['TS'])
        for k in r['KSET']:
            if k > 8:
                continue
            m = np.array(r['curves']['clv_%d_1e-07' % k]['log10r2_median']) * math.log(10)
            sel = (ts >= n / 4) & (ts <= n)
            sl = np.polyfit(ts[sel], m[sel], 1)[0]
            d2.append(sl * n / 2 / r['lyap_times_n'][k - 1])
d2 = np.array(d2)
out['D2_per_run'] = dict(count=int(d2.size), failures=int(np.sum(np.abs(d2 - 1) > 0.25)), range=[float(d2.min()), float(d2.max())],
                         note='k = 5, 7 were not simulated (KSET); per-path pooling is not possible from saved outputs')
d4, d5 = {}, {}
for n in sorted(R):
    per_net = []
    lifes = []
    for r in R[n]:
        e = np.array(r['flip_nd']['had_candidate'], bool)
        per_net.append(np.mean(np.array(r['flip_nd']['cascade'])[e]) / n)
        lifes.append(np.array(r['flip_nd']['lifetime_gates'])[e] / n)
    d4[str(n)] = float(np.median(per_net))
    d5[str(n)] = float(np.median(np.concatenate(lifes)))
out['D4_eligible'] = dict(values=d4, hit=bool(all(0.08 <= d4[str(n)] <= 0.4 for n in sorted(R) if n >= 16)))
m5 = [d5[str(n)] for n in sorted(R) if 32 <= n <= 256]
out['D5_eligible'] = dict(values=d5, hit=bool(max(m5) / min(m5) < 1.5))
json.dump(out, open(f'{D}/results/posthoc_review2.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'plateau_medians'}, indent=1))
