"""E7.1 summary: shared spread error retained by each intervention, on the eight E6 nets.
R_int = Cov(d_int, dbar_-i) / Cov(d_base, dbar_-i), dbar_-i = mean of the other seven nets' replicate-2 log-sd deviations
(share_test.npz) on the same (first 4,000) test episodes; pull slope = OLS slope of the log-sd deviation on the Bayes
log-sd; offset = mean log-sd deviation. Bootstrap (500) over episodes for the eight-net average.
usage: python qint_summ.py -> results/qint_summary.json, results/qint_tables.md
"""
import json, os
import numpy as np
import devpred as DP
import qshare as SH

NETS = [nm for nm, _ in SH.MAIN + SH.CROSS]
GROUP = {'L2_s0.05': 'A', 'L2seed1_s0.05': 'A', 'L2mx0_s0.05': 'A', 'L2mx1_s0.05': 'A', 'L2seed2_s0.05': 'B', 'L2ws0_s0.05': 'B', 'L2ws3_s0.05': 'B', 'L2mx4_s0.05': 'B'}
INTS = ['I1', 'I1c', 'I2', 'I3', 'I4a', 'I4b', 'I5', 'I6', 'I7', 'A', 'B']
LABEL = {'I1': 'I1 derived readout (ridge decodes)', 'I1c': 'I1c derived readout (calibrated decodes)', 'I2': 'I2 retrained head on r2',
         'I3': 'I3 retrained MLP readout from a2', 'I4a': 'I4a MLP2 write mean-ablated', 'I4b': 'I4b MLP2 skipped', 'I5': 'I5 noise off everywhere',
         'I6': 'I6 noise off at layer 2', 'I7': 'I7 single-draw predictive (diagnostic)', 'A': 'term A: per-pass spread', 'B': 'term B: between-draw jitter'}
Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); n = 4000; lsb = 0.5 * np.log(Tz['var_bayes'][:n])
have = [nm for nm in NETS if os.path.exists(os.path.join('runs', nm, 'qint.npz'))]
Q = {nm: np.load(os.path.join('runs', nm, 'qint.npz')) for nm in have}
R2 = {nm: np.load(os.path.join('runs', nm, 'share_test.npz'))['r2_logsd'][:n] for nm in NETS}
def get(nm, it):
    z = Q[nm]
    return z[it] if it in ('A', 'B') else z[f'{it}_logsd']
def cov(a, b): return float(np.mean((a - a.mean()) * (b - b.mean())))
def slope(dv): return float(np.polyfit(lsb, dv, 1)[0])
out = dict(nets=have, per_net={}, average={})
for nm in have:
    others = np.mean([R2[o] for o in NETS if o != nm], 0); base = get(nm, 'base')
    cb = cov(base, others)
    row = dict(group=GROUP[nm], base=dict(slope=slope(base), offset=float(base.mean()), cov_shared=cb))
    for it in INTS:
        dv = get(nm, it); row[it] = dict(retained=cov(dv, others) / cb, slope=slope(dv), offset=float(dv.mean()), R2_vs_base=DP.r2(base, dv))
    out['per_net'][nm] = row
rng = np.random.default_rng(0); boots = [rng.integers(0, n, n) for _ in range(500)]
for it in INTS:
    vals = [out['per_net'][nm][it]['retained'] for nm in have]
    bs = []
    for ii in boots:
        r = []
        for nm in have:
            others = np.mean([R2[o][ii] for o in NETS if o != nm], 0); base = get(nm, 'base')[ii]; dv = get(nm, it)[ii]
            r.append(cov(dv, others) / cov(base, others))
        bs.append(float(np.mean(r)))
    out['average'][it] = dict(retained=float(np.mean(vals)), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                              retained_A=float(np.mean([out['per_net'][nm][it]['retained'] for nm in have if GROUP[nm] == 'A'] or [np.nan])),
                              retained_B=float(np.mean([out['per_net'][nm][it]['retained'] for nm in have if GROUP[nm] == 'B'] or [np.nan])),
                              slope=float(np.mean([out['per_net'][nm][it]['slope'] for nm in have])),
                              offset=float(np.mean([out['per_net'][nm][it]['offset'] for nm in have])))
out['average']['base'] = dict(slope=float(np.mean([out['per_net'][nm]['base']['slope'] for nm in have])), offset=float(np.mean([out['per_net'][nm]['base']['offset'] for nm in have])))
json.dump(out, open('results/qint_summary.json', 'w'), indent=1)
f = lambda x: f'{x:.2f}'.replace('-', '−')
L = [f'### Shared spread error retained by each intervention ({len(have)} nets; 1 = unchanged, 0 = removed)\n',
     '| intervention | retained, mean over nets [95% CI] | group A / group B | pull slope on Bayes log-sd (baseline ' + f(out['average']['base']['slope']) + ') | mean offset (baseline ' + f(out['average']['base']['offset']) + ') |',
     '|---|---|---|---|---|']
for it in INTS:
    a = out['average'][it]
    L.append(f"| {LABEL[it]} | {f(a['retained'])} [{f(a['ci'][0])}, {f(a['ci'][1])}] | {f(a['retained_A'])} / {f(a['retained_B'])} | {f(a['slope'])} | {f(a['offset'])} |")
txt = '\n'.join(L); open('results/qint_tables.md', 'w').write(txt); print(txt)
