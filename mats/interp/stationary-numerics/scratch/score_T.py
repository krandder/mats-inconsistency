"""Post hoc: exact tangent (linear) response along CLVs -> results/tangent_fits.json, figs/response_collapse.png
(panel a replaced by tangent data). Fit per (n, k, net): median ln R(t) = a - 2 c (k + 1/4) t/n over t in [n/8, n];
compare c with the rank-conditioned exponents gamma_k = -(k + 1/4)/n - (k^2 + k + 1/24)/n^2 (astra r1 P8), which give
c_k = 1 + (k^2 + k + 1/24)/((k + 1/4) n) for the CLV relative to the radius (the radius itself is stationary)."""
import json, glob, math, sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')
from plotstyle import plt, setup, seq, CAT, INK, INK2

D = '/home/agent/workspaces/mats/work/stationary-numerics'
setup()
runs = {}
for f in sorted(glob.glob(f'{D}/runs/T_n*_s*.json')):
    d = json.load(open(f))
    runs.setdefault(d['n'], []).append(d)
NS = sorted(runs)
fits = {}
for n in NS:
    for k in range(1, runs[n][0]['K'] + 1):
        cs = []
        for r in runs[n]:
            ts = np.array(r['TS']); m = r['lnR_median'][str(k)]
            if m[0] is None:
                continue
            m = np.array(m)
            sel = (ts >= n / 8) & (ts <= n)
            sl = np.polyfit(ts[sel] / n, m[sel], 1)[0]
            cs.append(-sl / (2 * (k + 0.25)))
        if cs:
            pred = 1 + (k * k + k + 1 / 24) / ((k + 0.25) * n)
            fits[f'{n}_{k}'] = dict(n=n, k=k, c_mean=float(np.mean(cs)), c_se=float(np.std(cs, ddof=1) / math.sqrt(len(cs))) if len(cs) > 1 else None,
                                   c_pred_second_order=pred, nets=len(cs))
rl = {str(n): [r['rankloss_fraction'] for r in runs[n]] for n in NS}
allc = np.array([v['c_mean'] for v in fits.values() if v['n'] >= 64])
allp = np.array([v['c_pred_second_order'] for v in fits.values() if v['n'] >= 64])
out = dict(fits=fits, rankloss=rl, c_mean_n_ge_64=float(allc.mean()), c_over_pred_mean=float(np.mean(allc / allp)),
           c_over_pred_range=[float(np.min(allc / allp)), float(np.max(allc / allp))])
json.dump(out, open(f'{D}/results/tangent_fits.json', 'w'), indent=1)
print('rank loss', rl)
print('mean c (n>=64) %.3f; c/pred mean %.3f range %s' % (out['c_mean_n_ge_64'], out['c_over_pred_mean'], out['c_over_pred_range']))
for n in NS:
    print(n, ' '.join('k%d:%.2f/%.2f' % (v['k'], v['c_mean'], v['c_pred_second_order']) for v in fits.values() if v['n'] == n))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for i, n in enumerate([n for n in NS if n >= 64]):
    for k in range(1, runs[n][0]['K'] + 1):
        arr = [np.array(r['lnR_median'][str(k)]) for r in runs[n] if r['lnR_median'][str(k)][0] is not None]
        ts = np.array(runs[n][0]['TS'])
        m = np.mean(arr, 0)
        ax[0].plot((k + 0.25) * ts / n, m, color=seq(i, 3), lw=0.8, alpha=0.8, label=f'n = {n}' if k == 1 else None)
xx = np.linspace(0, 40, 50)
ax[0].plot(xx, -2 * xx, color=INK, ls='--', lw=1.1, label='slope -2')
ax[0].set_xlim(0, 30); ax[0].set_ylim(-65, 2); ax[0].set_xlabel('(k + 1/4) t / n (beyond 30: double-precision floor of the CLVs)'); ax[0].set_ylabel('median ln R(t), tangent propagation')
ax[0].set_title('Linear response along CLV k = 1..24 (n >= 64)'); ax[0].legend(fontsize=7)
for i, n in enumerate([n for n in NS if n >= 64]):
    ks = [v['k'] for v in fits.values() if v['n'] == n]
    cm = [v['c_mean'] for v in fits.values() if v['n'] == n]
    ce = [v['c_se'] or 0 for v in fits.values() if v['n'] == n]
    cp = [v['c_pred_second_order'] for v in fits.values() if v['n'] == n]
    ax[1].errorbar(ks, cm, yerr=ce, fmt='o', ms=3, color=seq(i, 3), label=f'n = {n}: fitted c')
    ax[1].plot(ks, cp, '-', color=seq(i, 3), lw=1, label=f'n = {n}: 1 + (k^2+k+1/24)/((k+1/4) n)')
ax[1].axhline(1, color=INK2, lw=0.7)
ax[1].set_xlabel('CLV index k'); ax[1].set_ylabel('decay coefficient c')
ax[1].set_title('Rate = 2c(k + 1/4)/n; c from the rank-conditioned law'); ax[1].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(f'{D}/figs/response_tangent.png'); plt.close(fig)
print('fig ok')
