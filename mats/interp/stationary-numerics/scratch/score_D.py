"""Score PREREG Part D (response kernels, flip response, collapse) -> results/partD_scores.json and figures."""
import json, glob, math, sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')
from plotstyle import plt, setup, seq, CAT, INK, INK2

D = '/home/agent/workspaces/mats/work/stationary-numerics'
setup()
runs = {}
for f in sorted(glob.glob(f'{D}/runs/D_n*_s*.json')):
    d = json.load(open(f))
    runs.setdefault(d['n'], []).append(d)
NS = sorted(runs)
LN10 = math.log(10)


def key(k, e):
    return 'clv_%d_%g' % (k, e)


def logR_median(n, k, ei):
    """seed-averaged median of log10 R(t) = log10(r_t^2 / eps^2) at the run's TS."""
    rs = runs[n]
    e = rs[0]['EPS'][ei]
    arr = np.array([np.array(r['curves'][key(k, e)]['log10r2_median']) for r in rs]) - 2 * math.log10(e)
    return np.array(rs[0]['TS']), arr.mean(0), arr.std(0) / math.sqrt(len(rs))


S = {}
# D1: linear collapse in t/n, k <= 4, n = 64, 128, 256
grid = np.linspace(0.1, 1.0, 19)
d1 = {}
ok1 = True
for k in (1, 2, 3, 4):
    curves = {}
    for n in (64, 128, 256):
        if n in runs and k in runs[n][0]['KSET']:
            ts, m, s = logR_median(n, k, 0)
            curves[n] = np.interp(grid, ts / n, m)
    if len(curves) == 3:
        mx = max(np.max(np.abs(curves[a] - curves[b])) for a in curves for b in curves if a < b)
        d1[k] = float(mx)
        ok1 &= mx < 0.5
S['D1'] = dict(hit=bool(ok1), max_pairwise_diff_log10=d1)
# D2: slope over [n/4, n] times n/2 vs n lambda_k
d2 = {}
ok2 = True
for n in NS:
    if n < 64:
        continue
    lam = np.mean([r['lyap_times_n'] for r in runs[n]], 0)
    for k in runs[n][0]['KSET']:
        if k > 8:
            continue
        ts, m, s = logR_median(n, k, 0)
        sel = (ts >= n / 4) & (ts <= n)
        sl = np.polyfit(ts[sel], m[sel] * LN10, 1)[0]
        est = sl * n / 2
        d2[f'{n}_{k}'] = dict(slope_times_n_over_2=float(est), n_lambda_k=float(lam[k - 1]), ratio=float(est / lam[k - 1]))
        ok2 &= abs(est / lam[k - 1] - 1) <= 0.25
S['D2'] = dict(hit=bool(ok2), rows=d2)
# D3: nonlinear floor at t = n
d3 = {}
ok3 = True
betas = []
for n in NS:
    if n < 32:
        continue
    for k in runs[n][0]['KSET']:
        if k < 4:
            continue
        E = runs[n][0]['EPS']
        R = []
        for e in E:
            R.append(np.mean([np.array(r['curves'][key(k, e)]['r2'])[n] for r in runs[n]]) / e ** 2)
        ratio = R[2] / max(R[0], 1e-300)
        beta = math.log(R[2] / R[1]) / math.log(E[2] / E[1])
        d3[f'{n}_{k}'] = dict(R_lin=R[0], R_eps1=R[1], R_eps2=R[2], ratio_eps2_lin=ratio, beta=beta)
        ok3 &= ratio > 10
        betas.append(beta)
bmed = float(np.median(betas))
S['D3'] = dict(hit=bool(ok3 and 0.5 <= bmed <= 1.2), beta_median=bmed, rows=d3)
# D4, D5: flips
d4, d5 = {}, {}
for n in NS:
    casc = [np.mean(r['flip_nd']['cascade']) / n for r in runs[n]]
    d4[n] = float(np.median(casc))
    life = np.concatenate([np.array(r['flip_nd']['lifetime_gates']) for r in runs[n]]) / n
    d5[n] = dict(median=float(np.median(life)), q25=float(np.quantile(life, 0.25)), q75=float(np.quantile(life, 0.75)),
                 censored_frac=float(np.mean(life >= 2 * 1.0 - 1.0 / n)))
S['D4'] = dict(hit=bool(all(0.08 <= d4[n] <= 0.4 for n in NS if n >= 16)), cascade_over_n=d4)
meds = [d5[n]['median'] for n in NS if 32 <= n <= 256]
S['D5'] = dict(hit=bool(max(meds) / max(min(meds), 1e-9) < 1.5), lifetime_over_n=d5)
json.dump(S, open(f'{D}/results/partD_scores.json', 'w'), indent=1)
for k, v in S.items():
    print(k, {kk: vv for kk, vv in v.items() if kk.startswith('hit')})
print('D1', d1)
print('D2', {k: round(v['ratio'], 2) for k, v in d2.items()})
print('D3 beta median', round(bmed, 2), {k: (round(v['ratio_eps2_lin'], 1), round(v['beta'], 2)) for k, v in d3.items()})
print('D4', d4)
print('D5', {n: (round(v['median'], 2), round(v['censored_frac'], 2)) for n, v in d5.items()})

# ---- figures ----
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
# (a) linear response collapse: ln R vs (k + 1/4) t / n
n32 = [n for n in NS if n >= 32]
for i, n in enumerate(n32):
    for k in runs[n][0]['KSET']:
        ts, m, s = logR_median(n, k, 0)
        x = (k + 0.25) * ts / n
        ax[0].plot(x, m * LN10, color=seq(i, len(n32)), lw=0.9, alpha=0.8, label=f'n = {n}' if k == 1 else None)
xx = np.linspace(0, 6, 50)
ax[0].plot(xx, -2 * xx, color=INK, ls='--', lw=1.2, label='slope -2 (rank-conditioned law)')
ax[0].set_xlim(0, 6); ax[0].set_ylim(-14, 1)
ax[0].set_xlabel('(k + 1/4) t / n'); ax[0].set_ylabel('median ln R_lin(t), all CLV k')
ax[0].set_title('Linear response along CLV k collapses'); ax[0].legend(fontsize=7)
# (b) finite perturbations: mean R(t) for one k at the three amplitudes, several n, against t/n
kk = 8
nk = [n for n in NS if n >= 32 and kk in runs[n][0]['KSET']]
for i, n in enumerate(nk):
    E = runs[n][0]['EPS']
    for j, e in enumerate(E):
        R = np.mean([np.array(r['curves'][key(kk, e)]['r2']) for r in runs[n]], 0) / e ** 2
        t = np.arange(R.size)
        ax[1].plot(t[1:] / n, R[1:], color=seq(i, len(nk)), ls=['-', '--', ':'][j], lw=1.1,
                   label=(f'n = {n}' if j == 0 else None))
ax[1].set_yscale('log'); ax[1].set_xlabel('t / n'); ax[1].set_ylabel('mean R(t) = r_t^2 / eps^2')
ax[1].set_title(f'CLV k = {kk}: linear (solid) vs eps = 0.3, 3 / sqrt(n)'); ax[1].legend(fontsize=7)
ax[1].set_ylim(1e-8, 2)
# (c) flip lifetimes survival in t/n
for i, n in enumerate(NS):
    life = np.concatenate([np.array(r['flip_nd']['lifetime_gates']) for r in runs[n]]) / n
    xs = np.sort(life)
    sv = 1 - np.arange(1, xs.size + 1) / xs.size
    ax[2].step(xs, sv, where='post', color=seq(i, len(NS)), label=f'n = {n}')
ax[2].set_yscale('log'); ax[2].set_xlabel('gate lifetime / n (censored at 2)'); ax[2].set_ylabel('P(lifetime > x)')
ax[2].set_title('One noise-decided flip: gate-difference lifetimes'); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{D}/figs/response_collapse.png'); plt.close(fig)
print('fig ok')
