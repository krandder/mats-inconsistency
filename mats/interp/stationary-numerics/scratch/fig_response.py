"""Revised response figure (after the GPT-6 review): figs/response_collapse.png.
Panel a: CLV k = 8 at n >= 64, tangent response (median, solid) against the two finite sizes (mean, dashed/dotted).
Panel b: survival of the 100x-decay lifetime after one noise-decided gate flip (eligible paths only)."""
import json, glob, math, sys
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')
import numpy as np
from plotstyle import plt, setup, seq, CAT, INK, INK2

D = '/home/agent/workspaces/mats/work/stationary-numerics'
setup()
runs, T = {}, {}
for f in sorted(glob.glob(f'{D}/runs/D_n*_s*.json')):
    d = json.load(open(f)); runs.setdefault(d['n'], []).append(d)
for f in sorted(glob.glob(f'{D}/runs/T_n*_s*.json')):
    d = json.load(open(f)); T.setdefault(d['n'], []).append(d)
NS = sorted(runs)
kk = 8
nk = [n for n in NS if n >= 64 and kk in runs[n][0]['KSET']]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
for i, n in enumerate(nk):
    E = runs[n][0]['EPS']
    for j, e in enumerate(E[1:]):
        tsD = np.array(runs[n][0]['TS'])
        lm = np.mean([np.array(r['curves']['clv_%d_%g' % (kk, e)]['log10r2_median']) for r in runs[n]], 0) - 2 * math.log10(e)
        ax[0].plot(tsD / n, 10 ** lm, color=seq(i, len(nk)), ls=['--', ':'][j], lw=1.1,
                   label=(f'n = {n}' if j == 0 else None))
    if n in T:
        ts = np.array(T[n][0]['TS'])
        lm = np.mean([np.array(r['lnR_median'][str(kk)]) for r in T[n]], 0)
        ax[0].plot(ts / n, np.exp(lm), color=seq(i, len(nk)), ls='-', lw=1.2)
ax[0].set_yscale('log'); ax[0].set_ylim(1e-12, 2)
ax[0].set_xlabel('t / n'); ax[0].set_ylabel('R(t) = r_t^2 / eps^2')
ax[0].set_title('CLV 8, medians over paths: tangent (solid) vs eps = 0.3 (dashed), 3 (dotted) / sqrt(n)', fontsize=8.5)
ax[0].legend(fontsize=7)
for i, n in enumerate(NS):
    life = np.concatenate([np.array(r['flip_nd']['lifetime_r2'])[np.array(r['flip_nd']['had_candidate'], bool)]
                           for r in runs[n]]) / n
    obs = np.sort(life[life <= 2.0])  # lifetimes beyond the 2n window are censored: keep their mass, stop at 2
    sv = 1 - np.arange(1, obs.size + 1) / life.size
    xs = np.concatenate([[0.0], obs, [2.0]]); ys = np.concatenate([[1.0], sv, [sv[-1] if sv.size else 1.0]])
    ax[1].step(xs, ys, where='post', color=seq(i, len(NS)), label=f'n = {n} ({np.mean(life > 2):.0%} beyond 2n)')
ax[1].set_xlim(0, 2.05); ax[1].set_yscale('log'); ax[1].set_xlabel('layers until |dh|^2/|h|^2 falls 100x, / n (censored at 2)')
ax[1].set_ylabel('P(lifetime > x)'); ax[1].set_title('One noise-decided gate flip: lifetimes scale with n', fontsize=9)
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{D}/figs/response_collapse.png'); plt.close(fig)
print('ok')
