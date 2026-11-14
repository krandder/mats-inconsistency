"""Figures for the report (static PNG, default font, light surface). Categorical slots 1-4 of the validated reference
palette (blue, orange, aqua, yellow) for widths 32, 64, 128, 256; markers differ by width as a secondary encoding."""
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

D_ = '/home/agent/workspaces/mats/work/input-bits'
COL = {32: '#2a78d6', 64: '#eb6834', 128: '#1baf7a', 256: '#eda100'}
MK = {32: 'o', 64: 's', 128: '^', 256: 'D'}
INK, MUTED = '#0b0b0b', '#52514e'
S = [4000, 4001, 4002, 4003]
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'lines.linewidth': 2,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})


def kernel(Lmax):
    rho = 0.0; out = {}
    for l in range(1, Lmax + 1):
        th = math.acos(rho); out[l] = dict(ham=th / math.pi, delta=1 - float(ib.kappa(rho)))
        rho = float(ib.kappa(rho))
    return out


K = kernel(2048)
# ---- Figure 1: what carries the input, and the bits
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for n in (32, 64, 128, 256):
    A = [json.load(open(f'{D_}/runs/p1a_n{n}_s{s}.json')) for s in S]
    Ls = sorted(int(L) for L in A[0]['split'])
    sd = [np.median([a['split'][str(L)]['s_dir'] for a in A]) for L in Ls]
    sa = [np.median([a['split'][str(L)]['s_amp'] for a in A]) for L in Ls]
    ax[0].plot(Ls, sd, '-', marker=MK[n], color=COL[n], ms=6, label=f'n = {n}: direction')
    ax[0].plot(Ls, sa, '--', marker=MK[n], color=COL[n], ms=5, alpha=0.8)
    m = min(len(a['layers']) for a in A)
    hm = np.median([[x['hamming'] for x in a['layers'][:m]] for a in A], 0)
    ls = np.arange(1, m + 1)
    ax[1].loglog(ls, hm, '-', color=COL[n], lw=2, label=f'n = {n}')
    ax[1].loglog(ls, [n * K[l]['ham'] for l in ls], ':', color=COL[n], lw=1.5)
ax[0].set_xscale('log', base=2)
ax[0].set_xlabel('depth L'); ax[0].set_ylabel('share of tr Var_x(a) (median of 4 nets)')
ax[0].set_title('Solid: output direction. Dashed: amplitude |g(u)|', fontsize=10, color=INK)
ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8, loc='center right', frameon=False)
ax[0].annotate('ARC 256 x 32', xy=(32, 0.47), xytext=(60, 0.62), fontsize=8, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED))
ax[1].set_xlabel('layer l'); ax[1].set_ylabel('gates differing between two inputs')
ax[1].set_title('Input gate bits (solid: nets, dotted: kernel n theta/pi)', fontsize=10, color=INK)
ax[1].set_ylim(0.3, 200); ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f'{D_}/figs/fig1_split_bits.png', dpi=130); plt.close(fig)
# ---- Figure 2: synthesis
P2 = json.load(open(f'{D_}/results/part2_scores.json'))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for c in P2['H2.2']['cells']:
    ax[0].scatter(c['G_pred'], c['G'], color=COL[c['n']], marker=MK[c['n']], s=28, edgecolor='#fcfcfb', linewidth=0.8)
for n in (32, 64, 128, 256):
    ax[0].scatter([], [], color=COL[n], marker=MK[n], label=f'n = {n}')
lim = [1, 6]
ax[0].plot(lim, lim, '-', color=MUTED, lw=1); ax[0].plot(lim, [1.5 * x for x in lim], ':', color=MUTED, lw=1)
ax[0].plot(lim, [x / 1.5 for x in lim], ':', color=MUTED, lw=1)
ax[0].set_xlim(1, 5); ax[0].set_ylim(0.8, 6)
ax[0].set_xlabel('predicted 1/(1 - S1 - (1 - rho2(N/n)) W2_off)'); ax[0].set_ylabel('measured RQMC gain (median of 4 nets)')
ax[0].set_title('Input regime: 48 cells (n, L, N); dotted = factor 1.5', fontsize=10, color=INK)
ax[0].legend(fontsize=8, frameon=False, loc='upper left')
cells = P2['H2.3']['cells']
ns = sorted(int(k) for k in cells)
ax[1].plot(ns, [cells[str(k)]['G1024'] for k in ns], '-o', color='#2a78d6', label='RQMC gain at N = 1024')
ax[1].plot(ns, [1 / (1 - cells[str(k)]['S1']) for k in ns], '-s', color='#eb6834', label='1/(1 - first-order share)')
ax[1].set_xscale('log', base=2); ax[1].set_xlabel('width n (noise regime)'); ax[1].set_ylabel('gain over MC')
ax[1].set_ylim(1, 7)
a2 = P2['additions']['A2A']['cells']
g0 = [np.median([r['G0'][0] for r in a2[str(n)]['rows']]) for n in (8, 16)]
g8 = [np.median([r['G8192'][0] for r in a2[str(n)]['rows']]) for n in (8, 16)]
ax[1].plot([8, 16], g0, 'o', mfc='none', mec='#1baf7a', ms=8, mew=2, label='N = 4096, integrand in Sobol dims 0+')
ax[1].plot([8, 16], g8, 'v', color='#1baf7a', ms=8, label='N = 4096, same integrand in dims 8192+')
for n, a_, b_ in zip((8, 16), g0, g8):
    ax[1].annotate('', xy=(n, b_ + 0.15), xytext=(n, a_ - 0.15), arrowprops=dict(arrowstyle='->', color='#1baf7a', lw=1.2))
ax[1].set_ylim(1, 9)
ax[1].set_title('Noise regime: first-order share flat, gain falls', fontsize=10, color=INK)
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f'{D_}/figs/fig2_synthesis.png', dpi=130); plt.close(fig)
print('figs written')
# ---- Figure 3: Part 3 asymptotic constants vs width at fixed L/n
R3 = json.load(open(f'{D_}/results/part3_scores.json'))['per_run']
import collections
gg = collections.defaultdict(list)
for o in R3:
    if o['accounting'] == 'cost':
        gg[(o['method'], o['n'], o['L'])].append(o['gain'])
MC4 = {'RQMC-RB': ('#2a78d6', 'o'), 'ML2': ('#eb6834', 's'), 'MLD': ('#1baf7a', '^'), 'ML2-RQMC': ('#eda100', 'D')}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for k, (lab, f) in enumerate([('L = 32 (fixed depth)', lambda n: 32), ('L = n', lambda n: n), ('L = 2n', lambda n: 2 * n)]):
    for m, (c, mk) in MC4.items():
        ns = [64, 128, 256]
        ys = [np.median(gg[(m, n, f(n))]) for n in ns]
        ax[k].plot(ns, ys, '-', marker=mk, color=c, ms=7, label=m)
    ax[k].set_xscale('log', base=2); ax[k].set_yscale('log'); ax[k].set_xticks([64, 128, 256]); ax[k].set_xticklabels(['64', '128', '256'])
    ax[k].set_title(lab, fontsize=10, color=INK); ax[k].set_xlabel('width n')
    ax[k].axhline(1, color=MUTED, lw=1)
    ax[k].set_yticks([0.5, 1, 2, 4, 8]); ax[k].set_yticklabels(['0.5', '1', '2', '4', '8']); ax[k].minorticks_off()
ax[0].set_ylabel('gain over MC at 2^12 L mv (pilot excluded)')
ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f'{D_}/figs/fig3_samplers.png', dpi=130); plt.close(fig)
print('fig3 written')
