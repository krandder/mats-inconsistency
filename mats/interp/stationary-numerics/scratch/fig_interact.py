"""Figures: interaction order across width (Part I + Gram chain + L0), causal freezing (Part I2), and the exact
gate cascade after one refresh (Gram chain) in collapsed coordinates."""
import sys, json, glob, math, os
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')
import numpy as np
from plotstyle import plt, setup, seq, CAT, INK, INK2

D = '/home/agent/workspaces/mats/work/stationary-numerics'
setup()
PI = json.load(open(f'{D}/results/partI_scores.json'))
L0 = {int(k): v for k, v in json.load(open(f'{D}/results/l0_theory.json')).items()}
med = {int(k): v for k, v in PI['scores']['I1a']['medians'].items()}
se = {int(k): v for k, v in PI['scores']['I1a']['median_se'].items()}
NS = sorted(med)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for i, (k, lab) in enumerate((('h', 'state h_0'), ('u', 'direction h_0/|h_0|'), ('a', 'output relu(W_1 h_0)'))):
    ax[0].errorbar(NS, [med[n][k] for n in NS], yerr=[se[n][k] for n in NS], fmt='o-', color=CAT[i], ms=4, capsize=2,
                   label=f'{lab}: networks, 8 per width')
nl = sorted(L0)
ax[0].plot(nl, [L0[n]['D_ang_u'] for n in nl], '--', color=CAT[1], lw=1.2, label='direction: infinite-width noisy kernel (L0)')
gw = {}
for n in (8, 16, 32, 64, 128, 256, 512, 1024):
    f = f'{D}/runs/gram_window_n{n}.json'
    if os.path.exists(f):
        gw[n] = json.load(open(f))
gn = sorted(gw)
ax[0].plot(gn, [gw[n]['D']['u'] for n in gn], 's:', color=CAT[1], ms=4, lw=1, label='direction: exact Gram chain (annealed)')
sl = {16: 2.49, 64: 2.135, 256: 3.055}
ax[0].plot(list(sl), list(sl.values()), 'x', color=INK2, ms=7, label='state: stationary lane eps-curve (a lower bound)')
ax[0].set_xscale('log', base=2); ax[0].set_xlabel('width n'); ax[0].set_ylabel('mean interaction order D (layers)')
ax[0].set_title('Interaction order grows with width'); ax[0].legend(fontsize=7)
ax[1].errorbar(NS, [med[n]['g'] for n in NS], yerr=[se[n]['g'] for n in NS], fmt='o-', color=CAT[3], ms=4, capsize=2,
               label='output gates: networks')
ax[1].plot(nl, [L0[n]['D_g'] for n in nl], '--', color=CAT[3], lw=1.2, label='output gates: L0')
ax[1].plot(gn, [gw[n]['D']['g'] for n in gn], 's:', color=CAT[3], ms=4, lw=1, label='output gates: exact Gram chain')
x = np.array([16, 1024])
ax[1].plot(x, 7.6 * (x / 16) ** 0.5, color=INK2, lw=0.8, label='slope 1/2')
ax[1].set_xscale('log', base=2); ax[1].set_yscale('log'); ax[1].set_xlabel('width n'); ax[1].set_ylabel('D of the output gate pattern')
ax[1].set_title('Gate patterns interact much more'); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{D}/figs/interaction_order.png'); plt.close(fig)

# causal: excess fraction vs margin threshold
E = PI['scores']['I2b']['E_h']
ff = PI['scores']['I2b']['frac_free']
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
kap = [0, 0.5, 1, 2, 4]
for i, n in enumerate(sorted(E, key=int)):
    y = [0] + [E[n][f'free_{k}'] for k in ('0.5', '1', '2', '4')]
    ax[0].plot(kap, y, 'o-', color=seq(i, len(E)), ms=4, label=f'n = {n}')
ax[0].axhline(1, color=INK2, lw=0.8)
ax[0].set_xlabel('units free to switch: margin below k noise SDs (others frozen)')
ax[0].set_ylabel('share of the excess order D_h - 1 recovered')
ax[0].set_title('Which gates carry the interactions'); ax[0].legend(fontsize=8)
labs = ['recent_q', 'recent_n', 'clamp']
names = ['only the last n/4 layers switch', 'only the last n layers switch', 'radius clamped']
w = 0.18
for i, n in enumerate(sorted(E, key=int)):
    ax[1].bar(np.arange(3) + (i - 1.5) * w, [E[n][l] for l in labs], width=w, color=seq(i, len(E)), label=f'n = {n}')
ax[1].axhline(1, color=INK2, lw=0.8)
ax[1].set_xticks(np.arange(3)); ax[1].set_xticklabels([x.replace(' layers', '\nlayers') for x in names], fontsize=8)
ax[1].set_ylabel('share of the excess order D_h - 1'); ax[1].set_title('Recent gate switching carries it; the radius is not needed')
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{D}/figs/interaction_causal.png'); plt.close(fig)

# exact gate cascade after one refresh, collapsed
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for i, n in enumerate(gn):
    c = np.array(gw[n]['cascade_flips_per_layer'])
    t = np.arange(c.size)
    ax[0].plot(t[1:] / math.sqrt(n), c[1:] / math.sqrt(n), color=seq(i, len(gn)), label=f'n = {n}')
    ax[1].plot(t[1:] / n, t[1:] * c[1:] / n, color=seq(i, len(gn)), label=f'n = {n}')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('layers t after the refresh / sqrt(n)')
ax[0].set_ylabel('gate flips per layer / sqrt(n)'); ax[0].set_title('Early cascade against t / sqrt(n)')
ax[0].legend(fontsize=7, ncol=2)
ax[1].axhline(3.0, color=INK2, lw=0.8, ls='--', label='3 (finite-angle law, t >> sqrt n)')
ax[1].set_xscale('log'); ax[1].set_xlabel('t / n'); ax[1].set_ylabel('t x flips per layer / n')
ax[1].set_title('t x flips / n: the 3n/t law needs n >> 1e4'); ax[1].legend(fontsize=7)
cas = [gw[n]['cascade_2n_total_over_n'] for n in gn]
ax[2].plot(gn, cas, 'o-', color=CAT[0], ms=4, label='exact Gram chain')
ax[2].plot(nl, [L0[n]['cascade_over_n'] for n in nl], '--', color=CAT[0], lw=1.2, label='L0 kernel theory')
ax[2].plot([64, 64, 256, 256], [2.38, 2.56, 3.78, 4.52], 'x', color=INK2, ms=7, label='stationary lane, networks')
ax[2].set_xscale('log', base=2); ax[2].set_xlabel('width n'); ax[2].set_ylabel('flips over the next 2n layers / n')
ax[2].set_title('Cascade size grows like log n'); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{D}/figs/flip_cascade_exact.png'); plt.close(fig)
print('ok')
