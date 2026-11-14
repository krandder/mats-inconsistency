"""Figures for the exact radius law: figs/radius_density.png, figs/radius_corrections.png, figs/radius_left_tail.png."""
import sys, os, math, json
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')
import numpy as np
from plotstyle import plt, setup, seq, CAT, INK, INK2
from radius_chain import RadiusLaw

D = '/home/agent/workspaces/mats/work/stationary-numerics'
setup()
rc = json.load(open(f'{D}/results/radius_corrections.json'))

# 1. density of X = Q/n (in log x) for several n against the inverse gamma
ns = [2, 4, 8, 32, 256]
x = np.exp(np.linspace(np.log(0.02), np.log(300), 500))
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for k, n in enumerate(ns):
    R = RadiusLaw(n)
    F = np.array([R.Q_cdf(v * n) for v in x])
    dens = np.gradient(F, np.log(x))  # density of log X
    ax[0].plot(x, dens, color=seq(k, len(ns)), label=f'n = {n}')
F0 = np.exp(-0.4 / x)
ax[0].plot(x, np.gradient(F0, np.log(x)), color=INK, ls='--', lw=1.2, label='n = inf: 2/(5Z)')
ax[0].set_xscale('log'); ax[0].set_xlabel('X = |h|^2 / (n^2 sigma^2)'); ax[0].set_ylabel('density of log X')
ax[0].set_title('Exact stationary radius law at finite n'); ax[0].legend(fontsize=8)
# 1/n collapse: n (F_n - F0) against F1
xs = np.exp(np.linspace(np.log(0.05), np.log(50), 300))
F1 = -(65 * xs - 47) * np.exp(-0.4 / xs) / (375 * xs ** 2)
for k, n in enumerate([8, 32, 128, 1024]):
    R = RadiusLaw(n)
    Fn = np.array([R.Q_cdf(v * n) for v in xs])
    ax[1].plot(xs, n * (Fn - np.exp(-0.4 / xs)), color=seq(k + 1, 5), label=f'n = {n}')
ax[1].plot(xs, F1, color=INK, ls='--', lw=1.2, label='F1(x), closed form')
ax[1].axhline(0, color=INK2, lw=0.6)
ax[1].set_xscale('log'); ax[1].set_xlabel('x'); ax[1].set_ylabel('n [P(Q <= x n) - exp(-2/(5x))]')
ax[1].set_title('Finite-n correction collapses onto F1'); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{D}/figs/radius_density.png'); plt.close(fig)

# 2. left tail rates
ns_all = sorted(int(k) for k in rc)
lr = [-math.log(rc[str(n)]['left_tail']['P(r<=2)']) / n for n in ns_all]
lq = [-math.log(rc[str(n)]['left_tail']['P(Q<=1)']) / n for n in ns_all]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(ns_all, lq, 'o-', color=CAT[0], ms=4, label='exact: -ln P(|h|^2 <= n sigma^2) / n')
ax.plot(ns_all, lr, 's-', color=CAT[1], ms=4, label='exact: -ln P(r <= 2) / n')
ax.axhline(0.31071, color=CAT[0], ls='--', lw=1, label='large-deviation rate K(1) = 0.3107')
ax.axhline(0.21424, color=CAT[1], ls='--', lw=1, label='large-deviation rate J(2) = 0.2142')
ax.axhline(0.4, color=INK2, ls=':', lw=1.2, label='inverse-gamma extrapolation 0.4')
ax.set_xscale('log', base=2); ax.set_xlabel('width n'); ax.set_ylabel('decay rate per unit width')
ax.set_title('Near-floor layers: exact left tail of the radius'); ax.legend(fontsize=7.5, loc='upper right')
ax.set_ylim(0.15, 0.62)
fig.tight_layout(); fig.savefig(f'{D}/figs/radius_left_tail.png'); plt.close(fig)
print('ok')
