"""K1 worker3: share of trader wealth held by active funds over time, per arm (main solid, null dashed), mean of seeds."""
import csv, glob, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(os.path.dirname(HERE), 'runs')
ARMS = ['bh-within', 'crp-within', 'crp-cross', 'eg-within', 'eg-cross', 'fof-cross']
COL = dict(zip(ARMS, ['#7f7f7f', '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']))
for N in (1000, 10000):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arm in ARMS:
        S, Sn, x = [], [], None
        for p in sorted(glob.glob(os.path.join(RUNS, f'w3-{arm}-N{N}-s*', 'metrics.csv'))):
            r = list(csv.DictReader(open(p)))
            x = [int(z['step']) for z in r]
            S.append([float(z['share_fund']) for z in r]); Sn.append([float(z['null_share_fund']) for z in r])
        if not S:
            continue
        L = min(map(len, S)); S = np.array([s[:L] for s in S]); Sn = np.array([s[:L] for s in Sn]); x = x[:L]
        ax.plot(x, S.mean(0), color=COL[arm], label=f'{arm} (n={len(S)})')
        ax.fill_between(x, S.min(0), S.max(0), color=COL[arm], alpha=0.12, lw=0)
        ax.plot(x, Sn.mean(0), color=COL[arm], ls='--', lw=1)
    ax.axhline(0.5, color='k', lw=0.5)
    ax.set_xlabel('step'); ax.set_ylabel('wealth share of active funds (vs buy-and-hold groups)')
    ax.set_title(f'K1 funds of funds, N={N}: solid = real beliefs, dashed = shuffled null')
    ax.legend(fontsize=7, frameon=False); fig.tight_layout()
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    fig.savefig(os.path.join(HERE, 'out', f'fund_share_N{N}.png'), dpi=120); plt.close(fig)
print('ok')
