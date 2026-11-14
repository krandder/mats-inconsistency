"""Aggregate worker1 runs: K1/runs/w1-*/summary.json -> w1/cells.csv (mean, sd over seeds) + w1/*.png."""
import csv, glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), 'runs')
F = ['alive', 'clades_1pct', 'eff_clades', 'top_clade_share', 'top_species_share', 'gini_traders', 'gini_null_traders',
     'eff_traders', 'births', 'bankrupt', 'W_traders', 'R_total', 'sink_compute', 'niches_occupied',
     'top_clade_share_null', 'eff_clades_null', 'alive_null', 'cons_resid_ledger', 'cons_resid_settle']
TAIL = ['tail_brier_all', 'tail_brier_all_null', 'tail_brier_oracle_all', 'tail_bce_all', 'tail_ent_all',
        'tail_turnover_traders']
cells = {}
for p in sorted(glob.glob(os.path.join(RUNS, 'w1-*', 'summary.json'))):
    s = json.load(open(p)); c = s['cfg']
    key = (c['N'], c['Q'], 'on' if c['niches'] else 'off', c['T'])
    row = {k: s['final'].get(k, np.nan) for k in F}
    row.update({k: s.get(k, np.nan) for k in TAIL})
    # species holding the most wealth at the end
    sp = {k[7:]: v for k, v in s['final'].items() if k.startswith('wealth_')}
    row['top_species'] = max(sp, key=sp.get)
    row['mono'] = float(bool(s['monoculture']))
    row['mono_null'] = float(bool(s.get('monoculture_null')))
    cells.setdefault(key, []).append(row)
out = os.path.join(HERE, 'cells.csv')
cols = ['N', 'Q', 'niches', 'T', 'seeds'] + [f'{k}_{m}' for k in F + TAIL + ['mono', 'mono_null'] for m in ('mean', 'sd')] + ['top_species']
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(cols)
    for key in sorted(cells):
        rs = cells[key]
        r = list(key) + [len(rs)]
        for k in F + TAIL + ['mono', 'mono_null']:
            v = np.array([x[k] for x in rs], float)
            r += [np.nanmean(v), np.nanstd(v, ddof=1) if len(v) > 1 else np.nan]
        r.append('/'.join(x['top_species'] for x in rs))
        w.writerow(r)
print('wrote', out, len(cells), 'cells')

# trajectories: top clade share and gini vs null over time, one panel per N
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
except Exception as e:
    print('no matplotlib', e); sys.exit()
Ns = sorted({k[0] for k in cells})
fig, ax = plt.subplots(2, len(Ns), figsize=(5 * len(Ns), 7), squeeze=False)
col = {(1, 'off'): '#444444', (64, 'on'): '#3274D9', (64, 'off'): '#9BBEF0', (4096, 'on'): '#E0522D', (4096, 'off'): '#F2A58F'}
for j, N in enumerate(Ns):
    for (Q, nic), cc in col.items():
        for p in sorted(glob.glob(os.path.join(RUNS, f'w1-n{N//1000}k-q{Q}-{nic}-s*', 'metrics.csv'))):
            d = list(csv.DictReader(open(p)))
            if not d: continue
            st = np.array([float(r['step']) for r in d])
            lab = f'Q{Q} {nic}' if p.endswith('s1/metrics.csv') or '-s1-' in p else None
            ax[0, j].plot(st, [float(r['top_clade_share']) for r in d], color=cc, lw=1, label=lab)
            ax[1, j].plot(st, [float(r['gini_traders']) for r in d], color=cc, lw=1, label=lab)
            ax[1, j].plot(st, [float(r['gini_null_traders']) for r in d], color=cc, lw=1, ls='--')
    ax[0, j].set_title(f'N = {N:,} traders'); ax[0, j].set_ylabel('top clade wealth share'); ax[0, j].set_ylim(0, 1)
    ax[1, j].set_ylabel('Gini (dashed: beliefs-shuffled null)'); ax[1, j].set_xlabel('step'); ax[1, j].set_ylim(0, 1)
    ax[0, j].axhline(0.9, color='k', lw=0.5, ls=':')
ax[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'trajectories.png'), dpi=110)
print('wrote trajectories.png')
