"""Curves for RESULTS.md: KL vs M and Solvable log loss vs M (mean over seeds), PNG + CSV. Usage: plot.py OUTDIR TAG"""
import json, sys, glob, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O, TAG = sys.argv[1], sys.argv[2]
E = [json.load(open(f)) for f in sorted(glob.glob(f'{O}/eval_s6_seed*.json'))]
MS = [0, 1, 2, 4, 8, 16]
rows = []
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for arm, lab in [('P1', 'P1 random order'), ('P2', 'P2 stiff-gated VoI'), ('P3', 'P3 confidence-first')]:
    y = [np.mean([e['cells']['0']['P0']['kl'] for e in E])] + [np.mean([e['cells'][str(m)][arm]['kl'] for e in E]) for m in MS[1:]]
    s = [0] + [np.std([e['cells'][str(m)][arm]['kl'] for e in E]) for m in MS[1:]]
    ax[0].errorbar(MS, y, yerr=s, marker='o', label=lab, capsize=3)
    rows += [dict(curve='cell_KL', arm=arm, M=m, mean=round(v, 5), sd=round(q, 5), seeds=len(E)) for m, v, q in zip(MS, y, s)]
ax[0].set_xscale('symlog', linthresh=1); ax[0].set_xlabel('M (chains)'); ax[0].set_ylabel('KL(exact || estimate), nats/cell'); ax[0].set_title(f'{TAG}: cell marginals (M=0 is P0)'); ax[0].legend()
for k, (sub, title) in enumerate([('ll_neg', 'unsolvable boards'), ('ll_pos', 'solvable boards')]):
    for arm, ls in [('P0', ':'), ('witness', '--'), ('null', '-'), ('count', '-.')]:
        y = [np.mean([e['solvable']['P2'][str(m)][arm][sub] for e in E]) for m in MS]
        ax[k + 1].plot(MS, y, ls, marker='o', label=arm)
        rows += [dict(curve=f'solvable_{sub}_P2chains', arm=arm, M=m, mean=round(v, 5), sd='', seeds=len(E)) for m, v in zip(MS, y)]
    ax[k + 1].set_xscale('symlog', linthresh=1); ax[k + 1].set_xlabel('M (chains, P2)'); ax[k + 1].set_ylabel('Solvable log loss'); ax[k + 1].set_title(f'{TAG}: {title}'); ax[k + 1].legend()
plt.tight_layout(); plt.savefig(f'{O}/curves_{TAG}.png', dpi=110)
with open(f'{O}/curves_{TAG}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['curve', 'arm', 'M', 'mean', 'sd', 'seeds']); w.writeheader(); w.writerows(rows)
print('ok', f'{O}/curves_{TAG}.png')
