"""Summarise grid K (runs/k2_*.json): per (arm, n, L) medians over nets of nMSE_K2, N_eq, MC's 1/N_eq, RQMC at N_eq,
the K2-to-MC ratio nMSE_K2 * N_eq, RQMC's gain over MC, and the PREREG win count (K2 below both on >= 3 of 4 nets).
usage: python3 scratch/summ_k2.py  -> logs/k2_table.md"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for path in sorted(glob.glob(os.path.join(HERE, '..', 'runs', 'k2_*_n*.json'))):
    d = json.load(open(path))
    for L in ('2', '4', '8', '16'):
        nets = [v[L] for v in d['nets'].values()]
        k2 = np.array([x['nmse_k2'] for x in nets]); neq = np.array([x['N_eq'] for x in nets])
        rq = np.array([x['nmse_rqmc_eq'] for x in nets]); mc = 1 / neq
        wins = int(np.sum((k2 < mc) & (k2 < rq)))
        rm = [x.get('nmse_k2_relu_mehler') for x in nets]
        rows.append(dict(arm=d['arm'], n=d['n'], L=int(L), k2=float(np.median(k2)), neq=float(np.median(neq)),
                         ratio=float(np.median(k2 * neq)), rq_gain=float(np.median(mc / rq)), wins=wins, nn=len(nets),
                         k2_mehler=(float(np.median(rm)) if rm[0] is not None else None),
                         rq1024_gain=float(np.median([(1 / 1024) / x['nmse_rqmc_1024'] for x in nets])),
                         rq16k_gain=float(np.median([(1 / 16384) / x['nmse_rqmc_16384'] for x in nets]))))
order = ['relu', 'softplus16', 'softplus8', 'softplus4', 'softplus2', 'softplus1', 'gelu', 'silu', 'gsrelu0.5', 'smoothstep1',
         'tanhb', 'tanh']
rows.sort(key=lambda r: (r['n'], r['L'], order.index(r['arm']) if r['arm'] in order else 99))
out = ['| n | L | arm | median nMSE_K2 | N_eq | K2/MC (nMSE_K2 * N_eq) | RQMC gain at N_eq | RQMC gain 2^10 / 2^14 | K2 wins (of nets) |',
       '|---|---|---|---|---|---|---|---|---|']
for r in rows:
    out.append(f"| {r['n']} | {r['L']} | {r['arm']} | {r['k2']:.2e} | {r['neq']:.0f} | {r['ratio']:.3g} | {r['rq_gain']:.2f} | "
               f"{r['rq1024_gain']:.2f} / {r['rq16k_gain']:.2f} | {r['wins']}/{r['nn']} |")
open(os.path.join(HERE, '..', 'logs', 'k2_table.md'), 'w').write('\n'.join(out) + '\n')
json.dump(rows, open(os.path.join(HERE, '..', 'runs', 'k2_summary.json'), 'w'), indent=1)
print('\n'.join(out))
