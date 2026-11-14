"""Part 3 tables for the report -> results/part3_tables.md (medians over the 2 nets)."""
import json, collections
import numpy as np

D_ = '/home/agent/workspaces/mats/work/input-bits'
S = json.load(open(f'{D_}/results/part3_scores.json'))
out = S['per_run']
ph = S['rules']['posthoc_largest_budget']
M = ['RBMC', 'RQMC', 'RQMC-RB', 'RQMC-ROT', 'CVQ', 'RQMC-CVQ', 'ML2', 'MLD', 'ML2-RQMC']
g = collections.defaultdict(list); e = collections.defaultdict(list); el = collections.defaultdict(list)
for o in out:
    if o['accounting'] == 'cost':
        g[(o['n'], o['L'], o['method'])].append(o['gain']); e[(o['n'], o['L'], o['method'])].append(o['exponent'])
        el[(o['n'], o['L'], o['method'])].append(o['exponent_last'])
gc = collections.defaultdict(list)
for o in ph:
    gc[(o['n'], o['L'], o['method'])].append(o['gain_charged'])
cells = sorted({(o['n'], o['L']) for o in out})
L1 = ['## Gain over plain MC at C* = 2^12 L mv, setup and pilot excluded (median of 2 nets)',
      '| n | L | ' + ' | '.join(M) + ' |', '|---|---|' + '---|' * len(M)]
for n, L in cells:
    L1.append(f'| {n} | {L} | ' + ' | '.join(f'{np.median(g[(n, L, m)]):.2f}' if g[(n, L, m)] else '-' for m in M) + ' |')
L1 += ['', '## Fitted exponent of nMSE in compute over N = 2^6..2^12 (last step 2^10 -> 2^12 in brackets), RQMC-type methods',
       '| n | L | RQMC | RQMC-RB | RQMC-ROT | RQMC-CVQ | ML2-RQMC |', '|---|---|---|---|---|---|---|']
for n, L in cells:
    L1.append(f'| {n} | {L} | ' + ' | '.join(f'{np.median(e[(n, L, m)]):.2f} ({np.median(el[(n, L, m)]):.2f})' if e[(n, L, m)] else '-'
                                         for m in ['RQMC', 'RQMC-RB', 'RQMC-ROT', 'RQMC-CVQ', 'ML2-RQMC']) + ' |')
L1 += ['', '## Gain at each method\'s largest measured budget with setup and pilot charged (post hoc accounting)',
       '| n | L | ' + ' | '.join(M) + ' |', '|---|---|' + '---|' * len(M)]
for n, L in cells:
    L1.append(f'| {n} | {L} | ' + ' | '.join(f'{np.median(gc[(n, L, m)]):.2f}' if gc[(n, L, m)] else '-' for m in M) + ' |')
cv = collections.defaultdict(list)
for f in __import__('glob').glob(f'{D_}/runs/p3_n*_L*_s*.json'):
    r = json.load(open(f)); cv[(r['n'], r['L'])].append(r['cvq']['ratio'])
L1 += ['', '## CVQ residual variance share (V(a - c)/V)', '| n | L | nets |', '|---|---|---|']
for k in sorted(cv):
    L1.append(f'| {k[0]} | {k[1]} | ' + ', '.join(f'{x:.2f}' for x in cv[k]) + ' |')
open(f'{D_}/results/part3_tables.md', 'w').write('\n'.join(L1) + '\n')
print('\n'.join(L1))
