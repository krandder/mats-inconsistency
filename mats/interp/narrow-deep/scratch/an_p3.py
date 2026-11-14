"""Tables for Part 3 (random nets) and Part 4 estimation (trained nets) -> stdout (markdown)."""
import json, glob, os, math
import numpy as np

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R3 = os.path.join(W, 'runs', 'p3')


def g(x):
    if x is None: return '-'
    if x == 0: return '0'
    return f'{x:.2g}'


print('## Part 3: deep part from the exact law at layer l0 (oracle), nMSE of E[output]\n')
print('| net | trVar share check | K2 from input (FLOPs in MC paths) | ' + ' | '.join(f'l0={l}' for l in (4, 8, 16, 32, 64, 128)) + ' |')
print('|---|---|---|' + '---|' * 6)
for f in sorted(glob.glob(os.path.join(R3, 'truth_*_att.json'))):
    a = json.load(open(f)); nm = os.path.basename(f).replace('truth_', '').replace('_att.json', '')
    if a.get('dead'):
        print(f'| {nm} | dead | - |' + ' - |' * 6); continue
    k = a['k2_input']
    cells = []
    for l0 in (4, 8, 16, 32, 64, 128):
        row = a['l0'].get(str(l0))
        if row is None: cells.append('-'); continue
        cells.append(f"K2b {g(row['att_k2b']['nmse'])}, st {g(row['att_k2b_st']['nmse'])}" + (f", lin {g(row['att_lin']['nmse'])}" if 'att_lin' in row else ''))
    print(f"| {nm} | - | {g(k['nmse'])} ({k['flops']/a['flops_path']:.0f}) | " + ' | '.join(cells) + ' |')

print('\n## Part 3: estimators at equal FLOPs (nMSE; budget = N_eq full MC paths)\n')
for pat, title in ((os.path.join(R3, 'truth_*_samplers.json'), 'random nets'),
                   (os.path.join(W, 'runs', 'p4', 'est_*.json'), 'trained nets')):
    files = sorted(glob.glob(pat))
    if not files: continue
    print(f'### {title}\n')
    print('| net | N_eq | MC | ANT | RQMC | best HYB (which) | K2 from input (cost in paths) |')
    print('|---|---|---|---|---|---|---|')
    for f in files:
        s = json.load(open(f)); nm = os.path.basename(f).replace('truth_', '').replace('_samplers.json', '').replace('est_', '').replace('.json', '')
        k2s = ''
        if 'k2' in s:
            k2s = f"{g(s['k2']['nmse'])} ({s['k2']['flops_in_paths']:.0f})"
        else:
            af = f.replace('_samplers.json', '_att.json')
            if os.path.exists(af):
                a = json.load(open(af))
                if not a.get('dead'): k2s = f"{g(a['k2_input']['nmse'])} ({a['k2_input']['flops']/a['flops_path']:.0f})"
        for neq, row in s['est'].items():
            hy = {k: v['nmse'] for k, v in row.items() if k.startswith('HYB')}
            best = min(hy.items(), key=lambda kv: kv[1]) if hy else (None, None)
            print(f"| {nm} | {neq} | {g(row['MC']['nmse'])} | {g(row['ANT']['nmse'])} | {g(row['RQMC']['nmse'])} | {g(best[1])} ({best[0]}) | {k2s} |")
