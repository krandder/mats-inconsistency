"""Tables for E2 (qstruct.json) and E3b (qstruct_b.json).
usage: python qstruct_summ.py [b] -> results/qstruct_tables.md (or qstruct_b_tables.md)
"""
import json, os, sys

TAG = 'qstruct_b' if 'b' in sys.argv[1:] else 'qstruct'
RUNS = [('2L seed 4', 'runs/L2_s0.05'), ('2L seed 1', 'runs/L2seed1_s0.05'), ('2L seed 2', 'runs/L2seed2_s0.05'), ('3L seed 13', 'runs/L3_s0.05')]
J = {nm: json.load(open(os.path.join(rd, TAG + '.json'))) for nm, rd in RUNS if os.path.exists(os.path.join(rd, TAG + '.json'))}
FULL = {'2L': 18016, '3L': 26432}
f = lambda x: 'n/a' if x is None else (f'{x:.2f}' if x > -10 else (f'{x:.0f}' if x > -1000 else '< −1000')).replace('-', '−')
out = []
out.append('### Stage A: backward build (real maps, restricted attention)\n')
out.append('| model | ε | kept attention edges | needed nodes | layer-1 within-source share | KL bits | proxy R² mean / log-sd / KL |')
out.append('|---|---|---|---|---|---|---|')
for nm, j in J.items():
    for r in j['stageA']:
        out.append(f"| {nm} | {r['tag'].split('=')[1]} | {r['cost']['edges']} | {r['cost']['nodes']} | {r['within_source_L1']:.2f} | {r['KL_bits']:,.0f} | "
                   f"{f(r['proxy']['mean'])} / {f(r['proxy']['logsd'])} / {f(r['proxy']['kl'])} |")
out.append('\n### Single candidates in the full network (proxy R² mean / log-sd / KL)\n')
names = list(J)
cands = []
for j in J.values():
    for k in j.get('single', {}):
        if k not in cands: cands.append(k)
out.append('| candidate | ' + ' | '.join(names) + ' |'); out.append('|---|' + '---|' * len(names))
for c in cands:
    cells = []
    for nm, j in J.items():
        r = j.get('single', {}).get(c); cells.append('–' if r is None else f"{f(r['mean'])} / {f(r['logsd'])} / {f(r['kl'])}")
    out.append(f'| {c} | ' + ' | '.join(cells) + ' |')
out.append('\n### Stage B: greedy path (accepted move, cost after it, proxy R² mean, KL bits)\n')
for nm, j in J.items():
    depth = '3L' if '3L' in nm else '2L'
    out.append(f"**{nm}** (start ε = {j.get('stageB_start_eps')}; network = {FULL[depth]:,} parameters)\n")
    out.append('| step | move | parameters (% of network) | edges | nodes | KL bits | proxy R² mean / log-sd / KL |'); out.append('|---|---|---|---|---|---|---|')
    for i, r in enumerate(j['stageB']):
        mv = r['moves'][-1] if i else 'start'
        out.append(f"| {i} | {mv} | {r['cost']['params']:,} ({100 * r['cost']['params'] / FULL[depth]:.0f}%) | {r['cost']['edges']} | {r['cost']['nodes']} | {r['KL_bits']:,.0f} | "
                   f"{f(r['proxy']['mean'])} / {f(r['proxy']['logsd'])} / {f(r['proxy']['kl'])} |")
    out.append('')
out.append('### Pareto points: K = 64 independent-draw R² on the 2,000 test episodes, scenario agreement (of 9)\n')
out.append('| model | stage, step | parameters (% of network) | edges | KL bits | proxy R² mean | K = 64 R² mean / log-sd / KL | floor (mean) | scenarios agree |')
out.append('|---|---|---|---|---|---|---|---|---|')
for nm, j in J.items():
    depth = '3L' if '3L' in nm else '2L'
    for r in j.get('pareto', []):
        c = r['cost']
        out.append(f"| {nm} | {r['stage']} {r['index']} | {c['params']:,} ({100 * c['params'] / FULL[depth]:.0f}%) | {c['edges']} | {r['KL_bits']:,.0f} | {f(r['proxy_mean'])} | "
                   f"{f(r['R2']['mean'])} / {f(r['R2']['logsd'])} / {f(r['R2']['kl'])} | {j['floor']['mean']:.3f} | {r['scenario_agree']} |")
txt = '\n'.join(out)
open(f'results/{TAG}_tables.md', 'w').write(txt)
print(txt)
