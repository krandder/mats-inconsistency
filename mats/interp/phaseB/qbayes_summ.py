"""Tables for E3a (qbayes.json): whole network described relative to Bayes.
usage: python qbayes_summ.py -> results/qbayes_tables.md
"""
import json, os

RUNS = [('2L seed 4', 'runs/L2_s0.05'), ('2L seed 1', 'runs/L2seed1_s0.05'), ('2L seed 2', 'runs/L2seed2_s0.05'), ('3L seed 13', 'runs/L3_s0.05')]
J = {nm: json.load(open(os.path.join(rd, 'qbayes.json'))) for nm, rd in RUNS if os.path.exists(os.path.join(rd, 'qbayes.json'))}
f = lambda x: (f'{x:.2f}' if x > -10 else (f'{x:.0f}' if x > -1000 else '< −1000')).replace('-', '−')
out = ['### E3a: predictive head = MLP(exact Bayes quantities); K = 64 R² mean / log-sd / KL on 2,000 test episodes\n']
names = list(J)
out.append('| width h (parameters) | ' + ' | '.join(names) + ' |'); out.append('|---|' + '---|' * len(names))
hs = sorted({r['hidden'] for j in J.values() for r in j['rows']})
for h in hs:
    cells = []; cost = ''
    for nm, j in J.items():
        r = next((x for x in j['rows'] if x['hidden'] == h), None)
        if r is None: cells.append('–'); continue
        cost = r['cost']; cells.append(f"{f(r['R2']['mean'])} / {f(r['R2']['logsd'])} / {f(r['R2']['kl'])} (scen {r['scenario_agree']}/9)")
    out.append(f'| {h} ({cost}) | ' + ' | '.join(cells) + ' |')
out.append('| floor | ' + ' | '.join(f"{j['floor']['mean']:.3f} / {j['floor']['logsd']:.3f} / {j['floor']['kl']:.3f}" for j in J.values()) + ' |')
out.append('\n### Scenario outcomes (hit = the pre-registered pattern is present): network vs best-R² head\n')
out.append('| model | ' + ' | '.join(f'S{i}' for i in range(1, 10)) + ' |'); out.append('|---|' + '---|' * 9)
for nm, j in J.items():
    best = max(j['rows'], key=lambda r: r['R2']['mean'])
    net = j['network_scenarios']
    out.append(f'| {nm} network | ' + ' | '.join('hit' if net[f'S{i}']['hit'] else '–' for i in range(1, 10)) + ' |')
    out.append(f"| {nm} head h={best['hidden']} | " + ' | '.join('hit' if best['scenarios'][f'S{i}'] else '–' for i in range(1, 10)) + ' |')
txt = '\n'.join(out); open('results/qbayes_tables.md', 'w').write(txt); print(txt)
