"""Tables for E5 (qsys.json).
usage: python qsys_summ.py -> results/qsys_tables.md
"""
import json, os

RUNS = [('2L seed 4', 'runs/L2_s0.05'), ('2L seed 1', 'runs/L2seed1_s0.05'), ('2L seed 2', 'runs/L2seed2_s0.05'), ('3L seed 13', 'runs/L3_s0.05')]
J = {nm: json.load(open(os.path.join(rd, 'qsys.json'))) for nm, rd in RUNS if os.path.exists(os.path.join(rd, 'qsys.json'))}
F = ('mean', 'logsd', 'kl')
f = lambda x: (f'{x:.2f}' if x > -10 else (f'{x:.0f}' if x > -1000 else '< −1000')).replace('-', '−')
out = ['### Systematic fraction and learnability ceiling (held-out R² on 2,000 test episodes; fraction = R² / floor)\n']
out.append('| model | floor mean / log-sd / KL | systematic R² mean / log-sd / KL (fraction) | ceiling R² mean / log-sd / KL (fraction) | regressor chosen (mean) |')
out.append('|---|---|---|---|---|')
for nm, j in J.items():
    if 'systematic' not in j: continue
    fl = j['floor']; sy = j['systematic']; ce = j.get('ceiling')
    sys_ = ' / '.join(f"{f(sy[k]['R2'])} ({sy[k]['fraction']:.2f})" for k in F)
    cel = ' / '.join(f"{f(ce[k]['R2'])} ({ce[k]['fraction']:.2f})" for k in F) if ce else '–'
    out.append(f"| {nm} | {fl['mean']:.3f} / {fl['logsd']:.3f} / {fl['kl']:.3f} | {sys_} | {cel} | {sy['mean']['model']} |")
out.append('\n### Descriptions scored on the systematic part S(x): R² against S (raw) / R² of E[description | z] against S (projected, post hoc) / R² against the full deviation\n')
for nm, j in J.items():
    if 'descriptions' not in j: continue
    out.append(f'**{nm}**\n')
    out.append('| description | mean | log-sd | KL |'); out.append('|---|---|---|---|')
    for d, v in j['descriptions'].items():
        out.append(f'| {d} | ' + ' | '.join(f"{f(v[k]['R2_systematic'])} / {f(v[k]['R2_projected'])} / {f(v[k]['R2_full'])}" for k in F) + ' |')
    out.append('')
txt = '\n'.join(out); open('results/qsys_tables.md', 'w').write(txt); print(txt)
