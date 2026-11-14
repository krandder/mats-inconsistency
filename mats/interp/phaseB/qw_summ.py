"""Tables for the question-weighted fits (amendment 7) from runs/*/qw.json.
usage: python qw_summ.py [smoke] -> results/qw_tables.md
"""
import json, os, sys
import numpy as np

RUNS = [('2L seed 4', 'runs/L2_s0.05'), ('2L seed 1', 'runs/L2seed1_s0.05'), ('2L seed 2', 'runs/L2seed2_s0.05'), ('3L seed 13', 'runs/L3_s0.05')]
tag = 'qw_smoke.json' if 'smoke' in sys.argv else 'qw.json'
J = {nm: json.load(open(os.path.join(rd, tag))) for nm, rd in RUNS if os.path.exists(os.path.join(rd, tag))}
f = lambda x: 'n/a' if x is None else (f'{x:.2f}' if x > -10 else (f'{x:.0f}' if x > -1000 else '< -1000'))
out = []
out.append('### K = 64 noise floor and "network = Bayes"\n')
out.append('| model | floor R² mean / log-sd / KL | network = Bayes R² mean / log-sd / KL | dev sd mean / log-sd / KL |')
out.append('|---|---|---|---|')
for nm, j in J.items():
    nf, z, sd = j['noise_floor'], j['zero'], j['dev_sd']
    out.append(f"| {nm} | {nf['mean']:.3f} / {nf['logsd']:.3f} / {nf['kl']:.3f} | {z['mean']:.2f} / {z['logsd']:.2f} / {z['kl']:.2f} | {sd['mean']:.3f} / {sd['logsd']:.3f} / {sd['kl']:.3f} |")


def row(j, name):
    r = [x for x in j['rows'] if x['config'] == name]
    return r[0] if r else None


def sites_of(j):
    return [s for s in j['fits']]


out.append('\n### Single-site surrogates: R²(mean), closed form / unweighted / weighted (cost as % of the site)\n')
names = list(J)
out.append('| site : family | ' + ' | '.join(names) + ' |')
out.append('|---|' + '---|' * len(names))
allsites = []
for j in J.values():
    for s in sites_of(j):
        if s not in allsites: allsites.append(s)
for s in allsites:
    fams = [('LR2', 'LR4', 'LR8') if s[0] != 'a' else ('fp2', 'fp4', 'fpfull')][0]
    for fam in fams:
        cells = []
        for nm, j in J.items():
            if s not in j['fits']: cells.append('–'); continue
            rs = [row(j, f'{s}:{fam}:{k}') for k in 'cuw']
            if any(r is None for r in rs): cells.append('–'); continue
            ex = j['exact_cost']['emb' if s == 'emb' else s[0]]
            cells.append(' / '.join(f(r['R2_mean']) for r in rs) + f" ({100 * rs[0]['cost'] / ex:.0f}%)")
        out.append(f'| {s} : {fam} | ' + ' | '.join(cells) + ' |')

out.append('\n### Same, R²(log-sd) and R²(KL) for the weighted fits at the ≤ 25% level (rank 4; attention pattern + rank 2)\n')
out.append('| site | ' + ' | '.join(names) + ' |')
out.append('|---|' + '---|' * len(names))
for s in allsites:
    fam = 'fp2' if s[0] == 'a' else 'LR4'
    cells = []
    for nm, j in J.items():
        r = row(j, f'{s}:{fam}:w')
        cells.append('–' if r is None else f"{f(r['R2_logsd'])} / {f(r['R2_kl'])}")
    out.append(f'| {s} ({fam}) | ' + ' | '.join(cells) + ' |')

out.append('\n### Fit diagnostics (held-out 1,024 fit episodes): J-weighted error E||J(g−f)||² and site KL (nats), rank 4 / pattern + rank 2\n')
out.append('| site | ' + ' | '.join(names) + ' |')
out.append('|---|' + '---|' * len(names))
for s in allsites:
    lev = '2' if s[0] == 'a' else '4'
    cells = []
    for nm, j in J.items():
        F = j['fits'].get(s)
        if F is None: cells.append('–'); continue
        cells.append(' / '.join(f"{F[f'{lev}:{k}']['Jerr']:.2g}" for k in 'cuw') + ' ; KL ' + ' / '.join(f"{F[f'{lev}:{k}']['KL']:.0f}" for k in 'cuw'))
    out.append(f'| {s} | ' + ' | '.join(cells) + ' |')

out.append('\n### Whole chains (every site replaced): R² mean / log-sd / KL\n')
out.append('| chain | parameters (2L / 3L; network sites total 18,016 / 26,432) | ' + ' | '.join(names) + ' |')
out.append('|---|---|' + '---|' * len(names))
for ctag, lab in (('le25', '≤ 25% (emb r4, attn pattern + r2, MLP r4)'), ('r8', 'rank 8 (attn pattern + r4)')):
    for kind in ('closed', 'unweighted', 'weighted', 'e2e'):
        nm0 = f'chain-{ctag}-{kind}'; cells = []; costs = []
        for nm, j in J.items():
            r = row(j, nm0)
            if r is None: cells.append('–'); continue
            cells.append(f"{f(r['R2_mean'])} / {f(r['R2_logsd'])} / {f(r['R2_kl'])}")
            if r['cost'] not in costs: costs.append(r['cost'])
        cost = ' / '.join(str(c) for c in costs)
        out.append(f'| {lab}: {kind} | {cost} | ' + ' | '.join(cells) + ' |')

out.append('\n### Per-source DAG (restricted attention): real weights renormalised / refitted unweighted / refitted weighted, R² mean / log-sd / KL\n')
out.append('| variant | ' + ' | '.join(names) + ' |')
out.append('|---|' + '---|' * len(names))
for v in ('D1', 'D1+lat', 'D1-L2lat'):
    for kind in ('real', 'unweighted', 'weighted'):
        cells = []; any_ = False
        for nm, j in J.items():
            r = row(j, f'DAG-{v}-{kind}')
            if r is None: cells.append('–'); continue
            any_ = True; cells.append(f"{f(r['R2_mean'])} / {f(r['R2_logsd'])} / {f(r['R2_kl'])}")
        if any_: out.append(f'| {v}: {kind} | ' + ' | '.join(cells) + ' |')

# predictions
out.append('\n### Predictions (amendment 7)\n')
p39 = all(j['noise_floor']['mean'] >= 0.95 and j['noise_floor']['logsd'] >= 0.98 and j['noise_floor']['kl'] >= 0.98 for j in J.values())
out.append(f"- P39 (floor ≥ 0.95 mean, ≥ 0.98 log-sd and KL in every model): {'HOLDS' if p39 else 'FAILS'} — " +
           '; '.join(f"{nm} {j['noise_floor']['mean']:.3f} / {j['noise_floor']['logsd']:.3f} / {j['noise_floor']['kl']:.3f}" for nm, j in J.items()))
hits, tot, att = 0, 0, []
for nm, j in J.items():
    for s in sites_of(j):
        if s[0] == 'a':
            r = row(j, f'{s}:fp2:w')
            if r: att.append(r['R2_mean'])
        else:
            u, w = row(j, f'{s}:LR4:u'), row(j, f'{s}:LR4:w')
            if u and w and u['R2_mean'] < 0:
                tot += 1; hits += w['R2_mean'] >= 0.5
out.append(f"- P40 (rank-4 weighted ≥ 0.5 wherever unweighted < 0; attention pattern + rank 2 in (0, 0.5)): emb/MLP {hits} of {tot}; attention R² {', '.join(f(a) for a in att)}")
p41 = []
for nm, j in J.items():
    e, w = row(j, 'chain-le25-e2e'), row(j, 'chain-le25-weighted')
    if e and w: p41.append(f"{nm} e2e {f(e['R2_mean'])} vs weighted {f(w['R2_mean'])}")
out.append('- P41 (e2e ≤ 25% chain ≥ 0.5 and beats the weighted chain): ' + '; '.join(p41))
p42 = []
for nm, j in J.items():
    r = row(j, 'DAG-D1-weighted')
    if not r: continue
    p42.append(f"{nm} D1 weighted {f(r['R2_mean'])} (floor {j['noise_floor']['mean']:.2f})")
    r2 = row(j, 'DAG-D1+lat-weighted')
    if r2: p42[-1] += f", D1+lat weighted {f(r2['R2_mean'])}"
out.append('- P42 (weighted D1 ≥ floor − 0.1 in own-source seeds; seed 2 D1+lat ≥ 0.5): ' + '; '.join(p42))
txt = '\n'.join(out)
open(os.path.join('results', 'qw_tables_smoke.md' if 'smoke' in sys.argv else 'qw_tables.md'), 'w').write(txt)
print(txt)
