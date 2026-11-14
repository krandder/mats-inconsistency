"""Compact report tables from results/eval_scores.json (medians over the census nets per configuration)."""
import json, numpy as np
d = json.load(open('results/eval_scores.json'))
rows = [r for r in d['rows'] if r['arm'] == 'gap-mid']
cfgs = sorted(set((r['n'], r['L']) for r in rows))
def med(vals):
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else None
f = lambda x: '-' if x is None else (f'{x:.1e}' if (x < 0.01 or x >= 1000) else f'{x:.2g}')
print('| n | L | B&B point-estimate nMSE at 2^16 / 2^20 / 2^22 passes | RQMC : B&B at 2^16 / 2^20 / 2^22 | RB-MC : B&B at 2^22 | exact mass | certified nMSE (final) |')
print('|---|---|---|---|---|---|---|')
for (n, L) in cfgs:
    rr = [r for r in rows if r['n'] == n and r['L'] == L]
    def cell(k, key):
        out = []
        for r in rr:
            c = r['k'].get(str(k)) or r['k'].get(k)
            if c is None: out.append(None); continue
            if key == 'bnb': out.append(c['bnb'])
            elif key == 'rq': out.append(c['rqmc'] / c['bnb'] if (c['rqmc'] and c['bnb']) else None)
            elif key == 'rb': out.append(c['rbmc'] / c['bnb'] if c['bnb'] else None)
        return med(out)
    def unres(k):  # any net whose B&B error is below 10x the truth's own error at this checkpoint
        for r in rr:
            c = r['k'].get(str(k)) or r['k'].get(k)
            if c is not None and not c['resolved']:
                return '*'
        return ''
    b = ' / '.join(f(cell(k, 'bnb')) + unres(k) for k in (16, 20, 22))
    q = ' / '.join(f(cell(k, 'rq')) + unres(k) for k in (16, 20, 22))
    rb = f(cell(22, 'rb')) + unres(22)
    em = med([r['final']['mass_exact'] for r in rr]); ce = med([r['final']['cert'] for r in rr])
    print(f'| {n} | {L} | {b} | {q} | {rb} | {em:.2f} | {f(ce)} |')
