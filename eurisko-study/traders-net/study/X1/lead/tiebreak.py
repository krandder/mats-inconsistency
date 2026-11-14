#!/usr/bin/env python3
"""X1 lead: resolve w3/w4 outside-2sd rows with the seed-5 tie-break (rule: X1/PREREG.md sections 2 and 5).
Inputs: X1/w3/results/repro.json (seed 4), X1/w4/repro.csv (seed 4), X1/lead/results/repro_s5_w3.json and
X1/lead/results/repro_s5_w4.csv (seed 5; their column is still named seed4).
Output: X1/lead/results/tiebreak.json and tiebreak.csv.  Run: python3 X1/lead/tiebreak.py (from study/)."""
import csv, json, os, datetime, statistics as stt
H = os.path.dirname(os.path.abspath(__file__)); X1 = os.path.dirname(H)
R = os.path.join(H, 'results')

def w3rows(p):
    return {r['id']: dict(s123=r['seeds123'], mean=r['mean'], sd=r['sd'], x=r['seed4'], qual=r.get('qual_holds'))
            for r in json.load(open(p))['rows']}
def w4rows(p):
    out = {}
    for r in csv.DictReader(open(p)):
        x = float(r['seed4']) if r['seed4'] not in ('', 'None') else None
        out[r['quantity']] = dict(s123=[float(r['s1']), float(r['s2']), float(r['s3'])], mean=float(r['mean']), sd=float(r['sd']), x=x, qual=None)
    return out

S4 = {'w3': w3rows(os.path.join(X1, 'w3/results/repro.json')), 'w4': w4rows(os.path.join(X1, 'w4/repro.csv'))}
S5 = {'w3': w3rows(os.path.join(R, 'repro_s5_w3.json')), 'w4': w4rows(os.path.join(R, 'repro_s5_w4.csv'))}

# row -> (worker, quantities, kind, qualitative check on a fresh value (None = use the w3 file's qual_holds))
Q = {
 'M5-1': ('w3', ['R-M5a', 'R-M5b', 'R-M5d.P4', 'R-M5d.P16'], 'quant', None),
 'M5-4': ('w3', ['R-M5c.P4', 'R-M5c.P16'], 'quant', None),
 'M6-2': ('w3', ['R-M6b.B'], 'qual', None),
 'M6-3': ('w3', ['R-M6b.A', 'R-M6b.ES'], 'quant', None),
 'M6-4': ('w3', ['R-M6c.paired', 'R-M6c.cg'], 'quant', None),
 'M6-9': ('w3', ['R-M6d'], 'qual', None),
 'M8-wholerun': ('w4', ['w1 prequential_bce none'], 'quant', None),
 'M8-spearman': ('w4', ['w1 census none spearman_income', 'w1 census none spearman_absag'], 'qual', 'spearman'),
 'M8-stop': ('w4', ['w2 absag - random bce_postshift_switched'], 'qual', lambda v: v > -0.0161),
 'M8-postshift-table': ('w4', ['w2 absag bce_postshift_switched', 'w2 random bce_postshift_switched'], 'quant', None),
 'M9c-thin': ('w4', ['A-FT K64-K1 learn_bce_ex_lfsr'], 'quant', lambda v: v > 0),
 'K1-shift': ('w4', ['w2 S-on10k recovery_steps', 'w2 S-on10k newborn_share_trk_end'], 'quant', None),
}
RANK = {'verified': 0, 'unverified': 1, 'refuted': 2}

def side(q, x):
    if x is None: return None
    return 0 if abs(x - q['mean']) <= 2 * q['sd'] else (1 if x > q['mean'] else -1)

out, flat = {}, []
for row, (w, qs, kind, qf) in Q.items():
    qres = []
    for qn in qs:
        a, b = S4[w].get(qn), S5[w].get(qn)
        if a is None: continue
        x4, x5 = a['x'], (b or {}).get('x')
        s4, s5 = side(a, x4), side(a, x5)
        def qual(x, src):
            if x is None: return None
            if qf == 'spearman':
                return None
            if callable(qf): return bool(qf(x))
            return src.get('qual')
        q4, q5 = qual(x4, a), qual(x5, b or {})
        if x5 is None: st = 'unverified'; why = 'seed 5 missing'
        elif s4 == 0 and s5 == 0: st = 'verified'; why = 'seeds 4 and 5 inside 2 sd'
        elif s4 != 0 and s5 != 0 and s4 == s5:
            if kind == 'quant': st = 'refuted'; why = 'seeds 4 and 5 outside 2 sd on the same side (quoted mean +- sd does not hold)'
            elif q4 is not False and q5 is not False: st = 'unverified'; why = 'both outside, same side; qualitative claim holds; sd understated'
            else: st = 'refuted'; why = 'both outside, same side; qualitative claim fails'
        else: st = 'unverified'; why = 'mixed: one fresh seed inside, one outside' if (s4 == 0) != (s5 == 0) else 'outside on opposite sides'
        pool = a['s123'] + [x for x in (x4, x5) if x is not None]
        r = dict(row=row, worker=w, quantity=qn, kind=kind, mean123=a['mean'], sd123=a['sd'], seed4=x4, seed5=x5,
                 z4=None if x4 is None or not a['sd'] else (x4 - a['mean']) / a['sd'],
                 z5=None if x5 is None or not a['sd'] else (x5 - a['mean']) / a['sd'],
                 qual4=q4, qual5=q5, status=st, why=why, mean5=stt.mean(pool), sd5=stt.stdev(pool) if len(pool) > 1 else None,
                 sd_ratio=(stt.stdev(pool) / a['sd']) if a['sd'] else None)
        qres.append(r); flat.append(r)
    # spearman qualitative: |a.g| > income on each fresh seed
    if qf == 'spearman':
        g = {r['quantity'].split()[-1]: r for r in qres}
        for sd in ('seed4', 'seed5'):
            i, a_ = g['spearman_income'][sd], g['spearman_absag'][sd]
            ok = None if i is None or a_ is None else bool(a_ > i)
            for r in qres: r['qual' + sd[-1]] = ok
        for r in qres:
            if r['status'] == 'refuted' or (r['status'] == 'unverified' and 'same side' in r['why']):
                r['status'] = 'unverified' if (r['qual4'] is not False and r['qual5'] is not False) else 'refuted'
    worst = max(qres, key=lambda r: RANK[r['status']]) if qres else None
    out[row] = dict(worker=w, kind=kind, status=worst['status'] if worst else 'unverified',
                    why=worst['why'] if worst else 'no quantities', quantities=qres)

now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
json.dump(dict(checked_at=now, rule='X1/PREREG.md sections 2 and 5', rows=out), open(os.path.join(R, 'tiebreak.json'), 'w'), indent=1)
with open(os.path.join(R, 'tiebreak.csv'), 'w', newline='') as f:
    k = list(flat[0].keys()); w = csv.DictWriter(f, k); w.writeheader(); w.writerows(flat)
for row, v in out.items():
    print(f"{row:20s} {v['status']:10s} {v['why']}")
    for r in v['quantities']:
        f_ = lambda x: 'NA' if x is None else f'{x:.4g}'
        print(f"    {r['quantity'][:40]:40s} m={f_(r['mean123'])} sd={f_(r['sd123'])} s4={f_(r['seed4'])} (z {f_(r['z4'])}) s5={f_(r['seed5'])} (z {f_(r['z5'])}) q={r['qual4']}/{r['qual5']} 5seed={f_(r['mean5'])}+-{f_(r['sd5'])} -> {r['status']}")
