"""Score the registered B&B evaluation (PREREG P1-P8) and tabulate. Writes results/eval_tables.md and
results/eval_scores.json. Run after the queue."""
import json, math, os, glob
import numpy as np

R = os.path.join(os.path.dirname(__file__), '..')
census = json.load(open(os.path.join(R, 'runs', 'census.json')))
KS = list(range(10, 23, 2))  # scored budgets 2^k passes


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def rqmc_at(t, N):
    ms = sorted(int(m) for m in t['rqmc'])
    if N > 2 ** ms[-1] * 1.01:
        return None  # never extrapolate (a 1% overshoot of the last measured N, from batch granularity, is clamped)
    N = min(N, 2 ** ms[-1])
    xs = np.array([float(m) for m in ms]); ys = np.log(np.array([t['rqmc'][str(m)]['nmse'] for m in ms]))
    return float(np.exp(np.interp(math.log2(N), xs, ys)))


def rec_at(b, N):
    """The B&B record taken at checkpoint N (the first state whose cost reached N passes; its own cost r['passes'] is
    used for every comparison, so all comparisons are at equal cost)."""
    recs = [r for r in b['records'] if r['checkpoint'] == N]
    return recs[0] if recs else None


def bnb_at(b, N, t, key='point'):
    r = rec_at(b, N)
    if r is None:
        return None
    tr = np.array(t['truth']); V = t['V']
    if key == 'cert':
        return float(np.sum(((np.array(r['ub']) - np.array(r['lb'])) / 2) ** 2) / V)
    return float(np.sum((np.array(r[key]) - tr) ** 2) / V)


def reached(b):
    return max(r['passes'] for r in b['records'])


rows = []
for key, c in census.items():
    n, L = c['n'], c['L']
    for s in c['picked']:
        t = load(os.path.join(R, 'runs', f'truth_n{n}_L{L}_seed{s}.json'))
        if t is None:
            continue
        for arm in ('gap-mid', 'instab-mid'):
            b = load(os.path.join(R, 'runs', f'bnb_{arm}_n{n}_L{L}_seed{s}.json'))
            if b is None:
                continue
            row = dict(n=n, L=L, seed=s, arm=arm, truth_nmse=t['truth_nmse'], rho_rb=t['rho_rb'],
                       reached=reached(b), stop=b['stop'], k={})
            for k in KS:
                r = rec_at(b, 2 ** k)
                if r is None:
                    continue
                P = r['passes']  # actual cost of this state
                e = bnb_at(b, 2 ** k, t)
                row['k'][k] = dict(passes=P, bnb=e, cert=bnb_at(b, 2 ** k, t, 'cert'), mid=bnb_at(b, 2 ** k, t, 'mid'),
                                   rqmc=rqmc_at(t, P), rbmc=t['rho_rb'] / P, mc=1.0 / P,
                                   rqmc_2xcost=rqmc_at(t, 2 * P),  # RQMC at twice the charged cost (FLOP-undercount sensitivity)
                                   resolved=(e is not None and e >= 10 * t['truth_nmse']))
            fin = [r for r in b['records'] if r['checkpoint'] == 'final'][0]
            row['final'] = dict(passes=fin['passes'], leaves=fin['leaves'], mass_exact=fin['mass_exact'],
                                cert=float(np.sum(((np.array(fin['ub']) - np.array(fin['lb'])) / 2) ** 2) / t['V']),
                                point=float(np.sum((np.array(fin['point']) - np.array(t['truth'])) ** 2) / t['V']))
            qd = {int(m): v['nmse'] for m, v in t['qmc_det'].items()}
            rq = {int(m): v['nmse'] for m, v in t['rqmc'].items()}
            row['qmc_ratio'] = {m: qd[m] / rq[m] for m in qd if m in rq and rq[m] > 0 and qd[m] > 0}
            rows.append(row)


def ratio(cell):  # RQMC nMSE / B&B nMSE (> 1: B&B wins)
    if cell['bnb'] is None or cell['rqmc'] is None or cell['bnb'] <= 0:
        return None
    return cell['rqmc'] / cell['bnb']


scores = {}
prim = [r for r in rows if r['arm'] == 'gap-mid']
# P1
p1 = [r for r in prim if r['n'] == 3 and r['L'] in (4, 8)]
win = 0
for r in p1:
    ok = any((ratio(r['k'][k]) or 0) >= 3 and r['k'][k]['resolved'] for k in r['k'] if 16 <= k <= 22)
    win += ok
scores['P1'] = dict(nets=len(p1), wins=win, hit=(win >= 4 and len(p1) == 6))
# sensitivity (post hoc, astra's FLOP-undercount finding): the same wins if the B&B cost twice its charge
win2 = 0
for r in p1:
    win2 += any(r['k'][k]['bnb'] and r['k'][k]['rqmc_2xcost'] and r['k'][k]['rqmc_2xcost'] / r['k'][k]['bnb'] >= 3
                and r['k'][k]['resolved'] for k in r['k'] if 16 <= k <= 22)
scores['P1_cost_x2_sensitivity'] = dict(wins=win2)
# P2
p2 = [r for r in prim if r['n'] == 4]
cnt = 0
for r in p2:
    ks = [k for k in r['k'] if 12 <= k <= 22 and ratio(r['k'][k]) is not None]
    ok = len(ks) > 0 and all(ratio(r['k'][k]) <= 0.1 for k in ks)
    cnt += ok
scores['P2'] = dict(nets=len(p2), satisfied=cnt, hit=(cnt >= 9))
# P3
p3 = [r for r in prim if r['n'] in (6, 8)]
cnt = 0
for r in p3:
    ks = [k for k in r['k'] if 14 <= k <= 22 and ratio(r['k'][k]) is not None]
    cnt += (len(ks) > 0 and all(ratio(r['k'][k]) <= 0.01 for k in ks))
scores['P3'] = dict(nets=len(p3), satisfied=cnt, hit=(cnt == len(p3) and len(p3) == 30))
# P4
p4 = [r for r in prim if r['n'] in (16, 32)]
cnt = 0; certok = 0
for r in p4:
    ks = [k for k in r['k'] if k >= 16 and ratio(r['k'][k]) is not None]
    if ks:
        cnt += ratio(r['k'][max(ks)]) <= 1e-3
    certok += all(r['k'][k]['cert'] >= 1 for k in r['k'])
scores['P4'] = dict(nets=len(p4), loses_1000x=cnt, cert_vacuous=certok, hit=(cnt == len(p4) == certok == 18))
# P5
c3 = [r for r in prim if r['n'] == 3 and r['L'] <= 8]
c6 = [r for r in prim if r['n'] >= 6]
ok3 = sum(any(r['k'][k]['cert'] <= 1e-6 for k in r['k']) or r['final']['cert'] <= 1e-6 for r in c3)
ok6 = sum(any(r['k'][k]['cert'] <= 1e-6 for k in r['k']) or r['final']['cert'] <= 1e-6 for r in c6)
scores['P5'] = dict(n3_nets=len(c3), n3_cert=ok3, n6plus_cert=ok6, hit=(ok3 >= 4 and ok6 == 0))
# P6
p6 = {}
for L in (8, 16):
    med = []
    for n in (3, 4, 6, 8):
        v = [r['k'][20]['bnb'] for r in prim if r['n'] == n and r['L'] == L and 20 in r['k'] and r['k'][20]['bnb'] is not None]
        med.append(float(np.median(v)) if v else None)
    mono = all(a is not None and b is not None and b > a for a, b in zip(med, med[1:]))
    big = med[0] is not None and med[-1] is not None and med[-1] >= 100 * med[0]
    p6[L] = dict(medians=med, monotone=mono, factor_100=big)
scores['P6'] = dict(by_L=p6, hit=all(v['monotone'] and v['factor_100'] for v in p6.values()))
# P7
p7 = {}
for m in (10, 14, 18):
    v = [r['qmc_ratio'][m] for r in prim if r['n'] <= 8 and m in r['qmc_ratio']]
    p7[m] = float(np.exp(np.mean(np.log(v)))) if v else None
scores['P7'] = dict(geomean_qmcdet_over_rqmc=p7, hit=all(g is not None and 1 / 3 <= g <= 3 for g in p7.values()))
# P8
cnt = 0
for r in p3:
    ks = [k for k in r['k'] if r['k'][k]['bnb'] is not None]
    cnt += (len(ks) > 0 and all(r['k'][k]['bnb'] > r['k'][k]['rbmc'] for k in ks))
scores['P8'] = dict(nets=len(p3), loses_to_rbmc_everywhere=cnt, hit=(cnt >= 2 * len(p3) / 3))

# cost to precision (post hoc descriptive): smallest checkpoint where the B&B point nMSE <= thr, and RQMC's N
def first_budget(b, t, thr):
    tr = np.array(t['truth']); V = t['V']
    for r in b['records']:
        if r['checkpoint'] == 'final':
            continue
        if np.sum((np.array(r['point']) - tr) ** 2) / V <= thr:
            return r['passes']
    return None
def rqmc_budget(t, thr):
    ms = sorted(int(m) for m in t['rqmc'])
    for m in ms:
        if t['rqmc'][str(m)]['nmse'] <= thr:
            return 2 ** m
    return None
ctp = []
for key, c in census.items():
    n, L = c['n'], c['L']
    for s in c['picked']:
        t = load(os.path.join(R, 'runs', f'truth_n{n}_L{L}_seed{s}.json'))
        b = load(os.path.join(R, 'runs', f'bnb_gap-mid_n{n}_L{L}_seed{s}.json'))
        if t is None or b is None:
            continue
        ctp.append(dict(n=n, L=L, seed=s, **{f'bnb_{thr:g}': first_budget(b, t, thr) for thr in (1e-3, 1e-4, 1e-6)},
                        **{f'rqmc_{thr:g}': rqmc_budget(t, thr) for thr in (1e-3, 1e-4, 1e-6)}))
json.dump(dict(scores=scores, rows=rows, cost_to_precision=ctp), open(os.path.join(R, 'results', 'eval_scores.json'), 'w'), indent=1)

# tables: median over nets per (n, L, arm) at each scored budget: B&B nMSE, ratio RQMC/B&B, ratio RBMC/B&B
lines = ['# B&B evaluation tables (medians over the 3 census nets per configuration)', '',
         'Columns per budget 2^k passes: B&B point nMSE / ratio RQMC:B&B (>1 = B&B better) / ratio RB-MC:B&B.', '']
for arm in ('gap-mid', 'instab-mid'):
    lines.append(f'## arm {arm}')
    lines.append('| n | L | ' + ' | '.join(f'2^{k}' for k in KS) + ' | final exact mass | final cert nMSE | truth nMSE |')
    lines.append('|' + '---|' * (len(KS) + 5))
    for key in sorted(census, key=lambda x: (census[x]['n'], census[x]['L'])):
        n, L = census[key]['n'], census[key]['L']
        rr = [r for r in rows if r['n'] == n and r['L'] == L and r['arm'] == arm]
        if not rr:
            continue
        cells = []
        for k in KS:
            vals = [r['k'][k] for r in rr if k in r['k']]
            if not vals:
                cells.append('-'); continue
            e = np.median([v['bnb'] for v in vals])
            q = [ratio(v) for v in vals if ratio(v) is not None]
            m = np.median([v['rbmc'] / v['bnb'] for v in vals])
            cells.append(f'{e:.1e} / {np.median(q):.2g} / {m:.2g}' if q else f'{e:.1e} / - / {m:.2g}')
        fe = np.median([r['final']['mass_exact'] for r in rr]); fc = np.median([r['final']['cert'] for r in rr])
        tn = np.median([r['truth_nmse'] for r in rr])
        lines.append(f'| {n} | {L} | ' + ' | '.join(cells) + f' | {fe:.3f} | {fc:.1e} | {tn:.1e} |')
    lines.append('')
lines.append('## Cost to precision (passes; median over nets; None = not reached within the run)')
lines.append('| n | L | B&B to 1e-3 | B&B to 1e-4 | B&B to 1e-6 | RQMC to 1e-3 | RQMC to 1e-4 | RQMC to 1e-6 |')
lines.append('|---|---|---|---|---|---|---|---|')
def med(v):
    v2 = [x for x in v if x is not None]
    return f'{np.median(v2):.2g} ({len(v2)}/{len(v)})' if v2 else f'- (0/{len(v)})'
for key in sorted(census, key=lambda x: (census[x]['n'], census[x]['L'])):
    n, L = census[key]['n'], census[key]['L']
    cc = [c for c in ctp if c['n'] == n and c['L'] == L]
    if cc:
        lines.append(f'| {n} | {L} | ' + ' | '.join(med([c[f'{a}_{thr:g}'] for c in cc]) for a in ('bnb', 'rqmc') for thr in (1e-3, 1e-4, 1e-6)) + ' |')
lines.append('')
lines.append('## Scores')
for k, v in scores.items():
    lines.append(f'- {k}: {json.dumps(v)}')
open(os.path.join(R, 'results', 'eval_tables.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines[-12:]))
