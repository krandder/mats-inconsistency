"""E72 analysis (PREREG-B amendment 80): is the big-miss departure the information loss of block 1's own-source pooled statistic?
Inputs: results/e72/<cohort>_s<seed>.json (e72_run.py: the net's registered rom and type; CB rom of M0-M3), results/e72_patch/<cohort>.json
(e72_patch.py: route contributions c_i, c_ii, c_iii and the unpatched rom). Types: the registered middle-range rom sign (A < 0).
Sets: A = the A-type nets of the registered held-out cohorts (ctrl, fix, own, nocons, h4, b1h4, mag, ownmag, w64ctrl, w128ctrl,
w64mag, w128mag); routing-A = those A nets outside the membership-given cohorts (own, ownmag); B = the B-type nets of the learned-pool
cohorts among them (own and ownmag have none); uni = the uniform-pool calibration cohort. Part 2 sets: the A and B nets of the
standard-forward cohorts (ctrl, fix, h4, b1h4, mag, w64ctrl, w128ctrl, w64mag, w128mag). Descriptive: d3ctrl, d3mag; secondary
(reported separately with the same tests): E71's h2w8, h3w16, h4w16.
Predictions (registered):
  P281  |M3 - net| <= 0.03 (rom_large) in >= 75% of A;
  P282  M3 rom_middle < 0 in >= 75% of A;
  P283  over the A cohorts with >= 3 A nets (at least 8 such cohorts), Spearman of cohort-mean M3 against cohort-mean net rom_large >= 0.6;
  P284  M1 rom_large < net rom_large - 0.03 in >= 75% of routing-A;
  P285  M3 rom_large >= 1.5 x net rom_large in >= 75% of B (nets with net rom_large > 0);
  P286  Part 2, A: |c_i - net| <= 0.03 and |c_iii| <= 0.02 (rom_large) in >= 75%;
  P287  Part 2, B: c_i >= 1.3 x net and c_iii < 0 (rom_large) in >= 75%;
  P288  uni: |M3 - net| <= 0.03 (rom_large) in >= 75%.
Addendum 1 (results/e72_detq/<tag>.json from e72_detq.py; base and detq rom):
  P289  detq - base > 0 (rom_large) in >= 75% of the registered cohorts' nets;
  P290  A: M3 < detq - 0.02 (rom_large) in >= 75%;   P291  B: M3 > detq + 0.05 (rom_large) in >= 75%.
usage: python e72_summ.py -> results/e72.json, results/e72_tables.md
"""
import glob, json, os
import numpy as np

REG = ('ctrl', 'fix', 'own', 'nocons', 'h4', 'b1h4', 'mag', 'ownmag', 'w64ctrl', 'w128ctrl', 'w64mag', 'w128mag')
MEMB = ('own', 'ownmag')
STD = ('ctrl', 'fix', 'h4', 'b1h4', 'mag', 'w64ctrl', 'w128ctrl', 'w64mag', 'w128mag')
DESC = ('d3ctrl', 'd3mag'); SEC = ('h2w8', 'h3w16', 'h4w16')


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def frac(xs):
    return [int(sum(xs)), len(xs), float(np.mean(xs)) if xs else None]


def tests(P1, P2, cohorts, memb, std):
    pr = {}
    A = [o for c in cohorts for o in P1.get(c, []) if o['type'] == 'A' and o.get('M3')]
    RA = [o for o in A if o['cohort'] not in memb]
    B = [o for c in cohorts if c not in memb for o in P1.get(c, []) if o['type'] == 'B' and o.get('M3') and o['net_rom_large'] > 0]
    f = frac([abs(o['M3']['large'] - o['net_rom_large']) <= 0.03 for o in A]); pr['P281'] = dict(n=f, holds=bool(A) and f[2] >= 0.75)
    f = frac([o['M3']['middle'] < 0 for o in A]); pr['P282'] = dict(n=f, holds=bool(A) and f[2] >= 0.75)
    cm = [(c, np.mean([o['M3']['large'] for o in A if o['cohort'] == c]), np.mean([o['net_rom_large'] for o in A if o['cohort'] == c]))
          for c in cohorts if sum(o['cohort'] == c for o in A) >= 3]
    rho = spearman([x[1] for x in cm], [x[2] for x in cm]) if len(cm) >= 3 else None
    pr['P283'] = dict(cohorts=[[c, float(a), float(b)] for c, a, b in cm], spearman=rho, holds=bool(len(cm) >= 8 and rho is not None and rho >= 0.6))
    f = frac([o['M1']['large'] < o['net_rom_large'] - 0.03 for o in RA if o.get('M1')]); pr['P284'] = dict(n=f, holds=bool(RA) and f[2] >= 0.75)
    f = frac([o['M3']['large'] >= 1.5 * o['net_rom_large'] for o in B]); pr['P285'] = dict(n=f, holds=bool(B) and f[2] >= 0.75)
    pa = [r for c in std for r in P2.get(c, {}).values() if r['type'] == 'A']; pb = [r for c in std for r in P2.get(c, {}).values() if r['type'] == 'B']
    f = frac([abs(r['c_i_large'] - r['net_rom_large']) <= 0.03 and abs(r['c_iii_large']) <= 0.02 for r in pa]); pr['P286'] = dict(n=f, holds=bool(pa) and f[2] >= 0.75)
    f = frac([r['c_i_large'] >= 1.3 * r['net_rom_large'] and r['c_iii_large'] < 0 for r in pb]); pr['P287'] = dict(n=f, holds=bool(pb) and f[2] >= 0.75)
    return pr


def tests_detq(P1, D, cohorts, memb):
    pr = {}
    nets = [o for c in cohorts for o in P1.get(c, []) if o['tag'] in D]
    f = frac([D[o['tag']]['detq']['large'] - D[o['tag']]['base']['large'] > 0 for o in nets]); pr['P289'] = dict(n=f, holds=bool(nets) and f[2] >= 0.75)
    A = [o for o in nets if o['type'] == 'A' and o.get('M3')]
    B = [o for o in nets if o['type'] == 'B' and o.get('M3') and o['cohort'] not in memb and o['net_rom_large'] > 0]
    f = frac([o['M3']['large'] < D[o['tag']]['detq']['large'] - 0.02 for o in A]); pr['P290'] = dict(n=f, holds=bool(A) and f[2] >= 0.75)
    f = frac([o['M3']['large'] > D[o['tag']]['detq']['large'] + 0.05 for o in B]); pr['P291'] = dict(n=f, holds=bool(B) and f[2] >= 0.75)
    return pr


def main():
    P1 = {}
    for p in sorted(glob.glob('results/e72/*.json')):
        o = json.load(open(p)); o['cohort'] = o['tag'].rsplit('_s', 1)[0]; P1.setdefault(o['cohort'], []).append(o)
    P2 = {}
    for p in sorted(glob.glob('results/e72_patch/*.json')):
        c = os.path.basename(p)[:-5]; d = json.load(open(p))
        for t, r in d.items():
            r['type'] = 'A' if r['net_rom_middle'] < 0 else 'B'
        P2[c] = d
    D = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob('results/e72_detq/*.json')}
    res = dict(predictions=tests(P1, P2, REG, MEMB, STD), secondary=tests(P1, P2, SEC, (), SEC) if any(c in P1 for c in SEC) else None)
    res['predictions'].update(tests_detq(P1, D, REG, MEMB))
    if res['secondary']:
        res['secondary'].update(tests_detq(P1, D, SEC, ()))
    uni = [o for o in P1.get('uni', []) if o.get('M3')]
    f = frac([abs(o['M3']['large'] - o['net_rom_large']) <= 0.03 for o in uni]); res['predictions']['P288'] = dict(n=f, holds=bool(uni) and f[2] >= 0.75)
    rows = ['# E72 tables (generated by e72_summ.py)', '', '## Part 1: CB readers of the own-source pool, cohort means by type (rom_large / rom_middle)', '',
            '| cohort | type | nets | net | M0 | M1 | M2 | M3 | M3 - net (large): median [min, max] | c | pooling heads |', '|---|---|---|---|---|---|---|---|---|---|---|']
    mm = lambda L, k, r: np.mean([o[k][r] for o in L if o.get(k)]) if any(o.get(k) for o in L) else float('nan')
    for c in REG + ('uni',) + DESC + SEC:
        for t in ('A', 'B'):
            L = [o for o in P1.get(c, []) if o['type'] == t]
            if not L:
                continue
            d = [o['M3']['large'] - o['net_rom_large'] for o in L if o.get('M3')]
            rows.append('| %s | %s | %d | %.3f / %+.3f | %.3f / %+.3f | %.3f / %+.3f | %.3f / %+.3f | %.3f / %+.3f | %s | %.2f | %s |' % (
                c, t, len(L), np.mean([o['net_rom_large'] for o in L]), np.mean([o['net_rom_middle'] for o in L]),
                mm(L, 'M0', 'large'), mm(L, 'M0', 'middle'), mm(L, 'M1', 'large'), mm(L, 'M1', 'middle'), mm(L, 'M2', 'large'), mm(L, 'M2', 'middle'),
                mm(L, 'M3', 'large'), mm(L, 'M3', 'middle'), ('%+.3f [%+.3f, %+.3f]' % (np.median(d), min(d), max(d))) if d else '-',
                np.median([o['c'] for o in L]), '/'.join(str(len(o['pool'])) for o in L)))
    rows += ['', '## Part 2: route contributions (rom_large; medians by cohort and type)', '', '| cohort | type | nets | net | rom (recomputed) | c_i | c_ii | c_iii | detach_iii |', '|---|---|---|---|---|---|---|---|---|']
    for c in STD + DESC + SEC:
        for t in ('A', 'B'):
            L = [r for r in P2.get(c, {}).values() if r['type'] == t]
            if L:
                md = lambda k: np.median([r[k] for r in L])
                rows.append('| %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (c, t, len(L), md('net_rom_large'), md('rom_large'), md('c_i_large'), md('c_ii_large'), md('c_iii_large'), md('detach_iii_large')))
    rows += ['', '## Addendum 1: decomposition rom_base = M3 + (detq - M3) + (base - detq), rom_large means by cohort and type', '',
             '| cohort | type | nets | rom_base | M3 | detq - M3 | base - detq | detq |', '|---|---|---|---|---|---|---|---|']
    for c in REG + ('uni',) + DESC + SEC:
        for t in ('A', 'B'):
            L = [o for o in P1.get(c, []) if o['type'] == t and o.get('M3') and o['tag'] in D]
            if L:
                b = np.mean([D[o['tag']]['base']['large'] for o in L]); dq = np.mean([D[o['tag']]['detq']['large'] for o in L]); m3 = np.mean([o['M3']['large'] for o in L])
                rows.append('| %s | %s | %d | %.3f | %.3f | %+.3f | %+.3f | %.3f |' % (c, t, len(L), b, m3, dq - m3, b - dq, dq))
    rows += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in res['predictions'].items()]
    if res['secondary']:
        rows += ['', '## Secondary set (E71 arms), same tests', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in res['secondary'].items()]
    json.dump(res, open('results/e72.json', 'w'), indent=1, default=float)
    open('results/e72_tables.md', 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
