"""mknote_tables.py WORLD [WORLD ...] : the compact tables of NOTE.md, written to parts/tables_*.md from each world's
summary.json (and the post hoc summaries where they exist). EVALUATION CODE. Cells are "world a / world b"."""
import json
import os
import sys

import numpy as np

NS = [64, 256, 1024, 4096, 20480]
worlds = sys.argv[1:]
SS = [json.load(open('out/w%s/summary.json' % w)) for w in worlds]
fn = lambda n: format(n, ',')


def mean(S, arm, lam, n, f, any_r=False):
    rs = [f(m) for m in S['runs'].values() if m.get('arm') == arm and m.get('n') == n and (lam is None or m.get('lam') == lam)
          and (any_r or m.get('r') == 0 or n == 0)]
    return float(np.mean(rs)) if rs else None


def cell(vals, fmt='%.3f'):
    return ' / '.join('-' if v is None else fmt % v for v in vals)


ARMS = [('i', 1.0, '(i) pairs, weight 1'), ('i', 10.0, '(i) pairs, weight 10'), ('iio', 1.0, '(ii-o) other order, 1'),
        ('iio', 10.0, '(ii-o) other order, 10'), ('iis', 0.1, '(ii-s) score, 0.1'), ('iis', 1.0, '(ii-s) score, 1'),
        ('iis', 10.0, '(ii-s) score, 10'), ('iii', 1.0, '(iii) over compute, 1'), ('iiisym', 10.0, '(iii-sym), 10'),
        ('synth', 1.0, "outcomes sampled from M, 1")]
# ---------------- A: excess by n
L = ['| arm, weight | no real outcome | ' + ' | '.join('n = ' + fn(n) for n in NS) + ' |', '|---|' + '---|' * (len(NS) + 1)]
f = lambda m: m['t1_avg']
L.append('| real outcomes only | - | ' + ' | '.join(cell([mean(S, 'data', None, n, f) for S in SS]) for n in NS) + ' |')
for a, lam, lab in [x for x in ARMS if x[0] != 'iiisym']:
    L.append('| %s | %s | ' % (lab, cell([mean(S, a, lam, 0, f) for S in SS])) + ' | '.join(cell([mean(S, a, lam, n, f) for S in SS]) for n in NS) + ' |')
L.append('| one-pass net trained on the n real vectors | - | ' + ' | '.join(cell([mean(S, 'direct', None, n, f, True) for S in SS]) for n in NS) + ' |')
open('parts/tables_A.md', 'w').write('\n'.join(L) + '\n')
# ---------------- A2: real log loss, four rows
L = ['| arm, weight | ' + ' | '.join('n = ' + fn(n) for n in NS) + ' |', '|---|' + '---|' * len(NS)]
f = lambda m: m['real_avg']
L.append('| real outcomes only | ' + ' | '.join(cell([mean(S, 'data', None, n, f) for S in SS]) for n in NS) + ' |')
for a, lam, lab in [ARMS[1], ARMS[5], ARMS[9]]:
    L.append('| %s | ' % lab + ' | '.join(cell([mean(S, a, lam, n, f) for S in SS]) for n in NS) + ' |')
open('parts/tables_A2.md', 'w').write('\n'.join(L) + '\n')
# ---------------- B: gain with paired differences
L = ['| arm, weight | ' + ' | '.join('n = ' + fn(n) for n in NS) + ' |', '|---|' + '---|' * len(NS)]
for a, lam, lab in [x for x in ARMS if x[0] != 'iiisym']:
    row = []
    for n in NS:
        cs = []
        for S in SS:
            b0, a0 = mean(S, 'data', None, n, lambda m: m['t1_avg']), mean(S, a, lam, n, lambda m: m['t1_avg'])
            ds = [S['cmp'][t]['d_excess'] for t, m in S['runs'].items() if m.get('arm') == a and m.get('lam') == lam and m.get('n') == n and t in S['cmp']]
            if a0 is None or not ds:
                cs.append('-')
                continue
            star = '*' if all((d[1] > 0) == (d[2] > 0) and (d[0] > 0) == (ds[0][0] > 0) for d in ds) else ''
            cs.append('%+.0f%%%s' % (100 * (1 - a0 / b0), star))
        row.append(' / '.join(cs))
    L.append('| %s | ' % lab + ' | '.join(row) + ' |')
open('parts/tables_B.md', 'w').write('\n'.join(L) + '\n')
# ---------------- C: cross-order variance
Ks = ['1', '4']
L = ['| predictor | pair mismatch D, K = 1 | D, K = 4 | V_ord, K = 1 | V_ord, K = 4 |', '|---|---|---|---|---|']


def vrow(name, get):
    row = []
    for sec, k1, k2 in (('t2', 'D', 'D_noise'), ('t3', 'V_ord', 'V_noise')):
        for K in Ks:
            a = [get(S, lambda m: m[sec][K][k1]) for S in SS]
            b = [get(S, lambda m: m[sec][K].get(k2)) for S in SS]
            fmt = '%.3f' if sec == 't2' else '%.2f'
            row.append(cell(a, fmt) + ' (' + cell(b, fmt) + ')')
    L.append('| %s | ' % name + ' | '.join(row) + ' |')


fx = lambda nm: (lambda S, f: f(S['runs']['fixed-%s_n0_d0_r0' % nm]))
vrow('one pass of M (M itself)', fx('one'))
vrow("the score's exact weights on the K trajectories", fx('exu'))
for n in (64, 1024, 20480):
    vrow('real outcomes only, n = %s' % fn(n), lambda S, f, n=n: mean(S, 'data', None, n, f))
for a, lam, lab, ns in [ARMS[1] + ((64, 20480),), ARMS[3] + ((20480,),), ARMS[5] + ((20480,),)]:
    for n in ns:
        vrow('%s, n = %s' % (lab, fn(n)), lambda S, f, a=a, lam=lam, n=n: mean(S, a, lam, n, f))
open('parts/tables_C.md', 'w').write('\n'.join(L) + '\n')
# ---------------- D: drag at K = 4
L = ['| arm, weight | n = 64 | n = 1,024 | n = 20,480 |', '|---|---|---|---|']
for a, lam, lab in ARMS[:9] + [ARMS[9]]:
    row = []
    for n in (64, 1024, 20480):
        cs = []
        for S in SS:
            c = S['cmp'].get('%s@%g_n%d_d0_r0' % (a, lam, n), {}).get('drag_K4')
            cs.append('-' if c is None else '%+.3f%s' % (c['adj_better'][0], '*' if (c['adj_better'][1] > 0) == (c['adj_better'][2] > 0) else ''))
        row.append(' / '.join(cs))
    L.append('| %s | ' % lab + ' | '.join(row) + ' |')
open('parts/tables_D.md', 'w').write('\n'.join(L) + '\n')
# ---------------- E: post hoc, real outcomes only on contexts with >= 25 revealed checks
if all(os.path.exists('out/w%s_ph25/summary.json' % w) for w in worlds):
    PP = [json.load(open('out/w%s_ph25/summary.json' % w)) for w in worlds]
    L = ['| arm, weight | n = 1,024: all queries | 0-9 revealed | 10-19 | 20-29 | n = 20,480: all queries | 0-9 | 10-19 | 20-29 |', '|---|' + '---|' * 8]
    rows = [('data', None, 'real outcomes only (restricted)'), ('i', 10.0, '+ (i) pairs, 10'), ('iio', 10.0, '+ (ii-o), 10'),
            ('iis', 1.0, '+ (ii-s) score, 1'), ('iii', 1.0, '+ (iii) over compute, 1'), ('synth', 1.0, '+ outcomes sampled from M')]
    for a, lam, lab in rows:
        row = []
        for n in (1024, 20480):
            for k in ('excess', 'exc_small', 'exc_mid', 'exc_large'):
                row.append(cell([mean(P, a, lam, n, lambda m: m['t1']['4'][k]) for P in PP]))
        L.append('| %s | ' % lab + ' | '.join(row) + ' |')
    row = []
    for n in (1024, 20480):
        for k in ('excess', 'exc_small', 'exc_mid', 'exc_large'):
            row.append(cell([S['runs']['data_n%d_d0_r0' % n]['t1']['4'][k] for S in SS]))
    L.append('| real outcomes on every query (section 2a) | ' + ' | '.join(row) + ' |')
    row = []
    for n in (1024, 20480):
        for k in ('excess', 'exc_small', 'exc_mid', 'exc_large'):
            row.append(cell([S['runs']['fixed-one_n0_d0_r0']['t1']['4'][k] for S in SS]))
    L.append('| one pass of M | ' + ' | '.join(row) + ' |')
    open('parts/tables_E.md', 'w').write('\n'.join(L) + '\n')
# ---------------- references, as one JSON for the text
ref = {w: S['ref'] for w, S in zip(worlds, SS)}
for w, S in zip(worlds, SS):
    ref[w]['bad'] = S['bad']
    ref[w]['n_queries'] = S['n_t1_queries']
    for nm in ('exu', 'meanu'):
        ref[w][nm] = {K: S['runs']['fixed-%s_n0_d0_r0' % nm]['t1'][K]['excess'] for K in Ks}
    ref[w]['one_by_size'] = [S['runs']['fixed-one_n0_d0_r0']['t1']['4'][k] for k in ('exc_small', 'exc_mid', 'exc_large')]
    for n in (64, 20480):
        ref[w]['data%d_by_K_size' % n] = {K: [S['runs']['data_n%d_d0_r0' % n]['t1'][K][k] for k in ('excess', 'exc_small', 'exc_mid', 'exc_large', 'exc_bad', 'exc_good') if k in S['runs']['data_n%d_d0_r0' % n]['t1'][K]] for K in Ks}
    ref[w]['n0'] = {a: mean(S, a, 1.0, 0, lambda m: m['t1_avg']) for a in ('i', 'iio', 'iis', 'iii', 'synth')}
json.dump(ref, open('parts/refs.json', 'w'), indent=1)
print(open('parts/tables_A.md').read())
print(open('parts/tables_B.md').read())
