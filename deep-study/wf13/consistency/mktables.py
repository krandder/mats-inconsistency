"""mktables.py OUTROOT WORLD [WORLD ...] : tables (markdown, OUTROOT/tables.md, OUTROOT/tables.json) and figures from
each world's summary.json (written by creport.py). EVALUATION CODE. Cells are "world a / world b / ..."."""
import json
import os
import sys

import numpy as np

NS = [64, 256, 1024, 4096, 20480]


def cell(vals, fmt='%.3f'):
    return ' / '.join('-' if v is None else fmt % v for v in vals)


def runs_of(S, arm, lam, n):
    return [m for m in S['runs'].values() if m.get('arm') == arm and m.get('n') == n and (lam is None or m.get('lam') == lam)]


def mean(S, arm, lam, n, f, r0=True):
    rs = [m for m in runs_of(S, arm, lam, n) if (not r0 or m.get('r') == 0 or n == 0)]
    return float(np.mean([f(m) for m in rs])) if rs else None


if __name__ == '__main__':
    root, worlds = sys.argv[1], sys.argv[2:]
    SS = [json.load(open('%s/w%s/summary.json' % (root, w))) for w in worlds]
    S0 = SS[0]
    Ks = sorted(int(k) for k in next(m for m in S0['runs'].values() if m.get('arm') == 'data')['t1'])
    arms = sorted({(m['arm'], m['lam']) for S in SS for m in S['runs'].values() if m.get('arm') not in (None, 'direct', 'data') and m.get('n')},
                  key=lambda a: (['i', 'iio', 'iis', 'iii', 'iiisym', 'synth'].index(a[0]), a[1]))
    L, J = [], {}
    add = L.append
    add('World(s): %s. Cells: %s. Mean over the matched runs (d = 0, 1; r = 0).\n' % (', '.join(worlds), ' / '.join('w' + w for w in worlds)))
    # ---- Table 1: excess against the exact posterior, by n
    for key, title, f in (('excess', 'Excess log loss against the exact posterior (nats; mean over K = %s)' % Ks, lambda m: m['t1_avg']),
                          ('real', 'Log loss on the held-out real outcomes (nats)', lambda m: m['real_avg'])):
        add('**%s**\n' % title)
        add('| arm @ weight | n = 0 (term alone) | ' + ' | '.join('n = %s' % format(n, ',') for n in NS) + ' |')
        add('|---|' + '---|' * (len(NS) + 1))
        add('| real outcomes only | - | ' + ' | '.join(cell([mean(S, 'data', None, n, f) for S in SS]) for n in NS) + ' |')
        J[key] = {'data': {n: [mean(S, 'data', None, n, f) for S in SS] for n in NS}}
        for arm, lam in arms:
            row = [cell([mean(S, arm, lam, 0, f) for S in SS])] + [cell([mean(S, arm, lam, n, f) for S in SS]) for n in NS]
            if all(set(c) <= set('- /') for c in row):
                continue
            add('| %s @ %g | ' % (arm, lam) + ' | '.join(row) + ' |')
            J[key]['%s@%g' % (arm, lam)] = {n: [mean(S, arm, lam, n, f) for S in SS] for n in [0] + NS}
        add('| one-pass net trained on the n real vectors | - | ' + ' | '.join(cell([mean(S, 'direct', None, n, f, r0=False) for S in SS]) for n in NS) + ' |')
        if key == 'excess':
            fx = lambda nm: cell([S['runs'].get('fixed-%s_n0_d0_r0' % nm, {}).get('t1_avg') for S in SS])
            add('\nReferences (no training on these real vectors): one pass of M (size M, the network used) %s; size S %s; size L %s; the score with exact weights on the same trajectories %s; plain average %s; the score T, exact %s.\n'
                % (cell([S['ref'].get('p_one') for S in SS]), cell([S['ref'].get('p_one_S') for S in SS]), cell([S['ref'].get('p_one_L') for S in SS]),
                   fx('exu'), fx('meanu'), cell([S['ref'].get('score_T') for S in SS], '%.4f')))
            for K in Ks:
                add('exu at K = %d: %s; data-only n = 20,480 at K = %d: %s.' % (K, cell([S['runs'].get('fixed-exu_n0_d0_r0', {}).get('t1', {}).get(str(K), {}).get('excess') for S in SS]),
                                                                                   K, cell([mean(S, 'data', None, 20480, lambda m: m['t1'][str(K)]['excess']) for S in SS])))
        else:
            add('\nThe exact posterior itself scores %s.\n' % cell([S['ref'].get('real_truth') for S in SS]))
    # ---- Table 2: gain and paired differences
    add('**Gain G = 1 - excess(arm) / excess(real outcomes only), and the paired difference of excess with its 95 % interval (arm minus data-only; mean over the two matched runs; negative = better)**\n')
    add('| arm @ weight | ' + ' | '.join('n = %s' % format(n, ',') for n in NS) + ' |')
    add('|---|' + '---|' * len(NS))
    J['gain'] = {}
    for arm, lam in arms:
        row = []
        J['gain']['%s@%g' % (arm, lam)] = {}
        for n in NS:
            cs = []
            for S in SS:
                b, a = mean(S, 'data', None, n, lambda m: m['t1_avg']), mean(S, arm, lam, n, lambda m: m['t1_avg'])
                ds = [S['cmp'][t]['d_excess'] for t, m in S['runs'].items() if m.get('arm') == arm and m.get('lam') == lam and m.get('n') == n and t in S['cmp']]
                if a is None or not ds:
                    cs.append(None)
                    continue
                lo, hi = float(np.mean([d[1] for d in ds])), float(np.mean([d[2] for d in ds]))
                sig_ = '*' if all((d[1] > 0) == (d[2] > 0) and (d[0] > 0) == (ds[0][0] > 0) for d in ds) else ''
                cs.append((1 - a / b, float(np.mean([d[0] for d in ds])), sig_))
            J['gain']['%s@%g' % (arm, lam)][n] = [None if c is None else c[:2] for c in cs]
            row.append(' / '.join('-' if c is None else '%+.0f%% (%+.3f%s)' % (100 * c[0], c[1], c[2]) for c in cs))
        add('| %s @ %g | ' % (arm, lam) + ' | '.join(row) + ' |')
    add('\n* = in both matched runs the 95 % interval of the paired difference excludes 0, same sign.\n')
    # ---- registered H1 reading: per form, the weight with the lower validation loss on real outcomes
    add('**H1 as registered: per form, the weight with the lower mean validation loss (real outcomes of the n vectors); its gain G**\n')
    add('| form | ' + ' | '.join('n = %s' % format(n, ',') for n in NS) + ' |')
    add('|---|' + '---|' * len(NS))
    J['h1'] = {}
    for form in ['i', 'iio', 'iis', 'iii', 'synth']:
        row = []
        J['h1'][form] = {}
        for n in NS:
            cs = []
            for S in SS:
                cand = sorted({m['lam'] for m in S['runs'].values() if m.get('arm') == form and m.get('n') == n})
                best = None
                for lam in cand:
                    v = mean(S, form, lam, n, lambda m: m['best_val'])
                    if v is not None and (best is None or v < best[0]):
                        best = (v, lam)
                if best is None:
                    cs.append(None)
                    continue
                b, a = mean(S, 'data', None, n, lambda m: m['t1_avg']), mean(S, form, best[1], n, lambda m: m['t1_avg'])
                cs.append((best[1], 1 - a / b))
            J['h1'][form][n] = cs
            row.append(' / '.join('-' if c is None else '%+.0f%% (w %g)' % (100 * c[1], c[0]) for c in cs))
        add('| %s | ' % form + ' | '.join(row) + ' |')
    add('')
    # ---- Table 3: cross-order variance by K
    add('**Cross-order variance on held-out real vectors. Pair mismatch D (noise floor) and V_ord (noise floor), by K**\n')
    add('| predictor | ' + ' | '.join('D, K = %d' % K for K in Ks) + ' | ' + ' | '.join('V_ord, K = %d' % K for K in Ks) + ' |')
    add('|---|' + '---|' * (2 * len(Ks)))
    J['var'] = {}

    def vrow(name, getter):
        row = []
        for sec, k1, k2 in (('t2', 'D', 'D_noise'), ('t3', 'V_ord', 'V_noise')):
            for K in Ks:
                a = [getter(S, lambda m: m[sec][str(K)][k1]) for S in SS]
                b = [getter(S, lambda m: m[sec][str(K)].get(k2)) if all(k2 in m[sec][str(K)] for m in S['runs'].values() if 't2' in m and str(K) in m.get('t2', {})) else None for S in SS]
                J['var'].setdefault(name, {})['%s_K%d' % (k1, K)] = [a, b]
                row.append(cell(a, '%.3f') + (' (' + cell(b, '%.3f') + ')' if any(v is not None for v in b) else ''))
        add('| %s | ' % name + ' | '.join(row) + ' |')
    fixed = lambda nm: (lambda S, f: f(S['runs']['fixed-%s_n0_d0_r0' % nm]) if 'fixed-%s_n0_d0_r0' % nm in S['runs'] else None)
    vrow('one pass of M (M itself)', fixed('one'))
    vrow('the score, exact weights on K trajectories', fixed('exu'))
    vrow('plain average of M(x given h)', fixed('meanu'))
    for n in NS:
        vrow('real outcomes only, n = %s' % format(n, ','), lambda S, f, n=n: mean(S, 'data', None, n, f))
    for arm, lam in arms:
        for n in (0, 64, 1024, 20480):
            if any(runs_of(S, arm, lam, n) for S in SS):
                vrow('%s @ %g, n = %s' % (arm, lam, format(n, ',')), lambda S, f, arm=arm, lam=lam, n=n: mean(S, arm, lam, n, f))
    # ---- Table 4: drag
    add('\n**Drag (H5), K = %d: placebo-adjusted' % Ks[-1] + ' change of the KL to the exact joint, for the order that was better / worse under data-only [95 % interval]. Positive = got worse.**\n')
    add('| arm @ weight | n | better order | worse order | data-only gap between the two orders |')
    add('|---|---|---|---|---|')
    J['drag'] = {}
    for arm, lam in arms:
        for n in NS:
            cs = []
            for S in SS:
                t = '%s@%g_n%d_d0_r0' % (arm, lam, n)
                c = S['cmp'].get(t, {}).get('drag_K%d' % Ks[-1])
                cs.append(c)
            if all(c is None for c in cs):
                continue
            J['drag']['%s@%g_n%d' % (arm, lam, n)] = cs
            f = lambda k: ' / '.join('-' if c is None else '%+.3f [%+.3f, %+.3f]' % tuple(c[k]) for c in cs)
            add('| %s @ %g | %s | %s | %s | %s |' % (arm, lam, format(n, ','), f('adj_better'), f('adj_worse'), cell([None if c is None else c['gap_base'] for c in cs])))
    # ---- extra: by context size and by bad checks, per-K
    add('\n**Excess by context size (0-9 / 10-19 / 20-29 checks revealed), K = %d**\n' % Ks[-1])
    add('| arm | ' + ' | '.join('n = %s' % format(n, ',') for n in NS) + ' |')
    add('|---|' + '---|' * len(NS))
    for arm, lam in [('data', None)] + arms:
        row = []
        for n in NS:
            row.append(' ; '.join(cell([mean(S, arm, lam, n, lambda m: m['t1'][str(Ks[-1])]['exc_' + s]) for S in SS]) for s in ('small', 'mid', 'large')))
        add('| %s | ' % (arm if lam is None else '%s @ %g' % (arm, lam)) + ' | '.join(row) + ' |')
    for S, w in zip(SS, worlds):
        if S.get('bad'):
            add('\n**World %s, checks the score gets wrong %s: excess on those / on the rest, K = %d**\n' % (w, S['bad'], Ks[-1]))
            add('| arm | n = 0 | ' + ' | '.join('n = %s' % format(n, ',') for n in NS) + ' |')
            add('|---|' + '---|' * (len(NS) + 1))
            for arm, lam in [('data', None)] + arms:
                g = lambda n: '-' if mean(S, arm, lam, n, lambda m: m['t1'][str(Ks[-1])]['exc_bad']) is None else '%.3f / %.3f' % (
                    mean(S, arm, lam, n, lambda m: m['t1'][str(Ks[-1])]['exc_bad']), mean(S, arm, lam, n, lambda m: m['t1'][str(Ks[-1])]['exc_good']))
                add('| %s | ' % (arm if lam is None else '%s @ %g' % (arm, lam)) + ' | '.join(g(n) for n in [0] + NS) + ' |')
    open(root + '/tables.md', 'w').write('\n'.join(L) + '\n')
    json.dump(J, open(root + '/tables.json', 'w'), indent=0)
    print('\n'.join(L))
