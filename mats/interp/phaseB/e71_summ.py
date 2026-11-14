"""E71 analysis (PREREG-B amendment 79): head count against head width in block 1.
Cells (block 1's heads x head dimension; block 2 always 2 x 16; seeds 0-11, the same episodes and ctrl-paired initial values):
  ctrl  = 2 x 16 (E37, GPU: results/e37/e37_ctrl_s<seed>.npz; heads results/e63/ctrl.json)
  b1h4  = 4 x 8  (E70, GPU: results/e70/e70_b1h4_s<seed>.npz; heads results/e70_heads/)
  h2w8  = 2 x 8, h3w16 = 3 x 16, h4w16 = 4 x 16 (E71, CPU: results/e71/e71_<arm>_s<seed>.npz; heads results/e71_heads/)
Acquired: gap < 0.10. Type A: rom_middle < 0. Pooling head P_h >= 0.5, consensus head C_h >= 0.5. Sign share: sum(sign_pc) / sum(sign_kl)
over acquired nets. Paired ratios: mean KL over the seeds acquired in both cells; bootstrap 2,000 resamples of seeds (numpy seed 710).
Predictions (registered in amendment 79):
  P275  h4w16: >= 75% of acquired nets A-type, and >= 75% with >= 2 pooling heads and >= 1 consensus head;
  P276  h2w8: <= 50% of acquired nets A-type;
  P277  h3w16: as P275;
  P278  h4w16 and h3w16 each: sign share <= 0.100 and paired mean-KL ratio to ctrl <= 0.75 (point estimates);
  P279  h2w8 / b1h4 paired mean-KL ratio: lower 95% limit >= 1.25;
  P280  A lineages of b1h4, h3w16 and h4w16 together: median first checkpoint (of 2k, 3k, 5k, 10k, 20k, 40k, 60k, 80k, 90k) with
        >= 2 pooling heads is <= 5,000 steps (ctrl's A lineages, E68: 10,000).
Descriptive: per-seed table; per cell A fraction, stalls, mean / median KL, sign share, rom_large, lab_b2, xq, head counts, pooling-head
L_own and K; the paired log-KL contrasts for head count (h4w16 vs ctrl at dimension 16, b1h4 vs h2w8 at dimension 8) and head
dimension (h2w8 vs ctrl at 2 heads, b1h4 vs h4w16 at 4 heads); trajectory summaries per cell.
usage: python e71_summ.py -> results/e71.json, results/e71_tables.md
"""
import glob, json, os
import numpy as np

rng = np.random.default_rng(710)
CELLS = ('ctrl', 'h2w8', 'h3w16', 'h4w16', 'b1h4')
SHAPE = {'ctrl': '2 x 16', 'h2w8': '2 x 8', 'h3w16': '3 x 16', 'h4w16': '4 x 16', 'b1h4': '4 x 8'}
STEPS = [2000, 3000, 5000, 10000, 20000, 40000, 60000, 80000, 90000]


def ev(p):
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: float(z[k]) for k in z.files if z[k].ndim == 0 and np.issubdtype(z[k].dtype, np.number)}


def boot(f, seeds):
    if not seeds:
        return None
    vals = [f(list(rng.choice(seeds, len(seeds)))) for _ in range(2000)]
    return [float(f(seeds)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(seeds)]


def heads_of(cell):
    out = {}
    if cell == 'ctrl':
        d = json.load(open('results/e63/ctrl.json'))
        for tag, o in d.items():
            s = int(tag.rsplit('_s', 1)[1]); C = o.get('C_h', [])
            out[s] = dict(n_pool=o['n_pool'], n_consensus=int(sum(c >= 0.5 for c in C)), K_h=o.get('Kodd_mean_h', []),
                          L_own_h=o.get('L_own_h', []), pool_heads=o.get('pool_heads', []))
        return out
    pat = 'results/e70_heads/e70_b1h4_s*.json' if cell == 'b1h4' else 'results/e71_heads/e71_%s_s*.json' % cell
    for p in glob.glob(pat):
        o = json.load(open(p)); out[int(o['tag'].rsplit('_s', 1)[1])] = o
    return out


def first_pair(cell, seed):
    tag = ('e70_b1h4_s%d' if cell == 'b1h4' else 'e71_' + cell + '_s%d') % seed
    for st in STEPS:
        p = 'results/e71_traj/%s@%d.json' % (tag, st)
        if not os.path.exists(p):
            return None, None
        o = json.load(open(p))
        if o['n_pool'] >= 2:
            return st, o['opposite_pair']
    return 'never', None


def main():
    C = {c: {s: ev(('results/e37/e37_ctrl_s%d.npz' if c == 'ctrl' else 'results/e70/e70_b1h4_s%d.npz' if c == 'b1h4'
                    else 'results/e71/e71_' + c + '_s%d.npz') % s) for s in range(12)} for c in CELLS}
    acq = {c: [s for s, r in d.items() if r is not None and r['gap'] < 0.10] for c, d in C.items()}
    H = {c: heads_of(c) for c in CELLS}
    g = lambda c, j: np.mean([C[c][s]['gap'] for s in j])
    res = dict(acquired=acq, cells={}, predictions={}, contrasts={}, trajectories={}); pr = res['predictions']
    for c in CELLS:
        a = acq[c]
        if not a:
            continue
        typ = {s: 'A' if C[c][s]['rom_middle'] < 0 else 'B' for s in a}
        both = [s for s in a if s in H[c] and H[c][s]['n_pool'] >= 2 and H[c][s]['n_consensus'] >= 1]
        pk = [H[c][s]['L_own_h'][h] for s in a if s in H[c] for h in H[c][s]['pool_heads']]
        opp = [s for s in a if s in H[c] and H[c][s]['n_pool'] >= 2 and (H[c][s].get('opposite_pair') if 'opposite_pair' in H[c][s]
               else any(x * y < 0 for i, x in enumerate([H[c][s]['K_h'][h] for h in H[c][s]['pool_heads']])
                        for y in [H[c][s]['K_h'][h] for h in H[c][s]['pool_heads']][i + 1:]))]
        res['cells'][c] = dict(shape=SHAPE[c], n=sum(r is not None for r in C[c].values()), acquired=len(a), A=sum(t == 'A' for t in typ.values()),
                               types=typ, kl_mean=float(g(c, a)), kl_median=float(np.median([C[c][s]['gap'] for s in a])),
                               sign_share=float(sum(C[c][s]['sign_pc'] for s in a) / sum(C[c][s]['sign_kl'] for s in a)),
                               rom_large=float(np.mean([C[c][s]['rom_large'] for s in a])), rom_middle=float(np.mean([C[c][s]['rom_middle'] for s in a])),
                               lab_b2=float(np.mean([C[c][s]['lab_b2'] for s in a])), xq=float(np.mean([C[c][s]['xq'] for s in a])),
                               n_pool={s: H[c][s]['n_pool'] for s in a if s in H[c]}, n_consensus={s: H[c][s]['n_consensus'] for s in a if s in H[c]},
                               pool_and_consensus=len(both), opposite_pair=len(opp),
                               L_own_pool_median=float(np.median(pk)) if pk else None,
                               kl_ratio_ctrl=boot(lambda j, c=c: g(c, j) / g('ctrl', j), [s for s in a if s in acq['ctrl']]) if c != 'ctrl' else None)
    cl = res['cells']
    for p, c in (('P275', 'h4w16'), ('P277', 'h3w16')):
        if c in cl:
            x = cl[c]; pr[p] = dict(A=x['A'], pool_and_consensus=x['pool_and_consensus'], acquired=x['acquired'],
                                    holds=bool(x['A'] >= 0.75 * x['acquired'] and x['pool_and_consensus'] >= 0.75 * x['acquired']))
            if x['acquired'] < 6:
                pr[p]['holds'] = 'undecided'                                   # amendment 79: fewer than 6 acquired nets
    if 'h2w8' in cl:
        x = cl['h2w8']; pr['P276'] = dict(A=x['A'], acquired=x['acquired'], holds=bool(x['A'] <= 0.5 * x['acquired']) if x['acquired'] >= 6 else 'undecided')
    if 'h4w16' in cl and 'h3w16' in cl:
        part = {c: dict(sign_share=cl[c]['sign_share'], kl_ratio_ctrl=cl[c]['kl_ratio_ctrl'],
                        holds=bool(cl[c]['sign_share'] <= 0.100 and cl[c]['kl_ratio_ctrl'][0] <= 0.75)) for c in ('h4w16', 'h3w16')}
        pr['P278'] = dict(part, holds=bool(all(v['holds'] for v in part.values())))
    if 'h2w8' in cl and 'b1h4' in cl:
        r = boot(lambda j: g('h2w8', j) / g('b1h4', j), [s for s in acq['h2w8'] if s in acq['b1h4']])
        pr['P279'] = dict(ratio=r, holds=bool(r[1] >= 1.25))
    lin = []
    for c in ('b1h4', 'h3w16', 'h4w16', 'h2w8'):
        rows = {}
        for s in acq.get(c, []):
            st, opp = first_pair(c, s); rows[s] = dict(type=cl[c]['types'][s], first_pair=st, opposite_at_first=opp)
            if c != 'h2w8' and cl[c]['types'][s] == 'A':
                lin.append(st)
        res['trajectories'][c] = rows
    if lin and all(x is not None for x in lin):
        num = [x if x != 'never' else 10 ** 9 for x in lin]
        pr['P280'] = dict(n=len(lin), median_first_pair=float(np.median(num)), counts={str(k): lin.count(k) for k in sorted(set(lin), key=str)},
                          holds=bool(np.median(num) <= 5000))
    for name, (c1, c0) in {'count_at_dim16 (h4w16 / ctrl)': ('h4w16', 'ctrl'), 'count_at_dim8 (b1h4 / h2w8)': ('b1h4', 'h2w8'),
                           'three_heads_at_dim16 (h3w16 / ctrl)': ('h3w16', 'ctrl'), 'dim_at_2_heads (h2w8 / ctrl)': ('h2w8', 'ctrl'),
                           'dim_at_4_heads (b1h4 / h4w16)': ('b1h4', 'h4w16')}.items():
        j = [s for s in acq.get(c1, []) if s in acq.get(c0, [])]
        res['contrasts'][name] = boot(lambda jj: float(np.mean([np.log(C[c1][s]['gap'] / C[c0][s]['gap']) for s in jj])), j)
    md = ['# E71 tables (generated by e71_summ.py)', '', '| seed | ' + ' | '.join('%s (%s): KL, type, rom_large, pool / cons' % (c, SHAPE[c]) for c in CELLS) + ' |',
          '|---|' + '---|' * len(CELLS)]
    t = lambda c, s: '-' if C[c][s] is None else ('%.4f %s %.3f%s' % (C[c][s]['gap'], 'stall' if C[c][s]['gap'] >= 0.10 else ('A' if C[c][s]['rom_middle'] < 0 else 'B'),
                                                                      C[c][s]['rom_large'], (' %d / %d' % (H[c][s]['n_pool'], H[c][s]['n_consensus'])) if s in H[c] else ''))
    for s in range(12):
        md.append('| %d | ' % s + ' | '.join(t(c, s) for c in CELLS) + ' |')
    md += ['', '| cell | shape | acquired | A | pool + cons | opposite pair | mean KL | median KL | KL ratio to ctrl [95%] | sign share | rom_large | lab_b2 | xq | L_own (pool, median) |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for c in CELLS:
        if c in cl:
            x = cl[c]; r = x['kl_ratio_ctrl']
            md.append('| %s | %s | %d / %d | %d | %d | %d | %.4f | %.4f | %s | %.3f | %.3f | %.3f | %.3f | %s |' % (
                c, x['shape'], x['acquired'], x['n'], x['A'], x['pool_and_consensus'], x['opposite_pair'], x['kl_mean'], x['kl_median'],
                '-' if r is None else '%.2f [%.2f, %.2f]' % tuple(r[:3]), x['sign_share'], x['rom_large'], x['lab_b2'], x['xq'],
                '-' if x['L_own_pool_median'] is None else '%.1f' % x['L_own_pool_median']))
    md += ['', '## Paired log-KL contrasts (mean log ratio [95%], seeds)', ''] + ['- %s: %s' % (k, v) for k, v in res['contrasts'].items()]
    md += ['', '## Trajectories (first checkpoint with >= 2 pooling heads; opposite signs there)', '']
    for c, rows in res['trajectories'].items():
        md.append('- %s: ' % c + '; '.join('s%d %s %s %s' % (s, r['type'], r['first_pair'], r['opposite_at_first']) for s, r in sorted(rows.items())))
    md += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    json.dump(res, open('results/e71.json', 'w'), indent=1, default=float)
    open('results/e71_tables.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
