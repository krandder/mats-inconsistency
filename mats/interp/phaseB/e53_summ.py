"""E53 analysis (PREREG-B amendment 61): is the big-miss departure made by the learned weighting of the source's pool, and does a
net whose only route from a source's errors is a uniform pool read the pooled statistic like Bayes (constrained Bayes)?
Arms (world 1, fixed 90,000-step schedule; pfn_uni.py): uni and unif in the ctrl format (seeds 0-11); unif_noln in the mag and sq
formats (seeds 0-7). Baselines with the same initialisation of every parameter and the same episodes: E37's ctrl and mag nets,
E50's no-LayerNorm mag and sq nets (mag_noln, sq_noln), E48's sq nets.
Inputs: results/e53/e53_<arm>_s<seed>.npz (e53_eval.py), results/e37/, results/e50/, results/e48w1/.
Acquired: gap < 0.10. r(a/b) = mean K_a / mean K_b over seeds acquired in both; rom compared as means over seeds acquired in both
arms; d(a-b) = mean (K_a - K_b) over seeds acquired in both. 2,000 bootstrap resamples of seeds, with multiplicities.
Constrained-Bayes values (e53_cb.py; written into the amendment before any training): a Bayes reader of mean_q |r_q| has, on E12's
pairs, rom_large 0.131 and rom_middle 0.201; its KL from Bayes on the test set is KCB_MABS (quadrature).
Predictions (registered in amendment 61):
  P208  rom_large(uni) <= 0.5 x rom_large(ctrl), paired seeds;
  P209  rom_large(unif) <= 0.5 x rom_large(ctrl), paired seeds;
  P210  unif_noln, mag format: mean rom_large in [0.091, 0.171] and mean rom_middle in [0.141, 0.261] over acquired nets;
  P211  d(unifnoln_mag - unifnoln_sq) in [0.7, 1.3] x KCB_MABS, paired seeds.
usage: python e53_summ.py -> results/e53.json, results/e53_tables.md
"""
import glob, json, os
import numpy as np

rng = np.random.default_rng(53)
NB = 2000
KCB_MABS = 0.00464         # quadrature on the 16,000 test episodes (amendment 61)
CB_ROM = dict(large=0.131, middle=0.201)


def rd(f):
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in dict(np.load(f)).items()}


def seed_of(f):
    return int(os.path.basename(f)[:-4].rsplit('_s', 1)[1])


def ci(xs):
    return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]


def main():
    R = {}
    for a, pat in (('uni', 'results/e53/e53_uni_s*.npz'), ('unif', 'results/e53/e53_unif_s*.npz'),
                   ('unifnoln_mag', 'results/e53/e53_unifnoln_mag_s*.npz'), ('unifnoln_sq', 'results/e53/e53_unifnoln_sq_s*.npz'),
                   ('ctrl', 'results/e37/e37_ctrl_s*.npz'), ('mag', 'results/e37/e37_mag_s*.npz'), ('sq', 'results/e48w1/e48_w1sq_s*.npz'),
                   ('mag_noln', 'results/e50/e50_mag_s*.npz'), ('sq_noln', 'results/e50/e50_sq_s*.npz')):
        for f in sorted(glob.glob(pat)):
            R[(a, seed_of(f))] = rd(f)
    seeds = sorted({s for (a, s) in R if a in ('uni', 'unif', 'unifnoln_mag', 'unifnoln_sq')})
    acq = {k: v['gap'] < 0.10 for k, v in R.items()}
    arms = ('ctrl', 'uni', 'unif', 'mag', 'unifnoln_mag', 'mag_noln', 'sq', 'unifnoln_sq', 'sq_noln')
    res = dict(per_net={'%s_s%d' % k: v for k, v in R.items() if k[1] in seeds and k[0] in arms}, arms={}, ratios={}, diffs={}, rom={},
               predictions={}, constants=dict(KCB_MABS=KCB_MABS, CB_ROM=CB_ROM))
    for a in arms:
        pool = [s for s in seeds if (a, s) in R]
        if a.startswith('unifnoln'):
            pool = [s for s in pool if s < 8]
        acs = [s for s in pool if acq[(a, s)]]
        o = dict(n=len(pool), n_acquired=len(acs), acquired=acs)
        if acs:
            v = lambda k: [R[(a, s)][k] for s in acs]
            o.update(K=float(np.mean(v('gap'))), K_median=float(np.median(v('gap'))), rom_large=float(np.mean(v('rom_large'))),
                     rom_middle=float(np.mean(v('rom_middle'))), n_A=int(sum(x < 0 for x in v('rom_middle'))),
                     n_B=int(sum(x > 0 for x in v('rom_middle'))), lab=float(np.mean(v('lab_last'))), xq=float(np.mean(v('xq'))))
            for k in ('rom_large', 'rom_middle'):
                bs = [np.mean([R[(a, s)][k] for s in rng.choice(acs, len(acs))]) for _ in range(NB)]
                o[k + '_ci'] = ci(bs)
            have = [s for s in acs if 'sign_pc' in R[(a, s)]]
            if have:
                o['c_sign'] = float(sum(R[(a, s)]['sign_pc'] for s in have) / sum(R[(a, s)]['sign_kl'] for s in have))
        res['arms'][a] = o
    both = lambda a, b: [s for s in seeds if all((x, s) in R and acq[(x, s)] for x in (a, b))]
    for a, b in (('uni', 'ctrl'), ('unif', 'ctrl'), ('unif', 'uni'), ('unif', 'mag'), ('unifnoln_sq', 'mag'), ('unifnoln_sq', 'sq_noln'),
                 ('unifnoln_mag', 'mag_noln'), ('unifnoln_mag', 'unifnoln_sq'), ('unifnoln_sq', 'unif')):
        ss = both(a, b)
        if len(ss) >= 3:
            f = lambda q: np.mean([R[(a, s)]['gap'] for s in q]) / np.mean([R[(b, s)]['gap'] for s in q])
            bs = [f(list(rng.choice(ss, len(ss)))) for _ in range(NB)]
            res['ratios']['%s/%s' % (a, b)] = [float(f(ss))] + ci(bs) + [len(ss)]
    for a, b in (('unifnoln_mag', 'unifnoln_sq'),):
        ss = both(a, b)
        if len(ss) >= 3:
            f = lambda q: np.mean([R[(a, s)]['gap'] - R[(b, s)]['gap'] for s in q])
            bs = [f(list(rng.choice(ss, len(ss)))) for _ in range(NB)]
            res['diffs']['%s-%s' % (a, b)] = [float(f(ss))] + ci(bs) + [len(ss)]
    for a, b in (('uni', 'ctrl'), ('unif', 'ctrl'), ('unifnoln_mag', 'mag_noln'), ('unifnoln_sq', 'sq_noln')):
        ss = both(a, b)
        if len(ss) >= 3:
            f = lambda q: (np.mean([R[(a, s)]['rom_large'] for s in q]), np.mean([R[(b, s)]['rom_large'] for s in q]))
            bs = [f(list(rng.choice(ss, len(ss)))) for _ in range(NB)]
            ra, rb = f(ss)
            res['rom']['%s/%s' % (a, b)] = dict(a=float(ra), b=float(rb), seeds=ss, a_minus_half_b_ci=ci([x[0] - 0.5 * x[1] for x in bs]))
    pr = res['predictions']
    for p, a in (('P208', 'uni'), ('P209', 'unif')):
        q = res['rom'].get('%s/ctrl' % a)
        if q:
            pr[p] = dict(rom_arm=q['a'], rom_ctrl=q['b'], ci=q['a_minus_half_b_ci'], n=len(q['seeds']), holds=bool(q['a'] <= 0.5 * q['b']))
    o = res['arms'].get('unifnoln_mag', {})
    if o.get('n_acquired'):
        pr['P210'] = dict(rom_large=o['rom_large'], rom_large_ci=o['rom_large_ci'], rom_middle=o['rom_middle'], rom_middle_ci=o['rom_middle_ci'],
                          cb=CB_ROM, n=o['n_acquired'],
                          holds=bool(0.091 <= o['rom_large'] <= 0.171 and 0.141 <= o['rom_middle'] <= 0.261))
    d = res['diffs'].get('unifnoln_mag-unifnoln_sq')
    if d:
        pr['P211'] = dict(diff=d, band=[0.7 * KCB_MABS, 1.3 * KCB_MABS], holds=bool(0.7 * KCB_MABS <= d[0] <= 1.3 * KCB_MABS))
    md = ['# E53 tables (generated by e53_summ.py)', '',
          '| arm | nets | acquired | mean KL | median KL | A / B | rom large [95% CI] | rom middle [95% CI] | lab | xq | sign share |',
          '|---' * 11 + '|']
    for a in arms:
        o = res['arms'][a]
        if not o.get('n_acquired'):
            md.append('| %s | %d | 0 |' % (a, o['n'])); continue
        md.append('| %s | %d | %d | %.4f | %.4f | %d / %d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f | %.4f | %s |' % (
            a, o['n'], o['n_acquired'], o['K'], o['K_median'], o['n_A'], o['n_B'], o['rom_large'], *o['rom_large_ci'], o['rom_middle'],
            *o['rom_middle_ci'], o['lab'], o['xq'], '%.3f' % o['c_sign'] if 'c_sign' in o else '-'))
    md += ['', '| ratio of means (paired seeds) | value [95% CI] (seeds) |', '|---|---|']
    md += ['| %s | %.2f [%.2f, %.2f] (%d) |' % ((k,) + tuple(v)) for k, v in res['ratios'].items()]
    md += ['', '| mean paired difference of KL | value [95% CI] (seeds) |', '|---|---|']
    md += ['| %s | %.4f [%.4f, %.4f] (%d) |' % ((k,) + tuple(v)) for k, v in res['diffs'].items()]
    md += ['', '| large-range rom, paired seeds | a | b | a - 0.5 b [95% CI] | seeds |', '|---|---|---|---|---|']
    md += ['| %s | %.3f | %.3f | [%.3f, %.3f] | %d |' % (k, v['a'], v['b'], v['a_minus_half_b_ci'][0], v['a_minus_half_b_ci'][1], len(v['seeds']))
           for k, v in res['rom'].items()]
    md += ['', '| seed | ' + ' | '.join(arms) + ' |', '|---' * (len(arms) + 1) + '|']
    for s in seeds:
        cell = lambda a: '-' if (a, s) not in R else '%.4f %s %.3f' % (R[(a, s)]['gap'], ('B' if R[(a, s)]['rom_middle'] > 0 else 'A') if acq[(a, s)] else 'stall',
                                                                 R[(a, s)]['rom_large'])
        md.append('| %d | %s |' % (s, ' | '.join(cell(a) for a in arms)))
    md += ['', 'Cells: test KL, type (or stall), large-range rom.', '', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    os.makedirs('results', exist_ok=True)
    json.dump(res, open('results/e53.json', 'w'), indent=1, default=float)
    open('results/e53_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1, default=float))


if __name__ == '__main__':
    main()
