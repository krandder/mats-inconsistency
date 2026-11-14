"""E49 analysis (PREREG-B amendment 57): does a per-token MLP before the first attention (an architecture change, ctrl format)
remove much of the departure in both worlds?
Inputs: results/e49w1/e49_w1pre_s<seed>.npz and results/e49t2/e49_d<D>pre_s<seed>.npz (e49_eval.py); the baselines E37's
results/e37/e37_ctrl_s<seed>.npz (world 1) and E40's results/e40/e40_d<D>ctrl_s<seed>.npz (world 2): same seeds, the same
initialisation of every shared parameter, the same episodes and schedule. Descriptive companions when present: E37's mag, E48's
sq (world 1) and E40's canon, E48's prod (world 2).
Acquired: world 1 gap < 0.10; world 2 gap < 0.5 x the level-blind reference's gap. r(pre/ctrl) = mean K_pre / mean K_ctrl over
seeds acquired in both; c_sign = pooled sign-flip KL reduction / pooled family KL (uncorrected). 2,000 bootstrap resamples of
seeds, with multiplicities.
Predictions (registered in amendment 57):
  P198  world 1: r(pre/ctrl) <= 0.6;
  P199  world 2, D = 2: r(pre/ctrl) <= 0.6;
  P200  c_sign(pre) <= 0.5 x c_sign(ctrl), in world 1 and in world 2 at D = 2.
usage: python e49_summ.py -> results/e49.json, results/e49_tables.md
"""
import glob, json, os
import numpy as np

rng = np.random.default_rng(49)
NB = 2000


def rd(f):
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in dict(np.load(f)).items()}


def seed_of(f):
    return int(os.path.basename(f)[:-4].rsplit('_s', 1)[1])


def ratio(R, a, b, seeds):
    if len(seeds) < 3:
        return None
    f = lambda ss: np.mean([R[(a, s)]['gap'] for s in ss]) / np.mean([R[(b, s)]['gap'] for s in ss])
    bs = [f(list(rng.choice(seeds, len(seeds)))) for _ in range(NB)]
    return [float(f(seeds)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(seeds)]


def csign(R, a, seeds):
    have = [s for s in seeds if 'sign_pc' in R[(a, s)]]
    return float(sum(R[(a, s)]['sign_pc'] for s in have) / sum(R[(a, s)]['sign_kl'] for s in have)) if have else None


def world(R, acq, seeds, arms, extra_keys):
    o = {}
    for a in arms:
        acs = [s for s in seeds if (a, s) in R and acq[(a, s)]]
        q = dict(n=sum((a, s) in R for s in seeds), n_acquired=len(acs), acquired=acs)
        if acs:
            q.update(K=float(np.mean([R[(a, s)]['gap'] for s in acs])), K_median=float(np.median([R[(a, s)]['gap'] for s in acs])),
                     c_sign=csign(R, a, acs))
            for k in extra_keys:
                if all(k in R[(a, s)] for s in acs):
                    q[k] = float(np.mean([R[(a, s)][k] for s in acs]))
            if all('rom_middle' in R[(a, s)] for s in acs):
                q['n_A'] = int(sum(R[(a, s)]['rom_middle'] < 0 for s in acs)); q['n_B'] = len(acs) - q['n_A']
        o[a] = q
    both = lambda a, b: [s for s in seeds if all((x, s) in R and acq[(x, s)] for x in (a, b))]
    o['ratios'] = {'%s/%s' % (a, b): ratio(R, a, b, both(a, b)) for a in arms for b in arms
                   if a != b and (a, b) in (('pre', 'ctrl'), ('pre', 'mag'), ('pre', 'sq'), ('pre', 'canon'), ('pre', 'prod'))}
    o['per_seed'] = {s: {a: R[(a, s)]['gap'] for a in arms if (a, s) in R} for s in seeds}
    return o


def main():
    res = dict(w1={}, w2={}, predictions={})
    R = {}
    for a, pat in (('pre', 'results/e49w1/e49_w1pre_s*.npz'), ('ctrl', 'results/e37/e37_ctrl_s*.npz'), ('mag', 'results/e37/e37_mag_s*.npz'),
                   ('sq', 'results/e48w1/e48_w1sq_s*.npz')):
        for f in sorted(glob.glob(pat)):
            R[(a, seed_of(f))] = rd(f)
    seeds = sorted({s for (a, s) in R if a == 'pre'})
    acq = {k: v['gap'] < 0.10 for k, v in R.items()}
    res['w1'] = world(R, acq, seeds, ('ctrl', 'pre', 'mag', 'sq'), ('rom_large', 'rom_middle', 'lab_last', 'xq'))
    for D in (1, 2):
        R2 = {}
        for a, pat in (('pre', 'results/e49t2/e49_d%dpre_s*.npz'), ('ctrl', 'results/e40/e40_d%dctrl_s*.npz'),
                       ('canon', 'results/e40/e40_d%dcanon_s*.npz'), ('prod', 'results/e48t2/e48_d%dprod_s*.npz')):
            for f in sorted(glob.glob(pat % D)):
                R2[(a, seed_of(f))] = rd(f)
        seeds2 = sorted({s for (a, s) in R2 if a == 'pre'})
        acq2 = {k: v['gap'] < 0.5 * v['blind_gap'] for k, v in R2.items()}
        if seeds2:
            res['w2'][D] = world(R2, acq2, seeds2, ('ctrl', 'pre', 'canon', 'prod'), ('neg_pc',))
    pr = res['predictions']
    r1 = res['w1'].get('ratios', {}).get('pre/ctrl')
    if r1:
        pr['P198'] = dict(pre_over_ctrl=r1, holds=bool(r1[0] <= 0.6))
    r2 = res['w2'].get(2, {}).get('ratios', {}).get('pre/ctrl')
    if r2:
        pr['P199'] = dict(pre_over_ctrl_D2=r2, holds=bool(r2[0] <= 0.6))
    c1p, c1c = res['w1'].get('pre', {}).get('c_sign'), res['w1'].get('ctrl', {}).get('c_sign')
    c2p, c2c = res['w2'].get(2, {}).get('pre', {}).get('c_sign'), res['w2'].get(2, {}).get('ctrl', {}).get('c_sign')
    if None not in (c1p, c1c, c2p, c2c):
        pr['P200'] = dict(w1=[c1p, c1c], w2_D2=[c2p, c2c], holds=bool(c1p <= 0.5 * c1c and c2p <= 0.5 * c2c))
    fr = lambda r: '-' if not r else '%.2f [%.2f, %.2f] (%d)' % tuple(r)
    md = ['# E49 tables (generated by e49_summ.py)']
    for name, o in [('World 1', res['w1'])] + [('World 2, D = %d' % D, res['w2'][D]) for D in sorted(res['w2'])]:
        if not o:
            continue
        md += ['', '## ' + name, '', '| arm | nets | acquired | mean KL | median KL | A / B | sign red. / KL |', '|---' * 7 + '|']
        for a, q in o.items():
            if a in ('ratios', 'per_seed'):
                continue
            if not q.get('n_acquired'):
                md.append('| %s | %d | 0 |' % (a, q['n'])); continue
            md.append('| %s | %d | %d | %.4f | %.4f | %s | %s |' % (a, q['n'], q['n_acquired'], q['K'], q['K_median'],
                                                             '%d / %d' % (q['n_A'], q['n_B']) if 'n_A' in q else '-',
                                                             '%.3f' % q['c_sign'] if q.get('c_sign') is not None else '-'))
        md += ['', '| ratio of means (paired seeds) | value [95% CI] (seeds) |', '|---|---|'] + ['| %s | %s |' % (k, fr(v)) for k, v in o['ratios'].items()]
        arms = [a for a in o if a not in ('ratios', 'per_seed')]
        md += ['', '| seed | ' + ' | '.join(arms) + ' |', '|---' * (len(arms) + 1) + '|']
        for s, v in o['per_seed'].items():
            md.append('| %d | %s |' % (s, ' | '.join('%.4f' % v[a] if a in v else '-' for a in arms)))
    md += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    json.dump(res, open('results/e49.json', 'w'), indent=1, default=float)
    open('results/e49_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1, default=float))


if __name__ == '__main__':
    main()
