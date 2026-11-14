"""E48 analysis (PREREG-B amendment 56): do input formats that give the posterior's sufficient statistics LINEARLY, per token,
gain beyond removing the signs, in both worlds? And do visible signs still cost once those statistics are linear?
Inputs:
  world 1: results/e48w1/e48_w1{sq,sgnsq}_s<seed>.npz (e48_eval_w1.py); E37's results/e37/e37_{ctrl,mag}_s<seed>.npz and E39's
           results/e39/e39_sgp_s<seed>.npz (same seeds, initialisations of shared parameters, episodes and schedule);
  world 2: results/e48t2/e48_d<D>{prod,sgnprod}_s<seed>.npz (e48_eval_t2.py); E40's results/e40/e40_d<D>{ctrl,canon}_s<seed>.npz.
Acquired: world 1 gap < 0.10; world 2 gap < 0.5 x the level-blind reference's gap (E40's rule).
r(a/b) = mean K_a / mean K_b over seeds acquired in both (paired ratio of means). c_sign(a) = pooled sign-flip KL reduction /
pooled family KL over a's acquired nets (uncorrected). Intervals: 2,000 bootstrap resamples of seeds (with multiplicities).
Predictions (registered in amendment 56):
  P193  world 2: r(prod/canon) <= 0.6 at D = 2 and at D = 1;
  P194  world 2, D = 2: r(sgnprod/prod) <= 1.25 and c_sign(sgnprod) <= 0.25 x c_sign(ctrl);
  P195  world 1: r(sq/mag) <= 0.8;
  P196  world 1: mean rom_large(sq) <= 0.5 x mean rom_large(mag) over seeds acquired in both;
  P197  world 1: r(sgnsq/sq) <= 1.25 and c_sign(sgnsq) <= 0.25 x c_sign(ctrl).
usage: python e48_summ.py -> results/e48.json, results/e48_tables.md
"""
import glob, json, os
import numpy as np

rng = np.random.default_rng(48)
NB = 2000


def rd(f):
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in dict(np.load(f)).items()}


def seed_of(f):
    return int(os.path.basename(f)[:-4].rsplit('_s', 1)[1])


def load():
    W1, W2 = {}, {}
    for a, pat in (('sq', 'results/e48w1/e48_w1sq_s*.npz'), ('sgnsq', 'results/e48w1/e48_w1sgnsq_s*.npz'),
                   ('ctrl', 'results/e37/e37_ctrl_s*.npz'), ('mag', 'results/e37/e37_mag_s*.npz'), ('sgp', 'results/e39/e39_sgp_s*.npz')):
        for f in sorted(glob.glob(pat)):
            W1[(a, seed_of(f))] = rd(f)
    for D in (1, 2):
        for a, pat in (('prod', 'results/e48t2/e48_d%dprod_s*.npz'), ('sgnprod', 'results/e48t2/e48_d%dsgnprod_s*.npz'),
                       ('ctrl', 'results/e40/e40_d%dctrl_s*.npz'), ('canon', 'results/e40/e40_d%dcanon_s*.npz')):
            for f in sorted(glob.glob(pat % D)):
                W2[(D, a, seed_of(f))] = rd(f)
    return W1, W2


def ratio(R, key, a, b, seeds):
    """Paired ratio of means K_a / K_b over `seeds`, with a bootstrap interval; key(arm, seed) -> R's key."""
    if len(seeds) < 3:
        return None
    f = lambda ss: np.mean([R[key(a, s)]['gap'] for s in ss]) / np.mean([R[key(b, s)]['gap'] for s in ss])
    bs = [f(list(rng.choice(seeds, len(seeds)))) for _ in range(NB)]
    return [float(f(seeds)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(seeds)]


def csign(R, key, a, seeds):
    have = [s for s in seeds if 'sign_pc' in R[key(a, s)]]
    if not have:
        return None
    return float(sum(R[key(a, s)]['sign_pc'] for s in have) / sum(R[key(a, s)]['sign_kl'] for s in have))


def main():
    W1, W2 = load()
    res = dict(per_net={}, w1={}, w2={}, predictions={})
    # ---------------- world 1
    acq1 = {k: v['gap'] < 0.10 for k, v in W1.items()}
    k1 = lambda a, s: (a, s)
    seeds1 = sorted({s for (a, s) in W1 if a in ('sq', 'sgnsq')})
    for (a, s), v in W1.items():
        if s in seeds1:
            res['per_net']['w1_%s_s%d' % (a, s)] = v
    for a in ('ctrl', 'mag', 'sgp', 'sq', 'sgnsq'):
        acs = [s for s in seeds1 if (a, s) in W1 and acq1[(a, s)]]
        o = dict(n=sum((a, s) in W1 for s in seeds1), n_acquired=len(acs), acquired=acs)
        if acs:
            v = lambda k: [W1[(a, s)][k] for s in acs]
            o.update(K=float(np.mean(v('gap'))), K_median=float(np.median(v('gap'))), n_A=int(sum(x < 0 for x in v('rom_middle'))),
                     n_B=int(sum(x > 0 for x in v('rom_middle'))), rom_large=float(np.mean(v('rom_large'))),
                     rom_middle=float(np.mean(v('rom_middle'))), lab=float(np.mean(v('lab_last'))), xq=float(np.mean(v('xq'))),
                     c_sign=csign(W1, k1, a, acs))
        res['w1'][a] = o
    both = lambda a, b: [s for s in seeds1 if all((x, s) in W1 and acq1[(x, s)] for x in (a, b))]
    w1r = {}
    for a, b in (('sq', 'mag'), ('sgnsq', 'sq'), ('sq', 'ctrl'), ('sgnsq', 'ctrl'), ('mag', 'ctrl'), ('sgnsq', 'sgp'), ('sgp', 'mag')):
        w1r['%s/%s' % (a, b)] = ratio(W1, k1, a, b, both(a, b))
    res['w1']['ratios'] = w1r
    cm = both('sq', 'mag')
    if len(cm) >= 3:
        f = lambda ss: (np.mean([W1[('sq', s)]['rom_large'] for s in ss]), np.mean([W1[('mag', s)]['rom_large'] for s in ss]))
        bs = [f(list(rng.choice(cm, len(cm)))) for _ in range(NB)]
        rs, rm = f(cm)
        res['w1']['rom_large_paired'] = dict(sq=float(rs), mag=float(rm), seeds=cm,
                                             diff_ci=[float(np.percentile([b[0] - 0.5 * b[1] for b in bs], 2.5)),
                                                      float(np.percentile([b[0] - 0.5 * b[1] for b in bs], 97.5))])
    # ---------------- world 2
    acq2 = {k: v['gap'] < 0.5 * v['blind_gap'] for k, v in W2.items()}
    for D in (1, 2):
        k2 = lambda a, s, D=D: (D, a, s)
        seeds2 = sorted({s for (d, a, s) in W2 if d == D and a in ('prod', 'sgnprod')})
        for (d, a, s), v in W2.items():
            if d == D and s in seeds2:
                res['per_net']['w2_d%d%s_s%d' % (d, a, s)] = v
        o2 = {}
        for a in ('ctrl', 'canon', 'prod', 'sgnprod'):
            acs = [s for s in seeds2 if (D, a, s) in W2 and acq2[(D, a, s)]]
            q = dict(n=sum((D, a, s) in W2 for s in seeds2), n_acquired=len(acs), acquired=acs)
            if acs:
                g = [W2[(D, a, s)]['gap'] for s in acs]
                q.update(K=float(np.mean(g)), K_median=float(np.median(g)),
                         K_by_level=[float(np.mean([W2[(D, a, s)]['gap_lev%d' % j] for s in acs])) for j in range(3)],
                         c_sign=csign(W2, k2, a, acs))
            o2[a] = q
        both2 = lambda a, b: [s for s in seeds2 if all((D, x, s) in W2 and acq2[(D, x, s)] for x in (a, b))]
        rr = {}
        for a, b in (('prod', 'canon'), ('sgnprod', 'prod'), ('prod', 'ctrl'), ('sgnprod', 'ctrl'), ('canon', 'ctrl'), ('sgnprod', 'canon')):
            rr['%s/%s' % (a, b)] = ratio(W2, k2, a, b, both2(a, b))
        o2['ratios'] = rr
        o2['per_seed'] = {s: {a: W2[(D, a, s)]['gap'] for a in ('ctrl', 'canon', 'prod', 'sgnprod') if (D, a, s) in W2} for s in seeds2}
        res['w2'][D] = o2
    # ---------------- predictions
    pr = res['predictions']
    r1, r2 = res['w2'].get(1, {}).get('ratios', {}).get('prod/canon'), res['w2'].get(2, {}).get('ratios', {}).get('prod/canon')
    if r1 and r2:
        pr['P193'] = dict(D2=r2, D1=r1, holds=bool(r2[0] <= 0.6 and r1[0] <= 0.6))
    o = res['w2'].get(2, {})
    rsp = o.get('ratios', {}).get('sgnprod/prod')
    cs, cc = o.get('sgnprod', {}).get('c_sign'), o.get('ctrl', {}).get('c_sign')
    if rsp:
        pr['P194'] = dict(sgnprod_over_prod=rsp, c_sign_sgnprod=cs, c_sign_ctrl=cc,
                          holds=bool(rsp[0] <= 1.25 and cs is not None and cc is not None and cs <= 0.25 * cc))
    if w1r.get('sq/mag'):
        pr['P195'] = dict(sq_over_mag=w1r['sq/mag'], holds=bool(w1r['sq/mag'][0] <= 0.8))
    if 'rom_large_paired' in res['w1']:
        q = res['w1']['rom_large_paired']
        pr['P196'] = dict(rom_large_sq=q['sq'], rom_large_mag=q['mag'], sq_minus_half_mag_ci=q['diff_ci'], holds=bool(q['sq'] <= 0.5 * q['mag']))
    if w1r.get('sgnsq/sq'):
        cs1, cc1 = res['w1']['sgnsq'].get('c_sign'), res['w1']['ctrl'].get('c_sign')
        pr['P197'] = dict(sgnsq_over_sq=w1r['sgnsq/sq'], c_sign_sgnsq=cs1, c_sign_ctrl=cc1,
                          holds=bool(w1r['sgnsq/sq'][0] <= 1.25 and cs1 is not None and cc1 is not None and cs1 <= 0.25 * cc1))
    # ---------------- tables
    fr = lambda r: '-' if not r else '%.2f [%.2f, %.2f] (%d)' % tuple(r)
    md = ['# E48 tables (generated by e48_summ.py)', '', '## World 1 (sources; KL in nats; E37 ctrl/mag and E39 sgp are the same seeds)', '',
          '| arm | nets | acquired | mean KL | median KL | A / B | rom large | rom middle | lab (block 2) | xq | sign red. / KL |', '|---' * 11 + '|']
    for a in ('ctrl', 'mag', 'sgp', 'sq', 'sgnsq'):
        q = res['w1'][a]
        if not q.get('n_acquired'):
            md.append('| %s | %d | 0 |' % (a, q['n'])); continue
        md.append('| %s | %d | %d | %.4f | %.4f | %d / %d | %.3f | %.3f | %.3f | %.4f | %s |' % (
            a, q['n'], q['n_acquired'], q['K'], q['K_median'], q['n_A'], q['n_B'], q['rom_large'], q['rom_middle'], q['lab'], q['xq'],
            '%.3f' % q['c_sign'] if q.get('c_sign') is not None else '-'))
    md += ['', '| ratio of means (paired seeds) | value [95% CI] (seeds) |', '|---|---|'] + ['| %s | %s |' % (k, fr(v)) for k, v in w1r.items()]
    md += ['', '| seed | ctrl | mag | sgp | sq | sgnsq |', '|---|---|---|---|---|---|']
    for s in seeds1:
        cell = lambda a: '-' if (a, s) not in W1 else '%.4f %s' % (W1[(a, s)]['gap'], ('B' if W1[(a, s)]['rom_middle'] > 0 else 'A') if acq1[(a, s)] else 'stall')
        md.append('| %d | %s | %s | %s | %s | %s |' % (s, cell('ctrl'), cell('mag'), cell('sgp'), cell('sq'), cell('sgnsq')))
    for D in (1, 2):
        o = res['w2'].get(D)
        if not o:
            continue
        md += ['', '## World 2, D = %d (regression; E40 ctrl/canon are the same seeds)' % D, '',
               '| format | nets | acquired | mean KL | median KL | KL by level (0.1, 0.5, 2.0) | sign red. / KL |', '|---' * 7 + '|']
        for a in ('ctrl', 'canon', 'prod', 'sgnprod'):
            q = o[a]
            if not q.get('n_acquired'):
                md.append('| %s | %d | 0 |' % (a, q['n'])); continue
            md.append('| %s | %d | %d | %.4f | %.4f | %s | %s |' % (a, q['n'], q['n_acquired'], q['K'], q['K_median'],
                                                             ', '.join('%.4f' % x for x in q['K_by_level']),
                                                             '%.3f' % q['c_sign'] if q.get('c_sign') is not None else '-'))
        md += ['', '| ratio of means (paired seeds) | value [95% CI] (seeds) |', '|---|---|'] + ['| %s | %s |' % (k, fr(v)) for k, v in o['ratios'].items()]
        md += ['', '| seed | ctrl | canon | prod | sgnprod |', '|---|---|---|---|---|']
        for s, v in o['per_seed'].items():
            md.append('| %d | %s |' % (s, ' | '.join('%.4f' % v[a] if a in v else '-' for a in ('ctrl', 'canon', 'prod', 'sgnprod'))))
    md += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    json.dump(res, open('results/e48.json', 'w'), indent=1, default=float)
    open('results/e48_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1, default=float))


if __name__ == '__main__':
    main()
