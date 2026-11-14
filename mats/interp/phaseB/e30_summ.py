"""E30 analysis (PREREG-B amendment 38).
K_arm = mean test KL (gap) over the arm's acquired nets (gap < 0.10). c_sign, c_code = the ctrl nets' pooled sign-flip and
source-code KL-reduction shares (uncorrected; sum of reductions / sum of the family's mean KL). Intervals: 2,000 bootstrap
resamples of the seeds (the same seeds for every arm, so paired comparisons keep their pairing), over seeds acquired in every
arm involved.
Predictions:
  P150  1 - K_mag / K_ctrl lies in [0.5, 1.5] x c_sign;
  P151  K_sgnmag <= 1.1 x K_mag, and the sgnmag nets' pooled sign-flip reduction share <= 0.25 x c_sign;
  P152  1 - K_fix / K_ctrl >= 0.5 x c_code;
  P153  among seeds acquired in ctrl and mag, the paired reduction K_ctrl - K_mag is larger on average for the seeds whose ctrl net
        is B type (middle-range rom > 0) than for A-type seeds (testable with >= 2 of each).
usage: python e30_summ.py -> results/e30.json, results/e30_tables.md
"""
import glob, json, os
import numpy as np

ARMS = ['ctrl', 'mag', 'sgnmag', 'fix']


def main():
    R = {}
    IN = os.environ.get('E30_OUT', 'results/e30'); RES = os.environ.get('E30_RES', 'results/e30')    # overrides: checks only
    for f in sorted(glob.glob(os.path.join(IN, '*_s*.npz'))):
        z = dict(np.load(f)); tag = os.path.basename(f)[:-4]; arm, s = tag.rsplit('_s', 1)
        R[(arm, int(s))] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in z.items()}
    seeds = sorted({s for (_, s) in R})
    acq = {(a, s): R[(a, s)]['gap'] < 0.10 for (a, s) in R}
    res = dict(per_net={'%s_s%d' % k: v for k, v in R.items()}, arms={})
    for a in ARMS:
        ss = [s for s in seeds if (a, s) in R]
        g = [R[(a, s)]['gap'] for s in ss if acq[(a, s)]]
        res['arms'][a] = dict(n=len(ss), n_acquired=len(g), K=float(np.mean(g)) if g else None,
                              K_median=float(np.median(g)) if g else None,
                              n_B=int(sum(R[(a, s)]['rom_middle'] > 0 for s in ss if acq[(a, s)])),
                              n_A=int(sum(R[(a, s)]['rom_middle'] < 0 for s in ss if acq[(a, s)])),
                              rom_large=float(np.mean([R[(a, s)]['rom_large'] for s in ss if acq[(a, s)]])) if g else None,
                              rom_middle=float(np.mean([R[(a, s)]['rom_middle'] for s in ss if acq[(a, s)]])) if g else None)
        for c in ('sign', 'code'):
            have = [s for s in ss if acq[(a, s)] and (c + '_pc') in R[(a, s)]]
            if have:
                res['arms'][a]['c_' + c] = float(sum(R[(a, s)][c + '_pc'] for s in have) / sum(R[(a, s)][c + '_kl'] for s in have))
    rng = np.random.default_rng(30)

    def boot(stat, arms):
        common = [s for s in seeds if all((a, s) in R and acq[(a, s)] for a in arms)]
        if len(common) < 3:
            return None, len(common)
        pt = stat(common)
        bs = [stat(list(rng.choice(common, len(common)))) for _ in range(2000)]
        return [float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], len(common)

    K = lambda a, ss: np.mean([R[(a, s)]['gap'] for s in ss])
    cs = lambda ss: sum(R[('ctrl', s)]['sign_pc'] for s in ss) / sum(R[('ctrl', s)]['sign_kl'] for s in ss)
    cc = lambda ss: sum(R[('ctrl', s)]['code_pc'] for s in ss) / sum(R[('ctrl', s)]['code_kl'] for s in ss)
    pr = {}
    red_mag, n1 = boot(lambda ss: 1 - K('mag', ss) / K('ctrl', ss), ['ctrl', 'mag'])
    csign, _ = boot(cs, ['ctrl', 'mag'])
    if red_mag:
        lo, hi = 0.5 * csign[0], 1.5 * csign[0]
        pr['P150'] = dict(reduction=red_mag, c_sign=csign, band=[lo, hi], n_seeds=n1, holds=bool(lo <= red_mag[0] <= hi))
    r_sm, n2 = boot(lambda ss: K('sgnmag', ss) / K('mag', ss), ['mag', 'sgnmag'])
    if r_sm and 'c_sign' in res['arms']['sgnmag'] and 'c_sign' in res['arms']['ctrl']:
        sg = res['arms']['sgnmag']['c_sign']; ct = res['arms']['ctrl']['c_sign']
        pr['P151'] = dict(K_sgnmag_over_K_mag=r_sm, sgnmag_c_sign=sg, ctrl_c_sign=ct, n_seeds=n2,
                          holds=bool(r_sm[0] <= 1.1 and sg <= 0.25 * ct))
    red_fix, n3 = boot(lambda ss: 1 - K('fix', ss) / K('ctrl', ss), ['ctrl', 'fix'])
    ccode, _ = boot(cc, ['ctrl', 'fix'])
    if red_fix:
        pr['P152'] = dict(reduction=red_fix, c_code=ccode, n_seeds=n3, holds=bool(red_fix[0] >= 0.5 * ccode[0]))
    common = [s for s in seeds if all((a, s) in R and acq[(a, s)] for a in ('ctrl', 'mag'))]
    Bs = [s for s in common if R[('ctrl', s)]['rom_middle'] > 0]; As = [s for s in common if R[('ctrl', s)]['rom_middle'] < 0]
    if len(Bs) >= 2 and len(As) >= 2:
        dB = float(np.mean([R[('ctrl', s)]['gap'] - R[('mag', s)]['gap'] for s in Bs]))
        dA = float(np.mean([R[('ctrl', s)]['gap'] - R[('mag', s)]['gap'] for s in As]))
        pr['P153'] = dict(reduction_B=dB, reduction_A=dA, n_B=len(Bs), n_A=len(As), holds=bool(dB > dA))
    else:
        pr['P153'] = dict(n_B=len(Bs), n_A=len(As), testable=False)
    # descriptive comparisons asked for by the coordinator
    d = {}
    for a in ('mag', 'sgnmag', 'fix'):
        v, n = boot(lambda ss, a=a: K(a, ss) / K('ctrl', ss), ['ctrl', a]); d['K_%s_over_K_ctrl' % a] = v
    v, _ = boot(lambda ss: K('fix', ss) / (K('ctrl', ss) * (1 - cc(ss))), ['ctrl', 'fix']); d['K_fix_over_ctrl_code_averaged'] = v
    v, _ = boot(lambda ss: K('mag', ss) / (K('ctrl', ss) * (1 - cs(ss))), ['ctrl', 'mag']); d['K_mag_over_ctrl_sign_averaged'] = v
    v, _ = boot(lambda ss: K('sgnmag', ss) / K('ctrl', ss), ['ctrl', 'sgnmag']); d['K_sgnmag_over_K_ctrl'] = v
    res['descriptive'] = d; res['predictions'] = pr
    md = ['# E30 tables (generated by e30_summ.py)', '', '| arm | nets | acquired | mean KL | median KL | A / B | rom large | rom middle | sign reduction / KL | code reduction / KL |',
          '|---' * 10 + '|']
    for a, o in res['arms'].items():
        md.append('| %s | %d | %d | %s | %s | %d / %d | %s | %s | %s | %s |' % (
            a, o['n'], o['n_acquired'], '%.4f' % o['K'] if o['K'] is not None else '-', '%.4f' % o['K_median'] if o['K_median'] is not None else '-',
            o['n_A'], o['n_B'], '%.3f' % o['rom_large'] if o['rom_large'] is not None else '-',
            '%.3f' % o['rom_middle'] if o['rom_middle'] is not None else '-',
            '%.3f' % o['c_sign'] if 'c_sign' in o else '-', '%.3f' % o['c_code'] if 'c_code' in o else '-'))
    md += ['', '| net | gap | rom large | rom middle | sign red. | code red. |', '|---' * 6 + '|']
    for k in sorted(res['per_net']):
        o = res['per_net'][k]
        md.append('| %s | %.4f | %.3f | %.3f | %s | %s |' % (k, o['gap'], o['rom_large'], o['rom_middle'],
                  '%.5f' % o['sign_pc'] if 'sign_pc' in o else '-', '%.5f' % o['code_pc'] if 'code_pc' in o else '-'))
    md += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    md += ['', '## Descriptive', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in d.items()]
    json.dump(res, open(RES + '.json', 'w'), indent=1, default=float)
    open(RES + '_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1, default=float)); print(json.dumps(d, indent=1, default=float))


if __name__ == '__main__':
    main()
