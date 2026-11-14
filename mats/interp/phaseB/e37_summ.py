"""E37 analysis (PREREG-B amendment 45): when is the solution chosen? A format-switch test of the selection hypothesis.
Inputs: results/e37/<tag>.npz from e37_eval.py (tags e37_<arm>_s<seed>), results/e37_traj/<tag>.npz from e37_traj.py, the runs'
checkpoints (branch-point check) and E30's results (reproduction check).
Definitions (acquired = test gap < 0.10; K = test KL in the net's final format; lab = the last block's query-row attention mass
on the 12 labelled tokens):
  R(a)    = sum_s (K_ctrl,s - K_a,s) / sum_s (K_ctrl,s - K_mag,s), over seeds acquired in ctrl, mag and a: how mag-like arm a's
            final KL is (1 = as mag from scratch, 0 = as ctrl);
  Rlab(a) = sum_s (lab_ctrl,s - lab_a,s) / sum_s (lab_ctrl,s - lab_mag,s), the same for the routing marker; testable when the mean
            of lab_ctrl - lab_mag over those seeds is >= 0.05 in absolute value;
  type    B if rom_middle > 0, A if < 0.
Intervals: 2,000 bootstrap resamples of the seeds (paired across arms).
Predictions (selection hypothesis):
  P166  R(c2m2k) >= 0.8;
  P167  R(c2m30k) <= 0.5 and R(mag60) >= 0.8;
  P168  Rlab(c2m30k) <= 0.5 (if testable);
  P169  among seeds whose ctrl net is B type: >= half of their acquired c2m30k nets are B type, and <= a quarter of their acquired
        c2m2k nets are B type;
  P170  R(m2c10k) >= 0.5 and R(m2c30k) >= 0.5.
usage: python e37_summ.py -> results/e37.json, results/e37_tables.md
"""
import glob, json, os
import numpy as np
import torch

ARMS = ['ctrl', 'mag', 'c2m2k', 'c2m3k', 'c2m6k', 'c2m10k', 'c2m30k', 'm2c3k', 'm2c10k', 'm2c30k', 'mag60']
SWITCH = {'c2m2k': 2000, 'c2m3k': 3000, 'c2m6k': 6000, 'c2m10k': 10000, 'c2m30k': 30000, 'm2c3k': 3000, 'm2c10k': 10000,
          'm2c30k': 30000}
rng = np.random.default_rng(37)


def load():
    R = {}
    for f in sorted(glob.glob('results/e37/e37_*_s*.npz')):
        tag = os.path.basename(f)[4:-4]; arm, s = tag.rsplit('_s', 1)
        R[(arm, int(s))] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in dict(np.load(f)).items()}
    return R


def main():
    R = load()
    seeds = sorted({s for (_, s) in R})
    acq = {k: v['gap'] < 0.10 for k, v in R.items()}
    ok = lambda arms, s: all((a, s) in R and acq[(a, s)] for a in arms)

    def boot(stat, arms):
        common = [s for s in seeds if ok(arms, s)]
        if len(common) < 3:
            return None, common
        pt = stat(common)
        bs = [stat(list(rng.choice(common, len(common)))) for _ in range(2000)]
        return [float(pt), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], common

    def ratio(a, key):
        def f(ss):
            den = sum(R[('ctrl', s)][key] - R[('mag', s)][key] for s in ss)
            return sum(R[('ctrl', s)][key] - R[(a, s)][key] for s in ss) / den if den != 0 else np.nan
        return f

    res = dict(per_net={'%s_s%d' % k: v for k, v in R.items()}, arms={}, R={}, Rlab={})
    for a in ARMS:
        ss = [s for s in seeds if (a, s) in R]; acs = [s for s in ss if acq[(a, s)]]
        o = dict(n=len(ss), n_acquired=len(acs))
        if acs:
            o.update(K=float(np.mean([R[(a, s)]['gap'] for s in acs])), K_median=float(np.median([R[(a, s)]['gap'] for s in acs])),
                     n_A=int(sum(R[(a, s)]['rom_middle'] < 0 for s in acs)), n_B=int(sum(R[(a, s)]['rom_middle'] > 0 for s in acs)),
                     rom_large=float(np.mean([R[(a, s)]['rom_large'] for s in acs])),
                     lab=float(np.mean([R[(a, s)]['lab_last'] for s in acs])), xq=float(np.mean([R[(a, s)]['xq'] for s in acs])))
            have = [s for s in acs if 'sign_pc' in R[(a, s)]]
            if have:
                o['c_sign'] = float(sum(R[(a, s)]['sign_pc'] for s in have) / sum(R[(a, s)]['sign_kl'] for s in have))
        res['arms'][a] = o
        if a not in ('ctrl', 'mag'):
            v, c = boot(ratio(a, 'gap'), ['ctrl', 'mag', a]); res['R'][a] = dict(value=v, seeds=c)
            v, c = boot(ratio(a, 'lab_last'), ['ctrl', 'mag', a])
            dl = float(np.mean([R[('ctrl', s)]['lab_last'] - R[('mag', s)]['lab_last'] for s in c])) if c else None
            res['Rlab'][a] = dict(value=v, seeds=c, mean_lab_ctrl_minus_mag=dl, testable=bool(dl is not None and abs(dl) >= 0.05))
    pr = {}
    g = lambda a: (res['R'].get(a) or {}).get('value')
    if g('c2m2k'):
        pr['P166'] = dict(R_c2m2k=g('c2m2k'), holds=bool(g('c2m2k')[0] >= 0.8))
    if g('c2m30k') and g('mag60'):
        pr['P167'] = dict(R_c2m30k=g('c2m30k'), R_mag60=g('mag60'), holds=bool(g('c2m30k')[0] <= 0.5 and g('mag60')[0] >= 0.8))
    rl = res['Rlab'].get('c2m30k')
    if rl and rl['value']:
        pr['P168'] = dict(Rlab_c2m30k=rl['value'], mean_lab_ctrl_minus_mag=rl['mean_lab_ctrl_minus_mag'], testable=rl['testable'],
                          holds=bool(rl['testable'] and rl['value'][0] <= 0.5) if rl['testable'] else None)
    Bseeds = [s for s in seeds if ('ctrl', s) in R and acq[('ctrl', s)] and R[('ctrl', s)]['rom_middle'] > 0]
    cnt = {}
    for a in ('c2m2k', 'c2m30k'):
        have = [s for s in Bseeds if (a, s) in R and acq[(a, s)]]
        cnt[a] = (int(sum(R[(a, s)]['rom_middle'] > 0 for s in have)), len(have))
    pr['P169'] = dict(B_ctrl_seeds=Bseeds, c2m30k_B_of=cnt['c2m30k'], c2m2k_B_of=cnt['c2m2k'],
                      holds=bool(cnt['c2m30k'][1] > 0 and cnt['c2m30k'][0] >= 0.5 * cnt['c2m30k'][1]
                                 and cnt['c2m2k'][1] > 0 and cnt['c2m2k'][0] <= 0.25 * cnt['c2m2k'][1]))
    if g('m2c10k') and g('m2c30k'):
        pr['P170'] = dict(R_m2c10k=g('m2c10k'), R_m2c30k=g('m2c30k'), holds=bool(g('m2c10k')[0] >= 0.5 and g('m2c30k')[0] >= 0.5))
    res['predictions'] = pr
    # descriptive: commitment step (first switch step with R < 0.5)
    fw = [(SWITCH[a], g(a)[0]) for a in ('c2m2k', 'c2m3k', 'c2m6k', 'c2m10k', 'c2m30k') if g(a)]
    res['commitment_step'] = next((k for k, r in fw if r < 0.5), None)
    # reproduction of E30's 90k nets (same init, stream and schedule) and branch-point identity
    rep = {}
    for a in ('ctrl', 'mag'):
        for s in seeds:
            p = 'results/e30/%s_s%d.npz' % (a, s)
            if os.path.exists(p) and (a, s) in R:
                z = np.load(p)
                if int(z['total_steps']) == 90000:
                    rep['%s_s%d' % (a, s)] = float(abs(float(z['gap']) - R[(a, s)]['gap']))
    res['reproduction_vs_e30_abs_gap_diff'] = rep
    bp = {}
    for a, K in SWITCH.items():
        f0 = 'ctrl' if a.startswith('c2m') else 'mag'
        for s in seeds:
            p1, p0 = 'runs/e37/e37_%s_s%d/ckpt/step_%06d.pt' % (a, s, K), 'runs/e37/e37_%s_s%d/ckpt/step_%06d.pt' % (f0, s, K)
            if os.path.exists(p1) and os.path.exists(p0):
                A, B = torch.load(p1, map_location='cpu'), torch.load(p0, map_location='cpu')
                bp['%s_s%d' % (a, s)] = max(float((A[k] - B[k]).abs().max()) for k in A)
    res['branch_point_max_abs_diff'] = bp
    # descriptive trajectories: medians over seeds per arm and checkpoint step
    traj = {}
    for a in ARMS:
        rows = {}
        for f in sorted(glob.glob('results/e37_traj/e37_%s_s*.npz' % a)):
            z = np.load(f)
            for i, st in enumerate(z['step']):
                kl = z['kl1'][i] if z['fin'][i] else z['kl0'][i]
                rows.setdefault(int(st), []).append((kl, z['lab'][i], z['xq'][i], z['rom_middle'][i]))
        traj[a] = {st: dict(n=len(v), kl=float(np.median([x[0] for x in v])), lab=float(np.median([x[1] for x in v])),
                            xq=float(np.median([x[2] for x in v])),
                            frac_B=float(np.mean([x[3] > 0 for x in v if np.isfinite(x[3])])) if any(np.isfinite(x[3]) for x in v) else None)
                   for st, v in sorted(rows.items())}
    res['trajectories'] = traj
    md = ['# E37 tables (generated by e37_summ.py)', '',
          '| arm | nets | acquired | mean KL | median KL | A / B | rom large | lab (last block) | xq | sign red. / KL | R [95% CI] | Rlab [95% CI] |',
          '|---' * 12 + '|']
    fmt = lambda v: '-' if not v else '%.2f [%.2f, %.2f]' % tuple(v)
    for a in ARMS:
        o = res['arms'][a]
        if not o.get('n_acquired'):
            md.append('| %s | %d | 0 |' % (a, o['n'])); continue
        md.append('| %s | %d | %d | %.4f | %.4f | %d / %d | %.3f | %.3f | %.4f | %s | %s | %s |' % (
            a, o['n'], o['n_acquired'], o['K'], o['K_median'], o['n_A'], o['n_B'], o['rom_large'], o['lab'], o['xq'],
            '%.3f' % o['c_sign'] if 'c_sign' in o else '-', fmt((res['R'].get(a) or {}).get('value')),
            fmt((res['Rlab'].get(a) or {}).get('value'))))
    md += ['', '| seed | ' + ' | '.join(ARMS) + ' |', '|---' * (len(ARMS) + 1) + '|']
    for s in seeds:
        cells = []
        for a in ARMS:
            o = R.get((a, s))
            cells.append('-' if o is None else '%.4f %s %.2f' % (o['gap'], 'B' if o['rom_middle'] > 0 else 'A', o['lab_last']))
        md.append('| %d | ' % s + ' | '.join(cells) + ' |')
    md += ['', '(cells: test KL, type, last-block labelled attention mass)', '', '## Predictions', '']
    md += ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    md += ['', 'commitment step (first switch step with R < 0.5): %s' % res['commitment_step'],
           'reproduction of E30 90k nets, abs gap diff: %s' % json.dumps(rep),
           'branch points, max abs weight diff: max %s over %d' % (max(bp.values()) if bp else None, len(bp))]
    json.dump(res, open('results/e37.json', 'w'), indent=1, default=float)
    open('results/e37_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1, default=float))
    print('R', json.dumps({a: v['value'] for a, v in res['R'].items()}, default=float))
    print('commitment', res['commitment_step'], 'repro', rep, 'branch max', max(bp.values()) if bp else None)


if __name__ == '__main__':
    main()
