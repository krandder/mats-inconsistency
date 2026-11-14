"""E55 analysis (PREREG-B amendment 63): does block 1's LayerNorm cap scale c, read from a net's input embedding, predict its
big-miss departure?
Inputs: results/e55_geom.json and results/e55_traj.json (e55_run.py; weights only), results/e55_cbtable.json (e55_cbtable.py; the
constrained-Bayes departure of T = mean_q h_c(r_q), h_c(r) = 1 - (1 + r^2 / c^2)^(-1/2), on E12's pairs, for a grid of c), the
nets' registered evaluations (results/<sub>/<tag>.npz: gap, rom_large, rom_middle) and E37's trajectories (results/e37_traj/).
Acquired: gap < 0.10. Predicted rom of a net: linear interpolation of the table at its c (clamped to the grid; flagged).
Predictions (registered in amendment 63):
  P215  E53's uniform-pool nets (uni and unif, ctrl format, acquired, seeds 1-11; seed 0 was seen in exploration): in each arm the
        mean measured rom_large and rom_middle are each within 0.03 of the mean constrained-Bayes prediction at the nets' c;
  P216  nets with learned pooling, confirmatory cohorts with >= 8 acquired nets: within-cohort Spearman(c, rom_large) < 0 in >= 80%
        of the cohorts, and the median within-cohort Spearman <= -0.4;
  P217  width: the median c of E38's width-128 nets is larger than that of E37's width-32 nets, for ctrl and for mag;
  P218  training: in E37's acquired pure ctrl and mag runs, the within-run Spearman between c and rom_large over the checkpoints at
        >= 5,000 steps is negative in >= 80% of the runs.
usage: python e55_summ.py -> results/e55.json, results/e55_tables.md
"""
import json, os
import numpy as np


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    geom = json.load(open('results/e55_geom.json')); tab = json.load(open('results/e55_cbtable.json'))
    grid = np.array(tab['grid']); TL = np.array([tab['rom']['%.2f' % c]['large'] for c in grid]); TM = np.array([tab['rom']['%.2f' % c]['middle'] for c in grid])
    nets = {}
    for tag, g in geom.items():
        p = 'results/%s/%s.npz' % (g['sub'], tag if g['sub'] != 'e30' else tag)
        if not os.path.exists(p):
            continue
        r = np.load(p, allow_pickle=True)
        if float(r['gap']) >= 0.10:
            continue
        nets[tag] = dict(g, gap=float(r['gap']), rom_large=float(r['rom_large']), rom_middle=float(r['rom_middle']),
                         cb_large=float(np.interp(g['c'], grid, TL)), cb_middle=float(np.interp(g['c'], grid, TM)),
                         clamped=bool(g['c'] < grid[0] or g['c'] > grid[-1]))
    res = dict(cohorts={}, predictions={}, per_net=nets)
    for co in sorted({n['cohort'] for n in nets.values()}):
        v = [n for n in nets.values() if n['cohort'] == co]
        c = np.array([n['c'] for n in v]); rl = np.array([n['rom_large'] for n in v])
        res['cohorts'][co] = dict(n=len(v), exploration=v[0]['exploration'], c_median=float(np.median(c)), c_min=float(c.min()), c_max=float(c.max()),
                                  rom_large=float(rl.mean()), rom_middle=float(np.mean([n['rom_middle'] for n in v])),
                                  cb_large=float(np.mean([n['cb_large'] for n in v])), cb_middle=float(np.mean([n['cb_middle'] for n in v])),
                                  A_cv=float(np.median([n['A_cv'] for n in v])), rho=float(np.median([n['rho'] for n in v])),
                                  spearman=spearman(c, rl) if len(v) >= 3 else None)
    pr = res['predictions']
    ok, det = True, {}
    for arm in ('e53_uni', 'e53_unif'):
        v = [n for n in nets.values() if n['cohort'] == arm and n['seed'] >= 1]
        if not v:
            ok = None; break
        m = dict(n=len(v), rom_large=float(np.mean([n['rom_large'] for n in v])), cb_large=float(np.mean([n['cb_large'] for n in v])),
                 rom_middle=float(np.mean([n['rom_middle'] for n in v])), cb_middle=float(np.mean([n['cb_middle'] for n in v])))
        m['holds'] = bool(abs(m['rom_large'] - m['cb_large']) <= 0.03 and abs(m['rom_middle'] - m['cb_middle']) <= 0.03)
        ok = ok and m['holds']; det[arm] = m
    pr['P215'] = dict(det, holds=ok)
    conf = [co for co, o in res['cohorts'].items() if not o['exploration'] and not co.startswith('e53') and o['n'] >= 8]
    sp = [res['cohorts'][co]['spearman'] for co in conf]
    pr['P216'] = dict(cohorts=conf, spearman=sp, n_negative=int(sum(x < 0 for x in sp)), median=float(np.median(sp)) if sp else None,
                      holds=bool(sp and sum(x < 0 for x in sp) >= 0.8 * len(sp) and np.median(sp) <= -0.4))
    cm = lambda co: res['cohorts'].get(co, {}).get('c_median')
    pr['P217'] = dict(ctrl=[cm('e38_w128ctrl'), cm('x_e37_ctrl')], mag=[cm('e38_w128mag'), cm('x_e37_mag')],
                      holds=bool(cm('e38_w128ctrl') and cm('e38_w128ctrl') > cm('x_e37_ctrl') and cm('e38_w128mag') > cm('x_e37_mag')))
    traj = json.load(open('results/e55_traj.json')); rs = {}
    for tag, t in traj.items():
        fin = 'results/e37/%s.npz' % tag
        if not os.path.exists(fin) or float(np.load(fin)['gap']) >= 0.10:
            continue
        tz = np.load('results/e37_traj/%s.npz' % tag, allow_pickle=True)
        rl = dict(zip(tz['step'].tolist(), tz['rom_large'].tolist())); cc = dict(zip(t['step'], t['c']))
        st = [s for s in t['step'] if s >= 5000 and s in rl and np.isfinite(rl[s])]   # rom is measured at 6k-90k only (corrected after a first run)
        rs[tag] = dict(spearman=spearman([cc[s] for s in st], [rl[s] for s in st]), n=len(st), c_5k=cc.get(6000), c_90k=cc.get(90000),
                       rom_5k=rl.get(6000), rom_90k=rl.get(90000))
    sp = [x['spearman'] for x in rs.values()]
    res['traj'] = rs
    pr['P218'] = dict(n_runs=len(sp), n_negative=int(sum(x < 0 for x in sp)), median=float(np.median(sp)) if sp else None,
                      holds=bool(sp and sum(x < 0 for x in sp) >= 0.8 * len(sp)))
    md = ['# E55 tables (generated by e55_summ.py)', '', '| cohort | acquired nets | median c [min, max] | A cv | abs rho | rom large (measured / CB at c) | rom middle (measured / CB at c) | Spearman(c, rom large) |',
          '|---' * 8 + '|']
    for co, o in res['cohorts'].items():
        md.append('| %s%s | %d | %.2f [%.2f, %.2f] | %.3f | %.3f | %.3f / %.3f | %.3f / %.3f | %s |' % (
            co, ' (exploration)' if o['exploration'] else '', o['n'], o['c_median'], o['c_min'], o['c_max'], o['A_cv'], o['rho'],
            o['rom_large'], o['cb_large'], o['rom_middle'], o['cb_middle'], '%.2f' % o['spearman'] if o['spearman'] is not None else '-'))
    md += ['', '| E37 run (training, measured steps >= 5,000: 6k-90k) | Spearman(c, rom large) | c at 6k / 90k | rom large at 6k / 90k |', '|---|---|---|---|']
    md += ['| %s | %.2f | %.2f / %.2f | %.3f / %.3f |' % (k, x['spearman'], x['c_5k'], x['c_90k'], x['rom_5k'], x['rom_90k']) for k, x in sorted(rs.items())]
    md += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    json.dump(res, open('results/e55.json', 'w'), indent=1, default=float)
    open('results/e55_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1, default=float))


if __name__ == '__main__':
    main()
