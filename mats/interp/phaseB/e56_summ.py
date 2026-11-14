"""E56 analysis (PREREG-B amendment 64): does decoupling the summary readings from the labelled errors' input column raise the
learned LayerNorm cap scale and shrink the big-miss departure?
Arms: sep (learned pool) and unif sep (E53's uniform factorised pool), seeds 0-11, against E37's ctrl nets and E53's unif nets of
the same seeds (same initialisation of every shared parameter, same episodes).
Inputs: results/e56/<tag>.npz (e56_eval.py); results/e37/e37_ctrl_s*.npz, results/e53/e53_unif_s*.npz; cap scales: c of the E56 nets
computed here with lngeom.py exactly as e55_run.py (E27's first 500 bases, the net's own format; labelled tokens only), c of the
comparison nets from results/e55_geom.json; the constrained-Bayes table results/e55_cbtable.json.
Acquired: gap < 0.10. Paired seeds acquired in both. 2,000 bootstrap resamples of seeds, with multiplicities.
Predictions (registered in amendment 64):
  P219  sep against ctrl: the median over paired seeds of c(sep) / c(ctrl) >= 1.25, and mean rom_large(sep) <= 0.7 x mean
        rom_large(ctrl);
  P220  unif sep against unif: the median of c(unif sep) / c(unif) >= 1.5, and mean rom_large(unif sep) <= 0.6 x mean
        rom_large(unif);
  P221  unif sep: the mean measured rom_large and rom_middle are each within 0.03 of the mean constrained-Bayes prediction at the
        nets' c.
usage: python e56_summ.py -> results/e56.json, results/e56_tables.md
"""
import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import e56_fmt as F  # noqa: E402
import lngeom as G  # noqa: E402

rng = np.random.default_rng(56)
NB = 2000


def rd(f):
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in dict(np.load(f, allow_pickle=True)).items()}


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    tab = json.load(open('results/e55_cbtable.json')); grid = np.array(tab['grid'])
    TL = np.array([tab['rom']['%.2f' % c]['large'] for c in grid]); TM = np.array([tab['rom']['%.2f' % c]['middle'] for c in grid])
    geo = json.load(open('results/e55_geom.json'))
    z = np.load('results/e27_data.npz')
    Xs, _ = F.make_X(z['o'][:500, 0], z['y'][:500], z['semb'][:500], z['qemb'][:500], 'sep')
    R = {}
    for f in sorted(glob.glob('results/e56/*.npz')):
        tag = os.path.basename(f)[:-4]; r = rd(f)
        arm = 'unifsep' if r.get('variant') == 'unif' else 'sep'
        sd = torch.load(os.path.join('runs/e56', tag, 'model.pt'), map_location='cpu')
        g = G.geom_wb(sd['inp.weight'].numpy(), sd['inp.bias'].numpy(), Xs)
        W = sd['inp.weight'].numpy()
        r.update(c=g['c_median'], A_cv=g['A_cv'], rho=g['rho_abs_median'], U=g['U'], U2=float(np.linalg.norm(W[:, 36] - W[:, 36].mean())))
        R[(arm, int(r['seed']))] = r
    for s in range(12):
        for arm, tag in (('ctrl', 'e37_ctrl_s%d' % s), ('unif', 'e53_unif_s%d' % s)):
            sub = 'e37' if arm == 'ctrl' else 'e53'
            if os.path.exists('results/%s/%s.npz' % (sub, tag)) and tag in geo:
                r = rd('results/%s/%s.npz' % (sub, tag)); r.update(c=geo[tag]['c'], A_cv=geo[tag]['A_cv'], rho=geo[tag]['rho'], U=geo[tag]['U'])
                R[(arm, s)] = r
    for r in R.values():
        r['cb_large'] = float(np.interp(r['c'], grid, TL)); r['cb_middle'] = float(np.interp(r['c'], grid, TM))
    acq = {k: v['gap'] < 0.10 for k, v in R.items()}
    res = dict(arms={}, pairs={}, predictions={}, per_net={'%s_s%d' % k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for k, v in R.items()})
    for a in ('ctrl', 'sep', 'unif', 'unifsep'):
        acs = [s for s in range(12) if (a, s) in R and acq[(a, s)]]
        o = dict(n=sum((a, s) in R for s in range(12)), n_acquired=len(acs))
        if acs:
            v = lambda k: [R[(a, s)][k] for s in acs]
            o.update(K=float(np.mean(v('gap'))), rom_large=float(np.mean(v('rom_large'))), rom_middle=float(np.mean(v('rom_middle'))),
                     c_median=float(np.median(v('c'))), A_cv=float(np.median(v('A_cv'))), rho=float(np.median(v('rho'))),
                     cb_large=float(np.mean(v('cb_large'))), cb_middle=float(np.mean(v('cb_middle'))),
                     n_A=int(sum(x < 0 for x in v('rom_middle'))), lab=float(np.mean(v('lab_last'))))
            if 'U2' in R[(a, acs[0])]:
                o['U'] = float(np.median(v('U'))); o['U2'] = float(np.median(v('U2')))
            have = [s for s in acs if 'sign_pc' in R[(a, s)]]
            if have:
                o['c_sign'] = float(sum(R[(a, s)]['sign_pc'] for s in have) / sum(R[(a, s)]['sign_kl'] for s in have))
        res['arms'][a] = o
    pr = res['predictions']
    for p, a, b, cbar, rbar in (('P219', 'sep', 'ctrl', 1.25, 0.7), ('P220', 'unifsep', 'unif', 1.5, 0.6)):
        ss = [s for s in range(12) if all((x, s) in R and acq[(x, s)] for x in (a, b))]
        if len(ss) < 3:
            pr[p] = dict(n=len(ss), holds=None); continue
        cr = [R[(a, s)]['c'] / R[(b, s)]['c'] for s in ss]
        fr = lambda q: np.mean([R[(a, s)]['rom_large'] for s in q]) / np.mean([R[(b, s)]['rom_large'] for s in q])
        fk = lambda q: np.mean([R[(a, s)]['gap'] for s in q]) / np.mean([R[(b, s)]['gap'] for s in q])
        bs = [list(rng.choice(ss, len(ss))) for _ in range(NB)]
        res['pairs']['%s/%s' % (a, b)] = dict(seeds=ss, c_ratio_median=float(np.median(cr)), rom_ratio=float(fr(ss)), rom_ratio_ci=ci([fr(q) for q in bs]),
                                             kl_ratio=float(fk(ss)), kl_ratio_ci=ci([fk(q) for q in bs]))
        pr[p] = dict(c_ratio_median=float(np.median(cr)), rom_ratio=float(fr(ss)), n=len(ss),
                     holds=bool(np.median(cr) >= cbar and fr(ss) <= rbar))
    o = res['arms'].get('unifsep', {})
    if o.get('n_acquired'):
        pr['P221'] = dict(rom_large=o['rom_large'], cb_large=o['cb_large'], rom_middle=o['rom_middle'], cb_middle=o['cb_middle'],
                          holds=bool(abs(o['rom_large'] - o['cb_large']) <= 0.03 and abs(o['rom_middle'] - o['cb_middle']) <= 0.03))
    md = ['# E56 tables (generated by e56_summ.py)', '', '| arm | nets | acquired | mean KL | median c | A cv | abs rho | rom large (measured / CB at c) | rom middle (measured / CB at c) | A / B | lab | sign share |',
          '|---' * 12 + '|']
    for a, o in res['arms'].items():
        if not o.get('n_acquired'):
            md.append('| %s | %d | 0 |' % (a, o['n'])); continue
        md.append('| %s | %d | %d | %.4f | %.2f | %.3f | %.3f | %.3f / %.3f | %.3f / %.3f | %d / %d | %.3f | %s |' % (
            a, o['n'], o['n_acquired'], o['K'], o['c_median'], o['A_cv'], o['rho'], o['rom_large'], o['cb_large'], o['rom_middle'], o['cb_middle'],
            o['n_A'], o['n_acquired'] - o['n_A'], o['lab'], '%.3f' % o['c_sign'] if 'c_sign' in o else '-'))
    md += ['', '| pair | seeds | median c ratio | rom_large ratio [95% CI] | KL ratio [95% CI] |', '|---|---|---|---|---|']
    md += ['| %s | %d | %.2f | %.2f [%.2f, %.2f] | %.2f [%.2f, %.2f] |' % (k, len(v['seeds']), v['c_ratio_median'], v['rom_ratio'], *v['rom_ratio_ci'], v['kl_ratio'], *v['kl_ratio_ci'])
           for k, v in res['pairs'].items()]
    md += ['', '| seed | ' + ' | '.join(('ctrl', 'sep', 'unif', 'unifsep')) + ' |', '|---' * 5 + '|']
    for s in range(12):
        cell = lambda a: '-' if (a, s) not in R else '%.4f %s c %.2f rl %.3f' % (R[(a, s)]['gap'], ('B' if R[(a, s)]['rom_middle'] > 0 else 'A') if acq[(a, s)] else 'stall', R[(a, s)]['c'], R[(a, s)]['rom_large'])
        md.append('| %d | %s |' % (s, ' | '.join(cell(a) for a in ('ctrl', 'sep', 'unif', 'unifsep'))))
    md += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    json.dump(res, open('results/e56.json', 'w'), indent=1, default=float)
    open('results/e56_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1, default=float))


if __name__ == '__main__':
    main()
