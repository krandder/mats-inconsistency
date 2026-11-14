"""E10.3 summary (PREREG-B amendment 17 and addendum 2): response of effective weights to shifting the decoded reliability
in the QQ tokens' layer-1 state, against the Bayes response; controls; accuracy.
usage: python qrel_summ.py -> results/qrel_summary.json, results/qrel_tables.md
"""
import json, os
import numpy as np
import qmech as QM

f = QM.f


def fit(y, x):
    x, y = x.ravel(), y.ravel(); s = float(np.polyfit(x, y, 1)[0]); c = float(np.corrcoef(x, y)[0, 1]); return dict(slope=s, corr=c)


def main():
    B = dict(np.load('results/qrel_bayes.npz')); out = dict(per_net={}, by_group={})
    for nm in QM.ALL:
        p = os.path.join('runs', nm, 'qrel.npz')
        if not os.path.exists(p): continue
        z = np.load(p); r = dict(group=QM.GROUP[nm], dec_r2=float(z['dec_r2']))
        for k in ('val', 'key', 'rnd'): r[f'all_{k}'] = fit(z[f'all_{k}'], B['all'])
        for k in ('val', 'rnd'):
            own_n = np.concatenate([z[f'one{s}_{k}'][:, s] for s in range(3)]); own_b = np.concatenate([B[f'one{s}'][:, s] for s in range(3)])
            cr_n = np.concatenate([z[f'one{s}_{k}'][:, [t for t in range(3) if t != s]].ravel() for s in range(3)])
            cr_b = np.concatenate([B[f'one{s}'][:, [t for t in range(3) if t != s]].ravel() for s in range(3)])
            r[f'own_{k}'] = fit(own_n, own_b); r[f'cross_{k}'] = fit(cr_n, cr_b)
            r[f'own_{k}']['ratio_of_means'] = float(own_n.mean() / own_b.mean())
        for v in ('plus', 'minus'):
            r[f'msd_{v}'] = float((z[f'{v}_mean'] ** 2).mean() / (z['base_mean'] ** 2).mean())
            r[f'msd_logsd_{v}'] = float((z[f'{v}_logsd'] ** 2).mean() / (z['base_logsd'] ** 2).mean())
            r[f'kl_{v}'] = float(z[f'{v}_kl'].mean() / z['base_kl'].mean())
        r['supported'] = bool(r['dec_r2'] >= 0.5 and r['all_val']['slope'] >= 0.5 and r['all_val']['corr'] >= 0.5 and abs(r['all_rnd']['slope']) <= 0.1
                              and r['msd_plus'] <= 1.1 and r['msd_minus'] <= 1.1)
        out['per_net'][nm] = r
    keys = [('dec_r2',), ('all_val', 'slope'), ('all_val', 'corr'), ('all_key', 'slope'), ('all_key', 'corr'), ('all_rnd', 'slope'), ('all_rnd', 'corr'),
            ('own_val', 'slope'), ('own_val', 'corr'), ('own_val', 'ratio_of_means'), ('cross_val', 'slope'), ('cross_val', 'corr'), ('own_rnd', 'slope'), ('own_rnd', 'corr'),
            ('msd_plus',), ('msd_minus',), ('msd_logsd_plus',), ('msd_logsd_minus',), ('kl_plus',), ('kl_minus',)]
    get = lambda r, k: r[k[0]] if len(k) == 1 else r[k[0]][k[1]]
    for g, nets0 in (('A', QM.A_NETS), ('B', QM.B_NETS)):
        nets = [nm for nm in nets0 if nm in out['per_net']]
        if not nets: continue
        out['by_group'][g] = {'.'.join(k): dict(mean=float(np.mean([get(out['per_net'][nm], k) for nm in nets])), lo=float(min(get(out['per_net'][nm], k) for nm in nets)),
                                                hi=float(max(get(out['per_net'][nm], k) for nm in nets))) for k in keys}
        out['by_group'][g]['n'] = len(nets); out['by_group'][g]['n_supported'] = int(sum(out['per_net'][nm]['supported'] for nm in nets))
        G = out['by_group'][g]
        out['by_group'][g]['supported_on_means'] = bool(G['dec_r2']['mean'] >= 0.5 and G['all_val.slope']['mean'] >= 0.5 and G['all_val.corr']['mean'] >= 0.5
                                                        and abs(G['all_rnd.slope']['mean']) <= 0.1 and G['msd_plus']['mean'] <= 1.1 and G['msd_minus']['mean'] <= 1.1)
        print(g, {k: round(v['mean'], 3) for k, v in G.items() if isinstance(v, dict)}, 'supported nets', G['n_supported'], '/', G['n'], flush=True)
    if 'A' in out['by_group']:
        out['P88'] = dict(holds=bool(out['by_group']['A']['supported_on_means']))
    json.dump(out, open('results/qrel_summary.json', 'w'), indent=1)
    L = ['### E10.3 Shifting the decoded reliability in the QQ tokens\' layer-1 state (values only) vs the Bayes response (group mean [range])\n',
         '| measure | group A | group B |', '|---|---|---|']
    lab = {'dec_r2': 'decoding R² of log own labelled SS from the QQ layer-1 state (test)',
           'all_val.slope': 'all sources shifted, value route: slope of net response on Bayes response', 'all_val.corr': '  correlation',
           'all_key.slope': 'all shifted, keys/queries only: slope', 'all_key.corr': '  correlation',
           'all_rnd.slope': 'all shifted, random direction (values): slope', 'all_rnd.corr': '  correlation',
           'own_val.slope': 'one source at a time, own weight: slope', 'own_val.corr': '  correlation', 'own_val.ratio_of_means': '  ratio of mean responses (net / Bayes)',
           'cross_val.slope': 'one at a time, the other two weights: slope', 'cross_val.corr': '  correlation',
           'own_rnd.slope': 'one at a time, random direction: own slope', 'own_rnd.corr': '  correlation',
           'msd_plus': 'mean-deviation MSD ratio, shift +0.25 SD', 'msd_minus': 'mean-deviation MSD ratio, shift −0.25 SD',
           'msd_logsd_plus': 'log-sd MSD ratio, +0.25', 'msd_logsd_minus': 'log-sd MSD ratio, −0.25', 'kl_plus': 'KL ratio, +0.25', 'kl_minus': 'KL ratio, −0.25'}
    for k in lab:
        cells = []
        for g in ('A', 'B'):
            v = out['by_group'].get(g, {}).get(k)
            cells.append(f"{f(v['mean'])} [{f(v['lo'])}, {f(v['hi'])}]" if v else '–')
        L.append(f"| {lab[k]} | {cells[0]} | {cells[1]} |")
    for g in ('A', 'B'):
        if g in out['by_group']: L.append(f"\nGroup {g}: registered criteria met on group means: {out['by_group'][g]['supported_on_means']}; nets meeting all criteria: {out['by_group'][g]['n_supported']} of {out['by_group'][g]['n']}.")
    open('results/qrel_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
