"""E10.5 summary (PREREG-B amendment 17 addendum 5). usage: python qsize_summ.py -> results/qsize_summary.json, results/qsize_tables.md"""
import json, os
import numpy as np
import nnoise  # noqa: F401
import qbayes as QB
import qmech as QM
import qmech9 as Q9

f = QM.f; N4 = 4000; VARS = ['a10', 'a25', 'p10', 'p25']; DOSE = {'a10': 0.1, 'a25': 0.25, 'p10': 0.1, 'p25': 0.25}


def main():
    t8 = QM.test8000(); d4 = {k: v[:N4] for k, v in t8['d'].items()}
    p = QB.bayes_features(d4)[:, 20:23].astype(np.float64); p /= p.sum(1, keepdims=True)
    S = {nm: np.load(os.path.join('runs', nm, 'share_test.npz')) for nm in QM.ALL}
    out = dict(per_net={}, by_group={})
    for nm in QM.ALL:
        pth = os.path.join('runs', nm, 'qsize.npz')
        if not os.path.exists(pth): continue
        z = np.load(pth); q = np.load(os.path.join('runs', nm, 'qe9.npz'))['masses_qq'].sum(1); q = q / q.sum(1, keepdims=True)
        Sb, _ = Q9.sharp(z['w_base'], p); Sq = Q9.slope(q, p)
        ref = np.mean([S[o]['r2_mean'][:N4] for o in (QM.A_NETS if QM.GROUP[nm] == 'A' else QM.B_NETS) if o != nm], 0)
        r = dict(group=QM.GROUP[nm], S_base=Sb, S_q=Sq)
        for v in VARS:
            Sv, ok = Q9.sharp(z[f'w_{v}'], p); fl = (Sb - Sv) / (Sb - Sq)
            r[v] = dict(S=Sv, f=fl, f_per_dose=fl / DOSE[v], msd=float((z[f'{v}_mean'] ** 2).mean() / (z['base_mean'] ** 2).mean()),
                        msd_logsd=float((z[f'{v}_logsd'] ** 2).mean() / (z['base_logsd'] ** 2).mean()), kl=float(z[f'{v}_kl'].mean() / z['base_kl'].mean()),
                        retained=QM.cov(z[f'{v}_mean'], ref) / QM.cov(z['base_mean'], ref))
        out['per_net'][nm] = r
    for g, nets0 in (('A', QM.A_NETS), ('B', QM.B_NETS)):
        nets = [nm for nm in nets0 if nm in out['per_net']]
        if not nets: continue
        G = dict(n=len(nets))
        for v in VARS:
            G[v] = {k: dict(mean=float(np.mean([out['per_net'][nm][v][k] for nm in nets])), lo=float(min(out['per_net'][nm][v][k] for nm in nets)),
                            hi=float(max(out['per_net'][nm][v][k] for nm in nets))) for k in ('f', 'f_per_dose', 'msd', 'msd_logsd', 'kl', 'retained')}
        G['holds'] = bool(G['a10']['f_per_dose']['mean'] >= 0.5 and G['a25']['f_per_dose']['mean'] >= 0.5 and abs(G['p10']['f_per_dose']['mean']) <= 0.1
                          and abs(G['p25']['f_per_dose']['mean']) <= 0.1 and G['a10']['msd']['mean'] <= 1.1)
        out['by_group'][g] = G
        print(g, {v: (round(G[v]['f_per_dose']['mean'], 2), round(G[v]['msd']['mean'], 3), round(G[v]['retained']['mean'], 3)) for v in VARS}, 'holds', G['holds'], flush=True)
    if 'A' in out['by_group']: out['P89'] = dict(holds=out['by_group']['A']['holds'])
    json.dump(out, open('results/qsize_summary.json', 'w'), indent=1)
    L = ['### E10.5 Equalising the size of the reading\'s layer-1 encoding across sources (values only; group mean [range])\n',
         '| patch | group | flattening f | f per unit dose | mean MSD ratio | log-sd MSD ratio | KL ratio | shared mean-error retained |', '|---|---|---|---|---|---|---|---|']
    lab = {'a10': 'equalise 10%', 'a25': 'equalise 25%', 'p10': 'control: permuted sizes, 10%', 'p25': 'control: permuted sizes, 25%'}
    for v in VARS:
        for g in ('A', 'B'):
            if g not in out['by_group']: continue
            e = out['by_group'][g][v]; c = lambda k: f"{f(e[k]['mean'])} [{f(e[k]['lo'])}, {f(e[k]['hi'])}]"
            L.append(f"| {lab[v]} | {g} | {c('f')} | {c('f_per_dose')} | {e['msd']['mean']:.3f} [{e['msd']['lo']:.3f}, {e['msd']['hi']:.3f}] | {f(e['msd_logsd']['mean'])} | {f(e['kl']['mean'])} | {c('retained')} |")
    open('results/qsize_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
