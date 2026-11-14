"""E25 analysis (PREREG-B amendment 33): share of the big-miss weight contrast removed by replacing, at the modified source's own
query-question token only, its labelled tokens' attention terms with the spread version's (all four, the big-error one, the others).
usage: python e25_summ.py -> results/e25.json, results/e25_tables.md
"""
import json
import numpy as np
import e24_summ as S


def main():
    z = np.load('results/e12_pairs.npz'); rg = z['range']
    nets = json.load(open('results/e22_nets.json')); D = {r['tag']: dict(np.load(f"results/e25/{r['tag']}.npz")) for r in nets}
    out = dict(pooled={}, per_net={}, predictions={})
    s = rg == 'large'

    def removed(keys, X):
        num = sum((D[k]['ref'][s] - D[k][X][s]).sum() for k in keys); den = sum((D[k]['ref'][s] - D[k]['spread'][s]).sum() for k in keys)
        return float(num / den)
    keys = list(D)
    for name, L in (('all', keys), ('A_bases', S.A_BASES), ('B_bases', S.B_BASES)):
        out['pooled'][name] = {X: removed(L, X) for X in ('qq_all', 'qq_big', 'qq_small')}
    for k in keys:
        out['per_net'][k] = {X: removed([k], X) for X in ('qq_all', 'qq_big', 'qq_small')}
    P = out['pooled']['all']
    out['predictions']['P135'] = dict(qq_all=P['qq_all'], holds=bool(P['qq_all'] >= 0.7))
    out['predictions']['P136'] = dict(qq_big=P['qq_big'], qq_small=P['qq_small'], holds=bool(P['qq_big'] > P['qq_small']))
    out['counts'] = {X: int(sum(v[X] > 0.5 for v in out['per_net'].values())) for X in ('qq_all', 'qq_big', 'qq_small')} | dict(n=len(keys))
    json.dump(out, open('results/e25.json', 'w'), indent=1)
    print(out['predictions']); print(out['pooled']); print(out['counts'])
    T = ['| set | all four terms | big-error term only | other three terms only |', '|---|---|---|---|']
    for name, v in out['pooled'].items():
        T.append(f"| {name.replace('_', ' ')} | {v['qq_all']:+.2f} | {v['qq_big']:+.2f} | {v['qq_small']:+.2f} |")
    open('results/e25_tables.md', 'w').write('\n'.join(T) + '\n')


if __name__ == '__main__':
    main()
