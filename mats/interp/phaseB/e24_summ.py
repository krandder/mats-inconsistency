"""E24 analysis (PREREG-B amendment 31): share of the big-miss weight difference removed by each transplant from the spread version.
removed(X) = Σ (w_ref − w_X) ÷ Σ (w_ref − w_spread) over the large-range rows (and the middle range), pooled over nets and per net.
usage: python e24_summ.py -> results/e24.json, results/e24_tables.md
"""
import json
import numpy as np

A_BASES = ['w32_h2_L2_mix4_s0', 'w32_h2_L2_mix4_s1', 'w32_h2_L2_mix4_s2', 'w32_h2_L2_mix4_ws_s1', 'w32_h2_L2_mix4_ws_s4', 'w32_h2_L2_mix4_s6_nb']
B_BASES = ['w32_h2_L2_mix4_ws_s2', 'w32_h2_L2_mix4_ws_s0', 'w32_h2_L2_mix4_ws_s3', 'w32_h2_L2_mix4_s4', 'w32_h2_L2_mix4_s5_nb', 'w32_h2_L2_mix4_s7_nb']


def main():
    z = np.load('results/e12_pairs.npz'); rg = z['range']
    nets = json.load(open('results/e22_nets.json')); D = {}
    for r in nets:
        D[r['tag']] = dict(np.load(f"results/e24/{r['tag']}.npz"))
    out = dict(per_net={}, pooled={}, predictions={})
    for rr in ('large', 'middle'):
        s = rg == rr
        def removed(keys, X):
            num = sum((D[k]['ref'][s] - D[k][X][s]).sum() for k in keys); den = sum((D[k]['ref'][s] - D[k]['spread'][s]).sum() for k in keys)
            return float(num / den)
        keys = list(D)
        out['pooled'][rr] = {X: removed(keys, X) for X in ('att', 'val', 'both')}
        out['pooled'][rr]['ref_diff_mean'] = float(np.mean([(D[k]['ref'][s] - D[k]['spread'][s]).mean() for k in keys]))
        for grp, L in (('A_bases', A_BASES), ('B_bases', B_BASES)):
            out['pooled'][f'{rr}_{grp}'] = {X: removed(L, X) for X in ('att', 'val', 'both')}
        for k in keys:
            out['per_net'].setdefault(k, {})[rr] = {X: removed([k], X) for X in ('att', 'val', 'both')}
    pl = out['pooled']['large']
    out['predictions']['P133'] = dict(both=pl['both'], holds=bool(pl['both'] >= 0.7))
    out['predictions']['P134'] = dict(val=pl['val'], att=pl['att'], holds=bool(pl['val'] > pl['att']))
    n = len(out['per_net'])
    out['counts'] = {X: int(sum(v['large'][X] > 0.5 for v in out['per_net'].values())) for X in ('att', 'val', 'both')} | dict(n=n)
    json.dump(out, open('results/e24.json', 'w'), indent=1)
    print(out['predictions']); print(out['pooled']); print(out['counts'])
    T = ['| set | range | removed by attention row | removed by values | removed by both |', '|---|---|---|---|---|']
    for key in ('large', 'large_A_bases', 'large_B_bases', 'middle', 'middle_A_bases', 'middle_B_bases'):
        v = out['pooled'][key]
        T.append(f"| {key.replace('_', ' ')} | | {v['att']:+.2f} | {v['val']:+.2f} | {v['both']:+.2f} |")
    open('results/e24_tables.md', 'w').write('\n'.join(T) + '\n')


if __name__ == '__main__':
    main()
