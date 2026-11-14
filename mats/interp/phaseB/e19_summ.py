"""E19 analysis (PREREG-B amendment 26): does a per-token MLP before the first attention remove the departure, where the same MLP
after the last block does not? Baselines: the same-seed recipe nets base_mix4_s5-7 (E15 values, recomputed from results/e15).
usage: python e19_summ.py -> results/e19.json, results/e19_tables.md
"""
import json, os
import numpy as np

SEEDS = (5, 6, 7)
NBOOT = 1000


def stats(q, z):
    src, rg = z['src'], z['range']; rows = np.arange(len(src)); Wb = z['WB0'][rows, src]; oth = np.stack([(src + 1) % 3, (src + 2) % 3], 1)
    dw = q['ws1'][rows, src] - q['ws0'][rows, src]; dwo = (q['ws1'][rows[:, None], oth] - q['ws0'][rows[:, None], oth]).mean(1)
    WT = np.load('results/qweights_world.npz', allow_pickle=True)['test'].item()['WB']
    d = dict(gap=float(q['gap']), slope=float(np.cov(q['test_w'].ravel(), WT.ravel())[0, 1] / np.var(WT.ravel(), ddof=1)))
    for r in ('large', 'middle'):
        s = rg == r
        d[r] = dict(rom=float(dw[s].sum() / Wb[s].sum()), spec=float((dw[s].sum() - dwo[s].sum()) / Wb[s].sum()), mean_dw=float(dw[s].mean()))
    return d, dw


def main():
    z = np.load('results/e12_pairs.npz'); base, rg = z['base'], z['range']
    P, DW = {}, {}
    for s in SEEDS:
        P[('base', s)], DW[('base', s)] = stats(np.load(f'results/e15/w32_h2_L2_mix4_s{s}_nb__best.npz'), z)
        for v in ('pre', 'post'):
            f = f'results/e19/var_{v}_s{s}__best.npz'
            if os.path.exists(f):
                P[(v, s)], DW[(v, s)] = stats(np.load(f), z)
    acq = {k: v['gap'] < 0.10 for k, v in P.items()}
    ub = np.arange(3000); idx = {b: np.where(base == b)[0] for b in ub}
    rng = np.random.default_rng(0); boots = [np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]) for _ in range(NBOOT)]

    def pooled(keys):
        sel = rg == 'large'
        def st(ii):
            jj = ii[sel[ii]]; return float(np.mean([DW[k][jj].mean() for k in keys]))
        bs = [st(ii) for ii in boots]
        return dict(mean_dw=st(np.arange(len(rg))), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])
    out = dict(per_net={f'{k[0]}_s{k[1]}': v for k, v in P.items()}, acquired={f'{k[0]}_s{k[1]}': bool(v) for k, v in acq.items()}, arms={}, predictions={})
    for v in ('base', 'pre', 'post'):
        keys = [(v, s) for s in SEEDS if (v, s) in P and acq[(v, s)]]
        if keys:
            out['arms'][v] = dict(n=len(keys), rom_large=float(np.mean([P[k]['large']['rom'] for k in keys])),
                                  rom_middle=float(np.mean([P[k]['middle']['rom'] for k in keys])), gap=float(np.mean([P[k]['gap'] for k in keys])),
                                  slope=float(np.mean([P[k]['slope'] for k in keys])), **pooled(keys))
    A = out['arms']
    pairs_pre = [s for s in SEEDS if ('pre', s) in P and acq[('pre', s)] and acq[('base', s)]]
    pairs_post = [s for s in SEEDS if ('post', s) in P and acq[('post', s)] and acq[('base', s)]]
    if len(pairs_pre) >= 2:
        bm = np.mean([P[('base', s)]['large']['rom'] for s in pairs_pre]); pm = np.mean([P[('pre', s)]['large']['rom'] for s in pairs_pre])
        nlow = sum(P[('pre', s)]['large']['rom'] < P[('base', s)]['large']['rom'] for s in pairs_pre)
        out['predictions']['P118'] = dict(pre=float(pm), base=float(bm), ratio=float(pm / bm), n_lower=int(nlow), n=len(pairs_pre),
                                          holds=bool(pm < 0.5 * bm and nlow == len(pairs_pre)))
    if len(pairs_post) >= 2:
        bm = np.mean([P[('base', s)]['large']['rom'] for s in pairs_post]); qm = np.mean([P[('post', s)]['large']['rom'] for s in pairs_post])
        out['predictions']['P119'] = dict(post=float(qm), base=float(bm), ratio=float(qm / bm), holds=bool(qm >= 0.75 * bm))
    json.dump(out, open('results/e19.json', 'w'), indent=1)
    for k, v in out['predictions'].items():
        print(k, v)
    T = ['| net | acquired | test gap | slope on W | rom large | rom middle | spec large |', '|---|---|---|---|---|---|---|']
    for (v, s), d in sorted(P.items(), key=lambda t: (t[0][1], t[0][0])):
        T.append(f"| {v} s{s} | {acq[(v, s)]} | {d['gap']:.4f} | {d['slope']:.2f} | {d['large']['rom']:+.3f} | {d['middle']['rom']:+.3f} | {d['large']['spec']:+.3f} |")
    T += ['', '| arm | nets | gap | rom large | mean Δw large [95% CI] | rom middle |', '|---|---|---|---|---|---|']
    for v, d in A.items():
        T.append(f"| {v} | {d['n']} | {d['gap']:.4f} | {d['rom_large']:+.3f} | {d['mean_dw']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] | {d['rom_middle']:+.3f} |")
    open('results/e19_tables.md', 'w').write('\n'.join(T) + '\n')


if __name__ == '__main__':
    main()
