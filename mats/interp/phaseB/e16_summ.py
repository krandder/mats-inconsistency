"""E16 analysis (PREREG-B amendment 23): training dynamics of the clean bases base_mix4_s5 (B lineage) and s6 (A lineage).
usage: python e16_summ.py -> results/e16.json, results/e16_tables.md, results/e16_curves.png
"""
import glob, json, os
import numpy as np
from scipy.stats import spearmanr, pearsonr
import nnoise  # noqa: F401

TP = '/home/agent/workspaces/mats/work/tiny-pfn'
LIN = {5: 'B', 6: 'A'}


def r2_pwl(x, y, fit, test):
    """Held-out R^2 of y from a continuous piecewise-linear function of x, knots at the fit-set deciles (least squares)."""
    kn = np.quantile(x[fit], np.linspace(0.1, 0.9, 9))
    F = lambda v: np.c_[np.ones_like(v), v, np.maximum(v[:, None] - kn[None], 0)]
    b = np.linalg.lstsq(F(x[fit]), y[fit], rcond=None)[0]
    pr = F(x[test]) @ b
    return float(1 - ((y[test] - pr) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum())


def series(seed):
    z = np.load('results/e12_pairs.npz'); src, rg = z['src'], z['range']; m = len(src); rows = np.arange(m); Wb = z['WB0'][rows, src]
    oth = np.stack([(src + 1) % 3, (src + 2) % 3], 1)
    wt = np.load('results/qweights_world.npz', allow_pickle=True)['test'].item(); WT, mB = wt['WB'], wt['mB']
    ss = np.load(os.path.join(TP, 'runs/testset.npz'))['ss_resid'][:4000]
    x = np.log(ss.ravel()); fit = np.arange(x.size) < 6000; test = ~fit                # episodes 0-1999 fit, 2000-3999 held out
    S = []
    for p in sorted(glob.glob(f'results/e16/s{seed}_step*.npz')):
        q = np.load(p); d = dict(step=int(q['step']), gap=float(q['gap']))
        dw = q['ws1'][rows, src] - q['ws0'][rows, src]; dwo = (q['ws1'][rows[:, None], oth] - q['ws0'][rows[:, None], oth]).mean(1)
        for r in ('large', 'middle'):
            s = rg == r
            d[f'rom_{r}'] = float(dw[s].sum() / Wb[s].sum()); d[f'spec_{r}'] = float((dw[s].sum() - dwo[s].sum()) / Wb[s].sum())
            d[f'tot_{r}'] = float(q['ws1'][s].sum() / q['ws0'][s].sum() - 1)
            d[f'gsph_{r}'] = float((q['gs1'] - q['gs0'])[s].sum() / q['gs0'][s].sum())
        w = q['test_w'].ravel(); d['slope'] = float(np.cov(w, WT.ravel())[0, 1] / np.var(WT.ravel(), ddof=1))
        d['code_r2'] = r2_pwl(x, q['test_gs'].ravel(), fit, test)
        d['dev'] = q['test_m'] - mB
        S.append(d)
    return S


def first(S, cond, k=1):
    """Step of the first checkpoint from which cond holds at k consecutive checkpoints (S sorted by step)."""
    for i in range(len(S) - k + 1):
        if all(cond(S[j]) for j in range(i, i + k)):
            return S[i]['step']
    return None


def main():
    L = {s: series(s) for s in (5, 6)}
    out = dict(lineages={}, predictions={})
    meta = {s: json.load(open(f'runs/base_mix4_s{s}/meta.json')) for s in (5, 6)}
    for s, S in L.items():
        best = meta[s]['best_step']; dfin = [d for d in S if d['step'] == best][0]
        t_acq = first(S, lambda d: d['gap'] < 0.10, 3); t_dep = first(S, lambda d: d['spec_large'] > 0.05, 3)
        band = [d for d in S if d['step'] < t_acq and 0.14 <= d['gap'] <= 0.20]
        post = [d for d in S if d['step'] >= t_acq]; post1k = [d for d in S if d['step'] >= t_acq + 1000]
        rl = [d['rom_large'] for d in post]
        o = dict(group=LIN[s], best_step=best, t_acq=t_acq, t_dep=t_dep, final=dict((k, v) for k, v in dfin.items() if k != 'dev'),
                 band_n=len(band), band_steps=[d['step'] for d in band],
                 band_median_tot_large=float(np.median([d['tot_large'] for d in band])) if band else None,
                 band_median_spec_large=float(np.median([d['spec_large'] for d in band])) if band else None,
                 max_rom_large_post=float(max(rl)), step_max_rom_large=post[int(np.argmax(rl))]['step'],
                 spearman_rom_large_step_post1k=float(spearmanr([d['step'] for d in post1k], [d['rom_large'] for d in post1k])[0]),
                 first3_middle_post1k=[d['rom_middle'] for d in post1k[:3]],
                 frac_middle_pos_post1k=float(np.mean([d['rom_middle'] > 0 for d in post1k])),
                 spearman_gsph_rom_post=float(spearmanr([d['gsph_large'] for d in post], rl)[0]),
                 spearman_gap_rom_post1k=float(spearmanr([d['gap'] for d in post1k], [d['rom_large'] for d in post1k])[0]))
        fin_dev = dfin['dev']
        o['self_corr'] = [(d['step'], float(pearsonr(d['dev'], fin_dev)[0])) for d in S if d['step'] > 0]
        o['first_self_corr_0.8'] = first([dict(step=a, c=b) for a, b in o['self_corr']], lambda d: d['c'] > 0.8)
        neg = [d['step'] for d in post1k if d['rom_middle'] < 0]
        o['last_cross_to_negative_middle'] = None
        for i in range(1, len(post1k)):
            if post1k[i - 1]['rom_middle'] >= 0 and post1k[i]['rom_middle'] < 0:
                o['last_cross_to_negative_middle'] = post1k[i]['step']
        o['n_middle_negative_post1k'] = len(neg)
        out['lineages'][s] = o
    pr = out['predictions']; A, B = out['lineages'][6], out['lineages'][5]
    pr['P106'] = dict(diff={s: (o['t_dep'] - o['t_acq']) if o['t_dep'] is not None else None for s, o in out['lineages'].items()})
    pr['P106']['holds'] = bool(all(v is not None and abs(v) <= 500 for v in pr['P106']['diff'].values()))
    ok107 = [o['band_n'] >= 3 and o['band_median_tot_large'] > 0.02 and o['band_median_spec_large'] < 0 for o in out['lineages'].values()]
    pr['P107'] = dict(testable=bool(all(o['band_n'] >= 3 for o in out['lineages'].values())), holds=bool(all(ok107)))
    ok108 = [o['max_rom_large_post'] > 1.33 * o['final']['rom_large'] and o['spearman_rom_large_step_post1k'] < -0.5 for o in out['lineages'].values()]
    pr['P108'] = dict(holds=bool(all(ok108)))
    pr['P109'] = dict(A_first3_pos=bool(all(v > 0 for v in A['first3_middle_post1k'])), A_final_neg=bool(A['final']['rom_middle'] < 0),
                      B_frac_pos=B['frac_middle_pos_post1k'])
    pr['P109']['holds'] = bool(pr['P109']['A_first3_pos'] and pr['P109']['A_final_neg'] and B['frac_middle_pos_post1k'] >= 0.9)
    pr['P110'] = dict(holds=bool(all(o['spearman_gsph_rom_post'] > 0.5 for o in out['lineages'].values())))
    # P111: cross-lineage correlation of deviation patterns at matched steps
    d5 = {d['step']: d['dev'] for d in L[5]}; d6 = {d['step']: d['dev'] for d in L[6]}
    cross = [(st, float(pearsonr(d5[st], d6[st])[0])) for st in sorted(set(d5) & set(d6)) if st > 0]
    ta = max(A['t_acq'], B['t_acq'])
    early = [c for st, c in cross if ta + 1000 <= st <= ta + 5000]; final = dict(cross)[90000]
    out['cross_corr'] = cross
    pr['P111'] = dict(early_mean=float(np.mean(early)), n_early=len(early), final=final, holds=bool(np.mean(early) >= final + 0.1))
    json.dump(out, open('results/e16.json', 'w'), indent=1, default=float)
    for k, v in pr.items():
        print(k, v)
    for s, o in out['lineages'].items():
        print(s, {k: v for k, v in o.items() if k not in ('self_corr', 'final', 'band_steps')})
    # table of the trajectories at selected steps
    sel = [0, 500, 1000, 1500, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 5000, 7000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 85000, 90000]
    T = ['| step | lineage | gap | slope on W | rom large | rom middle | spec large | total weight large | Δg_sph/g large | code R² |', '|---|---|---|---|---|---|---|---|---|---|']
    for st in sel:
        for s in (6, 5):
            dd = [d for d in L[s] if d['step'] == st]
            if dd:
                d = dd[0]
                T.append(f"| {st} | {LIN[s]} (s{s}) | {d['gap']:.4f} | {d['slope']:.2f} | {d['rom_large']:+.3f} | {d['rom_middle']:+.3f} | {d['spec_large']:+.3f} | {d['tot_large']:+.3f} | {d['gsph_large']:+.2f} | {d['code_r2']:.2f} |")
    open('results/e16_tables.md', 'w').write('\n'.join(T) + '\n')
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        keys = [('gap', 'test gap to Bayes (nats)'), ('slope', 'slope of weights on Bayes weights'), ('rom_large', 'rom, large errors'),
                ('rom_middle', 'rom, middle errors'), ('spec_large', 'spec, large errors'), ('tot_large', 'total-weight change, large errors')]
        fig, ax = plt.subplots(2, 3, figsize=(13, 7))
        for a, (k, lab) in zip(ax.ravel(), keys):
            for s, c in ((6, 'C0'), (5, 'C3')):
                st = [d['step'] for d in L[s] if d['step'] > 0]; v = [d[k] for d in L[s] if d['step'] > 0]
                a.plot(st, v, color=c, label=f'{LIN[s]} lineage (s{s})')
            a.set_xscale('log'); a.set_title(lab); a.axhline(0, color='grey', lw=0.5)
        ax[0, 0].legend(); fig.tight_layout(); fig.savefig('results/e16_curves.png', dpi=110)
    except Exception as e:
        print('figure failed', e)


if __name__ == '__main__':
    main()
