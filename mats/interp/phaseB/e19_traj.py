"""E19 descriptive (amendment 26, "also reported"): large-range rom over training for the pre-MLP nets at seeds 5 and 6 against
E16's baseline trajectories of the same seeds (same shared-weight initialisation and data stream).
usage: python e19_traj.py -> results/e19_traj.json, results/e19_traj.png
"""
import glob, json, os
import numpy as np


def series(files):
    z = np.load('results/e12_pairs.npz'); src, rg = z['src'], z['range']; rows = np.arange(len(src)); Wb = z['WB0'][rows, src]
    oth = np.stack([(src + 1) % 3, (src + 2) % 3], 1)
    WT = np.load('results/qweights_world.npz', allow_pickle=True)['test'].item()['WB']
    out = []
    for f in files:
        q = np.load(f); step = int(os.path.basename(f).split('step')[-1].split('.')[0].lstrip('_'))
        dw = q['ws1'][rows, src] - q['ws0'][rows, src]; dwo = (q['ws1'][rows[:, None], oth] - q['ws0'][rows[:, None], oth]).mean(1)
        s, m = rg == 'large', rg == 'middle'
        out.append(dict(step=step, gap=float(q['gap']), rom_large=float(dw[s].sum() / Wb[s].sum()), rom_middle=float(dw[m].sum() / Wb[m].sum()),
                        spec_large=float((dw[s].sum() - dwo[s].sum()) / Wb[s].sum()),
                        slope=float(np.cov(q['test_w'].ravel(), WT.ravel())[0, 1] / np.var(WT.ravel(), ddof=1))))
    return sorted(out, key=lambda d: d['step'])


def first(S, cond, k=3):
    for i in range(len(S) - k + 1):
        if all(cond(S[j]) for j in range(i, i + k)):
            return S[i]['step']
    return None


def main():
    res = {}
    for s in (5, 6):
        pre = series(sorted(glob.glob(f'results/e19/var_pre_s{s}__step_*.npz')))
        base = series(sorted(glob.glob(f'results/e16/s{s}_step*.npz')))
        r = {}
        for name, S in (('pre', pre), ('base', base)):
            on = first(S, lambda d: d['spec_large'] > 0.05); post = [d for d in S if on is not None and d['step'] >= on]
            r[name] = dict(onset=on, slope06=first(S, lambda d: d['slope'] >= 0.6), gap_lt_010=first(S, lambda d: d['gap'] < 0.10),
                           rom_at_onset=post[0]['rom_large'] if post else None, rom_peak=max(d['rom_large'] for d in post) if post else None,
                           rom_final=S[-1]['rom_large'], step_final=S[-1]['step'],
                           rom_at={st: next((d['rom_large'] for d in S if d['step'] == st), None) for st in (5000, 10000, 20000, 40000, 80000)})
        res[s] = r
        res[f'{s}_series'] = dict(pre=pre, base=base)
        print(s, {k: v for k, v in r.items()})
    json.dump(res, open('results/e19_traj.json', 'w'), indent=1)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, s in enumerate((5, 6)):
        for name, c in (('base', 'C3'), ('pre', 'C0')):
            S = res[f'{s}_series'][name]
            ax[i].plot([d['step'] for d in S if d['step'] > 0], [d['rom_large'] for d in S if d['step'] > 0], color=c,
                       label={'base': 'baseline (E16 replay)', 'pre': 'per-token MLP before attention'}[name])
        ax[i].set_xscale('log'); ax[i].axhline(0, color='grey', lw=0.5); ax[i].set_title(f'seed {s}: rom, large errors'); ax[i].set_xlabel('step')
    ax[0].legend(); fig.tight_layout(); fig.savefig('results/e19_traj.png', dpi=110)


if __name__ == '__main__':
    main()
