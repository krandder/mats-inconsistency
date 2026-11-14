"""Figure for E37 (descriptive; reads results/e37.json and results/e37_traj/*.npz) -> results/e37_fig.png.
Panel 1: R (how mag-like the final KL is; 1 = as mag from scratch, 0 = as ctrl) by switch step, ctrl->mag and mag->ctrl, with
95% intervals; mag60 (the learning-speed control) as a separate marker.
Panel 2: final block-2 query attention mass on the labelled tokens per arm (one dot per seed; bar = median).
Panel 3: median test KL over seeds in the format in force, by checkpoint, for ctrl, mag and the ctrl->mag arms.
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#1f1f1e', '#8a8986'


def main():
    res = json.load(open('results/e37.json'))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    plt.rcParams.update({'font.size': 10})
    a = ax[0]
    for pre, col, lab in (('c2m', BLUE, 'ctrl then mag'), ('m2c', ORANGE, 'mag then ctrl')):
        pts = sorted((int(k[3:-1]) * 1000, v['value']) for k, v in res['R'].items() if k.startswith(pre) and v['value'])
        if pts:
            x = [p[0] for p in pts]; y = [p[1][0] for p in pts]
            lo = [p[1][0] - p[1][1] for p in pts]; hi = [p[1][2] - p[1][0] for p in pts]
            a.errorbar(x, y, yerr=[lo, hi], color=col, marker='o', ms=7, lw=2, capsize=3, label=lab)
    m60 = (res['R'].get('mag60') or {}).get('value')
    if m60:
        a.errorbar([30000 * 1.15], [m60[0]], yerr=[[m60[0] - m60[1]], [m60[2] - m60[0]]], color=AQUA, marker='D', ms=7, capsize=3,
                   lw=2, label='mag, 60k-step budget')
    a.axhline(1, color=MUTED, ls='--', lw=1); a.axhline(0, color=MUTED, ls='--', lw=1)
    a.text(2050, 1.04, 'as mag from scratch', color=MUTED, fontsize=8); a.text(2050, 0.04, 'as ctrl', color=MUTED, fontsize=8)
    a.set_xscale('log'); a.set_xticks([2000, 3000, 6000, 10000, 30000]); a.set_xticklabels(['2k', '3k', '6k', '10k', '30k'])
    a.minorticks_off(); a.set_xlabel('switch step K'); a.set_ylabel('R (final KL: 0 = ctrl, 1 = mag)')
    a.set_title('How mag-like the final KL is'); a.legend(frameon=False, fontsize=8, loc='lower left')
    a = ax[1]
    arms = ['ctrl', 'mag', 'c2m2k', 'c2m3k', 'c2m6k', 'c2m10k', 'c2m30k', 'm2c3k', 'm2c10k', 'm2c30k', 'mag60']
    for i, arm in enumerate(arms):
        v = [o['lab_last'] for k, o in res['per_net'].items() if k.rsplit('_s', 1)[0] == arm and o['gap'] < 0.10]
        if not v:
            continue
        col = BLUE if arm.startswith('c2m') else ORANGE if arm.startswith('m2c') else INK
        a.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=14, color=col, alpha=0.8)
        a.hlines(np.median(v), i - 0.3, i + 0.3, color=col, lw=2)
    a.set_xticks(range(len(arms))); a.set_xticklabels(arms, rotation=45, ha='right', fontsize=8)
    a.set_ylabel('block-2 query attention on labelled tokens'); a.set_title('Routing marker at the end (E34/E35)')
    a = ax[2]
    for arm, col, ls in (('ctrl', INK, '-'), ('mag', MUTED, '-'), ('c2m3k', BLUE, ':'), ('c2m10k', BLUE, '--'), ('c2m30k', BLUE, '-')):
        T = res['trajectories'].get(arm, {})
        st = sorted(int(s) for s in T if int(s) >= 1000)
        if st:
            a.plot(st, [T[str(s)]['kl'] for s in st], color=col, ls=ls, lw=2, label=arm)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('training step'); a.set_ylabel('median test KL (format in force)')
    a.set_title('KL over training'); a.legend(frameon=False, fontsize=8)
    for x in ax:
        x.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig('results/e37_fig.png', dpi=130)
    print('results/e37_fig.png')


if __name__ == '__main__':
    main()
