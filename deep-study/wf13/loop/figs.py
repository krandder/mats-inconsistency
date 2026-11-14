"""figs.py -- three plain figures from the registered outputs. -> fig/fig1_sudoku.png, fig/fig2_parity.png, fig/fig3_scale.png"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = '/home/agent/workspaces/mats/work/deep-study/wf13/loop/'
# ---- figure 1: Sudoku, openings needed to close the interval, by round
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, sn in enumerate(('hard', 'mixed')):
    for fn, ls in zip(sorted(glob.glob(HERE + 'out/s_s[12].json')), ('-', '--')):
        r = json.load(open(fn))
        if 'arms' not in r or 'final_mrvorder' not in r['arms']:
            continue
        A, R = r['arms'], r['cfg']['R']
        ax[k].plot(range(R + 1), [A['round%d' % i][sn]['opened_censored_mean'] for i in range(R + 1)], 'o' + ls, color='C0',
                   label='loop, seed %d' % r['seed'])
        ax[k].axhline(A['oneshot'][sn]['opened_censored_mean'], color='C1', ls=ls, label='one-shot control, seed %d' % r['seed'])
        ax[k].axhline(A['hand_mrv'][sn]['opened_censored_mean'], color='C2', ls=ls, label='hand-made order (fewest candidates)' if ls == '-' else None)
        ax[k].axhline(A['bad_random'][sn]['opened_censored_mean'], color='C3', ls=ls, label='random order' if ls == '-' else None)
    ax[k].set_yscale('log'); ax[k].set_xlabel('round of distillation'); ax[k].set_title('%s test puzzles' % sn)
    ax[k].set_ylabel('openings to close the interval (mean)')
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(HERE + 'fig/fig1_sudoku.png', dpi=130); plt.close(fig)

# ---- figure 2: parity, error and interval width against compute (network calls), per world
fs = sorted(glob.glob(HERE + 'out/p_w10[1-3]_s1.json'))
fig, ax = plt.subplots(1, max(len(fs), 1), figsize=(4.8 * max(len(fs), 1), 4.4), squeeze=False)
for k, fn in enumerate(fs):
    r = json.load(open(fn))
    if 'baselines' not in r:
        continue
    A, mk, km, R = r['arms'], r['cfg']['marks'], r['cfg']['kmarks'], r['cfg']['R']
    a = ax[0][k]
    fin, r0, r1 = A['round%d' % R], A['round0'], A['round1']
    a.plot(mk, fin['w_aud'], 's-', color='0.5', label='audit interval width (final policy)')
    a.plot(mk, fin['w_cert'], 's:', color='0.5', label='certain interval width (final policy)')
    a.plot(mk, fin['err_aud'], 'o-', color='C0', label='tree: audit point, final policy')
    a.plot(mk, A['hand_spec']['err_aud'], 'o--', color='C2', label='tree: specification alone (0 calls; x = openings)')
    for arm, c, lab in ((r0, 'C3', 'importance form, untrained'), (r1, 'C1', 'importance form, round 1'), (fin, 'C4', 'importance form, final')):
        a.plot(arm['is_calls'], arm['is_err'], '^-', color=c, label=lab)
    a.axhline(r['baselines']['onepass_M_abs'], color='k', ls='--', lw=0.8, label='one pass of M (cost of 41 calls)')
    a.axhline(fin['onepass_abs'], color='C4', ls=':', lw=1.0, label="final policy's own one pass (1 call)")
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('calls of the rollout policy per instance')
    a.set_ylabel('mean abs error, or interval width'); a.set_title('parity world %d' % r['world'])
ax[0][0].legend(fontsize=6, loc='lower left')
fig.tight_layout(); fig.savefig(HERE + 'fig/fig2_parity.png', dpi=130); plt.close(fig)

# ---- figure 3: scale
try:
    r = json.load(open(HERE + 'out/pscale.json'))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for k, hid in enumerate(sorted(r['hidden'], key=int)):
        v = r['hidden'][hid]['spec_alone']
        for key, ls, lab in (('w_cert', 'o-', 'certain'), ('w_aud4k', 's--', 'audit, 4,096 draws'), ('w_aud64k', '^:', 'audit, 65,536 draws')):
            ax[k].plot(r['marks'], [np.median([i[a][key] for i in v]) for a in range(len(r['marks']))], ls, label=lab)
        ax[k].set_xscale('log'); ax[k].set_title('%s hidden assumptions' % hid); ax[k].set_xlabel('openings (specification alone)')
    ax[0].set_ylabel('median interval width'); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(HERE + 'fig/fig3_scale.png', dpi=130); plt.close(fig)
except FileNotFoundError:
    pass
print('figures written')
