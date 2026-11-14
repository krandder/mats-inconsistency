"""E35 analysis (PREREG-B amendment 43). Window medians of block 2's query attention to the labelled tokens: W1 = 10,000-15,000 steps,
W2 = 20,000-30,000 steps (inclusive), and the final checkpoint.
  P162  A lineage (s6): median in W2 <= 0.7 x median in W1; B lineage (s5): >= 0.9 x.
  check (not a prediction; known from E34's bases, s6 0.256 and s5 0.387): the final checkpoints' labelled masses.
Also: the full trajectories, with E31's xq and sign-flip cost for comparison (results/e35_tables.md, results/e35_dyn.png).
usage: python e35_summ.py
"""
import json
import numpy as np


def med(T, lo, hi):
    return float(np.median([t['lab'] for t in T if lo <= t['step'] <= hi]))


def main():
    r = json.load(open('results/e35.json')); e31 = json.load(open('results/e31.json'))
    out = {}
    for s, lab in (('5', 'B'), ('6', 'A')):
        T = r[s]; w1, w2 = med(T, 10000, 15000), med(T, 20000, 30000)
        out[lab] = dict(W1=w1, W2=w2, ratio=w2 / w1, final=T[-1]['lab'], final_step=T[-1]['step'])
    pr = dict(P162=dict(A_ratio=out['A']['ratio'], B_ratio=out['B']['ratio'], holds=bool(out['A']['ratio'] <= 0.7 and out['B']['ratio'] >= 0.9)),
              check_final=dict(A_final=out['A']['final'], B_final=out['B']['final']))
    md = ['# E35 tables', '', '| lineage | step | block-2 query attention: labelled | summary | block-1 xq (E31) | sign-flip cost (E31) |', '|---' * 6 + '|']
    for s, lab in (('5', 'B'), ('6', 'A')):
        x31 = {t['step']: t for t in e31[lab]['traj']}
        for t in r[s]:
            if t['step'] in (0, 1000, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 12000, 15000, 17000, 20000, 25000, 30000, 40000, 60000, 80000, 89000, 90000):
                e = x31.get(t['step'], {})
                md.append('| %s | %d | %.3f | %.3f | %s | %s |' % (lab, t['step'], t['lab'], t['summ'], '%.4f' % e['xq'] if e else '-',
                                                                 '%.5f' % e['pc_flip_ind'] if e else '-'))
    md += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    open('results/e35_tables.md', 'w').write('\n'.join(md) + '\n')
    json.dump(dict(windows=out, predictions=pr), open('results/e35_summary.json', 'w'), indent=1)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for s, lab in (('5', 'B'), ('6', 'A')):
            st = [t['step'] for t in r[s]]
            ax[0].plot(st, [t['lab'] for t in r[s]], label=f'{lab}: block-2 query to labelled')
            ax[0].plot(st, [t['summ'] for t in r[s]], '--', label=f'{lab}: block-2 query to summaries')
            T = e31[lab]['traj']; ax[1].plot([t['step'] for t in T], [t['xq'] for t in T], label=f'{lab}: block-1 xq')
        for a in ax:
            a.set_xscale('symlog', linthresh=1000); a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig('results/e35_dyn.png', dpi=110)
    except Exception as ex:
        print('plot failed', ex)
    print(json.dumps(pr, indent=1))


if __name__ == '__main__':
    main()
