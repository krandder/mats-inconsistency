"""E54 post hoc (not registered; written after seeing E54's registered results): the scale effect dls by the size of the residual.
For each world-2 net, dls = log-sd_big - log-sd_spread averaged within quartiles of the pairs' residual sum of squares (RSS;
quartile edges per D over the 4,000 bases), and the lam coefficient within the bottom and top halves of RSS. Added after the
review: within each true noise level, dls above minus below that level's median RSS (RSS quartiles also differ in noise level).
usage: python e54_post.py -> results/e54_post.json
"""
import collections, glob, json
import numpy as np

P = {D: np.load('results/e54_pairs_d%d.npz' % D) for D in (1, 2)}
G = collections.defaultdict(list)
for f in sorted(glob.glob('results/e54/*.npz')):
    z = np.load(f, allow_pickle=True); D = int(z['D'])
    arm = ('pre' if str(z['variant']) == 'pre' else str(z['fmt'])) + '_d%d' % D
    rss = P[D]['rss']; b = np.digitize(rss, np.quantile(rss, [0.25, 0.5, 0.75]))
    dls = z['ls_big'] - z['ls_spr']
    mB = (P[D]['bayes_wt'] * P[D]['bayes_mu']).sum(1); dm = z['m_big'] - z['m_spr']
    lam = []
    for half in (b <= 1, b >= 2):
        A = np.c_[P[D]['dshift'][half], mB[half], np.ones(half.sum())]
        lam.append(float(np.linalg.lstsq(A, dm[half], rcond=None)[0][0]))
    wl = []                                                   # within each true noise level: dls above minus below that level's median RSS
    for j in range(3):
        s = P[D]['li'] == j; med = np.median(rss[s])
        wl.append(float(dls[s & (rss > med)].mean() - dls[s & (rss <= med)].mean()))
    G[arm].append(dict(q=[float(dls[b == k].mean()) for k in range(4)], lam_low=lam[0], lam_high=lam[1], within_level=wl))
out = {}
for a, v in G.items():
    q = np.array([x['q'] for x in v])
    out[a] = dict(n=len(v), dls_by_rss_quartile=q.mean(0).tolist(), n_top_below_bottom=int((q[:, 3] < q[:, 0]).sum()),
                  n_top_negative=int((q[:, 3] < 0).sum()), n_top_below_q2=int((q[:, 3] < q[:, 1]).sum()),
                  lam_low_half=float(np.mean([x['lam_low'] for x in v])), lam_high_half=float(np.mean([x['lam_high'] for x in v])),
                  n_q2_pos_q4_neg=int(((q[:, 1] > 0) & (q[:, 3] < 0)).sum()),
                  within_level_high_minus_low=np.mean([x['within_level'] for x in v], 0).tolist(),
                  n_within_level_negative=[int(sum(x['within_level'][j] < 0 for x in v)) for j in range(3)])
json.dump(out, open('results/e54_post.json', 'w'), indent=1)
for a, o in out.items():
    print('%-11s' % a, ' '.join('%+.4f' % x for x in o['dls_by_rss_quartile']), '| top < bottom %d/12, top < 0 %d/12 | lam low/high half %.3f / %.3f'
          % (o['n_top_below_bottom'], o['n_top_negative'], o['lam_low_half'], o['lam_high_half'])
          + ' | Q2>0&Q4<0 %d | within level (0.1/0.5/2.0) high-low %s, nets < 0 %s' % (o['n_q2_pos_q4_neg'],
            ' '.join('%+.4f' % x for x in o['within_level_high_minus_low']), o['n_within_level_negative']))
