"""E0 table for the report: per layer, mean (and range) over nets 113-116 of the descriptive statistics.
python scratch/e0_table.py runs/e0"""
import json, sys, glob
import numpy as np

fs = sorted(glob.glob(f'{sys.argv[1]}/e0_256_64_11[3-6].json'))
D = [json.load(open(f)) for f in fs]
print(f'nets: {[d["seed"] for d in D]}\n')
print('| layer | spread | spread x l | PR of Cov_nu | top-1 share | sd log rho | PC1 skew / ex. kurt (signed) | PC2 skew / ex. kurt | '
      'gate-relevant neurons | their median abs skew / ex. kurt (t) | all neurons, t | all neurons, z | R^2 top-4 / top-16 |')
print('|---' * 13 + '|')
for l in (8, 16, 24, 32, 40, 48, 56, 63):
    r = [d['layers'][str(l)] for d in D]
    m = lambda k: np.mean([x[k] for x in r])
    rng = lambda k: (min(x[k] for x in r), max(x[k] for x in r))
    pc = lambda i, k: np.mean([x[k][i] for x in r])
    r2 = lambda k: np.mean([x['r2_topk'][str(k)] for x in r])
    print(f"| {l} | {m('spread'):.3f} | {m('spread_x_l'):.1f} ({rng('spread_x_l')[0]:.1f}-{rng('spread_x_l')[1]:.1f}) | "
          f"{m('pr'):.1f} | {np.mean([x['top_share'][0] for x in r]):.2f} | {m('logrho_sd'):.3f} | "
          f"{pc(0, 'pc_skew'):+.2f} / {pc(0, 'pc_kurt'):+.2f} | {pc(1, 'pc_skew'):+.2f} / {pc(1, 'pc_kurt'):+.2f} | "
          f"{m('nb_count_t'):.0f} ({rng('nb_count_t')[0]}-{rng('nb_count_t')[1]}) | "
          f"{m('t_nb_abs_skew_med'):.2f} / {m('t_nb_kurt_med'):+.2f} | {m('t_all_abs_skew_med'):.2f} / {m('t_all_kurt_med'):+.2f} | "
          f"{m('z_all_abs_skew_med'):.2f} / {m('z_all_kurt_med'):+.2f} | {r2(4):.2f} / {r2(16):.2f} |")
print('\nPer-net PC1 excess kurtosis by layer (signed scores, third moment >= 0):')
for d in D:
    print(d['seed'], ' '.join(f"{l}:{d['layers'][str(l)]['pc_kurt'][0]:+.2f}" for l in (16, 24, 32, 40, 48, 56, 63)))
print('Per-net fourth-order dependence among the top-5 components (median / max |corr| of squared scores):')
for d in D:
    print(d['seed'], ' '.join(f"{l}:{d['layers'][str(l)]['sqcorr_top5_med']:.2f}/{d['layers'][str(l)]['sqcorr_top5_max']:.2f}"
                              for l in (16, 32, 48, 63)))
