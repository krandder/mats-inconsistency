"""Tables from results/tokan.jsonl (perm, M2, M3, M5, M6) and results/m1.jsonl (M1)."""
import json, sys
from collections import OrderedDict

T = [json.loads(l) for l in open('results/tokan.jsonl')]
print('token-net analyses (W0)')
for r in T:
    name = r['run'].rstrip('/').split('/')[-1]
    print(f'\n{name}: KL to Bayes {r["kl_bayes"]:.4f}; permutation max output change {r["perm_maxdiff"]:.1e}; '
          f'M6 other/own {r["M6_other_over_own"]:.3f}')
    for m in r['M2']:
        print(f'  M2 layer {m["layer"]} head {m["head"]}: classes C(TV<=.05/.1/.2) = {m["C"]["0.05"]:.1f}/{m["C"]["0.1"]:.1f}/'
              f'{m["C"]["0.2"]:.1f} of {r["N"] * 5} rows; per-key share {m["col_share"]:.2f}; label R^2 {m["label_r2"]:.2f}')
    print('  M2 last layer, query log mass on query readings vs log E[1/v|resid], r by head:',
          [round(x, 2) for x in r['M2_query_logmass_vs_logprec_r']])
    print('  M5 query self-attention vs predictive variance, R^2 (net, Bayes) by head:',
          [(round(m['r2_net_var'], 2), round(m['r2_bayes_var'], 2), round(m['mean_a0'], 3)) for m in r['M5']])
    print('  M3 KL to net / to Bayes:', {k: (round(v['kl_to_net'], 3), round(v['kl_to_bayes'], 3)) for k, v in r['M3'].items()})
print('\nM1 interaction shares (net, Bayes, deviation): source cut | within-source cut')
for l in open('results/m1.jsonl'):
    r = json.loads(l)
    name = r['run'].rstrip('/').split('/')[-1]
    print(f'  {name:22s} src {r["src_net_share"]:.4f} {r["src_bayes_share"]:.1e} {r.get("src_dev_share", float("nan")):.3f} | '
          f'split {r["split_net_share"]:.4f} {r["split_bayes_share"]:.4f} {r.get("split_dev_share", float("nan")):.3f}')
