"""E15 post-hoc descriptives (written after the registered E15 results and the review): saved so every reported number
has a source. usage: python e15_post.py -> results/e15_post.json
  - total-weight response: relative change of the summed effective weights of the three sources (big miss − spread) over
    large-range rows, sum(ws1) / sum(ws0) − 1, per net;
  - head count against the middle-range sign (acquired nets);
  - 2-head acquired nets: Spearman of the gap with the middle- and large-range rom; middle-range sign flips over the cooldown;
  - noisy minus base rom for the 12 base/noisy pairs.
"""
import json
import numpy as np
from scipy.stats import spearmanr


def main():
    o = json.load(open('results/e15.json')); pn = o['per_net']; nets = json.load(open('results/e15_nets.json'))
    rg = np.load('results/e12_pairs.npz')['range']; s = rg == 'large'
    acq = [r for r in nets if r['depth'] == 2 and pn[r['tag'] + '__best']['gap'] < 0.10]
    non = [r for r in nets if r not in acq]

    def tot(r):
        q = np.load(f"results/e15/{r['tag']}__best.npz")
        return float(q['ws1'][s].sum() / q['ws0'][s].sum() - 1)
    out = {}
    for name, L in (('non_acquired', non), ('acquired', acq)):
        v = np.array([tot(r) for r in L])
        out[f'total_weight_change_large_{name}'] = dict(n=len(v), median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                                                       n_pos=int((v > 0).sum()))
    for h in (1, 2):
        L = [r for r in acq if r['heads'] == h]
        out[f'heads{h}_middle_neg'] = dict(n=len(L), n_neg=int(sum(pn[r['tag'] + '__best']['middle']['rom'] < 0 for r in L)))
    h2 = [r for r in acq if r['heads'] == 2]
    g = [pn[r['tag'] + '__best']['gap'] for r in h2]
    mid = [pn[r['tag'] + '__best']['middle']['rom'] for r in h2]; lg = [pn[r['tag'] + '__best']['large']['rom'] for r in h2]
    fl = [(pn[r['tag'] + '__phase1']['middle']['rom'], pn[r['tag'] + '__best']['middle']['rom']) for r in h2]
    out['heads2'] = dict(n=len(h2), spearman_gap_middle=float(spearmanr(g, mid)[0]), spearman_gap_large=float(spearmanr(g, lg)[0]),
                         median_gap_middle_neg=float(np.median([x for x, m in zip(g, mid) if m < 0])),
                         median_gap_middle_pos=float(np.median([x for x, m in zip(g, mid) if m >= 0])),
                         phase1_neg=int(sum(a < 0 for a, b in fl)), final_neg=int(sum(b < 0 for a, b in fl)),
                         flips_pos_to_neg=int(sum(a >= 0 and b < 0 for a, b in fl)), flips_neg_to_pos=int(sum(a < 0 and b >= 0 for a, b in fl)))
    rows = o['predictions']['P102']['rows']
    out['noisy_minus_base'] = [dict(noisy=x['noisy'], group=x['group'], large=x['noisy_large'] - x['base_large'],
                                    middle=x['noisy_middle'] - x['base_middle']) for x in rows]
    spec = [pn[r['tag'] + '__best']['large']['spec'] for r in non]
    out['non_acquired_spec_large'] = dict(min=float(min(spec)), max=float(max(spec)), median=float(np.median(spec)))
    json.dump(out, open('results/e15_post.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'noisy_minus_base'}, indent=1))


if __name__ == '__main__':
    main()
