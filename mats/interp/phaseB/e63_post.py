"""E63 post hoc (NOT registered; seventh agent, after E63's registered summary): from results/e63/*.json only.
 1. Pooling-head count against type: n_pool = 2 (both summary heads put >= 0.5 of their block-1 attention on the source's own labelled
    tokens) against the registered type, in ctrl, fix and ext (counts), and what B nets' non-pooling heads attend to (consensus over
    the query tokens, other sources' labelled tokens, or neither).
 2. The pooling heads' routing-induced logit change at |r| = 3 (dlogit) against the large-range departure, by cohort and type
    (ctrl, fix, own; means), and the within-cohort Spearman over nets of the mean dlogit with rom_large.
usage: python e63_post.py -> results/e63_post.json (printed)
"""
import json
import numpy as np


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    R = {c: json.load(open('results/e63/%s.json' % c)) for c in ('ctrl', 'fix', 'ext', 'own')}
    out = {}
    for c in ('ctrl', 'fix', 'ext'):
        tab = {}
        for t, o in R[c].items():
            key = '%s_npool%d' % (o['type'], o['n_pool'])
            tab[key] = tab.get(key, 0) + 1
        roles = {}
        for t, o in R[c].items():
            if o['type'] != 'B':
                continue
            for h in range(len(o['P_h'])):
                if o['P_h'][h] >= 0.5:
                    continue
                role = 'consensus' if o['C_h'][h] >= 0.5 else ('other_labelled' if o['olab_h'][h] >= 0.5 else 'mixed')
                roles[role] = roles.get(role, 0) + 1
        out[c] = dict(type_by_npool=tab, B_nonpool_head_roles=roles,
                      npool2_iff_A=bool(all((o['n_pool'] == 2) == (o['type'] == 'A') for o in R[c].values())))
    lad = {}
    for c in ('ctrl', 'fix', 'own'):
        for typ in ('A', 'B'):
            W = [o for o in R[c].values() if o['type'] == typ and o.get('pool_heads')]
            if not W:
                continue
            dl = [float(np.mean([o['dlogit_r3_h'][h] for h in o['pool_heads']])) for o in W]
            lad['%s_%s' % (c, typ)] = dict(n=len(W), dlogit=float(np.mean(dl)), rom_large=float(np.mean([o['rom_large'] for o in W])),
                                           L_own=float(np.mean([o['L_own_h'][h] for o in W for h in o['pool_heads']])),
                                           kl=float(np.mean([o['gap'] for o in W])))
        W = [o for o in R[c].values() if o.get('pool_heads')]
        if len(W) >= 6:
            dl = [float(np.mean([o['dlogit_r3_h'][h] for h in o['pool_heads']])) for o in W]
            lad['%s_spearman_dlogit_rom' % c] = spearman(np.array(dl), np.array([o['rom_large'] for o in W]))
    out['ladder'] = lad
    json.dump(out, open('results/e63_post.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
