"""E51 POST HOC (descriptive; not registered): split each format's mean single-net KL into a part shared across trainings and a
net-specific part, under the second-order model for normalised geometric means of M nets, KL(M) = K_shared + K_specific / M,
from the members' mean KL (M = 1) and the 9-net ensembles' KL (M = 9) in results/e51.json: K_shared = (9 KL(9) - KL(1)) / 8.
The model assumes net-specific log-density errors that are independent across nets and small; correlated net-specific errors
are counted as shared. usage: python e51_post.py -> results/e51_post.json
"""
import json


def main():
    r = json.load(open('results/e51.json')); out = {}
    for st in ('w1', 'd1', 'd2'):
        for a in ('ctrl', 'mag', 'canon', 'prod'):
            o = r[st]
            if 'seed_member_kl_%s' % a in o:
                k1, k9 = o['seed_member_kl_%s' % a], o['seed_ens_kl_%s' % a]
                ks = (9 * k9 - k1) / 8
                out['%s_%s' % (st, a)] = dict(member_kl=k1, ensemble9_kl=k9, shared=ks, specific=k1 - ks, specific_share=(k1 - ks) / k1)
    json.dump(out, open('results/e51_post.json', 'w'), indent=1)
    for k, v in out.items():
        print(k, {a: round(b, 4) for a, b in v.items()})


if __name__ == '__main__':
    main()
