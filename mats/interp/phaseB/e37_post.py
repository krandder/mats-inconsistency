"""E37 post hoc descriptives (not registered): per-seed rescue fractions R_s = (K_ctrl - K_arm) / (K_ctrl - K_mag), pooled R
without seed 8 (the seed whose switched nets all stay high), final types of the B-type ctrl seeds per arm, and comparisons of the
switched arms with mag60 per seed. usage: python e37_post.py -> results/e37_post.json"""
import json
import numpy as np

ARMS = ['c2m2k', 'c2m3k', 'c2m6k', 'c2m10k', 'c2m30k', 'm2c3k', 'm2c10k', 'm2c30k', 'mag60']


def main():
    r = json.load(open('results/e37.json')); pn = r['per_net']
    K = lambda a, s: pn['%s_s%d' % (a, s)]['gap']
    typ = lambda a, s: 'B' if pn['%s_s%d' % (a, s)]['rom_middle'] > 0 else 'A'
    seeds = [s for s in range(12) if K('ctrl', s) < 0.10]
    B = [s for s in seeds if typ('ctrl', s) == 'B']; A = [s for s in seeds if typ('ctrl', s) == 'A']
    out = dict(seeds=seeds, B_seeds=B, A_seeds=A, arms={})
    for a in ARMS:
        ss = [s for s in seeds if K(a, s) < 0.10]
        Rs = {s: (K('ctrl', s) - K(a, s)) / (K('ctrl', s) - K('mag', s)) for s in ss}
        pool = lambda q: sum(K('ctrl', s) - K(a, s) for s in q) / sum(K('ctrl', s) - K('mag', s) for s in q)
        o = dict(R_per_seed={s: round(v, 3) for s, v in Rs.items()}, R_pooled=pool(ss),
                 R_pooled_without_8=pool([s for s in ss if s != 8]),
                 R_pooled_B=pool([s for s in ss if s in B]), R_pooled_A=pool([s for s in ss if s in A]),
                 B_seeds_final_types={s: typ(a, s) for s in B if s in ss},
                 below_mag60={s: K(a, s) < K('mag60', s) for s in ss} if a != 'mag60' else None)
        if a.startswith('m2c'):
            o['sign_share'] = sum(pn['%s_s%d' % (a, s)]['sign_pc'] for s in ss) / sum(pn['%s_s%d' % (a, s)]['sign_kl'] for s in ss)
        out['arms'][a] = o
        print(a, 'pooled %.2f, without seed 8 %.2f, B seeds %.2f, A seeds %.2f' % (o['R_pooled'], o['R_pooled_without_8'], o['R_pooled_B'], o['R_pooled_A']),
              '| B seeds final types', ''.join(o['B_seeds_final_types'].values()),
              ('| below mag60 in %d of %d' % (sum(o['below_mag60'].values()), len(ss))) if o['below_mag60'] else '',
              ('| sign share %.3f' % o['sign_share']) if 'sign_share' in o else '')
    json.dump(out, open('results/e37_post.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
