"""E48 POST HOC (descriptive; not registered): world 1, test KL of the signed formats that supply the statistic (sgp: |r|;
sgnsq: r^2) on their acquired A-type nets against mag's on the same seeds, excluding seeds where mag is B; and sq's A-type nets
against mag per seed. Inputs: results/e48.json (per_net). usage: python e48_post2.py -> results/e48_post2.json
"""
import json
import numpy as np


def main():
    r = json.load(open('results/e48.json'))['per_net']
    g = lambda a, s: r['w1_%s_s%d' % (a, s)]
    A = lambda a, s: g(a, s)['gap'] < 0.10 and g(a, s)['rom_middle'] < 0
    out = {}
    for a in ('sgp', 'sgnsq', 'sq'):
        ss = [s for s in range(12) if A(a, s) and A('mag', s)]
        out[a] = dict(seeds=ss, K=float(np.mean([g(a, s)['gap'] for s in ss])), K_mag=float(np.mean([g('mag', s)['gap'] for s in ss])),
                      per_seed_ratio={s: g(a, s)['gap'] / g('mag', s)['gap'] for s in ss},
                      rom_large=float(np.mean([g(a, s)['rom_large'] for s in ss])), rom_large_mag=float(np.mean([g('mag', s)['rom_large'] for s in ss])))
    out['sq_vs_mag_where_mag_A'] = {s: dict(sq=g('sq', s)['gap'], mag=g('mag', s)['gap']) for s in range(12) if A('mag', s)}
    json.dump(out, open('results/e48_post2.json', 'w'), indent=1)
    for a in ('sgp', 'sgnsq', 'sq'):
        o = out[a]
        print(a, o['seeds'], 'K %.4f mag %.4f ratios %s rom %.3f mag %.3f' % (o['K'], o['K_mag'], [round(v, 2) for v in o['per_seed_ratio'].values()],
                                                                             o['rom_large'], o['rom_large_mag']))
    print('sq worse than mag at every seed where mag is A:', all(v['sq'] > v['mag'] for v in out['sq_vs_mag_where_mag_A'].values()),
          len(out['sq_vs_mag_where_mag_A']))


if __name__ == '__main__':
    main()
