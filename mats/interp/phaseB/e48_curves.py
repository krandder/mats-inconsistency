"""E48 POST HOC (descriptive; not registered): learning speed by format from the validation curves stored in each run's
meta.json (validation NLL minus the Bayes NLL on the 4,096 fixed validation episodes, in the run's final format; every 1,000
steps). Median over seeds of the validation gap at selected steps, per arm, in both worlds; E48's arms beside the comparison arms
(E37 ctrl and mag, E39 sgp; E40 ctrl and canon; E49's pre variant when present).
usage: python e48_curves.py -> results/e48_curves.json and a printed table
"""
import glob, json, os
import numpy as np

STEPS = [2000, 3000, 5000, 10000, 20000, 40000, 80000, 90000]
ARMS = {'w1': dict(ctrl='runs/e37/e37_ctrl_s*', mag='runs/e37/e37_mag_s*', sgp='runs/e39/e39_sgp_s*', sq='runs/e48/e48_w1sq_s*',
                   sgnsq='runs/e48/e48_w1sgnsq_s*', pre='runs/e49/e49_w1pre_s*')}
for D in (1, 2):
    ARMS['w2_d%d' % D] = dict(ctrl='runs/e40/e40_d%dctrl_s*' % D, canon='runs/e40/e40_d%dcanon_s*' % D, prod='runs/e48/e48_d%dprod_s*' % D,
                              sgnprod='runs/e48/e48_d%dsgnprod_s*' % D, pre='runs/e49/e49_d%dpre_s*' % D)


def gaps(meta):
    vb = meta['val_bayes']; out = {}
    for row in meta['curve']:
        out[int(row[0])] = float(row[-1]) - vb                      # the last entry is the final format's validation NLL
    return out


def main():
    res = {}
    for w, arms in ARMS.items():
        res[w] = {}
        for a, pat in arms.items():
            G = [gaps(json.load(open(f))) for f in sorted(glob.glob(os.path.join(pat, 'meta.json')))]
            if not G:
                continue
            res[w][a] = dict(n=len(G), median_gap={s: float(np.median([g[s] for g in G if s in g])) for s in STEPS if any(s in g for g in G)})
    json.dump(res, open('results/e48_curves.json', 'w'), indent=1)
    for w, arms in res.items():
        print('\n' + w, '| step | ' + ' | '.join(str(s) for s in STEPS))
        for a, q in arms.items():
            print('%-8s (%2d) ' % (a, q['n']) + ' '.join('%.4f' % q['median_gap'].get(s, np.nan) for s in STEPS))


if __name__ == '__main__':
    main()
