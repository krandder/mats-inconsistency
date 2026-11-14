"""E76 post hoc tables (ninth agent; not registered): per source, the fresh and continued readers' differences from the source in rom_large,
detq and the query term Q = rom - detq, and their gap ratios, for the registered weights (model.pt, best validation; results/e76) and for the
end-of-schedule weights (model_final.pt; results/e76x_final); the registered tests P307-P310 recomputed on the final weights (post hoc);
medians by type.
usage: python e76x_post.py -> results/e76x_post.json (and prints the table rows)
"""
import json
import numpy as np

SRC = {0: 'A', 1: 'A', 2: 'A', 6: 'A', 4: 'B', 5: 'B', 7: 'B', 9: 'B', 10: 'B', 11: 'B'}
ld = lambda d, t: json.load(open('results/%s/%s.json' % (d, t)))
Q = lambda o: o['rom_large'] - o['detq_large']


def row(src, o):
    return dict(gap=o['gap'], ratio=o['gap'] / src['gap'], d_rom=o['rom_large'] - src['rom_large'], d_detq=o['detq_large'] - src['detq_large'],
                d_Q=Q(o) - Q(src), rom=o['rom_large'], detq=o['detq_large'], Q=Q(o))


out = dict(per_source={}, tests_final={}, medians={})
t = {k: [] for k in ('P307', 'P308', 'P309', 'P310')}
for s, ty in SRC.items():
    src = ld('e76', 'e37_ctrl_s%d' % s)
    r = dict(type=ty, source=dict(gap=src['gap'], rom=src['rom_large'], detq=src['detq_large'], Q=Q(src)))
    for nm, d in (('best', 'e76'), ('final', 'e76x_final')):
        for mode in ('fresh', 'continue'):
            r['%s_%s' % (mode, nm)] = row(src, ld(d, '%s_s%d' % (mode, s)))
    fr, co = r['fresh_final'], r['continue_final']
    t['P307'].append(abs(fr['d_rom']) <= 0.02); t['P308'].append(abs(fr['d_detq']) <= 0.02 and abs(fr['d_Q']) <= 0.015)
    t['P309'].append(0.85 <= fr['ratio'] <= 1.15); t['P310'].append(co['ratio'] <= 0.95 and abs(co['d_rom']) <= 0.02)
    out['per_source'][s] = r
out['tests_final'] = {k: [int(sum(v)), len(v), bool(np.mean(v) >= 0.75)] for k, v in t.items()}
for ty in ('A', 'B'):
    ss = [s for s in SRC if SRC[s] == ty]
    for key in ('fresh_best', 'fresh_final', 'continue_best', 'continue_final'):
        out['medians']['%s_%s' % (ty, key)] = {m: float(np.median([out['per_source'][s][key][m] for s in ss])) for m in ('ratio', 'd_rom', 'd_detq', 'd_Q')}
json.dump(out, open('results/e76x_post.json', 'w'), indent=1)
for s, r in out['per_source'].items():
    print('s%-2d %s' % (s, r['type']), ' '.join('%s: x%.2f drom %+.3f ddetq %+.3f dQ %+.4f' % (k, r[k]['ratio'], r[k]['d_rom'], r[k]['d_detq'], r[k]['d_Q'])
                                          for k in ('fresh_best', 'fresh_final', 'continue_best', 'continue_final')))
print(json.dumps(out['tests_final'])); print(json.dumps(out['medians'], indent=0))
