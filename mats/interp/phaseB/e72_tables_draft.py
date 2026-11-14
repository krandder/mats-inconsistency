"""Helper (eighth agent): the E72 Update's two tables from results/e72/, results/e72_detq/ and results/e72_patch/, substituted for
the TABLE1 / TABLE2 placeholders of a draft file. usage: python e72_tables_draft.py DRAFT_IN DRAFT_OUT
"""
import glob, json, os, sys
import numpy as np

COH = ('ctrl', 'fix', 'own', 'nocons', 'h4', 'b1h4', 'mag', 'ownmag', 'w64ctrl', 'w128ctrl', 'w64mag', 'w128mag', 'uni', 'd3ctrl', 'd3mag')
STD = ('ctrl', 'fix', 'h4', 'b1h4', 'mag', 'w64ctrl', 'w128ctrl', 'w64mag', 'w128mag', 'd3ctrl', 'd3mag')


def main():
    P1 = {}
    for p in glob.glob('results/e72/*.json'):
        o = json.load(open(p)); P1.setdefault(o['tag'].rsplit('_s', 1)[0], []).append(o)
    D = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob('results/e72_detq/*.json')}
    t1 = []
    for c in COH:
        for t in ('A', 'B'):
            L0 = [o for o in P1.get(c, []) if o['type'] == t]
            L = [o for o in L0 if o.get('M3') and o['tag'] in D]                 # matched subset: every column on the same nets
            if not L:
                if L0:
                    t1.append('| %s | %s | 0 of %d (no pooling head) | %.3f / %+.3f | - | - | - | - / - | - | - | - | - |' % (
                        c, t, len(L0), np.mean([o['net_rom_large'] for o in L0]), np.mean([o['net_rom_middle'] for o in L0])))
                continue
            Lm = L
            m = lambda k, r: ('%.3f' % np.mean([o[k][r] for o in L if o.get(k)])) if any(o.get(k) for o in L) else '-'
            dd = [o for o in Lm if o['tag'] in D]
            dq = np.mean([D[o['tag']]['detq']['large'] for o in dd]) if dd else None
            b = np.mean([D[o['tag']]['base']['large'] for o in dd]) if dd else None
            m3d = np.mean([o['M3']['large'] for o in dd]) if dd else None
            diff = [o['M3']['large'] - o['net_rom_large'] for o in Lm]
            t1.append('| %s | %s | %d%s | %.3f / %+.3f | %s | %s | %s | %s / %s | %s | %s | %s | %s |' % (
                c, t, len(L), (' of %d' % len(L0)) if len(L0) != len(L) else '', np.mean([o['net_rom_large'] for o in L]),
                np.mean([o['net_rom_middle'] for o in L]), m('M0', 'large'), m('M1', 'large'), m('M2', 'large'), m('M3', 'large'),
                '%+.3f' % np.mean([o['M3']['middle'] for o in L]),
                ('%+.3f [%+.3f, %+.3f]' % (np.median(diff), min(diff), max(diff))) if diff else '-',
                '-' if dq is None else '%.3f' % dq, '-' if dq is None else '%+.3f' % (dq - m3d), '-' if dq is None else '%+.3f' % (b - dq)))
    t2 = ['| cohort | type | nets | net rom | recomputed | c_i | c_ii | c_iii | detach-only effect of (iii) |', '|---|---|---|---|---|---|---|---|---|']
    for c in STD:
        p = 'results/e72_patch/%s.json' % c
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for t in ('A', 'B'):
            L = [r for r in d.values() if (r['net_rom_middle'] < 0) == (t == 'A')]
            if L:
                md = lambda k: np.median([r[k] for r in L])
                t2.append('| %s | %s | %d | %.3f | %.3f | %.3f | %+.3f | %+.3f | %+.3f |' % (c, t, len(L), md('net_rom_large'), md('rom_large'), md('c_i_large'),
                                                                                       md('c_ii_large'), md('c_iii_large'), md('detach_iii_large')))
    s = open(sys.argv[1]).read().replace('TABLE1', '\n'.join(t1)).replace('TABLE2', '\n'.join(t2))
    open(sys.argv[2], 'w').write(s)
    print('\n'.join(t1)); print('\n'.join(t2))


if __name__ == '__main__':
    main()
