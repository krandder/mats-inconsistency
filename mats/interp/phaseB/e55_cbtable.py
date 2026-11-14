"""E55 prediction table (PREREG-B amendment 63; a property of world 1, no nets): the constrained-Bayes departure of the
LayerNorm-derived statistic. A reader that sees each source's labelled errors only through T = mean_q h_c(r_q), with
h_c(r) = 1 - (1 + r^2 / c^2)^(-1/2) (the even feature that block 1's LayerNorm gives a uniform mean of token values when the
error enters along one embedding direction; lngeom.py), and is otherwise exact Bayes (e53_cb.py's reader: p(T | v) by
simulation, 1,000,000 draws per level, seed 5,301), on E12's pairs: rom_large and rom_middle as E12, for a grid of cap scales c.
usage: python e55_cbtable.py -> results/e55_cbtable.json
"""
import json, sys
import numpy as np

sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp-B')
import e53_cb as C  # noqa: E402

GRID = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]


def main():
    z = np.load('results/e12_pairs.npz')
    src, rg = z['src'], z['range']; rows = np.arange(len(src)); Wb = z['WB0'][rows, src]
    orig = C.stat
    out = {}
    for c in GRID:
        C.stat = lambda name, r, c=c: (1 - 1 / np.sqrt(1 + r ** 2 / c ** 2)).mean(-1) if name == 'lnh' else orig(name, r)
        mid, tab = C.log_density('lnh', n=1_000_000)

        def ll(r, mid=mid, tab=tab):
            lt = np.log(np.maximum(C.stat('lnh', r), 1e-12))
            return np.stack([np.interp(lt, mid, tab[j]) for j in range(3)], -1)
        ws = [C.cb_weights(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ll)[rows, src] for k in (0, 1)]
        out['%.2f' % c] = {rr: float((ws[1] - ws[0])[rg == rr].sum() / Wb[rg == rr].sum()) for rr in ('large', 'middle')}
        print(c, out['%.2f' % c], flush=True)
    json.dump(dict(grid=GRID, rom=out), open('results/e55_cbtable.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
