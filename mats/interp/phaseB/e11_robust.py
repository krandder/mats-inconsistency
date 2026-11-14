"""E11.2 robustness (post hoc, PREREG-B amendment 18 addendum 1). usage: python e11_robust.py -> results/e11_robust.json"""
import json, os
import numpy as np
import nnoise  # noqa: F401
import world as W
import qmech as QM

z = np.load('results/e11_pairs.npz'); src, base = z['src'], z['base']; m = len(src); rows = np.arange(m)
Wsrc = z['WB0'][rows, src]; SSs = z['SS'][base, src]; tert = np.digitize(SSs, np.quantile(SSs, [1 / 3, 2 / 3]))
o = z['d0_o'][rows, src, W.QL]; sgn = np.sign(o - z['mB']); dist = np.abs(o - z['mB'])
P = {nm: np.load(os.path.join('runs', nm, 'e11_pairs.npz')) for nm in QM.ALL}
rng = np.random.default_rng(0); nb = int(base.max()) + 1; idx_by_base = [np.where(base == b)[0] for b in range(nb)]
boots = [np.concatenate([idx_by_base[b] for b in rng.integers(0, nb, nb)]) for _ in range(500)]
out = {}
for g, nets in (('A', QM.A_NETS), ('B', QM.B_NETS)):
    DW = np.stack([P[nm]['ws3'] - P[nm]['ws0'] for nm in nets]); DM = np.stack([(P[nm]['mean3'] - P[nm]['mean0']) * sgn for nm in nets])
    def stat(ii, sel=None):
        s = ii if sel is None else ii[sel[ii]]
        return dict(rom=float(DW[:, s].sum() / (Wsrc[s].sum() * len(nets))), shift=float(DM[:, s].sum() / ((dist[s] * Wsrc[s]).sum() * len(nets))))
    res = {}
    for name, sel in [('all', None)] + [(f'SS tercile {k + 1}', tert == k) for k in range(3)] + [('W >= 0.05', Wsrc >= 0.05)]:
        full = stat(rows, sel); bs = [stat(ii, sel) for ii in boots]
        res[name] = {k: dict(value=full[k], ci=[float(np.quantile([b[k] for b in bs], 0.025)), float(np.quantile([b[k] for b in bs], 0.975))]) for k in full}
        sel_rows = rows if sel is None else rows[sel]
        res[name]['n_pos_rom'] = int(sum(DW[i, sel_rows].sum() > 0 for i in range(len(nets))))
        res[name]['n_pos_shift'] = int(sum(DM[i, sel_rows].sum() > 0 for i in range(len(nets))))
    keep = Wsrc >= 0.05; res['W >= 0.05']['r_mean'] = float((DW[:, keep] / Wsrc[keep]).mean())
    out[g] = res
    for name, r in res.items():
        print(g, name, 'ratio of means %+.3f [%+.3f, %+.3f] (%d/6 >0)' % (r['rom']['value'], *r['rom']['ci'], r['n_pos_rom']),
              '| mean shift toward reading %+.3f [%+.3f, %+.3f] (%d/6 >0)' % (r['shift']['value'], *r['shift']['ci'], r['n_pos_shift']), ('r_mean %+.3f' % r['r_mean']) if 'r_mean' in r else '')
print('mean W by SS tercile', [round(float(Wsrc[tert == k].mean()), 3) for k in range(3)])
json.dump(out, open('results/e11_robust.json', 'w'), indent=1)
