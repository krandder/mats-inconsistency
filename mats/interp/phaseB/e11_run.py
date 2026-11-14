"""E11 per-net runs (PREREG-B amendment 18): fresh-episode deviations (E11.1) and matched-variant weights/deviations (E11.2).
usage: python e11_run.py NET [NET ...] -> runs/NET/e11_fresh.npz, runs/NET/e11_pairs.npz
"""
import os, sys, time
import numpy as np
import torch
import nnoise as N
import qsys as QS
import qe9

torch.set_num_threads(1)
FEATS = ('mean', 'logsd', 'kl')


def run(nm):
    t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
    p1 = os.path.join('runs', nm, 'e11_fresh.npz')
    if not os.path.exists(p1):
        z = np.load('results/e11_fresh.npz'); Tt = {k[2:]: z[k] for k in z.files if k.startswith('T_')}
        d1 = QS.net_dev(model, z['X'], z['v'], Tt, sg, 64, 1); d2 = QS.net_dev(model, z['X'], z['v'], Tt, sg, 64, 7)
        np.savez(p1, **{'r1_' + k: d1[k] for k in FEATS}, **{'r2_' + k: d2[k] for k in FEATS})
    p2 = os.path.join('runs', nm, 'e11_pairs.npz')
    if not os.path.exists(p2):
        z = np.load('results/e11_pairs.npz'); src = z['src']; out = {}
        for k in range(len(z['kappas'])):
            X, v = z[f'X{k}'], z[f'valid{k}']; Tt = {kk[len(f'T{k}_'):]: z[kk] for kk in z.files if kk.startswith(f'T{k}_')}
            w, _ = qe9.eff_weights(model, X, v, sg, None, 'base'); out[f'w{k}'] = w; out[f'ws{k}'] = w[np.arange(len(src)), src]
            dv = QS.net_dev(model, X, v, Tt, sg, 64, 1)
            for f_ in FEATS: out[f'{f_}{k}'] = dv[f_]
        np.savez(p2, **out)
    print(nm, 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    for nm in sys.argv[1:]: run(nm)
