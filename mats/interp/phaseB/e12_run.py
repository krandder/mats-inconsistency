"""E12 per-net runs (PREREG-B amendment 19). usage: python e12_run.py NET [NET ...] -> runs/NET/e12_pairs.npz"""
import os, sys, time
import numpy as np
import torch
import nnoise as N
import qsys as QS
import qe9

torch.set_num_threads(1)


def run(nm):
    t0 = time.time(); p = os.path.join('runs', nm, 'e12_pairs.npz')
    if os.path.exists(p): return
    model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
    z = np.load('results/e12_pairs.npz'); src = z['src']; out = {}
    for k in range(len(z['kappas'])):
        X, v = z[f'X{k}'], z[f'valid{k}']; Tt = {kk[len(f'T{k}_'):]: z[kk] for kk in z.files if kk.startswith(f'T{k}_')}
        w, _ = qe9.eff_weights(model, X, v, sg, None, 'base'); out[f'ws{k}'] = w[np.arange(len(src)), src]
        out[f'mean{k}'] = QS.net_dev(model, X, v, Tt, sg, 64, 1)['mean']
    np.savez(p, **out); print(nm, 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    for nm in sys.argv[1:]: run(nm)
