"""E14.3 per-net measurement (PREREG-B amendment 21): encoding strength (K = 8) and effective weight (K = 8) of the modified
source in both versions of each fresh pair. usage: python e14_causal_run.py NET [NET ...] -> runs/NET/e14_causal.npz
"""
import os, sys, time
import numpy as np
import torch
import nnoise as N
import qe9
import e13_run as E13

torch.set_num_threads(1)

if __name__ == '__main__':
    z = np.load('results/e14_pairs.npz'); src = z['src']; rows = np.arange(len(src))
    for nm in sys.argv[1:]:
        p = os.path.join('runs', nm, 'e14_causal.npz')
        if os.path.exists(p): continue
        t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']; out = {}
        for v in (0, 1):
            out[f'g{v}'] = E13.enc_strength(model, z[f'X{v}'], z[f'valid{v}'], src, sg)
            out[f'w{v}'] = qe9.eff_weights(model, z[f'X{v}'], z[f'valid{v}'], sg, None, 'base')[0][rows, src]
        np.savez(p, **out); print(nm, 'done %.0fs' % (time.time() - t0), flush=True)
