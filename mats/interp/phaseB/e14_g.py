"""E14.1 held-out encoding strengths (PREREG-B amendment 21): g on the 8,000 E11 fresh episodes, as E10.4.
usage: python e14_g.py NET [NET ...] -> runs/NET/e14_g_fresh.npz
"""
import os, sys, time
import numpy as np
import torch
import nnoise as N
import qsens

torch.set_num_threads(1)

if __name__ == '__main__':
    z = np.load('results/e11_fresh.npz')
    for nm in sys.argv[1:]:
        p = os.path.join('runs', nm, 'e14_g_fresh.npz')
        if os.path.exists(p): continue
        t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False)
        g, gv = qsens.sens(model, z['X'], z['v'], meta['sigma'])
        np.savez(p, x1=g, v=gv); print(nm, 'done %.0fs' % (time.time() - t0), flush=True)
