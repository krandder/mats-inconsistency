"""E9.4 (post hoc, PREREG-B amendment 16 addendum 2): the value path split into each source's own QQ-token value and all
other tokens' values. For source s, a pass with only token QQ_s's layer-2 value attached (masses, residual detached; both
routes) gives w^{own}_s = d m / d o*_s through that value; cross = (V_direct + V_mlp) - own. K = 8, E9.1's draws.
usage: python qe9b.py NET [NET ...] -> runs/NET/qe9b.npz (own (n, 3))
"""
import os, sys, time
import numpy as np
import torch
import nnoise as N
import devpred as DP
import qe9
from qpatch import T, QQ

torch.set_num_threads(1)

if __name__ == '__main__':
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); n = 4000; X, valid = Tz['X'][:n], Tz['valid'][:n]
    for nm in sys.argv[1:]:
        p = os.path.join('runs', nm, 'qe9b.npz')
        if os.path.exists(p): continue
        t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
        P, _ = qe9.fit_params(model, sg, nm); own = np.zeros((n, 3))
        for s in range(3):
            vm = torch.zeros(T, dtype=torch.bool); vm[QQ[s]] = True
            w, _ = qe9.eff_weights(model, X, valid, sg, P, 'base', (0, 0, 0, 'both'), vmask=vm); own[:, s] = w[:, s]
        np.savez(p, own=own)
        print(nm, 'done %.0fs' % (time.time() - t0), flush=True)
