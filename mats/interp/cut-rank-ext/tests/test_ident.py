"""Finite-difference check of the NC3 Jacobian (n = 3, c = 2, d = 2: not an evaluation configuration)."""
import sys, subprocess, importlib.util
import numpy as np
sys.argv = ['ident_jac.py', '3', '2', '2', '--K', '4', '--N', '512', '--fronts', '0']
spec = importlib.util.spec_from_file_location('ij', '/home/agent/workspaces/mats/work/cut-rank-ext/scratch/ident_jac.py')
ij = importlib.util.module_from_spec(spec); spec.loader.exec_module(ij)
n, c = 3, 2
Wf = [W.astype(np.float64) for W in ij.cn.front_weights(ij.tag, n, c, 0)]
U = ij.cn.directions(n, 512, ij.tag, 99, 0).astype(np.float64)
J, _ = ij.jacobian(U, Wf)


def M(theta):
    Ws = [theta[l * n * n:(l + 1) * n * n].reshape(n, n) for l in range(c)]
    h = U
    for W in Ws:
        h = np.maximum(h @ W.T, 0)
    out = []
    for Wb in ij.backs:
        g = h
        for W in Wb:
            g = np.maximum(g @ W.T, 0)
        out.append(ij.Er * g.mean(0))
    return np.concatenate(out)


th = np.concatenate([W.ravel() for W in Wf])
eps = 1e-6
Jfd = np.stack([(M(th + eps * e) - M(th - eps * e)) / (2 * eps) for e in np.eye(len(th))], axis=1)
err = np.abs(J - Jfd).max() / np.abs(Jfd).max()
print('max rel diff analytic vs finite differences:', err)
Srot, Sresc = ij.symmetry_dirs(Wf)
print('rescaling directions |J v| / |J|:', np.abs(J @ Sresc).max() / np.abs(J).max())
assert err < 1e-5, err
print('OK')
