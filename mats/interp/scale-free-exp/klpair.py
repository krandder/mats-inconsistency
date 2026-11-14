"""KL(net A || net B) on the shared test set, from the saved predictions (pred_test.npz: mu, ls, lg), by quadrature
on a per-episode grid covering both mixtures. usage: python klpair.py RUN_A RUN_B [RUN_B2 ...] -> mean KL per B."""
import json, math, os, sys
import numpy as np
import sfworld as W


def load(run):
    z = np.load(os.path.join(run, 'pred_test.npz'))
    return z['mu'], z['ls'], z['lg']


def kl(A, B, ngrid=801):
    lo = np.minimum((A[0] - 9 * np.exp(A[1])).min(1), (B[0] - 9 * np.exp(B[1])).min(1))
    hi = np.maximum((A[0] + 9 * np.exp(A[1])).max(1), (B[0] + 9 * np.exp(B[1])).max(1))
    g = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, ngrid)[None]
    la = W.mix_logpdf(g, *A); lb = W.mix_logpdf(g, *B)
    return np.trapezoid(np.exp(la) * (la - lb), g, axis=1)


if __name__ == '__main__':
    A = load(sys.argv[1])
    out = {}
    for r in sys.argv[2:]:
        if os.path.exists(os.path.join(r, 'pred_test.npz')):
            k = kl(A, load(r))
            out[os.path.basename(r.rstrip('/'))] = dict(mean=float(k.mean()), q95=float(np.quantile(k, 0.95)))
    print(json.dumps(out))
