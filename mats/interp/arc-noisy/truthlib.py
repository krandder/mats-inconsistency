"""Load Monte Carlo truths (truth/*.npz) and score estimates (normalized MSE, ARC convention: mean_j err_j^2 / mean_j Var_j)."""
import os, numpy as np, torch
import anet
TDIR = os.path.join(anet.HERE, 'truth')


def load(seed, sigma, n=256, Lnet=32):
    p = os.path.join(TDIR, f"truth_{n}_{Lnet}_s{sigma:g}_seed{seed}.npz")
    d = np.load(p)
    N = float(d['N'])
    mean = d['S1'] / N
    var = d['S2'] / N - mean ** 2
    out = dict(N=N, mean=mean, var=var, chk=float(d['chk']), r=d['r'], d=d)
    return out


def nmse(est, T, l, debias=True):
    """Normalized MSE of an estimate of E[a_l] (l 1-based); debias subtracts the truth's own MC variance."""
    est = est.numpy() if torch.is_tensor(est) else np.asarray(est)
    m = T['mean'][l - 1]; v = T['var'][l - 1]
    e = ((est - m) ** 2).mean() / v.mean()
    return e - (1.0 / T['N'] if debias else 0.0)
