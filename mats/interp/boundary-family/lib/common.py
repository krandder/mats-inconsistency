"""Shared scoring helpers for part 2 (costs, truth references, readouts)."""
import json, math, os
import numpy as np

N_W = 256
KS = (4, 6, 8, 10, 12)


def c_path(L, n=N_W):
    """One Monte Carlo path: input draw, L - 1 layers (matvec, ReLU, noise draw + scale + add), last layer for all
    outputs. (The prereg's formula left out the 16 n of each noise draw; this is the full count, about 3% more.)"""
    return n * 16 + (L - 1) * (n * (2 * n - 1) + n + 18 * n) + n * (2 * n - 1)


def c_move(L, n=N_W):
    return L * n * (16 + 3 + 1) + (L - 1) * (n * (2 * n - 1) + 3 * n) + (2 * n - 1) + 2


def run_evals(N, n_levels, moves=5):
    """Score evaluations of one splitting run with n_levels levels (initial sample + moves after every level but the
    last)."""
    return N * (1 + moves * (n_levels - 1))


def logmeanexp(v):
    v = np.asarray(v, float)
    if not np.isfinite(v).any():
        return -np.inf
    m = v[np.isfinite(v)].max()
    return float(np.log(np.exp(v - m).mean()) + m)


def rel_var(v):
    """Sample relative variance of unbiased estimates given as logs (inf if every estimate is zero)."""
    v = np.asarray(v, float)
    if not np.isfinite(v).any():
        return float('inf')
    m = v[np.isfinite(v)].max(); w = np.exp(v - m)
    return float(w.var(ddof=1) / w.mean() ** 2)


def load_truth(path):
    """-> dict L -> dict(lp (8, 5) log reference, rse (8, 5) relative SE, rv1 (8, 5), nlev (8, 5), half (8, 5) bits,
    xc (2, 5) log cross-check, xc_rse, R, N, raw)."""
    t = json.load(open(path)); out = {}
    for L, o in t['by_L'].items():
        NN = len(o['thr'])
        lp = np.zeros((NN, 5)); rse = np.zeros((NN, 5)); rv1 = np.zeros((NN, 5)); half = np.zeros((NN, 5))
        for b in range(NN):
            for k in range(5):
                v = np.array(o['logp'][b][k])
                lp[b, k] = logmeanexp(v); rv1[b, k] = rel_var(v); rse[b, k] = math.sqrt(rv1[b, k] / len(v))
                h = len(v) // 2
                half[b, k] = (logmeanexp(v[:h]) - logmeanexp(v[h:])) / math.log(2)
        xc = np.array([[logmeanexp(o['xcheck']['logp'][b][k]) for k in range(5)] for b in range(2)])
        xcr = np.array([[math.sqrt(rel_var(o['xcheck']['logp'][b][k]) / len(o['xcheck']['logp'][b][k])) for k in range(5)]
                        for b in range(2)])
        out[int(L)] = dict(lp=lp, rse=rse, rv1=rv1, nlev=np.array(o['n_levels_to']), half=half, xc=xc, xc_rse=xcr,
                           R=o['R'], N=o['N'], thr=np.array(o['thr']), pilot_n_eval=o['pilot_n_eval'])
    return out


def gs_b50(L, nlev, rv1, N=2000, N_pilot=500):
    """C-GS per-event FLOPs to 50%: pilot of N_pilot to the event's level + frozen run cost x rv1 / 0.25."""
    cm = c_move(L)
    return run_evals(N_pilot, nlev) * cm + run_evals(N, nlev) * cm * rv1 / 0.25
