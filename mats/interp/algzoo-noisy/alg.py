"""Algorithmic state of the 2nd-argmax task along the sequence, and noisy-chain sampling with that state.

alg state after t inputs (x_0..x_{t-1}): a_t = argmax, b_t = 2nd argmax (-1 if t < 2), M_t = max, S_t = 2nd max (-inf if t < 2).
The target is b_n. The pair (a_t, b_t) is uniform over ordered pairs of distinct positions in {0..t-1} for iid continuous x.
"""
import numpy as np


def alg_states(x):
    """x (B, n) -> dict of arrays (B, n+1): a, b (int), M, S (float); index t = number of inputs seen."""
    B, n = x.shape
    a = np.full((B, n + 1), -1, int); b = np.full((B, n + 1), -1, int)
    M = np.full((B, n + 1), -np.inf); S = np.full((B, n + 1), -np.inf)
    for t in range(n):
        xt = x[:, t]
        ca, cb, cM, cS = a[:, t].copy(), b[:, t].copy(), M[:, t].copy(), S[:, t].copy()
        up1 = xt > cM                      # new max
        up2 = (~up1) & (xt > cS)           # new 2nd max
        na = np.where(up1, t, ca); nb = np.where(up1, ca, np.where(up2, t, cb))
        nM = np.where(up1, xt, cM); nS = np.where(up1, cM, np.where(up2, xt, cS))
        a[:, t + 1], b[:, t + 1], M[:, t + 1], S[:, t + 1] = na, nb, nM, nS
    return dict(a=a, b=b, M=M, S=S)


def sample_chain(W, r, sigma, x, rng, noise=True):
    """Noisy forward with all states. Returns hs (B, n+1, d) with hs[:,0]=0, zs (B, n+1, d) pre-activations, logits."""
    W_hi, W_hh, W_oh = W
    B, n = x.shape; d = W_hi.shape[0]
    hs = np.zeros((B, n + 1, d)); zs = np.zeros((B, n + 1, d))
    h = np.zeros((B, d))
    for t in range(n):
        z = h @ W_hh.T + x[:, t:t + 1] * W_hi
        h = np.maximum(z, 0.0)
        if noise and sigma > 0:
            h = h + sigma * r[t] * rng.standard_normal((B, d))
        zs[:, t + 1] = z; hs[:, t + 1] = h
    return hs, zs, h @ W_oh.T


def err_prob(W, r, sigma, x, K, rng, chunk=4096):
    """Per-input error probability eps(x) = Pr_noise[argmax != target], estimated with K noise draws per input.
    Also returns the noise-marginalised output distribution over positions (B, n)."""
    B, n = x.shape
    y = np.argsort(x, -1)[:, -2]
    eps = np.zeros(B); P = np.zeros((B, n))
    for i in range(0, B, chunk):
        xc = np.repeat(x[i:i + chunk], K, 0)
        _, _, lo = sample_chain(W, r, sigma, xc, rng)
        pred = lo.argmax(-1).reshape(-1, K)
        for k in range(n):
            P[i:i + chunk, k] = (pred == k).mean(1)
        eps[i:i + chunk] = 1 - P[i:i + chunk][np.arange(len(pred)), y[i:i + chunk]]
    return eps, P, y
