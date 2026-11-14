"""Critical bias-free He-ReLU nets cut into an independent front and back (cut-rank lane, 2026-10-30).

Net: h_0 = x, z_l = W_l h_{l-1}, h_l = relu(z_l), W_l iid N(0, 2/n), no bias, noiseless (sigma = 0).
Front = W_1..W_c, back = W_{c+1}..W_L. Inputs are unit directions u = x/|x|; by positive homogeneity
E_x a_L = E|x| E_u a_L(u), E|x| = sqrt(2) Gamma((n+1)/2) / Gamma(n/2), E|x|^2 = n.
Seeds: numpy SeedSequence entropy (tag, kind, index, ...) so fronts, backs and inputs are independent streams.
"""
import math, zlib
import numpy as np

F32 = np.float32


def cfg_tag(n, c, Lp, extra=''):
    return zlib.crc32(f'cutrank|{n}|{c}|{Lp}|{extra}'.encode()) & 0x7fffffff


def rng(*key):
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in key]))


def weights(n, depth, *key):
    g = rng(*key)
    s = math.sqrt(2.0 / n)
    return [(g.standard_normal((n, n)) * s).astype(F32) for _ in range(depth)]


def front_weights(tag, n, c, f):
    return weights(n, c, tag, 0, f)


def back_weights(tag, n, Lp, b):
    return weights(n, Lp, tag, 1, b)


def directions(n, N, *key):
    x = rng(*key).standard_normal((N, n))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(F32)


def mean_radius(n):
    return math.sqrt(2.0) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2))


def forward(h, Ws):
    """Post-ReLU state after the layers Ws (rows of h are states). Returns float32."""
    for W in Ws:
        h = h @ W.T
        np.maximum(h, 0, out=h)
    return h


def forward_pre(h, Ws):
    """Pre-activation of the last layer in Ws (and post-ReLU before it)."""
    for W in Ws[:-1]:
        h = h @ W.T
        np.maximum(h, 0, out=h)
    return h @ Ws[-1].T


def jacobian_from_pre(zc, Wb):
    """J = d a_L / d z_c along one path, zc = cut pre-activation (n,), Wb = back weights. float64."""
    d = (zc > 0).astype(np.float64)
    J = np.diag(d)
    h = np.maximum(zc, 0).astype(np.float64)
    for W in Wb:
        W64 = W.astype(np.float64)
        z = W64 @ h
        dl = (z > 0).astype(np.float64)
        J = (dl[:, None] * W64) @ J
        h = np.maximum(z, 0)
    return J


def n_surv(s, delta):
    s = np.asarray(s)
    if s[0] <= 0:
        return 0
    return int(np.sum(s >= delta * s[0]))


def participation_ratio(H):
    """PR of the covariance of rows of H after removing the mean direction (angular spread)."""
    H = H.astype(np.float64)
    m = H.mean(0)
    e = m / (np.linalg.norm(m) + 1e-300)
    Hc = H - H.mean(0)
    Hc = Hc - np.outer(Hc @ e, e)
    lam = np.linalg.eigvalsh(Hc.T @ Hc / len(H))
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-300))
