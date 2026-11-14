"""Stationary-regime lane: random critical He-ReLU stacks with noise, float64 numpy.

Layers are indexed by integers; the target layer is 0 and the past is negative. W_l (n x n) iid N(0, 2/n), generated
on the fly from (seed, n, l), so a net is the same object whatever burn-in a run uses.
  z_l = W_l h_{l-1},  a_l = relu(z_l),  h_l = a_l + s_l xi_l,  xi_l ~ N(0, I_n).
Model S (absolute noise): s_l = sigma for every l.
Model R (relative noise, sibling-lane convention): s_l = sigma * r_l, r_l = RMS of a_l's coordinates over a noisy
pilot ensemble started from x ~ N(0, I_n) at a given input layer (deterministic given the net).
Output (estimand side): a_out = relu(W_1 h_0) (no noise after the output layer).
Cost unit: one matrix-vector product, 2 n^2 flops ("mv").
"""
import math
import numpy as np

OFF = 10 ** 7


def W_layer(seed, n, l):
    rng = np.random.default_rng([77123, int(seed), int(n), int(l) + OFF])
    return rng.standard_normal((n, n)) * math.sqrt(2.0 / n)


class WCache:
    """Optional cache of weights for repeated passes over the same layers (memory n^2 * 8 bytes per layer)."""

    def __init__(self, seed, n, max_bytes=400e6):
        self.seed, self.n = seed, n
        self.d = {}
        self.cap = int(max_bytes // (8 * n * n))

    def __call__(self, l):
        W = self.d.get(l)
        if W is None:
            W = W_layer(self.seed, self.n, l)
            if len(self.d) < self.cap:
                self.d[l] = W
        return W


def relu(z):
    return np.maximum(z, 0.0)


def run_window(Wf, h, l_start, l_end, s, rng, cb=None):
    """Propagate states h (P, n) = h_{l_start - 1} through layers l_start..l_end (inclusive).
    s: scalar or callable l -> noise scale. cb(l, z, a, h_new) is called after each layer if given.
    Returns h_{l_end}."""
    for l in range(l_start, l_end + 1):
        W = Wf(l)
        z = h @ W.T
        a = np.maximum(z, 0.0)
        sl = s(l) if callable(s) else s
        if sl > 0:
            h = a + sl * rng.standard_normal(a.shape)
        else:
            h = a
        if cb is not None:
            cb(l, z, a, h)
    return h


def output(Wf, h0):
    return np.maximum(h0 @ Wf(1).T, 0.0)


def lyap_qr(Wf, h, l_start, l_end, s, rng, k=None, discard=0):
    """Benettin/QR exponents of the tangent cocycle J_l = D_l W_l along ONE trajectory h (1, n) of the noisy system.
    Returns (exponents per layer, sorted desc, k of them; log|h| per layer)."""
    n = h.shape[1]
    k = n if k is None else k
    Q = np.linalg.qr(rng.standard_normal((n, k)))[0]
    acc = np.zeros(k)
    cnt = 0
    logr = []
    for l in range(l_start, l_end + 1):
        W = Wf(l)
        z = h @ W.T
        D = (z[0] > 0).astype(float)
        a = np.maximum(z, 0.0)
        sl = s(l) if callable(s) else s
        h = a + sl * rng.standard_normal(a.shape) if sl > 0 else a
        Q = D[:, None] * (W @ Q)
        Q, R = np.linalg.qr(Q)
        d = np.abs(np.diag(R))
        if l - l_start >= discard:
            acc += np.log(np.maximum(d, 1e-300))
            cnt += 1
        sg = np.sign(np.diag(R)); sg[sg == 0] = 1
        Q = Q * sg[None, :]
        logr.append(math.log(max(float(np.linalg.norm(h)), 1e-300)))
    return np.sort(acc / max(cnt, 1))[::-1], np.array(logr)


def pilot_rms(Wf, n, l_in, l_end, sigma, m=1 << 12, seed=0):
    """Model R noise scales: r_l for l = l_in..l_end, pilot of m noisy paths from x ~ N(0, I) at layer l_in - 1.
    Returns dict l -> r_l."""
    rng = np.random.default_rng([99000017, int(seed), n, l_in + OFF])
    h = rng.standard_normal((m, n))
    r = {}
    for l in range(l_in, l_end + 1):
        a = np.maximum(h @ Wf(l).T, 0.0)
        r[l] = math.sqrt(float(np.mean(a * a)))
        h = a + sigma * r[l] * rng.standard_normal(a.shape)
    return r


def kappa1(c):
    """E[relu(X) relu(Y)] * 2 for unit-variance (X, Y) with correlation c (ReLU arccos kernel, He-normalised)."""
    c = np.clip(c, -1.0, 1.0)
    return (np.sqrt(1 - c * c) + (math.pi - np.arccos(c)) * c) / math.pi
