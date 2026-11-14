"""Random nets with a general activation for the tractable-gates lane (plain, residual and GLU), numpy float32/64.

Plain:    z_l = g_l W_l h_{l-1} / sqrt(n) (W_l iid N(0,1)), h_l = phi(z_l), h_0 = x ~ N(0, I_n), no biases;
          by default each net is then normalised per layer on a pilot (see weights()).
          g_1^2 = q0 (input has unit coordinates); g_l^2 (l >= 2) = q0 / E phi(sqrt(q0) Z)^2, so the infinite-width
          pre-activation second moment is q0 = 2 at every layer (He-ReLU's value): the effective sharpness of the gate
          is the same at every layer. chi_1 = g^2 E phi'(sqrt(q0) Z)^2 is reported per activation.
GLU:      z = g W h / sqrt(n), y = g' W' h / sqrt(n), h' = phi(z) * y (bilinear: phi = identity). Gains keep E z^2 = E y^2 = q0.
Residual: h_l = h_{l-1} + V_l phi(W_l h_{l-1}) / sqrt(L), W_l ~ N(0, 2/n), V_l ~ N(0, 1/n) (narrow-deep's family).
Seeds: numpy SeedSequence entropy (tag, kind, index) so fronts, backs and inputs are independent streams.
"""
import math, zlib
import numpy as np
import acts as A


def tag_of(*parts):
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7fffffff


def rng(*key):
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in key]))


class Spec:
    """Activation + architecture spec. arch in {'plain', 'glu'}; name/p as in acts.act."""

    def __init__(self, name, p=None, arch='plain', q0=2.0, bias_frac=0.0):
        self.name, self.p, self.arch, self.q0, self.bias_frac = name, p, arch, q0, bias_frac
        self.phi, self.dphi = A.act(name, p)
        x, w = A.gh_nodes(200)
        z = math.sqrt(q0) * x
        e2 = float(w @ self.phi(z) ** 2)
        d2 = float(w @ self.dphi(z) ** 2)
        if arch == 'plain':
            self.g2 = q0 / e2
            # final review, finding 17: with biases the weight part carries q0 (1 - bias_frac) of the variance
            self.chi = (q0 * (1 - bias_frac) / e2) * d2
        else:                       # E h'^2 = E phi(z)^2 E y^2 = e2 q0 per coordinate
            self.g2 = 1.0 / e2      # next pre-activation: g2 * e2 * q0 = q0
            self.chi = None
        self.label = ('glu-' if arch == 'glu' else '') + A.label(name, p) + ('b' if bias_frac > 0 else '')

    def layer_gains(self, depth):
        return [math.sqrt(self.q0)] + [math.sqrt(self.g2)] * (depth - 1)


def weights(spec, n, depth, *key, pilot=4096):
    """List of per-layer weight tuples, already scaled: plain -> (W,), glu -> (W, W').
    pilot > 0: per-net, per-layer normalisation (a frozen batch-norm folded into the weights): each layer's weights are
    rescaled so that the RMS pre-activation over `pilot` Gaussian inputs x ~ N(0, I_n) is sqrt(q0). This fixes the
    realised gate sharpness per net (finite-width nets otherwise drift in scale, log-normally) and, for GLU, removes the
    unstable scale fixed point of the product. pilot = 0: infinite-width gains only."""
    out = _raw_weights(spec, n, depth, *key)
    bs = biases(spec, n, depth, *key)
    target = math.sqrt(spec.q0 * (1 - spec.bias_frac))
    if pilot:
        H = rng(*key, 77).standard_normal((pilot, n))
        for l, Wt in enumerate(out):
            zs = [H @ W.T.astype(np.float64) for W in Wt]
            for W, z in zip(Wt, zs):
                W *= np.float32(target / max(float(np.sqrt(np.mean(z ** 2))), 1e-30))
            zs = [H @ W.T.astype(np.float64) for W in Wt]
            if bs is not None:
                zs[0] = zs[0] + bs[l][None, :]
            H = spec.phi(zs[0]) if spec.arch == 'plain' else spec.phi(zs[0]) * zs[1]
    return out


def biases(spec, n, depth, *key):
    """Per-layer biases b_l ~ N(0, bias_frac * q0) (None when bias_frac = 0); deterministic in the key."""
    if spec.bias_frac <= 0:
        return None
    g = rng(*key, 55)
    return [(g.standard_normal(n) * math.sqrt(spec.bias_frac * spec.q0)).astype(np.float32) for _ in range(depth)]


def _raw_weights(spec, n, depth, *key):
    g = rng(*key)
    out = []
    for gl in spec.layer_gains(depth):
        if spec.arch == 'plain':
            out.append(((g.standard_normal((n, n)) * (gl / math.sqrt(n))).astype(np.float32),))
        else:
            out.append(((g.standard_normal((n, n)) * (gl / math.sqrt(n))).astype(np.float32),
                        (g.standard_normal((n, n)) * (gl / math.sqrt(n))).astype(np.float32)))
    return out


def fwd_cols(spec, X, Ws, bs=None):
    """Forward states stored as columns (n x N). Ws: list of weight tuples for consecutive layers; bs: biases or None."""
    for l, Wt in enumerate(Ws):
        if spec.arch == 'plain':
            Z = Wt[0] @ X
            if bs is not None:
                Z += bs[l][:, None]
            X = spec.phi(Z).astype(np.float32)
        else:
            X = (spec.phi(Wt[0] @ X) * (Wt[1] @ X)).astype(np.float32)
    return X


def fwd_rows(spec, H, Ws, bs=None):
    """Forward with states as rows (N x n), float64."""
    for l, Wt in enumerate(Ws):
        if spec.arch == 'plain':
            H = spec.phi(H @ Wt[0].T.astype(np.float64) + (0 if bs is None else bs[l].astype(np.float64)[None, :]))
        else:
            H = spec.phi(H @ Wt[0].T.astype(np.float64)) * (H @ Wt[1].T.astype(np.float64))
    return H


def preact_std(spec, Ws, n, N=20000, seed=0, bs=None):
    """Per-layer RMS pre-activation (gate input) on Gaussian inputs: the realised scale s_l (sharpness = beta s_l)."""
    H = rng(seed, 99).standard_normal((N, n))
    out = []
    for l, Wt in enumerate(Ws):
        Z = H @ Wt[0].T.astype(np.float64) + (0 if bs is None else bs[l].astype(np.float64)[None, :])
        out.append(float(np.sqrt(np.mean(Z ** 2))))
        H = spec.phi(Z) if spec.arch == 'plain' else spec.phi(Z) * (H @ Wt[1].T.astype(np.float64))
    return out
