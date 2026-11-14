"""Random critical He-ReLU nets, samplers and FLOP conventions for the unrestricted-estimators lane.

Net: W_l iid N(0, 2/n) (n x n), no bias, x ~ N(0, I_n); h_0 = x, h_l = relu(W_l h_{l-1}) (+ s_l xi_l for l < L),
output a_L = relu(W_L h_{L-1}) in R^n (no noise on the output). Noise s_l = sigma r_l, r_l = RMS of relu(W_l h_{l-1})
in a seeded pilot of the noisy net (narrow-deep convention).
FLOPs (flopscope, as work/narrow-deep/lib/nd.py): matmul (M,K)@(K,N): M N (2K-1); elementwise 1; normal cdf 96;
normal pdf 54; exp/log 16; normal draw 16. One noiseless forward pass of one input: L n (2n - 1) + L n = 2 n^2 L.
float64 numpy throughout.
"""
import math
import numpy as np
from scipy.special import gammaln

F_CDF, F_PDF, F_TRANS, F_NORMAL = 96, 54, 16, 16


def mm_flops(M, K, N):
    return M * N * (2 * K - 1)


def pass_flops(n, L):
    """FLOPs of one noiseless forward pass of one input."""
    return L * (mm_flops(1, n, n) + n)


def make_weights(seed, n, L):
    rng = np.random.default_rng([20261031, int(seed), int(n), int(L)])
    return [rng.standard_normal((n, n)) * math.sqrt(2.0 / n) for _ in range(L)]


def mean_radius(n):
    """E|x| for x ~ N(0, I_n)."""
    return math.exp(0.5 * math.log(2.0) + gammaln((n + 1) / 2.0) - gammaln(n / 2.0))


class Net:
    def __init__(self, seed, n, L, sigma=0.0, pilot_m=1 << 14):
        self.seed, self.n, self.L, self.sigma = int(seed), int(n), int(L), float(sigma)
        self.W = make_weights(seed, n, L)
        self.r = np.ones(L)
        if sigma > 0:
            self.r = self.pilot_rms(pilot_m)
        self.s = self.sigma * self.r

    def name(self):
        return f"n{self.n}_L{self.L}_s{self.sigma:g}_seed{self.seed}"

    def pilot_rms(self, m):
        rng = np.random.default_rng([99000017, self.seed, self.n, self.L])
        h = rng.standard_normal((m, self.n))
        r = np.empty(self.L)
        for l in range(self.L):
            a = np.maximum(h @ self.W[l].T, 0.0)
            r[l] = math.sqrt(float(np.mean(a * a)))
            if l < self.L - 1:
                h = a + rng.standard_normal(a.shape) * (self.sigma * r[l])
        return r

    def forward(self, X, rng=None, noise=None):
        """X (P, n) -> a_L (P, n). Noise on hidden layers only if sigma > 0 and noise is not False."""
        use_noise = (self.sigma > 0) if noise is None else bool(noise)
        h = X
        for l in range(self.L):
            h = np.maximum(h @ self.W[l].T, 0.0)
            if use_noise and l < self.L - 1:
                h = h + rng.standard_normal(h.shape) * self.s[l]
        return h

    def forward_pair(self, X, rng):
        """Noiseless and noisy outputs on the same inputs (paired, for the noise effect)."""
        h0 = X
        h1 = X
        for l in range(self.L):
            h0 = np.maximum(h0 @ self.W[l].T, 0.0)
            h1 = np.maximum(h1 @ self.W[l].T, 0.0)
            if l < self.L - 1:
                h1 = h1 + rng.standard_normal(h1.shape) * self.s[l]
        return h0, h1


def sphere_from_gauss(G):
    nr = np.linalg.norm(G, axis=1, keepdims=True)
    return G / nr


def rb_stats(net, U, chunk=1 << 15):
    """Radius-exact statistics on unit directions U (P, n), noiseless: g(u) = a_L(u) (homogeneous of degree 1).
    Returns (sum g, sum g^2) over the rows."""
    s1 = np.zeros(net.n)
    s2 = np.zeros(net.n)
    for i in range(0, U.shape[0], chunk):
        g = net.forward(U[i:i + chunk], noise=False)
        s1 += g.sum(0)
        s2 += (g * g).sum(0)
    return s1, s2


def sobol_directions(n, m_log2, scramble, seed=None, skip_first=True):
    """2^m_log2 Sobol points in [0,1]^n mapped to Gaussians by the inverse cdf and normalised to the sphere.
    Unscrambled: the first 2^m points of the unscrambled sequence, centred by a shift of 2^-(m+1) in every
    coordinate (so no point is 0 or the all-1/2 point, which would map to the origin)."""
    from scipy.stats import qmc
    from scipy.special import ndtri
    eng = qmc.Sobol(d=n, scramble=scramble, seed=seed)
    N = 1 << m_log2
    if scramble:
        P = eng.random_base2(m_log2)
    else:
        P = eng.random(N) + 2.0 ** (-(m_log2 + 1))
    P = np.clip(P, 1e-300, 1 - 1e-16)
    return sphere_from_gauss(ndtri(P))
