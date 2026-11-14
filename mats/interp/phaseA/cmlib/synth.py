"""Known-answer worlds for validation.

MixWorld   p(h|z) = 1/2 N(mu_z + a m_z, I) + 1/2 N(mu_z - a m_z, I)   (non-Gaussian, z in {0..K-1})
           q_t(h|z) = 1/2 N(mu_z ± t a m_z, I + (1-t^2) a^2 m_z m_z^T): t = 0 is the moment-matched Gaussian,
           t = 1 is p. The ratio depends on h only through tau = m_z.(h - mu_z); the orthogonal part is N(0, I) in
           both, so log w, E[G], L(alpha), KL and chi^2 reduce to 1-D quadrature (exact to ~1e-12).
           G(h, z) = g(tau) with g(tau) = tau^4 / 10 (closed form) or any g via quadrature.
ScaleWorld p = N(0, I_d), q = N(0, k^2 I_d), G = |h|^2 / d: Delta = k^2 - 1, E_q w^2 = (k^2 / sqrt(2k^2 - 1))^d.
IgnoreNet  hand-built 2-layer net on h = [A(2) | B(2) | C(2)], label y. A Gaussian given y; B has a strong
           bimodal coordinate b1 that the net ignores (zero weights) and a Gaussian coordinate b2 that it reads;
           C has a bimodal coordinate c1 that the net reads through ReLUs (if c_bimodal) else Gaussian.
"""
import numpy as np
from scipy.special import logsumexp as _lse

GH_X, GH_W = np.polynomial.hermite.hermgauss(80)


def _norm_logpdf(x, m, s2):
    return -0.5 * (x - m) ** 2 / s2 - 0.5 * np.log(2 * np.pi * s2)


class MixWorld:
    def __init__(self, d=8, K=3, a=1.0, mu_scale=2.0, seed=0, g='quartic'):
        r = np.random.default_rng(seed)
        self.d, self.K, self.a = d, K, a
        self.pz = np.ones(K) / K
        self.mu = r.standard_normal((K, d)) * mu_scale / np.sqrt(d)
        m = r.standard_normal((K, d)); self.m = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.gname = g

    # 1-D pieces ---------------------------------------------------------------
    def comps(self, t):
        """Mixture components (means, variance) of tau under q_t (t=1: p)."""
        s2 = 1 + (1 - t * t) * self.a ** 2
        return np.array([t * self.a, -t * self.a]), s2

    def logdens_tau(self, tau, t):
        ms, s2 = self.comps(t)
        return _lse(np.stack([_norm_logpdf(tau, mm, s2) for mm in ms]), axis=0) - np.log(2)

    def g(self, tau):
        if self.gname == 'quartic':
            return tau ** 4 / 10
        if self.gname == 'quadratic':
            return tau ** 2
        if self.gname == 'softplus':
            return np.logaddexp(0, 2 * tau - 1)
        raise ValueError(self.gname)

    def EG(self, t):
        ms, s2 = self.comps(t)
        return float(np.mean([np.sum(GH_W * self.g(mm + np.sqrt(2 * s2) * GH_X)) / np.sqrt(np.pi) for mm in ms]))

    def _grid(self, t):
        tau = np.linspace(-14, 14, 20001)
        return tau, self.logdens_tau(tau, 1.0), self.logdens_tau(tau, t)

    def delta(self, t):
        """Delta = E_q G - E_p G."""
        return self.EG(t) - self.EG(1.0)

    def L_alpha(self, t, alphas):
        """Exact tempered path: p_alpha ∝ q^(1-alpha) p^alpha; L(alpha) = E_{p_alpha} g(tau)."""
        tau, lp, lq = self._grid(t)
        out = []
        for al in alphas:
            lo = al * lp + (1 - al) * lq; w = np.exp(lo - lo.max())
            out.append(np.sum(w * self.g(tau)) / np.sum(w))
        return np.array(out)

    def kl_pq(self, t):
        tau, lp, lq = self._grid(t); dx = tau[1] - tau[0]
        return float(np.sum(np.exp(lp) * (lp - lq)) * dx)

    def chi2_pq(self, t):
        """chi^2(p||q) = E_q w^2 - 1."""
        tau, lp, lq = self._grid(t); dx = tau[1] - tau[0]
        return float(np.sum(np.exp(2 * lp - lq)) * dx - 1)

    # samplers ---------------------------------------------------------------------
    def sample_z(self, n, rng):
        return rng.choice(self.K, n, p=self.pz)

    def sample(self, z, t, rng):
        """Draw h ~ q_t(.|z) (t = 1: p)."""
        z = np.asarray(z); n = len(z)
        ms, s2 = self.comps(t)
        s = rng.choice([0, 1], n)
        m = self.m[z]
        eps = rng.standard_normal((n, self.d))
        along = (eps * m).sum(1)
        eps = eps - along[:, None] * m + (np.sqrt(s2) * along)[:, None] * m
        return self.mu[z] + ms[s][:, None] * m + eps

    def tau(self, H, z):
        return ((H - self.mu[z]) * self.m[z]).sum(1)

    def G(self, H, z):
        return self.g(self.tau(H, z))

    def log_w(self, H, z, t):
        """Exact log p(h|z) / q_t(h|z)."""
        tau = self.tau(H, z)
        return self.logdens_tau(tau, 1.0) - self.logdens_tau(tau, t)


class ScaleWorld:
    def __init__(self, d=31, k=1.2):
        self.d, self.k = d, k

    def sample_p(self, n, rng):
        return rng.standard_normal((n, self.d))

    def sample_q(self, n, rng):
        return self.k * rng.standard_normal((n, self.d))

    def G(self, H):
        return (H ** 2).sum(1) / self.d

    def delta(self):
        return self.k ** 2 - 1

    def log_w(self, H):
        return -0.5 * (H ** 2).sum(1) * (1 - 1 / self.k ** 2) + self.d * np.log(self.k)

    def chi2(self):
        k2 = self.k ** 2
        return float((k2 / np.sqrt(2 * k2 - 1)) ** self.d - 1) if 2 * k2 > 1 else float('inf')

    def L_alpha(self, alphas):
        """p_alpha = N(0, s_a^2 I) with 1/s_a^2 = alpha + (1-alpha)/k^2; L = s_a^2."""
        return np.array([1.0 / (a + (1 - a) / self.k ** 2) for a in alphas])


class IgnoreNet:
    """h = [a1 a2 | b1 b2 | c1 c2]; y in {0,1}. F(h) = relu(W1 h + b1) (8 units), G = CE(W2 F(h) + b2, y).
    W1 column for b1 is exactly zero (downstream ignores b1). Index sets: A=[0,1], B=[2,3], C=[4,5]."""
    A, B, C = [0, 1], [2, 3], [4, 5]

    def __init__(self, c_bimodal=True, seed=0, b_sep=2.0, c_sep=1.5):
        r = np.random.default_rng(seed)
        self.c_bimodal, self.b_sep, self.c_sep = c_bimodal, b_sep, c_sep
        self.muA = np.array([[1.0, 0.5], [-1.0, -0.5]])
        W1 = r.standard_normal((8, 6))
        W1[:, 2] = 0.0                       # b1 ignored
        W1[:, 4] *= 2.5                      # c1 read strongly
        self.W1, self.b1 = W1, r.standard_normal(8) * 0.5
        self.W2 = 0.4 * r.standard_normal((2, 8)); self.b2 = np.zeros(2)
        self.W2[1] += 0.4 * np.sign(self.W1[:, 4])   # make the label depend on c through the ReLUs

    def sample(self, y, rng):
        y = np.asarray(y); n = len(y)
        h = np.empty((n, 6))
        h[:, :2] = self.muA[y] + rng.standard_normal((n, 2))
        h[:, 2] = rng.choice([-1, 1], n) * self.b_sep + 0.5 * rng.standard_normal(n)
        h[:, 3] = rng.standard_normal(n)
        if self.c_bimodal:
            h[:, 4] = rng.choice([-1, 1], n) * self.c_sep + 0.6 * rng.standard_normal(n)
        else:
            h[:, 4] = np.sqrt(self.c_sep ** 2 + 0.36) * rng.standard_normal(n)
        h[:, 5] = 0.5 * h[:, 4] * (y * 2 - 1) + rng.standard_normal(n)
        return h

    def F(self, H, y=None):
        return np.maximum(H @ self.W1.T + self.b1, 0.0)

    def G_next(self, H1, y):
        lg = H1 @ self.W2.T + self.b2
        return np.logaddexp(lg[:, 0], lg[:, 1]) - lg[np.arange(len(y)), y]

    def G(self, H, y):
        return self.G_next(self.F(H), y)

    def logits(self, H):
        return self.F(H) @ self.W2.T + self.b2

    @staticmethod
    def G_from_logits(lg, y):
        return np.logaddexp(lg[:, 0], lg[:, 1]) - lg[np.arange(len(y)), y]
