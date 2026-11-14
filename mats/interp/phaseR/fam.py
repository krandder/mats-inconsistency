"""Tractable families for the forward chain (phase R). All densities in nats (convert with LOG2), torch float64.

G   moment-matched Gaussian (optionally with a defensive component (1-delta) q + delta N(mu, 4 Sigma))
NC  neuron code, singletons: product over neurons of 1-D Gaussian mixtures (quantile cells of s_i, + sigma^2)
GC  neuron code with one Gaussian dependence group: Gaussian copula over the NC marginals
KN  know-nothing: uniform on the LN sphere (radius sqrt(d), centred subspace) convolved with N(0, sigma^2 I)
"""
import math
import numpy as np
import torch
from scipy.special import ive, gammaln

DT = torch.float64
L2PI = math.log(2 * math.pi)


def ndtri_log(logp):
    """Phi^{-1}(exp(logp)) for logp <= log(1/2) (lower tail), robust down to logp ~ -1e4. torch."""
    p = logp.exp()
    u = torch.special.ndtri(p.clamp(1e-300, 0.5))
    small = logp < -30
    if small.any():
        t = -2 * logp[small]
        u0 = -torch.sqrt(t - torch.log(t) - L2PI)
        u = u.clone(); u[small] = u0
    for _ in range(3):                                    # Newton on log Phi(u) = logp
        lphi = torch.special.log_ndtr(u)
        ldens = -0.5 * u * u - 0.5 * L2PI
        u = u - (lphi - logp) * torch.exp(lphi - ldens)
    return u


class Gauss:
    def __init__(self, mu, S, delta=0.0, infl=4.0):
        self.mu = mu.to(DT); self.S = S.to(DT); self.d = len(mu)
        self.L = torch.linalg.cholesky(self.S); self.ld = 2 * self.L.diagonal().log().sum()
        self.delta, self.infl = delta, infl

    @staticmethod
    def fit(H, delta=0.0):
        mu = H.mean(0); C = torch.cov(H.T)
        return Gauss(mu, 0.5 * (C + C.T), delta)

    def _lp(self, H, scale=1.0):
        r = torch.linalg.solve_triangular(self.L, (H - self.mu).T, upper=False).T
        return -0.5 * (r * r).sum(1) / scale - 0.5 * self.ld - 0.5 * self.d * (L2PI + math.log(scale))

    def logpdf(self, H):
        lp = self._lp(H)
        if self.delta > 0:
            lp = torch.logaddexp(math.log(1 - self.delta) + lp, math.log(self.delta) + self._lp(H, self.infl))
        return lp

    def sample(self, n, gen):
        Z = torch.randn(n, self.d, generator=gen, dtype=DT)
        X = self.mu + Z @ self.L.T
        if self.delta > 0:
            m = torch.rand(n, generator=gen, dtype=DT) < self.delta
            X[m] = self.mu + math.sqrt(self.infl) * (Z[m] @ self.L.T)
        return X

    def marg_logpdf(self, i, x):
        v = self.S[i, i]
        return -0.5 * (x - self.mu[i]) ** 2 / v - 0.5 * (L2PI + torch.log(v))

    def gauss(self):
        return self.mu, self.S

    def prob_halfspace(self, a, t):
        """q(a.h > t), exact (with the defensive component)."""
        m = float(a @ self.mu); s = float(torch.sqrt(a @ self.S @ a))
        from scipy.stats import norm
        p = norm.logsf((t - m) / s)
        if self.delta > 0:
            p = np.logaddexp(math.log(1 - self.delta) + p, math.log(self.delta) + norm.logsf((t - m) / (s * math.sqrt(self.infl))))
        return float(p)                                   # log prob (nats)


class NC:
    """Product of per-neuron 1-D mixtures sum_k pi_ik N(m_ik, v_ik)."""

    def __init__(self, pi, m, v):
        self.lpi = pi.clamp_min(1e-300).log(); self.m = m; self.v = v; self.sd = v.sqrt(); self.d = m.shape[0]
        self._mom()

    def _mom(self):
        pi = self.lpi.exp()
        self.mean = (pi * self.m).sum(1)
        self.var = (pi * (self.v + self.m ** 2)).sum(1) - self.mean ** 2

    @staticmethod
    def fit_from_s(Sv, sigma, M=32):
        """Sv (N, d) noiseless pushforward values; marginal of h_i = law(s_i) * N(0, sigma^2) by M quantile cells."""
        N, d = Sv.shape
        Ss, _ = torch.sort(Sv, 0)
        edges = torch.linspace(0, N, M + 1).round().long()
        m = torch.stack([Ss[edges[k]:edges[k + 1]].mean(0) for k in range(M)], 1)            # (d, M)
        w = torch.stack([Ss[edges[k]:edges[k + 1]].var(0, unbiased=False) for k in range(M)], 1)
        pi = torch.stack([torch.full((d,), float(edges[k + 1] - edges[k]) / N, dtype=DT) for k in range(M)], 1)
        return NC(pi, m, w + sigma ** 2)

    def comp_logpdf(self, H):
        """(n, d, M) log pi + log N per component."""
        x = H[:, :, None]
        return self.lpi[None] - 0.5 * (x - self.m[None]) ** 2 / self.v[None] - 0.5 * (L2PI + self.v[None].log())

    def marg_logpdf_all(self, H):
        return torch.logsumexp(self.comp_logpdf(H), 2)    # (n, d)

    def logpdf(self, H):
        return self.marg_logpdf_all(H).sum(1)

    def log_cdf_sf(self, H):
        """log F_i(h_i), log (1 - F_i(h_i)) (n, d)."""
        zz = (H[:, :, None] - self.m[None]) / self.sd[None]
        lF = torch.logsumexp(self.lpi[None] + torch.special.log_ndtr(zz), 2)
        lS = torch.logsumexp(self.lpi[None] + torch.special.log_ndtr(-zz), 2)
        return lF, lS

    def normal_scores(self, H):
        lF, lS = self.log_cdf_sf(H)
        lo = lF <= lS
        u = torch.empty_like(H)
        u[lo] = ndtri_log(lF[lo]); u[~lo] = -ndtri_log(lS[~lo])
        return u

    def sample(self, n, gen):
        pi = self.lpi.exp()
        cum = pi.cumsum(1)
        r = torch.rand(n, self.d, generator=gen, dtype=DT)
        k = torch.searchsorted(cum.contiguous(), r.T.contiguous()).T.clamp_max(self.m.shape[1] - 1)   # (n, d)
        idx = torch.arange(self.d)[None].expand(n, -1)
        return self.m[idx, k] + self.sd[idx, k] * torch.randn(n, self.d, generator=gen, dtype=DT)

    def quantile_table(self, ngrid=4096):
        lo = (self.m - 8 * self.sd).min(1).values; hi = (self.m + 8 * self.sd).max(1).values
        t = torch.linspace(0, 1, ngrid, dtype=DT)
        X = lo[:, None] + (hi - lo)[:, None] * t[None]                       # (d, ngrid)
        F = (self.lpi.exp()[:, None, :] * torch.special.ndtr((X[:, :, None] - self.m[:, None, :]) / self.sd[:, None, :])).sum(2)
        return X, F

    def inv_cdf(self, U, tab):
        """U (n, d) uniforms -> h via monotone interpolation of the tabulated CDF."""
        X, F = tab
        out = torch.empty_like(U)
        for i in range(self.d):
            j = torch.searchsorted(F[i].contiguous(), U[:, i].contiguous()).clamp(1, F.shape[1] - 1)
            f0, f1 = F[i, j - 1], F[i, j]; x0, x1 = X[i, j - 1], X[i, j]
            out[:, i] = x0 + (x1 - x0) * ((U[:, i] - f0) / (f1 - f0).clamp_min(1e-300)).clamp(0, 1)
        return out

    def inv_cdf_u(self, u, tab, ucut=6.0, iters=64):
        """h_i = F_i^{-1}(Phi(u_i)) from normal scores u (n, d), with exact tails (review fix 6, 2026-10-29).
        |u| <= ucut: the tabulated CDF (accurate to a fraction of its grid step there). |u| > ucut, where the table is
        bounded at m -/+ 8 sd and Phi(u) rounds to 1 for large u: bisection on log F (lower tail) or log(1 - F) (upper
        tail) of the mixture, inside the bracket [min_k(m_k + sd_k u), max_k(m_k + sd_k u)], which contains the root for
        any u (F(h) <= Phi(max_k (h - m_k)/sd_k) and F(h) >= Phi(min_k (h - m_k)/sd_k))."""
        out = self.inv_cdf(torch.special.ndtr(u.clamp(-ucut, ucut)), tab)
        far = u.abs() > ucut
        if far.any():
            ii, jj = far.nonzero(as_tuple=True)
            uu = u[ii, jj]; m = self.m[jj]; sd = self.sd[jj]; lpi = self.lpi[jj]
            cand = m + sd * uu[:, None]
            a = cand.min(1).values; b = cand.max(1).values
            upper = uu > 0
            target = torch.where(upper, torch.special.log_ndtr(-uu), torch.special.log_ndtr(uu))
            for _ in range(iters):
                mid = 0.5 * (a + b)
                z = (mid[:, None] - m) / sd
                lF = torch.logsumexp(lpi + torch.special.log_ndtr(z), 1)
                lS = torch.logsumexp(lpi + torch.special.log_ndtr(-z), 1)
                go_up = torch.where(upper, lS > target, lF < target)
                a = torch.where(go_up, mid, a); b = torch.where(go_up, b, mid)
            out = out.clone(); out[ii, jj] = 0.5 * (a + b)
        return out

    def gauss(self):
        return self.mean, torch.diag(self.var)

    def marg_logpdf(self, i, x):
        return torch.logsumexp(self.lpi[i][None] - 0.5 * (x[:, None] - self.m[i][None]) ** 2 / self.v[i][None]
                               - 0.5 * (L2PI + self.v[i][None].log()), 1)


class GC:
    """Gaussian copula with NC marginals: log q(h) = sum_i log f_i(h_i) - 1/2 log det R - 1/2 u^T (R^{-1} - I) u."""

    def __init__(self, nc, R):
        self.nc = nc; self.d = nc.d
        self.R = R; self.LR = torch.linalg.cholesky(R); self.ldR = 2 * self.LR.diagonal().log().sum()
        self.Ri = torch.cholesky_inverse(self.LR)
        self.tab = nc.quantile_table()
        self._gm = None

    @staticmethod
    def fit_from_s(Sv, H, sigma, M=32, shrink=1e-3):
        nc = NC.fit_from_s(Sv, sigma, M)
        u = nc.normal_scores(H)
        u = u.clamp(-8, 8)
        R = torch.corrcoef(u.T)
        R = (1 - shrink) * R + shrink * torch.eye(nc.d, dtype=DT)
        return GC(nc, R)

    def logpdf(self, H):
        u = self.nc.normal_scores(H)
        quad = (u @ (self.Ri - torch.eye(self.d, dtype=DT)) * u).sum(1)
        return self.nc.logpdf(H) - 0.5 * self.ldR - 0.5 * quad

    def sample(self, n, gen):
        u = torch.randn(n, self.d, generator=gen, dtype=DT) @ self.LR.T
        return self.nc.inv_cdf_u(u, self.tab)

    def gauss(self):
        if self._gm is None:
            g = torch.Generator().manual_seed(77)
            X = self.sample(1 << 14, g); self._gm = (X.mean(0), torch.cov(X.T))
        return self._gm

    def marg_logpdf(self, i, x):
        return self.nc.marg_logpdf(i, x)


class KN:
    """Uniform on the LN sphere (radius sqrt(d), centred subspace) convolved with N(0, s2 I_d)."""

    def __init__(self, d, sigma):
        self.d, self.s2 = d, sigma ** 2; self.R = math.sqrt(d); self.n = d - 1
        self.u = torch.full((d,), 1 / math.sqrt(d), dtype=DT)

    def logpdf(self, H):
        hp = H @ self.u; rho = (H - hp[:, None] * self.u).norm(dim=1).numpy()
        n, R, s2 = self.n, self.R, self.s2; v = n / 2 - 1; k = np.maximum(rho * R / s2, 1e-12)
        lf = (-n / 2 * math.log(2 * math.pi * s2) - (rho ** 2 + R ** 2) / (2 * s2) + gammaln(n / 2)
              + v * (math.log(2) - np.log(k)) + np.log(np.maximum(ive(v, k), 1e-300)) + k)
        return torch.tensor(lf, dtype=DT) - 0.5 * math.log(2 * math.pi * s2) - hp ** 2 / (2 * s2)

    def sample(self, n, gen):
        z = torch.randn(n, self.d, generator=gen, dtype=DT); z = z - z.mean(1, keepdim=True)
        s = z * (self.R / z.norm(dim=1, keepdim=True))
        return s + math.sqrt(self.s2) * torch.randn(n, self.d, generator=gen, dtype=DT)

    def gauss(self):
        d = self.d; P = torch.eye(d, dtype=DT) - 1.0 / d
        return torch.zeros(d, dtype=DT), P * (d / (d - 1)) + self.s2 * torch.eye(d, dtype=DT)

    def marg_logpdf(self, i, x):
        # marginal of one coordinate: s_i = sqrt(d) * (centred unit vector)_i ~ scaled Beta; use a 1-D node mixture
        if not hasattr(self, '_mnodes'):
            g = torch.Generator().manual_seed(5); S = self.sample(1 << 15, g)
            self._mnodes = (S[:, 0] - math.sqrt(self.s2) * 0).clone()
        raise NotImplementedError


class Mix:
    """(1 - delta) q + delta N(mu, 4 Sigma) for any family q with gauss()."""

    def __init__(self, q, delta=0.01, infl=4.0):
        self.q, self.delta = q, delta; mu, S = q.gauss(); self.D = Gauss(mu, infl * S); self.d = q.d

    def logpdf(self, H):
        return torch.logaddexp(math.log(1 - self.delta) + self.q.logpdf(H), math.log(self.delta) + self.D.logpdf(H))

    def sample(self, n, gen):
        X = self.q.sample(n, gen); m = torch.rand(n, generator=gen, dtype=DT) < self.delta
        if m.any(): X[m] = self.D.sample(int(m.sum()), gen)
        return X

    def gauss(self):
        return self.q.gauss()
