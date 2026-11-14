"""Exact stationary law of the radius chain of Model S at finite width n (sigma = 1 by scale invariance).

Chain (astra P4, stationary lane FORMAL 3.1):  Q_l = |h_l|^2 / n,  Q' = A Q + B,
  M ~ Bin(n, 1/2), U ~ chi2_M, V ~ chi2_{n-M}, A = 2U/n, B = (U+V)/n.
Reduction used here (proof in notes/radius_proof.md):
  Q_l = (S_l / n) r_l,  S_l ~ chi2_n independent of r_l,  r_l = 1 + A_l r_{l-1}  (perpetuity with B = 1),
  and with u = log(r - 1):  u' = log A + softplus(u)   (A = 0 with probability 2^-n sends u to -inf, r = 1).
Stationary law of u: an atom pi0 at -inf plus a density p(u) with
  p(u') = int g(u' - softplus(u)) p(u) du + pi0 g(u'),   pi0 = 2^-n,
where g is the density of log A on {A > 0} (mass 1 - 2^-n).
Discretisation: Nystrom with the trapezoid rule on a uniform u-grid (spectrally accurate for this smooth
kernel); the stationary vector is found by inverse iteration and then polished by positive power steps.
"""
import math
import numpy as np
from scipy.special import gammaln, gammainc, gammaincc, logsumexp, digamma
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq

LOG2 = math.log(2.0)


def binom_logw(n):
    m = np.arange(0, n + 1)
    return gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1) - n * LOG2


class LogA:
    """Law of log A, A = (2/n) chi2_M, M ~ Bin(n, 1/2), restricted to A > 0."""

    def __init__(self, n, wcut=-745.0):
        self.n = n
        lw = binom_logw(n)
        m = np.arange(1, n + 1)
        keep = lw[1:] > wcut  # weights below exp(-745) underflow anyway
        self.m = m[keep].astype(float)
        self.lw = lw[1:][keep]
        self.c = self.lw - gammaln(self.m / 2.0)  # log of P(m)/Gamma(m/2)
        self.lt0 = math.log(n / 4.0)

    def logpdf(self, s, chunk=1 << 16):
        """log density of log A at s:  -t + log sum_m P(m) t^{m/2}/Gamma(m/2),  t = n e^s / 4."""
        s = np.asarray(s, float)
        out = np.empty(s.shape)
        sf, of = s.ravel(), out.ravel()
        for a in range(0, sf.size, chunk):
            lt = self.lt0 + sf[a:a + chunk]
            terms = self.c[None, :] + 0.5 * self.m[None, :] * lt[:, None]
            of[a:a + chunk] = logsumexp(terms, axis=1) - np.exp(lt)
        return out

    def pdf(self, s):
        return np.exp(self.logpdf(s))

    def cdf(self, s):
        """P(log A <= s, A > 0)."""
        s = np.asarray(s, float)
        t = np.exp(self.lt0 + s)
        return np.sum(np.exp(self.lw)[None, :] * gammainc(self.m[None, :] / 2.0, t.ravel()[:, None]), 1).reshape(s.shape)

    def sf(self, s):
        """P(log A > s)."""
        s = np.asarray(s, float)
        t = np.exp(self.lt0 + s)
        return np.sum(np.exp(self.lw)[None, :] * gammaincc(self.m[None, :] / 2.0, t.ravel()[:, None]), 1).reshape(s.shape)


def E_AlogA(n):
    lw = binom_logw(n)
    m = np.arange(1, n + 1, dtype=float)
    return float(np.sum(np.exp(lw[1:]) * (2 * m / n) * (math.log(4.0 / n) + digamma(m / 2 + 1))))


def kesten_C(n):
    return 1.0 / E_AlogA(n)


def softplus(u):
    return np.logaddexp(0.0, u)


class RadiusLaw:
    def __init__(self, n, h=None, umin=None, umax=None, polish=200, verbose=False):
        self.n = n
        sd = math.sqrt(5.0 / n) if n >= 8 else 0.8
        self.h = h if h is not None else sd / 3.0
        C = kesten_C(n)
        self.umax = umax if umax is not None else math.log(C) + 38.0
        if umin is None:
            umin = -2.5 * 38.0 if n <= 16 else (-40.0 if n <= 64 else -6.0)
        self.umin = umin
        self.u = np.arange(self.umin, self.umax + 0.5 * self.h, self.h)
        self.la = LogA(n)
        self.pi0_rate = 2.0 ** (-n)
        self._build()
        self._solve(polish, verbose)

    def _build(self):
        u, h = self.u, self.h
        N = u.size
        phi = softplus(u)
        G = np.empty((N, N))
        # columns where softplus(u_j) == u_j to double precision use on-grid differences
        ongrid = u > 37.5
        if ongrid.any():
            d = np.arange(-(N - 1), N) * h
            gd = self.la.pdf(d)
            for j in np.where(ongrid)[0]:
                G[:, j] = gd[(np.arange(N) - j) + (N - 1)]
        for j in np.where(~ongrid)[0]:
            G[:, j] = self.la.pdf(u - phi[j])
        self.G = G
        self.g0 = self.la.pdf(u)  # from the atom (r = 1, softplus(-inf) = 0)
        # transition matrix on (grid densities, atom): x = T x
        T = np.empty((N + 1, N + 1))
        T[:N, :N] = h * G
        T[:N, N] = self.g0
        T[N, :N] = self.pi0_rate * h
        T[N, N] = self.pi0_rate
        self.T = T
        # mass-conservation diagnostic: total outgoing mass per source column
        self.colmass = h * G.sum(0) + self.pi0_rate

    def _solve(self, polish, verbose):
        N = self.u.size
        T = self.T
        A = np.eye(N + 1) - T
        lu = lu_factor(A)
        x = np.ones(N + 1)
        for it in range(3):
            x = lu_solve(lu, x)
            x = np.abs(x)
            x /= self._mass(x)
        # positive power steps (keep relative accuracy in the tails)
        for it in range(polish):
            y = T @ x
            y /= self._mass(y)
            x = y
        self.lam = float(self._mass(T @ x) / self._mass(x))
        self.p = x[:N]
        self.pi0 = x[N]

    def _mass(self, x):
        return self.h * x[:-1].sum() + x[-1]

    # ---------------- observables ----------------
    def p_interp(self, uu):
        """Nystrom interpolant of the density of u at arbitrary points."""
        uu = np.atleast_1d(np.asarray(uu, float))
        phi = softplus(self.u)
        out = np.zeros(uu.shape)
        for k in range(uu.size):
            out[k] = self.h * np.dot(self.la.pdf(uu[k] - phi), self.p) + self.pi0 * self.la.pdf(np.array([uu[k]]))[0]
        return out

    def r_cdf(self, x):
        """P(r <= x), x >= 1, via the Nystrom interpolant (exact given p)."""
        if x <= 1.0:
            return 0.0 if x < 1.0 else self.pi0
        c = math.log(x - 1.0)
        phi = softplus(self.u)
        return float(self.pi0 + self.h * np.dot(self.la.cdf(c - phi), self.p) + self.pi0 * self.la.cdf(np.array([c]))[0])

    def r_sf(self, x):
        """P(r > x) without cancellation."""
        c = math.log(x - 1.0)
        phi = softplus(self.u)
        return float(self.h * np.dot(self.la.sf(c - phi), self.p) + self.pi0 * self.la.sf(np.array([c]))[0])

    def Q_cdf(self, y):
        """P(Q <= y), Q = (S/n) r, S ~ chi2_n independent: trapezoid over the u-grid."""
        n = self.n
        w = gammainc(n / 2.0, (n * y / 2.0) * np.exp(-softplus(self.u)))
        return float(self.pi0 * gammainc(n / 2.0, n * y / 2.0) + self.h * np.dot(w, self.p))

    def Q_sf(self, y):
        n = self.n
        w = gammaincc(n / 2.0, (n * y / 2.0) * np.exp(-softplus(self.u)))
        return float(self.pi0 * gammaincc(n / 2.0, n * y / 2.0) + self.h * np.dot(w, self.p))

    def Q_quantile(self, q):
        n = self.n
        f = (lambda y: self.Q_cdf(y) - q) if q <= 0.5 else (lambda y: (1 - q) - self.Q_sf(y))
        lo, hi = 1e-8, 10.0 * n
        while f(hi) < 0:
            hi *= 10
        return brentq(f, lo, hi, xtol=1e-14 * n, rtol=1e-14, maxiter=500)

    def Q_logmoment(self):
        """E log Q = E log(S/n) + E log r."""
        n = self.n
        elogS = math.log(2.0) + float(digamma(n / 2.0)) - math.log(n)
        elogr = self.h * np.dot(softplus(self.u), self.p)  # atom contributes log 1 = 0
        return elogS + elogr


def simulate_Q(n, T, P, seed=0, burn=None):
    """Direct simulation of Q' = A Q + B, P parallel chains, T recorded steps after burn-in."""
    rng = np.random.default_rng([4242, n, seed])
    burn = 40 * n if burn is None else burn
    Q = np.full(P, 0.58 * n)
    out = np.empty((T, P))
    for t in range(burn + T):
        M = rng.binomial(n, 0.5, P)
        U = rng.chisquare(np.maximum(M, 1)) * (M > 0)
        V = rng.chisquare(np.maximum(n - M, 1)) * (M < n)
        Q = (2 * U / n) * Q + (U + V) / n
        if t >= burn:
            out[t - burn] = Q
    return out
