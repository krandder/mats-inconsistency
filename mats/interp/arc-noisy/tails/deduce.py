"""Tail readouts from cumulants (deterministic given the cumulants).

edgeworth(t, k): Phi-bar(x) + phi(x) [k3/6 He2(x) + k4/24 He3(x) + k3^2/72 He5(x)] (standardised cumulants);
saddle(t, k):   Lugannani-Rice with the truncated cumulant generating function K(th) = sum_j k_j th^j / j!
                (j <= 3 or 4); the saddlepoint is the smallest th > 0 with K'(th) = t and K''(th) > 0. Returns nan if
                there is none (a truncated CGF need not be a CGF);
radial(t, mom, n, order): noiseless homogeneity z = |x| g(u), |x| ~ chi_n independent of u: the moments of g are
                E[z^k] / E[|x|^k]; the tail is the chi_n mixture of saddle() with the conditional cumulants r^k k_k(g).
"""
import math
import numpy as np
from scipy.special import log_ndtr, gammaln
from scipy.stats import norm, chi


def cumulants_from_moments(m):
    """m = (E z, E z^2, E z^3, E z^4) -> (k1, k2, k3, k4)."""
    m1, m2, m3, m4 = m
    k1 = m1
    k2 = m2 - m1 ** 2
    k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
    k4 = m4 - 4 * m3 * m1 - 3 * m2 ** 2 + 12 * m2 * m1 ** 2 - 6 * m1 ** 4
    return k1, k2, k3, k4


def gauss(t, k):
    return float(np.exp(log_ndtr(-(t - k[0]) / math.sqrt(k[1]))))


def edgeworth(t, k, order=4):
    s = math.sqrt(k[1]); x = (t - k[0]) / s
    g3 = k[2] / s ** 3 if len(k) > 2 else 0.0
    g4 = k[3] / s ** 4 if len(k) > 3 and order >= 4 else 0.0
    He2 = x * x - 1; He3 = x ** 3 - 3 * x; He5 = x ** 5 - 10 * x ** 3 + 15 * x
    corr = g3 / 6 * He2 + (g4 / 24 * He3 + g3 * g3 / 72 * He5 if order >= 4 else 0.0)
    return float(norm.sf(x) + norm.pdf(x) * corr)


def saddle(t, k, order=4):
    """Lugannani-Rice with the CGF truncated at `order`; the saddlepoint is the smallest real theta > 0 with
    K'(theta) = t and K''(theta) > 0 (PREREG-TAILS section 3), found exactly as a root of the cubic/quadratic K' - t.
    nan if there is none or if theta t - K(theta) <= 0."""
    k = list(k[:order]) + [0.0] * (4 - min(order, len(k)))
    k1, k2, k3, k4 = k
    Kpp = lambda th: k2 + k3 * th + k4 * th ** 2 / 2
    K = lambda th: k1 * th + k2 * th ** 2 / 2 + k3 * th ** 3 / 6 + k4 * th ** 4 / 24
    if t <= k1:
        return float('nan')
    coef = [k4 / 6, k3 / 2, k2, k1 - t]
    while coef and coef[0] == 0:
        coef = coef[1:]
    roots = np.roots(coef)
    cand = sorted(float(r.real) for r in roots if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real)) and r.real > 0
                  and Kpp(r.real) > 0)
    if not cand:
        return float('nan')
    th = cand[0]
    a = th * t - K(th)
    if a <= 0:
        return float('nan')
    w = math.sqrt(2 * a); u = th * math.sqrt(Kpp(th))
    return float(norm.sf(w) + norm.pdf(w) * (1 / u - 1 / w))


# analytic FLOP counts of the readouts (flopscope rates: sqrt/exp/log 16, cdf 96, pdf 54), per event
F_SADDLE = 400 + 2 * 16 + 96 + 54 + 40      # cubic roots (companion eigenvalues ~400), K, K'', sqrt x2, sf, pdf, arithmetic
F_GAUSS = 96 + 10
F_EDGE = 96 + 54 + 30
F_CUMULANTS = 30                              # moments -> cumulants, per output


def chi_moment(n, k):
    return math.exp(k / 2 * math.log(2) + gammaln((n + k) / 2) - gammaln(n / 2))


F_RADIAL_NODES = 161


def radial(t, mom, n=256, order=4, nodes=161):
    """chi_n mixture of the conditional saddlepoint; mom = raw moments of z (noiseless homogeneity assumed).
    Cost per event about nodes x (F_SADDLE + 10) FLOPs."""
    mg = [mom[j] / chi_moment(n, j + 1) for j in range(4)]
    kg = cumulants_from_moments(mg)
    r = np.linspace(chi.ppf(1e-9, n), chi.ppf(1 - 1e-12, n), nodes)
    w = chi.pdf(r, n); w = w / np.trapezoid(w, r)
    vals = []
    for ri in r:
        kc = [kg[j] * ri ** (j + 1) for j in range(4)]
        v = saddle(t, kc, order)
        vals.append(v if np.isfinite(v) else np.nan)
    vals = np.array(vals)
    if np.isnan(vals).any():
        return float('nan')
    return float(np.trapezoid(w * vals, r))


def nig_params(k):
    """NIG (alpha, beta, mu, delta) matching cumulants k1..k4, or None outside the NIG domain (exkurt > 4/3 skew^2 needed,
    with rho^2 = (s^2/x) / (3 - 4 s^2/x) < 1)."""
    k1, k2, k3, k4 = k
    if k2 <= 0:
        return None
    s = k3 / k2 ** 1.5; x = k4 / k2 ** 2
    if x <= 0:
        return None
    q = s * s / x
    if 3 - 4 * q <= 0:
        return None
    r2 = q / (3 - 4 * q)
    if r2 >= 1:
        return None
    rho = math.copysign(math.sqrt(r2), s) if s != 0 else 0.0
    A = 3 * (1 + 4 * r2) / x
    alpha = math.sqrt(A / k2) / (1 - r2)
    beta = rho * alpha
    gam = alpha * math.sqrt(1 - r2)
    delta = A / (alpha * math.sqrt(1 - r2))
    mu = k1 - delta * beta / gam
    return alpha, beta, mu, delta


def nig_tail(t, k):
    """P(X > t) for the NIG matched to k1..k4; nan outside the NIG domain."""
    from scipy.stats import norminvgauss
    p = nig_params(k)
    if p is None:
        return float('nan')
    alpha, beta, mu, delta = p
    return float(norminvgauss.sf(t, alpha * delta, beta * delta, loc=mu, scale=delta))


def radial_nig(t, mom, n=256, nodes=161):
    """chi_n mixture of the NIG matched to the cumulants of g = z / |x| (noiseless homogeneity)."""
    from scipy.stats import norminvgauss
    mg = [mom[j] / chi_moment(n, j + 1) for j in range(4)]
    p = nig_params(cumulants_from_moments(mg))
    if p is None:
        return float('nan')
    alpha, beta, mu, delta = p
    r = np.linspace(chi.ppf(1e-9, n), chi.ppf(1 - 1e-12, n), nodes)
    w = chi.pdf(r, n); w = w / np.trapezoid(w, r)
    vals = norminvgauss.sf(t / r, alpha * delta, beta * delta, loc=mu, scale=delta)
    return float(np.trapezoid(w * vals, r))
