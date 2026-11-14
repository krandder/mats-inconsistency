"""Activation family for the tractable-gates lane: definitions, exact Hermite spectra, Gaussian moments, gains.

Hermite convention: h_k(x) = He_k(x)/sqrt(k!) (orthonormal under N(0,1)); c_k(phi; s) = E[phi(s Z) h_k(Z)], so
E[phi(sX) phi(sY)] = sum_k c_k^2 t^k for standard (X, Y) with correlation t (Mehler).

Exact spectra (derivations in notes/theory.md section 2):
- relu:    c_0 = s/sqrt(2 pi), c_1 = s/2, even k >= 2: s (-1)^{k/2-1} (k-3)!! / sqrt(2 pi k!), odd k >= 3: 0.
- gsrelu (ReLU smoothed by Gaussian noise of std tau, f(z) = E relu(z + tau xi)): c_k = r^k S' c_k^relu(1),
  S' = sqrt(s^2 + tau^2), r = s / S'.
- gelu = gsrelu(tau=1) - npdf: the Gaussian bump npdf(a x) has c_k = (-a^2/(1+a^2))^{k/2} (k-1)!! / sqrt(2 pi (1+a^2) k!)
  for even k.
- Phi(a x): odd k: (-1)^{(k-1)/2} (k-2)!! a^k (1+a^2)^{-k/2} / sqrt(2 pi k!); c_0 = 1/2.
- softplus_beta, silu, tanh via the normal scale-mixture representation of the logistic law (Andrews-Mallows 1974,
  Stefanski 1991): L =d sigma * xi with sigma = 2K, K Kolmogorov-distributed, independent of xi ~ N(0,1). Then
  softplus_beta(z) = E relu(z + L/beta) = E_sigma gsrelu_{sigma/beta}(z); silu(z) = z P(L < z) = E_sigma sigma gelu(z/sigma);
  tanh(z) = 2 P(L < 2z) - 1 = E_sigma [2 Phi(2z/sigma) - 1]. The sigma-integral uses a fixed quadrature (checked against
  direct Gauss-Hermite in tests/test_acts.py).
- anything else: Gauss-Hermite quadrature (valid for k <= ~300 in float64).
"""
import math
import numpy as np
from scipy import special as sp

SQ2PI = math.sqrt(2 * math.pi)


# ----------------------------------------------------------------------------------------------- definitions
def npdf(x):
    return np.exp(-0.5 * x * x) / SQ2PI


def ncdf(x):
    return 0.5 * sp.erfc(-x / math.sqrt(2))


def _smoothstep(u):
    """C-infinity step: 0 for u <= -1, 1 for u >= 1, f(u+1)/(f(u+1)+f(1-u)) with f(x) = exp(-1/x) (x > 0)."""
    u = np.asarray(u, dtype=np.float64)
    a = np.clip(u + 1, 0, None); b = np.clip(1 - u, 0, None)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        fa = np.where(a > 0, np.exp(-1 / np.where(a > 0, a, 1)), 0.0)
        fb = np.where(b > 0, np.exp(-1 / np.where(b > 0, b, 1)), 0.0)
        out = fa / (fa + fb)
    return np.where(u >= 1, 1.0, np.where(u <= -1, 0.0, out))


def _dsmoothstep(u, h=1e-5):
    return (_smoothstep(u + h) - _smoothstep(u - h)) / (2 * h)


def act(name, p=None):
    """Return (phi, dphi) for activation `name` with parameter p (numpy, float64)."""
    if name == 'relu':
        return (lambda z: np.maximum(z, 0)), (lambda z: (z > 0).astype(np.float64))
    if name == 'softplus':        # p = beta
        b = float(p)
        return (lambda z: np.logaddexp(0, b * z) / b), (lambda z: sp.expit(b * z))
    if name == 'gelu':
        return (lambda z: z * ncdf(z)), (lambda z: ncdf(z) + z * npdf(z))
    if name == 'silu':
        return (lambda z: z * sp.expit(z)), (lambda z: sp.expit(z) * (1 + z * (1 - sp.expit(z))))
    if name == 'tanh':
        return np.tanh, (lambda z: 1 - np.tanh(z) ** 2)
    if name == 'gsrelu':          # p = tau (Gaussian smoothing width)
        t = float(p)
        return (lambda z: t * ((z / t) * ncdf(z / t) + npdf(z / t))), (lambda z: ncdf(z / t))
    if name == 'smoothstep':      # p = delta (margin): z * S(z/delta), exact zeros below -delta
        d = float(p)
        return (lambda z: z * _smoothstep(z / d)), (lambda z: _smoothstep(z / d) + (z / d) * _dsmoothstep(z / d))
    if name == 'relu2':
        return (lambda z: np.maximum(z, 0) ** 2), (lambda z: 2 * np.maximum(z, 0))
    if name == 'linear':
        return (lambda z: z), (lambda z: np.ones_like(z))
    raise ValueError(name)


def label(name, p=None):
    return name if p is None else f'{name}{p:g}'


# ----------------------------------------------------------------------------------------------- Hermite spectra
def _lfact(k):
    return sp.gammaln(np.asarray(k, dtype=np.float64) + 1)


def _ldfact(m):
    """log of m!! for integer m >= -1 (array)."""
    m = np.asarray(m, dtype=np.float64)
    out = np.zeros_like(m)
    ev = (m > 0) & (np.mod(m, 2) == 0)
    od = (m > 0) & (np.mod(m, 2) == 1)
    out[ev] = (m[ev] / 2) * math.log(2) + sp.gammaln(m[ev] / 2 + 1)
    out[od] = sp.gammaln(m[od] + 1) - ((m[od] - 1) / 2) * math.log(2) - sp.gammaln((m[od] - 1) / 2 + 1)
    return out


def relu_unit(K):
    """c_k of relu(x), k = 0..K (unit scale)."""
    k = np.arange(K + 1)
    c = np.zeros(K + 1)
    c[0] = 1 / SQ2PI
    if K >= 1:
        c[1] = 0.5
    ev = k[(k >= 2) & (k % 2 == 0)]
    sgn = np.where(((ev - 2) // 2) % 2 == 0, 1.0, -1.0)
    c[ev] = sgn * np.exp(_ldfact(ev - 3) - 0.5 * _lfact(ev)) / SQ2PI
    return c


def gsrelu_coef(K, s, tau):
    Sp = math.sqrt(s * s + tau * tau); r = s / Sp
    k = np.arange(K + 1)
    return relu_unit(K) * Sp * np.exp(k * math.log(r)) if r > 0 else np.r_[relu_unit(0) * Sp, np.zeros(K)]


def bump_coef(K, a):
    """c_k of npdf(a x)."""
    k = np.arange(K + 1)
    c = np.zeros(K + 1)
    ev = k[k % 2 == 0]
    q = a * a / (1 + a * a)
    sgn = np.where((ev // 2) % 2 == 0, 1.0, -1.0)
    with np.errstate(divide='ignore'):
        lq = math.log(q) if q > 0 else -np.inf
    lc = (ev / 2) * lq + _ldfact(ev - 1) - 0.5 * _lfact(ev) - 0.5 * math.log(1 + a * a)
    c[ev] = sgn * np.exp(lc) / SQ2PI
    if q == 0:
        c[0] = 1 / SQ2PI
    return c


def gelu_coef(K, s):
    return gsrelu_coef(K, s, 1.0) - bump_coef(K, s)


def ncdf_coef(K, a):
    """c_k of Phi(a x)."""
    k = np.arange(K + 1)
    c = np.zeros(K + 1)
    c[0] = 0.5
    od = k[k % 2 == 1]
    sgn = np.where(((od - 1) // 2) % 2 == 0, 1.0, -1.0)
    lc = _ldfact(od - 2) + od * math.log(abs(a)) - (od / 2) * math.log(1 + a * a) - 0.5 * _lfact(od)
    c[od] = sgn * np.exp(lc) / SQ2PI * (1 if a > 0 else -1) ** od
    return c


# --- Kolmogorov scale mixture for the logistic law: sigma = 2K -------------------------------------------------
def kolmogorov_pdf(x):
    """Density of the Kolmogorov distribution (theta-function form for small x, alternating series otherwise)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    sm = (x > 0) & (x < 1.0)
    lg = x >= 1.0
    xs = x[sm]
    if xs.size:
        j = np.arange(1, 40)[:, None]
        a = (2 * j - 1) ** 2 * math.pi ** 2 / 8
        # F(x) = sqrt(2 pi)/x sum exp(-a/x^2); f = dF/dx
        e = np.exp(-a / xs[None] ** 2)
        out[sm] = math.sqrt(2 * math.pi) * np.sum(e * (2 * a / xs[None] ** 4 - 1 / xs[None] ** 2), 0)
    xl = x[lg]
    if xl.size:
        j = np.arange(1, 60)[:, None]
        out[lg] = 8 * np.sum(((-1.0) ** (j - 1)) * j * j * xl[None] * np.exp(-2 * j * j * xl[None] ** 2), 0)
    return out


_KQ = None


def kol_quad(m=4000):
    """Nodes/weights for E_sigma g(sigma), sigma = 2K: trapezoid on a log grid (smooth, fast-decaying integrand)."""
    global _KQ
    if _KQ is None or len(_KQ[0]) != m:
        u = np.linspace(math.log(0.05), math.log(4.0), m)
        x = np.exp(u)
        w = kolmogorov_pdf(x) * x * (u[1] - u[0])
        w[0] *= 0.5; w[-1] *= 0.5
        w /= w.sum()
        _KQ = (2 * x, w)
    return _KQ


def softplus_coef(K, s, beta):
    sig, w = kol_quad()
    tau = sig / beta
    Sp = np.sqrt(s * s + tau * tau); r = s / Sp
    k = np.arange(K + 1)[:, None]
    return relu_unit(K) * np.sum(w[None] * Sp[None] * np.exp(k * np.log(r)[None]), 1)


def silu_coef(K, s):
    sig, w = kol_quad()
    acc = np.zeros(K + 1)
    for si, wi in zip(sig, w):
        acc += wi * si * gelu_coef(K, s / si)
    return acc


def tanh_coef(K, s):
    sig, w = kol_quad()
    acc = np.zeros(K + 1)
    for si, wi in zip(sig, w):
        acc += wi * 2 * ncdf_coef(K, 2 * s / si)
    acc[0] = 0.0
    return acc


def gh_nodes(Q):
    x, w = sp.roots_hermitenorm(Q)
    return x, w / w.sum()


def quad_coef(phi, K, s, Q=None, m=0.0):
    """c_k of x -> phi(m + s x) by Gauss-Hermite with Q nodes (recurrence on sqrt-weighted normalised Hermite)."""
    Q = Q or max(2 * K + 40, 200)
    x, w = gh_nodes(Q)
    fw = phi(m + s * x) * w
    out = np.empty(K + 1)
    h0 = np.ones_like(x); h1 = x.copy()
    out[0] = fw @ h0
    if K >= 1:
        out[1] = fw @ h1
    for k in range(1, K):
        h2 = (x * h1 - math.sqrt(k) * h0) / math.sqrt(k + 1)
        out[k + 1] = fw @ h2
        h0, h1 = h1, h2
    return out


def coef(name, p, K, s):
    """Exact (or quadrature) Hermite coefficients c_0..c_K of x -> phi(s x)."""
    if name == 'relu':
        return relu_unit(K) * s
    if name == 'gsrelu':
        return gsrelu_coef(K, s, p)
    if name == 'gelu':
        return gelu_coef(K, s)
    if name == 'softplus':
        return softplus_coef(K, s, p)
    if name == 'silu':
        return silu_coef(K, s)
    if name == 'tanh':
        return tanh_coef(K, s)
    phi, _ = act(name, p)
    return fine_coef(phi, K, s)


def moments(name, p, s):
    """(E phi(sZ), E phi(sZ)^2): exact for relu/relu2, Gauss-Hermite (600 nodes) for the analytic arms, fine trapezoid
    for non-analytic ones (smoothstep: Gauss-Hermite converges slowly on its exp(-1/x) transition)."""
    if name == 'relu':
        return s / SQ2PI, s * s / 2
    if name == 'relu2':
        return s * s / 2, 1.5 * s ** 4
    phi, _ = act(name, p)
    if name in ('smoothstep',):
        h = 0.002; x = np.arange(-45, 45 + h / 2, h); w = npdf(x) * h
        return float(w @ phi(s * x)), float(w @ phi(s * x) ** 2)
    x, w = gh_nodes(600)
    return float(w @ phi(s * x)), float(w @ phi(s * x) ** 2)


# ----------------------------------------------------------------------------------------------- Gaussian moments
def gmom(name, p, m, v, Q=96):
    """E phi(z), E phi(z)^2, E phi'(z) for z ~ N(m, v) (arrays m, v), by Gauss-Hermite (Q nodes)."""
    phi, dphi = act(name, p)
    x, w = gh_nodes(Q)
    s = np.sqrt(np.maximum(v, 0))
    Z = m[..., None] + s[..., None] * x
    F = phi(Z)
    return F @ w, (F * F) @ w, dphi(Z) @ w


def gain_for_q(name, p, q0=2.0, bias_var=0.0):
    """g^2 such that the infinite-width pre-activation second moment is a fixed point at q0:
    q0 = g^2 E phi(sqrt(q0) Z)^2 + bias_var. Also returns chi_1 = g^2 E phi'(sqrt(q0) Z)^2."""
    phi, dphi = act(name, p)
    x, w = gh_nodes(200)
    z = math.sqrt(q0) * x
    e2 = float(w @ phi(z) ** 2)
    g2 = (q0 - bias_var) / e2
    chi = g2 * float(w @ dphi(z) ** 2)
    return g2, chi


def hermite_tail(c):
    """Relative nonlinear tail: T_D / sum_{k>=2} c_k^2 for D = 1.. (index D)."""
    e = c ** 2
    tot = e[2:].sum()
    t = np.concatenate([[e[1:].sum()], e[1:].sum() - np.cumsum(e[1:])])  # t[D] = sum_{k>D} e_k
    return t / tot if tot > 0 else t


def fine_coef(phi, K, s, L=45.0, h=None):
    """c_k of x -> phi(s x), k = 0..K, by the trapezoid rule on [-L, L] with the stable Hermite-function recurrence
    (psi_k = h_k sqrt(npdf), bounded by ~1): spectrally accurate for smooth phi, valid for large K (no underflow)."""
    h = h or min(0.01, 0.5 / math.sqrt(max(K, 1)))
    x = np.arange(-L, L + h / 2, h)
    g = phi(s * x) * np.sqrt(npdf(x)) * h
    out = np.empty(K + 1)
    p0 = np.sqrt(npdf(x)); p1 = x * p0
    out[0] = g @ p0
    if K >= 1:
        out[1] = g @ p1
    for k in range(1, K):
        p2 = (x * p1 - math.sqrt(k) * p0) / math.sqrt(k + 1)
        out[k + 1] = g @ p2
        p0, p1 = p1, p2
    return out
