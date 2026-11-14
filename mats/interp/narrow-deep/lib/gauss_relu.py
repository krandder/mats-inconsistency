# Copied unchanged from work/arc-noisy/gauss_relu.py (2026-10-29) for the narrow-deep lane.
"""Exact Gaussian moments of ReLU outputs (no Edgeworth or Mehler truncation).

For z ~ N(m, C): E[ReLU(z_i)] = s_i G(a_i), G(a) = a Phi(a) + phi(a), a_i = m_i / s_i;
E[ReLU(z_i) ReLU(z_j)] = s_i s_j F(a_i, a_j, rho_ij) with
    F(a, b, rho) = (ab + rho) Phi2(a, b; rho) + a phi(b) Phi(A) + b phi(a) Phi(B) + r phi(a) phi(B),
    r = sqrt(1 - rho^2), A = (a - rho b) / r, B = (b - rho a) / r,
derived by Gaussian integration by parts (checked against Monte Carlo in test_gauss_relu.py).
Phi2(a, b; rho) = P(U < a, V < b) = bvnu(-a, -b, rho), Genz's BVND algorithm (20-point Gauss-Legendre in both
branches), vectorised in torch float64.
FLOPs (flopscope weights): bvnu ~ 40 nodes x (sin or sqrt/exp: 2 x 16 + 10) + 3 cdf + overhead ~ 2,000 per pair;
F ~ 2,000 + 3 x 96 + 3 x 54 + 30 ~ 2,500 per pair.
"""
import math
import numpy as np
import torch

DT = torch.float64
_X, _W = np.polynomial.legendre.leggauss(20)
GLX = torch.tensor(_X, dtype=DT); GLW = torch.tensor(_W, dtype=DT)
SQ2PI = math.sqrt(2 * math.pi)
FLOPS_PAIR = 2500
FLOPS_NEURON = 2 * 96 + 2 * 54 + 20


def ncdf(x):
    return torch.special.ndtr(x)


def npdf(x):
    return torch.exp(-0.5 * x * x) / SQ2PI


def bvnu(h, k, r):
    """P(X > h, Y > k), (X, Y) standard bivariate normal with correlation r (tensors of one shape)."""
    h, k, r = torch.broadcast_tensors(h, k, r)
    out = torch.empty_like(h)
    lo = r.abs() < 0.925
    if lo.any():
        hh, kk, rr = h[lo], k[lo], r[lo]
        hk = hh * kk; hs = (hh * hh + kk * kk) / 2; asr = torch.asin(rr)
        sn = torch.sin(asr[..., None] * (GLX + 1) / 2)                       # (m, 20)
        val = (GLW * torch.exp((sn * hk[..., None] - hs[..., None]) / (1 - sn * sn))).sum(-1)
        out[lo] = val * asr / (4 * math.pi) + ncdf(-hh) * ncdf(-kk)
    hi = ~lo
    if hi.any():
        hh, kk, rr = h[hi].clone(), k[hi].clone(), r[hi]
        neg = rr < 0
        kk = torch.where(neg, -kk, kk)
        hk = hh * kk
        bvn = torch.zeros_like(hh)
        one = rr.abs() < 1
        as_ = ((1 - rr) * (1 + rr)).clamp_min(0)
        a = as_.sqrt(); bs = (hh - kk) ** 2
        c = (4 - hk) / 8; d = (12 - hk) / 16
        asr = -(bs / as_.clamp_min(1e-300) + hk) / 2
        t1 = torch.where(asr > -100, a * torch.exp(asr.clamp_min(-100)) *
                         (1 - c * (bs - as_) * (1 - d * bs / 5) / 3 + c * d * as_ * as_ / 5), torch.zeros_like(a))
        b = bs.sqrt()
        t2 = torch.where(-hk < 100, torch.exp(-hk.clamp_max(100) / 2) * SQ2PI * ncdf(-b / a.clamp_min(1e-300)) * b *
                         (1 - c * bs * (1 - d * bs / 5) / 3), torch.zeros_like(a))
        bvn = t1 - t2
        a2 = a / 2
        xs = (a2[..., None] * (GLX + 1)) ** 2                                  # (m, 20)
        rs = (1 - xs).clamp_min(0).sqrt()
        asr2 = -(bs[..., None] / xs.clamp_min(1e-300) + hk[..., None]) / 2
        term = torch.where(asr2 > -100,
                           a2[..., None] * GLW * torch.exp(asr2.clamp_min(-100)) *
                           (torch.exp(-hk[..., None] * (1 - rs) / (2 * (1 + rs))) / rs.clamp_min(1e-300)
                            - (1 + c[..., None] * xs * (1 + d[..., None] * xs))),
                           torch.zeros_like(xs))
        bvn = bvn + term.sum(-1)
        bvn = torch.where(one, -bvn / (2 * math.pi), torch.zeros_like(bvn))
        pos = torch.where(~neg, bvn + ncdf(-torch.maximum(hh, kk)), torch.zeros_like(bvn))
        negv = -bvn + torch.where(kk > hh, ncdf(kk) - ncdf(hh), torch.zeros_like(bvn))
        out[hi] = torch.where(neg, negv, pos).clamp(0, 1)
    return out


def relu_mean(m, s):
    a = m / s
    return s * (a * ncdf(a) + npdf(a))


def relu_second(m, s):
    """E[ReLU(z)^2] for z ~ N(m, s^2)."""
    a = m / s
    return s * s * ((a * a + 1) * ncdf(a) + a * npdf(a))


def F_pair(a, b, rho):
    r = (1 - rho * rho).clamp_min(1e-30).sqrt()
    A = (a - rho * b) / r; B = (b - rho * a) / r
    P2 = bvnu(-a, -b, rho)
    return (a * b + rho) * P2 + a * npdf(b) * ncdf(A) + b * npdf(a) * ncdf(B) + r * npdf(a) * npdf(B)


def relu_moments(m, C, det=8.0):
    """Exact mean and covariance of ReLU(z), z ~ N(m, C). Pairs where both neurons are deterministic
    (|a| > det: always on or always off) use the closed forms (Cov = C_ij if both on, else 0)."""
    s = C.diagonal().clamp_min(1e-300).sqrt()
    a = m / s
    mu = s * (a * ncdf(a) + npdf(a))
    on = a > det; off = a < -det; kink = ~(on | off)
    n = m.shape[0]
    E2 = torch.zeros_like(C)
    # pairs with at least one kink neuron and no off neuron
    ik = torch.nonzero(kink).flatten(); io = torch.nonzero(on).flatten()
    rho = (C / (s[:, None] * s[None, :])).clamp(-1, 1)
    if ik.numel():
        cols = torch.nonzero(~off).flatten()
        aa = a[ik][:, None].expand(-1, cols.numel()); bb = a[cols][None, :].expand(ik.numel(), -1)
        rr = rho[ik][:, cols]
        Fv = F_pair(aa.reshape(-1), bb.reshape(-1), rr.reshape(-1)).reshape(ik.numel(), cols.numel())
        blk = s[ik][:, None] * s[cols][None, :] * Fv
        E2[ik[:, None], cols[None, :]] = blk
        E2[cols[:, None], ik[None, :]] = blk.T
    if io.numel():                                   # both always on: E[z_i z_j]
        E2[io[:, None], io[None, :]] = C[io][:, io] + m[io][:, None] * m[io][None, :]
    d2 = relu_second(m, s)
    E2.diagonal().copy_(d2)
    cov = E2 - mu[:, None] * mu[None, :]
    n_pairs = int(kink.sum()) * int((~off).sum())
    return mu, cov, n_pairs
