"""Backward-only arm (BO): meeting point at the input. E[a_L] ~ E_t[V_0(U t)], t ~ N(0, I_k), with U (8 x k) the
relevant input directions. Radius exact by positive homogeneity: E[chi_k] for the noiseless part, a radial Gauss rule
over chi_k for the noise correction; directions by a product midpoint rule on S^(k-1) in hyperspherical angles. Each node
is evaluated by a deterministic noiseless pass to c_d and the deep part (bvf.Deep). No random numbers."""
import math
import numpy as np
import torch
from scipy.special import roots_genlaguerre, gammaln
import bvf

DT = torch.float64


def chi_mean(k):
    return math.sqrt(2.0) * math.exp(gammaln((k + 1) / 2) - gammaln(k / 2))


def sphere_rule(k, m):
    """nodes (N, k) on S^(k-1), weights (N,) summing to 1: midpoint rule with m points per polar angle in [0, pi] and 2m
    points for the azimuth in [0, 2 pi)."""
    if k == 1:
        return torch.tensor([[1.0], [-1.0]], dtype=DT), torch.tensor([0.5, 0.5], dtype=DT)
    az = (torch.arange(2 * m, dtype=DT) + 0.5) * (2 * math.pi / (2 * m))
    if k == 2:
        nodes = torch.stack([torch.cos(az), torch.sin(az)], 1)
        return nodes, torch.full((2 * m,), 1.0 / (2 * m), dtype=DT)
    pol = (torch.arange(m, dtype=DT) + 0.5) * (math.pi / m)
    grids = torch.meshgrid(*([pol] * (k - 2) + [az]), indexing='ij')
    ang = torch.stack([gg.reshape(-1) for gg in grids], 1)          # (N, k-1)
    N = ang.shape[0]
    x = torch.zeros(N, k, dtype=DT)
    s = torch.ones(N, dtype=DT)
    w = torch.ones(N, dtype=DT)
    for j in range(k - 1):
        if j < k - 2:
            x[:, j] = s * torch.cos(ang[:, j])
            w = w * torch.sin(ang[:, j]) ** (k - 2 - j)
            s = s * torch.sin(ang[:, j])
        else:
            x[:, j] = s * torch.cos(ang[:, j])
            x[:, j + 1] = s * torch.sin(ang[:, j])
    return x, w / w.sum()


def radial_rule(k, q=8):
    """nodes rho_i and weights for E[g(rho)], rho ~ chi_k: rho^2/2 ~ Gamma(k/2) -> generalized Gauss-Laguerre."""
    x, w = roots_genlaguerre(q, k / 2 - 1)
    w = w / w.sum()
    return torch.tensor(np.sqrt(2 * x), dtype=DT), torch.tensor(w, dtype=DT)


def input_directions(net, c_d, deep, probes, fl):
    """sample-free sensitivity at the input: M = mean over probe inputs of J^T J (J = J_ref . J_{0->c_d} along the
    noiseless path), returns eigvecs (descending) and eigvals."""
    n = net.n
    M = torch.zeros(n, n, dtype=DT)
    for h in probes:
        _, Jm = bvf.jac_pass(net, h, 0, c_d, fl)
        Jt = deep.J @ Jm
        g = Jt.T @ Jt
        M += g / g.trace().clamp_min(1e-300) * (Jt.norm() ** 0)  # per-probe normalised (scale-free), then averaged
        fl.add(bvf.mmf(n, n, n) * 2 + n * n)
    ev, V = torch.linalg.eigh(0.5 * (M + M.T))
    return V.flip(1), ev.flip(0)


def bo_estimate(net, c_d, deep, U, m, fl, q=8):
    """E[a_L] ~ E_t[V_0(U t)], t ~ N(0, I_k)."""
    k = U.shape[1]
    nodes, w = sphere_rule(k, m)
    X = nodes @ U.T                                                   # unit-norm input points
    fl.add(len(nodes) * bvf.mmf(1, k, net.n))
    Q = bvf.noiseless_pass(net, X, 0, c_d, fl)                        # states at c_d for unit input radius
    lin = (Q @ deep.J.T)                                              # noiseless output per unit radius
    fl.add(len(nodes) * bvf.mmf(1, net.n, net.n))
    est = chi_mean(k) * (w[:, None] * lin).sum(0)
    if not deep.noise_table:
        return est, len(nodes)
    # noise correction: E_rho[corr(rho * (r.q)/rv)] with a radial rule over chi_k
    rr, rw = radial_rule(k, q)
    base = (Q @ deep.r) / deep.rv                                     # radial coordinate per unit input radius
    corr = torch.zeros(net.n, dtype=DT)
    for rho, wr in zip(rr.tolist(), rw.tolist()):
        H = rho * Q
        Vfull = deep.value(H, fl)                                     # J H + corr
        corr += wr * (w[:, None] * (Vfull - H @ deep.J.T)).sum(0)
    return est + corr, len(nodes)
