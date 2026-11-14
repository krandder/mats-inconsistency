"""Gaussian closure (K2: mean + full covariance) for plain and residual nets with a general activation, with FLOPs.

Plain layer: z = W h, (z ~ N(W mu, W C W^T) under the closure), h' = phi(z).
  Diagonal moments E phi(z_i), E phi(z_i)^2 by Gauss-Hermite (Q nodes); off-diagonal E phi(z_i) phi(z_j) by Mehler's
  series sum_k a_ik a_jk rho_ij^k, a_ik = E[phi(m_i + s_i X) h_k(X)] (order <= Kmeh, from the same nodes).
  ReLU optionally in closed form (narrow-deep's gauss_relu, imported read-only).
Residual layer (narrow-deep's family, trained sorting nets): u = W h, b = phi(u), h' = h + V b; joint Gaussian closure
  of (h, u): Cov(h, b) = C W^T diag(E phi'(u)) (Stein's lemma, exact for jointly Gaussian (h, u)).
FLOPs (program conventions): matmul M N (2K - 1); one activation evaluation EVAL[name]; elementwise 1; sqrt 16.
MC cost of one sample per plain layer: n(2n - 1) + n EVAL; per residual layer: 2 n(2n - 1) + n EVAL + n.
"""
import math, os, sys
import numpy as np
import acts as A

EVAL = dict(relu=1, softplus=34, gelu=20, silu=20, tanh=16, gsrelu=154, smoothstep=40, relu2=2, linear=0)
_GR = None


def _gr():
    global _GR
    if _GR is None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'narrow-deep', 'lib'))
        import gauss_relu as GR
        _GR = GR
    return _GR


def q_for(name, p):
    """Gauss-Hermite nodes per arm, chosen for ~1e-4 relative accuracy of the 1-D moments (near-kinked gates need more:
    measured on softplus_beta at beta = 8, 16; logs/test_k2.log)."""
    if name == 'softplus':
        return 64 if p <= 2 else 128 if p <= 4 else 256 if p <= 8 else 512
    if name == 'smoothstep':
        return 200
    return 64


def mm(M, K, N):
    return M * N * (2 * K - 1)


class Hermite:
    def __init__(self, Q, K):
        x, w = A.gh_nodes(Q)
        H = np.empty((K + 1, Q)); H[0] = 1.0
        if K >= 1:
            H[1] = x
        for k in range(1, K):
            H[k + 1] = (x * H[k] - math.sqrt(k) * H[k - 1]) / math.sqrt(k + 1)
        self.x, self.w, self.H, self.Q, self.K = x, w, H, Q, K


def act_moments(name, p, m, Cz, herm, closed_relu=True):
    """Gaussian closure through phi: returns (mean, Cov, E phi', flops) for z ~ N(m, Cz)."""
    n = len(m)
    phi, dphi = A.act(name, p)
    ev = EVAL[name]
    s = np.sqrt(np.maximum(np.diag(Cz), 1e-300))
    if name == 'relu' and closed_relu:
        import torch
        GR = _gr()
        mu_t, C_t, npairs = GR.relu_moments(torch.tensor(m, dtype=torch.float64), torch.tensor(Cz, dtype=torch.float64))
        dmean = A.ncdf(m / s)
        return mu_t.numpy(), C_t.numpy(), dmean, npairs * GR.FLOPS_PAIR + n * GR.FLOPS_NEURON + n * 112
    X = m[:, None] + s[:, None] * herm.x[None, :]                       # n x Q
    F = phi(X); D = dphi(X)
    fl = 2 * n * herm.Q + n * herm.Q * (ev + ev)                         # nodes, phi and phi'
    a = (F * herm.w[None, :]) @ herm.H.T                                  # n x (K+1)
    fl += n * herm.Q + mm(n, herm.Q, herm.K + 1)
    mean = a[:, 0]
    e2 = (F * F) @ herm.w; dmean = D @ herm.w
    fl += 4 * n * herm.Q
    rho = Cz / np.outer(s, s); np.fill_diagonal(rho, 0.0)
    fl += n * n + 16 * n
    acc = np.zeros((n, n)); P = np.ones((n, n))
    for k in range(herm.K + 1):
        acc += np.outer(a[:, k], a[:, k]) * P
        P *= rho
    fl += 4 * n * n * (herm.K + 1)
    C = acc - np.outer(mean, mean)
    np.fill_diagonal(C, np.maximum(e2 - mean * mean, 0.0))
    fl += 3 * n * n
    return mean, C, dmean, fl


def k2_plain(name, p, Ws, herm, closed_relu=True, record=None, bs=None):
    """Ws: list of (W,) float arrays; bs: biases or None. Input x ~ N(0, I). Returns list of (mean, Cov) after each layer
    and the cumulative flops list."""
    n = Ws[0][0].shape[0]
    mu = np.zeros(n); C = np.eye(n)
    outs, fls = [], []
    tot = 0
    for l, (W,) in enumerate(Ws):
        W = W.astype(np.float64)
        m = W @ mu + (0 if bs is None else bs[l].astype(np.float64)); Cz = W @ C @ W.T
        tot += mm(n, n, 1) + 2 * mm(n, n, n)
        mu, C, _, f = act_moments(name, p, m, Cz, herm, closed_relu)
        tot += f
        outs.append((mu.copy(), C.copy())); fls.append(tot)
    return outs, fls


def k2_residual(name, p, Ws, Vs, herm, closed_relu=True, bs=None):
    """h_l = h_{l-1} + V_l phi(W_l h_{l-1} + b_l) (trained sorting nets: V already includes any 1/sqrt(L)). x ~ N(0, I).
    W_l may be m x n and V_l n x m (hidden width m)."""
    n = Ws[0].shape[1]
    mu = np.zeros(n); C = np.eye(n)
    tot = 0
    for l, (W, V) in enumerate(zip(Ws, Vs)):
        W = W.astype(np.float64); V = V.astype(np.float64)
        m = W @ mu + (0 if bs is None else bs[l]); CW = C @ W.T; Cz = W @ CW
        tot += mm(n, n, 1) + 2 * mm(n, n, n)
        mb, Cb, dmean, f = act_moments(name, p, m, Cz, herm, closed_relu)
        tot += f
        Chb = CW * dmean[None, :]
        X = Chb @ V.T
        mu = mu + V @ mb
        C = C + X + X.T + V @ Cb @ V.T
        tot += n * n + mm(n, n, n) + 3 * n * n + mm(n, n, 1) + 2 * mm(n, n, n) + 2 * n
    return mu, C, tot


def mc_flops_plain(name, n, L, bias=False):
    return L * (mm(1, n, n) + n * EVAL[name] + (n if bias else 0))


def mc_flops_res(name, n, L):
    return L * (2 * mm(1, n, n) + n * EVAL[name] + n)
