"""Input-bits lane library: critical He-ReLU stacks with an input and absolute noise, float64 numpy.

Model (the stationary lane's Part C convention, `work/stationary/scratch/partC.py`):
  h_0 = x ~ N(0, I_n);  z_l = W_l h_{l-1};  h_l = relu(z_l) + sigma xi_l  (l = 1..L-1);  output a = relu(W_L h_{L-1})
  (no noise on the output). W_l = st.W_layer(seed, n, l), the stationary lane's weight stream, so a net is the same
  object whatever L a run uses (layers 1..L of seed s are shared by every depth).
sigma = 0 is ARC's noiseless family; there a(x) = |x| g(x/|x|) exactly (positive homogeneity, no biases).
Cost unit: one n x n matrix-vector product ("mv", 2 n^2 flops); one forward pass costs L mv.

Contents:
  Net                  weights, forward passes (optional noise, per-layer callback), unit-direction outputs
  mean_radius          m_n = E|x|
  var_split            exact three-term split V = V_rad + V_amp + V_dir of tr Var_x(a) (noiseless, unit inputs)
  gate_layer_stats     per-layer input gate statistics (marginal entropy sum, mean pairwise Hamming distance)
  saltelli             first-order and total Sobol indices of chosen input coordinates (trace over outputs)
  stein_shares         Hermite degree-1 and degree-2 shares of a function of Gaussian inputs (unbiased, two halves)
  gram_chain_inputs    exact annealed replica Gram chain started from given input Gram matrices (weights integrated out)
"""
import math
import os
import sys

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary/lib')
import st  # noqa: E402  (W_layer, the shared weight stream)


def mean_radius(n):
    """E|x| for x ~ N(0, I_n)."""
    return math.exp(0.5 * math.log(2.0) + gammaln((n + 1) / 2.0) - gammaln(n / 2.0))


def relu(z):
    return np.maximum(z, 0.0)


class Net:
    """Layers 1..L of the stationary lane's weight stream for (seed, n); noise sigma on hidden layers 1..L-1."""

    def __init__(self, seed, n, L, sigma=0.0):
        self.seed, self.n, self.L, self.sigma = int(seed), int(n), int(L), float(sigma)
        self.W = [st.W_layer(seed, n, l) for l in range(1, L + 1)]   # self.W[l-1] = W_l

    def forward(self, X, noise=None, cb=None, upto=None):
        """X (P, n) inputs at layer 0. noise: None (sigma = 0 or no noise), an rng (fresh N(0,1) per hidden layer), or a
        callable noise(l, shape) -> array. cb(l, z, h) after each layer l (h = post-noise state; at l = L, h = output).
        upto: stop after layer `upto` (returns h_upto, with noise if upto < L). Returns the output a (or h_upto)."""
        L = self.L if upto is None else upto
        h = X
        for l in range(1, L + 1):
            z = h @ self.W[l - 1].T
            h = np.maximum(z, 0.0)
            if l < self.L and self.sigma > 0 and noise is not None:
                xi = noise(l, h.shape) if callable(noise) else noise.standard_normal(h.shape)
                h = h + self.sigma * xi
            if cb is not None:
                cb(l, z, h)
        return h

    def g(self, U):
        """Noiseless output on unit directions U (P, n): a(x) = |x| g(x / |x|)."""
        h = U
        for W in self.W:
            h = np.maximum(h @ W.T, 0.0)
        return h


def sphere(G):
    return G / np.linalg.norm(G, axis=1, keepdims=True)


def ustat_mean_sq(V):
    """Unbiased estimate of |E v|^2 from rows of V (P, d): (|sum v|^2 - sum |v|^2) / (P (P - 1))."""
    P = V.shape[0]
    s = V.sum(0)
    return float((s @ s - np.sum(V * V)) / (P * (P - 1)))


def var_split(Gout, n):
    """Exact split of V = tr Var_x(a) for a noiseless net, from outputs Gout = g(U) on P iid uniform directions U.
    With A = |g(u)|, d = g/|g| and m_n = E|x|:
      V = (n - m_n^2) E A^2  +  m_n^2 Var_u A  +  m_n^2 [(E A)^2 - |E A d|^2]   (radius, amplitude, direction).
    Each term is estimated without bias (U-statistics). Returns dict with the three terms, V and shares."""
    mn = mean_radius(n)
    A = np.linalg.norm(Gout, axis=1)
    P = len(A)
    EA2 = float(np.mean(A * A))
    EA_sq = float((A.sum() ** 2 - np.sum(A * A)) / (P * (P - 1)))   # (E A)^2
    EAd_sq = ustat_mean_sq(Gout)                                     # |E A d|^2 = |E g|^2
    V_rad = (n - mn * mn) * EA2
    V_amp = mn * mn * (EA2 - EA_sq)
    V_dir = mn * mn * (EA_sq - EAd_sq)
    V = V_rad + V_amp + V_dir
    return dict(V=V, V_rad=V_rad, V_amp=V_amp, V_dir=V_dir, s_rad=V_rad / V, s_amp=V_amp / V, s_dir=V_dir / V,
                EA=float(A.mean()), EA2=EA2, logA_var=float(np.var(np.log(np.maximum(A, 1e-300)), ddof=1)))


def bits_entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def gate_layer_stats(Z):
    """Input gate statistics of one layer from pre-activations Z (P inputs, n units):
    H_marg = sum_i H(p_i) bits (upper bound on the joint gate entropy), Hamming = E #units whose gate differs between two
    independent inputs (unbiased: sum_i 2 p_i (1 - p_i) P/(P-1)), live = #units with 0 < p_i < 1."""
    G = Z > 0
    P = G.shape[0]
    p = G.mean(0)
    return dict(H_marg=float(bits_entropy(p).sum()), hamming=float(np.sum(2 * p * (1 - p)) * P / (P - 1)),
                live=int(np.sum((p > 0) & (p < 1))))


def saltelli(f, n, P, rng, coords, chunk=None):
    """First-order (Saltelli 2010) and total (Jansen) Sobol indices of input coordinates `coords` of f: R^n -> R^m with
    x ~ N(0, I_n); indices are traces over the m outputs, normalised by V = tr Var f.
      S_j V = E[f(B) . (f(A_B^j) - f(A))],  T_j V = E|f(A) - f(A_B^j)|^2 / 2,   A_B^j = A with column j from B.
    Returns dict(S, T, V, se_S, se_T) (per-coordinate arrays; se from the sample)."""
    A = rng.standard_normal((P, n))
    B = rng.standard_normal((P, n))
    fA, fB = f(A), f(B)
    allf = np.concatenate([fA, fB])
    V = float(np.sum(allf.var(0, ddof=1)))
    S, T, sS, sT = [], [], [], []
    for j in coords:
        AB = A.copy()
        AB[:, j] = B[:, j]
        fAB = f(AB)
        s_ = np.sum(fB * (fAB - fA), 1)
        t_ = 0.5 * np.sum((fA - fAB) ** 2, 1)
        S.append(s_.mean() / V); T.append(t_.mean() / V)
        sS.append(s_.std(ddof=1) / math.sqrt(P) / V); sT.append(t_.std(ddof=1) / math.sqrt(P) / V)
    return dict(S=np.array(S), T=np.array(T), se_S=np.array(sS), se_T=np.array(sT), V=V)


def stein_shares(X, F, deg2_outputs=16):
    """Hermite degree-1 and degree-2 variance shares of F(X), X ~ N(0, I_d) rows, F (P, m), via Stein's identities:
    degree 1: sum_c |E[F_c X]|^2;  degree 2: sum_c (1/2) |E[F_c (X X^T - I)]|_F^2 (first `deg2_outputs` outputs only,
    rescaled by their share of the variance). Unbiased: inner products of the two half-sample means. Returns dict with
    the two shares (of V = tr Var F), V, and the degree-2 subset size."""
    P, d = X.shape
    Fc = F - F.mean(0)
    V = float(np.sum(F.var(0, ddof=1)))
    h = P // 2
    g1 = Fc[:h].T @ X[:h] / h
    g2 = Fc[h:2 * h].T @ X[h:2 * h] / h
    lin = float(np.sum(g1 * g2))
    out = dict(V=V, s_lin=lin / V)
    if deg2_outputs and d <= 1024:
        J = min(deg2_outputs, F.shape[1])
        VJ = float(np.sum(F[:, :J].var(0, ddof=1)))
        acc = 0.0
        for c in range(J):
            H1 = (X[:h] * Fc[:h, c:c + 1]).T @ X[:h] / h - np.eye(d) * Fc[:h, c].mean()
            H2 = (X[h:2 * h] * Fc[h:2 * h, c:c + 1]).T @ X[h:2 * h] / h - np.eye(d) * Fc[h:2 * h, c].mean()
            acc += 0.5 * float(np.sum(H1 * H2))
        out.update(s_deg2=acc / VJ, deg2_J=J)
    return out


def kappa(c):
    c = np.clip(c, -1.0, 1.0)
    return (np.sqrt(np.maximum(1.0 - c * c, 0.0)) + (math.pi - np.arccos(c)) * c) / math.pi


def gram_chain_inputs(G0, L, n, rng, sigma=0.0, share_noise=False, record=None):
    """Exact annealed two-replica Gram chain (weights integrated out) for the input model.
    G0 (P, 2, 2): input Gram matrices x^(a).x^(b) / n. Layer l = 1..L: z_i ~ N(0, 2 G) iid over units i = 1..n,
    v_i = relu(z_i) + sigma xi_i (l < L only; xi shared between the replicas if share_noise), G' = sum_i v_i v_i^T / n.
    This is exact for replicas that share every weight matrix (a fresh W_l is rotation invariant on its input side).
    record(l, G_prev, G_new) is called after each layer. Returns the final Gram matrices (P, 2, 2) of the outputs."""
    G = np.array(G0, dtype=float)
    P = G.shape[0]
    for l in range(1, L + 1):
        g11, g22, g12 = G[:, 0, 0], G[:, 1, 1], G[:, 0, 1]
        rho = np.clip(g12 / np.sqrt(np.maximum(g11 * g22, 1e-300)), -1.0, 1.0)
        N = rng.standard_normal((P, n, 2))
        z1 = np.sqrt(2 * g11)[:, None] * N[:, :, 0]
        z2 = np.sqrt(2 * g22)[:, None] * (rho[:, None] * N[:, :, 0] + np.sqrt(np.maximum(1 - rho * rho, 0.0))[:, None] * N[:, :, 1])
        v1, v2 = np.maximum(z1, 0.0), np.maximum(z2, 0.0)
        if l < L and sigma > 0:
            if share_noise:
                xi = rng.standard_normal((P, n))
                v1 = v1 + sigma * xi; v2 = v2 + sigma * xi
            else:
                v1 = v1 + sigma * rng.standard_normal((P, n)); v2 = v2 + sigma * rng.standard_normal((P, n))
        Gn = np.empty_like(G)
        Gn[:, 0, 0] = np.mean(v1 * v1, 1); Gn[:, 1, 1] = np.mean(v2 * v2, 1)
        Gn[:, 0, 1] = Gn[:, 1, 0] = np.mean(v1 * v2, 1)
        if record is not None:
            record(l, G, Gn)
        G = Gn
    return G


def hermite(x, d):
    """Probabilists' Hermite polynomials He_1..He_4."""
    return [x, x * x - 1, x ** 3 - 3 * x, x ** 4 - 6 * x * x + 3][d - 1]


def univariate_shares(X, F, dmax=4):
    """Univariate Hermite shares of F(X), X ~ N(0, I_d): for degree k = 1..dmax, sum_j sum_c E[F_c He_k(X_j)]^2 / k! / V.
    Their sum is the first-order (additive) Sobol share S_1 truncated at degree dmax. Unbiased (two halves).
    Returns (list of shares by degree, V)."""
    P = X.shape[0]
    h = P // 2
    Fc = F - F.mean(0)
    V = float(np.sum(F.var(0, ddof=1)))
    out = []
    for k in range(1, dmax + 1):
        Hk = hermite(X, k)
        g1 = Fc[:h].T @ Hk[:h] / h
        g2 = Fc[h:2 * h].T @ Hk[h:2 * h] / h
        out.append(float(np.sum(g1 * g2)) / math.factorial(k) / V)
    return out, V


def deg2_share(X, F, outputs=None, chunk=64):
    """Hermite degree-2 share of F(X), split into diagonal (univariate x_j^2 terms) and off-diagonal parts:
    sum_c (1/2) |H_c|_F^2 / V with H_c = E[F_c (X X^T - I)], unbiased via two halves. outputs: indices of the outputs
    used (default all); the share is normalised by those outputs' variance. Cost O(P d^2 |outputs|)."""
    P, d = X.shape
    h = P // 2
    oc = np.arange(F.shape[1]) if outputs is None else np.asarray(outputs)
    Fc = F[:, oc] - F[:, oc].mean(0)
    VJ = float(np.sum(F[:, oc].var(0, ddof=1)))
    tot = 0.0
    diag = 0.0
    X1, X2 = X[:h], X[h:2 * h]
    for c0 in range(0, len(oc), chunk):
        for c in range(c0, min(c0 + chunk, len(oc))):
            H1 = (X1 * Fc[:h, c:c + 1]).T @ X1 / h
            H2 = (X2 * Fc[h:2 * h, c:c + 1]).T @ X2 / h
            m1, m2 = Fc[:h, c].mean(), Fc[h:2 * h, c].mean()
            H1[np.diag_indices(d)] -= m1
            H2[np.diag_indices(d)] -= m2
            tot += 0.5 * float(np.sum(H1 * H2))
            diag += 0.5 * float(np.sum(np.diag(H1) * np.diag(H2)))
    return dict(W2=tot / VJ, W2_diag=diag / VJ, W2_off=(tot - diag) / VJ, n_outputs=len(oc))


def even_share(f, X):
    """s_even = tr Var((f(x) + f(-x))/2) / tr Var f(x), on samples X (P, d). Antithetic gain = 1/(2 s_even)."""
    F = f(X)
    Fm = f(-X)
    E = 0.5 * (F + Fm)
    V = float(np.sum(np.concatenate([F, Fm]).var(0, ddof=1)))
    return float(np.sum(E.var(0, ddof=1))) / V, V


def sobol_gauss(d, m_log2, seed):
    """2^m scrambled Sobol' points (scipy LMS + digital shift) mapped to N(0, I_d) by the inverse cdf."""
    from scipy.stats import qmc
    from scipy.special import ndtri
    U = qmc.Sobol(d=d, scramble=True, seed=np.random.default_rng(seed)).random_base2(m_log2)
    return ndtri(np.clip(U, 1e-16, 1 - 1e-16))


def chain_split_terms(G, n, xn2):
    """Annealed three-term split integrands from final two-replica Gram matrices G (P, 2, 2) of the OUTPUTS (a.a'/n),
    for inputs with squared norms xn2 (P, 2) = |x|^2, |x'|^2. Returns per-chain arrays whose means give
    E_W of: tr Var (half the squared difference), V_rad, V_amp, V_dir (see var_split)."""
    mn = mean_radius(n)
    A1 = np.sqrt(n * G[:, 0, 0] / xn2[:, 0]); A2 = np.sqrt(n * G[:, 1, 1] / xn2[:, 1])
    cos = np.clip(G[:, 0, 1] / np.sqrt(np.maximum(G[:, 0, 0] * G[:, 1, 1], 1e-300)), -1, 1)
    half_sq = 0.5 * n * (G[:, 0, 0] + G[:, 1, 1] - 2 * G[:, 0, 1])
    return dict(V=half_sq, V_rad=(n - mn * mn) * 0.5 * (A1 ** 2 + A2 ** 2), V_amp=mn * mn * 0.5 * (A1 - A2) ** 2,
                V_dir=mn * mn * A1 * A2 * (1 - cos))


def chain_hamming(Gp, n):
    """Expected number of gate disagreements at the next layer given the previous Gram matrices Gp (P, 2, 2):
    n arccos(rho)/pi; a dead state (|h| = 0, every gate off) disagrees with a live one on n/2 units, two dead on 0."""
    g11, g22 = Gp[:, 0, 0], Gp[:, 1, 1]
    live1, live2 = g11 > 0, g22 > 0
    out = np.zeros(len(g11))
    both = live1 & live2
    rho = np.clip(Gp[both, 0, 1] / np.sqrt(g11[both] * g22[both]), -1, 1)
    out[both] = n * np.arccos(rho) / math.pi
    out[live1 ^ live2] = n / 2.0
    return out
