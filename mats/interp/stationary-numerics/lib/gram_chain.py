"""Exact replica Gram-matrix chain of Model S (sigma = 1), weights integrated out exactly.

R noise replicas h^(1..R) share every weight matrix W_l (iid N(0, 2/n)). Because a fresh W_l is rotation invariant on its
input side, the rows of W_l h^(a) are, given the past, iid over coordinates i with law N(0, 2G), where
G_ab = h^(a).h^(b)/n is the replicas' Gram matrix. Hence G_l is an exact Markov chain:
  G' = (1/n) sum_i v_i v_i^T,  v_i = relu(z_i) + xi_i,  z_i ~ N(0, 2G) iid,  xi_i ~ N(0, S) iid,
where S (R x R) is the noise-sharing matrix of that layer (S_ab = 1 if replicas a, b share the layer's noise, else 0;
S_aa = 1). This is the noisy version of the conditional-Gaussian Gram recursion (Roberts-Yaida eq. 4.69-4.71).
Its conditional mean is the kernel map  E[G'_ab | G] = sqrt(G_aa G_bb) kappa(rho_ab) + S_ab  (kappa(1) = 1), and its
conditional covariance is (1/n) x (the four-point source): the PDLT kernel + vertex recursion with a noise source.

Closed forms used for observables at the final layer (annealed over W_1, exact given G):
  |h1 - h2|^2 / n = G11 + G22 - 2 G12;  |u1 - u2|^2 = 2 (1 - rho);  (log|h1| - log|h2|)^2 = (log G11 - log G22)^2 / 4;
  E|a1 - a2|^2 / n = G11 + G22 - 2 sqrt(G11 G22) kappa(rho)  for a = relu(W_1 h);  E #gate disagreements = n arccos(rho)/pi.
"""
import math
import numpy as np


def kappa(c):
    c = np.clip(c, -1.0, 1.0)
    return (np.sqrt(np.maximum(1.0 - c * c, 0.0)) + (math.pi - np.arccos(c)) * c) / math.pi


def sqrt2G(G):
    """Batched symmetric square root of 2G (P, R, R); exact for rank-deficient G (identical replicas)."""
    w, V = np.linalg.eigh(2.0 * G)
    return np.einsum('pab,pb,pcb->pac', V, np.sqrt(np.maximum(w, 0.0)), V)


def _labels(share):
    R = share.shape[0]
    lab = -np.ones(R, int)
    c = 0
    for i in range(R):
        if lab[i] < 0:
            for j in range(R):
                if share[i, j]:
                    lab[j] = c
            c += 1
    return lab


def step(G, share, n, rng):
    """One exact layer for P chains. G: (P, R, R). share: (R, R) 0/1 equivalence matrix, or labels (P, R) per chain
    (replicas with equal labels share that layer's noise)."""
    P, R, _ = G.shape
    if R == 2:
        g11, g22, g12 = G[:, 0, 0], G[:, 1, 1], G[:, 0, 1]
        rho = np.clip(g12 / np.sqrt(g11 * g22), -1.0, 1.0)
        N = rng.standard_normal((P, n, 2))
        z1 = np.sqrt(2 * g11)[:, None] * N[:, :, 0]
        z2 = np.sqrt(2 * g22)[:, None] * (rho[:, None] * N[:, :, 0] + np.sqrt(1 - rho * rho)[:, None] * N[:, :, 1])
        z = np.stack([z1, z2], 2)
    else:
        z = np.einsum('pab,pib->pia', sqrt2G(G), rng.standard_normal((P, n, R)))
    a = np.maximum(z, 0.0)
    share = np.asarray(share)
    if share.ndim == 2 and share.shape == (R, R):
        lab = _labels(share)
        x = rng.standard_normal((P, n, lab.max() + 1))[:, :, lab]
    else:
        labs = share  # (P, R) integer labels
        x = np.take_along_axis(rng.standard_normal((P, n, R)), np.broadcast_to(labs[:, None, :], (P, n, R)), axis=2)
    v = a + x
    return np.einsum('pia,pib->pab', v, v) / n


def sample_Q_stationary(n, P, rng, steps=None):
    """Exact stationary radius Q = |h|^2/n by running the exact scalar chain (Q' = (S/n)(1 + 2 Q beta))."""
    steps = 80 * n if steps is None else steps
    Q = np.full(P, 0.58 * n)
    for _ in range(steps):
        M = rng.binomial(n, 0.5, P)
        b = np.where(M == 0, 0.0, np.where(M == n, 1.0, rng.beta(np.maximum(M, 1) / 2.0, np.maximum(n - M, 1) / 2.0)))
        Q = rng.chisquare(n, P) / n * (1.0 + 2.0 * Q * b)
    return Q


def diffs(G, n, i=0, j=1):
    """Squared-difference functionals between replicas i and j of the final Gram matrices (P, R, R)."""
    g11, g22, g12 = G[:, i, i], G[:, j, j], G[:, i, j]
    rho = np.clip(g12 / np.sqrt(g11 * g22), -1.0, 1.0)
    return dict(h=n * (g11 + g22 - 2 * g12), u=2 * (1 - rho), logR=0.25 * (np.log(g11) - np.log(g22)) ** 2,
                a=n * (g11 + g22 - 2 * np.sqrt(g11 * g22) * kappa(rho)), g=n * np.arccos(rho) / math.pi,
                rho=rho, Q=0.5 * (g11 + g22))
