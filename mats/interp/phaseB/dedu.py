"""Deductive chain toolkit (phase B, amendment 1).

Distributions over R^D are represented as finite Gaussian mixtures (class GM). The exact pushforward through one noisy
layer, K q(h) = E_{h'~q} N(h; f(h'), sigma^2 I), is represented by a *node mixture*: deterministic quadrature nodes of q
(scrambled-Sobol / grid with fixed seeds, weights) mapped through f, each carrying N(., sigma^2 I). No sample of the real
network's p is used anywhere in the chain; samples of p appear only in validation code.
"""
import math
import numpy as np
import torch
from scipy.stats import qmc, norm

LOG2 = math.log(2.0)
torch.set_num_threads(1)


class GM:
    """sum_c w_c N(mu_c, S_c); S_c full (C, D, D) or isotropic scalar s2 for all components (iso=s2)."""

    def __init__(self, w, mu, S=None, iso=None):
        self.w = np.asarray(w, np.float64) / np.sum(w); self.mu = np.asarray(mu, np.float64)
        self.D = self.mu.shape[1]; self.iso = iso
        if iso is None:
            self.S = np.asarray(S, np.float64)
            self.L = np.linalg.cholesky(self.S); self.Li = np.linalg.inv(self.L)
            self.ld = 2 * np.log(np.diagonal(self.L, axis1=1, axis2=2)).sum(1)

    def logpdf(self, H, chunk=None):
        H = torch.as_tensor(np.asarray(H, np.float64)); out = np.empty(len(H))
        chunk = chunk or max(1, int(1.5e7 // (len(self.w) * (1 if self.iso is not None else self.D))))
        lw = torch.as_tensor(np.log(np.maximum(self.w, 1e-300))); mu = torch.as_tensor(self.mu)
        if self.iso is not None:                                  # float32 kernel (exponent error ~1e-4 nats)
            mu32 = mu.float(); c2 = (mu32 ** 2).sum(1); s2 = self.iso; lw32 = lw.float()
            chunk = max(1, int(6e7 // len(self.w)))
            for i in range(0, len(H), chunk):
                t = H[i:i + chunk].float()
                d2 = ((t ** 2).sum(1, keepdim=True) + c2[None] - 2 * t @ mu32.T).clamp_min(0)
                out[i:i + chunk] = torch.logsumexp(lw32[None] - d2 / (2 * s2), 1).double().numpy()
            return out - 0.5 * self.D * math.log(2 * math.pi * self.iso)
        Li = torch.as_tensor(self.Li); ld = torch.as_tensor(self.ld)
        for i in range(0, len(H), chunk):
            r = H[i:i + chunk, None, :] - mu[None]
            z = torch.einsum('cde,nce->ncd', Li, r)
            out[i:i + chunk] = torch.logsumexp(lw[None] - 0.5 * (z ** 2).sum(2) - 0.5 * ld[None], 1).numpy()
        return out - 0.5 * self.D * math.log(2 * math.pi)

    def logpdf_t(self, H):
        """Differentiable torch version (for sup search); H torch (n, D)."""
        lw = torch.as_tensor(np.log(np.maximum(self.w, 1e-300))); mu = torch.as_tensor(self.mu)
        if self.iso is not None:
            H32 = H.float(); mu32 = mu.float()
            d2 = ((H32 ** 2).sum(1, keepdim=True) + (mu32 ** 2).sum(1)[None] - 2 * H32 @ mu32.T).clamp_min(0)
            return (torch.logsumexp(lw.float()[None] - d2 / (2 * self.iso), 1)).double() - 0.5 * self.D * math.log(2 * math.pi * self.iso)
        Li = torch.as_tensor(self.Li); ld = torch.as_tensor(self.ld)
        z = torch.einsum('cde,nce->ncd', Li, H[:, None] - mu[None])
        return torch.logsumexp(lw[None] - 0.5 * (z ** 2).sum(2) - 0.5 * ld[None], 1) - 0.5 * self.D * math.log(2 * math.pi)

    def nodes(self, n_per, seed=0):
        """Deterministic quadrature nodes: scrambled Sobol mapped through each component. Returns (pts, weights, comp)."""
        sob = qmc.Sobol(self.D, scramble=True, seed=seed).random(n_per)
        z = norm.ppf(np.clip(sob, 1e-12, 1 - 1e-12))
        if self.iso is not None:
            pts = self.mu[:, None] + math.sqrt(self.iso) * z[None]
        else:
            pts = self.mu[:, None] + np.einsum('cde,ne->cnd', self.L, z)
        wts = np.repeat(self.w[:, None] / n_per, n_per, 1)
        comp = np.repeat(np.arange(len(self.w))[:, None], n_per, 1)
        return pts.reshape(-1, self.D), wts.ravel(), comp.ravel()

    def prob_halfspace(self, a, t):
        """q(a.h > t) exactly."""
        if self.iso is not None:
            sd = math.sqrt(self.iso) * np.linalg.norm(a)
            return float((self.w * norm.sf((t - self.mu @ a) / sd)).sum())
        sd = np.sqrt(np.einsum('d,cde,e->c', a, self.S, a))
        return float((self.w * norm.sf((t - self.mu @ a) / sd)).sum())

    def mean_cov(self):
        m = self.w @ self.mu
        if self.iso is not None:
            S = np.einsum('c,cd,ce->de', self.w, self.mu - m, self.mu - m) + self.iso * np.eye(self.D)
        else:
            S = np.einsum('c,cde->de', self.w, self.S) + np.einsum('c,cd,ce->de', self.w, self.mu - m, self.mu - m)
        return m, S


def mix(a, b, delta):
    """(1 - delta) a + delta b, both full-covariance GM."""
    return GM(np.r_[(1 - delta) * a.w, delta * b.w], np.r_[a.mu, b.mu], np.r_[a.S, b.S])


def node_moments(centres, wts, s2, groups=None):
    """Moments of the node mixture sum_j wts_j N(centres_j, s2 I), optionally per group label."""
    D = centres.shape[1]
    def mom(c, w):
        w = w / w.sum(); m = w @ c
        return w.sum(), m, np.einsum('n,nd,ne->de', w, c - m, c - m) + s2 * np.eye(D)
    if groups is None:
        return mom(centres, wts)
    out = []
    for g in np.unique(groups):
        s = groups == g
        W = wts[s].sum()
        _, m, S = mom(centres[s], wts[s]); out.append((g, W, m, S))
    return out


def kl_q_vs_node(q, K, n_per=2048, seed=1):
    """KL(q || K) in bits by QMC over q; K is a GM (node mixture, iso). Returns (kl, se)."""
    pts, wts, _ = q.nodes(n_per, seed)
    d = (q.logpdf(pts) - K.logpdf(pts)) / LOG2
    kl = float(wts @ d)
    return kl, float(math.sqrt(max(wts @ (d - kl) ** 2, 0) / max(n_per, 1)))


def sup_log_ratio(A, B, cand, steps=60, top=32, lr=0.05):
    """Search estimate of sup_h log2 A(h)/B(h) (a lower bound on D_inf(A||B)): evaluate on candidate points, then
    gradient ascent from the best `top`. A, B are GM."""
    r = (A.logpdf(cand) - B.logpdf(cand)) / LOG2
    idx = np.argsort(-r)[:top]
    h = torch.tensor(cand[idx], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([h], lr=lr)
    for _ in range(steps):
        loss = -(A.logpdf_t(h) - B.logpdf_t(h)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        r2 = ((A.logpdf_t(h) - B.logpdf_t(h)) / LOG2).numpy()
    return float(max(r.max(), r2.max())), float(r.max())
