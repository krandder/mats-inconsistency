"""Mixture-density information estimators for Gaussian-noise channels (phase B).

Setting: H = mu(parent) + sigma * Z, Z ~ N(0, I_D), parent random. Then p(h) = E_parent N(h; mu(parent), sigma^2 I)
exactly, and with M sampled centres p_hat(h) = (1/M) sum_j N(h; mu_j, sigma^2 I).

For a target h_i generated from its own centre mu_i (row i of the centre bank):
  leave-in  (own centre included)  : E log p_hat_in  >= E log p     (InfoNCE identity)  -> h_in  <= h(H)
  leave-out (own centre excluded)  : E log p_hat_out <= E log p     (Jensen, unbiased p_hat) -> h_out >= h(H)
so h(H) is bracketed [h_in, h_out]; I(parent; H) = h(H) - h(sigma Z) in [h_in - hZ, h_out - hZ]. The lower end can
never exceed log2(M) (InfoNCE saturation); the bracket is wide exactly when M is not >> 2^I.
All outputs in bits.
"""
import math
import numpy as np
import torch

LOG2 = math.log(2.0)


def hZ_bits(D, sigma):
    """Differential entropy of N(0, sigma^2 I_D) in bits."""
    return 0.5 * D * math.log2(2 * math.pi * math.e * sigma ** 2)


def gauss_cap_bits(D, P, sigma):
    return 0.5 * D * math.log2(1 + P / sigma ** 2)


def _lse_kernel(T, C, sigma, own=None, chunk=256, cchunk=65536):
    """log sum_j exp(-|t_i - c_j|^2 / (2 sigma^2)) for every target; own[i] = index of i's own centre or -1.
    Returns (lse_all, lse_without_own) as float64 numpy."""
    T = torch.as_tensor(T, dtype=torch.float64); C = torch.as_tensor(C, dtype=torch.float64)
    N, M = len(T), len(C)
    c2 = (C ** 2).sum(1)
    out_in = np.empty(N); out_out = np.empty(N)
    s2 = 2 * sigma ** 2
    for i in range(0, N, chunk):
        t = T[i:i + chunk]; t2 = (t ** 2).sum(1, keepdim=True)
        parts_in, parts_out = [], []
        for j in range(0, M, cchunk):
            d2 = (t2 + c2[None, j:j + cchunk] - 2 * t @ C[j:j + cchunk].T).clamp_min(0)
            lk = -d2 / s2
            parts_in.append(torch.logsumexp(lk, 1))
            if own is not None:
                o = torch.as_tensor(own[i:i + chunk])
                rel = o - j
                msk = (rel >= 0) & (rel < lk.shape[1])
                lk2 = lk.clone()
                rows = torch.nonzero(msk).flatten()
                lk2[rows, rel[rows]] = -float('inf')
                parts_out.append(torch.logsumexp(lk2, 1))
        out_in[i:i + chunk] = torch.logsumexp(torch.stack(parts_in, 1), 1).numpy()
        if own is not None:
            out_out[i:i + chunk] = torch.logsumexp(torch.stack(parts_out, 1), 1).numpy()
    return out_in, (out_out if own is not None else None)


def logp_mix(T, C, sigma, own=None):
    """log p_hat(t) (nats) for the equal-weight isotropic mixture over centres C; with own, also leave-out."""
    D = T.shape[1]; M = len(C)
    norm = -0.5 * D * math.log(2 * math.pi * sigma ** 2)
    a, b = _lse_kernel(T, C, sigma, own)
    lin = a - math.log(M) + norm
    lout = None if b is None else b - math.log(M - 1) + norm
    return lin, lout


def entropy_bracket(T, C, sigma, own):
    """h(H) bracket in bits from targets T generated from centres C[own]. Returns dict with h_in, h_out (bits) and
    per-target SE."""
    lin, lout = logp_mix(T, C, sigma, own)
    f = lambda v: (float(-v.mean() / LOG2), float(v.std() / LOG2 / math.sqrt(len(v))))
    (hi, hise), (ho, hose) = f(lin), f(lout)
    return dict(h_in=hi, h_out=ho, se=max(hise, hose), M=len(C))


def cond_entropy_bracket(T, Cin, sigma):
    """h(H|X) bracket: Cin (N, K+1, D) = centres for each target's own x; Cin[:, 0] is the target's own centre and
    Cin[:, 1:] are K fresh, independent parent draws given the same x. Returns bits."""
    N, K1, D = Cin.shape
    T = torch.as_tensor(T, dtype=torch.float64); C = torch.as_tensor(Cin, dtype=torch.float64)
    lk = -((T[:, None] - C) ** 2).sum(-1) / (2 * sigma ** 2)                         # (N, K+1)
    norm = -0.5 * D * math.log(2 * math.pi * sigma ** 2)
    lin = torch.logsumexp(lk, 1) - math.log(K1) + norm
    lout = torch.logsumexp(lk[:, 1:], 1) - math.log(K1 - 1) + norm
    r = dict(hc_in=float(-lin.mean() / LOG2), hc_out=float(-lout.mean() / LOG2), K=K1 - 1)
    r['hc_gauss'] = gauss_cond_entropy(Cin[:, 1:], sigma) if K1 - 1 > D + 2 else float('inf')
    return r


def gauss_cond_entropy(C, sigma, seed=0):
    """Max-entropy upper bound on h(H|X): mean over x of 0.5 log2 det(2 pi e Cov(H|x)), Cov from K noisy samples
    (centres + sigma Z) with the Wishart log-det bias correction. C: (N, K, D) centres given each x."""
    from scipy.special import digamma
    N, K, D = C.shape
    g = np.random.default_rng(seed)
    H = C + sigma * g.standard_normal(C.shape)
    Hc = H - H.mean(1, keepdims=True)
    S = np.einsum('nkd,nke->nde', Hc, Hc) / (K - 1)
    ld = np.linalg.slogdet(S)[1]
    corr = sum(digamma((K - i) / 2) for i in range(1, D + 1)) + D * math.log(2 / (K - 1))
    return float((0.5 * (ld - corr) + 0.5 * D * math.log(2 * math.pi * math.e)).mean() / LOG2)


def mi_bracket(T, C, sigma, own, Cin=None):
    """I(X;H) bracket (bits). Without Cin, H|X is exactly N(mu(x), sigma^2) (deterministic parent)."""
    D = T.shape[1]
    e = entropy_bracket(T, C, sigma, own)
    hz = hZ_bits(D, sigma)
    if Cin is None:
        hc_in = hc_out = hz
    else:
        c = cond_entropy_bracket(T, Cin, sigma)
        hc_in, hc_out = max(c['hc_in'], hz), min(c['hc_out'], c['hc_gauss'])
    return dict(I_lo=e['h_in'] - hc_out, I_hi=e['h_out'] - hc_in, I_par_lo=e['h_in'] - hz, I_par_hi=e['h_out'] - hz,
                h_in=e['h_in'], h_out=e['h_out'], hc_in=hc_in, hc_out=hc_out, hZ=hz, se=e['se'], log2M=math.log2(len(C)))


def xgauss_bracket(T, means, covs, own, chunk=128):
    """x-level Gaussian-conditional estimator (an approximation, not a bound): p(h|x_j) ~ N(m_j, S_j) with m_j, S_j the
    moments of H given x_j from independent inner chains; p(h) ~ (1/M) sum_j N(h; m_j, S_j).
    Leave-in / leave-out over the x-level components gives the same InfoNCE / L1Out sandwich, but now the bank only has to
    resolve I(X;H), not I(parent;H). T[i] is a fresh sample of H given x_{own[i]}. Returns bits."""
    T = torch.as_tensor(np.asarray(T, np.float64)); M, D = means.shape
    L = torch.linalg.cholesky(torch.as_tensor(np.asarray(covs, np.float64)))
    Li = torch.linalg.inv(L); ld = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
    mu = torch.as_tensor(np.asarray(means, np.float64))
    N = len(T); lk = torch.empty(N, M, dtype=torch.float64)
    for j in range(0, M, chunk):
        r = T[None] - mu[j:j + chunk, None]                                   # (c, N, D)
        z = torch.einsum('cde,cne->cnd', Li[j:j + chunk], r)
        lk[:, j:j + chunk] = (-0.5 * (z ** 2).sum(2) - 0.5 * ld[j:j + chunk, None]).T
    lk = lk - 0.5 * D * math.log(2 * math.pi)
    o = torch.as_tensor(own)
    lown = lk[torch.arange(N), o]
    lin = torch.logsumexp(lk, 1) - math.log(M)
    lk[torch.arange(N), o] = -float('inf')
    lout = torch.logsumexp(lk, 1) - math.log(M - 1)
    hc = float(-lown.mean() / LOG2); h_in = float(-lin.mean() / LOG2); h_out = float(-lout.mean() / LOG2)
    return dict(xg_I_lo=h_in - hc, xg_I_hi=h_out - hc, xg_hc=hc, xg_h_in=h_in, xg_h_out=h_out, xg_M=M)


def moments_from_chains(C, sigma):
    """C: (n, K, D) centres given each x -> means (n, D), covs (n, D, D) of H|x = Cov(centres) + sigma^2 I."""
    m = C.mean(1); Cc = C - m[:, None]
    S = np.einsum('nkd,nke->nde', Cc, Cc) / (C.shape[1] - 1) + sigma ** 2 * np.eye(C.shape[2])[None]
    return m, S
