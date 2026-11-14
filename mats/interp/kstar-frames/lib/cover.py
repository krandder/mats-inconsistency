"""Pairwise scores of reference frames on test paths, with the reference frame projected onto the test path's active
set: G = orth(D_x F_y). Uses tr(P_G A) = tr((G^T G)^{-1} G^T A G) with G = D_x F_y, so no per-pair QR:
  C = V_x^T F_y  (V_x lies in range(D_x), so V_x^T D_x = V_x^T),  Gram = F_y^T D_x F_y,
  CS = tr(Gram^{-1} (S C)^T (S C)) / sum_{i<=k} s_i^2,  O = tr(Gram^{-1} C_k^T C_k) / k  (C_k = first k rows of C).
All arithmetic in float64 (v1 formed Gram in float32, which made pseudo-inverses of nearly singular Gram matrices
amplify rounding into CS > 1; found by the GPT-6 review, 2026-10-31; v1 kept in notes/cover_v1_fp32gram.py.bak)."""
import numpy as np
import torch


def psd_pinv(G, chunk=2048):
    """Pseudo-inverse of a batch of symmetric PSD k x k matrices via eigh, in chunks (batched cusolver eigh allocates
    workspace proportional to the batch: ~10 GB for 16384 matrices on the L4)."""
    flat = G.reshape(-1, G.shape[-2], G.shape[-1])
    out = torch.empty_like(flat)
    for i in range(0, flat.shape[0], chunk):
        ev, U = torch.linalg.eigh(flat[i:i + chunk])
        inv = torch.where(ev > 1e-10 * ev[..., -1:].clamp_min(1e-300), 1.0 / ev, torch.zeros_like(ev))
        out[i:i + chunk] = (U * inv[..., None, :]) @ U.transpose(-1, -2)
    return out.reshape(G.shape)


def psd_solve(G, X1, X2, ridge=1e-15):
    """Gram^+ X for a batch of PSD k x k Gram matrices with eigenvalues in [0, 1] (float64): Cholesky of G + ridge I
    (fast, batched); entries whose factorisation fails or whose smallest pivot^2 is below 1e-6 fall back to the eigh
    pseudo-inverse. The ridge (1e-15) only guards the factorisation; ill-conditioned cases go to eigh."""
    k = G.shape[-1]
    eye = torch.eye(k, dtype=G.dtype, device=G.device)
    Lc, info = torch.linalg.cholesky_ex(G + ridge * eye)
    Y1 = torch.cholesky_solve(X1, Lc)
    Y2 = torch.cholesky_solve(X2, Lc)
    diag = torch.diagonal(Lc, dim1=-2, dim2=-1)
    bad = (info != 0) | (diag.min(-1).values ** 2 < 1e-6)
    if bool(bad.any()):
        idx = bad.nonzero(as_tuple=True)
        Gi = psd_pinv(G[idx])
        Y1[idx] = Gi @ X1[idx]
        Y2[idx] = Gi @ X2[idx]
    return Y1, Y2


def pair_scores(Vx, sx, Dx, Fy, k, bx=32, by=512, ridge=1e-9, dtype=torch.float64):
    """Vx (NB, n, KK), sx (NB, KK), Dx (NB, n) 0/1, Fy (NA, n, k). Returns CS, O as (NB, NA) float32 (computed in dtype)."""
    NB, n, KK = Vx.shape
    NA = Fy.shape[0]
    CS = torch.zeros(NB, NA, dtype=torch.float32)
    O = torch.zeros(NB, NA, dtype=torch.float32)
    s2 = (sx.double() ** 2)
    top = s2[:, :k].sum(-1)
    for j0 in range(0, NA, by):
        F = Fy[j0:j0 + by].to(dtype)                                # (b, n, k)
        b = F.shape[0]
        Fflat = F.permute(1, 0, 2).reshape(n, b * k)                # (n, b k)
        Q = (F[:, :, :, None] * F[:, :, None, :]).permute(1, 0, 2, 3).reshape(n, b * k * k)   # (n, b k k)
        for i0 in range(0, NB, bx):
            V = Vx[i0:i0 + bx].to(dtype)                            # (a, n, KK)
            a = V.shape[0]
            C = (V.transpose(1, 2) @ Fflat).reshape(a, KK, b, k).permute(0, 2, 1, 3).double()   # (a, b, KK, k)
            Gram = (Dx[i0:i0 + a].to(dtype) @ Q).reshape(a, b, k, k).double()
            Cs = C * sx[i0:i0 + a].double()[:, None, :, None]
            Ck = C[:, :, :k, :]
            Ys, Yk = psd_solve(Gram, Cs.transpose(-1, -2) @ Cs, Ck.transpose(-1, -2) @ Ck)   # Gram^+ X
            CS[i0:i0 + a, j0:j0 + b] = (torch.diagonal(Ys, dim1=-2, dim2=-1).sum(-1) / top[i0:i0 + a, None]).float().cpu()
            O[i0:i0 + a, j0:j0 + b] = (torch.diagonal(Yk, dim1=-2, dim2=-1).sum(-1) / k).float().cpu()
    return CS, O


def greedy_cover(cov, frac=(0.5, 0.9, 0.99), max_centres=None):
    """cov: (Nsel, NA) bool, cov[x, y] = reference y covers path x. Greedy max-coverage over references.
    Returns list of chosen reference indices (in order) and the coverage after each pick."""
    cov = cov.clone()
    Nsel, NA = cov.shape
    covered = torch.zeros(Nsel, dtype=torch.bool)
    chosen, curve = [], []
    max_centres = max_centres or NA
    gains = cov.sum(0).to(torch.int64)
    while len(chosen) < max_centres:
        j = int(torch.argmax(gains))
        if gains[j] <= 0:
            break
        new = cov[:, j] & ~covered
        covered |= new
        chosen.append(j)
        curve.append(float(covered.float().mean()))
        gains -= cov[new].sum(0).to(torch.int64)
        if curve[-1] >= max(frac):
            break
    return chosen, curve


def r99(p, target=0.01):
    """Random-centre count: smallest R with mean exp(-R p) <= target (p = ball masses). inf if unreachable."""
    p = np.asarray(p, dtype=float)
    if np.mean(p <= 0) >= target:
        return float('inf')
    lo, hi = 0.0, 1.0
    f = lambda R: np.mean(np.exp(-R * p))
    while f(hi) > target:
        hi *= 2
        if hi > 1e300:
            return float('inf')
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi
