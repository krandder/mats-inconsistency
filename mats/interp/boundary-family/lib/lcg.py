"""Latent-conditional Gaussian (LCG) propagation with per-layer re-binning (boundary-family lane, prototype).

Family at layer l (pre-activation): z ~ sum_c pi_c N(m_c, S_c), the cells being the bins of a k-dimensional latent
xi = U_l^T (z - mu) on a B^k grid (quantile bins of each axis' marginal), with a Gaussian residual given the bin.
Step:
  1. re-bin: U_l = top-k eigenvectors of the mixture's total covariance; every current cell is sliced along the axes
     one after another (exact 1-D truncation of a Gaussian along u: conditional-Gaussian update; each slice
     moment-matched before the next axis); slices that land in the same bin are merged by moment matching;
  2. readout E[a_l] = sum_c pi_c E[ReLU(z) | c] (exact per cell);
  3. push every cell through layer l+1 with its exact Gaussian closure (gmprop.push).
Only the k-dimensional law is carried beyond Gaussian; the residual given the latent bin is Gaussian.
"""
import math
import torch
from gmprop import Cell, push, readout_mean
from gauss_relu import ncdf, npdf

DT = torch.float64


def gm_quantiles(cells, u, B):
    """B-quantile bin edges (B - 1 interior points) of the 1-D Gaussian mixture u . z."""
    w = torch.tensor([c.w for c in cells], dtype=DT)
    mu = torch.stack([u @ c.m for c in cells])
    sd = torch.stack([(u @ c.S @ u).clamp_min(1e-300).sqrt() for c in cells])
    lo = float((mu - 8 * sd).min()); hi = float((mu + 8 * sd).max())
    edges = []
    for q in [j / B for j in range(1, B)]:
        a, b = lo, hi
        for _ in range(60):
            x = 0.5 * (a + b)
            F = float((w * ncdf((x - mu) / sd)).sum() / w.sum())
            if F < q:
                a = x
            else:
                b = x
        edges.append(0.5 * (a + b))
    return edges


def slice_cell(c, u, edges):
    """Slices of N(c.m, c.S) along u by the bin edges: list of (bin index, Cell) with exact truncated moments."""
    mu_e = float(u @ c.m); Su = c.S @ u; var_e = float(u @ Su)
    sd_e = math.sqrt(max(var_e, 1e-300))
    bounds = [-math.inf] + list(edges) + [math.inf]
    out = []
    for j in range(len(bounds) - 1):
        a = (bounds[j] - mu_e) / sd_e; b = (bounds[j + 1] - mu_e) / sd_e
        ta = torch.tensor(a, dtype=DT); tb = torch.tensor(b, dtype=DT)
        P = float(ncdf(tb) - ncdf(ta))
        if P < 1e-12:
            continue
        pa = float(npdf(ta)) if math.isfinite(a) else 0.0
        pb = float(npdf(tb)) if math.isfinite(b) else 0.0
        aa = a * pa if math.isfinite(a) else 0.0
        bb = b * pb if math.isfinite(b) else 0.0
        Ez = (pa - pb) / P
        Vz = max(1 + (aa - bb) / P - Ez * Ez, 1e-12)
        m2 = c.m + (Su / sd_e) * Ez
        S2 = c.S - torch.outer(Su, Su) / var_e * (1 - Vz)
        out.append((j, Cell(c.w * P, m2, 0.5 * (S2 + S2.T))))
    return out


def rebin(cells, k, B):
    """Re-bin the mixture onto a B^k grid along the top-k axes of its total covariance."""
    w = torch.tensor([c.w for c in cells], dtype=DT); W_ = float(w.sum())
    mu = sum(c.w * c.m for c in cells) / W_
    Sig = sum(c.w * (c.S + torch.outer(c.m - mu, c.m - mu)) for c in cells) / W_
    ev, V = torch.linalg.eigh(Sig); U = V.flip(1)[:, :k]
    pieces = [((), c) for c in cells]
    for j in range(k):
        u = U[:, j]
        edges = gm_quantiles([p for _, p in pieces], u, B)
        new = []
        for key, p in pieces:
            for b, s in slice_cell(p, u, edges):
                new.append((key + (b,), s))
        pieces = new
    groups = {}
    for key, p in pieces:
        groups.setdefault(key, []).append(p)
    out = []
    for key, ps in groups.items():
        Wg = sum(p.w for p in ps)
        m = sum(p.w * p.m for p in ps) / Wg
        S = sum(p.w * (p.S + torch.outer(p.m - m, p.m - m)) for p in ps) / Wg
        out.append(Cell(Wg, m, 0.5 * (S + S.T)))
    return out


def run(net, L, k=2, B=4, l0=1, keep=True):
    """LCG from the exact Gaussian z_1 = W_1 x; re-binning from layer l0 on. Returns (est, None, per-layer estimates)."""
    n = net.n
    W = [w.to(DT) for w in net.W[:L]]
    s = [float(v) for v in net.s[:L]]
    cells = [Cell(1.0, torch.zeros(n, dtype=DT), W[0] @ W[0].T)]
    per = []
    for l in range(L):
        if l + 1 >= l0 and k > 0:
            cells = rebin(cells, k, B)
        per.append(readout_mean(cells))
        if l == L - 1:
            break
        cells = [push(c, W[l + 1], s[l] ** 2)[0] for c in cells]
    return per[-1], None, per
