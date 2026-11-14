"""Boundary-aware Gaussian-mixture propagation through a noisy ReLU MLP (prototype, boundary-family lane).

State at layer l: the pre-activation z_l ~ sum_c pi_c N(m_c, S_c) (cells). One step (layer l -> l+1):
  1. split: cells whose ReLU acts most nonlinearly are split, either along their nonlinearity-weighted principal
     direction (latent split, Huber's 3-component library) or exactly at one neuron's decision boundary (boundary split:
     two children carrying the exact masses Phi(t), Phi(-t) of the two sides and the moment-matched truncated
     Gaussian of each side, with that neuron's gate forced on / off in the child);
  2. per cell, exact moments of a = ReLU(z) under the cell's Gaussian (rectified-Gaussian means; covariances by the
     bivariate-normal formula for pairs with an uncertain neuron, closed form for decided pairs; forced gates linear);
  3. z_{l+1} | c ~ N(W mu_a, W Cov(a) W^T + s^2 W W^T) (exact first two moments of the cell's push-forward);
  4. merge cells back to C_max (Ward-type cost in the shared metric) if needed.
The estimate of E[a_L] is sum_c pi_c E[ReLU(z_L) | c] (exact per cell).
FLOPs are counted analytically (flopscope float32 rates as in work/arc-noisy: matmul M N (2K - 1), pair formula
FLOPS_PAIR, per-neuron rectified moments FLOPS_NEURON, eigen/cholesky ~ 10 n^3)."""
import math
import torch
from gauss_relu import relu_moments, relu_moments2, ncdf, npdf, FLOPS_PAIR, FLOPS_NEURON

DT = torch.float64
# moment-preserving 3-component split of N(0, 1): children N(a_j, b^2) with weights w_j matching the moments of
# N(0, 1) through order 5 (b^2 = 1/2, a = sqrt(3/2), w = 1/6, 2/3, 1/6)
LIB_W = torch.tensor([1 / 6, 2 / 3, 1 / 6], dtype=DT)
LIB_A = torch.tensor([-math.sqrt(1.5), 0.0, math.sqrt(1.5)], dtype=DT)
LIB_B = math.sqrt(0.5)


def g_nonlin(t):
    """Var(ReLU(t + Z)) - Phi(t)^2: variance of the ReLU of N(t, 1) not explained by its best linear fit."""
    P = ncdf(t); p = npdf(t)
    v = (t * t + 1) * P + t * p - (t * P + p) ** 2
    return (v - P * P).clamp_min(0)


class Cell:
    __slots__ = ('w', 'm', 'S', 'force')

    def __init__(self, w, m, S, force=None):
        self.w, self.m, self.S = w, m, S
        self.force = force or {}                 # neuron -> +1 (forced on) / -1 (forced off), current layer only


def cell_moments_a(c, det=8.0):
    """Mean and covariance of a = ReLU(z) for z ~ N(m, S) with forced gates; returns (mu, cov, flops)."""
    mu, cov, npairs = relu_moments2(c.m, c.S, det=det)
    n = c.m.shape[0]
    fl = npairs * FLOPS_PAIR + n * FLOPS_NEURON
    if c.force:
        sd = c.S.diagonal().clamp_min(1e-300).sqrt()
        P = ncdf(c.m / sd)
        on = [i for i, s in c.force.items() if s > 0]
        off = [i for i, s in c.force.items() if s < 0]
        for i in on:                             # a_i = z_i: Cov(z_i, ReLU(z_j)) = S_ij Phi(t_j) (Stein)
            mu[i] = c.m[i]
            row = c.S[i] * P
            cov[i, :] = row; cov[:, i] = row
        for i in on:                             # forced-on pairs (including the diagonal) are linear
            for j in on:
                cov[i, j] = c.S[i, j]
        for i in off:                            # off masks last, so no on-row restores them
            mu[i] = 0.0
            cov[i, :] = 0.0; cov[:, i] = 0.0
        fl += len(c.force) * 2 * n
    return mu, cov, fl


def push(c, W, s2):
    """z_{l+1} | c from the cell at layer l (W: (n, n) next weights, s2: noise variance after a_l)."""
    mu, cov, fl = cell_moments_a(c)
    n = W.shape[0]
    m2 = W @ mu
    S2 = W @ (cov + s2 * torch.eye(n, dtype=DT)) @ W.T
    S2 = 0.5 * (S2 + S2.T)
    fl += 2 * n * (2 * n - 1) * n + n * (2 * n - 1)
    return Cell(c.w, m2, S2), fl


def nonlin_score(c):
    sd = c.S.diagonal().clamp_min(1e-300).sqrt()
    t = c.m / sd
    g = g_nonlin(t)
    if c.force:
        for i in c.force:
            g[i] = 0.0
    return c.w * float((sd * sd * g).sum()), g, sd


def latent_split(c, g, sd):
    """Split along the nonlinearity-weighted principal direction of the cell (whitened), 3 children."""
    n = c.m.shape[0]
    ev, V = torch.linalg.eigh(c.S)
    ev = ev.clamp_min(0)
    Lh = V * ev.sqrt()                           # S = Lh Lh^T
    # M = sum_i g_i (Lh^T e_i)(Lh^T e_i)^T / s_i^2
    Bm = Lh / sd[:, None]                        # rows: Lh^T e_i / s_i
    M = Bm.T @ (Bm * g[:, None])
    _, U = torch.linalg.eigh(M)
    u = U[:, -1]
    d = Lh @ u                                   # direction in z-space, Var along it absorbed
    kids = []
    S2 = c.S - (1 - LIB_B ** 2) * torch.outer(d, d)
    S2 = 0.5 * (S2 + S2.T)
    for wj, aj in zip(LIB_W, LIB_A):
        kids.append(Cell(c.w * float(wj), c.m + float(aj) * d, S2.clone(), dict(c.force)))
    fl = 20 * n ** 3 + 3 * n * n
    return kids, fl


def boundary_split(c, i):
    """Exact split at z_i = 0: children = the two truncated sides, moment matched, gate i forced."""
    n = c.m.shape[0]
    si = float(c.S[i, i].sqrt()); t = float(c.m[i]) / si
    v = c.S[:, i] / si                           # z = m + v Z + independent rest, Z ~ N(0, 1)
    tt = torch.tensor(t, dtype=DT)
    Pon = float(ncdf(tt)); Poff = 1 - Pon
    kids = []
    for side, P in ((1, Pon), (-1, Poff)):
        if P < 1e-15:
            continue
        if side > 0:                             # Z > -t
            lam = float(npdf(tt)) / P
            ez = lam; vz = 1 - t * lam - lam * lam
        else:                                    # Z < -t
            lam = float(npdf(tt)) / P
            ez = -lam; vz = 1 + t * lam - lam * lam
        vz = max(vz, 1e-12)
        m2 = c.m + ez * v
        S2 = c.S - (1 - vz) * torch.outer(v, v)
        S2 = 0.5 * (S2 + S2.T)
        f = dict(c.force); f[i] = side
        kids.append(Cell(c.w * P, m2, S2, f))
    return kids, 6 * n * n


def merge_to(cells, Cmax):
    """Greedy Ward-type merging in the metric of the weighted average covariance until len(cells) <= Cmax."""
    if len(cells) <= Cmax:
        return cells, 0
    n = cells[0].m.shape[0]
    w = torch.tensor([c.w for c in cells], dtype=DT)
    Sbar = sum(c.w * c.S for c in cells) / w.sum()
    ev, V = torch.linalg.eigh(Sbar)
    Wh = V / ev.clamp_min(1e-12 * float(ev.max())).sqrt()          # whitening
    X = torch.stack([c.m for c in cells]) @ Wh
    fl = 10 * n ** 3 + len(cells) * n * (2 * n - 1)
    alive = list(range(len(cells)))
    cells = list(cells)
    while len(alive) > Cmax:
        Xa = X[alive]; wa = w[alive]
        D2 = torch.cdist(Xa, Xa) ** 2
        cost = (wa[:, None] * wa[None, :]) / (wa[:, None] + wa[None, :]) * D2
        cost.fill_diagonal_(float('inf'))
        k = int(torch.argmin(cost))
        a, b = alive[k // len(alive)], alive[k % len(alive)]
        ca, cb = cells[a], cells[b]
        W_ = ca.w + cb.w
        m = (ca.w * ca.m + cb.w * cb.m) / W_
        S = (ca.w * (ca.S + torch.outer(ca.m - m, ca.m - m)) + cb.w * (cb.S + torch.outer(cb.m - m, cb.m - m))) / W_
        f = {i: s for i, s in ca.force.items() if cb.force.get(i) == s}
        cells[a] = Cell(W_, m, S, f)
        X[a] = (ca.w * X[a] + cb.w * X[b]) / W_
        w[a] = W_
        alive.remove(b)
        fl += len(alive) ** 2 * (3 * n) + 3 * n * n
    return [cells[i] for i in alive], fl


def readout_mean(cells):
    m = 0.0
    for c in cells:
        sd = c.S.diagonal().clamp_min(1e-300).sqrt()
        t = c.m / sd
        e = sd * (t * ncdf(t) + npdf(t))
        for i, sgn in c.force.items():           # forced gates: the cell is the truncated side
            e[i] = c.m[i] if sgn > 0 else 0.0
        m = m + c.w * e
    return m


def run(net, L, Cmax=1, mode='latent', split_budget=None, verbose=False, keep=False, ret_cells=False, keep_frac=0.5,
        l0=1):
    """Propagate from z_1 = W_1 x (exact Gaussian) to z_L. mode: 'latent' (direction splits), 'boundary' (exact
    boundary splits on the highest-g neuron), 'none' (Gaussian closure, Cmax ignored).
    Returns (estimate of E[a_L], flops[, list of per-layer estimates])."""
    n = net.n
    W = [w.to(DT) for w in net.W[:L]]
    s = [float(v) for v in net.s[:L]]
    cells = [Cell(1.0, torch.zeros(n, dtype=DT), W[0] @ W[0].T)]
    fl = n * n * (2 * n - 1)
    per_layer = []
    for l in range(L):
        # split (at layer l, before the ReLU)
        if mode != 'none' and Cmax > 1 and l + 1 >= l0:
            nsplit = 0
            while len(cells) < Cmax:
                scores = [nonlin_score(c) for c in cells]
                k = max(range(len(cells)), key=lambda j: scores[j][0])
                sc, g, sd = scores[k]
                if sc <= 0:
                    break
                c = cells.pop(k)
                if mode == 'latent':
                    kids, f = latent_split(c, g, sd)
                else:
                    i = int(torch.argmax(g * sd * sd))
                    kids, f = boundary_split(c, i)
                cells.extend(kids); fl += f + len(cells) * 3 * n
                nsplit += 1
                if split_budget is not None and nsplit >= split_budget:
                    break
        if keep or l == L - 1:
            per_layer.append(readout_mean(cells))
        if l == L - 1:
            break
        new = []
        for c in cells:
            c2, f = push(c, W[l + 1], s[l] ** 2)
            new.append(c2); fl += f
        cells = new
        if mode != 'none' and Cmax > 1:
            ck = max(1, int(Cmax * keep_frac))
            if len(cells) > ck:
                cells, f = merge_to(cells, ck); fl += f
        if verbose:
            print(l + 1, len(cells), flush=True)
    est = per_layer[-1]
    out = (est, fl, per_layer) if keep else (est, fl)
    if ret_cells:
        out = out + (cells,)
    return out


def grid_split(c, k, nodes=3):
    """Tensor-product moment-preserving split of a Gaussian cell along its top-k PCA modes (3 nodes per mode:
    weights 1/6, 2/3, 1/6 at 0, +-sqrt(3/2) sd, child variance 1/2 along the mode)."""
    n = c.m.shape[0]
    ev, V = torch.linalg.eigh(c.S)
    ev = ev.flip(0).clamp_min(0); V = V.flip(1)
    kids = [c]
    for j in range(k):
        d = V[:, j] * float(ev[j]).__pow__(0.5)
        new = []
        for p in kids:
            S2 = p.S - (1 - LIB_B ** 2) * torch.outer(d, d)
            S2 = 0.5 * (S2 + S2.T)
            for wj, aj in zip(LIB_W, LIB_A):
                new.append(Cell(p.w * float(wj), p.m + float(aj) * d, S2, dict(p.force)))
        kids = new
    return kids, 10 * n ** 3 + len(kids) * n


def run_grid(net, L, k, l_s, resplit=0, Cmax=None):
    """Gaussian closure to layer l_s, then a 3^k grid split along the top-k PCA modes, then every cell pushed with its
    own Gaussian closure to layer L (optionally re-split each cell along its top `resplit` modes every layer, merging
    back to Cmax)."""
    n = net.n
    W = [w.to(DT) for w in net.W[:L]]
    s = [float(v) for v in net.s[:L]]
    cells = [Cell(1.0, torch.zeros(n, dtype=DT), W[0] @ W[0].T)]
    fl = n * n * (2 * n - 1)
    per = []
    for l in range(L):
        if l + 1 == l_s and k > 0:
            new = []
            for c in cells:
                kids, f = grid_split(c, k); new.extend(kids); fl += f
            cells = new
        elif resplit and l + 1 > l_s:
            new = []
            for c in cells:
                kids, f = grid_split(c, resplit); new.extend(kids); fl += f
            cells = new
        per.append(readout_mean(cells))
        if l == L - 1:
            break
        new = []
        for c in cells:
            c2, f = push(c, W[l + 1], s[l] ** 2); new.append(c2); fl += f
        cells = new
        if Cmax is not None and len(cells) > Cmax:
            cells, f = merge_to(cells, Cmax); fl += f
    return per[-1], fl, per
