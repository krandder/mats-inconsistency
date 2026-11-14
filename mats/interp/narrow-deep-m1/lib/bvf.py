"""Two-sided downstream for plain ReLU nets: the value function V_c(h) = E[a_L | h_c = h] on k relevant directions,
stored on a grid of directions with the radius exact (positive homogeneity), plus a deep part. Deterministic; every
FLOP is counted (flopscope weights).

Deep part (from c_d): reference ray v_ref at c_d; J_ref = noiseless Jacobian d a_L / d h_{c_d} along the path from v_ref
(exact on its linear region); noise correction from K2-Stein (noise included) started at point masses rho * v_ref on a
log-radius table: V_{c_d}(h) = J_ref h + [phi(rho) - rho a0], rho = (r . h)/(r . v_ref), r = J_ref^T a0 / |a0|^2,
a0 = J_ref v_ref.
Middle (c to c_d): noiseless deterministic passes (homogeneous), either direct from each grid node (variant 'direct') or
by backward recursion layer by layer on per-layer grids (variant 'recur', not used for the headline).
Grid: relevant directions U (n x k, orthonormal) and the dropped part as a linear function of t = U^T h (B, n x k), so
h(t) = (U + B) t. Directions in gnomonic coordinates around e0 (unit k-vector): t = (e0.t) (e0 + E x), x in
[-X, X]^(k-1) on G points per dimension; multilinear interpolation of q(x) = F_{c->c_d}((U + B)(e0 + E x)); query:
V(t) = V_deep((e0.t) q(x)).
"""
import math
import itertools
import torch
import k2 as K2

DT = torch.float64


class Flops:
    def __init__(self):
        self.n = 0.0

    def add(self, k):
        self.n += float(k)


def mmf(M, K, N):
    return M * N * (2 * K - 1)


def noiseless_pass(net, H, l_from, l_to, fl):
    """H (P, n) states after noise of layer l_from; noiseless layers l_from+1..l_to; returns the state at l_to."""
    P, n = H.shape
    for l in range(l_from, l_to):
        H = torch.relu(H @ net.W[l].T)
    fl.add(P * (l_to - l_from) * (mmf(1, n, n) + n))
    return H


def jac_pass(net, h, l_from, l_to, fl):
    """single noiseless path from h (n,) through layers l_from+1..l_to; returns end state and Jacobian d h_end / d h."""
    n = h.shape[0]
    J = torch.eye(n, dtype=DT)
    for l in range(l_from, l_to):
        z = net.W[l] @ h
        D = (z > 0).to(DT)
        h = torch.relu(z)
        J = D[:, None] * (net.W[l] @ J)
    fl.add((l_to - l_from) * (mmf(n, n, 1) + mmf(n, n, n) + 2 * n * n))
    return h, J


def future_classes(net, H, l_from, fl):
    """noiseless future gate-pattern hash (layers l_from+1..L) and output a_L for states H (P, n) at l_from."""
    P, n = H.shape
    MOD = (1 << 61) - 1
    pw = torch.tensor([1 << i for i in range(n)], dtype=torch.long)
    key = torch.zeros(P, dtype=torch.long)
    for l in range(l_from, net.L):
        z = H @ net.W[l].T
        code = ((z > 0).long() * pw).sum(1)
        key = (key * 1_000_003 + code * 7_919 + l) % MOD
        H = torch.relu(z)
    fl.add(P * (net.L - l_from) * (mmf(1, n, n) + 2 * n))
    return key, H


class Deep:
    def __init__(self, net, c_d, cand, cand_w, fl, nrho=10, span=(1e-6, 1e3), noise_table=False):
        """cand (P, n): candidate states at c_d (deterministic probes pushed noiselessly); the reference ray is taken
        from the future class (noiseless gate pattern c_d+1..L) carrying the largest sum of w |a_L|."""
        n, L = net.n, net.L
        self.net, self.c_d = net, c_d
        if cand.dim() == 1:
            cand = cand[None, :]; cand_w = torch.ones(1, dtype=DT)
        key, aL = future_classes(net, cand, c_d, fl)
        mass = cand_w * aL.norm(dim=1)
        uk, inv = torch.unique(key, return_inverse=True)
        cm = torch.zeros(len(uk), dtype=DT).index_add_(0, inv, mass)
        best = int(cm.argmax())
        sel = torch.nonzero(inv == best)[:, 0]
        j = sel[mass[sel].argmax()]
        v_ref = cand[j]
        self.class_share = float(cm[best] / cm.sum().clamp_min(1e-300))
        self.nclass = int(len(uk))
        rho_typ = float(v_ref.norm())
        v = v_ref / v_ref.norm()
        a0, J = jac_pass(net, v, c_d, L, fl)
        self.J, self.a0, self.v = J, a0, v
        nr = float(a0.pow(2).sum())
        self.r = (J.T @ a0) / max(nr, 1e-300)
        self.rv = float(self.r @ v)
        fl.add(4 * n * n)
        self.noise_table = noise_table
        if not noise_table:                                             # D0: noiseless deep part (Amendment 2)
            self.rhos = torch.tensor([0.0, 1.0], dtype=DT)
            self.corr = torch.zeros(2, n, dtype=DT)
            return
        rhos = [0.0] + [rho_typ * 10 ** e for e in torch.linspace(math.log10(span[0]), math.log10(span[1]), nrho).tolist()]
        phis = []
        for rho in rhos:
            mu = v * rho
            C = torch.zeros(n, n, dtype=DT)
            if net.sigma > 0 and c_d > 0:
                C = C + torch.diag(K2.noise_var(net, c_d - 1, n))      # the state at c_d carries layer c_d's noise
            m, _, f = K2.plain_prop(net, mu, C, c_d, L, exact=False)
            fl.add(f)
            phis.append(m)
        self.rhos = torch.tensor(rhos, dtype=DT)
        self.phi = torch.stack(phis)                                    # (nrho+1, n)
        self.corr = self.phi - self.rhos[:, None] * a0[None, :]         # noise correction table

    def value(self, H, fl):
        """H (P, n) states at c_d -> V (P, n)."""
        P, n = H.shape
        V = H @ self.J.T
        if not self.noise_table:
            fl.add(P * mmf(1, n, n))
            return V
        rho = (H @ self.r) / self.rv
        # interpolate the correction in log rho (linear from 0 to the first grid point)
        rh = self.rhos
        lr = torch.log(rh[1:])
        x = torch.log(rho.clamp_min(1e-300))
        idx = torch.searchsorted(lr, x).clamp(1, len(lr) - 1)
        x0, x1 = lr[idx - 1], lr[idx]
        wgt = ((x - x0) / (x1 - x0)).clamp(0, 1)
        c0, c1 = self.corr[1:][idx - 1], self.corr[1:][idx]
        cor = c0 + wgt[:, None] * (c1 - c0)
        small = rho < rh[1]
        if small.any():
            f = (rho[small] / rh[1]).clamp(0, 1)[:, None]
            cor[small] = self.corr[0][None, :] * (1 - f) + self.corr[1][None, :] * f
        fl.add(P * (mmf(1, n, n) + 2 * n + 40 + 3 * n))
        return V + cor


def unscented(mu, C):
    """2n+1 sigma points (weights) of N(mu, C), exact for cubics."""
    n = mu.shape[0]
    ev, U = torch.linalg.eigh(0.5 * (C + C.T))
    S = U * ev.clamp_min(0).sqrt()
    kap = 3.0 - n
    pts = [mu]; w = [kap / (n + kap)]
    for i in range(n):
        d = math.sqrt(n + kap) * S[:, i] if (n + kap) > 0 else math.sqrt(n) * S[:, i]
        pts += [mu + d, mu - d]; w += [1 / (2 * (n + kap))] * 2
    return torch.stack(pts), torch.tensor(w, dtype=DT)


def relevant_directions(net, c, c_d, deep, probes, pw, Mmix, Cmix, k, fl):
    """Sample-free sensitivity S = C^1/2 M C^1/2 from Jacobians along deterministic probe paths (rows of probes, weights
    pw), C the forward description's covariance at c. Returns U (n x k, orthonormal state directions), eigenvalues."""
    n = net.n
    M = torch.zeros(n, n, dtype=DT); tot = 0.0
    for h, w in zip(probes, pw.tolist()):
        if w <= 0:
            continue
        hd, Jm = jac_pass(net, h, c, c_d, fl)
        Jt = deep.J @ Jm
        M += w * (Jt.T @ Jt); tot += w
        fl.add(mmf(n, n, n) * 2)
    M = M / max(tot, 1e-300)
    M = M / M.trace().clamp_min(1e-300)
    ev, Uc = torch.linalg.eigh(0.5 * (Cmix + Cmix.T))
    Ch = Uc @ torch.diag(ev.clamp_min(0).sqrt()) @ Uc.T
    S = Ch @ M @ Ch
    eS, VS = torch.linalg.eigh(0.5 * (S + S.T)); eS, VS = eS.flip(0), VS.flip(1)
    U = torch.linalg.qr(Ch @ VS[:, :k])[0]
    fl.add(10 * n ** 3)
    return U, eS


class Grid:
    def __init__(self, net, c, c_d, deep, U, B, e0, G, X, fl):
        n, k = net.n, U.shape[1]
        self.net, self.c, self.c_d, self.deep = net, c, c_d, deep
        self.U, self.B, self.k, self.G, self.X = U, B, k, G, X
        self.e0 = e0 / e0.norm()
        # orthonormal complement E of e0 in R^k
        Q = torch.linalg.qr(torch.cat([self.e0[:, None], torch.eye(k, dtype=DT)], 1))[0]
        self.E = Q[:, 1:k]
        self.ax = torch.linspace(-X, X, G, dtype=DT)
        if k == 1:
            nodes = self.e0[None, :]
        else:
            mesh = torch.stack(torch.meshgrid(*([self.ax] * (k - 1)), indexing='ij'), -1).reshape(-1, k - 1)
            nodes = self.e0[None, :] + mesh @ self.E.T                   # (G^(k-1), k)
        H = nodes @ (U + B).T                                            # states at c
        fl.add(len(nodes) * (mmf(1, k, n) + k * k))
        self.q = noiseless_pass(net, H, c, c_d, fl)                      # (G^(k-1), n) at c_d
        self.nnodes = len(nodes)

    def value(self, T, fl):
        """T (P, k) relevant coordinates -> V (P, n); also returns the fraction outside the grid box."""
        P, k = T.shape
        rho = T @ self.e0
        if k == 1:
            Hd = rho[:, None] * self.q[0][None, :]
            fl.add(P * 2 * self.net.n)
            return self.deep.value(Hd, fl), 0.0
        x = (T @ self.E) / rho[:, None].clamp_min(1e-300)
        out = (x.abs() > self.X).any(1) | (rho <= 0)
        u = ((x.clamp(-self.X, self.X) + self.X) / (2 * self.X) * (self.G - 1))
        i0 = u.floor().clamp(0, self.G - 2).long()
        f = u - i0
        qv = torch.zeros(P, self.net.n, dtype=DT)
        strides = [self.G ** (k - 2 - d) for d in range(k - 1)]
        for corner in itertools.product((0, 1), repeat=k - 1):
            idx = torch.zeros(P, dtype=torch.long); wt = torch.ones(P, dtype=DT)
            for d, b in enumerate(corner):
                idx = idx + (i0[:, d] + b) * strides[d]
                wt = wt * (f[:, d] if b else (1 - f[:, d]))
            qv += wt[:, None] * self.q[idx]
        fl.add(P * (mmf(1, k, k) + 3 * k + (2 ** (k - 1)) * (2 * self.net.n + 2 * k)))
        Hd = rho.clamp_min(0)[:, None] * qv
        return self.deep.value(Hd, fl), float(out.double().mean())


def meet_mixture(grid, w, mu, C, fl, rule='unscented'):
    """sum_j w_j E_{N(mu_j, C_j)}[V_c] with V_c from the grid; t = U^T h. rule: 'mean' (1 node) or 'unscented'."""
    U = grid.U
    k = U.shape[1]
    tm = mu @ U                                                          # (B, k)
    fl.add(len(w) * mmf(1, mu.shape[1], k))
    if rule == 'mean':
        V, out = grid.value(tm, fl)
        return (w[:, None] * V).sum(0) / w.sum(), out
    Ct = U.T @ C @ U                                                     # (B, k, k)
    fl.add(len(w) * 2 * mmf(k, mu.shape[1], k))
    ev, Q = torch.linalg.eigh(0.5 * (Ct + Ct.transpose(1, 2)))
    S = Q * ev.clamp_min(0).sqrt()[:, None, :]
    # symmetric 2k-point rule (kappa = 0): +-sqrt(k) sd along each principal axis, weights 1/(2k); exact for cubics
    a = math.sqrt(k)
    pts, wts = [], []
    for i in range(k):
        pts += [tm + a * S[:, :, i], tm - a * S[:, :, i]]
        wts += [1 / (2 * k)] * 2
    fl.add(len(w) * (k ** 3 * 10 + 2 * k * k))
    tot = torch.zeros(mu.shape[1], dtype=DT); outs = 0.0
    for p, wt in zip(pts, wts):
        V, out = grid.value(p, fl)
        tot += wt * (w[:, None] * V).sum(0); outs += wt * out
    return tot / w.sum(), outs


def meet_direct(net, c, c_d, deep, w, mu, C, fl, U=None, spread_tol=None):
    """sum_j w_j E_{N(mu_j, C_j)}[V_c] with V_c evaluated exactly (noiseless pass c -> c_d, then the deep part) at the
    component means; if U (n x k) is given, components whose spread along U is large (max sd / |mu| > spread_tol) use
    the 2k-point rule along U instead of the mean alone. Deterministic."""
    B, n = mu.shape
    pts = [mu]; wts = [w]
    if U is not None and spread_tol is not None:
        k = U.shape[1]
        Ct = U.T @ C @ U
        ev, Q = torch.linalg.eigh(0.5 * (Ct + Ct.transpose(1, 2)))
        sd = ev.clamp_min(0).sqrt()
        big = (sd.max(1).values / mu.norm(dim=1).clamp_min(1e-300)) > spread_tol
        fl.add(B * (2 * mmf(k, n, k) + 10 * k ** 3))
        if big.any():
            # replace the mean node of big components by 2k nodes along U Q
            wts[0] = torch.where(big, torch.zeros_like(w), w)
            a = math.sqrt(k)
            for i in range(k):
                d = (U @ (Q[big][:, :, i] * sd[big][:, i:i + 1]).T).T * a     # (nb, n)
                for sgn in (1, -1):
                    pts.append(mu[big] + sgn * d); wts.append(w[big] / (2 * k))
    P = torch.cat(pts); Wt = torch.cat(wts)
    Hd = noiseless_pass(net, P, c, c_d, fl)
    V = deep.value(Hd, fl)
    meet_direct.last = (P, Wt, V)
    return (Wt[:, None] * V).sum(0) / w.sum(), len(P)


def tail_from_meet(w, mu, C, t_thr, fl):
    """P(|a_L| > t) from the last meet_direct call: each component mean's output norm s_j = |V(mu_j)|, radial spread
    from the component's covariance along mu_j scaled by the gain |V(mu_j)| / |mu_j|; Gaussian per component."""
    P, Wt, V = meet_direct.last
    B = mu.shape[0]
    s = V[:B].norm(dim=1)
    nm = mu.norm(dim=1).clamp_min(1e-300)
    u = mu / nm[:, None]
    vr = torch.einsum('bi,bij,bj->b', u, C, u).clamp_min(0)
    sd = (s / nm) * vr.sqrt()
    from scipy.stats import norm as _n
    p = torch.tensor(_n.sf(((t_thr - s) / sd.clamp_min(1e-300)).numpy()), dtype=DT)
    fl.add(B * (3 * mu.shape[1] ** 2 + 120))
    return float((w * p).sum() / w.sum())
