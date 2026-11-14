"""Amendment-1 families (nongauss-family lane, 29 Oct 2026, written after R1 replicate 0 of net 113 was seen).
lib/ng.py (the pre-registered families) is not changed; this module adds:

PBA{k}: sphere chart; a joint block on the subspace U = span(top-k weighted PCs of the direction law, top-k eigenvectors
        of the downstream sensitivity M = E[g g^T], g = d(r . a_L)/d z_l, r ~ N(0, I)); the block coordinates y = U^T
        (w - m) are drawn jointly (one fit path, weighted); the rest is regressed linearly on y (so second moments stay
        exact) and its residual is carried by independent components with exact weighted marginals (in the residual's
        own PC basis). Fully non-Gaussian.
PBS{k}: the same with U = top-k sensitivity eigenvectors only.
PBW{k}: (Amendment 2, Kelvin's "bidirectional" importance) the block on U = orth(R V_k), V_k the top-k eigenvectors of
        S = R M R, R = (P C P)^1/2, C the weighted covariance of the direction law, P = I - u u^T (u its mean
        direction): the directions in which the state both varies (forward) and to which the output is sensitive
        (backward), over the remaining horizon to L.
PB{k} for any k (e.g. 64, 128): ng.build's PB family.
Sensitivity: 2^NS paths of the fit set (default 2^13), fixed noise draws, autograd through the exact layers l+1..L."""
import torch
import ng

CH = ng.CH


def sensitivity(dn, Z, l, L, g, ns=1 << 13):
    """M = E[g g^T] at the cut l for the all-outputs mean at L (g = J^T r, r ~ N(0, I), one r per path)."""
    idx = torch.randint(0, Z.shape[0], (ns,), generator=g, device=Z.device)
    M = torch.zeros(dn.n, dn.n, dtype=ng.F64, device=Z.device)
    for a in range(0, ns, 4096):
        z = Z[idx[a:a + 4096]].clone().requires_grad_(True)
        h = z
        with torch.enable_grad():
            for k in range(l, L + 1):
                if k > l:
                    h = h @ dn.W[k - 1].T
                h = torch.relu(h)
                if k < L:
                    h = h + dn.s[k - 1] * torch.randn(h.shape, generator=g, device=h.device)
            r = torch.randn(h.shape, generator=g, device=h.device)
            (h * r).sum().backward()
        G = z.grad.double()
        M += G.T @ G
    return M / ns


class BlockU:
    """w = m + U y + B y + rest; y = U^T (w - m) drawn jointly from one (weighted) fit row; rest = residual of the
    orthogonal complement after linear regression on y, carried by independent weighted-empirical PC scores."""

    def __init__(self, Wd, w, U):
        self.m, C = ng.mean_cov(Wd, w)
        mf = self.m.float(); Uf = U.float()
        self.U = Uf
        Y = torch.cat([(Wd[a:a + CH] - mf) @ Uf for a in range(0, Wd.shape[0], CH)])          # (N, k)
        P = torch.eye(Wd.shape[1], dtype=ng.F64, device=Wd.device) - U @ U.T
        Cyy = U.T @ C @ U; Cry = P @ C @ U
        B = Cry @ torch.linalg.pinv(Cyy)                                                     # (n, k)
        Crr = P @ C @ P - B @ Cyy @ B.T
        ev, V = ng.eig_desc(0.5 * (Crr + Crr.T))
        self.B = B.float(); self.V = V.float()
        R = torch.cat([((Wd[a:a + CH] - mf) @ P.float() - Y[a:a + CH] @ self.B.T) @ self.V
                       for a in range(0, Wd.shape[0], CH)])                                  # residual scores (N, n)
        self.Y, self.R = Y, R
        self.cw = torch.cumsum(w.double(), 0)
        self.mf = mf

    def sample(self, b, g):
        i0 = ng.wsample(self.cw, (b,), g)
        y = self.Y[i0]
        idx = ng.wsample(self.cw, (b, self.R.shape[1]), g)
        r = self.R.gather(0, idx)
        return self.mf + y @ (self.U + self.B).T + r @ self.V.T


def build2(name, X, g, dn=None, l=None, L=None):
    if name.startswith('PBW'):
        k = int(name[3:])
        rho, Wd = ng.sphere(X)
        M = sensitivity(dn, X, l, L, g)
        m, C = ng.mean_cov(Wd, rho)
        u = m / m.norm(); P = torch.eye(X.shape[1], dtype=ng.F64, device=X.device) - torch.outer(u, u)
        ev, V = torch.linalg.eigh(P @ C @ P)
        R = (V * ev.clamp_min(0).sqrt()) @ V.T
        _, Vs = ng.eig_desc(R @ M @ R)
        U, _ = torch.linalg.qr(R @ Vs[:, :k])
        return ng.Polar(X, lambda Wd_, w: BlockU(Wd_, w, U))
    if name.startswith('PBA') or name.startswith('PBS'):
        k = int(name[3:])
        rho, Wd = ng.sphere(X)
        M = sensitivity(dn, X, l, L, g)
        _, A = ng.eig_desc(M)
        if name.startswith('PBA'):
            m, C = ng.mean_cov(Wd, rho); _, V = ng.eig_desc(C)
            U, _ = torch.linalg.qr(torch.cat([V[:, :k], A[:, :k]], 1))
        else:
            U = A[:, :k]
        return ng.Polar(X, lambda Wd_, w: BlockU(Wd_, w, U))
    return ng.build(name, X, g)
