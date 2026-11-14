"""lyapunov-dimension addendum (29 Oct 2026): the first-order optimal block as an oracle family, in a copy of the
nongauss lane's harness (lib/ng.py and lib/ng2.py are byte-identical copies; nothing in work/nongauss-family is edited).

PBO{k}: sphere chart (as PBW). R = (P C P)^1/2 with C the rho-weighted covariance of the direction law w = z/|z| and
        P = I - u u^T (u its mean direction); M the downstream sensitivity at the cut (ng2.sensitivity); V_k the top-k
        eigenvectors of S = R M R. The block is U = orth(Pi M R V_k), Pi the projector onto range(R). Conditioning on
        U^T w is conditioning on the whitened functionals R^+ V_k, the optimum of Theorem B in the report, because
        R^+ v = Pi M R v / mu for an eigenvector v of S with eigenvalue mu > 0. Everything else is as PBW: the block
        coordinates are drawn jointly from one weighted fit row, the rest is regressed linearly on them and carried by
        independent exact weighted marginals (ng2.BlockU). PBW{k} uses U = orth(R V_k) instead.
PBW{k}, PBO{k} here take a precomputed M, so both families of one (net, cut) use the same sensitivity estimate.
compose3: ng.compose with families that need (dn, l, L) (sensitivity at every re-projection layer)."""
import torch
import ng
import ng2

CH = ng.CH


def blocks_from_M(Wd, rho, M, k, kind):
    """U for PBW (kind 'W') or PBO (kind 'O')."""
    m, C = ng.mean_cov(Wd, rho)
    u = m / m.norm(); P = torch.eye(Wd.shape[1], dtype=ng.F64, device=Wd.device) - torch.outer(u, u)
    ev, V = torch.linalg.eigh(P @ C @ P)
    ev = ev.clamp_min(0)
    R = (V * ev.sqrt()) @ V.T
    mu, Vs = ng.eig_desc(R @ M @ R)
    if kind == 'W':
        A = R @ Vs[:, :k]
    else:
        keep = ev > 1e-10 * ev.max()
        Pi = V[:, keep] @ V[:, keep].T
        A = Pi @ M @ R @ Vs[:, :k]
    U, _ = torch.linalg.qr(A)
    return U


def build3(name, X, g, dn=None, l=None, L=None, M=None, ns=1 << 13):
    """PB{k}, PBW{k}, PBO{k} (M computed here if not given), and every family of ng2.build2 / ng.build."""
    if name.startswith('PBO') or name.startswith('PBW'):
        k = int(name[3:])
        rho, Wd = ng.sphere(X)
        if M is None:
            M = ng2.sensitivity(dn, X, l, L, g, ns=ns)
        U = blocks_from_M(Wd, rho, M, k, 'O' if name.startswith('PBO') else 'W')
        return ng.Polar(X, lambda Wd_, w: ng2.BlockU(Wd_, w, U))
    return ng2.build2(name, X, g, dn, l, L)


@torch.no_grad()
def compose3(dn, name, X0, l0, Lmax, proj, NC, g, chunk=CH):
    """ng.compose, with build3 (so sensitivity-based families re-estimate M at every re-projection layer)."""
    X = X0
    M1 = torch.zeros(Lmax - l0 + 1, dn.n, dtype=ng.F64, device=dn.dev); M2 = torch.zeros_like(M1)
    nproj, count = [], 0
    for l in range(l0, Lmax + 1):
        M1[l - l0], M2[l - l0] = ng.relu_moments(X)
        nproj.append(count)
        if l == Lmax:
            break
        if name != 'MC' and l in proj:
            fam = build3(name, X, g, dn, l, Lmax)
            X = torch.cat([fam.sample(min(chunk, NC - a), g).float() for a in range(0, NC, chunk)])
            del fam
            count += 1
        X = ng.push(dn, X, l, g)
    return M1, M2, nproj
