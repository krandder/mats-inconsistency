s = open('prop.py').read()
anchor = "        self.dbg.update(mo=mo, Po=Po)\n"
assert anchor in s
patch = open('_patch_av.txt').read() + \
"""            if self.K4:
                Ao = Ao + ein('cshb,ctshj->cthbj', mv, self.pad(Aa) - self.pad((h1 * al)[..., None] * Al))
"""
s = s.replace(anchor, patch + anchor)
s = s.replace("def gelu(z):", '''def _sig_derivs(x):
    p = torch.sigmoid(-x)
    q = p * (1 - p)
    return p, -q, q * (1 - 2 * p)


def _bivar_sigmoid(m, P, n):
    """f = sigmoid(-x); for jointly Gaussian l (mean m [C,...], cov P [C,N,N]) return E f, E f', E f'' and the matrices
    F_ij[a,b] = E[f^(i)(l_a) f^(j)(l_b)] (2-D Gauss-Hermite off the diagonal, 1-D on it) and G_ij = F_ij - F_i F_j."""
    C = m.shape[0]
    m = m.reshape(C, -1).to(F64); P = P.to(F64)
    var = P.diagonal(0, 1, 2).clamp_min(1e-12); sd = var.sqrt()
    rho = (P / (sd[:, :, None] * sd[:, None, :])).clamp(-0.999, 0.999)
    z, w = np.polynomial.hermite_e.hermegauss(n); w = w / w.sum()
    z = torch.tensor(z, dtype=F64, device=m.device); w = torch.tensor(w, dtype=F64, device=m.device)
    A = _sig_derivs(m[:, :, None] + sd[:, :, None] * z)                                  # 3 x [C,N,n]
    xb = m[:, None, :, None, None] + sd[:, None, :, None, None] * (
        rho[..., None, None] * z[:, None] + (1 - rho ** 2).sqrt()[..., None, None] * z[None, :])
    Bw = [torch.einsum('cabnm,m->cabn', b, w) for b in _sig_derivs(xb)]                 # 3 x [C,N,N,n]
    Fm = [(w * a).sum(-1) for a in A]
    G = {}
    eyeN = torch.eye(m.shape[1], dtype=torch.bool, device=m.device)
    for i in range(3):
        for j in range(3):
            if i + j > 2 and not (i == 1 and j == 1):
                continue
            Fij = torch.einsum('can,cabn,n->cab', A[i], Bw[j], w)
            Dij = (w * A[i] * A[j]).sum(-1)
            Fij = torch.where(eyeN, torch.diag_embed(Dij), Fij)
            G[f'{i}{j}f'] = Fij
            G[f'{i}{j}'] = Fij - Fm[i][:, :, None] * Fm[j][:, None, :]
    sh = m.shape
    return Fm[0], Fm[1], Fm[2], G


def gelu(z):''', 1)
s = s.replace("        self.rowproj = True\n", "        self.rowproj = False\n        self.av_stein = True\n        self.gh2 = 8\n", 1)
open('prop.py', 'w').write(s)
print('ok')
