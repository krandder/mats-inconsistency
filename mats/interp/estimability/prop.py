"""Deterministic, differentiable cumulant propagator for model.Net over the shrunken source world.

Input law: a finite mixture over partitions; given the partition the input is exactly Gaussian. Every partition
shape (orbit under source permutations) is propagated separately ("component", batch dim C) and the statistics
are mixed exactly at read-out (law of total cumulance; the network and the propagator are permutation-equivariant,
so the other members of an orbit are token permutations of the representative).

State per component, over the reference vector R = [residual stream (T*d coords); y*]:
  m  mean,  P  full joint covariance,
  higher cumulants (K=4 only) in "source" form:  kappa_r[x]_{i1..ir} = sum_j c_r[j] A[i1,j] ... A[ir,j],  r=3,4.
A source j is created at every scalar output of a non-linear op (its marginal kappa_3/kappa_4 not explained by
transported incoming sources); A holds each coordinate's linear response to source j. Linear maps transport A
exactly; non-linear ops transport it by linear response (Stein / first Hermite coefficient).
Truncation (stated in PREREG.md): cumulants of order >= 5 are zero; joint 3rd/4th cumulants are only those
representable by the sources (one-vertex creation + linear transport); multi-vertex diagrams (e.g. kappa_3 created
from products of covariances across coordinates) are dropped; covariances between distinct coordinates use the
Gaussian closure (Hermite expansion to 3rd order for elementwise ops, Isserlis for products, joint lognormal for
softmax); K=4 corrections enter means, variances, marginal kappa_3/4 and the exp-CGF of softmax.
K=2 = same code with all sources off (Gaussian closure, ARC "covariance propagation" generalised).
"""
import math
import numpy as np
import torch
import torch.nn.functional as Fn
import torch.utils.checkpoint
import world_small as W
from model import LN_EPS

GH_N = 48
_GX, _GW = np.polynomial.hermite_e.hermegauss(GH_N)
_GW = _GW / _GW.sum()
F64 = torch.float64


def _herm(x):
    x2 = x * x
    return x, x2 - 1, x * x2 - 3 * x, x2 * x2 - 6 * x2 + 3, x2 ** 3 - 15 * x2 * x2 + 45 * x2 - 15


def elementwise(fn, mu, var, k3, k4, K4):
    """Moments of fn(z), z with marginal cumulants (mu,var,k3,k4) (Edgeworth density if K4 else Gaussian).
    Returns mean, var, kappa3, kappa4 of fn(z) and Hermite coefficients h_k = E[fn^(k)(z)], k=1..3."""
    dt = mu.dtype
    x = torch.tensor(_GX, dtype=F64, device=mu.device); w = torch.tensor(_GW, dtype=F64, device=mu.device)
    mu, var = mu.to(F64), var.to(F64).clamp_min(1e-12)
    sd = var.sqrt()
    h1, h2, h3, h4, h6 = _herm(x)
    if K4 and k3 is not None:
        g3 = (k3.to(F64) / sd ** 3)[..., None]; g4 = (k4.to(F64) / var ** 2)[..., None]
        wt = w * (1 + g3 / 6 * h3 + g4 / 24 * h4 + g3 ** 2 / 72 * h6)
    else:
        wt = w.expand(mu.shape + (GH_N,))
    z = mu[..., None] + sd[..., None] * x
    f = fn(z)
    E = (wt * f).sum(-1)
    dv = f - E[..., None]
    V = (wt * dv ** 2).sum(-1).clamp_min(1e-30)
    m3 = (wt * dv ** 3).sum(-1); m4 = (wt * dv ** 4).sum(-1)
    # Hermite coefficients as E[f^(k)] (equal to E[f He_k]/sd^k by Stein; numerically stable as sd -> 0)
    d1, d2, d3 = DERIVS[fn](z)
    c1 = (wt * d1).sum(-1); c2 = (wt * d2).sum(-1); c3 = (wt * d3).sum(-1)
    return [t.to(dt) for t in (E, V, m3, m4 - 3 * V ** 2, c1, c2, c3)]


def qf_cumulants(Sg, mu, B):
    """kappa_1..4 of x^T B x for x ~ N(mu, Sg); Sg, B [..., n, n] (B symmetric), mu [..., n]. Exact (Gaussian)."""
    Sg, mu, B = Sg.to(F64), mu.to(F64), B.to(F64)
    BS = B @ Sg; BS2 = BS @ BS
    tr = lambda M: M.diagonal(0, -2, -1).sum(-1)
    Bm = (B @ mu[..., None])[..., 0]
    SBm = (Sg @ Bm[..., None])[..., 0]
    k1 = tr(BS) + (mu * Bm).sum(-1)
    k2 = 2 * tr(BS2) + 4 * (Bm * SBm).sum(-1)
    SBSBm = (Sg @ (B @ SBm[..., None]))[..., 0]
    k3 = 8 * (tr(BS2 @ BS) + 3 * (Bm * SBSBm).sum(-1))
    k4 = 48 * (tr(BS2 @ BS2) + 4 * (SBm * (B @ SBSBm[..., None])[..., 0]).sum(-1))
    return k1, k2, k3, k4


_LAG = {}


def _laguerre(al, n=32):
    if (al, n) not in _LAG:
        from scipy.special import roots_genlaguerre
        _LAG[(al, n)] = roots_genlaguerre(n, al)
    return _LAG[(al, n)]


def _sig_derivs(x):
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


def gelu(z):
    return 0.5 * z * (1 + torch.erf(z / math.sqrt(2)))


def _phi(z):
    return torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _gelu_d(z):
    p = _phi(z)
    return 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * p, (2 - z * z) * p, (z ** 3 - 4 * z) * p


def _softplus_d(z):
    s_ = torch.sigmoid(z); q = s_ * (1 - s_)
    return s_, q, q * (1 - 2 * s_)


def _signeg(z):
    return torch.sigmoid(-z)


def _signeg_d(z):
    p = torch.sigmoid(-z); q = p * (1 - p)
    return -q, q * (1 - 2 * p), -q * (1 - 6 * q)


DERIVS = {gelu: _gelu_d, Fn.softplus: _softplus_d, _signeg: _signeg_d}



def _psd(P, log=None):
    """Nearest PSD matrix (eigenvalue clipping) of a [C,n,n] covariance; straight-through (the correction is
    treated as a constant in the backward pass). log: list receiving the clipped negative mass relative to the trace."""
    Ps = 0.5 * (P + P.transpose(-1, -2))
    with torch.no_grad():
        L, U = torch.linalg.eigh(Ps.to(F64))
        neg = L.clamp_max(0)
        corr = -(U * neg[..., None, :]) @ U.transpose(-1, -2)
        if log is not None:
            log.append(float((-neg.sum(-1) / L.clamp_min(0).sum(-1).clamp_min(1e-30)).max()))
    return Ps + corr.to(P.dtype)


class Propagator:
    def __init__(self, K=4, dtype=torch.float32, device='cpu', components=None):
        self.K4 = (K == 4)
        self.ln2nd = False
        self.ln_pair = True
        self.ln_pair_n = 6
        self.ln_n = 32
        self.rowproj = False
        self.av_stein = True
        self.av_k3 = True
        self.psd = True
        self.debug = False
        self.checkpoint = True
        self.psd_log = []
        self.gh2 = 8
        self.dt, self.dev = dtype, device
        reps, probs, perms = W.shape_components()
        if components is not None:                      # for tests: explicit list of partition indices
            reps = np.array(components); probs = W.PRIOR[reps] / W.PRIOR[reps].sum(); perms = [tuple(range(W.S))]
        m0, C0 = W.component_gaussians()
        self.m0 = torch.tensor(m0[reps], dtype=dtype, device=device)
        self.C0 = torch.tensor(C0[reps], dtype=dtype, device=device)
        self.p = torch.tensor(probs, dtype=F64, device=device)
        self.perms = perms

    def _sel(self, T, H, w):
        key = (T, H, w, self.dt, str(self.dev))
        if not hasattr(self, '_selc'):
            self._selc = {}
        if key not in self._selc:
            sel = torch.zeros(T, T, H, T, T, H, dtype=self.dt)
            for t_ in range(T):
                for h_ in range(H):
                    sel[t_, :, h_, t_, w, h_] = 1.0
            self._selc[key] = sel.reshape(T * T * H, T * T * H).to(self.dev)
        return self._selc[key]

    # ------------------------------------------------------------ source registry
    def _reset(self, C):
        self.c3, self.c4, self.J, self.C = [], [], 0, C
        self.psd_log = []

    def cs(self):
        if self.J == 0:
            z = torch.zeros(self.C, 0, dtype=self.dt, device=self.dev); return z, z
        return torch.cat(self.c3, 1), torch.cat(self.c4, 1)

    def pad(self, A):
        return A if A.shape[-1] == self.J else Fn.pad(A, (0, self.J - A.shape[-1]))

    def marg(self, A):
        """marginal kappa_3, kappa_4 of coordinates with factor tensor A [C, ..., J]."""
        if not self.K4 or self.J == 0:
            z = torch.zeros(A.shape[:-1], dtype=self.dt, device=self.dev); return z, z
        A = self.pad(A); c3, c4 = self.cs()
        sh = (A.shape[0],) + (1,) * (A.dim() - 2) + (A.shape[-1],)
        return (A ** 3 * c3.view(sh)).sum(-1), (A ** 4 * c4.view(sh)).sum(-1)

    def new_sources(self, A, g3, g4):
        """Append one source per coordinate of a block (A [C, *shape, J'], g [C, *shape]); returns new A."""
        if not self.K4:
            return A
        C = A.shape[0]; shape = A.shape[1:-1]; n = int(np.prod(shape))
        A = self.pad(A)
        eye = torch.eye(n, dtype=self.dt, device=self.dev).expand(C, n, n).reshape((C,) + tuple(shape) + (n,))
        self.c3.append(g3.reshape(C, n).to(self.dt)); self.c4.append(g4.reshape(C, n).to(self.dt))
        self.J += n
        return torch.cat([A, eye], -1)

    def mixed(self, A1, A2):
        """M(a,b) = sum_j c3 (a^2 b + a b^2)/2 + c4 (4a^3 b + 6 a^2 b^2 + 4 a b^3)/24 for all pairs:
        A1 [C, n1, J], A2 [C, n2, J] -> [C, n1, n2]. Extra exponent in E[e^x e^y] = E e^x E e^y exp(Cov + M)."""
        if not self.K4 or self.J == 0:
            return torch.zeros(A1.shape[0], A1.shape[1], A2.shape[1], dtype=self.dt, device=self.dev)
        A1, A2 = self.pad(A1), self.pad(A2)
        c3, c4 = self.cs()
        e = lambda X, c, Y: torch.einsum('cij,cj,ckj->cik', X, c, Y)
        return (0.5 * (e(A1 ** 2, c3, A2) + e(A1, c3, A2 ** 2))
                + (4 * e(A1 ** 3, c4, A2) + 6 * e(A1 ** 2, c4, A2 ** 2) + 4 * e(A1, c4, A2 ** 3)) / 24)

    # ------------------------------------------------------------ ops on blocks
    # A block is (m [C,*s], P [C,*s,*s], R [C,*s,NR] = Cov(block, reference), A [C,*s,J]).
    def linear_tok(self, blk, Wt, b):
        """token-wise affine map: blk over (T, din) -> (T, dout)."""
        m, P, R, A = blk
        m2 = torch.einsum('cti,oi->cto', m, Wt) + b
        P2 = torch.einsum('oi,ctisj,pj->ctosp', Wt, P, Wt)
        R2 = torch.einsum('oi,ctir->ctor', Wt, R)
        A2 = torch.einsum('oi,ctij->ctoj', Wt, A)
        return m2, P2, R2, A2

    def layernorm(self, blk, g, b):
        """Within a token: exact moments of y = u (|u|^2/d + eps)^(-1/2) for Gaussian u, via
        r^k = Gamma(k/2)^-1 int_0^inf s^(k/2-1) e^(-s v) ds and tilted Gaussians (generalised Gauss-Laguerre in s).
        Across tokens and to the reference: linear response with the exact mean Jacobian E[dy/du] (+ the
        second-order Hermite term across tokens if self.ln2nd).
        K=4: new sources = marginal kappa_3/4 of y under Gaussian u; incoming sources transported by the Jacobian
        (their effect on LN means/variances is dropped)."""
        m, P, R, A = blk
        C, T, d = m.shape
        dt = self.dt
        A = self.pad(A)
        m = m - m.mean(-1, keepdim=True)
        P = P - P.mean(2, keepdim=True); P = P - P.mean(4, keepdim=True)
        R = R - R.mean(2, keepdim=True); A = A - A.mean(2, keepdim=True)
        Sd = torch.diagonal(P, 0, 1, 3).permute(0, 3, 1, 2).to(F64)          # [C,T,d,d]
        m64 = m.to(F64)
        Ev = ((Sd.diagonal(0, -2, -1).sum(-1) + (m64 ** 2).sum(-1)) / d).detach() + LN_EPS
        I = torch.eye(d, dtype=F64, device=self.dev)
        # one Gauss-Laguerre(alpha=-1/2) node set for all orders k: int s^(k/2-1) e^(-cs) G ds with weights x^((k-1)/2)
        x, w = _laguerre(-0.5, self.ln_n)
        x = torch.tensor(x, dtype=F64, device=self.dev); w = torch.tensor(w, dtype=F64, device=self.dev)
        sn = x / Ev[..., None]                                                   # [C,T,n]
        Bk = I + (2 * sn / d)[..., None, None] * Sd[:, :, None]                  # [C,T,n,d,d]
        Binv = torch.linalg.inv(Bk)
        logdet = torch.linalg.slogdet(Bk)[1]
        ms = (Binv @ m64[:, :, None, :, None])[..., 0]                           # [C,T,n,d]
        Ss = Binv @ Sd[:, :, None]
        logZ = -0.5 * logdet - (sn / d) * (m64[:, :, None] * ms).sum(-1) - sn * LN_EPS
        base = w * torch.exp(logZ + x)
        res = {}
        for k in (1, 2, 3, 4, 5):
            if k == 5 and not self.ln2nd:
                continue
            al = k / 2 - 1
            wt = base * x ** (al + 0.5) * Ev[..., None] ** (-al - 1) / math.gamma(k / 2)
            res[k] = (wt, ms, Ss)
        wt1, ms1, Ss1 = res[1]; wt2, ms2, Ss2 = res[2]; wt3, ms3, Ss3 = res[3]; wt4, ms4, Ss4 = res[4]
        Er = wt1.sum(-1)
        Ey = (wt1[..., None] * ms1).sum(2)                                       # [C,T,d]
        Eyy = (wt2[..., None, None] * (Ss2 + ms2[..., :, None] * ms2[..., None, :])).sum(2)
        Euur3 = (wt3[..., None, None] * (Ss3 + ms3[..., :, None] * ms3[..., None, :])).sum(2)
        Jac = Er[..., None, None] * I - Euur3 / d                                # [C,T,d(out),d(in)]
        Cw = Eyy - Ey[..., :, None] * Ey[..., None, :]                         # within-token covariance
        Jd = Jac.to(dt)
        Py = torch.einsum('ctia,ctasb,csjb->ctisj', Jd, P, Jd) if not (self.ln_pair and T > 1) else None
        if self.ln2nd and T > 1:
            # E[d2 y_i / du_a du_b] = -(d_ia E[u_b r^3] + d_ib E[u_a r^3] + d_ab E[u_i r^3])/d + 3 E[u_i u_a u_b r^5]/d^2
            wt5, ms5, Ss5 = res[5]
            Eur3 = (wt3[..., None] * ms3).sum(2)                                 # [C,T,d]
            Eu3r5 = (wt5[..., None, None, None] * (
                ms5[..., :, None, None] * ms5[..., None, :, None] * ms5[..., None, None, :]
                + ms5[..., :, None, None] * Ss5[..., None, :, :] + ms5[..., None, :, None] * Ss5[..., :, None, :]
                + ms5[..., None, None, :] * Ss5[..., :, :, None])).sum(2)        # [C,T,d,d,d]
            G2 = (-(I[None, None, :, :, None] * Eur3[:, :, None, None, :] + I[None, None, :, None, :] *
                    Eur3[:, :, None, :, None] + I[None, None, None, :, :] * Eur3[:, :, :, None, None]) / d
                  + 3 * Eu3r5 / d ** 2).to(dt)                                    # [C,T,i,a,b]
            X = torch.einsum('ctiab,ctasx,ctbsy->ctisxy', G2, P, P)
            Py = Py + 0.5 * torch.einsum('ctisxy,csjxy->ctisj', X, G2)
        if self.ln_pair and T > 1:
            # exact (for Gaussian u) cross-token second moments E[u_t u_t'^T r_t r_t'] by a 2-D Laplace integral:
            # r_t r_t' = pi^-1 int int (s s')^-1/2 e^{-s v_t - s' v_t'} ds ds', tilted joint Gaussian of (u_t, u_t').
            iu = torch.triu_indices(T, T, 1, device=self.dev)
            ta, tb = iu[0], iu[1]
            npair = ta.numel()
            Ptt = P.to(F64).permute(0, 1, 3, 2, 4)                                          # [C,T,T,d,d]
            SJ = torch.cat([torch.cat([Ptt[:, ta, ta], Ptt[:, ta, tb]], -1),
                            torch.cat([Ptt[:, tb, ta], Ptt[:, tb, tb]], -1)], -2)           # [C,npair,2d,2d]
            mJ = torch.cat([m64[:, ta], m64[:, tb]], -1)                                    # [C,npair,2d]
            x, w = _laguerre(-0.5, self.ln_pair_n)
            x = torch.tensor(x, dtype=F64, device=self.dev); w = torch.tensor(w, dtype=F64, device=self.dev)
            n_ = x.numel()
            ca, cb = Ev[:, ta], Ev[:, tb]                                                   # [C,npair]
            sa = x[None, None, :, None] / ca[..., None, None]                               # [C,npair,n,1]
            sb = x[None, None, None, :] / cb[..., None, None]                               # [C,npair,1,n]
            bvec = torch.cat([(2 * sa / d)[..., None].expand(C, npair, n_, n_, d),
                              (2 * sb / d)[..., None].expand(C, npair, n_, n_, d)], -1)    # [C,npair,n,n,2d]
            I2 = torch.eye(2 * d, dtype=F64, device=self.dev)
            Mx = I2 + SJ[:, :, None, None] * bvec[..., None, :]                             # I + Sigma B
            Minv = torch.linalg.inv(Mx)
            logdet = torch.linalg.slogdet(Mx)[1]
            mt = (Minv @ mJ[:, :, None, None, :, None])[..., 0]                            # tilted mean
            St = Minv @ SJ[:, :, None, None]                                               # tilted covariance
            quad = (mJ[:, :, None, None] * bvec * mt).sum(-1)
            logZ = -0.5 * logdet - 0.5 * quad - (sa + sb) * LN_EPS
            wt = (w[:, None] * w[None, :]) * torch.exp(logZ + x[:, None] + x[None, :]) / math.pi \
                / (ca * cb).sqrt()[..., None, None]                                         # [C,npair,n,n]
            Ecross = (wt[..., None, None] * (St[..., :d, d:] + mt[..., :d, None] * mt[..., None, d:])).sum((2, 3))
            Ccross = (Ecross - Ey[:, ta, :, None].to(F64) * Ey[:, tb, None, :].to(F64)).to(dt)   # [C,npair,d,d]
            Pcross = torch.zeros(C, T, T, d, d, dtype=dt, device=self.dev)
            Pcross = Pcross.index_put((torch.arange(C)[:, None], ta[None], tb[None]), Ccross)
            Pcross = Pcross.index_put((torch.arange(C)[:, None], tb[None], ta[None]), Ccross.transpose(-1, -2))
            Py = Pcross.permute(0, 1, 3, 2, 4)
        eyeT = torch.eye(T, dtype=dt, device=self.dev)
        Py = (Py * (1 - eyeT)[None, :, None, :, None] if T > 1 else 0) + torch.einsum('ctij,ts->ctisj', Cw.to(dt), eyeT)
        Ry = torch.einsum('ctia,ctar->ctir', Jd, R)
        Ay = torch.einsum('ctia,ctaj->ctij', Jd, A)
        Ey = Ey.to(dt)
        if self.K4:
            v3 = Ss3.diagonal(0, -2, -1); v4 = Ss4.diagonal(0, -2, -1)
            E3 = (wt3[..., None] * (ms3 ** 3 + 3 * ms3 * v3)).sum(2)
            E4 = (wt4[..., None] * (ms4 ** 4 + 6 * ms4 ** 2 * v4 + 3 * v4 ** 2)).sum(2)
            E1 = Ey.to(F64); E2 = Eyy.diagonal(0, -2, -1)
            V = E2 - E1 ** 2
            k3 = E3 - 3 * E1 * E2 + 2 * E1 ** 3
            k4 = E4 - 4 * E1 * E3 + 6 * E1 ** 2 * E2 - 3 * E1 ** 4 - 3 * V ** 2
            Ay = self.new_sources(Ay, k3.to(dt), k4.to(dt))
        if self.psd:
            Py = _psd(Py.reshape(C, T * d, T * d), self.psd_log).reshape(C, T, d, T, d)
        Ey = Ey * g + b
        Py = Py * g[None, None, :, None, None] * g[None, None, None, None, :]
        Ry = Ry * g[None, None, :, None]
        Ay = Ay * g[None, None, :, None]
        return Ey, Py, Ry, Ay

    def gelu_op(self, blk):
        m, P, R, A = blk
        C, T, n = m.shape
        Pf = P.reshape(C, T * n, T * n)
        V = torch.diagonal(Pf, 0, 1, 2).reshape(C, T, n)
        k3, k4 = self.marg(A)
        E, Vg, k3g, k4g, h1, h2, h3 = elementwise(gelu, m, V, k3, k4, self.K4)
        f = lambda h: h.reshape(C, T * n)
        H1, H2, H3 = f(h1), f(h2), f(h3)
        Pg = (H1[:, :, None] * H1[:, None] * Pf + H2[:, :, None] * H2[:, None] * Pf ** 2 / 2
              + H3[:, :, None] * H3[:, None] * Pf ** 3 / 6)
        Pg = Pg - torch.diag_embed(torch.diagonal(Pg, 0, 1, 2)) + torch.diag_embed(f(Vg))
        Rg = h1[..., None] * R
        Ag = h1[..., None] * self.pad(A)
        if self.K4:
            t3, t4 = self.marg(Ag)
            Ag = self.new_sources(Ag, k3g - t3, k4g - t4)
        return E, Pg.reshape(C, T, n, T, n), Rg, Ag

    def attention(self, blk, Wqkv, bqkv, Wo, bo, H, allow):
        m, P, R, A = blk
        C, T, d = m.shape
        D = d // H; sc = 1 / math.sqrt(D)
        mz, Pz, Rz, Az = self.linear_tok((m, P, R, self.pad(A)), Wqkv, bqkv)
        mz = mz.reshape(C, T, 3, H, D)
        Pz = Pz.reshape(C, T, 3, H, D, T, 3, H, D)
        Rz = Rz.reshape(C, T, 3, H, D, -1); Az = Az.reshape(C, T, 3, H, D, -1)
        mq, mk, mv = mz[:, :, 0], mz[:, :, 1], mz[:, :, 2]
        Pqq, Pkk, Pvv = Pz[:, :, 0, :, :, :, 0], Pz[:, :, 1, :, :, :, 1], Pz[:, :, 2, :, :, :, 2]
        Pqk, Pqv, Pkv = Pz[:, :, 0, :, :, :, 1], Pz[:, :, 0, :, :, :, 2], Pz[:, :, 1, :, :, :, 2]
        Pkq = Pqk.permute(0, 4, 5, 6, 1, 2, 3)
        Rq, Rk, Rv = Rz[:, :, 0], Rz[:, :, 1], Rz[:, :, 2]
        Aq, Ak, Av = Az[:, :, 0], Az[:, :, 1], Az[:, :, 2]
        ein = torch.einsum
        # ---- scores s[t,s,h]
        trqk = torch.diagonal(torch.diagonal(Pqk, 0, 2, 5), 0, 2, 4).sum(-1)        # [C,T,S,H]
        ms = sc * (ein('ctha,csha->ctsh', mq, mk) + trqk)
        Pss = sc ** 2 * (ein('ctha,cugb,cshawgb->ctshuwg', mq, mq, Pkk)
                         + ein('ctha,cwgb,cshaugb->ctshuwg', mq, mk, Pkq)
                         + ein('csha,cugb,cthawgb->ctshuwg', mk, mq, Pqk)
                         + ein('csha,cwgb,cthaugb->ctshuwg', mk, mk, Pqq)
                         + ein('cthaugb,cshawgb->ctshuwg', Pqq, Pkk)
                         + ein('cthawgb,cshaugb->ctshuwg', Pqk, Pkq))
        Rs = sc * (ein('csha,cthar->ctshr', mk, Rq) + ein('ctha,cshar->ctshr', mq, Rk))
        Psv = sc * (ein('csha,cthaugb->ctshugb', mk, Pqv) + ein('ctha,cshaugb->ctshugb', mq, Pkv))
        As = sc * (ein('csha,cthaj->ctshj', mk, Aq) + ein('ctha,cshaj->ctshj', mq, Ak))
        if self.K4:
            dq = torch.diagonal(torch.diagonal(Pqq, 0, 1, 4), 0, 1, 3)              # [C,a,b,T,H]
            dq = dq.permute(0, 3, 4, 1, 2)                                            # [C,T,H,a,b]
            dk = torch.diagonal(torch.diagonal(Pkk, 0, 1, 4), 0, 1, 3).permute(0, 3, 4, 1, 2)
            xqk = torch.diagonal(Pqk, 0, 2, 5).permute(0, 1, 3, 5, 2, 4)              # [C,T,S,H,a,b]
            Sg = torch.cat([torch.cat([dq[:, :, None].expand(C, T, T, H, D, D), xqk], -1),
                            torch.cat([xqk.transpose(-1, -2), dk[:, None].expand(C, T, T, H, D, D)], -1)], -2)
            mu2 = torch.cat([mq[:, :, None].expand(C, T, T, H, D), mk[:, None].expand(C, T, T, H, D)], -1)
            I = torch.eye(D, dtype=self.dt, device=self.dev); Z = torch.zeros_like(I)
            Bm = sc * 0.5 * torch.cat([torch.cat([Z, I], 1), torch.cat([I, Z], 1)], 0)
            _, _, q3, q4 = qf_cumulants(Sg, mu2, Bm)
            As = self.new_sources(As, q3, q4)
        # ---- softmax as a[t,s,h] = sigmoid(-l[t,s,h]), l = log sum_{w != s} exp(s_tw - s_ts)
        # l is closed by Fenton-Wilkinson (moment-matched lognormal) on the exactly-Gaussian differences;
        # cross-covariances of l from E[D D'] (joint lognormal); the sigmoid via Gauss-Hermite (Edgeworth if K=4).
        self.dbg = dict(ms=ms, Pss=Pss) if self.debug else {}
        # Clark-style pairwise reduction Z <- Z + softplus(s_tw - Z) over keys w != s (1-D Gauss-Hermite per step,
        # running Z kept jointly Gaussian with all scores/reference/values; joint covariance of all running Z's).
        n_s = T * T * H
        Pf = Pss.reshape(C, n_s, n_s)
        Vs = torch.diagonal(Pf, 0, 1, 2).reshape(C, T, T, H)
        mZ = torch.zeros(C, T, T, H, dtype=self.dt, device=self.dev)
        cZ = torch.zeros(C, T, T, H, T, T, H, dtype=self.dt, device=self.dev)
        KZ = torch.zeros(C, n_s, n_s, dtype=self.dt, device=self.dev)
        started = torch.zeros(T, T, dtype=torch.bool, device=self.dev)
        bZ = torch.zeros(C, T, T, H, T, dtype=self.dt, device=self.dev)
        eyeT = torch.eye(T, dtype=torch.bool, device=self.dev)
        Asp = self.pad(As) if self.K4 else None
        for w in range(T):
            valid = allow[:, w][:, None] & allow & ~eyeT[w][None, :]                   # [T(t),S(s)] update at key w
            st_ = (valid & started)[None, :, :, None]; init = (valid & ~started)[None, :, :, None]
            ms_w = ms[:, :, w, :][:, :, None, :]                                      # [C,T,1,H]
            Vs_w = Vs[:, :, w, :][:, :, None, :]
            cZx = torch.diagonal(torch.diagonal(cZ[:, :, :, :, :, w, :], 0, 1, 4), 0, 2, 3).permute(0, 2, 1, 3)  # [C,T,S,H]
            mdl = ms_w - mZ
            Vdl = (Vs_w + KZ.diagonal(0, 1, 2).reshape(C, T, T, H) - 2 * cZx).clamp_min(1e-12)
            Esp, Vsp, _, _, g1, g2, g3 = elementwise(Fn.softplus, mdl, Vdl, None, None, False)
            p = torch.where(st_, g1, torch.where(init, torch.ones_like(g1), torch.zeros_like(g1)))
            G1 = p
            G2 = torch.where(st_, g2, torch.zeros_like(g2)); G3 = torch.where(st_, g3, torch.zeros_like(g3))
            Ve = torch.where(st_, Vsp, torch.where(init, Vs_w.expand_as(Vsp), torch.zeros_like(Vsp)))
            # covariance of each running Z^a's incoming score s_{x_a} with everything
            Csx = Pss[:, :, w][:, :, None].expand(C, T, T, H, T, T, H)                 # Cov(s_{t w h}, score x)
            # Cov(delta^a, delta^b) = Css(x_a,x_b) - cZ^b(x_a) - cZ^a(x_b) + K_ab
            cZ_f = cZ.reshape(C, n_s, n_s)
            sel = self._sel(T, H, w)                                                   # x_a = (t, w, h)
            Cxx = sel @ Pf @ sel.T if False else torch.einsum('ax,cxy,by->cab', sel, Pf, sel)
            cZb_xa = torch.einsum('cbx,ax->cab', cZ_f, sel)                            # cZ^b(x_a) at [a,b]
            Cdd = Cxx - cZb_xa - cZb_xa.transpose(1, 2) + KZ
            f = lambda z: z.reshape(C, n_s)
            H1, H2, H3 = f(G1), f(G2), f(G3)
            Cee = (H1[:, :, None] * H1[:, None] * Cdd + H2[:, :, None] * H2[:, None] * Cdd ** 2 / 2
                   + H3[:, :, None] * H3[:, None] * Cdd ** 3 / 6)
            Cee = Cee - torch.diag_embed(torch.diagonal(Cee, 0, 1, 2)) + torch.diag_embed(f(Ve))
            # Cov(Z^a, e^b) = p_b (cZ^a(x_b) - K_ab)
            CZe = H1[:, None, :] * (cZb_xa.transpose(1, 2) - KZ)
            KZ = KZ + CZe + CZe.transpose(1, 2) + Cee
            mZ = torch.where(st_, mZ + Esp, torch.where(init, ms_w.expand_as(mZ), mZ))
            pe = p[..., None]
            cZ = (1 - p)[..., None, None, None] * cZ + p[..., None, None, None] * Csx
            oh = torch.zeros(T, dtype=self.dt, device=self.dev); oh[w] = 1.0
            bZ = (1 - p)[..., None] * bZ + p[..., None] * oh
            started = started | valid
        rowok = started[None]                                                          # [1,T,S]
        # the cross-covariances with the reference / values and the factors follow the same linear recursion as
        # beta, so they are formed once from beta (identical result, far less memory than updating them per step)
        rZ = torch.einsum('ctshw,ctwhr->ctshr', bZ, Rs)
        vZ = torch.einsum('ctshw,ctwhugb->ctshugb', bZ, Psv)
        AZ = torch.einsum('ctshw,ctwhj->ctshj', bZ, Asp) if self.K4 else None
        selfsel = torch.eye(n_s, dtype=self.dt, device=self.dev)
        cZ_f = cZ.reshape(C, n_s, n_s)
        ml = mZ - ms
        Pll = KZ - cZ_f - cZ_f.transpose(1, 2) + Pf
        Rl = rZ - Rs
        Plv = vZ - Psv
        bl = bZ - torch.eye(T, dtype=self.dt, device=self.dev)[None, None, :, None, :].permute(0, 1, 2, 3, 4)
        Al = (AZ - Asp) if self.K4 else self.pad(As)
        Pll = Pll.reshape(C, T, T, H, T, T, H)
        Vl = torch.diagonal(Pll.reshape(C, n_s, n_s), 0, 1, 2).reshape(C, T, T, H).clamp_min(1e-12)
        k3l, k4l = self.marg(Al)
        Ea, Va, k3a, k4a, h1, h2, h3 = elementwise(_signeg, ml, Vl, k3l, k4l, self.K4)
        al = (allow[None, :, :, None] & rowok[..., None]).to(self.dt)                 # [1,T,S,1]
        f = lambda z: z.reshape(C, n_s)
        H1, H2, H3 = f(h1 * al), f(h2 * al), f(h3 * al)
        Pllf = Pll.reshape(C, n_s, n_s)
        Paa = (H1[:, :, None] * H1[:, None] * Pllf + H2[:, :, None] * H2[:, None] * Pllf ** 2 / 2
               + H3[:, :, None] * H3[:, None] * Pllf ** 3 / 6)
        Paa = Paa - torch.diag_embed(torch.diagonal(Paa, 0, 1, 2)) + torch.diag_embed(f(Va * al))
        Paa = Paa.reshape(C, T, T, H, T, T, H)
        Ea = Ea * al
        Ra = (h1 * al)[..., None] * Rl
        Pav = (h1 * al)[..., None, None, None] * Plv
        Aa = (h1 * al)[..., None] * Al
        if self.rowproj:
            # exact constraint sum_s a_ts = 1: orthogonal projection of means and of every covariance/factor row
            nk = al.sum(2, keepdim=True)                                                # [1,T,1,1]
            Ea = Ea + (1 - Ea.sum(2, keepdim=True)) / nk * al
            proj = lambda X, dim: X - al.view(al.shape + (1,) * (X.dim() - 4)) * X.sum(dim, keepdim=True) / nk.view(nk.shape + (1,) * (X.dim() - 4))
            Ra = proj(Ra, 2); Pav = proj(Pav, 2); Aa = proj(Aa, 2)
            Paa = proj(Paa, 2)
            Paa = Paa.permute(0, 4, 5, 6, 1, 2, 3)
            Paa = proj(Paa, 2).permute(0, 4, 5, 6, 1, 2, 3)
        if self.K4:
            t3, t4 = self.marg(Aa)
            Aa = self.new_sources(Aa, (k3a - t3) * al, (k4a - t4) * al)
        if self.debug: self.dbg.update(Ea=Ea, Paa=Paa, ml=ml, Pll=Pll)
        # ---- o[t,h,b] = sum_s a[t,s,h] v[s,h,b]
        dav = torch.diagonal(torch.diagonal(Pav, 0, 2, 4), 0, 2, 3)                  # [C,T,B,S,H]: Cov(a_tsh, v_shb)
        mo = ein('ctsh,cshb->cthb', Ea, mv) + dav.sum(3).permute(0, 1, 3, 2)
        Pva = Pav.permute(0, 4, 5, 6, 1, 2, 3)                                        # Cov(v_shb, a_uwg)
        Po = (ein('ctsh,cuwg,cshbwge->cthbuge', Ea, Ea, Pvv)
              + ein('cshb,cwge,ctshuwg->cthbuge', mv, mv, Paa)
              + ein('ctsh,cwge,cshbuwg->cthbuge', Ea, mv, Pva)
              + ein('cshb,cuwg,ctshwge->cthbuge', mv, Ea, Pav)
              + ein('ctshuwg,cshbwge->cthbuge', Paa, Pvv)
              + ein('ctshwge,cshbuwg->cthbuge', Pav, Pva))
        Ro = ein('ctsh,cshbr->cthbr', Ea, Rv) + ein('cshb,ctshr->cthbr', mv, Ra)
        Ao = ein('ctsh,cshbj->cthbj', Ea, self.pad(Av)) + ein('cshb,ctshj->cthbj', mv, self.pad(Aa))
        if self.K4:
            Caa_t = torch.diagonal(torch.diagonal(Paa, 0, 1, 4), 0, 2, 4)            # [C,S,W,T,H]
            Caa_t = Caa_t.permute(0, 3, 4, 1, 2)                                       # [C,T,H,S,W]
            Cvv_b = torch.diagonal(torch.diagonal(Pvv, 0, 2, 5), 0, 2, 4)             # [C,S,W,H,D]
            Cvv_b = Cvv_b.permute(0, 3, 4, 1, 2)                                       # [C,H,D,S,W]
            Cav_b = torch.diagonal(Pav, 0, 3, 5)                                       # [C,T,S,W,D,H]
            Cav_b = Cav_b.permute(0, 1, 5, 4, 2, 3)                                    # [C,T,H,D,S,W]
            SA = Caa_t[:, :, :, None].expand(C, T, H, D, T, T)
            SV = Cvv_b[:, None].expand(C, T, H, D, T, T)
            Sg = torch.cat([torch.cat([SA, Cav_b], -1), torch.cat([Cav_b.transpose(-1, -2), SV], -1)], -2)
            mu2 = torch.cat([Ea.permute(0, 1, 3, 2)[:, :, :, None].expand(C, T, H, D, T),
                             mv.permute(0, 2, 3, 1)[:, None].expand(C, T, H, D, T)], -1)
            I = torch.eye(T, dtype=self.dt, device=self.dev); Z = torch.zeros_like(I)
            Bm = 0.5 * torch.cat([torch.cat([Z, I], 1), torch.cat([I, Z], 1)], 0)
            _, _, q3, q4 = qf_cumulants(Sg, mu2, Bm)
            Ao = self.new_sources(Ao, q3, q4)
        if self.av_stein:
            # (l, v) are jointly Gaussian in the closure and a = f(l) = sigmoid(-l): moments of o = sum_s f(l_ts) v_s are
            # computed exactly for that Gaussian by Stein's lemma twice + bivariate Gauss-Hermite for E[f^(i)(l) f^(j)(l')].
            F0, F1, F2, G = _bivar_sigmoid(ml, Pll.reshape(C, n_s, n_s), self.gh2)
            alf = al.expand(C, T, T, H).reshape(C, n_s)
            mk2 = alf[:, :, None] * alf[:, None, :]
            G = {k_: (v_ * mk2).reshape(C, T, T, H, T, T, H).to(self.dt) for k_, v_ in G.items()}
            F0, F1, F2 = [(x_.reshape(C, T, T, H) * al).to(self.dt) for x_ in (F0, F1, F2)]
            c1 = torch.diagonal(torch.diagonal(Plv, 0, 2, 4), 0, 2, 3).permute(0, 1, 3, 4, 2)   # [C,T,S,H,B] Cov(v_shb, l_tsh)
            mo = ein('ctsh,cshb->cthb', F0, mv) + ein('ctsh,ctshb->cthb', F1, c1)
            Po = (ein('cshb,cwge,ctshuwg->cthbuge', mv, mv, G['00'])
                  + ein('cshb,ctshwge,ctshuwg->cthbuge', mv, Plv, G['10f'])
                  + ein('cshb,cuwge,ctshuwg->cthbuge', mv, c1, G['01'])
                  + ein('ctshb,cwge,ctshuwg->cthbuge', c1, mv, G['10'])
                  + ein('ctshb,ctshwge,ctshuwg->cthbuge', c1, Plv, G['20f'])
                  + ein('ctshb,cuwge,ctshuwg->cthbuge', c1, c1, G['11'])
                  + ein('cuwgshb,cwge,ctshuwg->cthbuge', Plv, mv, G['01f'])
                  + ein('cuwgshb,ctshwge,ctshuwg->cthbuge', Plv, Plv, G['11f'])
                  + ein('cuwgshb,cuwge,ctshuwg->cthbuge', Plv, c1, G['02f'])
                  + ein('cshbwge,ctshuwg->cthbuge', Pvv, G['00f']))
            if self.av_k3:
                # kappa3(v_x, l_b, v_y) = sum_sigma beta[b,sigma] kappa3(v_x, s_sigma, v_y), with the exact Wick form for
                # bilinear scores: kappa3(v_x, q.k, v_y) = sum_a Cov(v_x,q_a)Cov(k_a,v_y) + Cov(v_x,k_a)Cov(q_a,v_y)
                Pkv_ = Pkv; Pqv_ = Pqv                                                     # [C,T,H,A,U,G,B]
                PkvD = torch.diagonal(Pkv_, 0, 2, 5).permute(0, 1, 5, 2, 3, 4)            # [C,Z,G,A,W,E]
                PqvD = torch.diagonal(Pqv_, 0, 2, 5).permute(0, 1, 5, 2, 3, 4)            # [C,U,G,A,W,E]
                blz = bl.permute(0, 1, 2, 3, 4)                                            # [C,U,W,G,Z]
                M1 = ein('cuwgz,czgawe->cuwgae', blz, PkvD)
                K3 = ein('cugashb,cuwgae->cshbuwge', Pqv_, M1)
                M2 = ein('cuwgz,czgashb->cuwgashb', blz, Pkv_)
                K3 = sc * (K3 + ein('cuwgashb,cugawe->cshbuwge', M2, PqvD))              # [C,S,H,B,U,W,G,E]
                Tk = ein('ctsh,cuwg,cshbuwge->cthbuge', F0, F1, K3)
                Po = Po + Tk + Tk.permute(0, 4, 5, 6, 1, 2, 3)
                if self.debug: self.dbg["Tk"] = Tk + Tk.permute(0, 4, 5, 6, 1, 2, 3)
            Ro = (ein('ctsh,cshb,ctshr->cthbr', F1, mv, Rl) + ein('ctsh,ctshb,ctshr->cthbr', F2, c1, Rl)
                  + ein('ctsh,cshbr->cthbr', F0, Rv))
            if self.av_k3:
                # Cov(o, r) gains sum_s E[f'] kappa3(l_tsh, v_shb, r), kappa3(l, v, r) = sum_z beta[tsh,z] kappa3(s_tzh, v, r),
                # kappa3(s_tzh, v, r) = sc * sum_a [Cov(q_tha, v) Cov(k_zha, r) + Cov(k_zha, v) Cov(q_tha, r)]
                PqvD2 = torch.diagonal(Pqv, 0, 2, 5)                                      # [C,T,A,S,B,H]
                PkvD2 = torch.diagonal(Pkv, 0, 2, 5)                                      # [C,Z,A,S,B,H]
                Fb = F1[..., None] * bl                                                   # [C,T,S,H,Z]
                Wq = ein('ctshz,ctasbh->cthabz', Fb, PqvD2)
                Vk = ein('ctshz,czasbh->cthab', Fb, PkvD2)

                def k3ref(Rk_, Rq_):
                    return sc * (ein('cthabz,czhar->cthbr', Wq, Rk_) + ein('cthab,cthar->cthbr', Vk, Rq_))
            Alp = self.pad(Al)
            Ao = (ein('ctsh,cshb,ctshj->cthbj', F1, mv, Alp) + ein('ctsh,ctshb,ctshj->cthbj', F2, c1, Alp)
                  + ein('ctsh,cshbj->cthbj', F0, self.pad(Av)))
            if self.av_k3:
                Ro = Ro + k3ref(Rk, Rq)
                if self.K4:
                    Ao = Ao + k3ref(self.pad(Ak), self.pad(Aq))
            if self.K4:
                Ao = Ao + ein('cshb,ctshj->cthbj', mv, self.pad(Aa) - self.pad((h1 * al)[..., None] * Al))
        if self.debug: self.dbg.update(mo=mo, Po=Po)
        # ---- output projection
        mo = mo.reshape(C, T, d); Po = Po.reshape(C, T, d, T, d)
        Ro = Ro.reshape(C, T, d, -1); Ao = self.pad(Ao).reshape(C, T, d, -1)
        return self.linear_tok((mo, Po, Ro, Ao), Wo, bo)

    # ------------------------------------------------------------ residual state
    def add_branch(self, st, blk):
        m, P, A = st
        mb, Pb, Rb, Ab = blk
        C, T, d = mb.shape
        N = T * d
        mb, Pb, Rb, Ab = mb.reshape(C, N), Pb.reshape(C, N, N), Rb.reshape(C, N, -1), self.pad(Ab).reshape(C, N, -1)
        m = m.clone(); m[:, :N] = m[:, :N] + mb
        Rf = torch.cat([Rb, torch.zeros(C, 1, Rb.shape[-1], dtype=self.dt, device=self.dev)], 1)
        Pn = P + Rf + Rf.transpose(1, 2)
        Pn = Pn.clone(); Pn[:, :N, :N] = Pn[:, :N, :N] + Pb
        A = self.pad(A).clone()
        A[:, :N] = A[:, :N] + Ab
        if self.psd:
            Pn = _psd(Pn, self.psd_log)
        return m, Pn, A

    def res_block(self, st, T, d):
        """residual stream as a block (m [C,T,d], P, R, A)."""
        m, P, A = st
        C = m.shape[0]; N = T * d
        return (m[:, :N].reshape(C, T, d), P[:, :N, :N].reshape(C, T, d, T, d),
                P[:, :N, :].reshape(C, T, d, -1), self.pad(A)[:, :N].reshape(C, T, d, -1))

    # ------------------------------------------------------------ full run
    def run(self, model, upto=None, want_points=True):
        dt, dev = self.dt, self.dev
        C = self.m0.shape[0]
        self._reset(C)
        T, Fd, d, H = W.T, W.F, model.d, model.h
        N = T * d
        Win, bin_ = model.inp.weight.to(dt), model.inp.bias.to(dt)
        M = torch.zeros(N + 1, T * Fd + 1, dtype=dt, device=dev)
        for t in range(T):
            M[t * d:(t + 1) * d, t * Fd:(t + 1) * Fd] = Win
        M[N, T * Fd] = 1.0
        bb = torch.cat([bin_.repeat(T), torch.zeros(1, dtype=dt, device=dev)])
        m = self.m0 @ M.T + bb
        P = M @ self.C0 @ M.T
        A = torch.zeros(C, N + 1, 0, dtype=dt, device=dev)
        st = (m, P, A)
        points = [self.point_stats(st, N)] if want_points else []
        allow = model.allow
        blocks = model.blocks if upto is None else model.blocks[:upto]
        def block_fn(blk, m_, P_, A_, c3_, c4_):
            self.c3 = [c3_] if c3_.shape[1] else []
            self.c4 = [c4_] if c4_.shape[1] else []
            self.J = c3_.shape[1]
            st_ = (m_, P_, A_)
            p = lambda t: t.to(dt)
            y = self.layernorm(self.res_block(st_, T, d), p(blk.ln1.weight), p(blk.ln1.bias))
            att = self.attention(y, p(blk.qkv.weight), p(blk.qkv.bias), p(blk.o.weight), p(blk.o.bias), H, allow)
            st1 = self.add_branch(st_, att)
            y2 = self.layernorm(self.res_block(st1, T, d), p(blk.ln2.weight), p(blk.ln2.bias))
            h = self.linear_tok(y2, p(blk.fc1.weight), p(blk.fc1.bias))
            g = self.gelu_op(h)
            mm = self.linear_tok(g, p(blk.fc2.weight), p(blk.fc2.bias))
            st2 = self.add_branch(st1, mm)
            c3n, c4n = self.cs()
            return st1[0], st1[1], self.pad(st1[2]), st2[0], st2[1], self.pad(st2[2]), c3n, c4n

        for blk in blocks:
            c3_, c4_ = self.cs()
            args = (st[0], st[1], self.pad(st[2]), c3_, c4_)
            if self.checkpoint and torch.is_grad_enabled():
                out_ = torch.utils.checkpoint.checkpoint(block_fn, blk, *args, use_reentrant=False)
            else:
                out_ = block_fn(blk, *args)
            m1, P1, A1, m2, P2, A2, c3n, c4n = out_
            self.c3 = [c3n] if c3n.shape[1] else []
            self.c4 = [c4n] if c4n.shape[1] else []
            self.J = c3n.shape[1]
            st1, st = (m1, P1, A1), (m2, P2, A2)
            if want_points:
                points.append(self.point_stats(st1, N)); points.append(self.point_stats(st, N))
        # final LN on the query token, head
        mres, Pres, Rres, Ares = self.res_block(st, T, d)
        q = (mres[:, -1:], Pres[:, -1:, :, -1:], Rres[:, -1:], Ares[:, -1:])
        z = self.layernorm(q, model.lnf.weight.to(dt), model.lnf.bias.to(dt))
        mo, Po, Ro, Ao = self.linear_tok(z, model.head.weight.to(dt), model.head.bias.to(dt))
        mo, Po, Ro, Ao = mo[:, 0], Po[:, 0, :, 0], Ro[:, 0], self.pad(Ao)[:, 0]
        out = self.readout(st, mo, Po, Ro, Ao, N)
        return dict(points=points, out=out)

    def readout(self, st, mo, Po, Ro, Ao, N):
        m, P, A = st
        my, Vy = m[:, N], P[:, N, N]
        mmu, mls = mo[:, 0], mo[:, 1]
        Vmu, Vls, Cml = Po[:, 0, 0], Po[:, 1, 1], Po[:, 0, 1]
        Cym, Cyl = Ro[:, 0, N], Ro[:, 1, N]
        mw = my - mmu; Vw = Vy + Vmu - 2 * Cym; Cwl = Cyl - Cml
        t = -2.0
        C0 = t * mls + t * t * Vls / 2
        Cs = mw + t * Cwl
        Css = Vw
        k3mu = k4mu = k3ls = k4ls = torch.zeros_like(mmu)
        if self.K4 and self.J:
            c3, c4 = self.cs()
            aw, al = -Ao[:, 0], Ao[:, 1]
            C0 = C0 + (c3 * (t * al) ** 3).sum(-1) / 6 + (c4 * (t * al) ** 4).sum(-1) / 24
            Cs = Cs + (c3 * t ** 2 * al ** 2 * aw).sum(-1) / 2 + (c4 * t ** 3 * al ** 3 * aw).sum(-1) / 6
            Css = Css + (c3 * t * aw ** 2 * al).sum(-1) + (c4 * t ** 2 * aw ** 2 * al ** 2).sum(-1) / 2
            k3mu, k4mu = (c3 * Ao[:, 0] ** 3).sum(-1), (c4 * Ao[:, 0] ** 4).sum(-1)
            k3ls, k4ls = (c3 * al ** 3).sum(-1), (c4 * al ** 4).sum(-1)
        Enll = mls + 0.5 * math.log(2 * math.pi) + 0.5 * (Cs ** 2 + Css) * torch.exp(C0)
        p = self.p.to(mmu.dtype)
        mix = lambda x: (p * x).sum()
        Emu, Els = mix(mmu), mix(mls)
        return dict(E_mu=Emu, E_logs=Els, E_nll=mix(Enll),
                    V_mu=mix(Vmu + mmu ** 2) - Emu ** 2, V_logs=mix(Vls + mls ** 2) - Els ** 2,
                    comp=dict(mu=mmu, logs=mls, nll=Enll, Vmu=Vmu, Vls=Vls, k3mu=k3mu, k4mu=k4mu))

    # ------------------------------------------------------------ mixture statistics of the residual stream
    def point_stats(self, st, N):
        m, P, A = st
        m, P = m[:, :N], P[:, :N, :N]
        k3, k4 = self.marg(A[:, :N]) if self.J else (torch.zeros_like(m), torch.zeros_like(m))
        return self.mixture(m, P, k3, k4)

    def mixture(self, m, P, k3, k4):
        """Mix component stats over permutation orbits and shapes. Returns mean, cov, marginal k3, k4."""
        C, N = m.shape
        d = N // W.T
        idxs = []
        for pm in self.perms:
            tok = list(range(W.T))
            for s in range(W.S):
                for q in range(W.Q):
                    tok[s * W.Q + q] = pm[s] * W.Q + q
            idx = (torch.tensor(tok)[:, None] * d + torch.arange(d)[None]).reshape(-1)
            idxs.append(idx.to(m.device))
        np_ = len(idxs)
        p = self.p.to(m.dtype)
        mu = sum((p[:, None] * m[:, ix]).sum(0) for ix in idxs) / np_
        S2 = sum((p[:, None, None] * (P[:, ix][:, :, ix] + m[:, ix, None] * m[:, None, ix])).sum(0) for ix in idxs) / np_
        cov = S2 - mu[:, None] * mu[None]
        var_c = torch.diagonal(P, 0, 1, 2)
        # raw marginal moments per component, then mix (float64)
        mm, vv, a3, a4 = m.to(F64), var_c.to(F64), k3.to(F64), k4.to(F64)
        r1, r2 = mm, vv + mm ** 2
        r3 = a3 + 3 * mm * vv + mm ** 3
        r4 = a4 + 4 * mm * a3 + 3 * vv ** 2 + 6 * mm ** 2 * vv + mm ** 4
        R = [sum((self.p[:, None] * r[:, ix]).sum(0) for ix in idxs) / np_ for r in (r1, r2, r3, r4)]
        M1 = R[0]
        c2 = R[1] - M1 ** 2
        c3 = R[2] - 3 * M1 * R[1] + 2 * M1 ** 3
        c4m = R[3] - 4 * M1 * R[2] + 6 * M1 ** 2 * R[1] - 3 * M1 ** 4    # 4th central moment
        k4m = c4m - 3 * c2 ** 2
        return dict(mean=mu, cov=cov, k3=c3.to(m.dtype), k4=k4m.to(m.dtype))


def empirical_stats(x):
    """x [B, N] -> mean, cov, marginal k3, k4 (biased plug-in estimators; differentiable)."""
    mu = x.mean(0)
    xc = x - mu
    B = x.shape[0]
    cov = xc.T @ xc / B
    v = (xc ** 2).mean(0)
    k3 = (xc ** 3).mean(0)
    k4 = (xc ** 4).mean(0) - 3 * v ** 2
    return dict(mean=mu, cov=cov, k3=k3, k4=k4)
