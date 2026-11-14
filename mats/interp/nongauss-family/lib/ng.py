"""nongauss-family lane (29 Oct 2026): oracle families at a cut, exact propagation, re-projection (composition).
v2 (after the GPT-6 design critique): sphere chart, weighted fits, PEMP and PBk families, family-specific sampling
floors.

Nets: ARC's He-ReLU family as in work/arc-noisy/anet.py (copied in lib/anet.py, lib/bf.py). Layer indices are 1-based:
z_l = W_l h_{l-1} (pre-activation), a_l = ReLU(z_l), h_l = a_l + s_l eps (s_l = sigma r_l; net.s[l-1]).
Families are fitted to samples X (N, n) of z_l and expose .sample(b, gen) -> (b, n) float32 samples of z_l.

Polar (sphere) chart: z = rho w, rho = |z|, w = z / |z| (defined for every z != 0). For a degree-1 positively
homogeneous f, E f(z) = E[rho] E_nu f(w) with nu the rho-weighted law of w. Polar families fit nu with weights
proportional to rho and draw rho independently from its unweighted empirical law (exact for noiseless homogeneous
observables; the noise breaks homogeneity slightly, measured by PEMP against EMP).
Every family is fully specified here; device-agnostic (CUDA if available)."""
import math
import torch

F32 = torch.float32
F64 = torch.float64
CH = 1 << 16


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DNet:
    """A net's weights and noise scales on a device (float32 weights)."""

    def __init__(self, net, dev):
        self.W = [w.to(device=dev, dtype=F32) for w in net.W]
        self.s = [float(v) for v in net.s]
        self.n, self.L, self.dev = net.n, net.L, dev


@torch.no_grad()
def truth_moments(dn, N, seed, chunk=CH):
    """Plain Monte Carlo: E[a_l], E[a_l^2] for l = 1..L (float64, (L, n))."""
    g = torch.Generator(device=dn.dev).manual_seed(seed)
    S1 = torch.zeros(dn.L, dn.n, dtype=F64, device=dn.dev); S2 = torch.zeros_like(S1)
    done = 0
    while done < N:
        b = min(chunk, N - done)
        h = torch.randn(b, dn.n, generator=g, device=dn.dev)
        for k in range(1, dn.L + 1):
            a = torch.relu(h @ dn.W[k - 1].T)
            ad = a.double(); S1[k - 1] += ad.sum(0); S2[k - 1] += (ad * ad).sum(0)
            if k < dn.L:
                h = a + dn.s[k - 1] * torch.randn(b, dn.n, generator=g, device=dn.dev)
        done += b
    return S1 / N, S2 / N


# ------------------------------------------------------------------------------------------------ statistics
def wsample(cw, shape, g):
    """indices drawn with probabilities proportional to the weights whose cumulative sum is cw (float64)."""
    u = torch.rand(shape, generator=g, device=cw.device, dtype=F64) * cw[-1]
    return torch.searchsorted(cw, u).clamp_max(cw.shape[0] - 1)


@torch.no_grad()
def mean_cov(X, w=None, chunk=CH):
    """float64 (weighted) mean and covariance of float32 X (N, n); w (N,) nonnegative weights or None."""
    N, n = X.shape
    c0 = X[:min(N, 4096)].double().mean(0)
    S0 = 0.0
    S1 = torch.zeros(n, dtype=F64, device=X.device); S2 = torch.zeros(n, n, dtype=F64, device=X.device)
    for a in range(0, N, chunk):
        x = X[a:a + chunk].double() - c0
        if w is None:
            S0 += x.shape[0]; S1 += x.sum(0); S2 += x.T @ x
        else:
            ww = w[a:a + chunk].double()
            S0 += float(ww.sum()); S1 += (ww[:, None] * x).sum(0); S2 += (x * ww[:, None]).T @ x
    m = S1 / S0
    C = S2 / S0 - torch.outer(m, m)
    if w is None:
        C = C * S0 / max(S0 - 1, 1)
    return m + c0, C


def eig_desc(C):
    ev, V = torch.linalg.eigh(C)
    return ev.flip(0).clamp_min(0), V.flip(1)


def wquantile(y, w, qs):
    """weighted quantiles of y (N,) with weights w (N,), at levels qs (k,)."""
    o = torch.argsort(y)
    cw = torch.cumsum(w[o].double(), 0); cw = cw / cw[-1]
    i = torch.searchsorted(cw, qs.double().to(y.device)).clamp_max(y.shape[0] - 1)
    return y[o][i]


@torch.no_grad()
def sphere(X, chunk=CH):
    """rho = |z| (float32, (N,)) and w = z / |z| (float32, (N, n))."""
    rho = torch.cat([X[a:a + chunk].double().norm(dim=1).float() for a in range(0, X.shape[0], chunk)])
    return rho, X / rho[:, None]


# ------------------------------------------------------------------------------------------------ families
class Emp:
    """Resample of the fit set, optionally weighted (structured sampling; reference floor only)."""

    def __init__(self, X, w=None):
        self.X = X
        self.cw = None if w is None else torch.cumsum(w.double(), 0)

    def sample(self, b, g):
        if self.cw is None:
            i = torch.randint(0, self.X.shape[0], (b,), generator=g, device=self.X.device)
        else:
            i = wsample(self.cw, (b,), g)
        return self.X[i]


class Lin:
    """x = m + V e, V the eigenvectors of the (weighted) covariance of X, components e_k drawn by `shape`:
    'gauss' : e_k ~ N(0, ev_k), independent                              (Gaussian; control)
    'rad'   : e_k = +-sqrt(ev_k), independent (two-point: symmetric, excess kurtosis -2)
    'emp'   : e_k independent, each with the exact (weighted) empirical law of the k-th PC score of X
    'blockK': the top-K scores drawn jointly (one fit-set row, weighted), the other components independent as in 'emp'
    All but 'gauss' are fully non-Gaussian. Second moments are exact up to sampling for every shape."""

    def __init__(self, X, shape, w=None):
        self.shape = shape
        self.m, C = mean_cov(X, w)
        self.ev, self.V = eig_desc(C)
        self.mf, self.Vf = self.m.float(), self.V.float()
        self.sd = self.ev.sqrt().float()
        self.cw = None; self.k = 0
        if shape == 'emp' or shape.startswith('block'):
            self.Y = torch.cat([(X[a:a + CH] - self.mf) @ self.Vf for a in range(0, X.shape[0], CH)])
            ww = torch.ones(X.shape[0], device=X.device) if w is None else w
            self.cw = torch.cumsum(ww.double(), 0)
            self.k = int(shape[5:]) if shape.startswith('block') else 0

    def sample(self, b, g):
        n = self.V.shape[0]; dev = self.V.device
        if self.shape == 'gauss':
            E = torch.randn(b, n, generator=g, device=dev) * self.sd
        elif self.shape == 'rad':
            E = (torch.randint(0, 2, (b, n), generator=g, device=dev) * 2 - 1).float() * self.sd
        else:
            idx = wsample(self.cw, (b, n), g)
            if self.k > 0:
                idx[:, :self.k] = wsample(self.cw, (b, 1), g)
            E = self.Y.gather(0, idx)
        return self.mf + E @ self.Vf.T


class Cells:
    """Quantile cells (nb bins per axis, weighted quantiles) on the top-k principal components of the (weighted) fit
    set; each cell is a Lin(shape) of its members with their weights, i.e. its own mean, covariance and eigenbasis;
    cell probability = total weight. Samples may leave their cell (a mixture, not a truncated family)."""

    def __init__(self, X, shape, w=None, k=3, nb=4):
        ww = torch.ones(X.shape[0], device=X.device) if w is None else w
        m, C = mean_cov(X, ww); ev, V = eig_desc(C)
        Y = (X - m.float()) @ V[:, :k].float()
        lab = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        q = torch.linspace(0, 1, nb + 1, dtype=F64, device=X.device)[1:-1]
        for j in range(k):
            qs = wquantile(Y[:, j].contiguous(), ww, q)
            lab = lab * nb + torch.bucketize(Y[:, j].contiguous(), qs)
        self.cells, cw = [], []
        for c in torch.unique(lab):
            S = lab == c
            if int(S.sum()) < 2 * X.shape[1]:
                continue
            self.cells.append(Lin(X[S], shape, ww[S])); cw.append(float(ww[S].double().sum()))
        if not self.cells:
            raise ValueError('no cell has enough members')
        self.p = torch.tensor(cw, dtype=F64, device=X.device); self.p /= self.p.sum()

    def sample(self, b, g):
        lab = torch.multinomial(self.p.float(), b, replacement=True, generator=g)
        cnt = torch.bincount(lab, minlength=len(self.cells)).tolist()
        return torch.cat([c.sample(k, g) for c, k in zip(self.cells, cnt) if k > 0], 0)


class Polar:
    """Sphere-chart family: inner family fitted to w = z / |z| with weights rho = |z|; rho drawn independently from
    its unweighted empirical law."""

    def __init__(self, X, inner):
        self.rho, Wd = sphere(X)
        self.inner = inner(Wd, self.rho)

    def sample(self, b, g):
        d = self.inner.sample(b, g)
        r = self.rho[torch.randint(0, self.rho.shape[0], (b,), generator=g, device=self.rho.device)]
        return r[:, None] * d


class Att:
    """Attractor transfer: the joint whitened PC scores of nu at a reference (another layer, or another net), re-coloured
    with the current layer's weighted PC basis. Components in eigenvalue order; ranks: components with eigenvalue above
    1e-10 x the top one in both; signs of both bases fixed so every score has a nonnegative weighted third moment.
    Reference rows drawn with the reference weights. Second moments of nu at the current layer exact up to sampling."""

    def __init__(self, X, Xref):
        self.rho, Wd = sphere(X)
        rr, Wr = sphere(Xref)
        m, C = mean_cov(Wd, self.rho); ev, V = eig_desc(C)
        mr, Cr = mean_cov(Wr, rr); evr, Vr = eig_desc(Cr)
        k = int(min((ev > 1e-10 * ev[0]).sum(), (evr > 1e-10 * evr[0]).sum()))
        Yc = torch.cat([(Wd[a:a + CH] - m.float()) @ V[:, :k].float() for a in range(0, Wd.shape[0], CH)])
        Zr = torch.cat([(Wr[a:a + CH] - mr.float()) @ Vr[:, :k].float() for a in range(0, Wr.shape[0], CH)])
        Zr = Zr / evr[:k].sqrt().float()
        wc = (self.rho / self.rho.sum()).double(); wr = (rr / rr.sum()).double()
        sc = torch.sign((wc[:, None] * Yc.double() ** 3).sum(0)).float(); sc[sc == 0] = 1
        sr = torch.sign((wr[:, None] * Zr.double() ** 3).sum(0)).float(); sr[sr == 0] = 1
        self.Z = Zr * sr
        self.cw = torch.cumsum(rr.double(), 0)
        self.B = (V[:, :k].float() * sc) * ev[:k].sqrt().float()
        self.m = m.float(); self.k = k

    def sample(self, b, g):
        z = self.Z[wsample(self.cw, (b,), g)]
        d = self.m + z @ self.B.T
        r = self.rho[torch.randint(0, self.rho.shape[0], (b,), generator=g, device=self.rho.device)]
        return r[:, None] * d


# name -> (description, fully non-Gaussian?)
FAMILIES = {
    'EMP': ('resample of the fit set (floor)', None),
    'PEMP': ('sphere chart: w resampled with weights rho, rho independent (reduction control)', None),
    'G': ('one Gaussian on z', False),
    'PCA3': ('64 cells on top-3 PCs of z, Gaussian within', False),
    'I': ('z = m + V e, independent PC components with exact marginals', True),
    'PG': ('sphere chart, Gaussian nu', False),
    'PI': ('sphere chart, independent PC components with exact weighted marginals', True),
    'PR': ('sphere chart, two-point components', True),
    'PB2': ('sphere chart, top-2 PC block joint + independent rest', True),
    'PB4': ('sphere chart, top-4 block', True),
    'PB8': ('sphere chart, top-8 block', True),
    'PB16': ('sphere chart, top-16 block', True),
    'PB32': ('sphere chart, top-32 block', True),
    'PC3G': ('sphere chart, 64 cells on top-3 PCs, Gaussian within', False),
    'PC3I': ('sphere chart, 64 cells, independent within-cell components with exact marginals', True),
    'PC3R': ('sphere chart, 64 cells, two-point within', True),
    'ATT8': ('attractor transfer from layer l - 8 (same net)', True),
    'ATT16': ('attractor transfer from layer l - 16 (same net)', True),
    'ATTX': ('attractor transfer from the same layer of exploration net 112', True),
}


def build(name, X, g=None, Xref=None):
    if name == 'EMP':
        return Emp(X)
    if name == 'PEMP':
        return Polar(X, lambda Wd, w: Emp(Wd, w))
    if name == 'G':
        return Lin(X, 'gauss')
    if name == 'I':
        return Lin(X, 'emp')
    if name == 'PCA3':
        return Cells(X, 'gauss')
    if name in ('PG', 'PI', 'PR'):
        shape = dict(PG='gauss', PI='emp', PR='rad')[name]
        return Polar(X, lambda Wd, w: Lin(Wd, shape, w))
    if name.startswith('PB'):
        k = int(name[2:])
        return Polar(X, lambda Wd, w: Lin(Wd, f'block{k}', w))
    if name in ('PC3G', 'PC3I', 'PC3R'):
        shape = dict(PC3G='gauss', PC3I='emp', PC3R='rad')[name]
        return Polar(X, lambda Wd, w: Cells(Wd, shape, w))
    if name.startswith('ATT'):
        if Xref is None:
            raise ValueError('ATT needs a reference')
        return Att(X, Xref)
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------ propagation
@torch.no_grad()
def propagate(dn, fam, NR, l, Lmax, g, chunk=CH):
    """Draw NR samples of z_l from fam, run the exact noisy layers l+1..Lmax.
    Returns (mean, second moment) of a_L for L = l..Lmax ((Lmax - l + 1, n) float64 each)."""
    S1 = torch.zeros(Lmax - l + 1, dn.n, dtype=F64, device=dn.dev); S2 = torch.zeros_like(S1)
    done = 0
    while done < NR:
        b = min(chunk, NR - done)
        z = fam.sample(b, g).float()
        for k in range(l, Lmax + 1):
            if k > l:
                z = h @ dn.W[k - 1].T
            a = torch.relu(z)
            ad = a.double(); S1[k - l] += ad.sum(0); S2[k - l] += (ad * ad).sum(0)
            if k < Lmax:
                h = a + dn.s[k - 1] * torch.randn(b, dn.n, generator=g, device=dn.dev)
        done += b
    return S1 / NR, S2 / NR


@torch.no_grad()
def push(dn, X, l, g, chunk=CH):
    """z_l samples -> z_{l+1} samples (ReLU, noise s_l, W_{l+1})."""
    Xn = torch.empty_like(X)
    for a in range(0, X.shape[0], chunk):
        h = torch.relu(X[a:a + chunk])
        h = h + dn.s[l - 1] * torch.randn(h.shape[0], dn.n, generator=g, device=dn.dev)
        Xn[a:a + chunk] = h @ dn.W[l].T
    return Xn


@torch.no_grad()
def relu_moments(X, chunk=CH):
    S1 = sum(torch.relu(X[a:a + chunk]).double().sum(0) for a in range(0, X.shape[0], chunk))
    S2 = sum((torch.relu(X[a:a + chunk]).double() ** 2).sum(0) for a in range(0, X.shape[0], chunk))
    return S1 / X.shape[0], S2 / X.shape[0]


@torch.no_grad()
def compose(dn, name, X0, l0, Lmax, proj, NC, g, chunk=CH):
    """Re-projection chain from true paths X0 of z_{l0}: at every layer l in proj (l0 <= l < Lmax), fit `name` to the
    current samples and replace them by NC fresh draws ('MC' never projects: plain Monte Carlo on the X0 paths).
    Returns (mean, second moment) of a_L estimates for L = l0..Lmax from the samples arriving at L (before any
    projection at L), and the number of projections made before each L."""
    X = X0
    M1 = torch.zeros(Lmax - l0 + 1, dn.n, dtype=F64, device=dn.dev); M2 = torch.zeros_like(M1)
    nproj, count = [], 0
    for l in range(l0, Lmax + 1):
        M1[l - l0], M2[l - l0] = relu_moments(X)
        nproj.append(count)
        if l == Lmax:
            break
        if name != 'MC' and l in proj:
            fam = build(name, X, g)
            X = torch.cat([fam.sample(min(chunk, NC - a), g).float() for a in range(0, NC, chunk)])
            del fam
            count += 1
        X = push(dn, X, l, g)
    return M1, M2, nproj


def scores(M1, M2, TM, TV, l, NT, NR):
    """Per output layer L = l..: dict with nMSE (est vs truth, minus the truth floor 1/NT and the family's own
    sampling floor tr Cov_F(a_L) / (NR tr Cov_P(a_L))), the raw MSE and mean truth variance, and the floor used."""
    out = dict(nmse=[], mse=[], tv=[], floor=[])
    for i in range(M1.shape[0]):
        L = l + i
        tv = float(TV[L - 1].mean())
        fv = float((M2[i] - M1[i] ** 2).clamp_min(0).mean())
        mse = float(((M1[i] - TM[L - 1]) ** 2).mean())
        fl = 1.0 / NT + fv / (tv * NR)
        out['nmse'].append(mse / tv - fl); out['mse'].append(mse); out['tv'].append(tv); out['floor'].append(fl)
    return out
