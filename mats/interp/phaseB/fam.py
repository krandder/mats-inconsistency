"""q families for one site (phase B task 3). Every family is a finite Gaussian mixture with tractable log density,
sampling, and a description length bits_desc (PREREG-B.md). All covariances are floored at sigma^2 I.
"""
import math
import numpy as np
import torch
import mi

LOG2 = math.log(2.0)


def floor_cov(S, s2):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.maximum(w, s2)) @ V.T


def half_logdet_ratio_bits(S, s2):
    """0.5 log2 det(S / s2) over eigenvalues floored at s2 (>= 0)."""
    w = np.linalg.eigvalsh(S)
    return float(0.5 * np.log2(np.maximum(w, s2) / s2).sum())


class SharedCovMixture:
    """q(h) = sum_j pi_j N(h; mu_j, Sigma) with one shared covariance (F1, F3)."""

    def __init__(self, means, Sigma, weights=None):
        self.mu = np.asarray(means, np.float64); self.S = Sigma
        self.w = np.full(len(self.mu), 1 / len(self.mu)) if weights is None else np.asarray(weights) / np.sum(weights)
        self.L = np.linalg.cholesky(Sigma); self.Li = np.linalg.inv(self.L)
        self.ld = 2 * np.log(np.diag(self.L)).sum()
        self.muw = self.mu @ self.Li.T
        self.D = Sigma.shape[0]

    def logpdf(self, H, chunk=256):
        Hw = torch.as_tensor(np.asarray(H, np.float64) @ self.Li.T); C = torch.as_tensor(self.muw)
        lw = torch.as_tensor(np.log(self.w)); c2 = (C ** 2).sum(1)
        out = np.empty(len(Hw))
        for i in range(0, len(Hw), chunk):
            t = Hw[i:i + chunk]
            d2 = ((t ** 2).sum(1, keepdim=True) + c2[None] - 2 * t @ C.T).clamp_min(0)
            out[i:i + chunk] = torch.logsumexp(lw[None] - 0.5 * d2, 1).numpy()
        return out - 0.5 * self.ld - 0.5 * self.D * math.log(2 * math.pi)

    def sample(self, n, rng, return_comp=False):
        j = rng.choice(len(self.mu), n, p=self.w)
        h = self.mu[j] + rng.standard_normal((n, self.D)) @ self.L.T
        return (h, j) if return_comp else h


class FullCovMixture:
    """q(h) = sum_c pi_c N(h; mu_c, Sigma_c) (F2)."""

    def __init__(self, means, covs, weights):
        self.mu = np.asarray(means); self.S = np.asarray(covs); self.w = np.asarray(weights) / np.sum(weights)
        self.L = np.linalg.cholesky(self.S); self.Li = np.linalg.inv(self.L)
        self.ld = 2 * np.log(np.diagonal(self.L, axis1=1, axis2=2)).sum(1)
        self.D = self.mu.shape[1]

    def logpdf(self, H, chunk=512):
        H = np.asarray(H, np.float64); out = np.empty(len(H))
        for i in range(0, len(H), chunk):
            r = H[i:i + chunk, None, :] - self.mu[None]                       # (n, C, D)
            z = np.einsum('cde,nce->ncd', self.Li, r)
            lp = np.log(self.w)[None] - 0.5 * (z ** 2).sum(2) - 0.5 * self.ld[None] - 0.5 * self.D * math.log(2 * math.pi)
            mx = lp.max(1, keepdims=True); out[i:i + chunk] = mx[:, 0] + np.log(np.exp(lp - mx).sum(1))
        return out

    def sample(self, n, rng):
        c = rng.choice(len(self.mu), n, p=self.w)
        return self.mu[c] + np.einsum('nde,ne->nd', self.L[c], rng.standard_normal((n, self.D)))


class LowRankMixture:
    """q(h) = sum_S pi_S N(h; m_S, U_S U_S^T + s2 I), U_S (D x k) (F4)."""

    def __init__(self, means, U, s2, weights):
        self.mu = np.asarray(means); self.U = np.asarray(U); self.s2 = s2; self.w = np.asarray(weights) / np.sum(weights)
        self.D, self.k = self.U.shape[1], self.U.shape[2]
        G = np.einsum('sdk,sdl->skl', self.U, self.U) / s2 + np.eye(self.k)[None]   # I + U^T U / s2
        self.Gi = np.linalg.inv(G); self.ldG = np.linalg.slogdet(G)[1]
        self.ld = self.D * math.log(s2) + self.ldG

    def logpdf(self, H, chunk=64):
        H = torch.as_tensor(np.asarray(H, np.float64)); mu = torch.as_tensor(self.mu); U = torch.as_tensor(self.U)
        Gi = torch.as_tensor(self.Gi); lw = torch.as_tensor(np.log(self.w)); ld = torch.as_tensor(self.ld)
        out = np.empty(len(H))
        for i in range(0, len(H), chunk):
            r = H[i:i + chunk, None, :] - mu[None]                                  # (n, S, D)
            a = torch.einsum('nsd,sdk->nsk', r, U)                                  # U^T r
            quad = ((r ** 2).sum(2) - torch.einsum('nsk,skl,nsl->ns', a, Gi, a) / self.s2) / self.s2
            lp = lw[None] - 0.5 * quad - 0.5 * ld[None] - 0.5 * self.D * math.log(2 * math.pi)
            out[i:i + chunk] = torch.logsumexp(lp, 1).numpy()
        return out

    def sample(self, n, rng):
        c = rng.choice(len(self.mu), n, p=self.w)
        return self.mu[c] + np.einsum('ndk,nk->nd', self.U[c], rng.standard_normal((n, self.k))) + math.sqrt(self.s2) * rng.standard_normal((n, self.D))


# ------------------------------------------------------------------------------------------------ fitting
def ridge(Z, H, lam=1e-3):
    Zc = np.c_[Z, np.ones(len(Z))]
    A = np.linalg.solve(Zc.T @ Zc + lam * len(Z) * np.diag(np.r_[np.ones(Z.shape[1]), 0]), Zc.T @ H)
    return A


def fit_F1(Hfit, Zfit, Zbank, s2):
    """Gaussian conditional on named latents z: mean linear in z, shared full residual covariance; q marginalises z
    over a bank of world draws."""
    mz, sz = Zfit.mean(0), Zfit.std(0) + 1e-9
    A = ridge((Zfit - mz) / sz, Hfit)
    pred = np.c_[(Zfit - mz) / sz, np.ones(len(Zfit))] @ A
    S = floor_cov(np.cov((Hfit - pred).T), s2)
    means = np.c_[(Zbank - mz) / sz, np.ones(len(Zbank))] @ A
    q = SharedCovMixture(means, S)
    q.bits_code = 0.0                                     # no discrete code; the continuous part is in h(q)
    q.desc_mode = 'rate'
    q.r2 = float(1 - ((Hfit - pred) ** 2).sum() / ((Hfit - Hfit.mean(0)) ** 2).sum())
    return q


def probe_r2(H, Z, ntr):
    out = []
    for j in range(Z.shape[1]):
        A = ridge(H[:ntr], Z[:ntr, j:j + 1], 1e-3)
        p = np.c_[H[ntr:], np.ones(len(H) - ntr)] @ A
        out.append(float(1 - ((Z[ntr:, j:j + 1] - p) ** 2).sum() / ((Z[ntr:, j] - Z[ntr:, j].mean()) ** 2).sum()))
    return np.array(out)


def fit_F2(Hfit, Zfit, s2, ndir=3, K=4, shrink=50):
    r2 = probe_r2(Hfit, Zfit, int(0.8 * len(Hfit)))
    top = np.argsort(-r2)[:ndir]
    edges = [np.quantile(Zfit[:, j], np.linspace(0, 1, K + 1)[1:-1]) for j in top]
    cell = np.zeros(len(Hfit), int)
    for e, j in zip(edges, top):
        cell = cell * K + np.searchsorted(e, Zfit[:, j])
    Sp = np.cov(Hfit.T)
    pooled = np.zeros_like(Sp); means, covs, ws = [], [], []
    ids = np.unique(cell)
    resid = np.zeros_like(Hfit)
    for c in ids:
        s = cell == c; m = Hfit[s].mean(0); resid[s] = Hfit[s] - m
    Spool = np.cov(resid.T)
    for c in ids:
        s = cell == c; n = s.sum()
        Sc = np.cov(Hfit[s].T) if n > 2 else Spool
        w = n / (n + shrink)
        covs.append(floor_cov(w * Sc + (1 - w) * Spool, s2)); means.append(Hfit[s].mean(0)); ws.append(n)
    q = FullCovMixture(np.array(means), np.array(covs), np.array(ws, float))
    p = q.w
    q.bits_code = float(-(p * np.log2(p)).sum())
    q.bits_cont = float(sum(pi * half_logdet_ratio_bits(S, s2) for pi, S in zip(p, covs)))
    q.desc_mode = 'code'; q.top = top.tolist(); q.top_r2 = r2[top].tolist(); q.ncell = len(ids)
    return q


def fit_F3(Hfit, pat, s2, minc=5):
    """Mixture over activation patterns (rows of the boolean matrix pat), per-pattern mean, shared covariance;
    patterns with < minc fit samples pooled into one component."""
    keys, inv, cnt = np.unique(pat, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    p = cnt / cnt.sum(); Hpat = float(-(p * np.log2(p)).sum())
    big = cnt >= minc
    comp = np.where(big[inv], inv, -1)
    ids = np.unique(comp)
    means, ws = [], []
    resid = np.zeros_like(Hfit)
    for c in ids:
        s = comp == c; m = Hfit[s].mean(0); resid[s] = Hfit[s] - m; means.append(m); ws.append(s.sum())
    S = floor_cov(np.cov(resid.T), s2)
    q = SharedCovMixture(np.array(means), S, np.array(ws, float))
    q.bits_code = Hpat
    q.bits_cont = half_logdet_ratio_bits(S, s2)
    q.desc_mode = 'code'; q.npat = int(len(keys)); q.ncomp = int(len(ids)); q.frac_rare = float(1 - cnt[big].sum() / cnt.sum())
    # Miller-Madow corrected entropy, and the coverage of the plug-in (patterns seen once)
    q.Hpat_mm = Hpat + (len(keys) - 1) / (2 * cnt.sum() * LOG2); q.singletons = float((cnt == 1).sum() / cnt.sum())
    return q


class TopKSAE(torch.nn.Module):
    def __init__(self, D, F, k):
        super().__init__()
        self.k = k
        self.E = torch.nn.Linear(D, F); self.Dm = torch.nn.Parameter(torch.randn(F, D) / math.sqrt(D)); self.bd = torch.nn.Parameter(torch.zeros(D))

    def pre(self, h):
        return self.E(h - self.bd)

    def code(self, h):
        z = self.pre(h)
        v, i = z.topk(self.k, -1)
        a = torch.zeros_like(z).scatter(-1, i, torch.relu(v))
        return a

    def forward(self, h):
        a = self.code(h)
        return a @ self.Dm + self.bd, a


def train_sae(H, F, k, steps=2000, seed=0, lr=2e-3, bs=512):
    torch.manual_seed(seed)
    D = H.shape[1]; Ht = torch.as_tensor(H, dtype=torch.float32)
    m = TopKSAE(D, F, k)
    with torch.no_grad():
        m.bd.copy_(Ht.mean(0))
        idx = torch.randperm(len(Ht))[:F]
        dirs = Ht[idx] - Ht.mean(0); dirs = dirs / dirs.norm(dim=1, keepdim=True)
        m.Dm.copy_(dirs); m.E.weight.copy_(dirs); m.E.bias.zero_()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        b = Ht[torch.randint(len(Ht), (bs,), generator=g)]
        rec, _ = m(b)
        loss = ((rec - b) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            m.Dm.div_(m.Dm.norm(dim=1, keepdim=True).clamp_min(1e-6))   # unit-norm atoms
    return m


def fit_F4(Hfit, s2, F, k, seed=0, sae=None):
    sae = sae or train_sae(Hfit, F, k, seed=seed)
    with torch.no_grad():
        Ht = torch.as_tensor(Hfit, dtype=torch.float32)
        rec, A = sae(Ht)
    A = A.numpy().astype(np.float64); rec = rec.numpy().astype(np.float64)
    Dm = sae.Dm.detach().numpy().astype(np.float64); bd = sae.bd.detach().numpy().astype(np.float64)
    act = A > 0
    D = Hfit.shape[1]
    se2 = max(float(((Hfit - rec) ** 2).mean()), s2)
    mean_a = np.array([A[act[:, i], i].mean() if act[:, i].any() else 0.0 for i in range(F)])
    var_a = np.array([A[act[:, i], i].var() if act[:, i].sum() > 1 else 0.0 for i in range(F)])
    # supports (as sorted index tuples, padded with -1 when fewer than k are positive)
    sup = np.full((len(A), k), -1)
    for r in range(len(A)):
        ii = np.flatnonzero(act[r]); sup[r, :len(ii)] = ii[:k]
    keys, inv, cnt = np.unique(sup, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    means = np.zeros((len(keys), D)); U = np.zeros((len(keys), D, k))
    cont = np.zeros(len(keys))
    for c, key in enumerate(keys):
        means[c] = bd
        for j, i in enumerate(key):
            if i < 0: continue
            means[c] += mean_a[i] * Dm[i]
            U[c, :, j] = math.sqrt(var_a[i]) * Dm[i]
            cont[c] += 0.5 * math.log2(1 + var_a[i] * (Dm[i] @ Dm[i]) / se2)
    q = LowRankMixture(means, U, se2, cnt.astype(float))
    p = cnt / cnt.sum()
    q.bits_code = float(-(p * np.log2(p)).sum())
    q.bits_cont = float((p * cont).sum() + 0.5 * D * math.log2(se2 / s2))
    q.desc_mode = 'code'; q.nsup = int(len(keys)); q.se2 = se2; q.sae = sae
    q.fve = float(1 - ((Hfit - rec) ** 2).sum() / ((Hfit - Hfit.mean(0)) ** 2).sum())
    q.Hsup_mm = q.bits_code + (len(keys) - 1) / (2 * cnt.sum() * LOG2)
    return q


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(q, Ht, hbr, bank, sigma, rng, nq=4096):
    """Ht: eval targets from p (n, D); hbr: dict(h_in, h_out) bracket of h(p) in bits; bank: p centres for log p_hat.
    Returns eps bracket (bits), wasted bits (over-estimate), h(q), bits_rate, bits_desc."""
    D = Ht.shape[1]; hz = mi.hZ_bits(D, sigma)
    CE = float(-q.logpdf(Ht).mean() / LOG2)
    Hq = q.sample(nq, rng); lq = q.logpdf(Hq)
    hq = float(-lq.mean() / LOG2)
    lp, _ = mi.logp_mix(Hq, bank, sigma)                                   # unbiased p_hat at q samples
    wasted = float((lq - lp).mean() / LOG2)
    rate = hq - hz
    desc = rate if q.desc_mode == 'rate' else q.bits_code + q.bits_cont
    return dict(CE=CE, eps_lo=CE - hbr['h_out'], eps_hi=CE - hbr['h_in'], wasted_ub=wasted, h_q=hq, bits_rate=rate,
                bits_desc=desc, bits_code=getattr(q, 'bits_code', 0.0), lost_code=desc - rate)
