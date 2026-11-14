"""log K q(h) for K = one noisy LN layer and q a tractable family: Laplace (Gauss-Newton) + importance sampling.

K q(h) = E_{h' ~ q}[ N(h; f(h'), s2 I) ].  Posterior over h' given h: q(h') N(h; f(h'), s2 I).
Gauss-Newton mode with q's Gaussian moments as the curvature prior, multivariate-t proposal around the mode,
weights with the exact q density. Returns the log-mean-weight (unbiased for K q(h) in the mean, so its log is biased low
by about var/2) and the effective sample size.
"""
import math
import torch

DT = torch.float64
L2PI = math.log(2 * math.pi)


def prop_factor(Hm, scale):
    """Factor Lc with Lc Lc^T = scale * Hm^{-1} (via Cholesky of Hm, jitter if needed) and log det of that covariance."""
    Hm = 0.5 * (Hm + Hm.transpose(1, 2)); d = Hm.shape[-1]
    jit = 0.0
    for _ in range(8):
        Lh, info = torch.linalg.cholesky_ex(Hm + jit * torch.eye(d, dtype=DT))
        if int(info.max()) == 0:
            break
        jit = max(1e-10, 10 * jit) * (Hm.diagonal(dim1=1, dim2=2).abs().max())
    Id = torch.eye(d, dtype=DT).expand_as(Hm)
    Linv = torch.linalg.solve_triangular(Lh, Id, upper=False)          # Lh^{-1}
    Lc = Linv.transpose(1, 2) * math.sqrt(scale)                        # Lh^{-T}: Lc Lc^T = Hm^{-1}
    ldC = -2 * torch.log(torch.diagonal(Lh, dim1=1, dim2=2)).sum(1) + d * math.log(scale)
    return Lc, ldC


_INV = {}


def inverse_regression(net, l, q, gen, n=4096):
    """Linear regression of h' on h under h' ~ q, h = f(h') + noise (initial guess for the posterior mode)."""
    key = (id(net), l, id(q))
    if key not in _INV:
        X = q.sample(n, gen); Hh = net.f(l, X) + net.sigma * torch.randn(n, net.d, generator=gen, dtype=DT)
        mx, mh = X.mean(0), Hh.mean(0); Xc, Hc = X - mx, Hh - mh
        Chh = Hc.T @ Hc / n + 1e-6 * torch.eye(net.d, dtype=DT); Cxh = Xc.T @ Hc / n
        B = torch.linalg.solve(Chh, Cxh.T).T
        _INV[key] = (B, mx - mh @ B.T)
    return _INV[key]


def gn_mode(net, l, mu, Si, H, x0, iters=40):
    s2 = net.sigma ** 2; n, d = H.shape
    x = x0.clone(); lam = torch.full((n,), 1e-2, dtype=DT); I = torch.eye(d, dtype=DT)

    def energy(x):
        r = H - net.f(l, x); dx = x - mu
        return 0.5 * ((dx @ Si) * dx).sum(1) + 0.5 * (r * r).sum(1) / s2

    E = energy(x)
    for _ in range(iters):
        r = H - net.f(l, x); J = net.jac(l, x); Jt = J.transpose(1, 2)
        g = (x - mu) @ Si - (Jt @ r[:, :, None])[:, :, 0] / s2
        Hm = Si[None] + Jt @ J / s2
        step = torch.linalg.solve(Hm + lam[:, None, None] * I, g[:, :, None])[:, :, 0]
        xn = x - step; En = energy(xn)
        acc = En < E
        x = torch.where(acc[:, None], xn, x); E = torch.where(acc, En, E)
        lam = torch.where(acc, lam / 3, lam * 5).clamp(1e-8, 1e8)
    J = net.jac(l, x)
    Hm = Si[None] + J.transpose(1, 2) @ J / s2
    return x, Hm, E


def log_kq(net, l, q, H, gen, n_is=512, nu=6.0, chunk=64, x0=None, iters=40, scale=1.2):
    """Returns (logKq (n,), ess (n,)) for points H (n, d) under layer l and prior family q."""
    mu, S = q.gauss(); Si = torch.linalg.inv(S); Si = 0.5 * (Si + Si.T)
    s2 = net.sigma ** 2; n, d = H.shape
    out = torch.empty(n, dtype=DT); ess = torch.empty(n, dtype=DT)
    lgc = math.lgamma((nu + d) / 2) - math.lgamma(nu / 2) - 0.5 * d * math.log(nu * math.pi)
    B, b0 = inverse_regression(net, l, q, gen)
    for a in range(0, n, chunk):
        Hc = H[a:a + chunk]; m = len(Hc)
        xr = Hc @ B.T + b0                               # linear-regression guess of the preimage
        x, Hm, E = gn_mode(net, l, mu, Si, Hc, xr, iters)
        if x0 is not None:                               # also start from a supplied point; keep the lower energy
            x2, Hm2, E2 = gn_mode(net, l, mu, Si, Hc, x0[a:a + chunk].clone(), iters)
            b = E2 < E
            x = torch.where(b[:, None], x2, x); Hm = torch.where(b[:, None, None], Hm2, Hm)
        Lc, ldC = prop_factor(Hm, scale)
        Z = torch.randn(m, n_is, d, generator=gen, dtype=DT)
        U = torch.distributions.Chi2(torch.tensor(nu, dtype=DT)).sample((m, n_is)) / nu
        Xs = x[:, None, :] + (Z @ Lc.transpose(1, 2)) / U.sqrt()[:, :, None]
        quad = (Z * Z).sum(2) / U
        lt = lgc - 0.5 * ldC[:, None] - 0.5 * (nu + d) * torch.log1p(quad / nu)
        Xf = Xs.reshape(-1, d)
        lq = q.logpdf(Xf).reshape(m, n_is)
        r = Hc[:, None, :] - net.f(l, Xf).reshape(m, n_is, d)
        ll = -0.5 * (r * r).sum(2) / s2 - 0.5 * d * (L2PI + math.log(s2))
        lw = lq + ll - lt
        lw = torch.where(torch.isfinite(lw), lw, torch.full_like(lw, -1e300))
        out[a:a + chunk] = torch.logsumexp(lw, 1) - math.log(n_is)
        w = torch.softmax(lw, 1); ess[a:a + chunk] = 1.0 / (w * w).sum(1)
    return out, ess


def grad_log_kq(net, l, q, H, gen, n_is=256, nu=6.0, iters=30):
    """d/dh log K q(h) = E_post[(f(h') - h)] / s2 via self-normalised IS (same proposal). Returns (logKq, grad)."""
    mu, S = q.gauss(); Si = torch.linalg.inv(S); Si = 0.5 * (Si + Si.T)
    s2 = net.sigma ** 2; m, d = H.shape
    B, b0 = inverse_regression(net, l, q, gen)
    x, Hm, _ = gn_mode(net, l, mu, Si, H, H @ B.T + b0, iters)
    Lc, ldC = prop_factor(Hm, 1.2)
    lgc = math.lgamma((nu + d) / 2) - math.lgamma(nu / 2) - 0.5 * d * math.log(nu * math.pi)
    Z = torch.randn(m, n_is, d, generator=gen, dtype=DT)
    U = torch.distributions.Chi2(torch.tensor(nu, dtype=DT)).sample((m, n_is)) / nu
    Xs = x[:, None, :] + (Z @ Lc.transpose(1, 2)) / U.sqrt()[:, :, None]
    lt = lgc - 0.5 * ldC[:, None] - 0.5 * (nu + d) * torch.log1p((Z * Z).sum(2) / U / nu)
    Xf = Xs.reshape(-1, d); F = net.f(l, Xf).reshape(m, n_is, d)
    r = H[:, None, :] - F
    lw = q.logpdf(Xf).reshape(m, n_is) - 0.5 * (r * r).sum(2) / s2 - 0.5 * d * (L2PI + math.log(s2)) - lt
    lw = torch.where(torch.isfinite(lw), lw, torch.full_like(lw, -1e300))
    w = torch.softmax(lw, 1)
    g = ((w[:, :, None] * F).sum(1) - H) / s2
    return torch.logsumexp(lw, 1) - math.log(n_is), g


# ------------------------------------------------------------------ IS in post-activation space (tanh): u = tanh(z)
# The LN fiber {u : LN(u) = s} = {beta 1 + r s : r > 0} is flat in u-space, so the posterior is close to Gaussian there.

class ZPrior:
    """Density of z = W h' for h' ~ q (exact for Gaussian q; for other q via q(W^{-1} z) / |det W|)."""

    def __init__(self, W, q):
        self.W = W; self.q = q; self.gaussian = hasattr(q, 'L') and getattr(q, 'delta', 0.0) == 0.0 and q.__class__.__name__ == 'Gauss'
        mu, S = q.gauss()
        self.mz = W @ mu; self.Cz = W @ S @ W.T; self.Cz = 0.5 * (self.Cz + self.Cz.T)
        self.Lz = torch.linalg.cholesky(self.Cz); self.ldz = 2 * self.Lz.diagonal().log().sum()
        self.Czi = torch.cholesky_inverse(self.Lz)
        self.Wi = torch.linalg.inv(W); self.ldW = torch.linalg.slogdet(W)[1]

    def logpdf(self, Z):
        if self.gaussian:
            r = torch.linalg.solve_triangular(self.Lz, (Z - self.mz).T, upper=False).T
            return -0.5 * (r * r).sum(-1) - 0.5 * self.ldz - 0.5 * Z.shape[-1] * L2PI
        return self.q.logpdf(Z @ self.Wi.T) - self.ldW


def _ln(u, sqd):
    c = u - u.mean(-1, keepdim=True)
    return c * (sqd / c.norm(dim=-1, keepdim=True).clamp_min(1e-30))


def log_kq_u(net, l, q, H, gen, n_is=512, nu=8.0, chunk=32, iters=40, scale=1.15, zp=None):
    """log K q(h) for a tanh layer by Laplace + IS over u = tanh(W h'). Returns (logKq, ess)."""
    assert net.act == 'tanh'
    W = net.W[l - 1]; d = net.d; s2 = net.sigma ** 2; sqd = math.sqrt(d)
    zp = zp or ZPrior(W, q)
    # inverse-regression init u ~ B h + b0 (from joint samples)
    X = q.sample(4096, gen); U0 = torch.tanh(X @ W.T); Hh = _ln(U0, sqd) + net.sigma * torch.randn(4096, d, generator=gen, dtype=DT)
    mu_, mh = U0.mean(0), Hh.mean(0); Chh = (Hh - mh).T @ (Hh - mh) / 4096 + 1e-6 * torch.eye(d, dtype=DT)
    B = torch.linalg.solve(Chh, ((U0 - mu_).T @ (Hh - mh) / 4096).T).T; b0 = mu_ - mh @ B.T
    I = torch.eye(d, dtype=DT)
    lgc = math.lgamma((nu + d) / 2) - math.lgamma(nu / 2) - 0.5 * d * math.log(nu * math.pi)
    n = len(H); out = torch.empty(n, dtype=DT); ess = torch.empty(n, dtype=DT)

    def neg_logpost(u, Hc):
        z = torch.atanh(u)
        lp = zp.logpdf(z) - torch.log1p(-u * u).sum(-1)
        r = Hc - _ln(u, sqd)
        return -(lp - 0.5 * (r * r).sum(-1) / s2)

    for a in range(0, n, chunk):
        Hc = H[a:a + chunk]; m = len(Hc)
        u = (Hc @ B.T + b0).clamp(-0.98, 0.98)
        lam = torch.full((m,), 1e-2, dtype=DT)
        with torch.no_grad():
            E = neg_logpost(u, Hc)
        for _ in range(iters):
            uu = u.clone().requires_grad_(True)
            Eg = neg_logpost(uu, Hc).sum(); g, = torch.autograd.grad(Eg, uu)
            with torch.no_grad():
                c = u - u.mean(-1, keepdim=True); nc = c.norm(dim=-1, keepdim=True); ch = c / nc
                JL = (sqd / nc)[:, :, None] * (I[None] - ch[:, :, None] * ch[:, None, :] - 1.0 / d)
                D = 1.0 / (1 - u * u)
                Hp = D[:, :, None] * zp.Czi[None] * D[:, None, :] + torch.diag_embed(2 * D)   # GN prior curvature
                Hm = Hp + JL.transpose(1, 2) @ JL / s2
                step = torch.linalg.solve(Hm + lam[:, None, None] * I, g[:, :, None])[:, :, 0]
                un = u - step
                okb = (un.abs() < 0.999999).all(1)
                En = torch.where(okb, neg_logpost(un.clamp(-0.999999, 0.999999), Hc), torch.full_like(E, float('inf')))
                acc = En < E
                u = torch.where(acc[:, None], un, u); E = torch.where(acc, En, E)
                lam = torch.where(acc, lam / 3, lam * 5).clamp(1e-8, 1e8)
        with torch.no_grad():
            c = u - u.mean(-1, keepdim=True); nc = c.norm(dim=-1, keepdim=True); ch = c / nc
            JL = (sqd / nc)[:, :, None] * (I[None] - ch[:, :, None] * ch[:, None, :] - 1.0 / d)
            D = 1.0 / (1 - u * u)
            Hm = D[:, :, None] * zp.Czi[None] * D[:, None, :] + torch.diag_embed(2 * D) + JL.transpose(1, 2) @ JL / s2
            Lc, ldC = prop_factor(Hm, scale)
            Z = torch.randn(m, n_is, d, generator=gen, dtype=DT)
            Ug = torch.distributions.Chi2(torch.tensor(nu, dtype=DT)).sample((m, n_is)) / nu
            Us = u[:, None, :] + (Z @ Lc.transpose(1, 2)) / Ug.sqrt()[:, :, None]
            lt = lgc - 0.5 * ldC[:, None] - 0.5 * (nu + d) * torch.log1p((Z * Z).sum(2) / Ug / nu)
            inside = (Us.abs() < 1).all(2)
            Usc = Us.clamp(-1 + 1e-12, 1 - 1e-12)
            lw = -neg_logpost(Usc.reshape(-1, d), Hc.repeat_interleave(n_is, 0)).reshape(m, n_is) - lt
            lw = lw - 0.5 * d * (L2PI + math.log(s2))
            lw = torch.where(inside & torch.isfinite(lw), lw, torch.full_like(lw, -1e300))
            out[a:a + chunk] = torch.logsumexp(lw, 1) - math.log(n_is)
            w = torch.softmax(lw, 1); ess[a:a + chunk] = 1.0 / (w * w).sum(1)
    return out, ess


# ------------------------------------------------------------------ tempered SMC for log K q(h) (robust, unbiased in K q(h))
def log_kq_smc(net, l, q, H, gen, Np=256, ess_frac=0.5, n_moves=5, max_steps=600, chunk=32):
    """Targets q(h') N(h; f(h'), s2 I)^t, t: 0 -> 1, adaptive steps (conditional ESS = ess_frac Np), multinomial
    resampling, random-walk MH moves with the particle-cloud covariance. Returns (logKq (n,), n_steps (n,))."""
    s2 = net.sigma ** 2; n, d = H.shape
    const = -0.5 * d * (L2PI + math.log(s2))
    out = torch.empty(n, dtype=DT); steps = torch.empty(n, dtype=DT)
    for a in range(0, n, chunk):
        Hc = H[a:a + chunk]; B = len(Hc)
        X = q.sample(B * Np, gen).reshape(B, Np, d)
        lq = q.logpdf(X.reshape(-1, d)).reshape(B, Np)

        def loglik(X):
            r = Hc[:, None, :] - net.f(l, X.reshape(-1, d)).reshape(B, Np, d)
            return -0.5 * (r * r).sum(2) / s2

        ll = loglik(X)
        t = torch.zeros(B, dtype=DT); logZ = torch.zeros(B, dtype=DT); ns = torch.zeros(B, dtype=DT)
        scale = torch.full((B,), 2.38 / math.sqrt(d), dtype=DT)
        for _ in range(max_steps):
            act = t < 1 - 1e-12
            if not act.any():
                break
            # bisection for dt: ESS(exp(dt * ll)) = ess_frac * Np
            lo = torch.zeros(B, dtype=DT); hi = (1 - t).clone()
            def ess(dt):
                w = torch.softmax(dt[:, None] * ll, 1); return 1.0 / (w * w).sum(1)
            full_ok = ess(hi) >= ess_frac * Np
            for _b in range(40):
                mid = 0.5 * (lo + hi); ok = ess(mid) >= ess_frac * Np
                lo = torch.where(ok, mid, lo); hi = torch.where(ok, hi, mid)
            dt = torch.where(full_ok, 1 - t, lo.clamp_min(1e-12))
            dt = torch.where(act, dt, torch.zeros_like(dt))
            lw = dt[:, None] * ll
            logZ = logZ + torch.where(act, torch.logsumexp(lw, 1) - math.log(Np), torch.zeros_like(logZ))
            t = t + dt; ns = ns + act.to(DT)
            idx = torch.multinomial(torch.softmax(lw, 1), Np, replacement=True, generator=gen)
            X = torch.gather(X, 1, idx[:, :, None].expand(-1, -1, d)); ll = torch.gather(ll, 1, idx); lq = torch.gather(lq, 1, idx)
            # moves
            Xm = X.mean(1, keepdim=True); C = (X - Xm).transpose(1, 2) @ (X - Xm) / Np + 1e-10 * torch.eye(d, dtype=DT)
            Lc = torch.linalg.cholesky(C + 1e-8 * C.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(d, dtype=DT))
            acc_rate = torch.zeros(B, dtype=DT)
            for _m in range(n_moves):
                Z = torch.randn(B, Np, d, generator=gen, dtype=DT)
                Xp = X + scale[:, None, None] * (Z @ Lc.transpose(1, 2))
                lqp = q.logpdf(Xp.reshape(-1, d)).reshape(B, Np); llp = loglik(Xp)
                la = (lqp - lq) + t[:, None] * (llp - ll)
                accp = (torch.rand(B, Np, generator=gen, dtype=DT).log() < la) & act[:, None]
                X = torch.where(accp[:, :, None], Xp, X); ll = torch.where(accp, llp, ll); lq = torch.where(accp, lqp, lq)
                acc_rate = acc_rate + accp.to(DT).mean(1) / n_moves
            scale = torch.where(act, scale * torch.exp(acc_rate - 0.25), scale)
        out[a:a + B] = logZ + const; steps[a:a + B] = ns
    return out, steps
