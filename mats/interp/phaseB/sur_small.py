"""Surrogate chains on the trained small noisy nets (PREREG-B amendment 4).

Real chain: h_l = f_l(h_{l-1}) + sigma e_l, f_l(h) = LN0(tanh(W_l h + b_l)), h_0... ; surrogate chain: same with g_l.
Exact chain rule: KL(real path || surrogate path) = sum_l E_real ||f_l(h_{l-1}) - g_l(h_{l-1})||^2 / (2 sigma^2) (nats).
(1) per-layer KL vs description cost (parameters) for LR-r, LIN-r, TC-F-k, exact;
(2) Donsker-Varadhan two-sided bounds on E[F] for F = squared task error and F = output mean, vs plain sampling's 95% CI
    at equal forward passes;
(3) skip-merge of layer pairs with the linearised propagated noise covariance sigma^2 (I + J J^T); Gaussian check against
    the sampled real two-step kernel; DP over cut points minimising total KL + beta * parameters.
usage: python sur_small.py SIGMA -> results/sur_small_s{SIGMA}.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.special import logsumexp
import e2 as E2
import dec as Dc

torch.set_num_threads(1)
L, DIN, D = E2.L, E2.DIN, 8
OUT = E2.OUT


def ln0(a):
    c = a - a.mean(-1, keepdims=True); return c * (math.sqrt(a.shape[-1]) / np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-9))


def paths(net, sigma, n, rng):
    x = rng.standard_normal((n, DIN)); hs = [x]; mus = []
    h = x
    for l in range(L):
        m = net.f(l, h); h = m + sigma * rng.standard_normal(m.shape); mus.append(m); hs.append(h)
    return hs, mus                                                   # hs[l] = input of layer l (hs[0] = x), mus[l] = f_l


# ------------------------------------------------------------------ surrogate families (fit on (parent, f(parent)))
def rrr(X, Y, r):
    """Reduced-rank least squares Y ~ mu_Y + B (X - mu_X), rank r. Returns (B, muX, muY)."""
    muX, muY = X.mean(0), Y.mean(0); Xc, Yc = X - muX, Y - muY
    Sx = Xc.T @ Xc / len(X) + 1e-9 * np.eye(X.shape[1])
    w, V = np.linalg.eigh(Sx); Sh = (V * np.sqrt(w)) @ V.T; Shi = (V / np.sqrt(w)) @ V.T
    Bf = np.linalg.solve(Sx, Xc.T @ Yc / len(X)).T                      # (dy, dx)
    U, s, Vt = np.linalg.svd(Bf @ Sh, full_matrices=False)
    B = (U[:, :r] * s[:r]) @ Vt[:r] @ Shi
    return B, muX, muY


class LR:
    """Rank-r pre-activation map, same tanh + LN0: g(h) = LN0(tanh(A h + c))."""
    def __init__(self, net, l, Hfit, r):
        Z = Hfit @ net.W[l].T * net.g + net.b[l]
        B, muX, muZ = rrr(Hfit, Z, r); self.B, self.c = B, muZ - B @ muX; self.r = r
        self.cost = r * (Hfit.shape[1] + D) + D
    def __call__(self, H): return ln0(np.tanh(H @ self.B.T + self.c))


class LIN:
    """Rank-r affine map (no nonlinearity) fitted to f."""
    def __init__(self, net, l, Hfit, Yfit, r):
        B, muX, muY = rrr(Hfit, Yfit, r); self.B, self.c = B, muY - B @ muX
        self.cost = r * (Hfit.shape[1] + D) + D
    def __call__(self, H): return H @ self.B.T + self.c


class TC:
    """TopK transcoder g(h) = c + Dec TopK(Enc h + e), MSE-trained."""
    def __init__(self, Hfit, Yfit, F, k, steps=3000, seed=0):
        torch.manual_seed(seed); din = Hfit.shape[1]
        self.E = torch.nn.Linear(din, F).double(); self.Dm = torch.nn.Linear(F, D).double(); self.k = k
        opt = torch.optim.Adam(list(self.E.parameters()) + list(self.Dm.parameters()), lr=3e-3)
        Xt, Yt = torch.tensor(Hfit), torch.tensor(Yfit); g = torch.Generator().manual_seed(seed)
        for s in range(steps):
            i = torch.randint(len(Xt), (1024,), generator=g)
            loss = ((self.fwd(Xt[i]) - Yt[i]) ** 2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        self.cost = F * (din + D) + F + D
    def fwd(self, X):
        z = self.E(X); v, i = z.topk(self.k, -1); a = torch.zeros_like(z).scatter(-1, i, torch.relu(v)); return self.Dm(a)
    def __call__(self, H):
        with torch.no_grad(): return self.fwd(torch.tensor(H)).numpy()


def exact_cost(l): return (DIN if l == 0 else D) * D + D


# ------------------------------------------------------------------ Donsker-Varadhan
LAMS = np.r_[np.geomspace(1e-3, 50, 60)]


def dv_bounds(KL, Fq):
    """Two-sided DV bounds on E_p F from KL (nats) and surrogate samples Fq; returns (lo, hi, lam_lo, lam_hi)."""
    m = Fq.mean(); best_hi, best_lo, lh, ll = np.inf, -np.inf, None, None
    for lam in LAMS:
        hi = (KL + logsumexp(lam * (Fq - m)) - math.log(len(Fq))) / lam + m
        lo = -(KL + logsumexp(-lam * (Fq - m)) - math.log(len(Fq))) / lam + m
        if hi < best_hi: best_hi, lh = hi, lam
        if lo > best_lo: best_lo, ll = lo, lam
    return float(best_lo), float(best_hi), lh, ll


def surrogate_paths(net, gs, sigma, n, rng):
    x = rng.standard_normal((n, DIN)); h = x
    for l in range(L):
        m = gs[l](h) if gs[l] is not None else net.f(l, h); h = m + sigma * rng.standard_normal(m.shape)
    return x, h


def run(sigma):
    t_all = time.time()
    net, w_ro, b_ro = Dc.train_net_ro(D, sigma); U = np.random.default_rng(0).standard_normal((4, DIN))
    target = lambda x: np.sin(x @ U.T).sum(1)
    hs_f, mus_f = paths(net, sigma, 2 ** 15, np.random.default_rng(1))
    hs_e, mus_e = paths(net, sigma, 2 ** 15, np.random.default_rng(2))
    s2x2 = 2 * sigma ** 2
    out = dict(sigma=sigma, layers=[], configs=[], merges=[], dp=[])
    fams = {}
    # ---- (1) per-layer frontier
    for l in range(L):
        Hf, Yf, He, Ye = hs_f[l], mus_f[l], hs_e[l], mus_e[l]
        din = Hf.shape[1]; cand = {'exact': None}
        for r in range(1, min(din, D) + 1):
            if r in (1, 2, 3, 4, 6, 8): cand[f'LR{r}'] = LR(net, l, Hf, r); cand[f'LIN{r}'] = LIN(net, l, Hf, Yf, r)
        for F, k in ((16, 1), (16, 2), (64, 2), (64, 4)):
            cand[f'TC{F}k{k}'] = TC(Hf, Yf, F, k, seed=l)
        rows = {}
        for name, g in cand.items():
            if g is None:
                rows[name] = dict(KL=0.0, cost=exact_cost(l)); continue
            err = ((Ye - g(He)) ** 2).sum(1) / s2x2
            rows[name] = dict(KL=float(err.mean()), KL_se=float(err.std() / math.sqrt(len(err))), cost=int(g.cost))
        fams[l] = cand
        out['layers'].append(dict(layer=l + 1, rows=rows))
        print('layer', l + 1, {k: (round(v['KL'], 3), v['cost']) for k, v in rows.items()}, flush=True)
    # ---- (2) DV bounds for chain configurations
    x_ref, h_ref = surrogate_paths(net, [None] * L, sigma, 2 ** 20, np.random.default_rng(3))
    yhat = h_ref @ w_ro + b_ro; F_ref = {'sqerr': (yhat - target(x_ref)) ** 2, 'mean': yhat}
    truth = {k: float(v.mean()) for k, v in F_ref.items()}; sd = {k: float(v.std()) for k, v in F_ref.items()}
    nKL, nQ = 2 ** 14, 2 ** 14; n_eq = 2 * nKL + nQ
    configs = {}
    for fam in ('LR1', 'LR2', 'LR4', 'LIN4', 'LIN8', 'TC64k2', 'TC64k4'):
        configs['all-' + fam] = {l: fam for l in range(L)}
    for beta in (1e-3, 1e-2, 1e-1):                                       # per-layer Lagrangian choice
        configs[f'best-beta{beta:g}'] = {l: min(out['layers'][l]['rows'].items(), key=lambda kv: kv[1]['KL'] + beta * kv[1]['cost'])[0] for l in range(L)}
    def resolve(l, nm):
        if nm in fams[l]: return nm
        for base in ('LIN', 'LR'):
            if nm.startswith(base):
                r = int(nm[len(base):]); avail = sorted(int(k[len(base):]) for k in fams[l] if k.startswith(base) and k[len(base):].isdigit())
                return base + str(max(x for x in avail if x <= r))
        return 'exact'
    for cname, choice in configs.items():
        choice = {l: resolve(l, choice[l]) for l in range(L)}
        gs = [fams[l][choice[l]] for l in range(L)]
        KL = sum(out['layers'][l]['rows'][choice[l]]['KL'] for l in range(L))
        cost = sum(out['layers'][l]['rows'][choice[l]]['cost'] for l in range(L))
        xq, hq = surrogate_paths(net, gs, sigma, nQ, np.random.default_rng(4))
        yq = hq @ w_ro + b_ro; Fq = {'sqerr': (yq - target(xq)) ** 2, 'mean': yq}
        row = dict(config=cname, choice={str(l + 1): choice[l] for l in range(L)}, KL=KL, cost=cost, n_eq=n_eq)
        for f in ('sqerr', 'mean'):
            lo, hi, lh, ll = dv_bounds(KL, Fq[f])
            ci = 1.96 * sd[f] / math.sqrt(n_eq)
            row[f] = dict(truth=truth[f], q_mean=float(Fq[f].mean()), lo=lo, hi=hi, width=hi - lo, ci_width=2 * ci, width_over_ci=(hi - lo) / (2 * ci))
        out['configs'].append(row)
        print(cname, 'KL %.3f cost %d' % (KL, cost), {f: (round(row[f]['lo'], 3), round(row[f]['hi'], 3), round(row[f]['truth'], 3), round(row[f]['width_over_ci'], 1)) for f in ('sqerr', 'mean')}, flush=True)
    # ---- (3) skip-merge: pairs (l+1, l+2) from parent h = hs[l]
    def Jac(l, U_):                                                     # Jacobian of f_l at points U_ (n, d) -> (n, d, d)
        W = torch.tensor(net.W[l]); b = torch.tensor(net.b[l])
        def fn(u):
            a = torch.tanh(net.g * u @ W.T + b); c = a - a.mean(); return c * math.sqrt(D) / c.norm()
        return np.stack([torch.autograd.functional.jacobian(fn, torch.tensor(u)).numpy() for u in U_])
    rng = np.random.default_rng(9)
    for l in range(L - 1):                                                # merged block covers layers l and l+1 (0-based)
        He = hs_e[l][:4000]; U1 = net.f(l, He); M = net.f(l + 1, U1)
        J = Jac(l + 1, U1); Sig = sigma ** 2 * (np.eye(D)[None] + J @ np.transpose(J, (0, 2, 1)))
        # Gaussian check: KL(real two-step kernel || N(M, Sig)) on 100 inputs
        kls = []
        for i in range(100):
            h0 = He[i:i + 1]
            e1 = rng.standard_normal((4096, D)); mid = net.f(l, h0) + sigma * e1; C = net.f(l + 1, mid)
            ev = net.f(l + 1, net.f(l, h0) + sigma * rng.standard_normal((2048, D))) + sigma * rng.standard_normal((2048, D))
            d2 = ((ev[:, None, :] - C[None]) ** 2).sum(-1)
            lp = logsumexp(-d2 / (2 * sigma ** 2), 1) - math.log(len(C)) - 0.5 * D * math.log(2 * math.pi * sigma ** 2)
            Si = np.linalg.inv(Sig[i]); r_ = ev - M[i]
            lq = -0.5 * np.einsum('nd,de,ne->n', r_, Si, r_) - 0.5 * np.linalg.slogdet(2 * math.pi * Sig[i])[1]
            kls.append(float(np.mean(lp - lq)))
        # merged surrogates: exact composed mean with fixed average covariance; rank-r affine of the composed map
        Sbar = Sig.mean(0); Sbi = np.linalg.inv(Sbar); ldb = np.linalg.slogdet(Sbar)[1]
        def kl_gauss(Mhat):
            r_ = M - Mhat
            t = np.einsum('de,ned->n', Sbi, Sig) - D + ldb - np.linalg.slogdet(Sig)[1] + np.einsum('nd,de,ne->n', r_, Sbi, r_)
            return float(0.5 * t.mean())
        Hf = hs_f[l][:2 ** 14]; Mf = net.f(l + 1, net.f(l, Hf))
        mrows = {'exact-merge': dict(KL=kl_gauss(M), cost=exact_cost(l) + exact_cost(l + 1) + D * (D + 1) // 2)}
        for r in (1, 2, 4, 8):
            if r > Hf.shape[1]: continue
            B, muX, muY = rrr(Hf, Mf, r); Mhat = He @ B.T + (muY - B @ muX)
            mrows[f'mergeLIN{r}'] = dict(KL=kl_gauss(Mhat), cost=r * (Hf.shape[1] + D) + D + D * (D + 1) // 2)
        sep = {nm: out['layers'][l]['rows'][nm]['KL'] + out['layers'][l + 1]['rows'][nm]['KL'] for nm in ('LIN1', 'LIN2', 'LIN4', 'LIN8') if nm in out['layers'][l]['rows'] and nm in out['layers'][l + 1]['rows']}
        out['merges'].append(dict(layers=[l + 1, l + 2], gauss_check_KL_mean=float(np.mean(kls)), gauss_check_KL_max=float(np.max(kls)),
                                  expansion=float((J ** 2).sum((1, 2)).mean() / D), merged=mrows, separate_LIN=sep))
        print('merge', l + 1, l + 2, 'gauss check KL %.3f (max %.3f)' % (np.mean(kls), np.max(kls)), {k: (round(v['KL'], 3), v['cost']) for k, v in mrows.items()}, 'sep', {k: round(v, 3) for k, v in sep.items()}, flush=True)
    # ---- DP over cut points: segments of 1 (single-layer options) or 2 (merged options) layers
    for beta in (1e-4, 1e-3, 1e-2, 1e-1):
        best = [0.0] + [np.inf] * L; arg = [None] * (L + 1)
        for j in range(1, L + 1):
            o1 = min(out['layers'][j - 1]['rows'].items(), key=lambda kv: kv[1]['KL'] + beta * kv[1]['cost'])
            c1 = best[j - 1] + o1[1]['KL'] + beta * o1[1]['cost']
            if c1 < best[j]: best[j], arg[j] = c1, ('single', j, o1[0])
            if j >= 2:
                mg = out['merges'][j - 2]; o2 = min(mg['merged'].items(), key=lambda kv: kv[1]['KL'] + mg['gauss_check_KL_mean'] + beta * kv[1]['cost'])
                c2 = best[j - 2] + o2[1]['KL'] + mg['gauss_check_KL_mean'] + beta * o2[1]['cost']
                if c2 < best[j]: best[j], arg[j] = c2, ('merge', j - 1, j, o2[0])
        seg, j = [], L
        while j > 0:
            a = arg[j]; seg.append(a); j -= 1 if a[0] == 'single' else 2
        out['dp'].append(dict(beta=beta, objective=float(best[L]), segments=[list(map(str, s)) for s in reversed(seg)]))
        print('DP beta', beta, best[L], list(reversed(seg)), flush=True)
    out['seconds'] = time.time() - t_all
    json.dump(out, open(os.path.join(OUT, f'sur_small_s{sigma}.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    run(float(sys.argv[1]))
