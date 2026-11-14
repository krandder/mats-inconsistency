"""E1: the 5-neuron exact case. x ~ N(0, I_2); h1 = LN0(ReLU(W1 x + b1)) + s e1; h2 = LN0(ReLU(W2 h1 + b2)) + s e2;
y = w.h2. Deductive chains q0 -> q1 -> q2 per family; scorecard c_l, w_l, compute; certified bounds on p(y > t).

usage: python e1.py SIGMA SEED   -> results/e1_s{SIGMA}_n{SEED}.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.stats import norm, qmc, beta
import dedu as Dd

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DELTA = 0.01


def ln0relu(z):
    a = np.maximum(z, 0); c = a - a.mean(-1, keepdims=True)
    n = np.linalg.norm(c, axis=-1, keepdims=True)
    return np.where(n > 1e-9, c * (math.sqrt(z.shape[-1]) / np.maximum(n, 1e-9)), 0.0)


class Net:
    def __init__(self, seed):
        g = np.random.default_rng(seed)
        self.W1 = g.standard_normal((5, 2)) * 1.0; self.b1 = g.standard_normal(5) * 0.5
        self.W2 = g.standard_normal((5, 5)) * math.sqrt(2 / 5); self.b2 = g.standard_normal(5) * 0.3
        w = g.standard_normal(5); w -= w.mean(); self.w = w / np.linalg.norm(w)

    def f1(self, x): return ln0relu(x @ self.W1.T + self.b1)
    def f2(self, h): return ln0relu(h @ self.W2.T + self.b2)


class ProdCode:
    """Neuron code: z = B h + b, z split into groups; q(h) = |det B| prod_g GM_g(z_g)."""

    def __init__(self, B, b, groups, gms):
        self.B, self.b, self.groups, self.gms = B, b, [list(g) for g in groups], gms
        self.Bi = np.linalg.inv(B); self.ldB = np.linalg.slogdet(B)[1]; self.D = B.shape[0]

    def logpdf(self, H):
        Z = H @ self.B.T + self.b
        return self.ldB + sum(g.logpdf(Z[:, idx]) for idx, g in zip(self.groups, self.gms))

    def logpdf_t(self, H):
        Z = H @ torch.as_tensor(self.B.T) + torch.as_tensor(self.b)
        return self.ldB + sum(g.logpdf_t(Z[:, idx]) for idx, g in zip(self.groups, self.gms))

    def nodes(self, n, seed=0):
        u = qmc.Sobol(self.D, scramble=True, seed=seed).random(n)
        Z = np.zeros((n, self.D)); col = 0
        for idx, g in zip(self.groups, self.gms):
            k = len(idx); uu = u[:, col:col + k]; col += k
            cw = np.cumsum(g.w); c = np.minimum(np.searchsorted(cw, uu[:, 0] * cw[-1]), len(g.w) - 1)
            lo = np.r_[0, cw[:-1]][c]; v = np.clip((uu[:, 0] * cw[-1] - lo) / g.w[c], 1e-9, 1 - 1e-9)
            e = norm.ppf(np.c_[v, np.clip(uu[:, 1:], 1e-9, 1 - 1e-9)])
            Z[:, idx] = g.mu[c] + np.einsum('nde,ne->nd', g.L[c], e)
        return (Z - self.b) @ self.Bi.T, np.full(n, 1 / n), np.zeros(n, int)

    def prob_halfspace(self, a, t):
        """Only for a = first row of B with group (0,) a singleton: q(z_0 > t - b_0)."""
        assert np.allclose(a, self.B[0]) and list(self.groups[0]) == [0]
        return self.gms[0].prob_halfspace(np.ones(1), t + self.b[0])


def gm_from_groups(centres, wts, s2, labels):
    comps = Dd.node_moments(centres, wts, s2, labels)
    return Dd.GM([c[1] for c in comps], [c[2] for c in comps], [c[3] for c in comps])


def gm1d_quantile(z, wts, extra_var, nb=24):
    """1-D GM from weighted points: quantile bins, Gaussian per bin (+ extra_var)."""
    o = np.argsort(z); z, wts = z[o], wts[o]; cw = np.cumsum(wts) / wts.sum()
    lab = np.minimum((cw * nb).astype(int), nb - 1)
    return gm_from_groups(z[:, None], wts, extra_var, lab)


def fit_family(fam, centres, wts, s2, ctx):
    """Fit family member q~ to the node mixture sum wts N(centres, s2 I) (= K q). ctx carries labels/basis."""
    D = centres.shape[1]
    if fam == 'G':
        _, m, S = Dd.node_moments(centres, wts, s2); return Dd.GM([1.0], [m], [S])
    if fam in ('XG', 'PAT'):
        return gm_from_groups(centres, wts, s2, ctx['labels'])
    if fam.startswith('SPARSE'):
        return fit_sparse(centres, wts, s2, k=2, F=int(fam[6:]))
    if fam == 'NEU':
        B, b, groups = ctx['B'], ctx['b'], ctx['groups']
        Z = centres @ B.T + b; Sz = s2 * B @ B.T
        gms = []
        for idx in groups:
            idx = list(idx)
            if len(idx) == 1:
                gms.append(gm1d_quantile(Z[:, idx[0]], wts, Sz[idx[0], idx[0]]))
            else:
                _, m, S = Dd.node_moments(Z[:, idx], wts, 0.0)
                gms.append(Dd.GM([1.0], [m], [S + Sz[np.ix_(idx, idx)]]))
        return ProdCode(B, b, groups, gms)
    raise ValueError(fam)


def fit_sparse(centres, wts, s2, k, F, steps=1500, seed=0):
    from fam import TopKSAE
    torch.manual_seed(seed); D = centres.shape[1]
    g = torch.Generator().manual_seed(seed)
    C = torch.as_tensor(centres, dtype=torch.float32); P = torch.as_tensor(wts / wts.sum(), dtype=torch.float32)
    m = TopKSAE(D, F, k)
    with torch.no_grad():
        m.bd.copy_((P[:, None] * C).sum(0)); idx = torch.multinomial(P, F, generator=g)
        d = C[idx] - m.bd; d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-6); m.Dm.copy_(d); m.E.weight.copy_(d); m.E.bias.zero_()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        b = C[torch.multinomial(P, 1024, replacement=True, generator=g)]
        rec, _ = m(b); loss = ((rec - b) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): m.Dm.div_(m.Dm.norm(dim=1, keepdim=True).clamp_min(1e-6))
    with torch.no_grad():
        rec, A = m(C)
    A = A.numpy().astype(np.float64); rec = rec.numpy().astype(np.float64); w = wts / wts.sum()
    se2 = float(w @ ((centres - rec) ** 2).mean(1))
    sup = np.sort(np.argsort(-A, 1)[:, :k], 1); sup = np.where(np.take_along_axis(A, sup, 1) > 0, sup, -1)
    keys, inv = np.unique(sup, axis=0, return_inverse=True); inv = inv.ravel()
    Dm = m.Dm.detach().numpy().astype(np.float64); bd = m.bd.detach().numpy().astype(np.float64)
    ws, mus, Ss = [], [], []
    for c, key in enumerate(keys):
        s = inv == c; W = w[s].sum()
        if W <= 0: continue
        act = [i for i in key if i >= 0]
        a = A[s][:, act] if act else np.zeros((s.sum(), 0))
        am = (w[s] @ a) / W if act else np.zeros(0)
        av = (w[s] @ (a - am) ** 2) / W if act else np.zeros(0)
        Dk = Dm[act] if act else np.zeros((0, D))
        mus.append(bd + am @ Dk); Ss.append(Dk.T @ np.diag(av) @ Dk + (se2 + s2) * np.eye(D)); ws.append(W)
    q = Dd.GM(ws, mus, Ss); q.nsup = len(ws); q.se2 = se2
    return q


class KnowNothing:
    """Uniform on the LN0 sphere (radius sqrt(d) in the centred (d-1)-subspace) convolved with N(0, s2 I_d).
    Torch log-density implemented for d = 5 (Bessel order (d-1)/2 - 1 = 1)."""

    def __init__(self, d, s2):
        self.d, self.s2, self.D, self.R, self.n = d, s2, d, math.sqrt(d), d - 1
        self.u = np.ones(d) / math.sqrt(d); self.w = np.ones(1)

    def logpdf(self, H):
        from scipy.special import ive, gammaln
        hp = H @ self.u; rho = np.linalg.norm(H - hp[:, None] * self.u, axis=1)
        n, R, s2 = self.n, self.R, self.s2; v = n / 2 - 1; k = np.maximum(rho * R / s2, 1e-12)
        return (-n / 2 * math.log(2 * math.pi * s2) - (rho ** 2 + R ** 2) / (2 * s2) + gammaln(n / 2) + v * (math.log(2) - np.log(k))
                + np.log(ive(v, k)) + k - 0.5 * math.log(2 * math.pi * s2) - hp ** 2 / (2 * s2))

    def logpdf_t(self, H):
        assert self.d == 5
        u = torch.as_tensor(self.u); hp = H @ u; rho = (H - hp[:, None] * u).norm(dim=1)
        n, R, s2 = self.n, self.R, self.s2; k = (rho * R / s2).clamp_min(1e-12)
        return (-n / 2 * math.log(2 * math.pi * s2) - (rho ** 2 + R ** 2) / (2 * s2) + math.lgamma(n / 2) + (math.log(2) - torch.log(k))
                + torch.log(torch.special.i1e(k)) + k - 0.5 * math.log(2 * math.pi * s2) - hp ** 2 / (2 * s2))

    def nodes(self, n, seed=0):
        z = norm.ppf(np.clip(qmc.Sobol(2 * self.d, scramble=True, seed=seed).random(n), 1e-9, 1 - 1e-9))
        a = z[:, :self.d] - z[:, :self.d].mean(1, keepdims=True)
        sph = a * (self.R / np.linalg.norm(a, axis=1, keepdims=True))
        return sph + math.sqrt(self.s2) * z[:, self.d:], np.full(n, 1 / n), np.zeros(n, int)

    def prob_halfspace(self, a, t):
        pts, w, _ = self.nodes(2 ** 15, seed=5)
        return float(w @ norm.sf((t - (pts - math.sqrt(self.s2) * 0) @ a) / 1e-12) if False else w @ (pts @ a > t))


def defensive(centres, wts, s2):
    if os.environ.get('E1_DEF') == 'kn':
        return KnowNothing(centres.shape[1], s2)
    _, m, S = Dd.node_moments(centres, wts, s2)
    return Dd.GM([1.0], [m], [4 * S])


def withdef(qt, Wd):
    if isinstance(qt, ProdCode) or isinstance(Wd, KnowNothing):
        return MixAny(qt, Wd, DELTA)
    return Dd.mix(qt, Wd, DELTA)


class MixAny:
    def __init__(self, a, b, delta): self.a, self.b, self.d = a, b, delta; self.D = b.D

    def logpdf(self, H): return np.logaddexp(math.log(1 - self.d) + self.a.logpdf(H), math.log(self.d) + self.b.logpdf(H))

    def logpdf_t(self, H): return torch.logaddexp(math.log(1 - self.d) + self.a.logpdf_t(H), math.log(self.d) + self.b.logpdf_t(H))

    def nodes(self, n, seed=0):
        na = int(round(n * (1 - self.d))); nb = max(64, n - na)
        ka = len(self.a.w) if isinstance(self.a, Dd.GM) else 1
        pa, wa, ca = self.a.nodes(max(16, na // ka) if isinstance(self.a, Dd.GM) else na, seed); pb, wb, _ = self.b.nodes(nb, seed + 1)
        wa = wa / wa.sum(); wb = wb / wb.sum()
        return np.r_[pa, pb], np.r_[(1 - self.d) * wa, self.d * wb], np.r_[ca, np.full(len(pb), ka)]

    def prob_halfspace(self, a, t): return (1 - self.d) * self.a.prob_halfspace(a, t) + self.d * self.b.prob_halfspace(a, t)


def per_comp(q, n):
    return max(16, n // len(q.w)) if isinstance(q, Dd.GM) else n


def scorecard(Kq, q, qt, cand_pts, nq=2 ** 13):
    t = time.time()
    c, c_cand = Dd.sup_log_ratio(Kq, q, cand_pts)
    c0, _ = Dd.sup_log_ratio(Kq, qt, cand_pts)
    pts, w, _ = q.nodes(per_comp(q, nq), seed=7)
    d = (q.logpdf(pts) - Kq.logpdf(pts)) / Dd.LOG2
    waste = float(w @ d) / float(w.sum())
    return dict(c=c, c_nodef=c0, waste=waste, t_score=time.time() - t)


def push_nodes(q, f, n_per, seed):
    pts, w, comp = q.nodes(n_per, seed)
    return f(pts), w, comp, pts


def run(sigma, seed):
    s2 = sigma ** 2; net = Net(seed); res = dict(sigma=sigma, seed=seed, delta=DELTA, layers={})
    # q0 = p0: product grid on x (exact up to quadrature)
    ng = 150; xs = np.linspace(-6.5, 6.5, ng); X = np.stack(np.meshgrid(xs, xs, indexing='ij'), -1).reshape(-1, 2)
    wx = norm.pdf(X).prod(1); wx /= wx.sum()
    c1 = net.f1(X)
    P1 = Dd.GM(wx, c1, iso=s2)                                     # p1 = K q0 exactly (quadrature)
    # validation truth p2 = K p1 (nodes: coarser x grid x Sobol noise)
    ngv = 80; xv = np.linspace(-6.5, 6.5, ngv); Xv = np.stack(np.meshgrid(xv, xv, indexing='ij'), -1).reshape(-1, 2)
    wv = norm.pdf(Xv).prod(1); wv /= wv.sum()
    ev = norm.ppf(np.clip(qmc.Sobol(5, scramble=True, seed=3).random(16), 1e-9, 1 - 1e-9))
    H1v = (net.f1(Xv)[:, None] + sigma * ev[None]).reshape(-1, 5); w1v = np.repeat(wv / 16, 16)
    P2 = Dd.GM(w1v, net.f2(H1v), iso=s2)
    # labels for named-grid (input cells) and patterns
    def xcells(r):
        qx = norm.ppf(np.linspace(0, 1, r + 1)[1:-1])
        return np.searchsorted(qx, X[:, 0]) * r + np.searchsorted(qx, X[:, 1])
    pat1 = ((X @ net.W1.T + net.b1) > 0) @ (1 << np.arange(5))
    # neuron-code basis at layer 1 = layer-2 pre-activations; groups from weights: |corr| of z2 under q's Gaussian moments
    _, m1, S1 = Dd.node_moments(c1, wx, s2)
    Cz = net.W2 @ S1 @ net.W2.T; Rz = Cz / np.sqrt(np.outer(np.diag(Cz), np.diag(Cz)))
    groups = union_groups(np.abs(Rz) > 0.5)
    # basis at layer 2: readout + orthonormal complement
    Q, _ = np.linalg.qr(np.c_[net.w, np.random.default_rng(0).standard_normal((5, 4))]); B2 = Q.T; B2[0] = net.w
    ctx1 = {'PAT': dict(labels=pat1), 'NEU': dict(B=net.W2, b=net.b2, groups=groups), 'G': {}, 'SPARSE10': {}, 'SPARSE40': {}}
    for r_ in (2, 4, 8, 16): ctx1[f'XG{r_}'] = dict(labels=xcells(r_))
    res['neu_groups'] = [list(map(int, g)) for g in groups]
    cand1 = np.r_[c1[np.random.default_rng(1).choice(len(c1), 3000, p=wx)] + sigma * np.random.default_rng(2).standard_normal((3000, 5))]
    W1def = defensive(c1, wx, s2)
    # thresholds for rare events from the truth p2 (y = w.h2)
    ts = np.linspace(0.0, 4.5, 31)
    pA = np.array([P2.prob_halfspace(net.w, t) for t in ts])
    res['events'] = dict(t=ts.tolist(), p_true=pA.tolist())
    for name in ('G', 'XG2', 'XG4', 'XG8', 'XG16', 'PAT', 'NEU', 'SPARSE10', 'SPARSE40'):
        fam = 'XG' if name.startswith('XG') else name
        t0 = time.time(); r = {}
        qt1 = fit_family(fam, c1, wx, s2, ctx1[name]); q1 = withdef(qt1, W1def)
        r['n_comp1'] = int(len(getattr(qt1, 'w', [])))
        r['fit1_s'] = time.time() - t0
        sc1 = scorecard(P1, q1, qt1, cand1); r['L1'] = sc1
        # push q1 through layer 2 (nodes of q1, never p)
        npush = 2 ** 15
        t1 = time.time()
        if fam == 'XG':
            pts, w, comp = q1.nodes(512 if isinstance(q1, Dd.GM) else 512 * (len(q1.a.w) + 1), seed=11); c2 = net.f2(pts); lab2 = comp   # cells carried
        elif fam == 'PAT':
            pts, w, comp = q1.nodes(npush // max(1, len(q1.w)) + 64 if hasattr(q1, 'w') else npush, seed=11)
            c2 = net.f2(pts); lab2 = ((pts @ net.W2.T + net.b2) > 0) @ (1 << np.arange(5))
        else:
            pts, w, comp = q1.nodes(npush if fam == 'NEU' else max(npush // max(1, len(getattr(q1, 'w', [1]))), 512), seed=11)
            c2 = net.f2(pts); lab2 = None
        Kq1 = Dd.GM(w, c2, iso=s2); r['n_push'] = int(len(w)); r['t_push'] = time.time() - t1
        # event bound through the true last kernel (no refit at the output): p2(A) <= 2^c1 * (K q1)(A)
        r['KqA'] = [float((w @ norm.sf((t - c2 @ net.w) / sigma)) / w.sum()) for t in ts]
        ctx2 = {'G': {}, 'XG': dict(labels=lab2), 'PAT': dict(labels=lab2), 'SPARSE10': {}, 'SPARSE40': {},
                'NEU': dict(B=B2, b=np.zeros(5), groups=[(0,), (1, 2, 3, 4)])}
        qt2 = fit_family(fam, c2, w, s2, ctx2[fam if fam != 'SPARSE10' and fam != 'SPARSE40' else name]); q2 = withdef(qt2, defensive(c2, w, s2))
        cand2 = c2[np.random.default_rng(4).choice(len(c2), 3000, p=w / w.sum())] + sigma * np.random.default_rng(5).standard_normal((3000, 5))
        sc2 = scorecard(Kq1, q2, qt2, cand2); r['L2'] = sc2
        r['t_total'] = time.time() - t0
        eps = sc1['c'] + sc2['c']
        qA = np.array([q2.prob_halfspace(net.w, t) for t in ts])
        r['eps'] = eps; r['q2A'] = qA.tolist(); r['bound'] = np.minimum(1, 2 ** eps * qA).tolist()
        r['bound_push'] = np.minimum(1, 2 ** sc1['c'] * np.array(r['KqA'])).tolist()
        # validation against the truth (uses p; reported separately)
        v1 = Dd.sup_log_ratio(P1, q1, cand1)[0]
        cand2v = net.f2(H1v[np.random.default_rng(6).choice(len(H1v), 3000, p=w1v)]) + sigma * np.random.default_rng(7).standard_normal((3000, 5))
        v2 = Dd.sup_log_ratio(P2, q2, np.r_[cand2v, cand2])[0]
        pts, wq, _ = q2.nodes(per_comp(q2, 2 ** 12), seed=9); kl_q2p2 = float(wq @ ((q2.logpdf(pts) - P2.logpdf(pts)) / Dd.LOG2))
        r['val'] = dict(Dinf_p1_q1=v1, Dinf_p2_q2=v2, KL_q2_p2=kl_q2p2)
        # sampling at equal wall time: forward simulations of the real noisy net
        n_eq = sampling_budget(net, sigma, r['t_total'])
        r['n_samp_eq'] = n_eq
        r['samp_bound'] = [cp_upper(k_exp(p, n_eq), n_eq) for p in pA]
        r['n_comp2'] = int(len(getattr(qt2, 'w', [])))
        res['layers'][name] = r
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ('q2A', 'bound', 'samp_bound')},
              'L1', {k: round(v, 3) for k, v in sc1.items()}, 'L2', {k: round(v, 3) for k, v in sc2.items()}, flush=True)
        json.dump(res, open(os.path.join(OUT, f'e1_s{sigma}_n{seed}' + ('_kn' if os.environ.get('E1_DEF') == 'kn' else '') + '.json'), 'w'), indent=1)
    return res


def union_groups(adj):
    n = len(adj); parent = list(range(n))
    def f(i):
        while parent[i] != i: i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]: parent[f(i)] = f(j)
    gs = {}
    for i in range(n): gs.setdefault(f(i), []).append(i)
    return [tuple(v) for v in gs.values()]


def sampling_budget(net, sigma, seconds):
    """Number of forward simulations of the noisy net that fit in `seconds` (measured)."""
    g = np.random.default_rng(0); n = 200_000; t = time.time()
    x = g.standard_normal((n, 2)); h1 = net.f1(x) + sigma * g.standard_normal((n, 5)); h2 = net.f2(h1) + sigma * g.standard_normal((n, 5)); y = h2 @ net.w
    rate = n / (time.time() - t)
    return int(rate * seconds)


def k_exp(p, n):
    return int(round(p * n))                                         # expected count (typical-case sampling outcome)


def cp_upper(k, n, a=0.05):
    return 1.0 if n == 0 else float(beta.ppf(1 - a, k + 1, n - k)) if k < n else 1.0


if __name__ == '__main__':
    run(float(sys.argv[1]), int(sys.argv[2]))
