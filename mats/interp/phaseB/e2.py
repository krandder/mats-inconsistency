"""E2: fixed-size families with re-projection after every layer; waste / coverage vs family size S and depth;
random vs trained; know-nothing cover as the upper reference.

Net: x ~ N(0, I_4); h_l = LN0(tanh(g W_l h_{l-1} + b_l)) + s e_l (LN0: centre, scale to norm sqrt(d)), depth 6.
LN0 output lies on the (d-1)-sphere of radius sqrt(d) in the centred subspace, so per layer
    I(X;H_l) <= C = ((d-1)/2) log2(1 + d / ((d-1) s^2)).
Chain (deductive): q_0 = N(0, I) exactly; for l = 1..L: K q_{l-1} = node mixture (S x n_per QMC nodes of q_{l-1} pushed
through f_l, each N(., s^2 I)); q_l = MixS fitted to K q_{l-1} (weighted k-means cells of the node centres, Gaussian per
cell = cell moments + s^2 I) mixed with a delta = 0.01 defensive Gaussian. S is fixed across layers (never multiplies).
Per layer: waste_KL = KL(q_l || K q_{l-1}), coverage c = D_inf(K q_{l-1} || q_l) (search estimate), the same two for the
know-nothing cover (uniform on the LN0 sphere convolved with noise), compute time, expansion E_q ||J||_F^2/d.
Also I(X;H_l) of the real net (x-level Gaussian estimator; validation side, uses samples of p).

usage: python e2.py WIDTH SIGMA [nets]   -> results/e2_d{W}_s{S}.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.special import ive, gammaln
from scipy.stats import qmc, norm
import dedu as Dd
import mi

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DIN, L, DELTA = 4, 6, 0.01
SIZES = (1, 4, 16, 64, 256)


def ln0(a):
    c = a - a.mean(-1, keepdims=True); n = np.linalg.norm(c, axis=-1, keepdims=True)
    return c * (math.sqrt(a.shape[-1]) / np.maximum(n, 1e-9))


class TNet:
    def __init__(self, d, g, seed, Ws=None, bs=None):
        r = np.random.default_rng(seed); dims = [DIN] + [d] * L
        self.W = Ws or [r.standard_normal((dims[i + 1], dims[i])) / math.sqrt(dims[i]) for i in range(L)]
        self.b = bs or [r.standard_normal(dims[i + 1]) * 0.1 for i in range(L)]
        self.g, self.d = g, d

    def f(self, l, h): return ln0(np.tanh(self.g * h @ self.W[l].T + self.b[l]))

    def expansion(self, l, H):
        W = torch.tensor(self.W[l]); b = torch.tensor(self.b[l]); d = self.d
        def fn(h):
            a = torch.tanh(self.g * h @ W.T + b); c = a - a.mean(); return c * math.sqrt(d) / c.norm()
        vals = []
        for h in torch.tensor(H[:150]):
            J = torch.autograd.functional.jacobian(fn, h); vals.append(float((J ** 2).sum()) / J.shape[1])
        return float(np.mean(vals))


def train_net(d, sigma, seed=0, steps=5000):
    torch.manual_seed(seed); r = np.random.default_rng(seed); U = torch.tensor(r.standard_normal((4, DIN)))
    dims = [DIN] + [d] * L
    W = [torch.nn.Parameter(torch.tensor(r.standard_normal((dims[i + 1], dims[i])) / math.sqrt(dims[i]))) for i in range(L)]
    bb = [torch.nn.Parameter(torch.tensor(r.standard_normal(dims[i + 1]) * 0.1)) for i in range(L)]
    ro = torch.nn.Linear(d, 1).double(); opt = torch.optim.Adam(W + bb + list(ro.parameters()), lr=3e-3)
    def fwd(x):
        h = x
        for l in range(L):
            a = torch.tanh(h @ W[l].T + bb[l]); c = a - a.mean(1, keepdim=True)
            h = c * math.sqrt(d) / c.norm(dim=1, keepdim=True) + sigma * torch.randn_like(c)
        return ro(h)[:, 0]
    for s in range(steps):
        x = torch.randn(512, DIN, dtype=torch.float64); y = torch.sin(x @ U.T).sum(1)
        loss = ((fwd(x) - y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = torch.randn(20000, DIN, dtype=torch.float64); y = torch.sin(x @ U.T).sum(1)
        r2 = 1 - float(((fwd(x) - y) ** 2).mean() / y.var())
    return TNet(d, 1.0, 0, [w.detach().numpy() for w in W], [b.detach().numpy() for b in bb]), r2


class KnowNothing:
    """Uniform on the LN0 sphere (radius sqrt(d) in the centred (d-1)-subspace) convolved with N(0, s^2 I_d)."""

    def __init__(self, d, s2):
        self.d, self.s2, self.D = d, s2, d
        self.u = np.ones(d) / math.sqrt(d); self.R = math.sqrt(d); self.n = d - 1

    def logpdf(self, H):
        hp = H @ self.u; Hc = H - hp[:, None] * self.u; rho = np.linalg.norm(Hc, axis=1)
        n, R, s2 = self.n, self.R, self.s2; v = n / 2 - 1; k = np.maximum(rho * R / s2, 1e-12)
        lf = (-n / 2 * math.log(2 * math.pi * s2) - (rho ** 2 + R ** 2) / (2 * s2) + gammaln(n / 2)
              + v * (math.log(2) - np.log(k)) + np.log(ive(v, k)) + k)
        return lf - 0.5 * math.log(2 * math.pi * s2) - hp ** 2 / (2 * s2)

    def sample(self, n, rng):
        z = rng.standard_normal((n, self.d)); z -= z.mean(1, keepdims=True)
        s = z * (self.R / np.linalg.norm(z, axis=1, keepdims=True))
        return s + math.sqrt(self.s2) * rng.standard_normal((n, self.d))


def wkmeans(C, w, S, seed, iters=25):
    if S == 1:
        return np.zeros(len(C), int)
    g = torch.Generator().manual_seed(seed)
    Ct = torch.as_tensor(C, dtype=torch.float32); Wt = torch.as_tensor(w / w.sum(), dtype=torch.float32)
    idx = [int(torch.multinomial(Wt, 1, generator=g))]
    d2 = ((Ct - Ct[idx[0]]) ** 2).sum(1)
    for _ in range(S - 1):                                                # weighted k-means++
        p = Wt * d2; idx.append(int(torch.multinomial(p / p.sum(), 1, generator=g)))
        d2 = torch.minimum(d2, ((Ct - Ct[idx[-1]]) ** 2).sum(1))
    M = Ct[idx].clone()
    for _ in range(iters):
        lab = torch.cdist(Ct, M).argmin(1)
        for k in range(S):
            s = lab == k
            if s.any(): M[k] = (Wt[s, None] * Ct[s]).sum(0) / Wt[s].sum()
    return torch.cdist(Ct, M).argmin(1).numpy()


def fit_mix(C, w, s2, S, seed):
    lab = wkmeans(C, w, S, seed)
    comps = Dd.node_moments(C, w, s2, lab)
    qt = Dd.GM([c[1] for c in comps], [c[2] for c in comps], [c[3] for c in comps])
    _, m, St = Dd.node_moments(C, w, s2)
    return qt, Dd.mix(qt, Dd.GM([1.0], [m], [4 * St]), DELTA)


def chain(net, sigma, S, rng):
    s2 = sigma ** 2; rows = []
    q = Dd.GM([1.0], [np.zeros(DIN)], [np.eye(DIN)])
    kn = KnowNothing(net.d, s2)
    for l in range(L):
        t0 = time.time()
        n_per = max(128, 2 ** 15 // len(q.w))
        pts, w, _ = q.nodes(n_per, seed=10 + l)
        C = net.f(l, pts); Kq = Dd.GM(w, C, iso=s2)
        qt, qn = fit_mix(C, w, s2, S, seed=l)
        t_fit = time.time() - t0
        cand = C[rng.choice(len(C), 2500, p=w / w.sum())] + sigma * rng.standard_normal((2500, net.d))
        c, _ = Dd.sup_log_ratio(Kq, qn, cand, steps=20, top=16)
        pq, wq, _ = qn.nodes(max(16, 4096 // len(qn.w)), seed=7)
        waste = float(wq @ ((qn.logpdf(pq) - Kq.logpdf(pq)) / Dd.LOG2)) / wq.sum()
        hk = kn.sample(4096, rng); waste_kn = float(np.mean((kn.logpdf(hk) - Kq.logpdf(hk)) / Dd.LOG2))
        c_kn = float(np.max((Kq.logpdf(cand) - kn.logpdf(cand)) / Dd.LOG2))
        t_all = time.time() - t0
        ex = net.expansion(l + 1, pq) if l + 1 < L else float('nan')
        rows.append(dict(layer=l + 1, S=S, n_comp=int(len(qn.w)), n_nodes=int(len(w)), waste=waste, c=c, waste_kn=waste_kn,
                         c_kn=c_kn, t_fit=t_fit, t_all=t_all, expansion_next=ex))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        q = qn
    return rows


def real_mi(net, sigma, rng, Mx=2048, K=128):
    """I(X;H_l) per layer of the real noisy net (x-level Gaussian estimator, validation side)."""
    x = rng.standard_normal((Mx, DIN)); N = 1024
    def run(xx):
        out = []; h = xx
        for l in range(L):
            h = net.f(l, h) + sigma * rng.standard_normal((len(xx), net.d)); out.append(h)
        return out
    tgt = run(x[:N]); acc = [[] for _ in range(L)]
    for k in range(K):
        hs = run(x)
        for l in range(L): acc[l].append(hs[l])
    res = []
    for l in range(L):
        m, S = mi.moments_from_chains(np.stack(acc[l], 1) - 0, 0.0)
        r = mi.xgauss_bracket(tgt[l], m, S + 1e-9 * np.eye(net.d), np.arange(N)); res.append((r['xg_I_lo'], r['xg_I_hi']))
    return res


def run(d, sigma, which):
    out = dict(d=d, sigma=sigma, C=0.5 * (d - 1) * math.log2(1 + d / ((d - 1) * sigma ** 2)), nets={})
    rng = np.random.default_rng(0)
    nets = []
    if 'r1' in which: nets.append(('random_g1', TNet(d, 1.0, 0)))
    if 'r3' in which: nets.append(('random_g3', TNet(d, 3.0, 0)))
    if 'tr' in which:
        tn, r2 = train_net(d, sigma); out['trained_r2'] = r2; print('trained R2', r2, flush=True); nets.append(('trained', tn))
    path = os.path.join(OUT, f'e2_d{d}_s{sigma}.json')
    for name, net in nets:
        rec = dict(I=real_mi(net, sigma, rng), rows=[])
        print(name, 'I(X;H_l) xg', [(round(a, 2), round(b, 2)) for a, b in rec['I']], flush=True)
        for S in SIZES:
            rec['rows'] += chain(net, sigma, S, rng)
            out['nets'][name] = rec; json.dump(out, open(path, 'w'), indent=1)


if __name__ == '__main__':
    run(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'r1,r3,tr')
