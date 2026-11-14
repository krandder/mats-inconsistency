"""Decisive test on a TRAINED small noisy net (6 layers, width 8, sigma 0.5; same net as E2's 'trained'):
certified-style bounds on rare outputs from the forward chain vs the truth vs sampling at equal compute, bits of rarity
lost split into coverage cost per layer and waste; and per-layer diagnosis of where the waste lands.

Event A_t = {y > t}, y = w.h_L + b (the trained linear readout of the noisy last layer).
Truth: subset simulation over u = (x, all noise) (validated by MC). Chain: fixed-size S Gaussian cells + delta defensive
(e2.fit_mix), c_l = D_inf(K q_{l-1} || q_l) (search estimate, l = 1..L-1); bound(t) = 2^(sum c) (K_L q_{L-1})(A_t),
the last layer pushed through the TRUE kernel (closed form given h_{L-1}).
Diagnosis: P_l(A_t) = (K_{l+1..L} q_l)(A_t) (q_l pushed through the true remaining layers, by subset simulation over
(component, z, noise)); log2(P_l / p) = event mass that the description of layer l adds.

usage: python dec.py WIDTH SIGMA -> results/dec_d{W}_s{S}.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.stats import norm, beta, qmc
import dedu as Dd
import e2 as E2

torch.set_num_threads(1)
OUT = E2.OUT; L, DIN = E2.L, E2.DIN


def train_net_ro(d, sigma, seed=0, steps=5000):
    """Same training as e2.train_net (same seeds), also returning the readout."""
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
    net = E2.TNet(d, 1.0, 0, [w.detach().numpy() for w in W], [b.detach().numpy() for b in bb])
    return net, ro.weight.detach().numpy()[0], float(ro.bias.detach().numpy()[0])


class Path:
    """Forward simulation of the net from layer l0 (state h given) with explicit noise; score y."""

    def __init__(self, net, w, b, sigma):
        self.net, self.w, self.b, self.sg, self.d = net, w, b, sigma, net.d

    def from_layer(self, l0, h, noise):           # h: state after layer l0 (noisy); noise: (n, L - l0, d)
        for j, l in enumerate(range(l0, L)):
            h = self.net.f(l, h) + self.sg * noise[:, j]
        return h @ self.w + self.b

    def full(self, u):                            # u: (n, DIN + L*d)
        x = u[:, :DIN]; noise = u[:, DIN:].reshape(len(u), L, self.d)
        h = x
        for l in range(L):
            h = self.net.f(l, h) + self.sg * noise[:, l]
        return h @ self.w + self.b


def sus(score_fn, sample_fn, move_fn, rng, N=2000, p0=0.1, max_levels=13):
    """Generic subset simulation; state = tuple of arrays. Returns sorted (t, p) curve, passes."""
    st = sample_fn(N, rng); s = score_fn(st); passes = N; pk = 1.0; curve = []; rho = 0.8; last = 0.3
    for k in range(max_levels):
        o = np.argsort(-s); nk = int(p0 * N); g = s[o[nk - 1]]
        for t in np.quantile(s, np.linspace(0, 1 - p0, 12)):
            curve.append((float(t), pk * float((s > t).mean())))
        pk *= p0
        seeds = tuple(a[o[:nk]] for a in st); ss = s[o[:nk]]
        reps = N // nk; cur = tuple(np.repeat(a, reps, 0) for a in seeds); cs = np.repeat(ss, reps)
        rho = min(0.98, rho + 0.05) if last < 0.2 else (max(0.5, rho - 0.05) if last > 0.5 else rho)
        acc = 0
        for step in range(8):
            prop = move_fn(cur, rho, rng); ps = score_fn(prop); passes += len(ps)
            ok = ps >= g; acc += ok.mean()
            cur = tuple(np.where(ok.reshape((-1,) + (1,) * (a.ndim - 1)), b_, a) for a, b_ in zip(cur, prop)); cs = np.where(ok, ps, cs)
        st, s = cur, cs; last = acc / 8
    for t in np.quantile(s, np.linspace(0, 1, 12)):
        curve.append((float(t), pk * float((s > t).mean())))
    curve = sorted(curve)
    return curve, passes


def interp_p(curve, t):
    ts = np.array([c[0] for c in curve]); ps = np.array([c[1] for c in curve])
    ps = np.maximum.accumulate(ps[::-1])[::-1]
    lp = np.interp(t, ts, np.log(np.maximum(ps, 1e-300)), left=0.0, right=-np.inf)
    return np.exp(lp)


def run(d, sigma):
    t_start = time.time()
    net, w, b = train_net_ro(d, sigma); path = Path(net, w, b, sigma); s2 = sigma ** 2
    rng = np.random.default_rng(0); D = DIN + L * d
    # --- truth
    t0 = time.time()
    nmc = 2 ** 21; ymc = np.concatenate([path.full(rng.standard_normal((2 ** 16, D))) for _ in range(nmc // 2 ** 16)])
    rate = nmc / (time.time() - t0)
    pcn = lambda cur, rho, g: (rho * cur[0] + math.sqrt(1 - rho ** 2) * g.standard_normal(cur[0].shape),)
    truth = []
    for rep in range(3):
        c, passes = sus(lambda st: path.full(st[0]), lambda n, g: (g.standard_normal((n, D)),), pcn, np.random.default_rng(10 + rep))
        truth.append(c)
    yq = np.quantile(ymc, 1 - 2.0 ** -6)
    tmax = min(max(t for t, p in c if p > 0) for c in truth)
    tg = np.linspace(yq, tmax, 400)
    Pg = np.exp(np.mean([np.log(np.maximum(interp_p(c, tg), 1e-300)) for c in truth], 0))
    ks = [k for k in range(6, 42, 2) if Pg.min() <= 2.0 ** -k]
    ts = np.array([tg[np.argmin(np.abs(np.log2(Pg) + k))] for k in ks])
    P = np.exp(np.mean([np.log(np.maximum(interp_p(c, ts), 1e-300)) for c in truth], 0))
    Pmc = np.array([(ymc > t).mean() for t in ts])
    res = dict(d=d, sigma=sigma, t=ts.tolist(), p_true=P.tolist(), p_mc=Pmc.tolist(), nmc=nmc, rate=rate,
               p_true_reps=[interp_p(c, ts).tolist() for c in truth], chains={})
    print('truth: t range', ts[0], ts[-1], 'p', P[0], P[-1], flush=True)
    # --- chains
    for S in (16, 256):
        t1 = time.time(); q = Dd.GM([1.0], [np.zeros(DIN)], [np.eye(DIN)]); cs = []; qs = []
        for l in range(L - 1):
            n_per = max(128, 2 ** 15 // len(q.w)); pts, wq, _ = q.nodes(n_per, seed=10 + l)
            C = net.f(l, pts); Kq = Dd.GM(wq, C, iso=s2)
            qt, qn = E2.fit_mix(C, wq, s2, S, seed=l)
            cand = C[rng.choice(len(C), 2500, p=wq / wq.sum())] + sigma * rng.standard_normal((2500, d))
            c, _ = Dd.sup_log_ratio(Kq, qn, cand, steps=20, top=16)
            pq, wqq, _ = qn.nodes(max(16, 4096 // len(qn.w)), seed=7)
            waste = float(wqq @ ((qn.logpdf(pq) - Kq.logpdf(pq)) / Dd.LOG2)) / wqq.sum()
            cs.append(c); qs.append(qn); q = qn
            print(f'S {S} layer {l + 1}: c {c:.2f} waste {waste:.2f}', flush=True)
        # last layer through the true kernel: (K_L q_{L-1})(A_t) = E_q Phi((w.f_L(h) + b - t) / (sigma |w|))
        pts, wq, _ = q.nodes(max(512, 2 ** 16 // len(q.w)), seed=99)
        m = net.f(L - 1, pts) @ w + b; sdw = sigma * np.linalg.norm(w)
        KqA = np.array([(wq @ norm.sf((t - m) / sdw)) / wq.sum() for t in ts])
        t_chain = time.time() - t1
        eps = float(sum(cs)); bound = np.minimum(1.0, 2.0 ** eps * KqA)
        n_eq = int(rate * t_chain)
        samp = np.array([beta.ppf(0.95, int(round(p * n_eq)) + 1, n_eq - int(round(p * n_eq))) for p in P])
        # --- diagnosis: P_l(A) = q_l pushed through the true remaining layers (subset simulation)
        diag = []
        for l in range(1, L):
            ql = qs[l - 1]; nrest = L - l
            def samp_fn(n, g, ql=ql, nrest=nrest):
                c = g.choice(len(ql.w), n, p=ql.w); return (c, g.standard_normal((n, d)), g.standard_normal((n, nrest, d)))
            def score(st, ql=ql, l=l):
                c, z, nz = st; h = ql.mu[c] + np.einsum('nde,ne->nd', ql.L[c], z); return path.from_layer(l, h, nz)
            def move(cur, rho, g, ql=ql):
                c, z, nz = cur; c2 = c.copy(); swap = g.random(len(c)) < 0.2
                c2[swap] = g.choice(len(ql.w), swap.sum(), p=ql.w)            # independence proposal from the weights
                return (c2, rho * z + math.sqrt(1 - rho ** 2) * g.standard_normal(z.shape), rho * nz + math.sqrt(1 - rho ** 2) * g.standard_normal(nz.shape))
            cv, _ = sus(score, samp_fn, move, np.random.default_rng(50 + l))
            Pl = interp_p(cv, ts)
            diag.append(dict(layer=l, P_l=Pl.tolist()))
            print(f'S {S} diag layer {l}: log2(P_l/p):', [round(float(np.log2(max(Pl[i], 1e-300) / P[i])), 1) for i in range(len(ts))], flush=True)
        res['chains'][S] = dict(c=cs, eps=eps, KqA=KqA.tolist(), bound=bound.tolist(), t_chain=t_chain, n_eq=n_eq, samp_bound=samp.tolist(),
                                bits_lost=np.log2(np.maximum(bound, 1e-300) / P).tolist(), bits_waste=np.log2(np.maximum(KqA, 1e-300) / P).tolist(),
                                diag=diag)
        json.dump(res, open(os.path.join(OUT, f'dec_d{d}_s{sigma}.json'), 'w'), indent=1)
    print('done', time.time() - t_start, flush=True)


if __name__ == '__main__':
    run(int(sys.argv[1]), float(sys.argv[2]))
