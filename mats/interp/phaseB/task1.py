"""Task 1: information flow in a small random noisy network (phase B).

Net: x ~ N(0, I_din); s_l = LN0(ReLU(W_l h_{l-1} + b_l)) (LayerNorm without affine: centred, ||s_l||^2 = d);
h_l = s_l + sigma Z_l. s_l lies in the (d-1)-dim centred subspace with ||s||^2 = d, so
    I(X; H_l) <= C_l = ((d-1)/2) log2(1 + d / ((d-1) sigma^2))  bits.
Markov chain X -> H_1 -> H_2 -> H_3, so I(X;H_1) >= I(X;H_2) >= I(X;H_3).

usage: python task1.py validate | flow | coarse   (writes results/task1_*.json)
"""
import json, math, os, sys, time
import numpy as np
import torch
import mi

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'); os.makedirs(OUT, exist_ok=True)


def cap_ln(d, sigma):
    return 0.5 * (d - 1) * math.log2(1 + d / ((d - 1) * sigma ** 2))


class RandNet:
    def __init__(self, din, d, L, seed, gain=2.0 ** 0.5, bias=0.3):
        g = np.random.default_rng(seed)
        dims = [din] + [d] * L
        self.W = [g.standard_normal((dims[i + 1], dims[i])) * gain / math.sqrt(dims[i]) for i in range(L)]
        self.b = [g.standard_normal(dims[i + 1]) * bias for i in range(L)]
        self.L, self.d, self.din = L, d, din

    def pre(self, l, hprev):                        # pre-activation of layer l (1-based)
        return hprev @ self.W[l - 1].T + self.b[l - 1]

    def mu(self, l, hprev):                         # noise-free signal s_l given the previous noisy state
        a = np.maximum(self.pre(l, hprev), 0)
        a = a - a.mean(1, keepdims=True)
        return a * (math.sqrt(self.d) / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9))

    def run(self, x, sigma, rng):
        """Returns lists mus[l], hs[l], pats[l] for l = 1..L (index 0 = layer 1)."""
        mus, hs, pats = [], [], []
        h = x
        for l in range(1, self.L + 1):
            pats.append(self.pre(l, h) > 0)
            m = self.mu(l, h); h = m + sigma * rng.standard_normal(m.shape)
            mus.append(m); hs.append(h)
        return mus, hs, pats


def layer_mi(net, sigma, M, N, K, seed, din=None):
    """I(X;H_l) brackets for every layer. Bank of M episodes; the first N are targets; K inner parent chains per target."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((M, net.din))
    mus, hs, pats = net.run(x, sigma, rng)
    own = np.arange(N)
    # inner chains: re-run the first N inputs K times with fresh noise; centres for layer l come from h_{l-1} chains
    inner = [np.empty((N, K + 1, net.d)) for _ in range(net.L)]
    for l in range(net.L):
        inner[l][:, 0] = mus[l][:N]
    for k in range(K):
        mk, _, _ = net.run(x[:N], sigma, rng)
        for l in range(net.L):
            inner[l][:, k + 1] = mk[l]
    res = []
    for l in range(net.L):
        Cin = None if l == 0 else inner[l]
        r = mi.mi_bracket(hs[l][:N], mus[l], sigma, own, Cin)
        r['cap'] = cap_ln(net.d, sigma); r['layer'] = l + 1
        p = pats[l]; _, cnt = np.unique(p, axis=0, return_counts=True); pr = cnt / cnt.sum()
        r['H_pattern_plugin'] = float(-(pr * np.log2(pr)).sum()); r['n_patterns'] = int(len(cnt))
        res.append(r)
    return res, (x, mus, hs)


# ---------------------------------------------------------------- validation on linear-Gaussian channels (exact I)
def validate():
    out = []
    for D, din, scale in ((8, 8, 1.0), (8, 8, 3.0), (16, 16, 3.0), (16, 16, 10.0), (32, 32, 3.0)):
        g = np.random.default_rng(D * 100 + int(scale))
        A = g.standard_normal((D, din)) * scale / math.sqrt(din); B = g.standard_normal((D, D)) / math.sqrt(D)
        sigma = 1.0
        S1 = A @ A.T + sigma ** 2 * np.eye(D)
        I1 = 0.5 * np.linalg.slogdet(np.eye(D) + A @ A.T / sigma ** 2)[1] / math.log(2)
        S2 = B @ S1 @ B.T + sigma ** 2 * np.eye(D); S2c = sigma ** 2 * (B @ B.T + np.eye(D))
        I2 = 0.5 * (np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S2c)[1]) / math.log(2)
        for M in (2 ** 10, 2 ** 13, 2 ** 16):
            rng = np.random.default_rng(M)
            x = rng.standard_normal((M, din)); m1 = x @ A.T; h1 = m1 + sigma * rng.standard_normal(m1.shape)
            m2 = h1 @ B.T; h2 = m2 + sigma * rng.standard_normal(m2.shape)
            N, K = min(2048, M // 2), 256
            r1 = mi.mi_bracket(h1[:N], m1, sigma, np.arange(N))
            Cin = np.empty((N, K + 1, D)); Cin[:, 0] = m2[:N]
            for k in range(K):
                Cin[:, k + 1] = (m1[:N] + sigma * rng.standard_normal((N, D))) @ B.T
            r2 = mi.mi_bracket(h2[:N], m2, sigma, np.arange(N), Cin)
            row = dict(D=D, scale=scale, M=M, log2M=math.log2(M), I1=I1, I1_lo=r1['I_lo'], I1_hi=r1['I_hi'],
                       I2=I2, I2_lo=r2['I_lo'], I2_hi=r2['I_hi'], hc2_in=r2['hc_in'], hc2_out=r2['hc_out'],
                       hc2_true=0.5 * np.linalg.slogdet(2 * math.pi * math.e * S2c)[1] / math.log(2))
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            out.append(row)
    json.dump(out, open(os.path.join(OUT, 'task1_validate.json'), 'w'), indent=1)


def flow():
    out = []
    for din in (2, 8):
        for d in (8, 16, 32):
            for sigma in (0.1, 0.3, 1.0):
                for seed in (0, 1):
                    t = time.time()
                    net = RandNet(din, d, 3, seed=1000 * seed + d + din)
                    rs, _ = layer_mi(net, sigma, M=2 ** 15, N=2048, K=128, seed=seed + 7)
                    for r in rs:
                        r.update(din=din, d=d, sigma=sigma, seed=seed); out.append(r)
                    print(din, d, sigma, seed, [(round(r['I_lo'], 2), round(r['I_hi'], 2)) for r in rs],
                          'cap', round(rs[0]['cap'], 1), 'Hpat', [round(r['H_pattern_plugin'], 1) for r in rs],
                          f'{time.time() - t:.0f}s', flush=True)
                    json.dump(out, open(os.path.join(OUT, 'task1_flow.json'), 'w'), indent=1)


# ---------------------------------------------------------------- coarse-graining of input space
def lse_grouped(T, C, gT, gC, sigma, own):
    """Leave-in / leave-out log p_hat(t | group) with centres restricted to the target's group (nats)."""
    T = torch.as_tensor(T, dtype=torch.float64); C = torch.as_tensor(C, dtype=torch.float64)
    gT = torch.as_tensor(gT); gC = torch.as_tensor(gC); D = T.shape[1]
    nC = torch.bincount(gC, minlength=int(max(gC.max(), gT.max())) + 1).double()
    c2 = (C ** 2).sum(1); lin = np.empty(len(T)); lout = np.empty(len(T))
    norm = -0.5 * D * math.log(2 * math.pi * sigma ** 2)
    for i in range(0, len(T), 256):
        t = T[i:i + 256]
        lk = -((t ** 2).sum(1, keepdim=True) + c2[None] - 2 * t @ C.T).clamp_min(0) / (2 * sigma ** 2)
        lk = lk.masked_fill(gT[i:i + 256, None] != gC[None], -float('inf'))
        n = nC[gT[i:i + 256]]
        lin[i:i + 256] = (torch.logsumexp(lk, 1) - torch.log(n)).numpy() + norm
        o = torch.as_tensor(own[i:i + 256]); lk[torch.arange(len(t)), o] = -float('inf')
        lout[i:i + 256] = (torch.logsumexp(lk, 1) - torch.log((n - 1).clamp_min(1))).numpy() + norm
    return lin, lout


def kmeans(Z, k, seed, iters=30):
    g = np.random.default_rng(seed)
    C = Z[g.choice(len(Z), k, replace=False)].copy()
    Zt = torch.as_tensor(Z);
    for _ in range(iters):
        Ct = torch.as_tensor(C)
        lab = torch.cdist(Zt, Ct).argmin(1).numpy()
        for j in range(k):
            s = lab == j
            if s.any(): C[j] = Z[s].mean(0)
    return C, torch.cdist(Zt, torch.as_tensor(C)).argmin(1).numpy()


def coarse():
    """For nets where I(X;H_3) is in range: classes = k-means cells of the noise-averaged last-layer mean g(x)=E[H_3|x]
    (2^b cells). I(K;H_3) = h(H_3) - h(H_3|K) (both bracketed), share of I(X;H_3) captured, R^2 of g(x) by the
    cell mean; and the same with cells from a uniform grid on the input (input-space coarse-graining)."""
    out = []
    for din, d, sigma, seed in ((2, 16, 0.3, 0), (2, 32, 0.3, 0), (8, 16, 1.0, 0), (8, 32, 0.3, 0), (2, 8, 0.1, 0)):
        net = RandNet(din, d, 3, seed=1000 * seed + d + din)
        M, N = 2 ** 15, 2048
        rs, (x, mus, hs) = layer_mi(net, sigma, M=M, N=N, K=128, seed=seed + 7)
        I3 = (rs[2]['I_lo'], rs[2]['I_hi']); hH = (rs[2]['h_in'], rs[2]['h_out'])
        # noise-averaged mean of H_3 given x, from 32 inner chains
        rng = np.random.default_rng(99)
        g = np.mean([net.run(x, sigma, rng)[0][2] for _ in range(32)], 0)
        for b in range(1, 13):
            k = 2 ** b
            if k > M // 8: break
            _, lab = kmeans(g, k, seed=b)
            lin, lout = lse_grouped(hs[2][:N], mus[2], lab[:N], lab, sigma, np.arange(N))
            hc_in, hc_out = float(-lin.mean() / mi.LOG2), float(-lout.mean() / mi.LOG2)
            # I(K;H) bracket: h(H) in [h_in, h_out]; h(H|K) in [hc_in, hc_out]
            IK = (hH[0] - hc_out, hH[1] - hc_in)
            cm = np.zeros((k, d)); np.add.at(cm, lab, g); cnt = np.bincount(lab, minlength=k); cm /= np.maximum(cnt, 1)[:, None]
            r2 = 1 - ((g - cm[lab]) ** 2).sum() / ((g - g.mean(0)) ** 2).sum()
            _, cnt = np.unique(lab, return_counts=True); p = cnt / cnt.sum(); HK = float(-(p * np.log2(p)).sum())
            row = dict(din=din, d=d, sigma=sigma, b=b, HK=HK, IK_lo=IK[0], IK_hi=IK[1], I3_lo=I3[0], I3_hi=I3[1], r2_g=float(r2),
                       I1_lo=rs[0]['I_lo'], I1_hi=rs[0]['I_hi'], cap=rs[0]['cap'])
            out.append(row)
            print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
        json.dump(out, open(os.path.join(OUT, 'task1_coarse.json'), 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] not in ('xg', 'coarse2'): {'validate': validate, 'flow': flow, 'coarse': coarse}[sys.argv[1]]()


def xg():
    """x-level Gaussian-conditional I(X;H_l) for every flow config (validated on the linear-Gaussian case first)."""
    out = []
    # validation: linear 2-layer channel, exact I2
    for D, scale in ((8, 3.0), (16, 3.0), (32, 3.0)):
        g = np.random.default_rng(D * 100 + int(scale)); din = D; sigma = 1.0
        A = g.standard_normal((D, din)) * scale / math.sqrt(din); B = g.standard_normal((D, D)) / math.sqrt(D)
        S1 = A @ A.T + np.eye(D); S2 = B @ S1 @ B.T + np.eye(D); S2c = B @ B.T + np.eye(D)
        I2 = 0.5 * (np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S2c)[1]) / math.log(2)
        rng = np.random.default_rng(5); Mx, K = 4096, 256
        x = rng.standard_normal((Mx, din))
        C = np.stack([(x @ A.T + rng.standard_normal((Mx, D))) @ B.T for _ in range(K)], 1)
        m, S = mi.moments_from_chains(C, sigma)
        h = (x[:1024] @ A.T + rng.standard_normal((1024, D))) @ B.T + rng.standard_normal((1024, D))
        r = mi.xgauss_bracket(h, m, S, np.arange(1024)); r.update(check='linear', D=D, I_true=I2)
        print(r, flush=True); out.append(r)
    for din in (2, 8):
        for d in (8, 16, 32):
            for sigma in (0.1, 0.3, 1.0):
                for seed in (0, 1):
                    t = time.time()
                    net = RandNet(din, d, 3, seed=1000 * seed + d + din)
                    rng = np.random.default_rng(seed + 11); Mx, N, K = 4096, 1024, (256 if d == 32 else 128)
                    x = rng.standard_normal((Mx, din))
                    _, hs, _ = net.run(x[:N], sigma, rng)
                    acc = [[] for _ in range(3)]
                    for k in range(K):
                        mk, _, _ = net.run(x, sigma, rng)
                        for l in range(3): acc[l].append(mk[l])
                    row = dict(din=din, d=d, sigma=sigma, seed=seed)
                    for l in range(3):
                        m, S = mi.moments_from_chains(np.stack(acc[l], 1), sigma)
                        if l == 0: S = np.broadcast_to(sigma ** 2 * np.eye(d), S.shape).copy()   # layer 1: H|x exact
                        r = mi.xgauss_bracket(hs[l], m, S, np.arange(N))
                        row[f'xg{l + 1}_lo'] = r['xg_I_lo']; row[f'xg{l + 1}_hi'] = r['xg_I_hi']
                    out.append(row)
                    print(row, f'{time.time() - t:.0f}s', flush=True)
                    json.dump(out, open(os.path.join(OUT, 'task1_xg.json'), 'w'), indent=1)


if __name__ == '__main__' and sys.argv[1] == 'xg':
    xg()


def lk_matrix(T, means, covs, chunk=128):
    T = torch.as_tensor(np.asarray(T, np.float64)); M, D = means.shape
    L = torch.linalg.cholesky(torch.as_tensor(np.asarray(covs, np.float64)))
    Li = torch.linalg.inv(L); ld = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
    mu = torch.as_tensor(np.asarray(means, np.float64)); lk = torch.empty(len(T), M, dtype=torch.float64)
    for j in range(0, M, chunk):
        z = torch.einsum('cde,cne->cnd', Li[j:j + chunk], T[None] - mu[j:j + chunk, None])
        lk[:, j:j + chunk] = (-0.5 * (z ** 2).sum(2) - 0.5 * ld[j:j + chunk, None]).T
    return lk - 0.5 * D * math.log(2 * math.pi)


def coarse2():
    """Coarse-graining with the x-level Gaussian estimator. Classes K(x): (a) k-means cells of m(x) = E[H_3|x]
    (output-defined), (b) a product quantile grid on the input (din = 2 only). I(K;H_3) = h(H) - h(H|K), both by
    leave-in/leave-out over x-level components (h(H|K): components restricted to the target's cell)."""
    out = []
    for din, d, sigma, seed in ((2, 8, 0.3, 0), (2, 16, 0.3, 0), (2, 32, 0.3, 0), (2, 32, 1.0, 0), (8, 16, 0.3, 0), (8, 32, 0.3, 0), (8, 32, 1.0, 0)):
        net = RandNet(din, d, 3, seed=1000 * seed + d + din)
        rng = np.random.default_rng(seed + 21); Mx, N, K = 8192, 1024, (256 if d == 32 else 128)
        x = rng.standard_normal((Mx, din)); _, hs, _ = net.run(x[:N], sigma, rng)
        acc = []
        for k in range(K):
            acc.append(net.run(x, sigma, rng)[0][2])
        m, S = mi.moments_from_chains(np.stack(acc, 1), sigma); del acc
        lk = lk_matrix(hs[2], m, S); o = torch.arange(N)
        lown = lk[o, o]
        lin = torch.logsumexp(lk, 1) - math.log(Mx); lk2 = lk.clone(); lk2[o, o] = -float('inf')
        lout = torch.logsumexp(lk2, 1) - math.log(Mx - 1)
        hc = float(-lown.mean() / mi.LOG2); hH = (float(-lin.mean() / mi.LOG2), float(-lout.mean() / mi.LOG2))
        I3 = (hH[0] - hc, hH[1] - hc)
        print(din, d, sigma, 'I3 xg', I3, flush=True)
        schemes = {'kmeans_m': lambda b: kmeans(m, 2 ** b, seed=b)[1]}
        if din == 2:
            def grid(b):
                b1 = b // 2; b2 = b - b1
                e1 = np.quantile(x[:, 0], np.linspace(0, 1, 2 ** b1 + 1)[1:-1]); e2 = np.quantile(x[:, 1], np.linspace(0, 1, 2 ** b2 + 1)[1:-1])
                return np.searchsorted(e1, x[:, 0]) * 2 ** b2 + np.searchsorted(e2, x[:, 1])
            schemes['input_grid'] = grid
        for name, fn in schemes.items():
            for b in range(1, 12):
                lab = fn(b); labt = torch.as_tensor(lab)
                same = labt[:N, None] == labt[None]
                lkk = lk.masked_fill(~same, -float('inf')); n_c = same.sum(1).double()
                cin = torch.logsumexp(lkk, 1) - torch.log(n_c)
                lkk[o, o] = -float('inf'); cout = torch.logsumexp(lkk, 1) - torch.log((n_c - 1).clamp_min(1))
                hK = (float(-cin.mean() / mi.LOG2), float(-cout.mean() / mi.LOG2))
                IK = (hH[0] - hK[1], hH[1] - hK[0])
                cm = np.zeros((lab.max() + 1, d)); np.add.at(cm, lab, m); cnt = np.bincount(lab); cm /= np.maximum(cnt, 1)[:, None]
                r2 = 1 - ((m - cm[lab]) ** 2).sum() / ((m - m.mean(0)) ** 2).sum()
                p = cnt[cnt > 0] / cnt.sum(); HK = float(-(p * np.log2(p)).sum())
                row = dict(din=din, d=d, sigma=sigma, scheme=name, b=b, HK=HK, IK_lo=IK[0], IK_hi=IK[1], I3_lo=I3[0], I3_hi=I3[1], r2_m=float(r2))
                out.append(row)
                print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
                json.dump(out, open(os.path.join(OUT, 'task1_coarse2.json'), 'w'), indent=1)


if __name__ == '__main__' and sys.argv[1] == 'coarse2':
    coarse2()
