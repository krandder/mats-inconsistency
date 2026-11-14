"""(3) Backward functions psi_l for rare outputs: mechanistic (derived from the layer maps) vs black-box (learned), used
as twists in SMC over the layers of the trained small noisy net of dec.py (6 layers, width 8, sigma 0.5).

Event A_t = {y > t}, y = w.h_L + b. psi_{L-1} is exact for every method: P(A | h_{L-1}) = Phi((w.f_L(h) + b - t) / (sigma|w|)).
- mechanistic psi_l(h) = Phi((g_l(h) - t) / s_l(h)): g_l = noise-free propagation of h to y, s_l^2 = sigma^2 (|w|^2 +
  sum_{j>l} |dy/dh_j|^2) (linearised propagation of the remaining noise); built from the weights only.
- black-box psi_l(h) = Phi((mu(h) - t) / s(h)), (mu, log s) a small MLP fitted by Gaussian NLL to (h_l, fresh y) pairs from
  plain forward paths plus tail-biased paths (subset-simulation particles, downstream noise re-drawn).
Twisted SMC (Feynman-Kac): potentials G_0 = psi_0(x), G_l = psi_l(h_l)/psi_{l-1}(h_{l-1}), terminal = P(A|h_{L-1}) /
psi_{L-1}: unbiased for any positive psi. Compared at equal particle count and at equal wall time with plain MC and
untwisted SMC; relative RMSE of p_hat against the subset-simulation truth (dec.py).

usage: python twist.py -> results/twist_d8_s0.5.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.stats import norm
import dec as Dc
import e2 as E2

torch.set_num_threads(1)
L, DIN = E2.L, E2.DIN


def torch_layer(net, l):
    W = torch.tensor(net.W[l]); b = torch.tensor(net.b[l]); d = net.d
    def f(h):
        a = torch.tanh(net.g * h @ W.T + b); c = a - a.mean(-1, keepdim=True); return c * math.sqrt(d) / c.norm(dim=-1, keepdim=True)
    return f


class Mech:
    def __init__(self, net, w, b, sigma):
        self.fs = [torch_layer(net, l) for l in range(L)]; self.w = torch.tensor(w); self.b = b; self.sg = sigma

    def stats(self, l, h):
        """For states h at layer l (l = 0: input x): noise-free y and its linearised sd from the noise at sites l+1..L."""
        H = torch.tensor(h, requires_grad=True); zs = []; x = H
        for j in range(l, L):
            x = self.fs[j](x)
            if j < L - 1:
                z = torch.zeros_like(x, requires_grad=True); x = x + z; zs.append(z)
        y = x @ self.w + self.b
        var = self.sg ** 2 * float(self.w @ self.w) * torch.ones(len(h), dtype=torch.float64)
        if zs:
            gs = torch.autograd.grad(y.sum(), zs)
            for g in gs: var = var + self.sg ** 2 * (g ** 2).sum(1)
        return y.detach().numpy(), np.sqrt(var.detach().numpy())

    def logpsi(self, l, h, t):
        m, s = self.stats(l, h); return norm.logsf((t - m) / s)


class MechUT:
    """Mechanistic psi by forward moment propagation of a Gaussian description: for a particle h at layer l, h_{l+1} ~
    N(f(h), sigma^2 I) is pushed through each later layer by the unscented transform (2d+1 sigma points, moment-matched
    Gaussian + sigma^2 I after every layer); psi = Phi((m - t)/s) from the final Gaussian of y. Weights only, no learning."""

    def __init__(self, net, w, b, sigma):
        self.net, self.w, self.b, self.sg = net, w, b, sigma

    def logpsi(self, l, h, t):
        net, sg = self.net, self.sg; n = len(h); d = net.d
        m = net.f(l, h); C = np.broadcast_to(sg ** 2 * np.eye(d), (n, d, d)).copy()        # h_{l+1} | h
        for j in range(l + 1, L):
            Lc = np.linalg.cholesky(C + 1e-9 * np.eye(d)); k = d
            pts = np.concatenate([m[:, None], m[:, None] + math.sqrt(k) * np.transpose(Lc, (0, 2, 1)), m[:, None] - math.sqrt(k) * np.transpose(Lc, (0, 2, 1))], 1)
            wts = np.r_[0.0, np.full(2 * k, 1 / (2 * k))]
            F = net.f(j, pts.reshape(-1, d)).reshape(n, 2 * k + 1, d)
            m = np.einsum('k,nkd->nd', wts, F); Fc = F - m[:, None]
            C = np.einsum('k,nkd,nke->nde', wts, Fc, Fc) + sg ** 2 * np.eye(d)
        my = m @ self.w + self.b; vy = np.einsum('d,nde,e->n', self.w, C, self.w)
        return norm.logsf((t - my) / np.sqrt(vy))


class BlackBox:
    def __init__(self, d_in, seed):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2)).double()

    def fit(self, H, Y, steps=3000):
        Ht = torch.tensor(H); Yt = torch.tensor(Y); opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(0)
        for s in range(steps):
            i = torch.randint(len(Ht), (1024,), generator=g); out = self.net(Ht[i])
            mu, ls = out[:, 0], out[:, 1].clamp(-5, 3)
            loss = (ls + 0.5 * ((Yt[i] - mu) / ls.exp()) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        return self

    def logpsi(self, h, t):
        with torch.no_grad():
            out = self.net(torch.tensor(h)); m, s = out[:, 0].numpy(), np.exp(out[:, 1].clamp(-5, 3).numpy())
        return norm.logsf((t - m) / s)


def smc(net, path, logpsi, t, N, rng, sigma):
    """Twisted SMC; logpsi(l, h) for l = 0..L-2 (None = untwisted); exact terminal at L-1. Returns log p_hat."""
    d = net.d; x = rng.standard_normal((N, DIN)); logZ = 0.0
    lp_prev = logpsi(0, x) if logpsi else np.zeros(N)
    lw = lp_prev.copy(); h = x
    for l in range(L - 1):
        # resample by current weights
        m = lw.max(); wts = np.exp(lw - m); logZ += m + math.log(wts.mean())
        idx = rng.choice(N, N, p=wts / wts.sum()); h, lp_prev = h[idx], lp_prev[idx]
        h = net.f(l, h) + sigma * rng.standard_normal((N, d))
        if l < L - 2:
            lp = logpsi(l + 1, h) if logpsi else np.zeros(N)
        else:   # terminal: exact P(A | h_{L-1})
            mL = net.f(L - 1, h) @ path.w + path.b
            lp = norm.logsf((t - mL) / (sigma * np.linalg.norm(path.w)))
        lw = lp - lp_prev; lp_prev = lp
    m = lw.max(); logZ += m + math.log(np.exp(lw - m).mean())
    return logZ


def run(d=8, sigma=0.5):
    R = json.load(open(os.path.join(E2.OUT, f'dec_d{d}_s{sigma}.json')))
    ts, ptrue = np.array(R['t']), np.array(R['p_true'])
    net, w, b = Dc.train_net_ro(d, sigma); path = Dc.Path(net, w, b, sigma)
    rng = np.random.default_rng(7)
    # targets: thresholds at p ~ 2^-10, 2^-20, 2^-30 (closest available)
    sel = [int(np.argmin(np.abs(np.log2(ptrue) + k))) for k in (10, 20, 30, 40)]
    sel = sorted(set(sel))
    mech = Mech(net, w, b, sigma); mut = MechUT(net, w, b, sigma)
    # black-box training data: plain paths + tail-biased paths (subset simulation particles), fresh downstream noise
    D = DIN + L * d
    t0 = time.time()
    U = [rng.standard_normal((20000, D))]
    st = rng.standard_normal((2000, D))
    for k in range(10):                                         # crude tail-biased population: keep top 10%, pCN
        y = path.full(st); g_ = np.quantile(y, 0.9); seeds = st[y >= g_]
        cur = np.repeat(seeds, 10, 0)[:2000]
        for _ in range(5):
            prop = 0.8 * cur + 0.6 * rng.standard_normal(cur.shape); ok = path.full(prop) >= g_; cur[ok] = prop[ok]
        st = cur; U.append(cur.copy())
    U = np.concatenate(U)
    bbs = []
    for l in range(L - 1):
        x = U[:, :DIN]; nz = U[:, DIN:].reshape(len(U), L, d); h = x
        for j in range(l):
            h = net.f(j, h) + sigma * nz[:, j]
        Y = path.from_layer(l, h, rng.standard_normal((len(U), L - l, d)))
        bbs.append(BlackBox(h.shape[1], l).fit(h, Y))
    t_bb_train = time.time() - t0
    out = dict(t=ts[sel].tolist(), p_true=ptrue[sel].tolist(), t_bb_train=t_bb_train, rows=[])
    for i in sel:
        t = ts[i]; p = ptrue[i]
        methods = {'untwisted': None, 'mechanistic': lambda l, h, t=t: mech.logpsi(l, h, t), 'mech-UT': lambda l, h, t=t: mut.logpsi(l, h, t),
                   'black-box': lambda l, h, t=t: bbs[l].logpsi(h, t)}
        if os.environ.get('TWIST_ONLY'): methods = {k: v for k, v in methods.items() if k in os.environ['TWIST_ONLY'].split(',')}
        for name, fn in methods.items():
            for N in (1000, 4000):
                t1 = time.time(); est = [smc(net, path, fn, t, N, np.random.default_rng(100 + r), sigma) for r in range(20)]
                dt = (time.time() - t1) / 20
                est = np.array(est); rel = np.exp(est - math.log(p))
                row = dict(t=float(t), p=float(p), method=name, N=N, sec_per_run=dt, log2_mean_ratio=float(np.log2(rel.mean())),
                           rel_rmse=float(np.sqrt(((rel - 1) ** 2).mean())), sd_log2=float(np.std(est) / math.log(2)))
                # plain MC at the same wall time
                n_mc = int(dt * R['rate']); row['plain_mc_rel_sd'] = float(math.sqrt((1 - p) / (p * max(n_mc, 1)))); row['n_mc_eq'] = n_mc
                out['rows'].append(row); print(row, flush=True)
                json.dump(out, open(os.path.join(E2.OUT, f'twist_d{d}_s{sigma}' + os.environ.get('TWIST_TAG', '') + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 8, float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
