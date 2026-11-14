"""E5: deduction vs sampling for a trained tree net (W0, independent sources).

The tree's root state is z = sum_s phi(x_s). Given the shared variables y = (y_1..y_4, y*), the sources are iid, so
the law of z | y is the N-fold convolution of one source's message law. One source's law is computed by a
deterministic rule over its 5 errors (v in {0.1, 0.5, 2}, 1/3 each): scrambled Sobol points (fixed seed) mapped to a
widened normal with importance weights (covers the tails), or tensor Gauss-Hermite.
  exact  (m = 2): the per-source law is deposited (bilinear, mean-preserving) on a per-y adaptive 2-D grid; N-fold
                  convolution by FFT gives the law of z on the grid; the readout is evaluated on every cell with mass.
  gauss  (any m): CLT root: z | y ~ N(N mu_y, N Sigma_y) with the per-source mean and covariance from the same rule;
                  the readout expectation over nroot Gaussian root draws (no episode sampling).
Outer integral over y (5-D): the same kind of deterministic rule.
Estimands: E[NLL_net(y*)], E[mean_net], E[mean_net^2], E[mean_net * y*], tail probabilities P(NLL_net > L).
Reference and baseline: plain Monte Carlo over episodes. Departures from Bayes follow from world constants:
  E[(mean_net - mean_bayes)^2] = E[mean_net^2] - 2 E[mean_net y*] + E[mean_bayes^2];  gap = E[NLL_net] - H(y* | x).
usage: python est5.py RUN N {exact|gauss|mc|world} [options]
"""
import argparse, json, math, os, time
import numpy as np
import torch
import sfworld as W
import sfmodels as M

torch.set_num_threads(1)
LEVELS = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0)   # NLL tail thresholds (nats)


def load_tree(run, which='model.pt'):
    meta = json.load(open(os.path.join(run, 'meta.json')))
    net = M.TreeNet(9, meta['m'], h=meta['h'], K=meta['K'], pool=meta['pool'])
    net.load_state_dict(torch.load(os.path.join(run, which), map_location='cpu')); net.eval()
    return net, meta


def gh(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


def tensor_nodes(n, dim):
    x, w = gh(n)
    X = np.stack(np.meshgrid(*[x] * dim, indexing='ij'), -1).reshape(-1, dim)
    Wt = np.prod(np.stack(np.meshgrid(*[w] * dim, indexing='ij'), -1).reshape(-1, dim), 1)
    return X, Wt


def qmc_normal(n_log2, dim, scale, seed):
    """Scrambled Sobol points mapped to N(0, scale^2 I), self-normalised importance weights to N(0, I)."""
    from scipy.stats import qmc, norm
    u = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(n_log2)
    z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)) * scale
    lw = -0.5 * (z ** 2).sum(1) * (1 - 1 / scale ** 2)
    w = np.exp(lw - lw.max())
    return z, w / w.sum()


def rule(n, dim, kind, scale, seed):
    return tensor_nodes(n, dim) if kind == 'gh' else qmc_normal(n, dim, scale, seed)


def y_rule(n_y, kind):
    """Shared-variable rule. 'qmc': 2^n_y tail-covering Sobol points in 5-D. 'prod': y* by n_y-point Gauss-Hermite
    (resolves |y*| up to about sqrt(4 n_y)) times 32 Sobol points for the four labels (they matter less)."""
    if kind != 'prod':
        return rule(n_y, 5, kind, 1.3, 5)
    xs, ws = gh(n_y)
    L, WL = qmc_normal(5, 4, 1.0, 7)
    Y = np.concatenate([np.repeat(L, len(xs), 0), np.tile(xs, len(L))[:, None]], 1)
    return Y, np.repeat(WL, len(xs)) * np.tile(ws, len(L))


class Source:
    def __init__(self, n, kind='qmc', seed=11):
        kind = 'qmc' if kind == 'prod' else kind
        self.E, self.Wt = rule(n, 5, kind, 1.6, seed)

    def cloud(self, net, y):
        feats, wts = [], []
        for v in W.VLEV:
            e = self.E * math.sqrt(v)
            o = y[None, :] + e
            f = np.concatenate([o[:, :4], np.repeat(y[None, :4], len(e), 0), o[:, 4:5]], 1)
            feats.append(f); wts.append(self.Wt / 3)
        F = np.concatenate(feats).astype(np.float32)
        with torch.no_grad():
            msg = net.enc(torch.from_numpy(F)).numpy().astype(np.float64)
        return msg, np.concatenate(wts)


def nll_mix(mu, ls, lg, y):
    lw = lg - np.logaddexp.reduce(lg, axis=-1, keepdims=True)
    lp = lw - ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((y - mu) / np.exp(ls)) ** 2
    return -np.logaddexp.reduce(lp, axis=-1)


def readout(net, z, bs=200000):
    out = []
    with torch.no_grad():
        for i in range(0, len(z), bs):
            mu, ls, lg = M.split_head(net.read(torch.from_numpy(z[i:i + bs].astype(np.float32))), net.K)
            out.append((mu.numpy().astype(np.float64), ls.numpy().astype(np.float64), lg.numpy().astype(np.float64)))
    return tuple(np.concatenate([o[j] for o in out]) for j in range(3))


def mix_mean(mu, lg):
    w = np.exp(lg - np.logaddexp.reduce(lg, axis=-1, keepdims=True))
    return (w * mu).sum(-1)


def new_res():
    return dict(nll=0.0, mean=0.0, mean2=0.0, mean_y=0.0, tails={L: 0.0 for L in LEVELS})


def accumulate(res, wy, ystar, qq, nl, mc):
    res['nll'] += wy * (qq * nl).sum()
    res['mean'] += wy * (qq * mc).sum()
    res['mean2'] += wy * (qq * mc ** 2).sum()
    res['mean_y'] += wy * ystar * (qq * mc).sum()
    for L in LEVELS:
        res['tails'][L] += wy * qq[nl > L].sum()


def deduce_exact(net, N, n_src=14, n_y=10, G=640, kind='qmc'):
    """m = 2: exact N-fold convolution of the per-source message law on a per-y adaptive grid."""
    assert net.m == 2
    src = Source(n_src, kind)
    Y, WY = y_rule(n_y, kind)
    res = new_res(); t0 = time.time(); n_read = 0
    for y, wy in zip(Y, WY):
        msg, wt = src.cloud(net, y)
        lo, hi = msg.min(0), msg.max(0); span = hi - lo
        lo, hi = lo - 0.02 * span, hi + 0.02 * span
        Gs = max(8, G // N)
        h = (hi - lo) / (Gs - 1)
        F = Gs * N + 2
        u = (msg - lo) / h
        i0 = np.floor(u).astype(int); f = u - i0
        p = np.zeros((F, F))
        for dx in (0, 1):
            for dy in (0, 1):
                ww = wt * (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                np.add.at(p, (i0[:, 0] + dx, i0[:, 1] + dy), ww)
        q = np.fft.irfft2(np.fft.rfft2(p) ** N, s=(F, F))
        q = np.maximum(q, 0).reshape(-1); q /= q.sum()
        ii = np.nonzero(q > 1e-18)[0]
        ix, iy = np.unravel_index(ii, (F, F))
        zg = np.stack([N * lo[0] + ix * h[0], N * lo[1] + iy * h[1]], 1)
        mu, ls, lg = readout(net, zg); n_read += len(zg)
        qq = q[ii] / q[ii].sum()
        accumulate(res, wy, y[4], qq, nll_mix(mu, ls, lg, y[4]), mix_mean(mu, lg))
    res['seconds'] = time.time() - t0
    res['cost'] = dict(enc_evals=len(Y) * len(src.E) * 3, read_evals=n_read, y_nodes=len(Y), kind=kind)
    return res


def deduce_gauss(net, N, n_src=14, n_y=10, nroot=4096, seed=0, kind='qmc'):
    """Any m: z | y ~ N(N mu_y, N Sigma_y) from the per-source moments; readout over nroot root draws."""
    src = Source(n_src, kind)
    Y, WY = y_rule(n_y, kind)
    Z0 = np.random.default_rng(seed).standard_normal((nroot, net.m))
    res = new_res(); t0 = time.time()
    for y, wy in zip(Y, WY):
        msg, wt = src.cloud(net, y)
        m1 = (wt[:, None] * msg).sum(0)
        dm = msg - m1
        C = (wt[:, None] * dm).T @ dm
        Lc = np.linalg.cholesky(N * C + 1e-10 * np.eye(net.m))
        z = N * m1 + Z0 @ Lc.T
        mu, ls, lg = readout(net, z)
        accumulate(res, wy, y[4], np.full(nroot, 1.0 / nroot), nll_mix(mu, ls, lg, y[4]), mix_mean(mu, lg))
    res['seconds'] = time.time() - t0
    res['cost'] = dict(enc_evals=len(Y) * len(src.E) * 3, read_evals=len(Y) * nroot, y_nodes=len(Y), kind=kind)
    return res


def monte_carlo(net, N, M_eps, seed=0, chunk=200000):
    rng = np.random.default_rng([seed, 77])
    tails = {L: 0 for L in LEVELS}
    sums = dict(nll=0.0, nll2=0.0, mean=0.0, mean2=0.0, mean_y=0.0)
    t0 = time.time(); n = 0
    while n < M_eps:
        b = min(chunk, M_eps - n)
        d = W.sample(b, N, rng)
        xs = torch.from_numpy(W.src_features(d))
        with torch.no_grad():
            mu, ls, lg = [t.numpy().astype(np.float64) for t in net(xs)]
        ys = d['y'][:, W.QL]
        nl = nll_mix(mu, ls, lg, ys[:, None]); mc = mix_mean(mu, lg)
        sums['nll'] += nl.sum(); sums['nll2'] += (nl ** 2).sum(); sums['mean'] += mc.sum()
        sums['mean2'] += (mc ** 2).sum(); sums['mean_y'] += (mc * ys).sum()
        for L in LEVELS:
            tails[L] += int((nl > L).sum())
        n += b
    out = {k: sums[k] / n for k in ('nll', 'mean', 'mean2', 'mean_y')}
    out['nll_se'] = math.sqrt(max(0, sums['nll2'] / n - out['nll'] ** 2) / n)
    out['tails'] = {L: tails[L] / n for L in LEVELS}; out['tail_counts'] = tails
    out['n'] = n; out['seconds'] = time.time() - t0
    return out


def world_constants(N, M_eps=200000, seed=3):
    """H(y* | x) = E[NLL_bayes] and E[mean_bayes^2] by Monte Carlo with exact Bayes per episode."""
    rng = np.random.default_rng([seed, 99])
    d = W.sample(M_eps, N, rng)
    B = W.bayes(d)
    p = np.exp(B['logp'])
    ent = -np.trapezoid(p * B['logp'], B['grid'], axis=1)               # posterior differential entropy
    # low-variance identities: E[NLL_bayes] = E[entropy of the posterior]; E[mean_bayes^2] = 1 - E[posterior var]
    return dict(H=float(B['nll'].mean()), H_se=float(B['nll'].std() / math.sqrt(M_eps)),
                H_ent=float(ent.mean()), H_ent_se=float(ent.std() / math.sqrt(M_eps)),
                Emb2=float((B['mean'] ** 2).mean()), Emb2_se=float((B['mean'] ** 2).std() / math.sqrt(M_eps)),
                Emb2_var=float(1 - B['var'].mean()), Emb2_var_se=float(B['var'].std() / math.sqrt(M_eps)), n=M_eps)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('N', type=int); ap.add_argument('how')
    ap.add_argument('--nsrc', type=int, default=14); ap.add_argument('--ny', type=int, default=10)
    ap.add_argument('--G', type=int, default=640); ap.add_argument('--M', type=int, default=1000000)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--nroot', type=int, default=4096)
    ap.add_argument('--kind', default='qmc'); ap.add_argument('--which', default='model.pt')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    net = None if a.how == 'world' else load_tree(a.run, a.which)[0]
    if a.how == 'exact':
        r = deduce_exact(net, a.N, a.nsrc, a.ny, a.G, a.kind)
    elif a.how == 'gauss':
        r = deduce_gauss(net, a.N, a.nsrc, a.ny, a.nroot, a.seed, a.kind)
    elif a.how == 'mc':
        r = monte_carlo(net, a.N, a.M, a.seed)
    else:
        r = world_constants(a.N, a.M, a.seed)
    r = json.loads(json.dumps(r, default=float))
    r.update(how=a.how, N=a.N, run=a.run, args=vars(a))
    print(json.dumps(r))
    if a.out:
        with open(a.out, 'a') as f:
            f.write(json.dumps(r) + '\n')
