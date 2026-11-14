"""E75 exploration (eighth agent; a property of world 1, no nets): how far from exact Bayes is the best output head of each size?
For N test episodes (tiny-pfn's world, S = 3), the exact predictive p of y* is a mixture of 27 Gaussians. For K = 1, 2, 4, 8, the
K-component Gaussian mixture q minimising KL(p || q) (an M-projection) is fitted per episode by weighted EM on a fine grid (+-9 sd of
p's widest component around its extreme means, 1,201 points, weights p(x) dx), initialised from p's K heaviest components (K = 1:
moment matching, exact); KL(p || q) by quadrature on a finer grid. Reports the mean KL per K (nats), the quantiles, and the fraction of
episodes where it exceeds 0.01.
usage: python e75x_headfloor.py N [K ...] -> results/e75x_headfloor.json
"""
import json, sys
import numpy as np

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import world as W  # noqa: E402


def fit_em(x, px, means, sd, w, K, iters=300):
    """x (G,), px (n, G) weights summing to 1 per row; init means/sd/w (n, K). Returns (mu, sd, w)."""
    mu, s, pi = means.copy(), sd.copy(), w.copy()
    for _ in range(iters):
        lc = np.log(pi)[:, None, :] - np.log(s)[:, None, :] - 0.5 * ((x[None, :, None] - mu[:, None, :]) / s[:, None, :]) ** 2
        lc -= lc.max(2, keepdims=True); r = np.exp(lc); r /= r.sum(2, keepdims=True)                  # (n, G, K)
        rw = r * px[:, :, None]; nk = rw.sum(1) + 1e-300                                                 # (n, K)
        pi = nk / nk.sum(1, keepdims=True)
        mu = (rw * x[None, :, None]).sum(1) / nk
        s = np.sqrt(np.maximum((rw * (x[None, :, None] - mu[:, None, :]) ** 2).sum(1) / nk, 1e-8))
    return mu, s, pi


def logmix(x, mu, s, pi):
    lc = np.log(pi)[:, None, :] - np.log(s)[:, None, :] - 0.5 * np.log(2 * np.pi) - 0.5 * ((x[:, :, None] - mu[:, None, :]) / s[:, None, :]) ** 2
    m = lc.max(2); return m + np.log(np.exp(lc - m[:, :, None]).sum(2))


def main():
    N = int(sys.argv[1]); Ks = [int(k) for k in sys.argv[2:]] or [1, 2, 4, 8]
    d = W.sample_np(N, np.random.default_rng(75_000))
    out, nll, mom, wb = W.references(d)
    means, vars_ = out['bayes'][0], out['bayes'][1]                                                     # (N, 27), (27,)
    sd = np.sqrt(vars_)
    res = {}
    for K in Ks:
        kls = []
        for i in range(0, N, 500):
            m, w = means[i:i + 500], wb[i:i + 500]; n = len(m)
            lo = (m - 9 * sd[None]).min(1); hi = (m + 9 * sd[None]).max(1)
            # a common grid per episode: rescale a unit grid (EM on x per episode needs per-row grids: use t in [0, 1])
            t = np.linspace(0, 1, 1201); X = lo[:, None] + (hi - lo)[:, None] * t[None]                    # (n, G)
            lp = np.log(np.maximum((w[:, None, :] * np.exp(-0.5 * (X[:, :, None] - m[:, None, :]) ** 2 / vars_) / np.sqrt(2 * np.pi * vars_)).sum(2), 1e-300))
            px = np.exp(lp) * (hi - lo)[:, None] / 1200; px /= px.sum(1, keepdims=True)
            if K == 1:
                mu = (px * X).sum(1, keepdims=True); s = np.sqrt((px * (X - mu) ** 2).sum(1, keepdims=True)); pi = np.ones((n, 1))
            else:
                order = np.argsort(-w, 1)[:, :K]
                mu0 = np.take_along_axis(m, order, 1); s0 = np.repeat(sd[None], n, 0); s0 = np.take_along_axis(s0, order, 1)
                pi0 = np.take_along_axis(w, order, 1) + 1e-6; pi0 /= pi0.sum(1, keepdims=True)
                # EM in normalised coordinates u = (x - lo) / (hi - lo) so that one grid serves all rows
                scale = (hi - lo)[:, None]
                mu, s, pi = fit_em(t, px, (mu0 - lo[:, None]) / scale, s0 / scale, pi0, K)
                mu = lo[:, None] + mu * scale; s = s * scale
            lq = logmix(X, mu, s, pi)
            kls.append((px * (lp - lq)).sum(1))
        kl = np.concatenate(kls)
        res[K] = dict(mean=float(kl.mean()), median=float(np.median(kl)), q90=float(np.quantile(kl, 0.9)), q99=float(np.quantile(kl, 0.99)),
                      frac_above_001=float((kl > 0.01).mean()))
        print(K, json.dumps(res[K]), flush=True)
    json.dump(res, open('results/e75x_headfloor.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
