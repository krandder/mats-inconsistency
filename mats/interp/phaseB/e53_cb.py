"""Constrained Bayes (sixth agent, prediction input for E53; a property of world 1, no nets involved).
A reader that sees, for each source, only a statistic T(r_1..r_4) of its four labelled errors (not their sum of squares), plus
every query reading, and is otherwise exact Bayes: weights over the 27 variance combinations
    log w(c) = log prior + sum_s log p(T_s | v_{c_s}) + log p(o* | c)      (the last term is world.py's exact query marginal),
predictive mean sum_c w(c) m_c(o*), effective weight of source s = d mean / d o*_s (central differences, as qweights.py).
p(T | v) is estimated per level by simulation (N draws of four N(0, v) errors) and a histogram in log T, smoothed.
rom on E12's matched pairs (results/e12_pairs.npz): sum (w_bigmiss - w_spread)[src] / sum W_Bayes[src] over each range, as E12.
usage: python e53_cb.py [STAT ...]   STAT in ss, mabs, max, cap1, cap2, cap05, l1sq (default: all)
"""
import sys
import numpy as np

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import world as W  # noqa: E402

QL = W.QL


def stat(name, r):
    a = np.abs(r)
    if name == 'ss':
        return (r ** 2).sum(-1)
    if name == 'mabs':
        return a.mean(-1)
    if name == 'max':
        return a.max(-1)
    if name.startswith('cap'):                                  # mean of r^2 / (1 + r^2 / c^2): quadratic, then saturating at c^2
        c = {'cap1': 1.0, 'cap2': 2.0, 'cap05': 0.5}[name]
        return (r ** 2 / (1 + r ** 2 / c ** 2)).mean(-1)
    if name == 'l1sq':                                          # (sum |r|)^2: same units as SS, not sufficient
        return a.sum(-1) ** 2
    raise ValueError(name)


def log_density(name, n=4_000_000, bins=1200, seed=5301):
    """Per level j: log p(log T | v_j) on a common grid of log T (histogram, 3-bin smoothing); returns grid, table (NV, bins)."""
    rng = np.random.default_rng(seed)
    Ts = []
    for v in W.VLEV:
        r = rng.standard_normal((n, QL)) * np.sqrt(v)
        Ts.append(np.log(np.maximum(stat(name, r), 1e-12)))
    lo = min(t.min() for t in Ts) - 0.5; hi = max(t.max() for t in Ts) + 0.5
    edges = np.linspace(lo, hi, bins + 1); mid = 0.5 * (edges[1:] + edges[:-1]); dx = edges[1] - edges[0]
    tab = []
    for t in Ts:
        h, _ = np.histogram(t, edges)
        h = np.convolve(h.astype(float), np.ones(5) / 5, mode='same') + 1e-3   # light smoothing, floor
        tab.append(np.log(h / (h.sum() * dx)))
    return mid, np.array(tab)


def make_loglik(name):
    if name == 'ss':                                            # exact: log p(SS | v) = chi-square with QL dof, scaled
        def ll(r):
            ss = (r ** 2).sum(-1)
            return -0.5 * ss[..., None] / W.VLEV - 0.5 * QL * np.log(W.VLEV)   # up to a v-free constant
        return ll
    mid, tab = log_density(name)

    def ll(r):
        lt = np.log(np.maximum(stat(name, r), 1e-12))          # density of log T: the Jacobian is v-free, so it cancels
        return np.stack([np.interp(lt, mid, tab[j]) for j in range(W.NV)], -1)
    return ll


def cb_mean(o, y, ll):
    """o (n, S, Q), y (n, Q): constrained-Bayes predictive mean of y*."""
    r = o[:, :, :QL] - y[:, None, :QL]
    L = ll(r)                                                   # (n, S, NV)
    llc = sum(L[:, s, W.COMB[:, s]] for s in range(W.S))        # (n, 27)
    ostar = o[:, :, QL]
    quadq = np.einsum('ns,pst,nt->np', ostar, W.SIGQI, ostar)
    lw = W.LOGPRIOR[None] + llc - 0.5 * quadq - 0.5 * W.LOGDETQ[None]
    lw = lw - lw.max(1, keepdims=True); w = np.exp(lw); w /= w.sum(1, keepdims=True)
    means = np.einsum('ps,ns->np', 1.0 / W.VC, ostar) / W.PREC[None]
    return (w * means).sum(1)


def cb_weights(o, y, ll, h=1e-3):
    out = []
    for s in range(W.S):
        op = o.copy(); om = o.copy(); op[:, s, QL] += h; om[:, s, QL] -= h
        out.append((cb_mean(op, y, ll) - cb_mean(om, y, ll)) / (2 * h))
    return np.stack(out, 1)


def rom_cb(name, z):
    ll = make_loglik(name)
    src, rg = z['src'], z['range']; rows = np.arange(len(src))
    ws = [cb_weights(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ll)[rows, src] for k in (0, 1)]
    Wb = z['WB0'][rows, src]
    out = {}
    for rr in ('large', 'middle'):
        s = rg == rr
        out[rr] = float((ws[1] - ws[0])[s].sum() / Wb[s].sum())
        out[rr + '_wcb_over_wb'] = float(ws[0][s].sum() / Wb[s].sum())
    return out


if __name__ == '__main__':
    z = np.load('/home/agent/workspaces/mats/work/cm-interp-B/results/e12_pairs.npz')
    names = sys.argv[1:] or ['ss', 'mabs', 'max', 'cap1', 'cap2', 'cap05', 'l1sq']
    for nm in names:
        print(nm, {k: round(v, 4) for k, v in rom_cb(nm, z).items()}, flush=True)
