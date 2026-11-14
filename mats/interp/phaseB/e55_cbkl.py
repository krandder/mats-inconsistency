"""E55 descriptive (a property of world 1, no nets): the KL from Bayes of the constrained-Bayes reader of T = mean_q h_c(r_q),
h_c(r) = 1 - (1 + r^2 / c^2)^(-1/2), by quadrature (pfn2.kl_grid) on the first N test episodes, for a grid of cap scales c.
usage: python e55_cbkl.py N C1 C2 ... -> results/e55_cbkl.json
"""
import json, sys
import numpy as np

sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp-B')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e53_cb as C  # noqa: E402
import pfn2 as P2  # noqa: E402
import world as W  # noqa: E402


def main():
    n = int(sys.argv[1]); cs = [float(x) for x in sys.argv[2:]]
    tz = np.load('/home/agent/workspaces/mats/work/tiny-pfn/runs/testset.npz')
    o = tz['o'][:n].astype(np.float64); y = tz['y'][:n].astype(np.float64)
    T = {k: (tz[k][:n] if k != 'comp_vars' else tz[k]) for k in ('comp_means', 'comp_vars', 'w_bayes')}
    orig = C.stat; out = {}
    for c in cs:
        C.stat = lambda name, r, c=c: (1 - 1 / np.sqrt(1 + r ** 2 / c ** 2)).mean(-1) if name == 'lnh' else orig(name, r)
        mid, tab = C.log_density('lnh', n=1_000_000)
        r = o[:, :, :W.QL] - y[:, None, :W.QL]
        L = np.stack([np.interp(np.log(np.maximum(C.stat('lnh', r), 1e-12)), mid, tab[j]) for j in range(3)], -1)
        llc = sum(L[:, s, W.COMB[:, s]] for s in range(W.S))
        ostar = o[:, :, W.QL]
        lw = W.LOGPRIOR[None] + llc - 0.5 * np.einsum('ns,pst,nt->np', ostar, W.SIGQI, ostar) - 0.5 * W.LOGDETQ[None]
        lw = lw - lw.max(1, keepdims=True)
        means = np.einsum('ps,ns->np', 1.0 / W.VC, ostar) / W.PREC[None]
        kl = P2.kl_grid(T, (means, np.broadcast_to(0.5 * np.log(1.0 / W.PREC)[None], means.shape).copy(), lw))
        out['%.2f' % c] = [float(kl.mean()), float(kl.std() / np.sqrt(n))]
        print(c, out['%.2f' % c], flush=True)
    json.dump(dict(n=n, kl=out), open('results/e55_cbkl.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
