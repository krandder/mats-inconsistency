import sys
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp-B'); sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import numpy as np
import e53_cb as C
import world as W
import pfn2 as P2
tz = np.load('/home/agent/workspaces/mats/work/tiny-pfn/runs/testset.npz')
n = int(sys.argv[1]); names = sys.argv[2:]
o = tz['o'][:n].astype(np.float64); y = tz['y'][:n].astype(np.float64)
T = {k: tz[k][:n] if k != 'comp_vars' else tz[k] for k in ('comp_means', 'comp_vars', 'w_bayes')}
for nm in names:
    ll = C.make_loglik(nm)
    r = o[:, :, :W.QL] - y[:, None, :W.QL]
    L = ll(r); llc = sum(L[:, s, W.COMB[:, s]] for s in range(W.S))
    ostar = o[:, :, W.QL]
    quadq = np.einsum('ns,pst,nt->np', ostar, W.SIGQI, ostar)
    lw = W.LOGPRIOR[None] + llc - 0.5 * quadq - 0.5 * W.LOGDETQ[None]
    lw = lw - lw.max(1, keepdims=True)
    means = np.einsum('ps,ns->np', 1.0 / W.VC, ostar) / W.PREC[None]
    head = (means, np.broadcast_to(0.5 * np.log(1.0 / W.PREC)[None], means.shape).copy(), lw)
    kl = P2.kl_grid(T, head)
    print(nm, 'KL(Bayes||CB) by quadrature on %d test episodes: %.5f (se %.5f)' % (n, kl.mean(), kl.std() / np.sqrt(n)), flush=True)
