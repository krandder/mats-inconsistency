"""Is the residual K2 result trivial? Compare the full Gaussian closure with cruder sample-free closures.
usage: python p3_k2variants.py TRUTHPREFIX [TRUTHPREFIX ...]
- K2-full: mean + full covariance, exact ReLU moments (as p3_att).
- K2-diag: same, but the covariance is kept diagonal after every layer (drops all correlations).
- K1: zero-variance closure (push the mean through the noiseless map).
- MF-input: input-independent stand-in, E[h_L] from a Gaussian N(0, q I) per layer with q from the K2-full variances
  (no correlations, no means carried; ~ a mean-field answer).
"""
import sys, os, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd, k2

torch.set_num_threads(1)
DT = nd.DT


class Diag:
    """Wrap relu_mom so that covariances are diagonalised after each layer."""


for pref in sys.argv[1:]:
    S = json.load(open(pref + '.json'))
    net = nd.Net(S['fam'], S['seed'], S['n'], S['L'], S['sigma'], r=S['r'] if S['sigma'] > 0 else None)
    truth = torch.tensor(S['mean'], dtype=DT); trv = S['trvar']; n, L = net.n, net.L
    out = {}
    mu, C, fl = k2.prop(net, torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT), 0, L, exact=True)
    out['K2-full'] = float(((mu - truth) ** 2).sum() / trv)
    # diagonal closure, layer by layer
    mu = torch.zeros(n, dtype=DT); C = torch.eye(n, dtype=DT)
    for l in range(L):
        mu, C, _ = k2.prop(net, mu, C, l, l + 1, exact=True)
        C = torch.diag(C.diagonal())
    out['K2-diag'] = float(((mu - truth) ** 2).sum() / trv)
    # K1: noiseless push of the mean (starts at 0 -> stays 0 for plain; residual likewise)
    h = torch.zeros(n, dtype=DT)
    for l in range(L):
        a = torch.relu(net.W[l] @ h)
        h = a if net.fam == 'plain' else h + net.c * (net.V[l] @ a)
    out['K1'] = float(((h - truth) ** 2).sum() / trv)
    out['zero'] = float((truth ** 2).sum() / trv)
    print(os.path.basename(pref), ' '.join(f"{k}={v:.3g}" for k, v in out.items()), flush=True)
    json.dump(out, open(pref + '_k2var.json', 'w'))
