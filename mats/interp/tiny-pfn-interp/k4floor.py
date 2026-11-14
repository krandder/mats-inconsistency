"""Amendment 3b: best K=4 Gaussian mixture to the Bayes mixture per held-out episode (weighted EM on a grid)."""
import json, sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import pfn as P
D = np.load('/home/agent/workspaces/mats/work/tiny-pfn/runs/testset.npz')
te = np.arange(12000, 16000)
means, vars_, w = D['comp_means'][te], D['comp_vars'], D['w_bayes'][te]
sd = np.sqrt(vars_); G = 1201; Kc = 4
lo = (means - 9 * sd).min(1); hi = (means + 9 * sd).max(1)
x = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, G)[None]
p = (w[:, None, :] * np.exp(-0.5 * (x[:, :, None] - means[:, None]) ** 2 / vars_) / np.sqrt(2 * np.pi * vars_)).sum(2)
pw = p / p.sum(1, keepdims=True)                                     # grid weights
top = np.argsort(-w, 1)[:, :Kc]
mu = np.take_along_axis(means, top, 1); var = vars_[top].copy(); pi = np.take_along_axis(w, top, 1); pi /= pi.sum(1, keepdims=True)
for it in range(300):
    lr = np.log(pi)[:, None] - 0.5 * np.log(2 * np.pi * var)[:, None] - 0.5 * (x[:, :, None] - mu[:, None]) ** 2 / var[:, None]
    lr -= lr.max(2, keepdims=True); r = np.exp(lr); r /= r.sum(2, keepdims=True)
    rw = r * pw[:, :, None]; Nk = rw.sum(1) + 1e-12
    pi = Nk / Nk.sum(1, keepdims=True)
    mu = (rw * x[:, :, None]).sum(1) / Nk
    var = np.maximum((rw * (x[:, :, None] - mu[:, None]) ** 2).sum(1) / Nk, 1e-4)
T = {'comp_means': means, 'comp_vars': vars_, 'w_bayes': w}
kl = P.kl_grid(T, (mu, 0.5 * np.log(var), np.log(pi)), chunk=200)
json.dump(dict(k4_floor_kl=float(kl.mean()), k4_floor_kl_q90=float(np.quantile(kl, 0.9))), open('out/k4floor.json', 'w'))
print('K4 floor', kl.mean(), np.quantile(kl, [0.5, 0.9, 0.99]))
