"""Propagated-noise metric (the program's DAG node weights): how much noise injected at node (t, i) moves the question.

Along each input's noiseless path, with the gates of that path held fixed (linear response), the target margin
m = l_target - max_other l moves by sum_{t,i} (dm/dh_{t,i}) * sigma * r_{t,i} * eps_{t,i}.  Its noise s.d. is
s(x) = sigma * sqrt(sum_{t,i} (dm/dh_{t,i} r_{t,i})^2).  Reported for a test sigma:
 - predicted noisy error (linear response): E_x[ Phi(-m(x)/s(x)) ] for correct x, + E_x[Phi(+|m|/s)] for wrong x
   (pairwise with the top competitor only; a lower bound on the flip rate up to the linearisation);
 - measured error with noise at that sigma (2^20 samples);
 - node shares: E_x[ w(x) (dm/dh_{t,i} r_{t,i})^2 / sum ] with w(x) = phi(m/s)/s (density of crossing the boundary);
 - the margin in noise units, quantiles of |m|/s.
usage: python noisesens.py MODEL.npz SIGMA_TEST OUT.json   (r is recalibrated at SIGMA_TEST if the model's sigma is 0)
"""
import sys, json, math
import numpy as np
import torch
from scipy.stats import norm
import az

torch.set_num_threads(1)
path, sg, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
W, r, sigma0 = az.load_npz(path)
W_hi, W_hh, W_oh = W; d = W_hi.shape[0]; n = W_oh.shape[0]
m_ = az.model_from(W, r=r if sigma0 > 0 else None, sigma=sg)
if sigma0 == 0 or abs(sigma0 - sg) > 1e-12:
    m_.calibrate_r(2 ** 18, iters=3)
r = m_.r.double().numpy()
meas = az.accuracy(m_, 2 ** 20, noise=True, seed=5)
rng = np.random.default_rng(8)
B = 2 ** 17
x = rng.standard_normal((B, n)); y = np.argsort(x, -1)[:, -2]
hs = [np.zeros((B, d))]; zs = []
for t in range(n):
    z = hs[-1] @ W_hh.T + x[:, t:t + 1] * W_hi; zs.append(z); hs.append(np.maximum(z, 0))
lo = hs[-1] @ W_oh.T
lo2 = lo.copy(); lo2[np.arange(B), y] = -np.inf; j = lo2.argmax(1)
mg = lo[np.arange(B), y] - lo[np.arange(B), j]
g = W_oh[y] - W_oh[j]                       # dm/dh_n
contrib = np.zeros((B, n, d))
for t in range(n, 0, -1):                   # hs[t] = relu(zs[t-1]) + noise_t
    contrib[:, t - 1] = (g * r[t - 1]) ** 2
    gz = g * (zs[t - 1] > 0)
    g = gz @ W_hh
s = sg * np.sqrt(contrib.sum((1, 2)))
# all-competitor linear response (review 1, finding 2): logit noise covariance with the noiseless path's gates fixed,
# Sigma_l = sum_t G_t diag((sigma r_t)^2) G_t^T, G_t = d logits / d h_t; error = P(argmax(l0 + noise) != y) by K draws
Bs = 2 ** 15
Gm = np.broadcast_to(W_oh, (Bs, n, d)).copy(); Sig = np.zeros((Bs, n, n))
for t in range(n, 0, -1):
    Sig += np.einsum('bkd,bld->bkl', Gm * (sg * r[t - 1]), Gm * (sg * r[t - 1]))
    Gm = np.einsum('bkd,de->bke', Gm * (zs[t - 1][:Bs] > 0)[:, None, :], W_hh)
Lc = np.linalg.cholesky(Sig + 1e-12 * np.eye(n))
errs = []
for k in range(16):
    z = rng.standard_normal((Bs, n))
    lo_k = lo[:Bs] + np.einsum('bkl,bl->bk', Lc, z)
    errs.append((lo_k.argmax(1) != y[:Bs]).mean())
pred_all = float(np.mean(errs))
c = mg / np.maximum(s, 1e-300)
pred = np.where(mg > 0, norm.cdf(-c), norm.cdf(-c)).mean()     # P(sign flips to wrong) for correct; P(stays wrong) for wrong
pred_err = float(np.where(mg > 0, norm.cdf(-c), 1 - norm.cdf(c)).mean())
w = norm.pdf(c) / np.maximum(s, 1e-300)
share = (w[:, None, None] * contrib / np.maximum(contrib.sum((1, 2)), 1e-300)[:, None, None]).sum(0) / w.sum()
res = dict(model=path, sigma_test=sg, model_sigma=sigma0, err_noiseless=float((mg <= 0).mean()), err_pred_linear=pred_err,
           err_pred_linear_all_competitors=pred_all,
           err_measured=float(1 - meas['acc']), err_measured_se=meas['se'],
           abs_margin_over_noise_quantiles={q: float(np.quantile(np.abs(c), q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5)},
           node_share=share.tolist(), step_share=share.sum(1).tolist(), neuron_share=share.sum(0).tolist(),
           r_scale=r.tolist())
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('node_share', 'r_scale')}, indent=0))
