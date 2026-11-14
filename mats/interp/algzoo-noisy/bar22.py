"""ARC's bar for noisy M_{2,2} in three columns (PREREG Amendment 5, prediction B3), over the estim22 family.
(a) the closed form (fixed-gate linear response; exact22.py), deductive;
(b) control variate: p_lin (closed form) + mean over N coupled samples of [e_net(x, eps) - e_lin(x, eps)], where e_lin is
    the error indicator of the fixed-gate model on the same input and noise (its mean is exactly p_lin). Unbiased;
(c) Monte Carlo.
usage: python bar22.py ESTIM22.json MODEL.npz OUT.json
"""
import sys, json, math
import numpy as np
import az, exact22

E = json.load(open(sys.argv[1])); path = sys.argv[2]; out = sys.argv[3]
W0, r0, sigma = az.load_npz(path)
DELTA = E['DELTA']; FL = E['flops_per_sample']
Ns = [2 ** k for k in (6, 8, 10, 12, 14)]; R = 64
res = dict(model=path, sigma=sigma, a=[], b={N: [] for N in Ns}, c={N: [] for N in Ns}, flops_closed=None)
FL_LIN = 2 * FL                                   # network + fixed-gate model on the same sample
for row in E['rows']:
    k = row['k']
    g = np.random.default_rng(2000 + k)
    W = [w.copy() for w in W0] if k == 0 else [w + DELTA * g.standard_normal(w.shape) for w in W0]
    m = az.model_from(W, r=r0, sigma=sigma); m.calibrate_r(2 ** 18, iters=3); r = m.r.double().numpy()
    p_lin, fl = exact22.error_rate(W, r, sigma); res['flops_closed'] = fl
    truth = row['p_mc']
    res['a'].append(dict(k=k, err=p_lin - truth, truth_se=row['p_mc_se']))
    W_hi, W_hh, W_oh = W
    rng = np.random.default_rng(777 + k)
    for N in Ns:
        for rep in range(R):
            x = rng.standard_normal((N, 2)); e1 = rng.standard_normal((N, 2)); e2 = rng.standard_normal((N, 2))
            y = (x[:, 0] < x[:, 1]).astype(int) * 0 + (x[:, 0] > x[:, 1]).astype(int)   # target: 2nd argmax = argmin
            y = np.where(x[:, 0] < x[:, 1], 0, 1)
            h1 = np.maximum(x[:, :1] * W_hi, 0) + sigma * r[0] * e1
            z2 = h1 @ W_hh.T + x[:, 1:2] * W_hi
            h2 = np.maximum(z2, 0) + sigma * r[1] * e2
            en = (h2 @ W_oh.T).argmax(1) != y
            h10 = np.maximum(x[:, :1] * W_hi, 0); z20 = h10 @ W_hh.T + x[:, 1:2] * W_hi
            h2l = (z20 > 0) * z2 + sigma * r[1] * e2
            el = (h2l @ W_oh.T).argmax(1) != y
            res['b'][N].append((p_lin + (en.astype(float) - el.astype(float)).mean() - truth) ** 2)
            res['c'][N].append((en.mean() - truth) ** 2)
    print(k, 'truth %.6f closed %.6f' % (truth, p_lin), flush=True)
errs = np.array([a['err'] for a in res['a']]); tse = np.array([a['truth_se'] for a in res['a']])
summary = dict(a=dict(flops=res['flops_closed'], mse=float((errs ** 2).mean()), truth_noise=float((tse ** 2).mean())),
               b=[dict(N=N, flops=res['flops_closed'] + N * FL_LIN, mse=float(np.mean(res['b'][N]))) for N in Ns],
               c=[dict(N=N, flops=N * FL, mse=float(np.mean(res['c'][N]))) for N in Ns])
res['summary'] = summary; res['b'] = {str(k): v for k, v in res['b'].items()}; res['c'] = {str(k): v for k, v in res['c'].items()}
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(summary, indent=0))
