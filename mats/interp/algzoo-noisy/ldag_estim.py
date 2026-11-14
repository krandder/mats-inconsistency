"""ARC's bar with the accuracy-targeted structure (PREREG Amendment 4/4b), on the T16 family of estim.py.
 (i)  deductive: the ADF run on the dp-targeted surrogate (masked weights + mean biases); the means mu_t are estimated
      once per model from 2^12 sampled sequences, and that setup is counted in the FLOPs (so strictly this is
      'deductive after a sampled setup');
 (ii) hybrid (does not count toward the bar): Monte Carlo on the surrogate, which is cheaper per sequence; its MSE vs
      the network's truth includes the surrogate's bias.
usage: python ldag_estim.py MODEL.npz LDAG.json ESTIM.json OUT.json
"""
import sys, json, math, time
import numpy as np
import az, adf, dag

t0 = time.time()
path, lpath, epath, out = sys.argv[1:5]
W0, r0, sigma = az.load_npz(path)
d = W0[0].shape[0]; n = W0[2].shape[0]
L = json.load(open(lpath)); E = json.load(open(epath))
row = [x for x in L['runs'] if x['target'] == 'dp' and x['Nref_log2'] == 14 and x.get('edges')][0]
keep = dict(rec=np.zeros((n, d, d), bool), inp=np.zeros((n, d), bool), out=np.zeros(d, bool))
for (k, t, i, j) in row['edges']:
    if k == 'rec': keep['rec'][t, i, j] = True
    elif k == 'inp': keep['inp'][t, i] = True
    else: keep['out'][i] = True
n_kept = row['K']
P = np.array(E['truths']); NM = len(P); DELTA = E['DELTA']
FL_NET = E['flops_per_sample']
FL_SUR = 2 * (keep['rec'].sum() + keep['inp'].sum() + keep['out'].sum()) + n * (d + 2 * d) + 2 * n * d + n + n * math.log2(n)


def perturbed(k):
    if k == 0: return [w.copy() for w in W0]
    g = np.random.default_rng(1000 + k)
    return [w + DELTA * np.sqrt((w ** 2).mean()) * g.standard_normal(w.shape) for w in W0]


res = dict(model=path, edges_kept=int(n_kept), edges_total=L['n_edges'], flops_per_sample_net=FL_NET, flops_per_sample_sur=float(FL_SUR),
           truths=P.tolist(), adf=[], plugin=[])
adf_err = {0: [], 2: []}; adf_fl = {0: [], 2: []}
plug = {N: [] for N in (2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16)}
for k in range(NM):
    W = perturbed(k)
    m = az.model_from(W, r=r0, sigma=sigma); m.calibrate_r(2 ** 18, iters=2); r = m.r.double().numpy()
    net = dag.Net(W, r, sigma)
    rng = np.random.default_rng(90000 + k)
    _, hs, _ = net.run(rng.standard_normal((2 ** 12, n)), rng.standard_normal((2 ** 12, n, d))); mu = [h.mean(0) for h in hs]
    setup = 2 ** 12 * FL_NET
    for ks in (0, 2):
        A = adf.ADF(W, r, sigma, k_split=ks, keep=keep, mu=mu)
        c = A.run(); e, _ = A.error_rate(c, 'seq')
        adf_err[ks].append(e - P[k]); adf_fl[ks].append(A.fl.n + setup)
    for N in plug:
        ests = []
        for rep in range(8):
            x = rng.standard_normal((N, n)); eps = rng.standard_normal((N, n, d))
            lo, _, _ = net.run(x, eps, keep=keep, mu=mu)
            ests.append(float((lo.argmax(1) != np.argsort(x, -1)[:, -2]).mean()))
        plug[N] += [(e_ - P[k]) ** 2 for e_ in ests]
    print('model', k, 'truth %.4f' % P[k], 'adf k0 %.4f k2 %.4f' % (P[k] + adf_err[0][-1], P[k] + adf_err[2][-1]), flush=True)
for ks in (0, 2):
    ee = np.array(adf_err[ks])
    res['adf'].append(dict(k_split=ks, flops=float(np.mean(adf_fl[ks])), mse=float((ee ** 2).mean()), bias=float(ee.mean())))
mc = {x['N']: x['mse'] for x in E['est']['mc']}
for N, v in plug.items():
    fl = N * FL_SUR + 2 ** 12 * FL_NET
    res['plugin'].append(dict(N=N, flops=float(fl), mse=float(np.mean(v)), mse_se=float(np.std(v) / math.sqrt(len(v)))))
res['mc_reference'] = E['est']['mc']
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('truths', 'mc_reference')}, indent=0))
