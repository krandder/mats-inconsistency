"""ARC's bar on a noisy RNN: estimate the error rate of each model in a family (base + weight-perturbed copies) and
compare estimators by MSE (over models and repetitions) against FLOPs.
usage: python estim.py MODEL.npz OUT.json NMODELS DELTA [TRUTH_LOG2]
Perturbation: each weight matrix gets i.i.d. N(0, (DELTA * rms(matrix))^2) added; r is recalibrated per copy.
Estimators: Monte Carlo; stratified over the n(n-1) algorithmic cells (a, b) (exact conditional sampling, proportional
allocation); RQMC (scrambled Sobol over x, noise plain MC); the deductive alg-cell x gate ADF (adf.py), several budgets;
the constant (leave-one-out family mean). FLOPs per sampled sequence are counted as matmul/elementwise operations plus
20 per Gaussian draw (reported separately).
"""
import sys, json, time, math
import numpy as np
import torch
from scipy.stats import qmc, norm
import az, adf

torch.set_num_threads(1)
t0 = time.time()
path, out, NM, DELTA = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
TL = int(sys.argv[5]) if len(sys.argv) > 5 else 24
W0, r0, sigma = az.load_npz(path)
d = W0[0].shape[0]; n = W0[2].shape[0]
rng = np.random.default_rng(4040)

FL_SAMPLE = n * (2 * d * d + 2 * d + d + 2 * d) + 2 * n * d + n + n * math.log2(n)
RNG_SAMPLE = 20 * (n + n * d)


def perturbed(k):
    if k == 0: return [w.copy() for w in W0]
    g = np.random.default_rng(1000 + k)
    return [w + DELTA * np.sqrt((w ** 2).mean()) * g.standard_normal(w.shape) for w in W0]


@torch.no_grad()
def batch_err(m, x, gen):
    lo = m(torch.as_tensor(x, dtype=torch.float32), noise=True, gen=gen)
    return (lo.argmax(-1).numpy() != np.argsort(x, -1)[:, -2])


def strat_sample(N, rng):
    """N sequences with exactly N/(n(n-1)) per cell (a, b) (rounded), proportional allocation."""
    cells = [(a, b) for a in range(n) for b in range(n) if a != b]
    per = N // len(cells)
    X = np.empty((per * len(cells), n)); idx = 0; lab = []
    for (a, b) in cells:
        v = np.sort(rng.standard_normal((per, n)), 1)
        others = [j for j in range(n) if j not in (a, b)]
        perm = np.argsort(rng.random((per, n - 2)), 1)
        X[idx:idx + per, a] = v[:, -1]; X[idx:idx + per, b] = v[:, -2]
        X[idx:idx + per][:, others] = np.take_along_axis(v[:, :-2], perm, 1)
        idx += per; lab += [(a, b)] * per
    return X, per


models = []
for k in range(NM):
    W = perturbed(k)
    m = az.model_from(W, r=r0, sigma=sigma)
    m.calibrate_r(2 ** 18, iters=2)
    g = torch.Generator().manual_seed(500 + k)
    tot = 0; cnt = 0
    for i in range(2 ** (TL - 16)):
        x = np.random.default_rng(10 ** 6 + 1000 * k + i).standard_normal((2 ** 16, n))
        tot += batch_err(m, x, g).sum(); cnt += 2 ** 16
    p = tot / cnt
    models.append(dict(k=k, p=float(p), p_se=float(math.sqrt(p * (1 - p) / cnt)), r=m.r.double().numpy(), W=W, m=m))
    print('model', k, 'p %.5f' % p, flush=True)
P = np.array([mm['p'] for mm in models])
res = dict(model=path, sigma=sigma, NM=NM, DELTA=DELTA, truths=P.tolist(), truth_se=[mm['p_se'] for mm in models],
           flops_per_sample=FL_SAMPLE, rng_flops_per_sample=RNG_SAMPLE, est={})
Ns = [2 ** k for k in range(8, 17, 2)]
R = 8
for name in ('mc', 'strat', 'rqmc'):
    res['est'][name] = []
    for N in Ns:
        se2 = []
        for mm in models:
            for rep in range(R):
                g = torch.Generator().manual_seed(7 * N + 31 * rep + 1000 * mm['k'])
                rr = np.random.default_rng(13 * N + 17 * rep + 1000 * mm['k'])
                if name == 'mc':
                    x = rr.standard_normal((N, n)); e = batch_err(mm['m'], x, g).mean()
                elif name == 'strat':
                    x, per = strat_sample(N, rr); e = batch_err(mm['m'], x, g).mean()
                else:
                    sob = qmc.Sobol(d=n, scramble=True, seed=rr).random(N)
                    x = norm.ppf(np.clip(sob, 1e-12, 1 - 1e-12)); e = batch_err(mm['m'], x, g).mean()
                se2.append((e - mm['p']) ** 2)
        mse = float(np.mean(se2)); res['est'][name].append(dict(N=N, flops=N * FL_SAMPLE, mse=mse, mse_se=float(np.std(se2) / math.sqrt(len(se2)))))
        print(name, N, 'mse %.3g' % mse, flush=True)
    json.dump(res, open(out, 'w'), indent=1)
# constant (leave-one-out mean)
loo = np.array([(P.sum() - P[i]) / (len(P) - 1) for i in range(len(P))])
res['est']['constant'] = [dict(flops=0, mse=float(((loo - P) ** 2).mean()))]
# deductive ADF at several budgets
res['est']['adf'] = []
import os
for ks in [int(v) for v in os.environ.get('ADF_KS', '0,2,4,6').split(',') if v != '']:
    errs = []; fls = []; secs = []
    for mm in models:
        t1 = time.time()
        A = adf.ADF(mm['W'], mm['r'], sigma, k_split=ks)
        comps = A.run(); e, _ = A.error_rate(comps, 'seq')
        errs.append(e - mm['p']); fls.append(A.fl.n); secs.append(time.time() - t1)
    errs = np.array(errs)
    row = dict(k_split=ks, flops=float(np.mean(fls)), mse=float((errs ** 2).mean()), bias=float(errs.mean()),
               est=(P + errs).tolist(), seconds=float(np.mean(secs)))
    res['est']['adf'].append(row); print('adf', ks, row['flops'], row['mse'], row['bias'], flush=True)
    json.dump(res, open(out, 'w'), indent=1)
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
print('done', res['seconds'])
