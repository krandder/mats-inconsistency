"""ARC's bar on noisy M_{2,2}: the closed-form description (exact22.py) against Monte Carlo at equal FLOPs, over a
family of weight-perturbed copies (ARC's protocol for M_{4,3}: add N(0, DELTA^2) to every weight).
usage: python estim22.py MODEL.npz OUT.json NMODELS DELTA
"""
import sys, json, math, time
import numpy as np
import torch
import az, exact22

torch.set_num_threads(1)
t0 = time.time()
path, out, NM, DELTA = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
W0, r0, sigma = az.load_npz(path)
d, n = 2, 2
FL_SAMPLE = n * (2 * d * d + 2 * d + d + 2 * d) + 2 * n * d + n + 1
rows = []
for k in range(NM):
    g = np.random.default_rng(2000 + k)
    W = [w.copy() for w in W0] if k == 0 else [w + DELTA * g.standard_normal(w.shape) for w in W0]
    m = az.model_from(W, r=r0, sigma=sigma); m.calibrate_r(2 ** 18, iters=3); r = m.r.double().numpy()
    acc = az.accuracy(m, 2 ** 26, noise=True, seed=300 + k)
    p = 1 - acc['acc']
    pe, fl = exact22.error_rate(W, r, sigma)
    rows.append(dict(k=k, p_mc=p, p_mc_se=acc['se'], p_exact=pe, flops_exact=fl))
    print(k, 'MC %.6f +- %.6f  closed form %.6f  flops %d' % (p, acc['se'], pe, fl), flush=True)
P = np.array([r_['p_mc'] for r_ in rows]); E = np.array([r_['p_exact'] for r_ in rows]); SE = np.array([r_['p_mc_se'] for r_ in rows])
fl = float(np.mean([r_['flops_exact'] for r_ in rows]))
mse_exact = float(((E - P) ** 2).mean()); mse_exact_noise_corrected = float(max(((E - P) ** 2).mean() - (SE ** 2).mean(), 0))
N_eq = fl / FL_SAMPLE
mse_mc_eq = float((P * (1 - P)).mean() / N_eq)
res = dict(model=path, sigma=sigma, DELTA=DELTA, rows=rows, flops_per_sample=FL_SAMPLE, flops_exact_mean=fl,
           mse_exact_vs_truth=mse_exact, truth_se_sq_mean=float((SE ** 2).mean()), mse_exact_minus_truth_noise=mse_exact_noise_corrected,
           mc_samples_at_equal_flops=N_eq, mse_mc_at_equal_flops=mse_mc_eq,
           mc_samples_to_match_exact=float((P * (1 - P)).mean() / max(mse_exact, 1e-300)),
           p_mean=float(P.mean()), p_sd=float(P.std()), seconds=time.time() - t0)
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=0))
