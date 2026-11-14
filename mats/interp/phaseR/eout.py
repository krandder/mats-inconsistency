"""Task 3a: E[output] estimation, ARC's metric. usage: python eout.py d L act g sigma bias seed
Writes results/eout_<net>.json: per-output normalised MSE (by the single-sample variance) of
  K=2 (ARC-style deductive moment propagation, mean-field LN with delta-method mean correction),
  the sample-fitted Gaussian chain (our forward chain, G family) at N_fit = 2^10, 2^12, 2^14 per layer,
against the truth from plain MC (2^22 samples, last noise Rao-Blackwellised); plain MC with N samples has normalised
MSE exactly 1/N. FLOPs counted analytically (rnet.flops_*).
"""
import json, math, os, sys, time
import torch
import rnet as R, fam as F

torch.set_num_threads(1)
DT = torch.float64
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def run(d, L, act, g, sigma, bias, seed):
    net = R.RNet(d, L, act, g, sigma, seed, bias=bias)
    path = os.path.join(OUT, f'eout_{net.name()}.json')
    gen = torch.Generator().manual_seed(500 + seed)
    t0 = time.time()
    ntr = (1 << 22) if d <= 64 else (1 << 20)
    net.set_dtype(torch.float32)
    S1, S2, _ = net.mc_outputs(ntr, gen, chunk=1 << 14)
    net.set_dtype(DT)
    Ey = S1 / ntr; Vrb = S2 / ntr - Ey ** 2
    V1 = Vrb + sigma ** 2 * (net.Wout ** 2).sum(1)              # single-sample variance of y_j
    t_truth = time.time() - t0
    res = dict(d=d, L=L, act=act, g=g, sigma=sigma, bias=bias, seed=seed, n_truth=ntr,
               truth_nmse_floor=float((Vrb / V1).mean() / ntr), flops_sample=net.flops_sample(), t_truth=t_truth)
    t1 = time.time(); mu, S = R.k2_propagate(net); t_k2 = time.time() - t1
    e = net.Wout @ mu
    res['k2'] = dict(nmse=float((((e - Ey) ** 2) / V1).mean()), flops=net.flops_k2(), secs=t_k2)
    mu0, _ = R.k2_propagate(net, ln_corr=False)
    res['k2_nolncorr'] = dict(nmse=float((((net.Wout @ mu0 - Ey) ** 2) / V1).mean()))
    res['k2']['N_eq'] = res['k2']['flops'] / net.flops_sample()
    res['k2']['mc_nmse_at_equal_flops'] = 1.0 / res['k2']['N_eq']
    res['chain'] = []
    for nfit in (1 << 10, 1 << 12, 1 << 14):
        q = F.Gauss(torch.zeros(d, dtype=DT), torch.eye(d, dtype=DT))
        for l in range(1, L + 1):
            X = q.sample(nfit, gen); H = net.f(l, X) + sigma * torch.randn(nfit, d, generator=gen, dtype=DT)
            q = F.Gauss(H.mean(0), torch.cov(H.T) + 1e-9 * torch.eye(d, dtype=DT))
        fl = L * nfit * (2 * d * d + 12 * d) + L * nfit * d * d + 2 * d * d   # pushes + covariance fits + readout
        res['chain'].append(dict(nfit=nfit, nmse=float((((net.Wout @ q.mu - Ey) ** 2) / V1).mean()), flops=fl,
                                 mc_nmse_at_equal_flops=net.flops_sample() / fl))
    json.dump(res, open(path, 'w'), indent=1)
    print(net.name(), f"K2 nmse {res['k2']['nmse']:.2e} vs MC@equal-FLOPs {res['k2']['mc_nmse_at_equal_flops']:.2e} "
          f"(floor {res['truth_nmse_floor']:.1e}); chain " + ' '.join(f"{c['nfit']}:{c['nmse']:.2e}/{c['mc_nmse_at_equal_flops']:.1e}" for c in res['chain'])
          + f"  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == '__main__':
    a = sys.argv
    run(int(a[1]), int(a[2]), a[3], float(a[4]), float(a[5]), float(a[6]), int(a[7]))
