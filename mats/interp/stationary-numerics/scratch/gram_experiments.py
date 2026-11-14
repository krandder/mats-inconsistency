"""Exact two/four-replica Gram-chain experiments (annealed over weights, quenched over noise replicas).
Window of J = 4n layers after a shared deep past whose radius Q0 is drawn from the exact stationary law.
MODE window (R = 3): replica 0 base; replica 1 independent window noise (cloud pair (0,1): V and delta);
  replica 2 = replica 0 except one layer j uniform in the window (single refresh, pair (0,2): total effect T_j).
  Interaction order D_f = J * mean d_f(0,2) / mean d_f(0,1) (unbiased ratio-of-means estimator of sum_j T_j for
  f in h, u (direction), logR, a = relu(W_1 h), g (output gates)); also by bins of the final radius Q of replica 0,
  the age profile of T_j, and the gate-flip cascade n arccos(rho_02(t))/pi at every layer t after the refresh.
MODE pr (R = 4, all independent): annealed tr C and tr C^2 of the quenched cloud of h and of the direction u
  (tr C^2 = E[(x1-x2).(x3-x4)]^2/4 from four independent replicas), participation ratios.
usage: gram_experiments.py MODE n --P P --seed s --out file"""
import sys, os, math, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import numpy as np
import gram_chain as gc

ap = argparse.ArgumentParser()
ap.add_argument('mode'); ap.add_argument('n', type=int)
ap.add_argument('--P', type=int, default=1000); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
n, P = a.n, a.P
J = 4 * n
rng = np.random.default_rng([5150, n, a.seed, 0 if a.mode == 'window' else 1])
t0 = time.time()
Q0 = gc.sample_Q_stationary(n, P, rng)
res = dict(mode=a.mode, n=n, P=P, J=J, seed=a.seed)
QB = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e300]  # bins of Q_final / n

if a.mode == 'window':
    R = 3
    G = np.repeat(Q0[:, None, None], R, 1).repeat(R, 2).astype(float)
    jref = rng.integers(0, J, P)  # layer index (0 = oldest) of the refresh
    casc = np.zeros(2 * n + 1)
    ccnt = np.zeros(2 * n + 1)
    dcasc = np.zeros(2 * n + 1)
    for t in range(J):
        lab = np.tile(np.array([0, 1, 0]), (P, 1))
        lab[jref == t, 2] = 2
        G = gc.step(G, lab, n, rng)
        after = t - jref  # 0 at the refresh layer itself
        sel = (after >= 0) & (after <= 2 * n)
        if sel.any():
            d = gc.diffs(G[sel], n, 0, 2)
            k = after[sel]
            np.add.at(casc, k, d['g'])
            np.add.at(dcasc, k, d['u'])
            np.add.at(ccnt, k, 1.0)
    d01 = gc.diffs(G, n, 0, 1)
    d02 = gc.diffs(G, n, 0, 2)
    age = J - 1 - jref
    keys = ('h', 'u', 'logR', 'a', 'g')
    res['D'] = {k: float(J * d02[k].mean() / d01[k].mean()) for k in keys}
    # bootstrap SE over chains
    B = 200
    idx = rng.integers(0, P, (B, P))
    res['D_se'] = {k: float(np.std([J * d02[k][i].mean() / d01[k][i].mean() for i in idx])) for k in keys}
    Qf = G[:, 0, 0]
    res['D_by_Qbin'] = []
    for lo, hi in zip(QB[:-1], QB[1:]):
        m = (Qf / n >= lo) & (Qf / n < hi)
        if m.sum() >= 30:
            res['D_by_Qbin'].append(dict(lo=lo, hi=hi, count=int(m.sum()), Qmed_over_n=float(np.median(Qf[m]) / n),
                                         D={k: float(J * d02[k][m].mean() / d01[k][m].mean()) for k in keys},
                                         delta_mean=float(np.mean(1 - d01['rho'][m])),
                                         delta_median=float(np.median(1 - d01['rho'][m])),
                                         sR2=float(np.mean(d01['logR'][m]))))
    # age profile of T_j (normalised by V) in octave bins of age
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (age >= lo) & (age < hi)
        if m.sum() > 0:
            prof.append(dict(lo=lo, hi=hi, count=int(m.sum()),
                             T={k: float(d02[k][m].mean() / d01[k].mean()) for k in keys}))
    res['age_profile'] = prof
    with np.errstate(invalid='ignore'):
        res['cascade_flips_per_layer'] = (casc / np.maximum(ccnt, 1)).tolist()
        res['cascade_delta_u'] = (dcasc / np.maximum(ccnt, 1)).tolist()
    res['cascade_2n_total_over_n'] = float(np.sum(casc[1:] / np.maximum(ccnt[1:], 1)) / n)
    res['births_first_layer'] = float(casc[0] / max(ccnt[0], 1))
    res['births_over_sqrt_n'] = res['births_first_layer'] / math.sqrt(n)
    res['delta_cloud'] = dict(mean=float(np.mean(1 - d01['rho'])), median=float(np.median(1 - d01['rho'])))
    res['V_h_over_nQ'] = float(np.mean(d01['h'] / (2 * n * Qf)))
    res['sR2_mean'] = float(np.mean(d01['logR']))
    res['Qf_quantiles_over_n'] = [float(np.quantile(Qf, q) / n) for q in (0.1, 0.5, 0.9)]
    # per-chain records (small) for later binning
    res['chains'] = dict(Q0=Q0.tolist(), Qf=Qf.tolist(), age=age.tolist(),
                         d01={k: d01[k].tolist() for k in ('h', 'u', 'logR', 'a', 'g')},
                         d02={k: d02[k].tolist() for k in ('h', 'u', 'logR', 'a', 'g')})
elif a.mode == 'pr':
    R = 4
    G = np.repeat(Q0[:, None, None], R, 1).repeat(R, 2).astype(float)
    lab = np.tile(np.arange(R), (P, 1))
    for t in range(J):
        G = gc.step(G, lab, n, rng)
    dg = np.sqrt(np.einsum('pii->pi', G))
    Rho = G / (dg[:, :, None] * dg[:, None, :])
    xh = n * (G[:, 0, 2] - G[:, 0, 3] - G[:, 1, 2] + G[:, 1, 3])
    xu = Rho[:, 0, 2] - Rho[:, 0, 3] - Rho[:, 1, 2] + Rho[:, 1, 3]
    trCh = n * (G[:, 0, 0] + G[:, 1, 1] - 2 * G[:, 0, 1]) / 2
    trCu = (1 - Rho[:, 0, 1])
    Qf = G[:, 0, 0]
    res['by_Qbin'] = []
    for lo, hi in zip(QB[:-1], QB[1:]):
        m = (Qf / n >= lo) & (Qf / n < hi)
        if m.sum() >= 30:
            th, t2h = trCh[m].mean(), (xh[m] ** 2).mean() / 4
            tu, t2u = trCu[m].mean(), (xu[m] ** 2).mean() / 4
            res['by_Qbin'].append(dict(lo=lo, hi=hi, count=int(m.sum()), Qmed_over_n=float(np.median(Qf[m]) / n),
                                       trC_h=float(th), trC2_h=float(t2h), PR_h=float(th ** 2 / t2h),
                                       trC_u=float(tu), trC2_u=float(t2u), PR_u=float(tu ** 2 / t2u)))
    th, t2h = trCh.mean(), (xh ** 2).mean() / 4
    tu, t2u = trCu.mean(), (xu ** 2).mean() / 4
    res['PR_h'] = float(th ** 2 / t2h); res['PR_u'] = float(tu ** 2 / t2u)
    B = 200
    idx = rng.integers(0, P, (B, P))
    res['PR_u_se'] = float(np.std([trCu[i].mean() ** 2 / ((xu[i] ** 2).mean() / 4) for i in idx]))
    res['PR_h_se'] = float(np.std([trCh[i].mean() ** 2 / ((xh[i] ** 2).mean() / 4) for i in idx]))
res['sec'] = time.time() - t0
json.dump(res, open(a.out, 'w'))
print(json.dumps({k: v for k, v in res.items() if k not in ('chains', 'cascade_flips_per_layer', 'cascade_delta_u')}),
      flush=True)
