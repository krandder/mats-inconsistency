"""Part 1 (PREREG Part 1): exact annealed replica Gram chain for the input regime (weights integrated out), sigma = 0.
usage: part1chain.py n [--P 20000] [--out file]
Pairs of independent Gaussian inputs (chain A) and pairs differing in one coordinate (chain B: x and x^(j), coordinate j
redrawn); exact two-replica chain to max depth. Per layer: annealed Hamming E n arccos(rho_{l-1})/pi (chain A). Per depth L
in D (as part1a): annealed split terms E_W[V], E_W[V_rad], E_W[V_amp], E_W[V_dir] (chain A) with SEs; annealed
single-coordinate total effect E|a - a^(j)|^2/2 (chain B) and D_ann = n E|a - a^(j)|^2/2 / E_W[V] (the variance-weighted
mean dimension over nets). Kernel numbers alongside: rho_l = kappa^(l)(0), Hamming_ker = n arccos(rho_{l-1})/pi,
delta_L, Lambda(L) and the predictions V_dir ~ n delta_L (E A)^2..., V_amp ~ (E A)^2 m_n^2 Lambda/(4n). Seeds [7101, n]."""
import argparse, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('--P', type=int, default=20000)
ap.add_argument('--out')
a = ap.parse_args()
n, P = a.n, a.P
t0 = time.time()
D = sorted({max(1, n // 8), n // 4, n // 2, n, 2 * n, 32} | ({4 * n} if n <= 64 else set()))
Lmax = max(D)
rng = np.random.default_rng([7101, n])
mn = ib.mean_radius(n)


def J2(t):
    return 3 * np.sin(t) * np.cos(t) + (math.pi - t) * (1 + 2 * np.cos(t) ** 2)


# kernel numbers
rho = 0.0; Lam = 0.0; ker = {}
for l in range(1, Lmax + 1):
    th = math.acos(rho)
    Lam += 6 - (2 / math.pi) * J2(th)
    ham = n * th / math.pi
    rho = float(ib.kappa(rho))
    ker[l] = dict(rho=rho, delta=1 - rho, ham=ham, Lambda=Lam)
res = dict(n=n, P=P, depths=D, kernel=ker, chainA={}, chainB={}, ham_layers=[])
CH = 5000
accA = {L: {k: [] for k in ('V', 'V_rad', 'V_amp', 'V_dir')} for L in D}
hamsum = np.zeros(Lmax); hamsq = np.zeros(Lmax); hamN = 0
accB = {L: [] for L in D}
for c0 in range(0, P, CH):
    m = min(CH, P - c0)
    Xa = rng.standard_normal((m, n)); Xb = rng.standard_normal((m, n))
    j = rng.integers(0, n, m)
    Xc = Xa.copy(); Xc[np.arange(m), j] = rng.standard_normal(m)
    for (X1, X2, tag) in ((Xa, Xb, 'A'), (Xa, Xc, 'B')):
        G = np.empty((m, 2, 2))
        G[:, 0, 0] = np.sum(X1 * X1, 1) / n; G[:, 1, 1] = np.sum(X2 * X2, 1) / n
        G[:, 0, 1] = G[:, 1, 0] = np.sum(X1 * X2, 1) / n
        xn2 = np.stack([np.sum(X1 * X1, 1), np.sum(X2 * X2, 1)], 1)
        hl = []

        def rec(l, Gp, Gn, tag=tag, xn2=xn2):
            if tag == 'A':
                hl.append(ib.chain_hamming(Gp, n))
            if l in D:
                if tag == 'A':
                    T = ib.chain_split_terms(Gn, n, xn2)
                    for k in accA[l]:
                        accA[l][k].append(T[k])
                else:
                    accB[l].append(0.5 * n * (Gn[:, 0, 0] + Gn[:, 1, 1] - 2 * Gn[:, 0, 1]))
        ib.gram_chain_inputs(G, Lmax, n, rng, record=rec)
        if tag == 'A':
            H = np.array(hl)          # (Lmax, m)
            hamsum += H.sum(1); hamsq += (H * H).sum(1); hamN += m
    print(f'chunk {c0} done {time.time() - t0:.0f}s', flush=True)
for l in range(1, Lmax + 1):
    mu = hamsum[l - 1] / hamN
    se = math.sqrt(max(hamsq[l - 1] / hamN - mu * mu, 0) / hamN)
    res['ham_layers'].append(dict(l=l, ham=mu, se=se, ham_ker=ker[l]['ham']))
for L in D:
    out = {}
    for k, v in accA[L].items():
        v = np.concatenate(v)
        out[k] = float(v.mean()); out[k + '_se'] = float(v.std() / math.sqrt(len(v)))
    b = np.concatenate(accB[L])
    out['T1'] = float(b.mean()); out['T1_se'] = float(b.std() / math.sqrt(len(b)))
    out['D_ann'] = n * out['T1'] / out['V']
    out['s_rad'] = out['V_rad'] / out['V']; out['s_amp'] = out['V_amp'] / out['V']; out['s_dir'] = out['V_dir'] / out['V']
    k = ker[L]
    out['ratio_dir_amp_pred'] = 4 * n * k['delta'] / k['Lambda']
    out['ratio_dir_amp_chain'] = out['V_dir'] / out['V_amp']
    res['chainA'][str(L)] = out
    print(f'L={L}: s_rad {out["s_rad"]:.3f} s_amp {out["s_amp"]:.3f} s_dir {out["s_dir"]:.3f} V {out["V"]:.3f} '
          f'dir/amp chain {out["ratio_dir_amp_chain"]:.3f} kernel {out["ratio_dir_amp_pred"]:.3f} D_ann {out["D_ann"]:.2f}', flush=True)
res['sec'] = time.time() - t0
if a.out:
    json.dump(res, open(a.out, 'w'))
print('done', res['sec'])
