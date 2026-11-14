"""Declared fix (PREREG rule "a crash is fixed and rerun with a note"): ARC's released kprop code floors variances at
1e-10 in ABSOLUTE terms, while deep fronts have cut scales down to 1e-7, so the registered ARC2/ARC3 runs were corrupted on
low-scale fronts. ARC's closure is positively homogeneous ((mu, Sigma, kappa3) -> (t mu, t^2 Sigma, t^3 kappa3) scales
the output by t) apart from such floors, so the registered estimator is computed here on unit-scaled cut cumulants
(divided by s_f = sqrt(tr E[h h^T]/n)) and rescaled. K2X is recomputed the same way as a check (it should not change).
Writes runs/NAME_arcunit.npz with A2u, A3u, K2u on the test fronts. usage: arc_unit_rerun.py n c d [--arc3]
"""
import argparse, math, os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/cut-rank/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/ext/mlp_cumulant_propagation/src')
import crnet as cn
import gauss_relu as GR
from mlp_kprop import kprop_harmonic as KH
from mlp_kprop.wick import WICK_COEF_D
import logging
logging.disable(logging.WARNING)
torch.set_num_threads(1)
DT = torch.float64
ap = argparse.ArgumentParser()
for k in ('n', 'c', 'd'):
    ap.add_argument(k, type=int)
ap.add_argument('--arc3', action='store_true')
a = ap.parse_args()
n, c, d = a.n, a.c, a.d
HERE = os.path.dirname(os.path.abspath(__file__))
name = f'nl_n{n}_c{c}_d{d}'
D = np.load(os.path.join(HERE, '..', 'runs', name + '.npz'))
F, B, Np = int(D['F']), int(D['B']), int(D['Np'])
tag = cn.cfg_tag(n, c, d)
Er = cn.mean_radius(n)
Er3 = 2 ** 1.5 * math.exp(math.lgamma((n + 3) / 2) - math.lgamma(n / 2))
backs_t = [[torch.tensor(W.astype(np.float64), dtype=DT) for W in cn.back_weights(tag, n, d, b)] for b in range(B)]


def fwd(H, Ws):
    for W in Ws:
        H = H @ W.T
        np.maximum(H, 0, out=H)
    return H


def arc(K_in, Wb, kmax):
    K = KH.coerce_input(K_in, k_max=kmax, kind=KH.SIMPLE)
    for W in Wb:
        K = KH.linear_kprop(K, W, k_max=kmax, set_metric=2.0, bias=None)
        K = KH.nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D['relu'], k_max=kmax, kind=KH.SIMPLE, use_pK=True,
                            factor=False)
    return K[1].to_tensor().numpy()


def k2x(mu, Sig, Wb):
    m = torch.tensor(mu, dtype=DT); C = torch.tensor(Sig, dtype=DT)
    for W in Wb:
        mz = W @ m; Cz = W @ C @ W.T
        m, C, _ = GR.relu_moments(mz, 0.5 * (Cz + Cz.T))
    return m.numpy()


A2u = np.full((F, B, n), np.nan); A3u = np.full((F, B, n), np.nan); K2u = np.full((F, B, n), np.nan)
t0 = time.time(); fails2 = fails3 = 0
for f in range(F // 2, F):
    Wf = [W.astype(np.float64) for W in cn.front_weights(tag, n, c, f)]
    H0 = fwd(cn.directions(n, Np, tag, 21, f, 0).astype(np.float64), Wf)
    mu = Er * H0.mean(0); m2 = n * (H0.T @ H0) / Np
    assert np.allclose(mu, D['MU'][f]) and np.allclose(m2, D['S2'][f]), 'split-0 moments differ from the saved run'
    s = math.sqrt(np.trace(m2) / n)
    if s == 0:
        A2u[f] = A3u[f] = K2u[f] = 0.0
        continue
    mu1, m21 = mu / s, m2 / s ** 2
    Sig1 = m21 - np.outer(mu1, mu1)
    if a.arc3:
        H1 = H0 / s
        m3 = Er3 * np.einsum('si,sj,sk->ijk', H1, H1, H1) / Np
        k3 = (m3 - np.einsum('ij,k->ijk', m21, mu1) - np.einsum('ik,j->ijk', m21, mu1) - np.einsum('jk,i->ijk', m21, mu1)
              + 2 * np.einsum('i,j,k->ijk', mu1, mu1, mu1))
    for b in range(B):
        K2u[f, b] = s * k2x(mu1, Sig1, backs_t[b])
        try:
            A2u[f, b] = s * arc({1: torch.tensor(mu1, dtype=DT), 2: torch.tensor(Sig1, dtype=DT)}, backs_t[b], 2)
        except Exception:
            fails2 += 1
        if a.arc3:
            try:
                A3u[f, b] = s * arc({1: torch.tensor(mu1, dtype=DT), 2: torch.tensor(Sig1, dtype=DT),
                                     3: torch.tensor(k3, dtype=DT)}, backs_t[b], 3)
            except Exception:
                fails3 += 1
    if f % 32 == 0:
        print(f'front {f} t={time.time() - t0:.0f}s fails {fails2}/{fails3}', flush=True)
np.savez_compressed(os.path.join(HERE, '..', 'runs', name + '_arcunit.npz'), A2u=A2u, A3u=A3u, K2u=K2u)
print(json.dumps(dict(name=name, secs=round(time.time() - t0), fails2=fails2, fails3=fails3,
                      finished=time.strftime('%H:%M:%SZ', time.gmtime()))), flush=True)
