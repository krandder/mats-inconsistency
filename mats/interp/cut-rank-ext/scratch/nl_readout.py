"""NC1/NC2 (PREREG.md): linear against nonlinear readouts of the same front numbers, and bits on the mean.

usage: nl_readout.py n c d F B Np [--arc3] [--name NAME]
Nets: the cut-rank lane's ensemble (work/cut-rank/lib/crnet.py, same cfg_tag, front_weights, back_weights).
Inputs: unit directions from new streams (tag, 21, f, split), radius integrated exactly.
Split 0 -> front numbers (mu, Sigma, kappa3); splits 1, 2 -> two independent estimates of M(f, b) and E a^2.
Estimators: MF = g_b(mu); K2X = exact-moment Gaussian closure through the back; ARC2/ARC3 = ARC's kprop
(kprop_harmonic, kind SIMPLE) from the cut cumulants, test fronts only; MFq = g_b(quantised mu) for q = 1..12 bits per
coordinate (NC2). Linear readouts and SVD tails are computed in nl_analyze.py from the saved arrays.
Run under the arc-noisy venv (torch), nice -n 19 ionice -c3.
"""
import argparse, math, os, sys, time, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/agent/workspaces/mats/work/cut-rank/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/ext/mlp_cumulant_propagation/src')
import crnet as cn
import gauss_relu as GR
from mlp_kprop import kprop_harmonic as KH
from mlp_kprop.wick import WICK_COEF_D

torch.set_num_threads(1)
DT = torch.float64
ap = argparse.ArgumentParser()
for k in ('n', 'c', 'd', 'F', 'B', 'Np'):
    ap.add_argument(k, type=int)
ap.add_argument('--arc3', action='store_true')
ap.add_argument('--name', default='')
a = ap.parse_args()
n, c, d, F, B, Np = a.n, a.c, a.d, a.F, a.B, a.Np
name = a.name or f'nl_n{n}_c{c}_d{d}'
tag = cn.cfg_tag(n, c, d)
Er = cn.mean_radius(n)                                  # E|x|
Er3 = 2 ** 1.5 * math.exp(math.lgamma((n + 3) / 2) - math.lgamma(n / 2))   # E|x|^3
QS = list(range(1, 13))
t0 = time.time()
Fte = list(range(F // 2, F))                             # test fronts (ARC only on these)

backs = [[W.astype(np.float64) for W in cn.back_weights(tag, n, d, b)] for b in range(B)]
backs_t = [[torch.tensor(W, dtype=DT) for W in Wb] for Wb in backs]


def fwd(H, Ws):
    for W in Ws:
        H = H @ W.T
        np.maximum(H, 0, out=H)
    return H


def k2x(mu, Sig, Wb):
    m = torch.tensor(mu, dtype=DT); C = torch.tensor(Sig, dtype=DT)
    for W in Wb:
        mz = W @ m; Cz = W @ C @ W.T
        Cz = 0.5 * (Cz + Cz.T)
        m, C, _ = GR.relu_moments(mz, Cz)
    return m.numpy()


def arc(K_in, Wb, kmax):
    K = KH.coerce_input(K_in, k_max=kmax, kind=KH.SIMPLE)
    for W in Wb:
        K = KH.linear_kprop(K, W, k_max=kmax, set_metric=2.0, bias=None)
        K = KH.nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D['relu'], k_max=kmax, kind=KH.SIMPLE, use_pK=True,
                            factor=False)
    return K[1].to_tensor().numpy()


def quantise(mu, s_ref, q):
    """B = n q + 16 bits: norm log-uniform (16 bits) on [1e-6, 1e3] s_ref, direction coordinates uniform q bits."""
    r = np.linalg.norm(mu)
    if r == 0:
        return mu * 0
    lo, hi = math.log(1e-6 * s_ref), math.log(1e3 * s_ref)
    t = np.clip((math.log(r) - lo) / (hi - lo), 0, 1)
    rq = math.exp(lo + (np.round(t * (2 ** 16 - 1)) / (2 ** 16 - 1)) * (hi - lo))
    u = np.clip(mu / r, 0, 1)
    L = 2 ** q - 1
    uq = np.round(u * L) / L
    nq = np.linalg.norm(uq)
    if nq == 0:
        uq = np.zeros(n); uq[np.argmax(u)] = 1.0; nq = 1.0
    return rq * uq / nq


MB = np.zeros((F, B, n)); MC = np.zeros((F, B, n)); E2 = np.zeros((F, B, n))
MF = np.zeros((F, B, n)); K2 = np.zeros((F, B, n)); MFq = np.zeros((len(QS), F, B, n))
A2 = np.full((F, B, n), np.nan); A3 = np.full((F, B, n), np.nan)
MU = np.zeros((F, n)); S2 = np.zeros((F, n, n)); SIG = np.zeros((F, n, n))
t_arc = 0.0; arc_fail = 0; arc3_fail = 0
for f in range(F):
    Wf = [W.astype(np.float64) for W in cn.front_weights(tag, n, c, f)]
    H0 = fwd(cn.directions(n, Np, tag, 21, f, 0).astype(np.float64), Wf)
    mu = Er * H0.mean(0)
    m2 = n * (H0.T @ H0) / Np                               # E h h^T
    Sig = m2 - np.outer(mu, mu)
    MU[f] = mu; S2[f] = m2; SIG[f] = Sig
    if a.arc3 and f in Fte:
        m3 = Er3 * np.einsum('si,sj,sk->ijk', H0, H0, H0) / Np
        k3 = (m3 - np.einsum('ij,k->ijk', m2, mu) - np.einsum('ik,j->ijk', m2, mu) - np.einsum('jk,i->ijk', m2, mu)
              + 2 * np.einsum('i,j,k->ijk', mu, mu, mu))
    H1 = fwd(cn.directions(n, Np, tag, 21, f, 1).astype(np.float64), Wf)
    H2 = fwd(cn.directions(n, Np, tag, 21, f, 2).astype(np.float64), Wf)
    s_ref = math.sqrt(max(np.trace(m2) / n, 1e-300))
    for b in range(B):
        A1 = fwd(H1, backs[b]); A2_ = fwd(H2, backs[b])
        MB[f, b] = Er * A1.mean(0); MC[f, b] = Er * A2_.mean(0)
        E2[f, b] = n * 0.5 * ((A1 ** 2).mean(0) + (A2_ ** 2).mean(0))
        MF[f, b] = fwd(mu[None, :].copy(), backs[b])[0]
        for iq, q in enumerate(QS):
            MFq[iq, f, b] = fwd(quantise(mu, s_ref, q)[None, :], backs[b])[0]
        if np.all(mu == 0):
            K2[f, b] = 0.0
        else:
            K2[f, b] = k2x(mu, Sig, backs_t[b])
        if f in Fte:
            ta = time.time()
            if np.all(mu == 0):
                A2[f, b] = 0.0
                if a.arc3:
                    A3[f, b] = 0.0
            else:
                try:
                    A2[f, b] = arc({1: torch.tensor(mu, dtype=DT), 2: torch.tensor(Sig, dtype=DT)}, backs_t[b], 2)
                except Exception as e:
                    arc_fail += 1
                if a.arc3:
                    try:
                        A3[f, b] = arc({1: torch.tensor(mu, dtype=DT), 2: torch.tensor(Sig, dtype=DT),
                                        3: torch.tensor(k3, dtype=DT)}, backs_t[b], 3)
                    except Exception as e:
                        arc3_fail += 1
            t_arc += time.time() - ta
    if f % 32 == 0:
        print(f'front {f}/{F} t={time.time() - t0:.0f}s (arc {t_arc:.0f}s, fails {arc_fail}/{arc3_fail})', flush=True)

out = os.path.join(HERE, '..', 'runs', name + '.npz')
np.savez_compressed(out, MB=MB, MC=MC, E2=E2, MF=MF, K2=K2, MFq=MFq, A2=A2, A3=A3, MU=MU, S2=S2, SIG=SIG,
                    QS=np.array(QS), n=n, c=c, d=d, F=F, B=B, Np=Np)
summ = dict(name=name, n=n, c=c, d=d, F=F, B=B, Np=Np, arc3=a.arc3, arc_fail=arc_fail, arc3_fail=arc3_fail,
            secs=round(time.time() - t0), secs_arc=round(t_arc), finished=time.strftime('%H:%M:%SZ', time.gmtime()))
with open(os.path.join(HERE, '..', 'logs', 'runs.jsonl'), 'a') as fh:
    fh.write(json.dumps(summ) + '\n')
print(json.dumps(summ), flush=True)
