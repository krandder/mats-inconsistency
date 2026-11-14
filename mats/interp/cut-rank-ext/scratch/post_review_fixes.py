"""Post-review fixes (Amendment 2; final GPT-6 review findings 9-11). usage: post_review_fixes.py n c d MODE
MODE nc2g: NC2 with a GLOBAL norm codebook (16 bits log-uniform on [1e-12, 1e4] absolute, clipped; overflows counted),
           so the decoded mean depends on the transmitted bits only (the registered code used a front-dependent s_ref).
           Writes runs/nl_n{n}_c{c}_d{d}_nc2g.npz (MFqg, all fronts).
MODE arc3k4: ARC3 on unit-scaled cut cumulants WITH the fourth cumulant (Wu et al.'s basic K = 3 also needs the scalar
           harmonic component of cumulant 4). Writes runs/..._arc3k4.npz (A3k4, test fronts).
MODE conv: moment-sample convergence of ARC2 and K2X: 64 test fronts, cut moments from 8,192 (the registered split 0)
           against 65,536 fresh directions. Writes runs/..._conv.npz.
"""
import math, os, sys, time, json
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
n, c, d, MODE = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
HERE = os.path.dirname(os.path.abspath(__file__))
name = f'nl_n{n}_c{c}_d{d}'
D = np.load(os.path.join(HERE, '..', 'runs', name + '.npz'))
F, B, Np = int(D['F']), int(D['B']), int(D['Np'])
tag = cn.cfg_tag(n, c, d)
ER = {k: (n if k == 2 else n * (n + 2) if k == 4 else
          2 ** (k / 2) * math.exp(math.lgamma((n + k) / 2) - math.lgamma(n / 2))) for k in (1, 2, 3, 4)}
backs = [[W.astype(np.float64) for W in cn.back_weights(tag, n, d, b)] for b in range(B)]
backs_t = [[torch.tensor(W, dtype=DT) for W in Wb] for Wb in backs]
t0 = time.time()


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


def raw_moments(H, kmax):
    """m_k = E|x|^k mean_s(u^{(x) k}) for the cut states H (rows) of unit input directions."""
    N = len(H)
    out = {1: ER[1] * H.mean(0), 2: ER[2] * H.T @ H / N}
    if kmax >= 3:
        out[3] = ER[3] * np.einsum('si,sj,sk->ijk', H, H, H) / N
    if kmax >= 4:
        out[4] = ER[4] * np.einsum('si,sj,sk,sl->ijkl', H, H, H, H) / N
    return out


def cumulants4(m):
    m1, m2, m3, m4 = m[1], m[2], m[3], m[4]
    k3 = (m3 - np.einsum('ij,k->ijk', m2, m1) - np.einsum('ik,j->ijk', m2, m1) - np.einsum('jk,i->ijk', m2, m1)
          + 2 * np.einsum('i,j,k->ijk', m1, m1, m1))
    k4 = (m4
          - np.einsum('ijk,l->ijkl', m3, m1) - np.einsum('ijl,k->ijkl', m3, m1) - np.einsum('ikl,j->ijkl', m3, m1)
          - np.einsum('jkl,i->ijkl', m3, m1)
          - np.einsum('ij,kl->ijkl', m2, m2) - np.einsum('ik,jl->ijkl', m2, m2) - np.einsum('il,jk->ijkl', m2, m2)
          + 2 * (np.einsum('ij,k,l->ijkl', m2, m1, m1) + np.einsum('ik,j,l->ijkl', m2, m1, m1)
                 + np.einsum('il,j,k->ijkl', m2, m1, m1) + np.einsum('jk,i,l->ijkl', m2, m1, m1)
                 + np.einsum('jl,i,k->ijkl', m2, m1, m1) + np.einsum('kl,i,j->ijkl', m2, m1, m1))
          - 6 * np.einsum('i,j,k,l->ijkl', m1, m1, m1, m1))
    return k3, k4


out = {}
if MODE == 'nc2g':
    QS = list(range(1, 13)); lo, hi = math.log(1e-12), math.log(1e4)
    MFqg = np.zeros((len(QS), F, B, n)); overflow = 0
    for f in range(F):
        mu = D['MU'][f]; r = np.linalg.norm(mu)
        for iq, q in enumerate(QS):
            if r == 0:
                continue
            tq = (math.log(r) - lo) / (hi - lo)
            overflow += int(tq < 0 or tq > 1) if iq == 0 else 0
            tq = min(max(tq, 0.0), 1.0)
            rq = math.exp(lo + (round(tq * (2 ** 16 - 1)) / (2 ** 16 - 1)) * (hi - lo))
            u = np.clip(mu / r, 0, 1); L = 2 ** q - 1
            uq = np.round(u * L) / L; nq = np.linalg.norm(uq)
            if nq == 0:
                uq = np.zeros(n); uq[np.argmax(u)] = 1.0; nq = 1.0
            muq = rq * uq / nq
            for b in range(B):
                MFqg[iq, f, b] = fwd(muq[None, :].copy(), backs[b])[0]
    out = dict(MFqg=MFqg, overflow=overflow)
elif MODE == 'arc3k4':
    A3k4 = np.full((F, B, n), np.nan); fails = 0
    for f in range(F // 2, F):
        Wf = [W.astype(np.float64) for W in cn.front_weights(tag, n, c, f)]
        H0 = fwd(cn.directions(n, Np, tag, 21, f, 0).astype(np.float64), Wf)
        s = math.sqrt(np.trace(ER[2] * H0.T @ H0 / Np) / n)
        if s == 0:
            A3k4[f] = 0.0
            continue
        m = raw_moments(H0 / s, 4)
        k3, k4 = cumulants4(m)
        Sig = m[2] - np.outer(m[1], m[1])
        K_in = {1: torch.tensor(m[1], dtype=DT), 2: torch.tensor(Sig, dtype=DT), 3: torch.tensor(k3, dtype=DT),
                4: torch.tensor(k4, dtype=DT)}
        for b in range(B):
            try:
                A3k4[f, b] = s * arc({k: v.clone() for k, v in K_in.items()}, backs_t[b], 3)
            except Exception:
                fails += 1
        if f % 32 == 0:
            print(f'front {f} t={time.time() - t0:.0f}s fails {fails}', flush=True)
    out = dict(A3k4=A3k4, fails=fails)
elif MODE == 'conv':
    fr = list(range(F // 2, F // 2 + 64))
    res = {k: np.full((len(fr), B, n), np.nan) for k in ('A2_8k', 'A2_64k', 'K2_8k', 'K2_64k')}
    for i, f in enumerate(fr):
        Wf = [W.astype(np.float64) for W in cn.front_weights(tag, n, c, f)]
        for lab, U in (('8k', cn.directions(n, Np, tag, 21, f, 0)), ('64k', cn.directions(n, 65536, tag, 41, f))):
            H = fwd(U.astype(np.float64), Wf)
            s = math.sqrt(np.trace(ER[2] * H.T @ H / len(H)) / n)
            if s == 0:
                for k in ('A2_', 'K2_'):
                    res[k + lab][i] = 0.0
                continue
            m = raw_moments(H / s, 2)
            Sig = m[2] - np.outer(m[1], m[1])
            for b in range(B):
                res['A2_' + lab][i, b] = s * arc({1: torch.tensor(m[1], dtype=DT), 2: torch.tensor(Sig, dtype=DT)},
                                                 backs_t[b], 2)
                res['K2_' + lab][i, b] = s * k2x(m[1], Sig, backs_t[b])
    out = dict(fronts=np.array(fr), **res)
np.savez_compressed(os.path.join(HERE, '..', 'runs', f'{name}_{MODE}.npz'), **out)
print(json.dumps(dict(name=name, mode=MODE, secs=round(time.time() - t0),
                      finished=time.strftime('%H:%M:%SZ', time.gmtime()))), flush=True)
