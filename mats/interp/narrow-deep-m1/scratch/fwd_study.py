"""Forward-part study (PREREG P1, P4): how well each sample-free forward description carries the law to layer c.
usage: python fwd_study.py SEED SIGMA TRUTHJSON OUTJSON [TAUS=1e-3,1e-4] [CUTS_MEET=2,4,6,8]
Methods: KP-K (ARC kprop: K2 simple/augmented, K3 factorized simple/augmented, K4 factorized simple), K2X (exact-moment
Gaussian closure), GHM(tau).
Scores at c = 1..8: nMSE of E[a_c] vs the truth's layer moments. At the meet cuts: e_c = |E_{law_c}[V_c] - E a_L|^2 /
trVar(a_L), with the true noisy downstream (oracle evaluation, 2^20 draws of the method's law at c):
- K2X and KP-K: the Gaussian with the method's mean and covariance of h_c ("gauss" meet);
- KP-K, K >= 3: also the degree-K polynomial meet: V_c fitted by least squares on monomials of the standardised state
  (true law, 2^19 paths; oracle fit), integrated against KP-K's raw moments of order <= K ("poly" meet). Degree 2 with
  KP-K's mean and covariance is reported for every KP variant.
- GHM: draws from the mixture.
FLOPs: kprop by ARC's counter; K2X and GHM by flopscope counts.
"""
import sys, os, json, math, time, itertools
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import nd, k2, ghm
import kprop_noisy as KN
from mlp_kprop import kprop_harmonic as KH
from mlp_kprop.wick import WICK_COEF_D
from mlp_kprop.flop_utils import NamedFlopCounter, flop_name

torch.set_num_threads(1)
SEED, SIG, TJ, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
TAUS = [float(t) for t in sys.argv[5].split(',')] if len(sys.argv) > 5 else [1e-3, 1e-4]
CM = [int(c) for c in sys.argv[6].split(',')] if len(sys.argv) > 6 else [2, 4, 6, 8]
CMAX = 8
DT = nd.DT
T = json.load(open(TJ))
net = nd.Net('plain', SEED, 8, T['L'], SIG, r=T['r'])
n, L = net.n, net.L
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']
lay = {int(k): v for k, v in T['layers'].items()}
g = torch.Generator().manual_seed(51_000_001 + SEED)
PM = 1 << 18          # meet draws (resolution ~1/PM in nMSE; forward errors are far above it)
res = dict(seed=SEED, sigma=SIG, truth_paths=T['paths'], flops_path=net.flops_path(), methods={})
t0 = time.time()


def downstream_mean(h, c):
    """oracle: push states h (after noise of layer c) through the true noisy layers c+1..L; mean of a_L."""
    for l in range(c, L):
        a = torch.relu(h @ net.W[l].T)
        h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if (l < L - 1 and SIG > 0) else a
    return h.mean(0)


def meet_gauss(m, C, c):
    ev, U = torch.linalg.eigh(0.5 * (C + C.T))
    Lh = U * ev.clamp_min(0).sqrt()
    out = torch.zeros(n, dtype=DT)
    for b in range(4):
        z = torch.randn(PM // 4, n, generator=g, dtype=DT)
        out += downstream_mean(m + z @ Lh.T, c) / 4
    return float((out - truth).pow(2).sum() / trv)


def meet_mix(w, mu, C, c):
    out = torch.zeros(n, dtype=DT)
    for b in range(4):
        hs = ghm.sample_mixture(w, mu, C, PM // 4, g)
        out += downstream_mean(hs, c) / 4
    return float((out - truth).pow(2).sum() / trv)


def err_a(m_a, c):
    return float((m_a - torch.tensor(lay[c]['mean_a'], dtype=DT)).pow(2).sum() / lay[c]['trvar_a'])


# ---------------- polynomial meet: oracle fit of V_c on the true law (standardised coordinates)
def true_states(c, P):
    h = torch.randn(P, n, generator=g, dtype=DT)
    for l in range(c):
        a = torch.relu(h @ net.W[l].T)
        h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if SIG > 0 else a
    return h


def monos(K):
    out = [()]
    for d in range(1, K + 1):
        out += list(itertools.combinations_with_replacement(range(n), d))
    return out


def feats(y, ms):
    cols = [torch.ones(len(y), dtype=DT)]
    for mm in ms[1:]:
        v = y[:, mm[0]].clone()
        for j in mm[1:]:
            v = v * y[:, j]
        cols.append(v)
    return torch.stack(cols, 1)


def poly_fit_all(c, P=1 << 19, CH=1 << 15):
    """least-squares fits of V_c on monomials of degree <= 2, 3, 4 of the standardised state, from one set of paths."""
    mh = torch.tensor(lay[c]['mean_h'], dtype=DT); Ch = torch.tensor(lay[c]['cov_h'], dtype=DT)
    ev, U = torch.linalg.eigh(0.5 * (Ch + Ch.T)); A = U @ torch.diag(ev.clamp_min(1e-300 * ev.max()).rsqrt()) @ U.T
    ms = monos(4); F = len(ms)
    XtX = torch.zeros(F, F, dtype=DT); XtY = torch.zeros(F, n, dtype=DT)
    for b in range(P // CH):
        h = true_states(c, CH)
        y = (h - mh) @ A.T
        X = feats(y, ms)
        o = downstream_mean_batch(h, c)
        XtX += X.T @ X; XtY += X.T @ o
    out = {}
    for K in (2, 3, 4):
        FK = len(monos(K))
        M = XtX[:FK, :FK]
        lam = 1e-10 * M.diagonal().mean()
        beta = torch.linalg.solve(M + lam * torch.eye(FK, dtype=DT), XtY[:FK])
        out[K] = dict(mh=mh, A=A, ms=monos(K), beta=beta)
    return out


def downstream_mean_batch(h, c):
    for l in range(c, L):
        a = torch.relu(h @ net.W[l].T)
        h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if (l < L - 1 and SIG > 0) else a
    return h


def raw_moments_from_cumulants(k1, k2, k3, k4, ms):
    """E[prod y_idx] for each monomial from cumulants (orders up to 4; missing cumulants are zero)."""
    out = []
    for mm in ms:
        d = len(mm)
        if d == 0:
            out.append(1.0); continue
        # sum over set partitions of the index multiset positions
        idx = list(mm)
        tot = 0.0
        for part in set_partitions(list(range(d))):
            prod = 1.0
            for blk in part:
                kk = [idx[p] for p in blk]
                if len(kk) == 1: v = k1[kk[0]]
                elif len(kk) == 2: v = k2[kk[0], kk[1]]
                elif len(kk) == 3: v = k3[kk[0], kk[1], kk[2]] if k3 is not None else 0.0
                elif len(kk) == 4: v = k4[kk[0], kk[1], kk[2], kk[3]] if k4 is not None else 0.0
                else: v = 0.0
                prod = prod * float(v)
            tot += prod
        out.append(tot)
    return torch.tensor(out, dtype=DT)


def set_partitions(s):
    if not s:
        yield []
        return
    first, rest = s[0], s[1:]
    for p in set_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def transform_cumulants(K, mh, A):
    """cumulants of h (dense tensors) -> cumulants of y = A (h - mh)."""
    k1 = A @ (K[1] - mh)
    k2 = A @ K[2] @ A.T
    k3 = torch.einsum('ia,jb,kc,abc->ijk', A, A, A, K[3]) if K.get(3) is not None else None
    k4 = torch.einsum('ia,jb,kc,ld,abcd->ijkl', A, A, A, A, K[4]) if K.get(4) is not None else None
    return k1, k2, k3, k4


# ---------------- kprop with the tower kept (dense cumulants of h_c after noise)
@torch.no_grad()
def kprop_tower(k_max, factor, kind):
    kind_e = {'simple': KH.SIMPLE, 'augment': KH.AUGMENT}[kind]
    mlp = KN.arc_mlp(net, CMAX)
    counter = NamedFlopCounter(); counter.__enter__()
    towers = {}; flops_at = {}
    try:
        with flop_name('kprop'):
            K = KH.coerce_input({1: torch.zeros(n, dtype=DT), 2: torch.eye(n, dtype=DT)}, k_max=k_max, kind=kind_e)
            for l in range(CMAX):
                K = KH.linear_kprop(K, mlp.Ws[l].weight, k_max=k_max, set_metric=mlp.init_scale[l], bias=None)
                K = KH.nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D['relu'], k_max=k_max, kind=kind_e, use_pK=True, factor=factor)
                m_a = K[1].to_tensor().clone()
                if SIG > 0:
                    K = KN.add_noise_to_K2(K, float(net.s[l]) ** 2)
                dense = {}
                for o in (1, 2, 3, 4):
                    if o in K and K[o] is not None:
                        try:
                            dense[o] = K[o].to_tensor().clone()
                        except Exception:
                            dense[o] = None
                towers[l + 1] = (m_a, dense)
                flops_at[l + 1] = counter.total()
    finally:
        counter.__exit__(None, None, None)
    return towers, flops_at


polys = {}
for c in CM:
    pf = poly_fit_all(c)
    for Kd in (2, 3, 4):
        polys[(c, Kd)] = pf[Kd]
print(f"poly fits done ({time.time()-t0:.0f}s)", flush=True)

KPV = [('KP2', 2, False, 'simple'), ('KP2a', 2, False, 'augment'), ('KP3f', 3, True, 'simple'),
       ('KP3fa', 3, True, 'augment'), ('KP4f', 4, True, 'simple')]
for name, kmax, fac, kind in KPV:
    try:
        towers, fla = kprop_tower(kmax, fac, kind)
    except Exception as e:
        res['methods'][name] = dict(error=str(e)[:300]); print(name, 'failed', e, flush=True); continue
    row = dict(flops_at={c: fla[c] for c in fla}, err_a={}, meet_gauss={}, meet_poly={})
    for c in range(1, CMAX + 1):
        row['err_a'][c] = err_a(towers[c][0], c)
    for c in CM:
        dense = towers[c][1]
        m, C = dense[1], dense[2]
        try:
            row['meet_gauss'][c] = meet_gauss(m, 0.5 * (C + C.T), c)
        except Exception as e:
            row['meet_gauss'][c] = None
        for Kd in sorted(set([2, min(kmax, 4)])):
            P_ = polys[(c, Kd)]
            Kc = {1: dense[1], 2: dense[2], 3: dense.get(3) if Kd >= 3 else None, 4: dense.get(4) if Kd >= 4 else None}
            try:
                k1, k2_, k3, k4 = transform_cumulants(Kc, P_['mh'], P_['A'])
                mom = raw_moments_from_cumulants(k1, k2_, k3, k4, P_['ms'])
                est = mom @ P_['beta']
                row['meet_poly'][f'{c}_deg{Kd}'] = float((est - truth).pow(2).sum() / trv)
            except Exception as e:
                row['meet_poly'][f'{c}_deg{Kd}'] = None
    res['methods'][name] = row
    print(f"{name}: err_a(c=1..8)={[f'{row['err_a'][c]:.2g}' for c in range(1, 9)]} meet_gauss={ {c: (round(v, 6) if v is not None else None) for c, v in row['meet_gauss'].items()} } "
          f"meet_poly={ {k: (f'{v:.3g}' if v is not None else None) for k, v in row['meet_poly'].items()} } flops@8={fla[8]:.3g} ({time.time()-t0:.0f}s)", flush=True)

# K2X: exact-moment Gaussian closure
row = dict(flops_at={}, err_a={}, meet_gauss={})
mu = torch.zeros(n, dtype=DT); C = torch.eye(n, dtype=DT); fl = 0
for l in range(CMAX):
    W = net.W[l]
    m = W @ mu; Cz = W @ C @ W.T
    fl += k2.mmf(n, n, 1) + 2 * k2.mmf(n, n, n)
    mu, C, f = k2.relu_mom(m, Cz, True); fl += f
    row['err_a'][l + 1] = err_a(mu, l + 1)
    if SIG > 0:
        C = C + torch.eye(n, dtype=DT) * float(net.s[l]) ** 2; fl += n
    row['flops_at'][l + 1] = fl
    if (l + 1) in CM:
        row['meet_gauss'][l + 1] = meet_gauss(mu, C, l + 1)
res['methods']['K2X'] = row
print(f"K2X: err_a={[f'{row['err_a'][c]:.2g}' for c in range(1, 9)]} meet={ {c: round(v, 6) for c, v in row['meet_gauss'].items()} } flops@8={fl:.3g}", flush=True)

# GHM
for tau in TAUS:
    t1 = time.time()
    out, ftot = ghm.ghm_forward(net, CMAX, tau, verbose=False)
    row = dict(tau=tau, flops_at={}, ncomp={}, err_a={}, meet={})
    for c in range(1, CMAX + 1):
        w, mu_, C_, fc = out[c]
        row['flops_at'][c] = fc; row['ncomp'][c] = int(len(w))
        # E[a_c]: the GHM mean of h_c is E[a_c] (noise has mean 0)
        row['err_a'][c] = err_a(ghm.mixture_mean(w, mu_), c)
        if c in CM:
            row['meet'][c] = meet_mix(w, mu_, C_, c)
    res['methods'][f'GHM{tau:g}'] = row
    print(f"GHM tau={tau:g}: ncomp={[row['ncomp'][c] for c in range(1, 9)]} err_a={[f'{row['err_a'][c]:.2g}' for c in range(1, 9)]} "
          f"meet={ {c: f'{v:.3g}' for c, v in row['meet'].items()} } flops@c={ {c: f'{row['flops_at'][c]:.3g}' for c in CM} } ({time.time()-t1:.0f}s)", flush=True)
res['sec'] = time.time() - t0
json.dump(res, open(OUT, 'w'), indent=1)
