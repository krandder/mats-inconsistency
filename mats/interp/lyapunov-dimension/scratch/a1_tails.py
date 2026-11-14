"""A1: linear-response (pathwise) tails of block families at a cut, from saved M = E[J^T J] and Sigma = Cov(z_c).
For a block of linear functionals B (n x k) of the state, carried exactly, and the rest scrambled (replaced by an
independent copy with the same second moments), the pathwise output error energy at first order is
  2 tr(M Schur_B) / (2 tr(M Sigma))  with Schur_B = Sigma - Sigma B (B^T Sigma B)^-1 B^T Sigma
(the residual covariance after linear regression on the block). Families (radial/mean direction u removed by P,
because the polar families carry the scale exactly):
  PC  : B = top-k eigenvectors of P Sigma P                      (PB_k of the nongauss lane)
  S   : B = R^-1 V_k, V_k = top-k eigenvectors of S = R M R, R = (P Sigma P)^1/2   (the optimum: tail = sum_{j>k} ev_j(S))
  W   : B = orth(R V_k)                                           (PBW_k of the nongauss lane)
  SEN : B = top-k eigenvectors of P M P                           (PBS_k)
  A   : B = orth[top-k PCs, top-k eigenvectors of P M P]          (PBA_k, dim <= 2k)
Also: tails of Sigma alone (state PCs), of M alone, single-path average.
python a1_tails.py runs/n2mat_256_64_c32_s113_sig0.001.npz [...]"""
import sys, json, math
import numpy as np

KS = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160]


def sqrtm_psd(C):
    ev, V = np.linalg.eigh(0.5 * (C + C.T))
    ev = np.clip(ev, 0, None)
    return (V * np.sqrt(ev)) @ V.T, ev, V


def schur_tail(B, Sig, M, denom):
    if B.shape[1] == 0:
        return 1.0
    Q, _ = np.linalg.qr(B)
    G = Q.T @ Sig @ Q
    # pseudo-inverse for safety
    Gi = np.linalg.pinv(G, rcond=1e-12)
    SB = Sig @ Q
    Sch = Sig - SB @ Gi @ SB.T
    return float(np.trace(M @ Sch) / denom)


def kstar(tails_by_k, level):
    for k, t in tails_by_k:
        if t <= level:
            return k
    return None


out = {}
for fn in sys.argv[1:]:
    D = np.load(fn)
    Sig = D['Sigma']; mu = D['mu']
    n = Sig.shape[0]
    u = mu / np.linalg.norm(mu)
    P = np.eye(n) - np.outer(u, u)
    SigP = P @ Sig @ P
    R, evR, VR = sqrtm_psd(SigP)
    evC = np.sort(np.clip(np.linalg.eigvalsh(SigP), 0, None))[::-1]
    evc, Vc = np.linalg.eigh(SigP); Vc = Vc[:, ::-1]
    res = {}
    for key in [k for k in D.files if k.startswith('M')]:
        Lp = int(key[1:])
        M = D[key]
        MP = P @ M @ P
        S = R @ M @ R
        evS, VS = np.linalg.eigh(0.5 * (S + S.T)); evS = np.clip(evS[::-1], 0, None); VS = VS[:, ::-1]
        evM, VM = np.linalg.eigh(0.5 * (MP + MP.T)); evM = np.clip(evM[::-1], 0, None); VM = VM[:, ::-1]
        denom = float(np.trace(M @ SigP))
        # optimal S family: tail = sum_{j>k} evS_j / tr S
        trS = evS.sum()
        fam = {'S': [], 'PC': [], 'W': [], 'SEN': [], 'A': [], 'SIGMA_ONLY': [], 'M_ONLY': []}
        for k in KS:
            if k > n:
                continue
            fam['S'].append((k, float(evS[k:].sum() / trS)))
            fam['PC'].append((k, schur_tail(Vc[:, :k], SigP, M, denom)))
            fam['W'].append((k, schur_tail(R @ VS[:, :k], SigP, M, denom)))
            fam['SEN'].append((k, schur_tail(VM[:, :k], SigP, M, denom)))
            if k > 0:
                fam['A'].append((k, schur_tail(np.concatenate([Vc[:, :k], VM[:, :k]], 1), SigP, M, denom)))
            else:
                fam['A'].append((0, 1.0))
            fam['SIGMA_ONLY'].append((k, float(evC[k:].sum() / evC.sum())))
            fam['M_ONLY'].append((k, float(evM[k:].sum() / evM.sum())))
        ks_tab = {f: {lvl: kstar(v, lvl) for lvl in (1e-1, 3e-2, 1e-2, 5e-3, 1e-3, 1e-4)} for f, v in fam.items()}
        # dense k* for the S family (exact from eigenvalues)
        tailS = np.cumsum(evS[::-1])[::-1] / trS
        kS_dense = {lvl: int(np.argmax(np.append(tailS[1:], 0) <= lvl)) + 1 for lvl in (1e-1, 3e-2, 1e-2, 5e-3, 1e-3, 1e-4)}
        res[Lp] = dict(tails=fam, kstar=ks_tab, kS_dense=kS_dense, prS=float(trS ** 2 / (evS ** 2).sum()),
                       prSigP=float(evC.sum() ** 2 / (evC ** 2).sum()))
        print(f"{fn.split('/')[-1]} L' {Lp}: PR(S) {res[Lp]['prS']:.2f} PR(PSigmaP) {res[Lp]['prSigP']:.2f}", flush=True)
        for f, v in fam.items():
            print(f"   {f:10s} tail at k=" + ",".join(str(k) for k, _ in v) + ": "
                  + " ".join(f"{t:.2e}" for _, t in v), flush=True)
        print("   k*(tail <= 1e-1,3e-2,1e-2,5e-3,1e-3,1e-4): " + "; ".join(
            f"{f} " + "/".join(str(ks_tab[f][l]) for l in (1e-1, 3e-2, 1e-2, 5e-3, 1e-3, 1e-4)) for f in fam)
            + f"; S exact " + "/".join(str(kS_dense[l]) for l in (1e-1, 3e-2, 1e-2, 5e-3, 1e-3, 1e-4)), flush=True)
    out[fn] = res
json.dump(out, open('runs/a1_tails.json', 'w'), indent=1)
