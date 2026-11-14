"""Part 1c (PREREG Part 1): age and decay of input information by Lyapunov direction, one net, sigma = 0.
usage: part1c.py n seed [--K 4096] [--M 512] [--k 64] [--out file]
Reference input u0 (unit). Along its trajectory: the tangent maps J_l = diag(z_l(u0) > 0) W_l and the backward Lyapunov
(Gram-Schmidt) basis Q_l at layer l: QR of J_l Q_{l-1} with Q_0 = a random orthonormal n x k frame (k = min(n, 64)
directions; column 1 = most expanding). Two input clouds at each layer l:
- tangent cloud: images J_l...J_1 v of M random unit perturbations v of u0 (infinitesimal input differences);
- finite cloud: h_l(u) for K independent unit inputs u (finite input differences).
Per layer l in {1, 2, 4, ..., n, 2n}: for each cloud C (covariance about its mean for the finite cloud, second moment of
the images for the tangent cloud): share of tr C along Q_l's columns 1..k; radial share along the reference state
h_l(u0)/|h_l(u0)|; participation ratio (tr C)^2 / tr C^2 of the part orthogonal to the reference state ("angular PR");
angular size relative to |h_l|^2 (finite: tr C_ang / E|h|^2; tangent: tr T_ang/ E|T|^2), so their ratio is the
infinitesimal-to-finite sensitivity ratio. Seeds [7021, seed, n]."""
import argparse, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--K', type=int, default=4096); ap.add_argument('--M', type=int, default=512)
ap.add_argument('--k', type=int, default=64); ap.add_argument('--R', type=int, default=16)
ap.add_argument('--Mref', type=int, default=64); ap.add_argument('--out')
a = ap.parse_args()
n, seed = a.n, a.seed
k = min(n, a.k)
t0 = time.time()
Lmax = 2 * n
net = ib.Net(seed, n, Lmax)
rng = np.random.default_rng([7021, seed, n])
u0 = ib.sphere(rng.standard_normal((1, n)))
Ufin = ib.sphere(rng.standard_normal((a.K, n)))
Vt = ib.sphere(rng.standard_normal((a.M, n)))
Vt = Vt - (Vt @ u0.T) * u0          # perturbations orthogonal to u0 (pure direction changes of the input)
Q = np.linalg.qr(rng.standard_normal((n, k)))[0]
# extra references for an input-averaged Gamma (mean-square angle growth of infinitesimal input changes)
Ur = ib.sphere(rng.standard_normal((a.R, n)))
Tr = ib.sphere(rng.standard_normal((a.R, a.Mref, n)))
Tr = Tr - np.einsum('rmn,rn->rm', Tr, Ur)[:, :, None] * Ur[:, None, :]
hr = Ur.copy(); logSr = np.zeros(a.R)
h0 = u0.copy(); hf = Ufin.copy(); T = Vt.copy()
marks = sorted({2 ** i for i in range(0, 20) if 2 ** i <= Lmax} | {Lmax})
res = dict(n=n, seed=seed, K=a.K, M=a.M, k=k, layers={})
logS = 0.0
for l in range(1, Lmax + 1):
    W = net.W[l - 1]
    z0 = h0 @ W.T
    Dg = (z0[0] > 0).astype(float)
    h0 = np.maximum(z0, 0.0)
    hf = np.maximum(hf @ W.T, 0.0)
    T = (T @ W.T) * Dg[None, :]
    Q = Dg[:, None] * (W @ Q)
    Q, R = np.linalg.qr(Q)
    sg = np.sign(np.diag(R)); sg[sg == 0] = 1
    Q = Q * sg[None, :]
    zr = hr @ W.T
    Dr = (zr > 0).astype(float)
    hr = np.maximum(zr, 0.0)
    Tr = np.einsum('rmn,kn->rmk', Tr, W) * Dr[:, None, :]
    sr = np.sqrt(np.mean(np.sum(Tr * Tr, 2), 1))
    okr = sr > 0
    Tr[okr] /= sr[okr][:, None, None]
    logSr[okr] += np.log(sr[okr])
    nrm0 = float(np.linalg.norm(h0))
    if nrm0 == 0:
        res['dead_at'] = l
        break
    # rescale the tangent images to avoid under/overflow (shares and ratios are scale free)
    s = float(np.sqrt(np.mean(np.sum(T * T, 1))))
    if s > 0:
        T /= s
        logS += math.log(s)
    if l in marks:
        e = (h0 / nrm0)[0]
        Cf = np.cov(hf.T, bias=False)
        Ct = T.T @ T / len(T)
        out = {}
        for name, C, Eh2 in (('finite', Cf, float(np.mean(np.sum(hf * hf, 1)))), ('tangent', Ct, float(np.trace(Ct)))):
            tr = float(np.trace(C))
            shares = [float(Q[:, i] @ C @ Q[:, i]) / tr for i in range(k)]
            P_ang = np.eye(n) - np.outer(e, e)
            Ca = P_ang @ C @ P_ang
            tra = float(np.trace(Ca))
            out[name] = dict(shares_Q=shares, radial_share=float(e @ C @ e) / tr, ang_PR=tra ** 2 / float(np.sum(Ca * Ca)),
                             ang_rel=tra / Eh2)
        # Gamma_l: mean-square angle at layer l per unit input angle (infinitesimal), in units of the reference state
        Ta = T - (T @ e)[:, None] * e[None, :]
        Gam = math.exp(2 * logS) * float(np.mean(np.sum(Ta * Ta, 1))) / nrm0 ** 2
        out['Gamma'] = Gam
        nr = np.linalg.norm(hr, axis=1)
        live = nr > 0
        er = hr[live] / nr[live][:, None]
        Ta_r = Tr[live] - np.einsum('rmn,rn->rm', Tr[live], er)[:, :, None] * er[:, None, :]
        Gr = np.exp(2 * logSr[live]) * np.mean(np.sum(Ta_r * Ta_r, 2), 1) / nr[live] ** 2
        out['Gamma_refs'] = Gr.tolist()
        out['Gamma_mean'] = float(Gr.mean()) if len(Gr) else None
        out['sens_ratio_mean'] = float(Gr.mean()) / max(out['finite']['ang_rel'], 1e-300) if len(Gr) else None
        out['sens_ratio'] = Gam / max(out['finite']['ang_rel'], 1e-300)
        res['layers'][str(l)] = out
        f, t = out['finite'], out['tangent']
        print(f'l={l}: finite ang_rel {f["ang_rel"]:.3e} PR {f["ang_PR"]:.1f} radial {f["radial_share"]:.3f} Q1..4 '
              f'{np.round(f["shares_Q"][:4], 3).tolist()} | tangent ang_rel {t["ang_rel"]:.3e} PR {t["ang_PR"]:.1f} '
              f'Q1..4 {np.round(t["shares_Q"][:4], 3).tolist()} | Gamma {out["Gamma"]:.3f} mean {out["Gamma_mean"]:.3f} ratio {out["sens_ratio_mean"]:.2f} | {time.time() - t0:.0f}s', flush=True)
res['sec'] = time.time() - t0
if a.out:
    json.dump(res, open(a.out, 'w'))
print('done', res['sec'])
