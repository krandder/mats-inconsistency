"""Simulate the cut matrix for one configuration (cut-rank lane, PREREG.md sections 1-3 and Amendment 1).

usage: python3 scratch/run_cut.py n c Lp F B Np [--S 8] [--control] [--nro 4] [--kmax K] [--name NAME]
Saves runs/NAME.npz and appends a one-line JSON summary to logs/runs.jsonl. Heavy: nice -n 19 ionice -c3.
--control: identical-front control (front 0 repeated F times, independent inputs per row; true rank 1).
States are stored column-wise (n x samples), which is faster for n >= 16 on this machine.
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import crnet as cn

ap = argparse.ArgumentParser()
for k in ('n', 'c', 'Lp', 'F', 'B', 'Np'):
    ap.add_argument(k, type=int)
ap.add_argument('--S', type=int, default=8)
ap.add_argument('--control', action='store_true')
ap.add_argument('--nro', type=int, default=4)
ap.add_argument('--kmax', type=int, default=-1)
ap.add_argument('--name', default='')
ap.add_argument('--nmom', type=int, default=1 << 16)
a = ap.parse_args()
n, c, Lp, F, B, Np, S = a.n, a.c, a.Lp, a.F, a.B, a.Np, a.S
name = a.name or f'cut_n{n}_c{c}_L{Lp}' + ('_ctrl' if a.control else '')
tag = cn.cfg_tag(n, c, Lp)
kmax = a.kmax if a.kmax >= 0 else (3 if n <= 8 else 2 if n <= 16 else 1)
t0 = time.time()
Er = cn.mean_radius(n)
fid = (lambda f: 0) if a.control else (lambda f: f)


def fwdT(X, Ws):  # columns are states
    for W in Ws:
        X = W @ X
        np.maximum(X, 0, out=X)
    return X


# ---- fronts: cut states, columns ordered (front, split, sample) ------------------------------------------------
H = np.empty((n, F * S * Np), dtype=np.float32)
for f in range(F):
    Wf = cn.front_weights(tag, n, c, fid(f))
    for s in range(S):
        u = cn.directions(n, Np, tag, 2, f, s)
        H[:, (f * S + s) * Np:(f * S + s + 1) * Np] = fwdT(np.ascontiguousarray(u.T), Wf)
t_front = time.time() - t0


# ---- moments of the cut state (separate sample), features of order <= kmax --------------------------------------
def moment_features(h, kmax):
    feats = [np.ones(1)]
    h = h.astype(np.float64)
    if kmax >= 1:
        feats.append(h.mean(0))
    if kmax >= 2:
        iu = np.triu_indices(n)
        feats.append((h.T @ h / len(h))[iu])
    if kmax >= 3:
        T3m = ((h[:, :, None] * h[:, None, :]).reshape(len(h), n * n).T @ h / len(h)).reshape(n, n, n)
        idx = np.array([(i, j, k) for i in range(n) for j in range(i, n) for k in range(j, n)])
        feats.append(T3m[idx[:, 0], idx[:, 1], idx[:, 2]])
    return np.concatenate(feats)


mom = np.stack([moment_features(cn.forward(cn.directions(n, a.nmom, tag, 3, f), cn.front_weights(tag, n, c, fid(f))),
                                kmax) for f in range(F)])
t_mom = time.time() - t0 - t_front

# ---- diagnostics: per-path k*, synchronisation, participation ratio ----------------------------------------------
ks01, ks1, gaps = [], [], []
for f in range(8):
    Wf = cn.front_weights(tag, n, c, f)
    zc = cn.forward_pre(cn.directions(n, 8, tag, 4, f), Wf)
    for b in range(8):
        J = cn.jacobian_from_pre(zc[b].astype(np.float64), cn.back_weights(tag, n, Lp, b))
        sv = np.linalg.svd(J, compute_uv=False)
        ks01.append(cn.n_surv(sv, 0.01)); ks1.append(cn.n_surv(sv, 0.1))
        gaps.append(float(sv[1] / sv[0]) if sv[0] > 0 else float('nan'))
angles, gdis, prs = [], [], []
for f in range(8):
    Wf = cn.front_weights(tag, n, c, f)
    zc = cn.forward_pre(cn.directions(n, 256, tag, 5, f), Wf)
    hc = np.maximum(zc, 0).astype(np.float64)
    nh = np.linalg.norm(hc, axis=1)
    hu = hc[nh > 0] / nh[nh > 0, None]
    Cm = np.clip(hu @ hu.T, -1, 1)
    angles.append(float(np.median(np.arccos(Cm[np.triu_indices(len(hu), 1)]))))
    sg = zc > 0
    gdis.append(float(np.median((sg[:, None, :] != sg[None, :, :]).mean(2)[np.triu_indices(len(sg), 1)])))
    prs.append(cn.participation_ratio(H[:, f * S * Np:f * S * Np + Np].T))
t_diag = time.time() - t0 - t_front - t_mom

# ---- backs: ensemble matrix M (F, S, B, n), second moments, fixed-back readouts ----------------------------------
M = np.empty((F, S, B, n), dtype=np.float64)
S2 = np.empty((F, B, n), dtype=np.float64)
nro = min(a.nro, B)
R2 = np.empty((F, S, nro, n, 8), dtype=np.float32)
R3 = np.empty((F, S, nro, 64, 8), dtype=np.float32)
T2 = np.zeros((nro, n, 8)); T3 = np.empty((nro, 64, 8)); V3 = np.empty((nro, 64, n))
chunk = max(1, int((192 << 20) // (S * Np * n * 4)))
for b in range(B):
    Wb = cn.back_weights(tag, n, Lp, b)
    if b < nro:  # thresholds from a pilot of 8 extra fronts (never used as rows)
        pil = np.concatenate([cn.forward(cn.forward(cn.directions(n, Np, tag, 6, f), cn.front_weights(tag, n, c, f)), Wb)
                              for f in range(F, F + 8)]).astype(np.float64)
        for j in range(n):
            pos = pil[:, j][pil[:, j] > 0]
            if len(pos) > 8:
                T2[b, j, 1:] = np.quantile(pos, np.arange(1, 8) / 8)
        v = cn.rng(tag, 7, b).standard_normal((64, n)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        V3[b] = v
        T3[b] = np.quantile(pil @ v.T, np.arange(1, 9) / 9, axis=0).T
    for f0 in range(0, F, chunk):
        f1 = min(F, f0 + chunk)
        A = fwdT(H[:, f0 * S * Np:f1 * S * Np].copy(), Wb).reshape(n, f1 - f0, S, Np)
        M[f0:f1, :, b, :] = A.mean(3, dtype=np.float64).transpose(1, 2, 0)
        S2[f0:f1, b, :] = np.mean(np.square(A), axis=(2, 3), dtype=np.float64).T
        if b < nro:
            t2 = T2[b].astype(np.float32); t3 = T3[b].astype(np.float32); v3 = V3[b].astype(np.float32)
            for i in range(f1 - f0):
                Ai = A[:, i]                                    # (n, S, Np)
                R2[f0 + i, :, b] = (Ai[..., None] > t2[:, None, None, :]).mean(2).transpose(1, 0, 2)
                P = np.einsum('mn,nsk->msk', v3, Ai)            # (64, S, Np)
                R3[f0 + i, :, b] = (P[..., None] > t3[:, None, None, :]).mean(2).transpose(1, 0, 2)
t_back = time.time() - t0 - t_front - t_mom - t_diag

Mbar = M.mean(1)
V = n * S2 - Er ** 2 * Mbar ** 2   # Var_x a = n E_u a^2 - (E|x|)^2 (E_u a)^2
np.savez_compressed(os.path.join(HERE, '..', 'runs', name + '.npz'), M=M * Er, V=V, R2=R2, R3=R3, T2=T2, T3=T3,
                    mom=mom, ks01=np.array(ks01), ks1=np.array(ks1), gaps=np.array(gaps), angles=np.array(angles),
                    gdis=np.array(gdis), prs=np.array(prs))
summ = dict(name=name, n=n, c=c, Lp=Lp, F=F, B=B, Np=Np, S=S, control=a.control, kmax=kmax,
            kstar01=float(np.mean(ks01)), kstar1=float(np.mean(ks1)), s2s1_median=float(np.nanmedian(gaps)),
            angle_median=float(np.median(angles)), gate_dis_median=float(np.median(gdis)), PR=float(np.mean(prs)),
            Vbar=float(V.mean()), EM2=float(np.mean((M * Er) ** 2)), t_front=round(t_front, 1), t_mom=round(t_mom, 1),
            t_diag=round(t_diag, 1), t_back=round(t_back, 1), t_total=round(time.time() - t0, 1))
print(json.dumps(summ))
with open(os.path.join(HERE, '..', 'logs', 'runs.jsonl'), 'a') as fh:
    fh.write(json.dumps(summ) + '\n')
