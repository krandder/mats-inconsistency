"""Predictions for the PBO addendum, written before any PBO run at 256 x 64 (only the 64 x 16 CPU smoke test was run).
Proxy first-order tails (z chart, mean direction projected out; ../runs/n2mat_*.npz, M at the output L = 64) for the
optimal block (S), PCs (PB) and PBW at k in KS, per net and cut; predicted nMSE = C_cut * tail (beta = 1), with C from
the report's family-mean fits (cut 32: 2.69e-4; cut 48: beta = 1 refit on the same mean-level points).
python prereg_predict.py > ../PREREG-PBO-predictions.txt"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', '..', 'runs')
KS = [8, 16, 24, 32, 48, 64]


def sqrtm(C):
    e, V = np.linalg.eigh(0.5 * (C + C.T)); e = np.clip(e, 0, None)
    return (V * np.sqrt(e)) @ V.T


def schur_tail(B, Sig, M, denom):
    Q, _ = np.linalg.qr(B)
    G = Q.T @ Sig @ Q; SB = Sig @ Q
    return float(np.trace(M @ (Sig - SB @ np.linalg.pinv(G, rcond=1e-12) @ SB.T)) / denom)


def tails(npz, Lp):
    D = np.load(npz); Sig, mu, M = D['Sigma'], D['mu'], D[f'M{Lp}']
    n = Sig.shape[0]; u = mu / np.linalg.norm(mu); P = np.eye(n) - np.outer(u, u)
    SigP = P @ Sig @ P; R = sqrtm(SigP)
    ec, Vc = np.linalg.eigh(SigP); Vc = Vc[:, ::-1]
    S = R @ M @ R; eS, VS = np.linalg.eigh(0.5 * (S + S.T)); eS = np.clip(eS[::-1], 0, None); VS = VS[:, ::-1]
    denom = float(np.trace(M @ SigP))
    out = {}
    for k in KS:
        out[('PBO', k)] = float(eS[k:].sum() / eS.sum())
        out[('PB', k)] = schur_tail(Vc[:, :k], SigP, M, denom)
        out[('PBW', k)] = schur_tail(R @ VS[:, :k], SigP, M, denom)
    return out


# C at cut 48 with beta = 1, from the report's family-mean points
pts = json.load(open(os.path.join(RUNS, 'a4_fit_points.json')))
fam48 = {}
for p in pts:
    if p['cut'] == 48:
        fam48.setdefault(p['fam'], []).append((p['nmse'], p['tail']))
lr = [np.log(np.mean([a for a, b in v]) / np.mean([b for a, b in v])) for f, v in fam48.items()
      if np.mean([a for a, b in v]) > 2e-6]
C = {32: 2.69e-4, 48: float(np.exp(np.mean(lr)))}
print(f"C (beta = 1): cut 32 {C[32]:.2e} (report, 10 family means); cut 48 {C[48]:.2e} ({len(lr)} family means)")
pred = {}
for cut in (32, 48):
    Lp = 64 - cut
    for seed in (113, 114, 115, 116):
        npz = os.path.join(RUNS, f'n2mat_256_64_c{cut}_s{seed}_sig0.001.npz')
        t = tails(npz, Lp)
        row = {f'{fam}{k}': C[cut] * t[(fam, k)] for (fam, k) in t}
        pred[f'{cut}/{seed}'] = row
        kO = next((k for k in KS if row[f'PBO{k}'] <= 1e-5), None)
        kP = next((k for k in KS if row[f'PB{k}'] <= 1e-5), None)
        print(f"cut {cut} net {seed}: predicted nMSE at L = 64")
        for fam in ('PBO', 'PBW', 'PB'):
            print(f"   {fam:4s} k=" + ",".join(map(str, KS)) + ": " + " ".join(f"{row[f'{fam}{k}']:.1e}" for k in KS))
        print(f"   smallest k in the grid with predicted nMSE <= 1e-5: PBO {kO}, PB {kP}; "
              f"PB/PBO ratio at k = 16, 32: {row['PB16']/row['PBO16']:.1f}, {row['PB32']/row['PBO32']:.1f}; "
              f"PBW/PBO at k = 16, 32: {row['PBW16']/row['PBO16']:.1f}, {row['PBW32']/row['PBO32']:.1f}")
json.dump(dict(C=C, pred=pred), open(os.path.join(HERE, '..', 'PREREG-PBO-predictions.json'), 'w'), indent=1)
