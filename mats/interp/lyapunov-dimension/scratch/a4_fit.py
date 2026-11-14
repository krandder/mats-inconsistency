"""A4: does the linear-response tail predict the measured family errors?
For each net and cut with saved M (runs/n2mat_*) and the nongauss lane's measured nMSE (read-only), compute the
first-order tail of each family at its block size and fit nMSE = C * tail^beta (log-log least squares) over points
above a floor. Families: PB_k (top-k PCs; residual independent), PBA_k (PCs + sensitivity eigenvectors, dim <= 2k),
PBS_k (sensitivity eigenvectors), PBW_k (orth(Sigma^1/2 V_k)), and the optimum S_k (whitened, not measured).
python a4_fit.py"""
import glob, json, re, sys
import numpy as np

NG = '/home/agent/workspaces/mats/work/nongauss-family/runs/gpu/ngfam2c/a1/e1_256_64_{seed}.json'
EXTRA = {}   # R1 numbers (PB2..PB32) if they land: {(seed, cut, fam): nmse}


def sqrtm(C):
    e, V = np.linalg.eigh(0.5 * (C + C.T)); e = np.clip(e, 0, None)
    return (V * np.sqrt(e)) @ V.T


def schur_tail(B, Sig, M, denom):
    Q, _ = np.linalg.qr(B)
    G = Q.T @ Sig @ Q
    SB = Sig @ Q
    Sch = Sig - SB @ np.linalg.pinv(G, rcond=1e-12) @ SB.T
    return float(np.trace(M @ Sch) / denom)


def tails_for(npz, Lp):
    D = np.load(npz)
    Sig, mu, M = D['Sigma'], D['mu'], D[f'M{Lp}']
    n = Sig.shape[0]
    u = mu / np.linalg.norm(mu); P = np.eye(n) - np.outer(u, u)
    SigP = P @ Sig @ P
    R = sqrtm(SigP)
    ec, Vc = np.linalg.eigh(SigP); Vc = Vc[:, ::-1]
    S = R @ M @ R
    eS, VS = np.linalg.eigh(0.5 * (S + S.T)); eS = np.clip(eS[::-1], 0, None); VS = VS[:, ::-1]
    MP = P @ M @ P
    eM, VM = np.linalg.eigh(0.5 * (MP + MP.T)); VM = VM[:, ::-1]
    denom = float(np.trace(M @ SigP))
    out = {}
    for k in (2, 4, 8, 16, 32, 64, 128):
        out[f'PB{k}'] = schur_tail(Vc[:, :k], SigP, M, denom)
        out[f'PBS{k}'] = schur_tail(VM[:, :k], SigP, M, denom)
        out[f'PBW{k}'] = schur_tail(R @ VS[:, :k], SigP, M, denom)
        out[f'PBA{k}'] = schur_tail(np.concatenate([Vc[:, :k], VM[:, :k]], 1), SigP, M, denom)
        out[f'S{k}'] = float(eS[k:].sum() / eS.sum())
    return out


pts = []
for npz in sorted(glob.glob('runs/n2mat_256_64_c*_s*_sig0.001.npz')):
    m = re.search(r'_c(\d+)_s(\d+)_', npz); cut, seed = int(m.group(1)), int(m.group(2))
    Lp = 64 - cut
    D = np.load(npz)
    if f'M{Lp}' not in D.files:
        Lp = max(int(k[1:]) for k in D.files if k.startswith('M'))
    tl = tails_for(npz, Lp)
    try:
        ng = json.load(open(NG.format(seed=seed)))
    except FileNotFoundError:
        continue
    emp = ng['runs'].get(f'0/{cut}/EMP', {}).get('nmse', [np.nan])[-1]
    for key, rec in ng['runs'].items():
        rep, c, fam = key.split('/')
        if int(c) != cut or fam == 'EMP':
            continue
        pts.append(dict(seed=seed, cut=cut, fam=fam, nmse=rec['nmse'][-1], tail=tl.get(fam, np.nan), emp=emp, Lp=Lp))
    for (sd, c, fam), v in EXTRA.items():
        if sd == seed and c == cut:
            pts.append(dict(seed=seed, cut=cut, fam=fam, nmse=v, tail=tl.get(fam, np.nan), emp=emp, Lp=Lp))
    print(f"net {seed} cut {cut} (M at L' = {Lp}): optimal S tails k=8,16,32,64: "
          + " ".join(f"{tl[f'S{k}']:.2e}" for k in (8, 16, 32, 64)), flush=True)

print("\nseed cut fam       nMSE(L=64)   first-order tail")
for p in sorted(pts, key=lambda p: (p['cut'], p['fam'], p['seed'])):
    print(f"{p['seed']} {p['cut']:3d} {p['fam']:7s} {p['nmse']:10.2e}   {p['tail']:.3e}")
for cut in sorted({p['cut'] for p in pts}):
    for floor in (2e-6, 4e-6):
        Q = [p for p in pts if p['cut'] == cut and p['nmse'] > floor and np.isfinite(p['tail'])]
        if len(Q) < 4:
            continue
        x = np.log([p['tail'] for p in Q]); y = np.log([p['nmse'] for p in Q])
        A = np.vstack([np.ones_like(x), x]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rms = float(np.sqrt(((y - pred) ** 2).mean()))
        print(f"cut {cut}, points with nMSE > {floor:g}: n = {len(Q)}, beta = {coef[1]:.2f}, C = {np.exp(coef[0]):.2e}, "
              f"R^2 = {r2:.2f}, rms log-residual {rms:.2f} (factor {np.exp(rms):.2f})", flush=True)
        # per-family mean residual
        fams = sorted({p['fam'].rstrip('0123456789') for p in Q})
        print("   mean log-residual by family: " + ", ".join(
            f"{f} {np.mean([y[i] - pred[i] for i, p in enumerate(Q) if p['fam'].rstrip('0123456789') == f]):+.2f}" for f in fams))
# ---- net-mean level (adds the R1 PB ladder means from the nongauss summary, cut 32 -> L = 64, 4 nets)
R1_MEANS = {32: {'PB0': 1.8e-4, 'PB8': 1.1e-4, 'PB16': 4.4e-5, 'PB32': 1.6e-5}}
print("\nnet-mean level fit (cut 32): mean nMSE over nets vs mean first-order tail over nets")
for cut in sorted({p['cut'] for p in pts}):
    seeds = sorted({p['seed'] for p in pts if p['cut'] == cut})
    tl_by_seed = {}
    for sd in seeds:
        npz = f'runs/n2mat_256_64_c{cut}_s{sd}_sig0.001.npz'
        D = np.load(npz); Lp = 64 - cut if f'M{64 - cut}' in D.files else max(int(k[1:]) for k in D.files if k.startswith('M'))
        tl_by_seed[sd] = tails_for(npz, Lp)
    rows = []
    fams = sorted({p['fam'] for p in pts if p['cut'] == cut})
    for fam in fams:
        v = [p['nmse'] for p in pts if p['cut'] == cut and p['fam'] == fam]
        t = [tl_by_seed[sd][fam] for sd in seeds if fam in tl_by_seed[sd]]
        rows.append((fam, float(np.mean(v)), float(np.mean(t)), len(v)))
    for fam, v in R1_MEANS.get(cut, {}).items():
        t = 1.0 if fam == 'PB0' else float(np.mean([tl_by_seed[sd][fam] for sd in seeds]))
        rows.append((fam + '(R1)', v, t, 4))
    for fam, v, t, cnt in sorted(rows, key=lambda r: r[2], reverse=True):
        st = float(np.mean([tl_by_seed[sd].get('S' + ''.join(ch for ch in fam if ch.isdigit()), np.nan) for sd in seeds])) if any(ch.isdigit() for ch in fam) else np.nan
        print(f"  {fam:10s} mean nMSE {v:9.2e}  mean tail {t:.3e}  (optimal S-block tail at the same k: {st:.3e}; nets {cnt})")
    for floor in (2e-6, 4e-6):
        Q = [r for r in rows if r[1] > floor]
        x = np.log([r[2] for r in Q]); y = np.log([r[1] for r in Q])
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rms = float(np.sqrt(((y - pred) ** 2).mean()))
        print(f"  cut {cut}, mean-level points with nMSE > {floor:g}: n = {len(Q)}, beta = {coef[1]:.2f}, "
              f"C = {np.exp(coef[0]):.2e}, R^2 = {r2:.2f}, rms log-residual {rms:.2f} (factor {np.exp(rms):.2f})")
        for r, yy, pp in zip(Q, y, pred):
            print(f"      {r[0]:10s} log-residual {yy - pp:+.2f}")
json.dump(pts, open('runs/a4_fit_points.json', 'w'), indent=1)
