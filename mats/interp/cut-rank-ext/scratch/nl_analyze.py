"""Score NC1/NC2 (PREREG.md) from runs/NAME.npz written by nl_readout.py.

usage: nl_analyze.py runs/NAME.npz ...   (prints a summary; writes runs/NAME_scores.json)
Primary criterion: scale-normalised (front rows / s_f, back columns / t_b, as work/cut-rank/scratch/scale_norm.py).
Errors are on the test fronts (second half), relative to Vbar of the test fronts. MC floor = mean((MB - MC)^2)/4.
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cut-rank/lib')
import spec


def lin_fit(Xtr, Ytr, Xte, ridge=1e-8):
    """Least squares per column: Ytr (Ftr, C) on Xtr (Ftr, p); returns predictions on Xte."""
    G = Xtr.T @ Xtr
    G = G + ridge * np.trace(G) / len(G) * np.eye(len(G))
    W = np.linalg.solve(G, Xtr.T @ Ytr)
    return Xte @ W


def score(path, normalise=True):
    d = np.load(path)
    n = int(d['n']); F = int(d['F']); B = int(d['B'])
    MB, MC, E2 = d['MB'], d['MC'], d['E2']
    T = 0.5 * (MB + MC)
    V = np.clip(E2 - T ** 2, 0, None)
    if normalise:
        s_f = np.sqrt(E2.mean(axis=(1, 2)))
        alive = s_f > 0
        s_f = np.where(alive, s_f, np.inf)
        t_b = np.sqrt((E2[alive] / s_f[alive, None, None] ** 2).mean(axis=(0, 2)))
        t_b = np.where(t_b > 0, t_b, np.inf)
    else:
        s_f = np.ones(F); alive = np.ones(F, bool); t_b = np.ones(B)
    sc = s_f[:, None, None] * t_b[None, :, None]
    N = lambda X: X / sc
    te = np.zeros(F, bool); te[F // 2:] = True
    tr = ~te
    te &= alive; tr &= alive
    Tn, MBn, MCn = N(T), N(MB), N(MC)
    Vn = V / sc ** 2
    Vbar = float(Vn[te].mean())
    floor = float(((MBn[te] - MCn[te]) ** 2).mean() / 4)
    err = lambda X: float(np.nanmean((Tn[te] - N(X)[te]) ** 2))
    out = dict(name=os.path.basename(path)[:-4], normalised=normalise, n=n, F_alive=int(alive.sum()),
               F_test=int(te.sum()), Vbar=Vbar, mc_floor_rel=floor / Vbar)
    arrs = {k: d[k] for k in ('MF', 'K2', 'A2', 'A3')}
    pu = path[:-4] + '_arcunit.npz'                  # Amendment 1: unit-scaled ARC/K2X reruns (test fronts)
    if os.path.exists(pu):
        u = np.load(pu)
        arrs.update(A2u=u['A2u'], A3u=u['A3u'], K2u=u['K2u'])
    pk = path[:-4] + '_arc3k4.npz'                   # Amendment 2: ARC3 with the fourth cumulant (test fronts)
    if os.path.exists(pk):
        arrs.update(A3k4=np.load(pk)['A3k4'])
    names = dict(MF='MF', K2='K2X', A2='ARC2', A3='ARC3', A2u='ARC2u', A3u='ARC3u', K2u='K2Xu', A3k4='ARC3k4')
    for key, X in arrs.items():
        nm = names[key]
        if np.all(np.isnan(X[te])):
            out[nm] = None
        else:
            out[nm] = err(X) / Vbar
            out[nm + '_nan_frac'] = float(np.isnan(X[te]).mean())
            e_pair = ((Tn[te] - N(X)[te]) ** 2).mean(axis=2)            # per (front, back)
            out[nm + '_median'] = float(np.nanmedian(e_pair) / Vbar)
            out[nm + '_blowup_frac'] = float(np.nanmean(e_pair > 100 * Vbar))
    # linear readouts of the same front numbers (features normalised to degree 0)
    MU, S2 = d['MU'], d['S2']
    iu = np.triu_indices(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = MU / s_f[:, None]
        f2 = S2[:, iu[0], iu[1]] / s_f[:, None] ** 2
    X1 = np.concatenate([np.ones((F, 1)), f1], 1)
    X2 = np.concatenate([X1, f2], 1)
    Y = Tn.reshape(F, -1)
    for nm, X in (('LIN1', X1), ('LIN2', X2)):
        P = lin_fit(X[tr], Y[tr], X[te])
        out[nm] = float(((Y[te] - P) ** 2).mean()) / Vbar
    # optimal linear protocols: in-sample tail of the pair-averaged Gram on the test fronts
    tail, _ = spec.tails_in([MBn[te].reshape(te.sum(), -1), MCn[te].reshape(te.sum(), -1)])
    s1, s2 = n + 1, 1 + n + n * (n + 1) // 2
    out['SVD_s1'] = float(tail[min(s1, len(tail) - 1)]) / Vbar
    out['SVD_s2'] = float(tail[min(s2, len(tail) - 1)]) / Vbar
    out['s1'], out['s2'] = s1, s2
    out['r_in'] = {e: int(spec.eps_rank(tail, float(e) ** 2 * Vbar)) for e in ('0.3', '0.1', '0.03')}
    # NC2: bits on the mean
    QS = [int(q) for q in d["QS"]]; MFq = d["MFq"]
    e_mf = out['MF'] * Vbar
    errs_q = [err(MFq[i]) for i in range(len(QS))]
    out['MFq_rel'] = [e / Vbar for e in errs_q]
    exc = np.array(errs_q) - e_mf
    thr = 2 * max(floor, e_mf)
    sel = [i for i in range(len(QS)) if exc[i] > thr]
    if len(sel) >= 2:
        qq = np.array([QS[i] for i in sel], float); ll = np.log2(exc[sel])
        out['NC2_slope'] = float(np.polyfit(qq, ll, 1)[0]); out['NC2_slope_q'] = [int(q) for q in qq]
    else:
        out['NC2_slope'] = None; out['NC2_slope_q'] = [QS[i] for i in sel]
    hit = [QS[i] for i in range(len(QS)) if errs_q[i] / Vbar <= 0.01]
    out['NC2_qmin_0.01'] = int(hit[0]) if hit else None
    out['NC2_Bmin_0.01'] = int(n * hit[0] + 16) if hit else None
    r01 = out['r_in']['0.1']
    out['log2_r_in_0.1'] = math.log2(r01) if r01 > 0 else 0.0
    pg = path[:-4] + '_nc2g.npz'                     # Amendment 2: global norm codebook (no side information)
    if os.path.exists(pg):
        g = np.load(pg); MFqg = g['MFqg']
        errs_g = [err(MFqg[i]) for i in range(len(QS))]
        out['MFqg_rel'] = [e / Vbar for e in errs_g]
        out['NC2g_overflow'] = int(g['overflow'])
        exg = np.array(errs_g) - e_mf
        selg = [i for i in range(len(QS)) if exg[i] > thr]
        if len(selg) >= 2:
            qq = np.array([QS[i] for i in selg], float)
            out['NC2g_slope'] = float(np.polyfit(qq, np.log2(exg[selg]), 1)[0])
            out['NC2g_slope_q'] = [int(q) for q in qq]
        else:
            out['NC2g_slope'] = None; out['NC2g_slope_q'] = [QS[i] for i in selg]
        hitg = [QS[i] for i in range(len(QS)) if errs_g[i] / Vbar <= 0.01]
        out['NC2g_Bmin_0.01'] = int(n * hitg[0] + 16) if hitg else None
    return out


if __name__ == '__main__':
    for path in sys.argv[1:]:
        res = [score(path, True), score(path, False)]
        with open(path[:-4] + '_scores.json', 'w') as fh:
            json.dump(res, fh, indent=1)
        for r in res:
            crit = 'scale-normalised' if r['normalised'] else 'raw'
            f = lambda v: 'n/a' if v is None else f'{v:.3g}'
            g = lambda k: f(r.get(k))
            print(f"{r['name']} [{crit}] alive {r['F_alive']} test {r['F_test']} | MC floor {f(r['mc_floor_rel'])} | "
                  f"MF {f(r['MF'])} K2X {f(r['K2X'])} ARC2 {f(r['ARC2'])} ARC3 {f(r['ARC3'])} | unit-scaled: K2Xu "
                  f"{g('K2Xu')} ARC2u {g('ARC2u')} (median {g('ARC2u_median')}, blow-ups {g('ARC2u_blowup_frac')}) "
                  f"ARC3u {g('ARC3u')} (median {g('ARC3u_median')}, blow-ups {g('ARC3u_blowup_frac')}) | LIN1 {f(r['LIN1'])} "
                  f"LIN2 {f(r['LIN2'])} | SVD({r['s1']}) {f(r['SVD_s1'])} SVD({r['s2']}) {f(r['SVD_s2'])} | "
                  f"r_in {r['r_in']} | NC2 slope {f(r['NC2_slope'])} over q={r['NC2_slope_q']}, "
                  f"Bmin(0.01)={r['NC2_Bmin_0.01']}, log2 r_in(0.1)={r['log2_r_in_0.1']:.2f} | global code: slope "
                  f"{g('NC2g_slope')} over q={r.get('NC2g_slope_q')}, Bmin(0.01)={r.get('NC2g_Bmin_0.01')}, overflow "
                  f"{r.get('NC2g_overflow')} | ARC3k4 {g('ARC3k4')} (median {g('ARC3k4_median')}, blow-ups "
                  f"{g('ARC3k4_blowup_frac')})")
