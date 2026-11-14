"""Analyse run_cut.py outputs (cut-rank lane; estimators fixed in PREREG.md + Amendment 1).

usage: python3 scratch/analyze.py runs/NAME.npz [...]   -> runs/analysis_NAME.json and one summary line each.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import spec

EPS = (0.3, 0.1, 0.03, 0.01)


def split_list(X):  # X: (F, S, ...) -> list of S matrices (F, rest)
    F, S = X.shape[:2]
    return [X[:, s].reshape(F, -1).astype(np.float64) for s in range(S)]


def rank_block(Ms, Vbar):
    ti, lam = spec.tails_in(Ms)
    tc = spec.tails_cf(Ms)
    return dict(r_in={str(e): spec.eps_rank(ti, e * e * Vbar) for e in EPS},
                r_cf={str(e): spec.eps_rank(tc, e * e * Vbar) for e in EPS},
                tail_in_rel=[float(ti[s] / Vbar) for s in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256) if s < len(ti)],
                tail_cf_rel=[float(tc[s] / Vbar) for s in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256) if s < len(tc)],
                lam_rel_top=[float(x / Vbar) for x in lam[:12]])


def moment_regression(mom, Ms, Vbar, n, kmax):
    """Held-out residual energy of the least-squares linear readout from cut moments of order <= K."""
    F = mom.shape[0]
    sizes = [1, n, n * (n + 1) // 2, n * (n + 1) * (n + 2) // 6]
    tr = np.arange(F // 2); te = np.arange(F // 2, F)
    S = len(Ms)
    Yfit = np.mean(Ms[:S // 2], axis=0)          # response: mean of splits 0..S/2-1
    Ts = [m[te] for m in Ms[S // 2:]]             # scoring: pairs of the other half (unbiased)
    out = {}
    for K in range(kmax + 1):
        p = sum(sizes[:K + 1])
        X = mom[:, :p]
        sc = X[tr].std(0); sc[sc == 0] = 1.0
        Xs = X / sc
        beta, *_ = np.linalg.lstsq(Xs[tr], Yfit[tr], rcond=1e-10)
        pred = Xs[te] @ beta
        out[str(K)] = dict(p=int(p), resid_rel=float(np.mean(spec._pair_mean_err(Ts, pred)) / Vbar))
    return out


def moment_ridge(mom, Ms, Vbar, n, kmax):
    """POST HOC (not pre-registered): ridge version of moment_regression, ridge chosen by 2-fold CV on the train half
    (CV score uses the cross-split pair estimator, so noise in the response does not reward overfitting)."""
    F = mom.shape[0]
    sizes = [1, n, n * (n + 1) // 2, n * (n + 1) * (n + 2) // 6]
    tr = np.arange(F // 2); te = np.arange(F // 2, F)
    S = len(Ms)
    Yfit = np.mean(Ms[:S // 2], axis=0)
    Ts_all = Ms[S // 2:]
    out = {}
    for K in range(kmax + 1):
        p = sum(sizes[:K + 1])
        X = mom[:, :p]
        mu = X[tr].mean(0); sc = X[tr].std(0); sc[sc == 0] = 1.0
        Xs = (X - mu) / sc; Xs[:, 0] = 1.0
        p_prev = sum(sizes[:K])                      # penalise only the order-K block: lam -> inf gives order K-1
        pen = np.r_[np.zeros(p_prev), np.ones(p - p_prev)] if K > 0 else np.zeros(p)
        def fit(idx, lam):
            A = Xs[idx]; G = A.T @ A + lam * len(idx) * np.diag(pen)
            return np.linalg.lstsq(G, A.T @ Yfit[idx], rcond=1e-12)[0]
        h1, h2 = tr[:len(tr) // 2], tr[len(tr) // 2:]
        best = None
        for lam in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e4):
            sc_cv = 0.0
            for a_, b_ in ((h1, h2), (h2, h1)):
                pred = Xs[b_] @ fit(a_, lam)
                sc_cv += float(np.mean(spec._pair_mean_err([m[b_] for m in Ts_all], pred)))
            if best is None or sc_cv < best[0]:
                best = (sc_cv, lam)
        pred = Xs[te] @ fit(tr, best[1])
        out[str(K)] = dict(p=int(p), lam=best[1],
                           resid_rel=float(np.mean(spec._pair_mean_err([m[te] for m in Ts_all], pred)) / Vbar))
    return out


def analyse(path):
    d = np.load(path)
    name = os.path.basename(path)[:-4]
    M = d['M']                                     # (F, S, B, n)
    F, S, B, n = M.shape
    V = d['V']
    Vbar = float(V.mean())
    Ms = split_list(M)
    res = dict(name=name, F=F, S=S, B=B, n=n, Vbar=Vbar, EM2_rel=float(np.mean(M ** 2) / Vbar))
    res['ensemble'] = rank_block(Ms, Vbar)
    sat = {}
    for Fp in (64, 128, 256, 512):
        if Fp <= F:
            ti, _ = spec.tails_in([m[:Fp] for m in Ms])
            sat[str(Fp)] = {str(e): spec.eps_rank(ti, e * e * Vbar) for e in EPS}
    res['saturation_in'] = sat
    tho = spec.tails_heldout(Ms)                   # POST HOC upper estimate (held-out fronts)
    res['heldout_posthoc'] = dict(r_ho={str(e): spec.eps_rank(tho, e * e * Vbar) for e in EPS},
                                  tail_ho_rel=[float(tho[s] / Vbar) for s in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256) if s < len(tho)])
    pe = spec.probe_protocol(Ms, [0, 1, 2, 4, 8, 16, 32, 64, 128], np.random.default_rng(0))
    res['probe_rel'] = {str(k): v / Vbar for k, v in pe.items()}
    r01 = res['ensemble']['r_in']['0.1']
    s8 = max(1, min(F // 2, r01 // 2))
    e = spec.per_net_errors(Ms, s8, n)             # per (front, back): columns are (back, neuron) blocks of n
    Vfb = V.mean(2)                                # (F, B)
    res['per_net'] = dict(s=int(s8), frac_err_gt_0p01V=float(np.mean(e > 0.01 * Vfb)),
                          median_err_over_V=float(np.median(e[Vfb > 0] / Vfb[Vfb > 0])))
    # fixed-back readouts
    fb = []
    for b in range(d['R2'].shape[2]):
        row = {}
        for key in ('R2', 'R3'):
            R = d[key][:, :, b].reshape(F, S, -1).astype(np.float64)
            p = R.mean(1)
            Vb = float(np.mean(p * (1 - p)))
            Rs = split_list(R)
            ti, _ = spec.tails_in(Rs); tc = spec.tails_cf(Rs)
            row[key] = dict(Vbar=Vb, r_in={str(e): spec.eps_rank(ti, e * e * Vb) for e in EPS},
                            r_cf={str(e): spec.eps_rank(tc, e * e * Vb) for e in EPS})
        fb.append(row)
    res['fixed_back'] = fb
    cum = np.cumsum([1, n, n * (n + 1) // 2, n * (n + 1) * (n + 2) // 6])
    kmax = int(np.nonzero(cum == d['mom'].shape[1])[0][0])
    res['moments'] = moment_regression(d['mom'], Ms, Vbar, n, kmax)
    res['moments_ridge_posthoc'] = moment_ridge(d['mom'], Ms, Vbar, n, kmax)
    res['diag'] = dict(kstar01=float(d['ks01'].mean()), kstar1=float(d['ks1'].mean()),
                       angle=float(np.median(d['angles'])), gate_dis=float(np.median(d['gdis'])), PR=float(d['prs'].mean()))
    with open(os.path.join(HERE, '..', 'runs', 'analysis_' + name + '.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    E = res['ensemble']
    print(f"{name}: k*={res['diag']['kstar01']:.1f} angle={res['diag']['angle']:.3f} PR={res['diag']['PR']:.1f} "
          f"EM2/V={res['EM2_rel']:.1f} | r_in {[E['r_in'][str(e)] for e in EPS]} r_cf {[E['r_cf'][str(e)] for e in EPS]} "
          f"| sat {[sat[k]['0.1'] for k in sat]} | probe {[round(res['probe_rel'][k], 4) for k in ('1', '4', '16', '64')]} "
          f"| perNet frac>{res['per_net']['frac_err_gt_0p01V']:.2f} | mom {[round(v['resid_rel'], 4) for v in res['moments'].values()]}"
          f" ridge {[round(v['resid_rel'], 4) for v in res['moments_ridge_posthoc'].values()]}"
          f" | fixedR3 r_in(0.1) {[x['R3']['r_in']['0.1'] for x in fb]} | r_ho {list(res['heldout_posthoc']['r_ho'].values())}")
    return res


if __name__ == '__main__':
    for p in sys.argv[1:]:
        analyse(p)
