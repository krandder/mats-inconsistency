"""Adversarial re-runs on one tiny-PFN net (review of cm-interp round 1). usage: python e124_real.py MODEL NC
E1  G quadrature on SYNTHETIC outputs: 201-point Bayes grid (as used) vs 2001 points, same range (METHOD §2.4 asked for it).
E2  Is the 'noise share' off-manifold-specific? Same mean, residual drawn from the EMPIRICAL residual pool of the fit split
    (independent per token, like q0's independence) vs q0's Gaussian; and real + Gaussian vs real + empirical residual
    difference (same covariance). Under a local-quadratic response both pairs should match.
E4  Output-space classifier exactly as r1est.crossfit_cells (1 rep): (a) leakage of the within-fold test split (BCE vs
    out-of-fold BCE), (b) out-of-fold Platt slope, (c) interior L(alpha) under s * log w-hat, (d) top-decile share with
    region selection and estimation on the same draws (as round 1) vs on disjoint halves."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp                       # noqa: E402
from lib import r1est as E               # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

NAME, NC = sys.argv[1], int(sys.argv[2])
NFIT, M = 20000, 8
T0 = time.time()
log = lambda *a: print(f'[{NAME} {time.time() - T0:5.0f}s]', *a, flush=True)
model, meta, ev = tp.load(NAME); L = meta['depth']
want = {('post', k): None for k in range(1, L)}; want[('post', L)] = [tp.QI]


def prep(seed, n, M_):
    d0 = tp.sample_episodes(n, seed); cs0 = tp.cell_stats(d0)
    if M_:
        d = tp.cond_draws(d0, M_, seed + 1)
    else:
        d = d0; d['cell'] = np.arange(n)
    X, V = tp.tokens(d)
    head, st, _ = tp.run_real(model, X, V, want)
    GQ = tp.GQuad(cs0)
    return dict(d=d, cs=tp.take(cs0, d['cell']), cs0=cs0, V=V, st=st, GQ=GQ, cell=d['cell'], n=n, M=max(M_, 1),
                head=head, Gp=GQ(head, d['cell']).reshape(n, max(M_, 1)))


F = prep(60000, NFIT, 0); B = prep(63500, NC, M)
log('data ready, KL_model(B) %.5f' % B['Gp'].mean())
rng = np.random.default_rng(12345)
mid, fin = ('post', L - 1), ('post', L)
sites = [(mid, ('LAB', 'QQ', 'QT'), 'ALL'), (fin, ('QT',), 'QT')]
out = {}
heads_q0 = {}
for cut, tys, lab in sites:
    isfin = cut == fin
    key = f'h{cut[1]}-{lab}'
    parts = {}
    for t in tys:
        ix = [0] if isfin else tp.GROUPS[t]
        norm = tp.needs_norm(model, cut, t)
        Hf = tp.to_q(F['st'][cut][:, ix].astype(np.float64), norm)
        q = E.QTok(t, 'q0', norm=norm).fit(F['d'], F['cs'], Hf)
        muF = q.mean_cov(F['d'], F['cs'])[0]
        muB, covs, _ = q.mean_cov(B['d'], B['cs'])
        C = covs[0]
        realB = tp.to_q(B['st'][cut][:, ix].astype(np.float64), norm)
        parts[t] = dict(ix=ix, norm=norm, eF=Hf - muF, muB=muB, C=C, realB=realB)
    res = {}
    for mode in ('q0', 'emp', 'noise', 'noise_emp', 'mean'):
        h = B['st'][cut].copy()
        for t, P_ in parts.items():
            N_, nt, dm = P_['muB'].shape
            eps = (rng.standard_normal((N_ * nt, dm)) @ np.linalg.cholesky(P_['C']).T).reshape(N_, nt, dm)
            pick = lambda: P_['eF'][rng.integers(0, NFIT, (N_, nt)), np.arange(nt)[None, :]]
            if mode == 'q0':
                hs = P_['muB'] + eps
            elif mode == 'emp':
                hs = P_['muB'] + pick()
            elif mode == 'noise':
                hs = P_['realB'] + eps
            elif mode == 'noise_emp':
                hs = P_['realB'] + (pick() - pick()) / math.sqrt(2)
            else:
                hs = P_['muB']
            h[:, P_['ix']] = tp.to_q(hs, P_['norm'])
        head = tp.forward_qt(model, h, cut) if isfin else tp.forward_from(model, h, cut, B['V'])
        G = B['GQ'](head, B['cell']).reshape(NC, M)
        r = E.direct(G, B['Gp'], B=300)
        res[mode] = dict(delta=r['delta'], se=r['se'], rms=r['rms'])
        if mode == 'q0':
            heads_q0[key] = (head, G)
        log(key, mode, 'delta %.4f +- %.4f rms %.4f' % (r['delta'], r['se'], r['rms']))
    res['noise_share'] = res['noise']['delta'] / res['q0']['delta']
    res['additivity_mean_plus_noise_over_q0'] = (res['mean']['delta'] + res['noise']['delta']) / res['q0']['delta']
    out[key] = dict(E2=res)

# ---------------------------------------------------------------- E1: grid resolution on synthetic outputs
nsub = min(800, NC); sel = np.arange(nsub * M)
cs_sub = tp.take(B['cs0'], np.arange(nsub))
tp.NGRID = 2001
GQf = tp.GQuad(cs_sub)
tp.NGRID = 201
for key, (head, G) in heads_q0.items():
    hsub = tuple(a[sel] for a in head)
    g201 = G.ravel()[sel]
    g2001 = GQf(hsub, B['cell'][sel], chunk=1000)
    dif = g201 - g2001
    out[key]['E1'] = dict(max_abs=float(np.abs(dif).max()), mean_diff=float(dif.mean()), rel_mean=float(dif.mean() / g2001.mean()),
                          frac_logsd_at_clip=float(np.mean(hsub[1] <= -5.999)))
    log(key, 'E1 grid 201 vs 2001: max|diff| %.2e mean diff %.2e (rel %.2e), frac log sd at -6 clip %.4f' % (
        np.abs(dif).max(), dif.mean(), dif.mean() / g2001.mean(), np.mean(hsub[1] <= -5.999)))
hr = tuple(a[sel] for a in B['head'])
dr = B['Gp'].ravel()[sel] - GQf(hr, B['cell'][sel], chunk=1000)
log('E1 real outputs: max|diff| %.2e' % np.abs(dr).max())

# ---------------------------------------------------------------- E4: output-space classifier, as round 1
ZB = np.concatenate([B['cs']['ostar'], B['cs']['lss'], B['cs']['mb'][:, None], B['cs']['lvb'][:, None]], 1)
HVp = tp.head_vec(B['head'])
alphas = np.linspace(0, 1, 11)
for key, (head, G) in heads_q0.items():
    Fp = np.concatenate([HVp, ZB], 1); Fq = np.concatenate([tp.head_vec(head), ZB], 1)
    # identical to r1est.crossfit_cells(reps=1, seed=11), unrolled to keep the per-fold diagnostics
    from cmlib.ratio import fit_ratio, ratio_diagnostics
    rngf = np.random.default_rng(11); fold_c = rngf.integers(0, 2, NC); fold = fold_c[B['cell']]
    lwp, lwq = np.empty(len(Fp)), np.empty(len(Fq)); fold_bce, fold_trte, fold_ece = [], [], []
    for k in range(2):
        tr = fold != k
        rm, dk = fit_ratio(Fp[tr], Fq[tr], family='mlp', seed=11 + k)
        lwp[~tr] = rm.log_w(Fp[~tr]); lwq[~tr] = rm.log_w(Fq[~tr])
        fold_bce.append(dk['bce']); fold_trte.append(dk['train_minus_test_bce']); fold_ece.append(dk['ece'])
    dg = ratio_diagnostics(lwp, lwq)
    # (a) within-fold test BCE (fit_ratio's draw-level split shares cells with its train split) vs out-of-fold BCE
    oof_bce = dg['bce']
    # (b) out-of-fold Platt slope on the out-of-fold log w-hat (1 = calibrated scale; < 1 = overconfident)
    s = np.concatenate([lwp, lwq]); y = np.concatenate([np.ones(len(lwp)), np.zeros(len(lwq))])
    pl = LogisticRegression(C=1e6, max_iter=2000).fit(s[:, None], y)
    # (c) interior L(alpha) under s * log w-hat
    lwP, lwQ = lwp.reshape(NC, M), lwq.reshape(NC, M)
    curves = {}
    for sc in (0.5, 1.0, 2.0):
        Lc, _, ess = E._bridge_cells(B['Gp'], sc * lwP, G, sc * lwQ, alphas)
        Lm = Lc.mean(0)
        curves[str(sc)] = dict(L=[float(x) for x in Lm], gap_over_direct=float((Lm[0] - Lm[-1]) / (G.mean() - B['Gp'].mean())),
                               drop_by_0p2=float((Lm[0] - Lm[2]) / (Lm[0] - Lm[-1])), ess_cell_min=float(ess.mean(0).min()))
    # (d) top-decile share: same draws (round 1) vs disjoint halves
    D = G - B['Gp']

    def share(lw_sel, dcell):
        e = np.quantile(lw_sel, np.linspace(.1, .9, 9)); b = np.digitize(lw_sel, e)
        return float(dcell[b == 9].sum() / dcell.sum())
    same = share(lwP.mean(1), D.mean(1))
    split = 0.5 * (share(lwP[:, :4].mean(1), D[:, 4:].mean(1)) + share(lwP[:, 4:].mean(1), D[:, :4].mean(1)))
    # bootstrap SE of same - split over cells
    bs = []
    for _ in range(200):
        i = rng.integers(0, NC, NC)
        bs.append(share(lwP[i].mean(1), D[i].mean(1)) - 0.5 * (share(lwP[i][:, :4].mean(1), D[i][:, 4:].mean(1)) +
                                                              share(lwP[i][:, 4:].mean(1), D[i][:, :4].mean(1))))
    out[key]['E4'] = dict(auc=dg['auc'], Eq_w=float(np.exp(lwq).mean()), Ep_invw=float(np.exp(-lwp).mean()),
                          lw_q_quantiles=[float(x) for x in np.quantile(lwq, [0, .01, .5, .99, 1])],
                          lw_p_quantiles=[float(x) for x in np.quantile(lwp, [0, .01, .5, .99, 1])],
                          within_fold_test_bce=fold_bce, within_fold_train_minus_test=fold_trte, within_fold_ece=fold_ece,
                          oof_bce=oof_bce, oof_platt_slope=float(pl.coef_[0, 0]),
                          oof_platt_intercept=float(pl.intercept_[0]), curves=curves,
                          top_decile_share_same=same, top_decile_share_split=split,
                          same_minus_split_se=float(np.std(bs, ddof=1)))
    log(key, 'E4', json.dumps(out[key]['E4'])[:900])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'e124_{NAME}.json'), 'w'), indent=1)
log('done')
