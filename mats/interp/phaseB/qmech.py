"""E8 analysis (PREREG-B amendment 15 and its addendum).
E8.1: implied vs Bayes source weights; linearised mean deviation sum_s delta_s (o*_s - c); measured-delta ceiling;
compact additive piecewise-linear weight-error model g of six per-source quantities, fitted to group-averaged delta on fit
episodes and scored against the group's seed-averaged test mean deviation on the finite-ensemble scale.
E8.2: retained shared mean / spread error and mean-squared-deviation ratios of the layer-2 query-row patches.
E8.3 (conditional): held-out A net nb6.
usage: python qmech.py -> results/qmech.json, results/qmech_tables.md
"""
import json, os
import numpy as np
import nnoise  # noqa: F401  (puts tiny-pfn on sys.path)
import world as W
import devpred as DP
import qweights as QWT

A_NETS = ['L2_s0.05', 'L2seed1_s0.05', 'L2mx0_s0.05', 'L2mx1_s0.05', 'L2mx2_s0.05', 'L2nb6_s0.05']
B_NETS = ['L2seed2_s0.05', 'L2ws0_s0.05', 'L2ws3_s0.05', 'L2mx4_s0.05', 'L2nb5_s0.05', 'L2nb7_s0.05']
ALL = A_NETS + B_NETS; GROUP = {**{n: 'A' for n in A_NETS}, **{n: 'B' for n in B_NETS}}
N8, N4, NB = 8000, 4000, 500
VARIANTS = ['Pbayes', 'PbayesQQ', 'Ppost', 'PpostQQ', 'Ptrue', 'Puniform', 'Pmean']
INPUTS = ['log own labelled SS', 'log other SS (smaller)', 'log other SS (larger)', 'Bayes weight W_s', '|o*_s - mean of other o*|', 'max |ID cosine|']
HOLD = 'L2nb6_s0.05'
f = lambda x: f'{x:.2f}'.replace('-', '−')


def cov(a, b): return float(np.mean((a - a.mean()) * (b - b.mean())))


def r2(y, p): return DP.r2(y, p)


def per_source_feats(d, X, WB):
    o = d['o'][:, :, W.QL]; res = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]; lss = np.log((res ** 2).sum(2) + 1e-3)
    semb = np.stack([X[:, 5 * s, :16] for s in range(3)], 1).astype(np.float64); sn = semb / np.linalg.norm(semb, axis=2, keepdims=True)
    C = np.abs(np.einsum('nsd,ntd->nst', sn, sn)); rows = []
    for s in range(3):
        oth = [t for t in range(3) if t != s]
        rows.append(np.c_[lss[:, s], lss[:, oth].min(1), lss[:, oth].max(1), WB[:, s], np.abs(o[:, s] - o[:, oth].mean(1)), C[:, s, oth].max(1)])
    return np.stack(rows, 1)                                     # (n, 3, 6)


def basis(Z, knots, drop=None):
    cols = [np.ones((len(Z), 1))]
    for j in range(Z.shape[1]):
        if j == drop: continue
        cols.append(Z[:, j:j + 1]); cols.append(np.maximum(Z[:, j:j + 1] - knots[j][None], 0))
    return np.concatenate(cols, 1)


def fit_g(zF, delF, knots, drop=None):
    coef, *_ = np.linalg.lstsq(basis(zF.reshape(-1, zF.shape[2]), knots, drop), delF.reshape(-1), rcond=None); return coef


def pred_g(coef, z, knots, drop=None): return (basis(z.reshape(-1, z.shape[2]), knots, drop) @ coef).reshape(-1, 3)


def test8000():
    p = 'results/qmech_test8000.npz'
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); return z['t'].item()
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz'))
    d = dict(y=Tz['y'][:N8].copy(), o=Tz['o'][:N8].copy(), v=Tz['v'][:N8].copy(), vidx=Tz['vidx'][:N8].copy())
    t = dict(d=d, X=Tz['X'][:N8], mB=QWT.bayes_mean(d), WB=QWT.bayes_weights(d))
    np.savez(p, t=np.array(t, dtype=object)); return t


def shared_cov(dev, nets, idx=None):
    g = (lambda a: a) if idx is None else (lambda a: a[idx])
    return float(np.mean([0.5 * (cov(g(dev[a][0]), g(dev[b][1])) + cov(g(dev[a][1]), g(dev[b][0]))) for i, a in enumerate(nets) for b in nets[i + 1:]]))


def ens_score(y, p, dev, nets, idx=None):
    """R2 x Var(average) / mean within-group cross-net covariance (E6 finite-ensemble scale)."""
    yy, pp = (y, p) if idx is None else (y[idx], p[idx])
    R = r2(yy, pp); return R, R * float(np.var(yy)) / shared_cov(dev, nets, idx)


def verdict(s): return 'explains' if s >= 0.5 else ('partly' if s >= 0.25 else 'does not explain')


def design_direct(z, o, c, knots):
    """Columns sum_s B_k(z_s) (o*_s - c) for the directly fitted compact model (d)."""
    B = basis(z.reshape(-1, z.shape[2]), knots).reshape(len(z), 3, -1)
    return (B * (o - c)[:, :, None]).sum(1)


def pdep(coef, zF, knots, direct=False):
    """Partial dependence of g: value at the 10th / 50th / 90th percentile of each input, others at their medians."""
    med = np.median(zF.reshape(-1, zF.shape[2]), 0); out = {}
    for j, name in enumerate(INPUTS):
        qs = np.quantile(zF[:, :, j].ravel(), [0.1, 0.5, 0.9]); Z = np.repeat(med[None], 3, 0); Z[:, j] = qs
        out[name] = dict(at=[float(q) for q in qs], g=[float(v) for v in basis(Z, knots) @ coef])
    return out


def e81b(out, fit, t8, zF, z8, knots, dev, boots8):
    fb, tb = (lambda z: (z['fit'].item(), z['test'].item()))(np.load('results/qig_world_J8.npz', allow_pickle=True))
    G = {nm: np.load(os.path.join('runs', nm, 'qig_J8.npz')) for nm in ALL}
    oF, o8 = fit['d']['o'][:, :, W.QL], t8['d']['o'][:, :, W.QL]; cF, c8 = fit['mB'][:, None], t8['mB'][:, None]
    o4, c4 = o8[:N4], c8[:N4]; i4 = np.arange(N4); res = dict(groups={}, heldout={})
    chk = max(float(np.abs(G[nm]['fit_m'] - np.load(os.path.join('runs', nm, 'qweights.npz'))['fit_m']).max()) for nm in ALL)
    res['check_fit_m_equal_qweights'] = chk
    XdF, Xd8 = design_direct(zF, oF, cF, knots), design_direct(z8, o8, c8, knots)
    for g, nets in (('all', ALL), ('A', A_NETS), ('B', B_NETS)):
        avg8 = np.mean([dev[nm][0] for nm in nets], 0)
        off = np.mean([G[nm]['test_mref'] - tb['mBref'] for nm in nets], 0)
        comb = np.mean([(G[nm]['test_dbar'] * (o4 - c4)).sum(1) for nm in nets], 0)
        tot = np.mean([G[nm]['test_m'] - t8['mB'][:N4] for nm in nets], 0)
        r = dict(offset=ens_score(avg8[:N4], off, dev, nets, i4), combination=ens_score(avg8[:N4], comb, dev, nets, i4),
                 total_K8=ens_score(avg8[:N4], tot, dev, nets, i4), identity_R2=r2(tot - off, comb))
        delF = np.mean([G[nm]['fit_dbar'] for nm in nets], 0)
        cc = fit_g(zF, delF, knots); pc = (pred_g(cc, z8, knots) * (o8 - c8)).sum(1)
        yF = np.mean([G[nm]['fit_m'] - fit['mB'] for nm in nets], 0)
        cd, *_ = np.linalg.lstsq(XdF, yF, rcond=None); pd_ = Xd8 @ cd
        for nmod, pred, coef in (('compact_on_dbar', pc, cc), ('compact_direct', pd_, cd)):
            R, sc = ens_score(avg8, pred, dev, nets); bs = [ens_score(avg8, pred, dev, nets, ii)[1] for ii in boots8]
            r[nmod] = dict(R2=R, score=sc, ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], verdict=verdict(sc),
                           pdep=pdep(coef, zF, knots))
            drop = {}
            for j, name in enumerate(INPUTS):
                if nmod == 'compact_on_dbar':
                    cj = fit_g(zF, delF, knots, drop=j); pj = (pred_g(cj, z8, knots, drop=j) * (o8 - c8)).sum(1)
                else:
                    keep = [0] + [k for k in range(1, XdF.shape[1]) if (k - 1) // 7 != j]
                    cj, *_ = np.linalg.lstsq(XdF[:, keep], yF, rcond=None); pj = Xd8[:, keep] @ cj
                drop[name] = ens_score(avg8, pj, dev, nets)[1]
            r[nmod]['drop_one'] = drop
        r['dbar_fit_R2_compact'] = r2(delF.reshape(-1), pred_g(cc, zF, knots).reshape(-1))
        r['dbar_test_R2_compact'] = r2(np.mean([G[nm]['test_dbar'] for nm in nets], 0).reshape(-1), pred_g(cc, z8[:N4], knots).reshape(-1))
        res['groups'][g] = r
        print('E8.1b', g, {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else round(v['score'], 3)) for k, v in r.items() if k in ('offset', 'combination', 'total_K8', 'compact_on_dbar', 'compact_direct')}, flush=True)
    rest = [n for n in A_NETS if n != HOLD]; own = dev[HOLD][0]; refA = np.mean([dev[nm][1] for nm in rest], 0); Rr = r2(own, refA)
    cc = fit_g(zF, np.mean([G[nm]['fit_dbar'] for nm in rest], 0), knots); pc = (pred_g(cc, z8, knots) * (o8 - c8)).sum(1)
    cd, *_ = np.linalg.lstsq(XdF, np.mean([G[nm]['fit_m'] - fit['mB'] for nm in rest], 0), rcond=None); pd_ = Xd8 @ cd
    res['heldout'] = dict(net=HOLD, R2_reference=Rr, **{k: dict(R2=r2(own, p), ratio=r2(own, p) / Rr, holds=bool(r2(own, p) >= 0.5 * Rr))
                                                         for k, p in (('compact_on_dbar', pc), ('compact_direct', pd_))})
    print('E8.1b heldout', res['heldout'], flush=True)
    out['e81b'] = res


V2 = ['PbA10', 'PbA25', 'PbA50', 'PbH', 'PbH50']
ALPHA = {'PbA10': 0.1, 'PbA25': 0.25, 'PbA50': 0.5}


def e82b(out, S, rng):
    have = [nm for nm in ALL if os.path.exists(os.path.join('runs', nm, 'qpatch2.npz'))]
    if not have: return
    P2 = {nm: np.load(os.path.join('runs', nm, 'qpatch2.npz')) for nm in have}
    res = dict(nets=have, qq_head={nm: int(P2[nm]['qq_head']) for nm in have}, qq_mass={nm: [float(x) for x in P2[nm]['qq_mass']] for nm in have})
    P1 = {nm: np.load(os.path.join('runs', nm, 'qpatch.npz')) for nm in have if os.path.exists(os.path.join('runs', nm, 'qpatch.npz'))}
    res['base_identical_to_E82'] = {nm: float(np.abs(P2[nm]['base_mean'] - P1[nm]['base_mean']).max()) for nm in P1}
    boots4 = [rng.integers(0, N4, N4) for _ in range(NB)]
    for key in ('mean', 'logsd'):
        R2s = {nm: S[nm][f'r2_{key}'][:N4] for nm in ALL}
        ref = {nm: np.mean([R2s[o] for o in (A_NETS if GROUP[nm] == 'A' else B_NETS) if o != nm], 0) for nm in have}
        per = {}
        for nm in have:
            base = P2[nm][f'base_{key}']
            per[nm] = {vr: dict(retained_group=cov(P2[nm][f'{vr}_{key}'], ref[nm]) / cov(base, ref[nm]),
                                msd_ratio=float((P2[nm][f'{vr}_{key}'] ** 2).mean() / (base ** 2).mean()),
                                kl_ratio=float(P2[nm][f'{vr}_kl'].mean() / P2[nm]['base_kl'].mean())) for vr in V2}
        grp = {}
        for g in ('A', 'B'):
            nets = [nm for nm in have if GROUP[nm] == g]
            if not nets: continue
            grp[g] = {}
            for vr in V2:
                bs = [float(np.mean([cov(P2[nm][f'{vr}_{key}'][ii], ref[nm][ii]) / cov(P2[nm][f'base_{key}'][ii], ref[nm][ii]) for nm in nets])) for ii in boots4]
                grp[g][vr] = dict(n=len(nets), **{k: float(np.mean([per[nm][vr][k] for nm in nets])) for k in ('retained_group', 'msd_ratio', 'kl_ratio')},
                                  ci_group=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                  range_group=[float(min(per[nm][vr]['retained_group'] for nm in nets)), float(max(per[nm][vr]['retained_group'] for nm in nets))])
            grp[g]['phi'] = {vr: (1 - grp[g][vr]['retained_group']) / ALPHA[vr] for vr in ('PbA10', 'PbA25')}
        res[key] = dict(per_net=per, by_group=grp)
        print('E8.2b', key, {g: {vr: (round(v['retained_group'], 2), round(v['msd_ratio'], 2)) for vr, v in grp[g].items() if vr != 'phi'} for g in grp},
              {g: {k: round(v, 2) for k, v in grp[g]['phi'].items()} for g in grp}, flush=True)
    out['e82b'] = res


def e84(out, fit, t8, dev, boots8):
    import qbayes as QB
    Q = {nm: np.load(os.path.join('runs', nm, 'qattnw.npz')) for nm in ALL}
    oF, o8 = fit['d']['o'][:, :, W.QL], t8['d']['o'][:, :, W.QL]
    pF = QB.bayes_features(fit['d'])[:, 20:23].astype(np.float64); pF /= pF.sum(1, keepdims=True)
    p8 = QB.bayes_features(t8['d'])[:, 20:23].astype(np.float64); p8 /= p8.sum(1, keepdims=True)
    wF = np.clip(fit['WB'], 0, None); wF /= np.maximum(wF.sum(1, keepdims=True), 1e-12)
    w8 = np.clip(t8['WB'], 0, None); w8 /= np.maximum(w8.sum(1, keepdims=True), 1e-12)
    qn = lambda a: a.sum(1) / a.sum(1).sum(1, keepdims=True)                  # pool heads, normalise over the 3 sources
    WT = {nm: dict(fit_qq=qn(Q[nm]['fit_qq']), test_qq=qn(Q[nm]['test_qq']), fit_src=qn(Q[nm]['fit_src']), test_src=qn(Q[nm]['test_src'])) for nm in ALL}
    yF = {nm: np.load(os.path.join('runs', nm, 'qweights.npz'))['fit_m'] - fit['mB'] for nm in ALL}
    res = dict(models={}, heldout={}, flatness={})
    def term(nets, which, ref_kind, split):
        o, pr = (oF, pF if ref_kind == 'post' else wF) if split == 'fit' else (o8, p8 if ref_kind == 'post' else w8)
        if which == 'flat': return ((1.0 / 3 - pr) * o).sum(1)
        return np.mean([((WT[nm][f'{split}_{which}'] - pr) * o).sum(1) for nm in nets], 0)
    variants = {'M1 (QQ masses vs posterior share)': ('qq', 'post'), 'M0 flat control (1/3 vs posterior share)': ('flat', 'post'),
                'M1s (source masses vs posterior share)': ('src', 'post'), 'M1w (QQ masses vs clipped W_s)': ('qq', 'deriv')}
    for g, nets in (('all', ALL), ('A', A_NETS), ('B', B_NETS)):
        avg8 = np.mean([dev[nm][0] for nm in nets], 0); yFg = np.mean([yF[nm] for nm in nets], 0); res['models'][g] = {}
        for name, (which, rk) in variants.items():
            xF, x8 = term(nets, which, rk, 'fit'), term(nets, which, rk, 'test')
            k = float((xF * yFg).sum() / (xF * xF).sum()); pred = k * x8
            R, sc = ens_score(avg8, pred, dev, nets); bs = [ens_score(avg8, pred, dev, nets, ii)[1] for ii in boots8]
            res['models'][g][name] = dict(k=k, R2=R, score=sc, ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], verdict=verdict(sc),
                                          fit_R2=r2(yFg, k * xF))
        print('E8.4', g, {n_: (round(v['k'], 3), round(v['score'], 3)) for n_, v in res['models'][g].items()}, flush=True)
    rest = [n for n in A_NETS if n != HOLD]; own = dev[HOLD][0]; Rr = r2(own, np.mean([dev[nm][1] for nm in rest], 0))
    for name, (which, rk) in variants.items():
        xF = term(rest, which, rk, 'fit'); yFr = np.mean([yF[nm] for nm in rest], 0); k = float((xF * yFr).sum() / (xF * xF).sum())
        x8 = term([HOLD], which, rk, 'test'); Rc = r2(own, k * x8)
        res['heldout'][name] = dict(k=k, R2=Rc, R2_reference=Rr, ratio=Rc / Rr, holds=bool(Rc >= 0.5 * Rr))
    print('E8.4 heldout', {n_: (round(v['R2'], 3), round(v['ratio'], 3)) for n_, v in res['heldout'].items()}, flush=True)
    ent = lambda a: float(np.mean(-(a * np.log(np.clip(a, 1e-12, None))).sum(1)))
    for g, nets in (('A', A_NETS), ('B', B_NETS)):
        q = np.concatenate([WT[nm]['fit_qq'] for nm in nets]); pp = np.tile(pF, (len(nets), 1))
        slope = float(np.polyfit(pp.ravel(), q.ravel(), 1)[0])
        res['flatness'][g] = dict(slope_q_on_p=slope, corr=float(np.corrcoef(pp.ravel(), q.ravel())[0, 1]), entropy_q=ent(q), entropy_p=ent(pF),
                                  per_net_slope={nm: float(np.polyfit(pF.ravel(), WT[nm]['fit_qq'].ravel(), 1)[0]) for nm in nets})
    print('E8.4 flatness', {g: (round(v['slope_q_on_p'], 3), round(v['corr'], 3), round(v['entropy_q'], 3), round(v['entropy_p'], 3)) for g, v in res['flatness'].items()}, flush=True)
    out['e84'] = res


def main():
    fit, t4 = QWT.worlds(); t8 = test8000()
    assert np.abs(t8['WB'][:N4] - t4['WB']).max() < 1e-9 and np.abs(t8['mB'][:N4] - t4['mB']).max() < 1e-12
    oF, o8 = fit['d']['o'][:, :, W.QL], t8['d']['o'][:, :, W.QL]
    Wv = {nm: np.load(os.path.join('runs', nm, 'qweights.npz')) for nm in ALL}
    S = {nm: np.load(os.path.join('runs', nm, 'share_test.npz')) for nm in ALL}
    dev = {nm: (S[nm]['r1_mean'][:N8], S[nm]['r2_mean'][:N8]) for nm in ALL}
    out = dict(linearisation={}, ceiling={}, compact={}, drop_one={}, patches={}, heldout={}, predictions={})
    # ---- linearisation constant, chosen on the all-net average fit-episode deviation
    avg_dF = np.mean([Wv[nm]['fit_m'] - fit['mB'] for nm in ALL], 0)
    avg_delF = np.mean([Wv[nm]['fit_w'] for nm in ALL], 0) - fit['WB']
    lin = {'c = 0': r2(avg_dF, (avg_delF * oF).sum(1)), 'c = Bayes mean': r2(avg_dF, (avg_delF * (oF - fit['mB'][:, None])).sum(1))}
    per_net = {k: float(np.mean([r2(Wv[nm]['fit_m'] - fit['mB'], ((Wv[nm]['fit_w'] - fit['WB']) * (oF - (0 if k == 'c = 0' else fit['mB'][:, None]))).sum(1)) for nm in ALL]))
               for k in lin}
    use_mb = lin['c = Bayes mean'] >= lin['c = 0']
    out['linearisation'] = dict(all_net_average=lin, per_net_mean_R2=per_net, chosen='c = Bayes mean' if use_mb else 'c = 0')
    cF = fit['mB'][:, None] if use_mb else 0.0; c8 = t8['mB'][:, None] if use_mb else 0.0
    print('linearisation', out['linearisation'], flush=True)
    # ---- E8.1 ceiling and compact model per group
    zF, z8 = per_source_feats(fit['d'], fit['X'], fit['WB']), per_source_feats(t8['d'], t8['X'], t8['WB'])
    knots = [np.quantile(zF[:, :, j].ravel(), np.linspace(0.1, 0.9, 6)) for j in range(zF.shape[2])]
    rng = np.random.default_rng(0); boots8 = [rng.integers(0, N8, N8) for _ in range(NB)]
    i4 = np.arange(N4)
    for g, nets in (('all', ALL), ('A', A_NETS), ('B', B_NETS)):
        avg8 = np.mean([dev[nm][0] for nm in nets], 0)
        dT_meas = np.mean([Wv[nm]['test_w'] for nm in nets], 0) - t8['WB'][:N4]
        lin4 = (dT_meas * (o8[:N4] - (c8[:N4] if use_mb else 0))).sum(1)
        cR, cS = ens_score(avg8[:N4], lin4, dev, nets, i4)
        out['ceiling'][g] = dict(R2=cR, score=cS, n_test=N4)
        delF = np.mean([Wv[nm]['fit_w'] for nm in nets], 0) - fit['WB']
        coef = fit_g(zF, delF, knots); g8 = pred_g(coef, z8, knots); pred = (g8 * (o8 - c8)).sum(1)
        R, sc = ens_score(avg8, pred, dev, nets)
        bs = [ens_score(avg8, pred, dev, nets, ii)[1] for ii in boots8]
        # the measured-delta linearisation on the same 4,000 episodes, for a like-for-like comparison
        R4, s4 = ens_score(avg8[:N4], pred[:N4], dev, nets, i4)
        out['compact'][g] = dict(R2=R, score=sc, ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], verdict=verdict(sc),
                                 score_first4000=s4, n_params=int(coef.size), n_test=N8,
                                 weight_error_R2_test=r2(dT_meas.reshape(-1), g8[:N4].reshape(-1)),
                                 weight_error_R2_fit=r2(delF.reshape(-1), pred_g(coef, zF, knots).reshape(-1)),
                                 var_avg=float(np.var(avg8)), shared_cov=shared_cov(dev, nets))
        out['drop_one'][g] = {}
        for j, name in enumerate(INPUTS):
            cj = fit_g(zF, delF, knots, drop=j); pj = (pred_g(cj, z8, knots, drop=j) * (o8 - c8)).sum(1)
            out['drop_one'][g][name] = ens_score(avg8, pj, dev, nets)[1]
        print(g, 'ceiling R2 %.3f score %.3f | compact R2 %.3f score %.3f [%.3f, %.3f] | weight-error R2 test %.3f' %
              (cR, cS, R, sc, *out['compact'][g]['ci'], out['compact'][g]['weight_error_R2_test']), flush=True)
    if all(os.path.exists(os.path.join('runs', nm, 'qig_J8.npz')) for nm in ALL): e81b(out, fit, t8, zF, z8, knots, dev, boots8)
    # ---- E8.2 patches
    have = [nm for nm in ALL if os.path.exists(os.path.join('runs', nm, 'qpatch.npz'))]
    P = {nm: np.load(os.path.join('runs', nm, 'qpatch.npz')) for nm in have}
    boots4 = [rng.integers(0, N4, N4) for _ in range(NB)]
    out['patches']['nets'] = have
    for key in ('mean', 'logsd'):
        R2s = {nm: S[nm][f'r2_{key}'][:N4] for nm in ALL}
        ref = {}
        for nm in have:
            same = [o for o in (A_NETS if GROUP[nm] == 'A' else B_NETS) if o != nm]
            ref[nm] = (np.mean([R2s[o] for o in same], 0), np.mean([R2s[o] for o in ALL if o != nm], 0))
        per = {}
        for nm in have:
            base = P[nm][f'base_{key}']; sg_, sa_ = ref[nm]
            per[nm] = dict(group=GROUP[nm], base_msd=float((base ** 2).mean()), base_cov_group=cov(base, sg_))
            for vr in VARIANTS:
                dv = P[nm][f'{vr}_{key}']
                per[nm][vr] = dict(retained_group=cov(dv, sg_) / cov(base, sg_), retained_all=cov(dv, sa_) / cov(base, sa_),
                                   msd_ratio=float((dv ** 2).mean() / (base ** 2).mean()),
                                   kl_ratio=float(P[nm][f'{vr}_kl'].mean() / P[nm]['base_kl'].mean()))
        grp = {}
        for g in ('A', 'B'):
            nets = [nm for nm in have if GROUP[nm] == g]
            if not nets: continue
            grp[g] = {}
            for vr in VARIANTS:
                bs = [float(np.mean([cov(P[nm][f'{vr}_{key}'][ii], ref[nm][0][ii]) / cov(P[nm][f'base_{key}'][ii], ref[nm][0][ii]) for nm in nets])) for ii in boots4]
                grp[g][vr] = dict(n=len(nets), **{k: float(np.mean([per[nm][vr][k] for nm in nets])) for k in ('retained_group', 'retained_all', 'msd_ratio', 'kl_ratio')},
                                  ci_group=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                  range_group=[float(min(per[nm][vr]['retained_group'] for nm in nets)), float(max(per[nm][vr]['retained_group'] for nm in nets))])
        out['patches'][key] = dict(per_net=per, by_group=grp)
        print(key, {g: {vr: (round(v['retained_group'], 2), round(v['msd_ratio'], 2)) for vr, v in grp[g].items()} for g in grp}, flush=True)
    e82b(out, S, rng)
    if all(os.path.exists(os.path.join('runs', nm, 'qattnw.npz')) for nm in ALL): e84(out, fit, t8, dev, boots8)
    # ---- E8.3 conditional: held-out fresh A net nb6
    pb = out['patches']; okA = 'A' in pb.get('mean', {}).get('by_group', {})
    trig_c = out['compact']['A']['score'] >= 0.5
    trig_p = okA and (pb['mean']['by_group']['A']['Pbayes']['retained_group'] <= 0.5 or pb['logsd']['by_group']['A']['Pbayes']['retained_group'] <= 0.5)
    out['heldout'] = dict(triggered=bool(trig_c or trig_p), by_compact=bool(trig_c), by_patch=bool(trig_p))
    if trig_c or trig_p:
        rest = [n for n in A_NETS if n != HOLD]
        delF = np.mean([Wv[nm]['fit_w'] for nm in rest], 0) - fit['WB']
        coef = fit_g(zF, delF, knots); pred = (pred_g(coef, z8, knots) * (o8 - c8)).sum(1)
        own = dev[HOLD][0]; refA = np.mean([dev[nm][1] for nm in rest], 0)
        Rc, Rr = r2(own, pred), r2(own, refA)
        h = dict(net=HOLD, R2_compact=Rc, R2_reference=Rr, ratio=Rc / Rr if Rr > 0 else float('nan'), compact_holds=bool(Rr > 0 and Rc >= 0.5 * Rr))
        if HOLD in pb.get('mean', {}).get('per_net', {}):
            h.update(Pbayes_retained_mean=pb['mean']['per_net'][HOLD]['Pbayes']['retained_group'],
                     Pbayes_retained_logsd=pb['logsd']['per_net'][HOLD]['Pbayes']['retained_group'])
            h['patch_holds'] = bool(min(h['Pbayes_retained_mean'], h['Pbayes_retained_logsd']) <= 0.5) if trig_p else None
        out['heldout'].update(h)
        print('heldout', out['heldout'], flush=True)
    # ---- predictions
    pr = out['predictions']
    pr['P68'] = dict(value=out['ceiling']['all']['R2'], holds=bool(out['ceiling']['all']['R2'] >= 0.6))
    pr['P69'] = dict(A=out['compact']['A']['score'], B=out['compact']['B']['score'], holds=bool(out['compact']['A']['score'] >= 0.5 and out['compact']['B']['score'] < 0.5))
    if okA:
        gA_m, gA_s = pb['mean']['by_group']['A'], pb['logsd']['by_group']['A']
        pr['P70'] = dict(Pbayes_mean=gA_m['Pbayes']['retained_group'], Pbayes_spread=gA_s['Pbayes']['retained_group'],
                         Puniform_msd=gA_m['Puniform']['msd_ratio'], Pmean_msd=gA_m['Pmean']['msd_ratio'],
                         holds=bool(gA_m['Pbayes']['retained_group'] <= 0.5 and gA_s['Pbayes']['retained_group'] <= 0.7
                                    and gA_m['Puniform']['msd_ratio'] >= 1.5 and gA_m['Pmean']['msd_ratio'] >= 1.5))
    if 'e81b' in out:
        E = out['e81b']['groups']
        pr['P72'] = dict(combination=E['all']['combination'][1], offset=E['all']['offset'][1], holds=bool(E['all']['combination'][1] >= 0.6 and E['all']['offset'][1] <= 0.2))
        pr['P73'] = dict(on_dbar=E['A']['compact_on_dbar']['score'], direct=E['A']['compact_direct']['score'],
                         holds=bool(max(E['A']['compact_on_dbar']['score'], E['A']['compact_direct']['score']) >= 0.5))
        best = 'compact_on_dbar' if E['A']['compact_on_dbar']['score'] >= E['A']['compact_direct']['score'] else 'compact_direct'
        pr['P74'] = dict(ratio=out['e81b']['heldout'][best]['ratio'], holds=out['e81b']['heldout'][best]['holds'])
    if 'e82b' in out and 'A' in out['e82b']['mean']['by_group']:
        gA = out['e82b']['mean']['by_group']['A']; phis = list(gA['phi'].values())
        pr['P75'] = dict(phi_a10=gA['phi']['PbA10'], phi_a25=gA['phi']['PbA25'], msd_a10=gA['PbA10']['msd_ratio'],
                         holds=bool(min(phis) >= 0.5 and gA['PbA10']['msd_ratio'] < 1))
    if 'e84' in out:
        M = out['e84']['models']['A']; m1, m0 = M['M1 (QQ masses vs posterior share)'], M['M0 flat control (1/3 vs posterior share)']
        pr['P76'] = dict(M1=m1['score'], M0=m0['score'], holds=bool(m1['score'] >= 0.5 and m1['score'] > m0['score']))
        h = out['e84']['heldout']['M1 (QQ masses vs posterior share)']; pr['P77'] = dict(ratio=h['ratio'], holds=h['holds'])
        pr['P78'] = dict(slope=out['e84']['flatness']['A']['slope_q_on_p'], holds=bool(out['e84']['flatness']['A']['slope_q_on_p'] < 1))
    if out['heldout'].get('triggered'):
        pr['P71'] = dict(holds=bool(out['heldout'].get('compact_holds')) if trig_c else bool(out['heldout'].get('patch_holds')))
    json.dump(out, open('results/qmech.json', 'w'), indent=1)
    tables(out)


def tables(o):
    L = ['### E8.1 Linearisation constant (fit episodes, all-net average deviation)\n', '| c | R² (all-net average) | mean of per-net R² |', '|---|---|---|']
    for k in ('c = 0', 'c = Bayes mean'):
        L.append(f"| {k} | {f(o['linearisation']['all_net_average'][k])} | {f(o['linearisation']['per_net_mean_R2'][k])} |")
    L.append(f"\nChosen: {o['linearisation']['chosen']}.\n")
    L += ['### E8.1 Source mis-weighting vs the seed-averaged test mean deviation (finite-ensemble scale; 0.5 = explains)\n',
          '| nets | measured-δ ceiling: R² / score (4,000 ep.) | compact model: R² / score [95% CI] (8,000 ep.) | verdict | compact score, same 4,000 ep. | weight-error R² of g (test) |',
          '|---|---|---|---|---|---|']
    for g in ('all', 'A', 'B'):
        c, m = o['ceiling'][g], o['compact'][g]
        L.append(f"| {g} | {f(c['R2'])} / {f(c['score'])} | {f(m['R2'])} / {f(m['score'])} [{f(m['ci'][0])}, {f(m['ci'][1])}] | {m['verdict']} | {f(m['score_first4000'])} | {f(m['weight_error_R2_test'])} |")
    L += ['\n### E8.1 Compact model with one input dropped (score; informative)\n', '| input dropped | all | A | B |', '|---|---|---|---|']
    for name in INPUTS:
        L.append(f"| {name} | {f(o['drop_one']['all'][name])} | {f(o['drop_one']['A'][name])} | {f(o['drop_one']['B'][name])} |")
    if 'e81b' in o:
        E = o['e81b']
        L += ['\n### E8.1b (post hoc) Exact path decomposition of the mean deviation: dev(o) = dev(agreement point) + Σ_s (o*_s − μ_B) d̄_s\n',
              '| nets | agreement offset: R² / score | reading combination: R² / score | both (K = 8 total): R² / score | compact on d̄: score [95% CI] | compact fitted directly: score [95% CI] |',
              '|---|---|---|---|---|---|']
        for g in ('all', 'A', 'B'):
            r = E['groups'][g]
            L.append(f"| {g} | {f(r['offset'][0])} / {f(r['offset'][1])} | {f(r['combination'][0])} / {f(r['combination'][1])} | {f(r['total_K8'][0])} / {f(r['total_K8'][1])} | "
                     f"{f(r['compact_on_dbar']['score'])} [{f(r['compact_on_dbar']['ci'][0])}, {f(r['compact_on_dbar']['ci'][1])}] ({r['compact_on_dbar']['verdict']}) | "
                     f"{f(r['compact_direct']['score'])} [{f(r['compact_direct']['ci'][0])}, {f(r['compact_direct']['ci'][1])}] ({r['compact_direct']['verdict']}) |")
        h = E['heldout']
        L.append(f"\nHeld-out nb6: reference R² (other five A nets' average) {f(h['R2_reference'])}; compact on d̄ R² {f(h['compact_on_dbar']['R2'])} (ratio {f(h['compact_on_dbar']['ratio'])}); "
                 f"compact direct R² {f(h['compact_direct']['R2'])} (ratio {f(h['compact_direct']['ratio'])}); holds if ratio ≥ 0.5.\n")
        L += ['| input dropped (score) | on d̄: all / A / B | direct: all / A / B |', '|---|---|---|']
        for name in INPUTS:
            L.append(f"| {name} | " + ' / '.join(f(E['groups'][g]['compact_on_dbar']['drop_one'][name]) for g in ('all', 'A', 'B')) + ' | '
                     + ' / '.join(f(E['groups'][g]['compact_direct']['drop_one'][name]) for g in ('all', 'A', 'B')) + ' |')
        L += ['\nPartial dependence of g (group A, direct fit): g at the 10th / 50th / 90th percentile of one input, others at medians\n',
              '| input | values | g |', '|---|---|---|']
        for name, v in E['groups']['A']['compact_direct']['pdep'].items():
            L.append(f"| {name} | {' / '.join(f(x) for x in v['at'])} | {' / '.join(f'{x:.3f}' for x in v['g'])} |")
    for key, lab in (('mean', 'shared mean error'), ('logsd', 'shared spread (log-sd) error')):
        if key not in o['patches']: continue
        G = o['patches'][key]['by_group']
        L += [f"\n### E8.2 Layer-2 query-row patches: {lab} retained (1 = unchanged, 0 = removed) and mean squared deviation ratio\n",
              '| patch | A: retained [95% CI] | A: range over nets | A: MSD ratio | B: retained [95% CI] | B: range | B: MSD ratio | A / B retained vs all other nets |',
              '|---|---|---|---|---|---|---|---|']
        for vr in VARIANTS:
            a, b = G.get('A', {}).get(vr), G.get('B', {}).get(vr)
            cell = lambda x: (f"{f(x['retained_group'])} [{f(x['ci_group'][0])}, {f(x['ci_group'][1])}]", f"{f(x['range_group'][0])} to {f(x['range_group'][1])}", f(x['msd_ratio'])) if x else ('–', '–', '–')
            ca, cb = cell(a), cell(b)
            L.append(f"| {vr} | {ca[0]} | {ca[1]} | {ca[2]} | {cb[0]} | {cb[1]} | {cb[2]} | {f(a['retained_all']) if a else '–'} / {f(b['retained_all']) if b else '–'} |")
    if 'mean' in o['patches']:
        L += ['\n### E8.2 Mean KL(Bayes‖net) ratio, patched / base (informative)\n', '| patch | A | B |', '|---|---|---|']
        G = o['patches']['mean']['by_group']
        for vr in VARIANTS:
            L.append(f"| {vr} | {f(G['A'][vr]['kl_ratio']) if 'A' in G else '–'} | {f(G['B'][vr]['kl_ratio']) if 'B' in G else '–'} |")
        L += ['\n### E8.2 Per net (retained shared error vs same-group others; mean / log-sd)\n', '| net | group | Pbayes | Ppost | Ptrue | Puniform | Pmean |', '|---|---|---|---|---|---|---|']
        for nm in o['patches']['nets']:
            pm, ps = o['patches']['mean']['per_net'][nm], o['patches']['logsd']['per_net'][nm]
            L.append(f"| {nm} | {pm['group']} | " + ' | '.join(f"{f(pm[v]['retained_group'])} / {f(ps[v]['retained_group'])}" for v in ('Pbayes', 'Ppost', 'Ptrue', 'Puniform', 'Pmean')) + ' |')
    if 'e82b' in o:
        for key, lab in (('mean', 'shared mean error'), ('logsd', 'shared spread (log-sd) error')):
            G = o['e82b'][key]['by_group']
            L += [f"\n### E8.2b (post hoc) Dose-response and single-head Bayes patches: {lab} retained and MSD ratio\n",
                  '| patch | A: retained [95% CI] | A: MSD ratio | B: retained [95% CI] | B: MSD ratio |', '|---|---|---|---|---|']
            for vr in V2:
                a, b = G.get('A', {}).get(vr), G.get('B', {}).get(vr)
                c = lambda x: (f"{f(x['retained_group'])} [{f(x['ci_group'][0])}, {f(x['ci_group'][1])}]", f(x['msd_ratio'])) if x else ('–', '–')
                L.append(f"| {vr} | {c(a)[0]} | {c(a)[1]} | {c(b)[0]} | {c(b)[1]} |")
            L.append('\nCause share φ = (1 − retained)/a: ' + '; '.join(f"{g}: a = 0.1 {f(G[g]['phi']['PbA10'])}, a = 0.25 {f(G[g]['phi']['PbA25'])}" for g in G))
    if 'e84' in o:
        E = o['e84']
        L += ['\n### E8.4 (post hoc) Attention value-weighting model: dev = k × Σ_s (q_s − p_s) o*_s (one parameter k, fitted on fit episodes)\n',
              '| model | all: k / score [95% CI] | A: k / score [95% CI] | B: k / score [95% CI] | held-out nb6: R² (ratio to 0.55 reference) |', '|---|---|---|---|---|']
        for name in E['models']['A']:
            cells = [f"{E['models'][g][name]['k']:.2f} / {f(E['models'][g][name]['score'])} [{f(E['models'][g][name]['ci'][0])}, {f(E['models'][g][name]['ci'][1])}]" for g in ('all', 'A', 'B')]
            h = E['heldout'][name]
            L.append(f"| {name} | " + ' | '.join(cells) + f" | {f(h['R2'])} ({f(h['ratio'])}) |")
        L.append('\nFlatness (fit episodes): ' + '; '.join(f"{g}: slope of q on p {f(v['slope_q_on_p'])}, corr {f(v['corr'])}, mean entropy q {f(v['entropy_q'])} vs p {f(v['entropy_p'])}" for g, v in E['flatness'].items()))
    h = o['heldout']
    L.append('\n### E8.3 Held-out net\n')
    if not h.get('triggered'): L.append('Not triggered (group-A compact score < 0.5 and Pbayes retains > 0.5 of both shared errors in group A).')
    else:
        L.append(f"Triggered by {'compact model' if h['by_compact'] else ''}{' and ' if h['by_compact'] and h['by_patch'] else ''}{'patch' if h['by_patch'] else ''}. "
                 f"nb6: compact R² {f(h['R2_compact'])} vs other five A nets' average R² {f(h['R2_reference'])} (ratio {f(h['ratio'])}; holds if ≥ 0.5: {h['compact_holds']}).")
        if 'Pbayes_retained_mean' in h:
            L.append(f" Pbayes on nb6 retains {f(h['Pbayes_retained_mean'])} (mean) / {f(h['Pbayes_retained_logsd'])} (log-sd).")
    L.append('\n### Predictions\n')
    for k, v in o['predictions'].items(): L.append(f"- {k}: {'holds' if v['holds'] else 'fails'} ({', '.join(f'{a} {f(b)}' for a, b in v.items() if a != 'holds' and isinstance(b, float))})")
    open('results/qmech_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
