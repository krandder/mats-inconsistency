"""E6 steps 2 and 3 (PREREG-B amendment 13), run only if the pre-registered conditions hold.

step2: what carries the shared part of the mean deviation: seed-averaged deviation (main set) regressed on named features
       and on named + token-level features; per-source DAG and replayed E2 72% point of seeds 4 and 1 scored against it.
step3: the systematic spread (log-sd) error: permutation importance, a compact additive model on the top 3 named quantities,
       and ridge-probe localisation in seed 4.
usage: python qshare2.py step2|step3 -> results/share_step2.json | results/share_step3.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV, LinearRegression
import nnoise as N
import world as W
import devpred as DP
import dag as DG
import qw
import qsys as QS
import qshare as SH

torch.set_num_threads(1)
FEATS = SH.FEATS
NAMES = (['bimodality', 'reading spread', 'labelled-posterior entropy', 'coupling', 'max |source-ID cos|', 'max residual share', '|Bayes mean| (scen)']
         + [f'true v s{s}={j}' for s in range(3) for j in range(3)]
         + ['Bayes mean', 'Bayes log-sd'] + [f'lab post s{s} v{j}' for s in range(3) for j in range(3)] + [f'full post s{s} v{j}' for s in range(3) for j in range(3)]
         + [f'weight s{s}' for s in range(3)] + ['Bayes bimodality'] + [f'reliability s{s}' for s in range(3)]
         + ['query-reading sd', 'query-reading range', '|Bayes mean|', 'max |query reading|', 'cos s0s1', 'cos s0s2', 'cos s1s2'])


def fit_world():
    rng = np.random.default_rng(99_500); d = W.sample_np(65536, rng); X, v = W.tokens_np(d, rng, 16); return d, X, v, QS.tt(d)


def test_world():
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); n = SH.N_TEST
    d = dict(y=Tz['y'][:n], o=Tz['o'][:n], v=Tz['v'][:n], vidx=Tz['vidx'][:n])
    X, v, Tt = SH.test_set(); return d, X, v, Tt


def token_feats(d, X):
    """Token-level structure: all ID cosines, all readings and labels, labelled residuals."""
    semb = np.stack([X[:, 5 * s, :16] for s in range(3)], 1).astype(np.float64); qemb = X[:, :5, 16:32].astype(np.float64)
    sn = semb / np.linalg.norm(semb, axis=2, keepdims=True); qn = qemb / np.linalg.norm(qemb, axis=2, keepdims=True)
    ss = np.stack([(sn[:, a] * sn[:, b]).sum(1) for a, b in ((0, 1), (0, 2), (1, 2))], 1)
    sq = np.einsum('nsd,nqd->nsq', sn, qn).reshape(len(X), -1)
    qq = np.stack([(qn[:, a] * qn[:, b]).sum(1) for a in range(5) for b in range(a + 1, 5)], 1)
    o = d['o'].reshape(len(X), -1); y = d['y'][:, :W.QL]; res = (d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]).reshape(len(X), -1)
    return np.c_[ss, sq, qq, o, y, res, np.linalg.norm(semb, axis=2), np.linalg.norm(qemb, axis=2)].astype(np.float32)


def load_fit_avg():
    F = [np.load(os.path.join('runs', nm, 'qsys_dev.npz')) for nm, _ in SH.MAIN]
    return {k: np.mean([f['F_' + k] for f in F], 0) for k in FEATS}, {nm: {k: f['F_' + k] for k in FEATS} for (nm, _), f in zip(SH.MAIN, F)}


def load_test(nets):
    out = {}
    for nm in nets:
        z = np.load(os.path.join('runs', nm, 'share_test.npz')); out[nm] = {k: (z['r1_' + k], z['r2_' + k]) for k in FEATS}
    return out


def shared_var(T, k):
    nets = list(T); c = []
    for i, a in enumerate(nets):
        for b in nets[i + 1:]:
            cv = lambda x, y: float(np.mean((x - x.mean()) * (y - y.mean())))
            c.append(0.5 * (cv(T[a][k][0], T[b][k][1]) + cv(T[a][k][1], T[b][k][0])))      # independent noise draws
    return float(np.mean(c))


def step2():
    t0 = time.time(); out = dict(rows=[], descriptions={})
    dF, XF, vF, TF = fit_world(); dT, XT, vT, TT = test_world()
    avgF, _ = load_fit_avg(); T = load_test([nm for nm, _ in SH.MAIN])
    avgT = {k: np.mean([T[nm][k][0] for nm in T], 0) for k in FEATS}
    zF, _ = QS.named(dF, XF, TF); zT, _ = QS.named(dT, XT, TT)
    tF, tT = token_feats(dF, XF), token_feats(dT, XT)
    for name, (A, At) in {'named': (zF, zT), 'named + token-level': (np.c_[zF, tF], np.c_[zT, tT])}.items():
        row = dict(features=name)
        for k in FEATS:
            pred, how = QS.fit_systematic(A, avgF[k], At)
            r2 = DP.r2(avgT[k], pred); sv = shared_var(T, k)
            row[k] = dict(R2=r2, explained_over_shared=r2 * float(np.var(avgT[k])) / sv, model=how)
        out['rows'].append(row); print(row, flush=True)
        json.dump(out, open('results/share_step2.json', 'w'), indent=1)
    # descriptions of seeds 4 and 1 against the seed-averaged deviation
    for nm in ('L2_s0.05', 'L2seed1_s0.05'):
        model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
        sur = DG.make_sur(model, DG.masks('D1', meta['depth']))
        h, _, _ = DP.marginal(model, XT[:2000], vT[:2000], sg, sur, 64, 11); dv = qw.deviations_t(h, {kk: (vv[:2000] if kk != 'comp_vars' else vv) for kk, vv in TT.items()})
        out['descriptions'][f'{nm} DAG D1'] = {k: dict(R2_vs_seed_average=DP.r2(avgT[k][:2000], dv[k]), R2_vs_own=DP.r2(T[nm][k][0][:2000], dv[k])) for k in FEATS}
        out['descriptions'][f'{nm} network itself (replicate 2)'] = {k: dict(R2_vs_seed_average=DP.r2(avgT[k][:2000], T[nm][k][1][:2000]), R2_vs_own=DP.r2(T[nm][k][0][:2000], T[nm][k][1][:2000])) for k in FEATS}
        qs = json.load(open(os.path.join('runs', nm, 'qstruct.json'))); qy = json.load(open(os.path.join('runs', nm, 'qsys.json')))
        pk = qy['e2_picks']['all-MLP-rank-8']['index']
        Tt2 = {kk: (vv[:2000] if kk != 'comp_vars' else vv) for kk, vv in TT.items()}
        rep = QS.replay_e2(model, meta, qs, {'all-MLP-rank-8': pk}, XT[:2000], Tt2, 64)
        for dn, dv in rep.items():
            out['descriptions'][f'{nm} {dn}'] = {k: dict(R2_vs_seed_average=DP.r2(avgT[k][:2000], dv[k]), R2_vs_own=DP.r2(T[nm][k][0][:2000], dv[k])) for k in FEATS}
        print(nm, {d_: {k: round(v['R2_vs_seed_average'], 3) for k, v in r.items()} for d_, r in out['descriptions'].items() if d_.startswith(nm)}, flush=True)
        json.dump(out, open('results/share_step2.json', 'w'), indent=1)
    out['seconds'] = time.time() - t0; json.dump(out, open('results/share_step2.json', 'w'), indent=1)
    print('done', out['seconds'], flush=True)


def hinge_basis(x, knots):
    return np.c_[x, np.maximum(x[:, None] - knots[None], 0)]


def step3():
    t0 = time.time(); out = {}
    dF, XF, vF, TF = fit_world(); dT, XT, vT, TT = test_world()
    avgF, perF = load_fit_avg(); T = load_test([nm for nm, _ in SH.MAIN])
    avgT = {k: np.mean([T[nm][k][0] for nm in T], 0) for k in FEATS}
    zF, _ = QS.named(dF, XF, TF); zT, _ = QS.named(dT, XT, TT); k = 'logsd'
    assert zF.shape[1] == len(NAMES), (zF.shape, len(NAMES))
    # (a) permutation importance (HGB fitted on 90% of the fit episodes, importance on the other 10%)
    n = len(zF); perm = np.random.default_rng(0).permutation(n); va, tr = perm[:n // 10], perm[n // 10:]
    m = HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=63, max_iter=600, early_stopping=True, random_state=0).fit(zF[tr], avgF[k][tr])
    pi = permutation_importance(m, zF[va], avgF[k][va], n_repeats=5, random_state=0, scoring='r2')
    order = np.argsort(-pi.importances_mean)
    out['importance'] = [dict(feature=NAMES[i], drop_in_R2=float(pi.importances_mean[i]), sd=float(pi.importances_std[i])) for i in order[:12]]
    # pick the top 3 distinct quantities (skip exact duplicates of |Bayes mean|)
    top = []
    for i in order:
        if NAMES[i] in ('|Bayes mean| (scen)',) and '|Bayes mean|' in [NAMES[j] for j in top]: continue
        if NAMES[i] == '|Bayes mean|' and '|Bayes mean| (scen)' in [NAMES[j] for j in top]: continue
        top.append(i)
        if len(top) == 3: break
    out['top3'] = [NAMES[i] for i in top]
    print('importance', [(r['feature'], round(r['drop_in_R2'], 3)) for r in out['importance'][:6]], 'top3', out['top3'], flush=True)
    # (b) compact additive model: piecewise-linear in each of the top 3 (8 interior knots at quantiles), least squares
    def basis(Z):
        cols = []
        for i in top:
            kn = np.quantile(zF[:, i], np.linspace(0.1, 0.9, 8)); cols.append(hinge_basis(Z[:, i].astype(np.float64), kn))
        return np.concatenate(cols, 1)
    lr = LinearRegression().fit(basis(zF[tr]), avgF[k][tr]); pT = lr.predict(basis(zT))
    sv = shared_var(T, k)
    out['compact'] = dict(n_params=int(lr.coef_.size + 1), R2_vs_seed_average=DP.r2(avgT[k], pT), explained_over_shared=DP.r2(avgT[k], pT) * float(np.var(avgT[k])) / sv,
                          R2_per_net={nm: DP.r2(T[nm][k][0], pT) for nm in T}, full_named_R2_vs_seed_average=None)
    pfull, how = QS.fit_systematic(zF, avgF[k], zT); out['compact']['full_named_R2_vs_seed_average'] = DP.r2(avgT[k], pfull)
    out['compact']['full_named_explained_over_shared'] = DP.r2(avgT[k], pfull) * float(np.var(avgT[k])) / sv
    # one-dimensional shapes of the compact model (for the write-up): predicted deviation at quantiles of each quantity
    shapes = {}
    for j, i in enumerate(top):
        qs_ = np.quantile(zF[:, i], [0.05, 0.25, 0.5, 0.75, 0.95]); Z = np.tile(np.median(zF, 0), (5, 1)); Z[:, i] = qs_
        shapes[NAMES[i]] = dict(quantiles=qs_.tolist(), prediction=lr.predict(basis(Z)).tolist())
    out['compact']['shapes'] = shapes
    print('compact', {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in out['compact'].items() if kk not in ('shapes', 'R2_per_net')}, flush=True)
    json.dump(out, open('results/share_step3.json', 'w'), indent=1)
    # (c) localisation by ridge probes in seed 4: site states (one noisy pass) -> own log-sd deviation and the top-3 quantities
    model, meta = N.load_run('runs/L2_s0.05'); model.requires_grad_(False); sg = meta['sigma']
    ns = 16384; st, nz, allow = qw.trace_nz(model, XF[:ns], vF[:ns], sg, torch.Generator().manual_seed(5))
    own = perF['L2_s0.05'][k][:ns]; cut = 12288
    targets = {'own log-sd deviation': own, 'seed-averaged log-sd deviation': avgF[k][:ns]}
    for i in top: targets[NAMES[i]] = zF[:ns, i].astype(np.float64)
    probes = {}
    for site in qw.order(meta['depth']):
        S = st[site]
        views = {'query token': S[:, -1].numpy()}
        if site in ('emb', 'a1', 'm1', 'r1'):
            for s_ in range(3): views[f'source {s_} tokens'] = S[:, 5 * s_:5 * s_ + 5].reshape(ns, -1).numpy()
            views['all tokens'] = S.reshape(ns, -1).numpy()
        for vn, A in views.items():
            r = {}
            for tn, y in targets.items():
                pr = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(A[:cut], y[:cut]).predict(A[cut:])
                r[tn] = DP.r2(y[cut:], pr)
            probes[f'{site} / {vn}'] = r
        print(site, {vn: round(probes[f'{site} / {vn}']['own log-sd deviation'], 3) for vn in views}, flush=True)
        out['probes'] = probes; json.dump(out, open('results/share_step3.json', 'w'), indent=1)
    out['seconds'] = time.time() - t0; json.dump(out, open('results/share_step3.json', 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    {'step2': step2, 'step3': step3}[sys.argv[1]]()
