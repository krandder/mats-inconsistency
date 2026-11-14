"""E5 (PREREG-B amendment 12): systematic vs idiosyncratic deviation from Bayes.

S(x) = E[deviation | z(x)] (named episode features) by held-out regressors fitted on 65,536 fresh episodes (K = 64);
systematic fraction = held-out R^2 of S on the 2,000 test episodes / noise floor; learnability ceiling = a fresh
2-layer transformer of the PFN's shape trained on the same episodes to predict the deviation from the raw tokens;
compact descriptions (per-source DAG, E2 path points, E3a head, scenario-mechanism regression) scored by R^2 against S(x).
usage: python qsys.py RUN_DIR -> RUN_DIR/qsys.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import nnoise as N
import world as W
import qw
import devpred as DP
import dag as DG
import qbayes as QB
import qstruct as Q
from sur_pfn import chain_forward

torch.set_num_threads(1)
FEATS = ('mean', 'logsd', 'kl')
SMOKE = '--smoke' in sys.argv


def tt(d):
    ref, _, mom, wb = W.references(d)
    return dict(comp_means=ref['bayes'][0], comp_vars=ref['bayes'][1], w_bayes=wb, mean_bayes=mom['bayes'][0], var_bayes=mom['bayes'][1])


def net_dev(model, X, v, T, sg, K, seed, bs=2000):
    out = []
    for i in range(0, len(X), bs):
        T_ = {k: (a[i:i + bs] if k != 'comp_vars' else a) for k, a in T.items()}
        h, _, _ = DP.marginal(model, X[i:i + bs], v[i:i + bs], sg, {}, K, seed + i); out.append(qw.deviations_t(h, T_))
    return {k: np.concatenate([o[k] for o in out]) for k in FEATS}


def named(d, X, T):
    F = DP.scenario_features(d, X, T)
    b = QB.bayes_features(d)
    lab = W.source_posterior(d); rel = (lab * (-np.log(W.VLEV))[None, None, :]).sum(2)
    os_ = d['o'][:, :, W.QL]
    semb = np.stack([X[:, 5 * s, :16] for s in range(3)], 1).astype(np.float64); sn = semb / np.linalg.norm(semb, axis=2, keepdims=True)
    cos = np.stack([(sn[:, a] * sn[:, c]).sum(1) for a, c in ((0, 1), (0, 2), (1, 2))], 1)
    scen = np.c_[F['bimod'], F['spread'], F['ent_lab'], F['coupling'], F['cos'], F['share'].max(1), np.abs(F['mb']), np.eye(3)[d['vidx']].reshape(len(X), -1)]
    z = np.c_[scen, b, rel, os_.std(1), os_.max(1) - os_.min(1), np.abs(F['mb']), np.abs(os_).max(1), cos]
    return z.astype(np.float32), scen.astype(np.float32)


def fit_systematic(z, y, zt, seed=0):
    """Best of a small HGB grid and an MLP, chosen on a 10% validation split; returns test predictions and choice."""
    n = len(z); rng = np.random.default_rng(seed); perm = rng.permutation(n); va, tr = perm[:n // 10], perm[n // 10:]
    best = (-np.inf, None, None)
    for lr in ((0.1,) if SMOKE else (0.03, 0.1)):
        for leaves in ((15,) if SMOKE else (15, 63)):
            for l2 in ((0.0,) if SMOKE else (0.0, 1.0)):
                m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2, max_iter=600,
                                                  early_stopping=True, validation_fraction=0.1, random_state=seed).fit(z[tr], y[tr])
                r = DP.r2(y[va], m.predict(z[va]))
                if r > best[0]: best = (r, m, f'HGB lr={lr} leaves={leaves} l2={l2}')
    # MLP
    mu, sd = z[tr].mean(0), z[tr].std(0) + 1e-6; ym, ys = y[tr].mean(), y[tr].std() + 1e-12
    Z = torch.as_tensor((z - mu) / sd); Y = torch.as_tensor((y - ym) / ys, dtype=torch.float32)
    torch.manual_seed(seed); net = nn.Sequential(nn.Linear(z.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); g = torch.Generator().manual_seed(seed)
    trt, vat = torch.as_tensor(tr), torch.as_tensor(va); best_mlp = (-np.inf, None)
    for ep in range(3 if SMOKE else 40):
        for idx in torch.randperm(len(trt), generator=g).split(512):
            ii = trt[idx]; loss = ((net(Z[ii])[:, 0] - Y[ii]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): r = DP.r2(Y[vat].numpy(), net(Z[vat])[:, 0].numpy())
        if r > best_mlp[0]: best_mlp = (r, {k: v.clone() for k, v in net.state_dict().items()})
    if best_mlp[0] > best[0]:
        net.load_state_dict(best_mlp[1])
        with torch.no_grad(): pred = net(torch.as_tensor((zt - mu) / sd))[:, 0].numpy() * ys + ym
        return pred, f'MLP (val R2 {best_mlp[0]:.3f}; best HGB {best[0]:.3f})'
    return best[1].predict(zt), f'{best[2]} (val R2 {best[0]:.3f}; MLP {best_mlp[0]:.3f})'


class Ceiling(nn.Module):
    """Fresh transformer of the PFN's shape (no noise) regressing the three normalised deviation features."""
    def __init__(self, fin, depth, d, h):
        super().__init__(); self.inp = nn.Linear(fin, d); self.blocks = nn.ModuleList([N.NBlock(d, h) for _ in range(depth)])
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 3)
    def forward(self, X, allow):
        x = N.sphere(self.inp(X))
        for b in self.blocks:
            xa = N.sphere(x + b.attn_out(x, allow)); x = N.sphere(b.write_mlp(xa, N.sphere(b.hidden_pre(xa))))
        return self.out(self.ln(x[:, -1]))


def run(run_dir, n_fit=65536, K=64):
    if SMOKE: n_fit, K = 2048, 8
    t0 = time.time(); model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']; depth = meta['depth']
    out = dict(run=run_dir, n_fit=n_fit, K=K)
    rng = np.random.default_rng(99_500); d = W.sample_np(n_fit, rng); X, valid = W.tokens_np(d, rng, meta['ds']); T = tt(d)
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); n_test = 2000
    Tt = {k: Tz[k][:n_test] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    if SMOKE:
        n_test = 300; Tt = {k: (v[:n_test] if k != 'comp_vars' else v) for k, v in Tt.items()}
    Xt, vt = Tz['X'][:n_test], Tz['valid'][:n_test]
    dT = dict(y=Tz['y'][:n_test], o=Tz['o'][:n_test], v=Tz['v'][:n_test], vidx=Tz['vidx'][:n_test])
    cache = os.path.join(run_dir, 'qsys_dev_smoke.npz' if SMOKE else 'qsys_dev.npz')
    if os.path.exists(cache):
        zc = np.load(cache); devF = {k: zc['F_' + k] for k in FEATS}; devT = {k: zc['T_' + k] for k in FEATS}; devT2 = {k: zc['T2_' + k] for k in FEATS}
    else:
        devF = net_dev(model, X, valid, T, sg, K, 31); devT = net_dev(model, Xt, vt, Tt, sg, K, 1); devT2 = net_dev(model, Xt, vt, Tt, sg, K, 7)
        np.savez(cache, **{'F_' + k: v for k, v in devF.items()}, **{'T_' + k: v for k, v in devT.items()}, **{'T2_' + k: v for k, v in devT2.items()})
    out['floor'] = {k: DP.r2(devT[k], devT2[k]) for k in FEATS}
    print('deviations %.0fs' % (time.time() - t0), 'floor', out['floor'], flush=True)
    z, scen = named(d, X, T); zt, scent = named(dT, Xt, Tt); out['n_features'] = int(z.shape[1])
    # ---- systematic part
    S = {}; out['systematic'] = {}
    for k in FEATS:
        S[k], how = fit_systematic(z, devF[k], zt)
        r = DP.r2(devT[k], S[k]); out['systematic'][k] = dict(R2=r, fraction=r / out['floor'][k], model=how)
        print('systematic', k, round(r, 3), 'fraction', round(r / out['floor'][k], 3), how, flush=True)
    # ---- learnability ceiling: fresh transformer on raw tokens (same episodes)
    sds = np.array([devF[k].std() for k in FEATS]); Yt = torch.as_tensor(np.stack([devF[k] for k in FEATS], 1) / sds, dtype=torch.float32)
    torch.manual_seed(0); cm = Ceiling(X.shape[-1], depth, meta['width'], meta['heads']); opt = torch.optim.Adam(cm.parameters(), lr=1e-3)
    allow = model.allow_mask(torch.from_numpy(valid)); allowt = model.allow_mask(torch.from_numpy(vt))
    perm = np.random.default_rng(1).permutation(n_fit); va, tr = perm[:n_fit // 10], perm[n_fit // 10:]
    Xtorch = torch.from_numpy(X); best = (-np.inf, None); g = torch.Generator().manual_seed(2); trt = torch.as_tensor(tr)
    for ep in range(2 if SMOKE else 30):
        for idx in torch.randperm(len(trt), generator=g).split(256):
            ii = trt[idx]; loss = ((cm(Xtorch[ii], allow[ii]) - Yt[ii]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): pv = cm(Xtorch[va], allow[va]).numpy()
        r = float(np.mean([DP.r2(Yt[va, j].numpy(), pv[:, j]) for j in range(3)]))
        if r > best[0]: best = (r, {kk: vv.clone() for kk, vv in cm.state_dict().items()}, ep)
    cm.load_state_dict(best[1])
    with torch.no_grad(): pt = cm(torch.from_numpy(Xt), allowt).numpy() * sds
    out['ceiling'] = {k: dict(R2=DP.r2(devT[k], pt[:, j]), fraction=DP.r2(devT[k], pt[:, j]) / out['floor'][k]) for j, k in enumerate(FEATS)}
    out['ceiling_best_epoch'] = best[2]
    print('ceiling', {k: round(v['R2'], 3) for k, v in out['ceiling'].items()}, flush=True)
    json.dump(out, open(os.path.join(run_dir, 'qsys_smoke.json' if SMOKE else 'qsys.json'), 'w'), indent=1)
    # ---- compact descriptions: deviations on the test set (K = 64, noise seed 11)
    descs = {'network itself (independent draws, seed 7)': devT2}
    for v in (['D1'] + (['D1+lat'] if 'seed2' in run_dir else [])):
        sur = DG.make_sur(model, DG.masks(v, depth)); h, _, _ = DP.marginal(model, Xt, vt, sg, sur, K, 11); descs[f'DAG {v} (real weights)'] = qw.deviations_t(h, Tt)
    # scenario mechanisms: ridge regression on the scenario variables
    descs['scenario-mechanism regression'] = {k: Ridge(alpha=1.0).fit(scen, devF[k]).predict(scent) for k in FEATS}
    # E3a head, best width (refitted with the E3a recipe)
    qb = json.load(open(os.path.join(run_dir, 'qbayes.json'))); H = max(qb['rows'], key=lambda r: r['R2']['mean'])['hidden']
    rngb = np.random.default_rng(94_000); db = W.sample_np(16384, rngb); Xb, vb = W.tokens_np(db, rngb, meta['ds'])
    Fb = QB.bayes_features(db); mu_, sd_ = Fb.mean(0), Fb.std(0) + 1e-6; bF = torch.as_tensor((Fb - mu_) / sd_)
    refb, _, _, _ = W.references(db); Bm = torch.as_tensor(refb['bayes'][0], dtype=torch.float32); Bv = torch.as_tensor(refb['bayes'][1], dtype=torch.float32)
    torch.manual_seed(H); head = QB.BayesHead(bF.shape[1], H); opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    nst = 50 if SMOKE else 4000
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nst); gb = torch.Generator().manual_seed(H + 1)
    for s in range(nst):
        idx = torch.randint(16384, (256,), generator=gb)
        with torch.no_grad(): hn = chain_forward(model, Xb[idx.numpy()], vb[idx.numpy()], sg, {}, gb)
        loss = QB.kl_mix(hn, head(bF[idx]), Bm[idx], Bv).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad(): hd = [t_.numpy() for t_ in head(torch.as_tensor((QB.bayes_features(dT) - mu_) / sd_))]
    descs[f'E3a head (h = {H})'] = qw.deviations_t(hd, Tt)
    print('descriptions so far %.0fs' % (time.time() - t0), flush=True)
    # E2 path points, replayed with the same seeds (stage-B moves from qstruct.json)
    qs = json.load(open(os.path.join(run_dir, 'qstruct.json')))
    picks = {}
    for i, r in enumerate(qs['stageB']):
        surs = [m for m in r['moves'] if m.startswith('sur ')]
        if all(any(m == f'sur {s}{l}:LR8' for m in surs) for l in range(1, depth + 1) for s in 'mr') and 'all-MLP-rank-8' not in picks: picks['all-MLP-rank-8'] = i
    picks['cheapest'] = min(range(len(qs['stageB'])), key=lambda i: qs['stageB'][i]['cost']['params'])
    out['e2_picks'] = {k: dict(index=i, params=qs['stageB'][i]['cost']['params'], moves=qs['stageB'][i]['moves']) for k, i in picks.items()}
    descs.update(replay_e2(model, meta, qs, picks, Xt, Tt, K))
    # ---- scores
    out['descriptions'] = {}
    def project(y):                                       # post hoc: E[y | z] on the test episodes by 5-fold cross-fitting
        pr = np.zeros(len(y)); folds = np.arange(len(y)) % 5
        for f_ in range(5):
            m = HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=15, max_iter=300, random_state=0).fit(zt[folds != f_], y[folds != f_])
            pr[folds == f_] = m.predict(zt[folds == f_])
        return pr
    for nm, dv in descs.items():
        out['descriptions'][nm] = {k: dict(R2_systematic=DP.r2(S[k], dv[k]), R2_full=DP.r2(devT[k], dv[k]),
                                           R2_projected=DP.r2(S[k], project(dv[k]))) for k in FEATS}
        print(nm, {k: (round(v['R2_systematic'], 3), round(v['R2_projected'], 3), round(v['R2_full'], 3)) for k, v in out['descriptions'][nm].items()}, flush=True)
        json.dump(out, open(os.path.join(run_dir, 'qsys_smoke.json' if SMOKE else 'qsys.json'), 'w'), indent=1)
    out['seconds'] = time.time() - t0; json.dump(out, open(os.path.join(run_dir, 'qsys_smoke.json' if SMOKE else 'qsys.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


def replay_e2(model, meta, qs, picks, Xt, Tt, K):
    """Rebuild E2 stage-B descriptions with qstruct's own setup and seeds; evaluate on the test set (noise seed 11)."""
    sg = meta['sigma']; depth = meta['depth']; sites = qw.order(depth)
    n_fit, ntr, steps_w, steps_e = (512, 448, 30, 10) if SMOKE else (4096, 3584, 600, 300)
    U9 = json.load(open(os.path.join(qs['run'], 'devpred.json'))); sds = [U9['dev_sd'][k] for k in FEATS]
    rng = np.random.default_rng(92_000); d = W.sample_np(n_fit, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    st, nz, allow = qw.trace_nz(model, X, valid, sg, torch.Generator().manual_seed(92_001)); B = qw.bayes_t(d)
    Dd = {s: st[s].shape[-1] for s in sites}; means = {s: (st[s] - sg * nz[s]).mean(0) for s in sites}
    Js = {s: qw.jacobian(model, s, st, nz, allow, sg, B, sds) for s in sites}
    sig_eff = {}
    for l in range(1, depth + 1):
        b = model.blocks[l - 1]; a_ = st[f'a{l}'][:64].reshape(-1, model.d)[:512]
        with torch.no_grad(): m0 = N.sphere(b.hidden_pre(a_))
        Jf = torch.stack([torch.autograd.functional.jacobian(lambda mv: N.sphere(b.write_mlp(a_[i], mv)), m0[i]) for i in range(0, 512, 4)])
        sig_eff[l] = sg * math.sqrt(1 + float((Jf ** 2).sum((1, 2)).mean()) / model.d)
    feats_all = Q.named_feats(Q.reliability(X))
    cands = Q.build_cands_E2(dict(model=model, st=st, nz=nz, X=X, sg=sg, Js=Js, ntr=ntr, steps_w=steps_w, steps_e=steps_e, sds=sds,
                                  allow=allow, feats_all=feats_all, sig_eff=sig_eff, depth=depth))
    start = qs['stageB'][0]; eps0 = qs['stageB_start_eps']
    descA = Q.Desc(depth, Dd)                         # stage-B start: eps = 0 backward build (as in qstruct.run)
    need_r = {depth: {Q.QT}}
    edge_w = {int(k): np.array(v) for k, v in qs['edge_w'].items()}
    for l in range(depth, 0, -1):
        M = torch.as_tensor(DG.base_allow()).clone(); nxt = set(need_r[l])
        for t in need_r[l]:
            e = edge_w[l][t].copy(); e[~DG.base_allow()[t]] = 0
            order = np.argsort(-e); cs = np.cumsum(e[order]); tot = cs[-1]
            kk = int(np.searchsorted(cs, (1 - eps0) * tot - 1e-12) + 1) if tot > 0 else 1
            keep = [int(u) for u in order[:max(kk, 1)] if DG.base_allow()[t, u]]
            row = torch.zeros(Q.T, dtype=torch.bool); row[keep] = True; M[t] = row; nxt |= set(keep)
        descA.mask[l] = M; need_r[l - 1] = nxt
    out = {}; cur = descA
    descs = [cur]
    for r in qs['stageB'][1:]:
        cur = Q.apply_move(cur, r['moves'][-1], cands); descs.append(cur)
    shapes = {s: tuple(st[s].shape[1:]) for s in sites}
    for nm, i in picks.items():
        h = Q.marginal_desc(model, descs[i], Xt, Q.noise_iter(11, len(Xt), K, shapes), sg, means, sig_eff, K)
        out[f'E2 {nm} (step {i}, {qs["stageB"][i]["cost"]["params"]:,} parameters)'] = qw.deviations_t(h, Tt)
    return out


if __name__ == '__main__':
    run(sys.argv[1])
