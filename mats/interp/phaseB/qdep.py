"""E10.1-E10.2 (PREREG-B amendment 17): where each source's weight departs from the Bayes requirement, and whether a model
of that departure (fitted on fit episodes) predicts the shared mean error on held-out test episodes.
usage: python qdep.py -> results/qdep.json, results/qdep_tables.md
"""
import json, os
import numpy as np
import torch
import nnoise  # noqa: F401
import world as W
import qbayes as QB
import qweights as QWT
import qmech as QM
from sklearn.ensemble import HistGradientBoostingRegressor

A_NETS, B_NETS, ALL = QM.A_NETS, QM.B_NETS, QM.ALL
N4, N8, NB = 4000, 8000, 500
f = QM.f
FEATS = ['p_s', 'W_s (clipped, normalised)', 'log own SS', 'o*_s - mu_B', 'D_s', 'max |ID cos|', 'p other1', 'p other2',
         'log SS other1', 'log SS other2', 'o* - mu_B other1', 'o* - mu_B other2', 'reading sd', 'mu_B', 'Bayes log-sd', 'entropy of p']


def ep_quant(d, X, WB, mB):
    n = len(mB); o = d['o'][:, :, W.QL]
    p = QB.bayes_features(d)[:, 20:23].astype(np.float64); p /= p.sum(1, keepdims=True)
    wn = np.clip(WB, 0, None); wn /= np.maximum(wn.sum(1, keepdims=True), 1e-12)
    res = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]; lss = np.log((res ** 2).sum(2) + 1e-3)
    semb = np.stack([X[:, 5 * s, :16] for s in range(3)], 1).astype(np.float64); sn = semb / np.linalg.norm(semb, axis=2, keepdims=True)
    C = np.abs(np.einsum('nsd,ntd->nst', sn, sn))
    D = np.stack([np.abs(o[:, s] - o[:, [t for t in range(3) if t != s]].mean(1)) for s in range(3)], 1)
    _, _, mom, _ = W.references(d); lsd = 0.5 * np.log(mom['bayes'][1])
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
    return dict(o=o, p=p, wn=wn, lss=lss, C=C, D=D, lsd=lsd, ent=ent, sd=o.std(1), mB=mB, rank=np.argsort(np.argsort(-p, 1), 1))


def features(E):
    n = len(E['mB']); rows = []
    for s in range(3):
        oth = [t for t in range(3) if t != s]
        o1 = np.where(E['p'][:, oth[0]] >= E['p'][:, oth[1]], oth[0], oth[1]); o2 = np.where(o1 == oth[0], oth[1], oth[0])
        g = lambda a, idx: a[np.arange(n), idx]
        rows.append(np.c_[E['p'][:, s], E['wn'][:, s], E['lss'][:, s], E['o'][:, s] - E['mB'], E['D'][:, s], E['C'][:, s, oth].max(1),
                          g(E['p'], o1), g(E['p'], o2), g(E['lss'], o1), g(E['lss'], o2), g(E['o'], o1) - E['mB'], g(E['o'], o2) - E['mB'],
                          E['sd'], E['mB'], E['lsd'], E['ent']])
    return np.stack(rows, 1)                                                    # (n, 3, 16)


def mlp_direct(FF, oF, cF, yF, F8, seed=0, steps=3000):
    torch.manual_seed(seed); mu, sd = FF.reshape(-1, FF.shape[2]).mean(0), FF.reshape(-1, FF.shape[2]).std(0) + 1e-6
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    Fz, F8z = t((FF - mu) / sd), t((F8 - mu) / sd); off = t(oF - cF); y = t(yF)
    net = torch.nn.Sequential(torch.nn.Linear(FF.shape[2], 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); n = len(y); rng = np.random.default_rng(seed); perm = rng.permutation(n)
    va, tr = perm[: n // 10], perm[n // 10:]; best, best_state, bad = 1e9, None, 0
    pred = lambda Fx, ox: (net(Fx).squeeze(-1) * ox).sum(1)
    for it in range(steps):
        idx = torch.as_tensor(rng.choice(tr, 512))
        loss = ((pred(Fz[idx], off[idx]) - y[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0:
            with torch.no_grad(): v = float(((pred(Fz[va], off[va]) - y[va]) ** 2).mean())
            if v < best: best, best_state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 10: break
    net.load_state_dict(best_state)
    with torch.no_grad(): return net(F8z).squeeze(-1).numpy().astype(np.float64)


def main():
    fit, _ = QWT.worlds(); t8 = QM.test8000()
    EF = ep_quant(fit['d'], fit['X'], fit['WB'], fit['mB']); E8 = ep_quant(t8['d'], t8['X'], t8['WB'], t8['mB'])
    FF, F8 = features(EF), features(E8)
    S = {nm: np.load(os.path.join('runs', nm, 'share_test.npz')) for nm in ALL}
    dev = {nm: (S[nm]['r1_mean'][:N8], S[nm]['r2_mean'][:N8]) for nm in ALL}
    G = {nm: np.load(os.path.join('runs', nm, 'qig_J8.npz')) for nm in ALL}
    Wv = {nm: np.load(os.path.join('runs', nm, 'qweights.npz')) for nm in ALL}
    Q = {nm: np.load(os.path.join('runs', nm, 'qattnw.npz'))['test_qq'][:N4].sum(1) for nm in ALL}
    Q = {nm: q / q.sum(1, keepdims=True) for nm, q in Q.items()}
    out = dict(where={}, predict={}, predictions={})
    # ---- E10.1 where it departs (first 4,000 test episodes)
    rank4 = E8['rank'][:N4]; D4 = E8['D'][:N4]; dt = np.quantile(D4, [1 / 3, 2 / 3]); Dter = np.digitize(D4, dt)
    sd4 = E8['sd'][:N4]; DIS4 = sd4 >= np.quantile(sd4, 2 / 3)
    for g, nets in (('A', A_NETS), ('B', B_NETS)):
        cells = {}
        for k in range(3):
            for j in range(3):
                m = (rank4 == k) & (Dter == j)
                vals = {nm: dict(dbar=float(G[nm]['test_dbar'][m].mean()), delta=float((Wv[nm]['test_w'] - t8['WB'][:N4])[m].mean()),
                                 gain=float((G[nm]['test_dbar'] / Q[nm])[m].mean())) for nm in nets}
                cells[f'rank{k + 1}_D{j + 1}'] = {key: dict(mean=float(np.mean([v[key] for v in vals.values()])), lo=float(min(v[key] for v in vals.values())),
                                                            hi=float(max(v[key] for v in vals.values())),
                                                            n_neg=int(sum(v[key] < 0 for v in vals.values()))) for key in ('dbar', 'delta', 'gain')}
                cells[f'rank{k + 1}_D{j + 1}']['n'] = int(m.sum())
        dis = {}
        for k in range(3):
            m = (rank4 == k)[DIS4]
            vals = [float(G[nm]['test_dbar'][DIS4][m].mean()) for nm in nets]
            dis[f'rank{k + 1}'] = dict(mean=float(np.mean(vals)), lo=float(min(vals)), hi=float(max(vals)), n_neg=int(sum(v < 0 for v in vals)))
        # correlation of the group-average departure with p and D on all / disagreeing episodes
        avg = np.mean([G[nm]['test_dbar'] for nm in nets], 0)
        out['where'][g] = dict(cells=cells, disagreeing_third=dis, corr_dbar_p=float(np.corrcoef(avg.ravel(), E8['p'][:N4].ravel())[0, 1]),
                               corr_dbar_p_dis=float(np.corrcoef(avg[DIS4].ravel(), E8['p'][:N4][DIS4].ravel())[0, 1]),
                               mean_abs_dbar=float(np.abs(avg).mean()), mean_abs_dbar_dis=float(np.abs(avg[DIS4]).mean()))
        print('E10.1', g, {c: round(v['dbar']['mean'], 4) for c, v in cells.items()}, flush=True)
    # ---- E10.2 prediction on held-out test episodes
    rng = np.random.default_rng(0); boots = [rng.integers(0, N8, N8) for _ in range(NB)]
    sd8 = E8['sd']; DIS8 = sd8 >= np.quantile(sd8, 2 / 3); idis = np.where(DIS8)[0]
    off8 = E8['o'] - E8['mB'][:, None]; offF = EF['o'] - EF['mB'][:, None]
    for g, nets in (('all', ALL), ('A', A_NETS), ('B', B_NETS)):
        avg8 = np.mean([dev[nm][0] for nm in nets], 0)
        yd = np.mean([G[nm]['fit_dbar'] for nm in nets], 0)
        hgb = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, max_iter=500, early_stopping=True,
                                            validation_fraction=0.1, random_state=0).fit(FF.reshape(-1, FF.shape[2]), yd.reshape(-1))
        dh = hgb.predict(F8.reshape(-1, F8.shape[2])).reshape(-1, 3); pred = (off8 * dh).sum(1)
        R, sc = QM.ens_score(avg8, pred, dev, nets); bs = [QM.ens_score(avg8, pred, dev, nets, ii)[1] for ii in boots]
        Rd, scd = QM.ens_score(avg8, pred, dev, nets, idis)
        yF = np.mean([Wv[nm]['fit_m'] - fit['mB'] for nm in nets], 0)
        dm = mlp_direct(FF, EF['o'], EF['mB'][:, None], yF, F8); pm = (off8 * dm.reshape(-1, 3)).sum(1)
        Rm, scm = QM.ens_score(avg8, pm, dev, nets); bsm = [QM.ens_score(avg8, pm, dev, nets, ii)[1] for ii in boots]
        Rmd, scmd = QM.ens_score(avg8, pm, dev, nets, idis)
        # measured reference on the first 4,000 (not held out)
        meas = np.mean([(G[nm]['test_dbar'] * off8[:N4]).sum(1) for nm in nets], 0)
        Rr, scr = QM.ens_score(avg8[:N4], meas, dev, nets, np.arange(N4))
        i4d = np.where(DIS8[:N4])[0]; Rrd, scrd = QM.ens_score(avg8[:N4], meas, dev, nets, i4d)
        out['predict'][g] = dict(hgb=dict(R2=R, score=sc, ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], R2_dis=Rd, score_dis=scd,
                                          dbar_R2_test=float(QM.r2(np.mean([G[nm]['test_dbar'] for nm in nets], 0).ravel(), dh[:N4].ravel())), n_iter=int(hgb.n_iter_)),
                                 mlp_direct=dict(R2=Rm, score=scm, ci=[float(np.quantile(bsm, 0.025)), float(np.quantile(bsm, 0.975))], R2_dis=Rmd, score_dis=scmd),
                                 measured_ref=dict(R2=Rr, score=scr, R2_dis=Rrd, score_dis=scrd))
        print('E10.2', g, 'hgb score %.3f [%.3f, %.3f] dis %.3f | mlp-direct %.3f dis %.3f | measured ref %.3f dis %.3f | dbar R2 test %.3f' %
              (sc, *out['predict'][g]['hgb']['ci'], scd, scm, scmd, scr, scrd, out['predict'][g]['hgb']['dbar_R2_test']), flush=True)
    pr = out['predictions']
    pr['P86'] = {g: dict(n_neg=out['where'][g]['cells']['rank1_D3']['dbar']['n_neg'], mean=out['where'][g]['cells']['rank1_D3']['dbar']['mean']) for g in ('A', 'B')}
    pr['P86']['holds'] = bool(all(pr['P86'][g]['n_neg'] >= 5 for g in ('A', 'B')))
    pr['P87'] = dict(score=out['predict']['A']['hgb']['score'], holds=bool(out['predict']['A']['hgb']['score'] >= 0.5))
    json.dump(out, open('results/qdep.json', 'w'), indent=1)
    tables(out)


def tables(o):
    L = []
    for g in ('A', 'B'):
        w = o['where'][g]
        L += [f"\n### E10.1 Group {g}: mean departure d̄_s (path-averaged weight − Bayes), by reliability rank and disagreement D_s tercile (group mean [range over nets]; nets with d̄ < 0)\n",
              '| source | D low | D mid | D high |', '|---|---|---|---|']
        for k, lab in ((1, 'most reliable'), (2, 'middle'), (3, 'least reliable')):
            cells = [w['cells'][f'rank{k}_D{j}']['dbar'] for j in (1, 2, 3)]
            L.append(f"| {lab} | " + ' | '.join(f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] ({c['n_neg']}/6)" for c in cells) + ' |')
        L += ['\n| source | local δ: D low / mid / high | gain units d̄/q: D low / mid / high |', '|---|---|---|']
        for k, lab in ((1, 'most reliable'), (2, 'middle'), (3, 'least reliable')):
            L.append(f"| {lab} | " + ' / '.join(f"{w['cells'][f'rank{k}_D{j}']['delta']['mean']:+.3f}" for j in (1, 2, 3)) + ' | '
                     + ' / '.join(f"{w['cells'][f'rank{k}_D{j}']['gain']['mean']:+.3f}" for j in (1, 2, 3)) + ' |')
        dsp = w['disagreeing_third']
        L.append(f"\nDisagreeing third (by reading sd): d̄ most / middle / least reliable = {dsp['rank1']['mean']:+.3f} ({dsp['rank1']['n_neg']}/6 < 0) / {dsp['rank2']['mean']:+.3f} / {dsp['rank3']['mean']:+.3f}. "
                 f"corr(d̄, p): all {f(w['corr_dbar_p'])}, disagreeing {f(w['corr_dbar_p_dis'])}; mean |d̄|: all {w['mean_abs_dbar']:.3f}, disagreeing {w['mean_abs_dbar_dis']:.3f}.")
    L += ['\n### E10.2 Does a model of the departure predict the shared mean error on held-out test episodes? (score: finite-ensemble scale; ≥ 0.5 = well)\n',
          '| nets | departure model (HGB on d̄): score [95% CI] / disagreeing third | same form fitted directly (MLP): score [95% CI] / disagreeing | measured d̄, not held out: score / disagreeing | d̄ prediction R² (test) |',
          '|---|---|---|---|---|']
    for g in ('all', 'A', 'B'):
        p = o['predict'][g]; h, m, r = p['hgb'], p['mlp_direct'], p['measured_ref']
        L.append(f"| {g} | {f(h['score'])} [{f(h['ci'][0])}, {f(h['ci'][1])}] / {f(h['score_dis'])} | {f(m['score'])} [{f(m['ci'][0])}, {f(m['ci'][1])}] / {f(m['score_dis'])} | {f(r['score'])} / {f(r['score_dis'])} | {f(h['dbar_R2_test'])} |")
    L.append('\n### Predictions\n')
    for k, v in o['predictions'].items(): L.append(f"- {k}: {'holds' if v['holds'] else 'fails'} ({json.dumps({a: b for a, b in v.items() if a != 'holds'})})")
    open('results/qdep_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
