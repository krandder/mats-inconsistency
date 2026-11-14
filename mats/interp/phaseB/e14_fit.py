"""E14.1-E14.2 (PREREG-B amendment 21): which symmetric function of a source's four labelled errors fits layer 1's encoding
strength g; does it explain A's reversal on the E12 pairs; is it the same in A and B.
usage: python e14_fit.py -> results/e14.json, results/e14_tables.md, results/e14_models.pkl
"""
import json, os, pickle
import numpy as np
import torch
import nnoise  # noqa: F401
import world as W
import qmech as QM
from sklearn.ensemble import HistGradientBoostingRegressor

torch.set_num_threads(1)
TAUS = [round(0.1 * k, 1) for k in range(1, 31)]
FORMS = ['SS', 'SA', 'MAX', 'COUNT', 'SYM']


def stats(r):
    a = np.abs(r); return dict(SS=(r ** 2).sum(-1), SA=a.sum(-1), MAX=a.max(-1))


def pwl_basis(x, knots):
    return np.concatenate([np.ones((len(x), 1)), x[:, None]] + [np.maximum(x - k, 0)[:, None] for k in knots], 1)


def fit_pwl(s, g):
    x = np.log(s + 1e-6); knots = np.quantile(x, np.linspace(0.1, 0.9, 9)); coef, *_ = np.linalg.lstsq(pwl_basis(x, knots), g, rcond=None)
    return dict(knots=knots, coef=coef)


def pred_pwl(m, s): return pwl_basis(np.log(s + 1e-6), m['knots']) @ m['coef']


def fit_count(r, g):
    a = np.abs(r); best = None
    for tau in TAUS:
        c = (a > tau).sum(-1); means = {k: float(g[c == k].mean()) if (c == k).any() else float(g.mean()) for k in range(5)}
        p = np.array([means[k] for k in c]); r2 = QM.r2(g, p)
        if best is None or r2 > best[0]: best = (r2, tau, means)
    return dict(tau=best[1], means=best[2], fit_r2=best[0], gmean=float(g.mean()))


def pred_count(m, r):
    c = (np.abs(r) > m['tau']).sum(-1); return np.array([m['means'].get(k, m['gmean']) for k in c])


def sym_in(r): return np.sort(np.log(np.abs(r) + 1e-3), -1)


def fit_sym(r, g, seed=0, steps=3000):
    torch.manual_seed(seed); X = sym_in(r); mu, sd = X.mean(0), X.std(0) + 1e-6; gm, gs = g.mean(), g.std() + 1e-9
    Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32); yt = torch.as_tensor((g - gm) / gs, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.GELU(), torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(seed); perm = rng.permutation(len(g)); va, tr = perm[: len(g) // 10], perm[len(g) // 10:]
    best, state, bad = 1e9, None, 0
    for it in range(steps):
        idx = torch.as_tensor(rng.choice(tr, 512)); loss = ((net(Xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0:
            with torch.no_grad(): v = float(((net(Xt[va]).squeeze(-1) - yt[va]) ** 2).mean())
            if v < best: best, state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 10: break
    net.load_state_dict(state)
    return dict(state=state, mu=mu, sd=sd, gm=gm, gs=gs)


def pred_sym(m, r):
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.GELU(), torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, 1)); net.load_state_dict(m['state'])
    with torch.no_grad(): return net(torch.as_tensor((sym_in(r) - m['mu']) / m['sd'], dtype=torch.float32)).squeeze(-1).numpy() * m['gs'] + m['gm']


def fit_form(form, r, g):
    if form in ('SS', 'SA', 'MAX'): return fit_pwl(stats(r)[form], g)
    if form == 'COUNT': return fit_count(r, g)
    return fit_sym(r, g)


def pred_form(form, m, r):
    if form in ('SS', 'SA', 'MAX'): return pred_pwl(m, stats(r)[form])
    if form == 'COUNT': return pred_count(m, r)
    return pred_sym(m, r)


def other_feats(d, X, mB):
    o = d['o'][:, :, W.QL]; res = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]; lss = np.log((res ** 2).sum(2) + 1e-3)
    semb = np.stack([X[:, 5 * s, :16] for s in range(3)], 1).astype(np.float64); sn = semb / np.linalg.norm(semb, axis=2, keepdims=True)
    C = np.abs(np.einsum('nsd,ntd->nst', sn, sn)); rows = []
    for s in range(3):
        oth = [t for t in range(3) if t != s]
        rows.append(np.c_[np.sort(np.abs(res[:, s]), 1), o[:, s] - mB, np.sort(lss[:, oth], 1), C[:, s, oth].max(1), o.std(1)])
    return np.stack(rows, 1).reshape(-1, rows[0].shape[1])


def main():
    t8 = QM.test8000(); dF = {k: v[:4000] for k, v in t8['d'].items()}; rF = (dF['o'][:, :, :W.QL] - dF['y'][:, None, :W.QL]).reshape(-1, 4)
    zf = np.load('results/e11_fresh.npz'); dH = {k[2:]: zf[k] for k in zf.files if k.startswith('d_')}; rH = (dH['o'][:, :, :W.QL] - dH['y'][:, None, :W.QL]).reshape(-1, 4)
    oF = other_feats(dF, t8['X'][:4000], t8['mB'][:4000]); oH = other_feats(dH, zf['X'], zf['mB'])
    z12 = np.load('results/e12_pairs.npz'); src = z12['src']; rows = np.arange(len(src)); rg = z12['range']
    r_sp = z12['d0_o'][rows, src, :W.QL] - z12['d0_y'][rows, :W.QL]; r_big = z12['d1_o'][rows, src, :W.QL] - z12['d1_y'][rows, :W.QL]
    out = dict(per_net={}, by_group={}, same={}, predictions={}); models = {}
    gF = {nm: np.load(os.path.join('runs', nm, 'qsens.npz'))['x1'].reshape(-1) for nm in QM.ALL}
    gH = {nm: np.load(os.path.join('runs', nm, 'e14_g_fresh.npz'))['x1'].reshape(-1) for nm in QM.ALL}
    for nm in QM.ALL:
        r = dict(group=QM.GROUP[nm], heldout_R2={}, pairs={}); models[nm] = {}
        for form in FORMS:
            m = fit_form(form, rF, gF[nm]); models[nm][form] = m
            r['heldout_R2'][form] = QM.r2(gH[nm], pred_form(form, m, rH))
            dpred = pred_form(form, m, r_big) - pred_form(form, m, r_sp); dmeas = np.load(os.path.join('runs', nm, 'e13.npz'))
            dmeas = dmeas['g1'] - dmeas['g0']
            r['pairs'][form] = dict(large=float(dpred[rg == 'large'].mean()), middle=float(dpred[rg == 'middle'].mean()),
                                    corr=float(np.corrcoef(dpred, dmeas)[0, 1]) if dpred.std() > 1e-9 else float('nan'),
                                    meas_large=float(dmeas[rg == 'large'].mean()), meas_middle=float(dmeas[rg == 'middle'].mean()))
        r['tau'] = models[nm]['COUNT']['tau']
        hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(oF, gF[nm]); r['heldout_R2']['ceiling_HGB'] = QM.r2(gH[nm], hgb.predict(oH))
        out['per_net'][nm] = r
        print(nm, r['group'], {k: round(v, 3) for k, v in r['heldout_R2'].items()}, 'tau', r['tau'], flush=True)
    for g in ('A', 'B'):
        nets = QM.A_NETS if g == 'A' else QM.B_NETS; P = out['per_net']
        G = dict(heldout_R2={f: dict(mean=float(np.mean([P[nm]['heldout_R2'][f] for nm in nets])), lo=float(min(P[nm]['heldout_R2'][f] for nm in nets)),
                                     hi=float(max(P[nm]['heldout_R2'][f] for nm in nets))) for f in FORMS + ['ceiling_HGB']},
                 taus=[P[nm]['tau'] for nm in nets])
        named = ['SS', 'SA', 'MAX', 'COUNT']; best = max(named, key=lambda f: G['heldout_R2'][f]['mean'])
        G['best_named'] = best; sym = G['heldout_R2']['SYM']['mean']
        G['fits_well'] = bool(G['heldout_R2'][best]['mean'] >= 0.9 * sym and sym >= 0.5)
        G['reversal'] = {f: int(sum(P[nm]['pairs'][f]['large'] > 0 and P[nm]['pairs'][f]['middle'] < 0 for nm in nets)) for f in FORMS}
        G['pairs_mean'] = {f: dict(large=float(np.mean([P[nm]['pairs'][f]['large'] for nm in nets])), middle=float(np.mean([P[nm]['pairs'][f]['middle'] for nm in nets])),
                                   corr=float(np.nanmean([P[nm]['pairs'][f]['corr'] for nm in nets]))) for f in FORMS}
        G['measured'] = dict(large=float(np.mean([P[nm]['pairs']['SS']['meas_large'] for nm in nets])), middle=float(np.mean([P[nm]['pairs']['SS']['meas_middle'] for nm in nets])))
        out['by_group'][g] = G
        print(g, 'best named', best, 'fits well', G['fits_well'], {f: round(v['mean'], 3) for f, v in G['heldout_R2'].items()}, 'reversal counts', G['reversal'], flush=True)
    # same function: SYM fitted on each group's average g, scored on every net's held-out g
    for g in ('A', 'B'):
        nets = QM.A_NETS if g == 'A' else QM.B_NETS; m = fit_sym(rF, np.mean([gF[nm] for nm in nets], 0)); p = pred_sym(m, rH)
        out['same'][g] = dict(within=float(np.mean([QM.r2(gH[nm], p) for nm in nets])), across=float(np.mean([QM.r2(gH[nm], p) for nm in QM.ALL if nm not in nets])))
    out['same']['same_function'] = bool(out['same']['A']['across'] >= 0.9 * out['same']['B']['within'] and out['same']['B']['across'] >= 0.9 * out['same']['A']['within'])
    print('same', out['same'], flush=True)
    GA = out['by_group']['A']
    out['predictions'] = dict(P97=dict(best=GA['best_named'], fits_well=GA['fits_well'], reversal_count=GA['reversal']['COUNT'],
                                       holds=bool(GA['best_named'] == 'COUNT' and GA['fits_well'] and GA['reversal']['COUNT'] >= 5)),
                              P98=dict(holds=not out['same']['same_function']))
    json.dump(out, open('results/e14.json', 'w'), indent=1); pickle.dump(models, open('results/e14_models.pkl', 'wb'))
    f = QM.f
    L = ['### E14.1 Held-out R² of layer 1\'s encoding strength g from the source\'s four labelled errors (fit: 4,000 test episodes; held out: 8,000 fresh episodes; group mean [range])\n',
         '| form | group A | group B |', '|---|---|---|']
    lab = {'SS': 'sum of squares (Bayes)', 'SA': 'sum of absolute errors', 'MAX': 'largest error', 'COUNT': 'count above a threshold tau', 'SYM': 'learned symmetric function (ceiling for error-only forms)', 'ceiling_HGB': 'errors + other inputs (HGB; not error-only)'}
    for k, name in lab.items():
        a, b = out['by_group']['A']['heldout_R2'][k], out['by_group']['B']['heldout_R2'][k]
        L.append(f"| {name} | {f(a['mean'])} [{f(a['lo'])}, {f(a['hi'])}] | {f(b['mean'])} [{f(b['lo'])}, {f(b['hi'])}] |")
    L.append(f"\nBest named form: A {out['by_group']['A']['best_named']} (fits well: {out['by_group']['A']['fits_well']}), B {out['by_group']['B']['best_named']} (fits well: {out['by_group']['B']['fits_well']}). Fitted tau: A {out['by_group']['A']['taus']}, B {out['by_group']['B']['taus']}.")
    L += ['\n### E14.2 Predicted Δg on the E12 pairs (big miss − spread) vs E13\'s measured Δg (group means)\n',
          '| form | A: large / middle predicted | A nets with large > 0 and middle < 0 | A: corr with measured | B: large / middle predicted | B: corr |', '|---|---|---|---|---|---|']
    for k in FORMS:
        a, b = out['by_group']['A']['pairs_mean'][k], out['by_group']['B']['pairs_mean'][k]
        L.append(f"| {lab[k]} | {a['large']:+.3f} / {a['middle']:+.3f} | {out['by_group']['A']['reversal'][k]}/6 | {f(a['corr'])} | {b['large']:+.3f} / {b['middle']:+.3f} | {f(b['corr'])} |")
    L.append(f"| measured (E13) | {out['by_group']['A']['measured']['large']:+.3f} / {out['by_group']['A']['measured']['middle']:+.3f} | | | {out['by_group']['B']['measured']['large']:+.3f} / {out['by_group']['B']['measured']['middle']:+.3f} | |")
    s = out['same']; L.append(f"\nSame function in A and B (learned symmetric function fitted on a group's average g): A-fit R² within A {f(s['A']['within'])}, on B {f(s['A']['across'])}; B-fit within B {f(s['B']['within'])}, on A {f(s['B']['across'])}; same: {s['same_function']}.")
    open('results/e14_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
