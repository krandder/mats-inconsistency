"""E11 analysis (PREREG-B amendment 18): fresh-episode confirmation (E11.1), matched error-spread pairs (E11.2), the statistic the
nets act as if they used (E11.3, conditional). usage: python e11_summ.py -> results/e11.json, results/e11_tables.md
"""
import json, os, pickle
import numpy as np
import nnoise  # noqa: F401
import world as W
import qbayes as QB
import qweights as QWT
import qmech as QM
import qdep as QD
import qdep2 as QD2

A_NETS, B_NETS, ALL, GROUP = QM.A_NETS, QM.B_NETS, QM.ALL, QM.GROUP
NB = 500; f = QM.f; PGRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def e111(out):
    z = np.load('results/e11_fresh.npz'); d = {k[2:]: z[k] for k in z.files if k.startswith('d_')}
    E = QD.ep_quant(d, z['X'], z['WB'], z['mB']); F16 = QD.features(E); F29 = np.concatenate([F16, QD2.extra(d, z['X'], E)], 2)
    M = pickle.load(open('results/e11_frozen.pkl', 'rb'))
    have = [nm for nm in ALL if os.path.exists(os.path.join('runs', nm, 'e11_fresh.npz'))]
    S = {nm: np.load(os.path.join('runs', nm, 'e11_fresh.npz')) for nm in have}; dev = {nm: (S[nm]['r1_mean'], S[nm]['r2_mean']) for nm in have}
    n = len(z['mB']); rng = np.random.default_rng(0); boots = [rng.integers(0, n, n) for _ in range(NB)]
    off = E['o'] - E['mB'][:, None]; idis = np.where(E['sd'] >= np.quantile(E['sd'], 2 / 3))[0]
    res = dict(nets=have)
    for g, nets0 in (('all', ALL), ('A', A_NETS), ('B', B_NETS)):
        nets = [nm for nm in nets0 if nm in have]
        if len(nets) < 2: continue
        avg = np.mean([dev[nm][0] for nm in nets], 0); res[g] = dict(n=len(nets))
        for tag, F in (('E10.2b', F29), ('E10.2', F16)):
            pred = (off * M[(tag, g)].predict(F.reshape(-1, F.shape[2])).reshape(-1, 3)).sum(1)
            R, sc = QM.ens_score(avg, pred, dev, nets); bs = [QM.ens_score(avg, pred, dev, nets, ii)[1] for ii in boots]
            res[g][tag] = dict(R2=R, score=sc, ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], score_dis=QM.ens_score(avg, pred, dev, nets, idis)[1])
        print('E11.1', g, {t: (round(res[g][t]['score'], 3), [round(x, 3) for x in res[g][t]['ci']]) for t in ('E10.2b', 'E10.2')}, flush=True)
    out['e111'] = res


def e112(out):
    z = np.load('results/e11_pairs.npz'); src, base = z['src'], z['base']; m = len(src); K = len(z['kappas'])
    Wsrc = z['WB0'][np.arange(m), src]; SSs = z['SS'][base, src]; rows = np.arange(m)
    d0 = {k[3:]: z[k] for k in z.files if k.startswith('d0_')}
    p = QB.bayes_features(d0)[:, 20:23].astype(np.float64); p /= p.sum(1, keepdims=True); rank = np.argsort(np.argsort(-p, 1), 1)[rows, src]
    o = d0['o'][rows, src, W.QL]; sgn = np.sign(o - z['mB']); dist = np.abs(o - z['mB'])
    have = [nm for nm in ALL if os.path.exists(os.path.join('runs', nm, 'e11_pairs.npz'))]
    P = {nm: np.load(os.path.join('runs', nm, 'e11_pairs.npz')) for nm in have}
    top = SSs >= np.quantile(SSs, 2 / 3)
    res = dict(nets=have, W_quantiles=[float(x) for x in np.quantile(Wsrc, [0, 0.01, 0.1, 0.5])], per_net={}, by_group={})
    for nm in have:
        dw = P[nm]['ws3'] - P[nm]['ws0']; r = dw / Wsrc
        res['per_net'][nm] = dict(group=GROUP[nm], r=float(r.mean()), r_median=float(np.median(r)), ratio_of_means=float(dw.sum() / Wsrc.sum()),
                                  r_top=float(r[top].mean()), r_rank={k + 1: float(r[rank == k].mean()) for k in range(3)},
                                  dm_toward=float(((P[nm]['mean3'] - P[nm]['mean0']) * sgn).mean()), dm_frac=float(((P[nm]['mean3'] - P[nm]['mean0']) * sgn).sum() / (dist * Wsrc).sum()),
                                  dlogsd=float((P[nm]['logsd3'] - P[nm]['logsd0']).mean()),
                                  kappa_curve=[float(((P[nm][f'ws{k}'] - P[nm]['ws0']) / Wsrc).mean()) for k in range(K)])
    rng = np.random.default_rng(0); nb = int(base.max()) + 1
    idx_by_base = [np.where(base == b)[0] for b in range(nb)]
    for g, nets0 in (('A', A_NETS), ('B', B_NETS)):
        nets = [nm for nm in nets0 if nm in have]
        if not nets: continue
        rmat = np.stack([(P[nm]['ws3'] - P[nm]['ws0']) / Wsrc for nm in nets])          # (nets, m)
        bs = []
        for _ in range(NB):
            bb = rng.integers(0, nb, nb); ii = np.concatenate([idx_by_base[b] for b in bb]); bs.append(float(rmat[:, ii].mean()))
        G = dict(n=len(nets), r=float(rmat.mean()), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                 n_pos=int(sum(res['per_net'][nm]['r'] > 0 for nm in nets)), n_neg=int(sum(res['per_net'][nm]['r'] < 0 for nm in nets)))
        for k in ('r_median', 'ratio_of_means', 'r_top', 'dm_toward', 'dm_frac', 'dlogsd'):
            G[k] = dict(mean=float(np.mean([res['per_net'][nm][k] for nm in nets])), lo=float(min(res['per_net'][nm][k] for nm in nets)), hi=float(max(res['per_net'][nm][k] for nm in nets)))
        G['r_rank'] = {k: float(np.mean([res['per_net'][nm]['r_rank'][k] for nm in nets])) for k in (1, 2, 3)}
        G['kappa_curve'] = [float(np.mean([res['per_net'][nm]['kappa_curve'][k] for nm in nets])) for k in range(K)]
        same = max(G['n_pos'], G['n_neg'])
        G['counts'] = bool(abs(G['r']) >= 0.05 and same >= 5 and (G['ci'][0] > 0 or G['ci'][1] < 0))
        res['by_group'][g] = G
        print('E11.2', g, 'r %.3f [%.3f, %.3f]' % (G['r'], *G['ci']), 'pos %d neg %d' % (G['n_pos'], G['n_neg']), 'top-SS %.3f' % G['r_top']['mean'],
              'ratio-of-means %.3f' % G['ratio_of_means']['mean'], 'kappa curve', [round(x, 3) for x in G['kappa_curve']], 'counts', G['counts'], flush=True)
    out['e112'] = res
    return z, P, have, Wsrc, src, rows


def e113(out, z, P, have, Wsrc, src, rows):
    K = len(z['kappas']); g_ok = [g for g, G in out['e112']['by_group'].items() if G['counts']]
    if not g_ok: out['e113'] = dict(triggered=False); return
    rng = np.random.default_rng(1); e = np.abs(rng.standard_normal((1_000_000, 4)))
    cp = {p_: 4.0 / float(((e ** p_).sum(1) ** (2 / p_)).mean()) for p_ in PGRID}
    pseudo = {}
    for p_ in PGRID:
        Wp = []
        for k in range(K):
            d = {kk[len(f'd{k}_'):]: z[kk].copy() for kk in z.files if kk.startswith(f'd{k}_')}
            r = d['o'][rows, src, :W.QL] - d['y'][rows, :W.QL]
            Tp = cp[p_] * (np.abs(r) ** p_).sum(1) ** (2 / p_)
            d['o'][rows, src, :W.QL] = d['y'][rows, :W.QL] + np.sqrt(Tp / 4)[:, None]
            Wp.append(QWT.bayes_weights(d)[rows, src])
        pseudo[p_] = np.stack([Wp[k] - Wp[0] for k in range(1, K)])                     # (K-1, m)
    res = dict(triggered=True, groups=g_ok, c_p=cp, fits={})
    for g in ('A', 'B'):
        nets = [nm for nm in (A_NETS if g == 'A' else B_NETS) if nm in have]
        obs = np.stack([np.mean([P[nm][f'ws{k}'] - P[nm]['ws0'] for nm in nets], 0) for k in range(1, K)])
        sst = ((obs - obs.mean()) ** 2).sum()
        fits = {p_: dict(R2=float(1 - ((obs - pseudo[p_]) ** 2).sum() / sst), slope=float(np.polyfit(pseudo[p_].ravel(), obs.ravel(), 1)[0]) if p_ != 2.0 else float('nan'),
                         corr=float(np.corrcoef(pseudo[p_].ravel(), obs.ravel())[0, 1]) if p_ != 2.0 else float('nan')) for p_ in PGRID}
        best = max(PGRID, key=lambda p_: fits[p_]['R2'])
        res['fits'][g] = dict(per_p=fits, best_p=best, best=fits[best])
        print('E11.3', g, 'best p', best, {p_: round(v['R2'], 3) for p_, v in fits.items()}, flush=True)
    out['e113'] = res


def tables(o):
    L = ['### E11.1 Frozen departure models on 8,000 fresh episodes (score on the E8 scale; ≥ 0.5 = confirmed)\n',
         '| model | all nets | A | B | A, disagreeing third | earlier test score (all / A / B) |', '|---|---|---|---|---|---|']
    e = o['e111']; prev = {'E10.2b': '0.68 / 0.60 / 0.69', 'E10.2': '0.42 / 0.35 / 0.39'}
    for tag in ('E10.2b', 'E10.2'):
        c = lambda g: f"{f(e[g][tag]['score'])} [{f(e[g][tag]['ci'][0])}, {f(e[g][tag]['ci'][1])}]" if g in e else '–'
        L.append(f"| {tag} ({'post hoc, 29 features' if tag == 'E10.2b' else 'registered, 16 features'}) | {c('all')} | {c('A')} | {c('B')} | {f(e['A'][tag]['score_dis'])} | {prev[tag]} |")
    E = o['e112']; L += [f"\n### E11.2 Matched error-spread pairs: one concentrated error (kappa 0.85) against the same total spread evenly (0.25); Bayes identical within pairs\n",
                         '| group | r = Δw / W_s [95% CI] | nets > 0 | median r | Σ Δw / Σ W | top-SS tercile r | by reliability rank (1 / 2 / 3) | mean shift toward the source\'s reading (fraction of W·|o* − μ_B|) | Δ log-sd | kappa curve 0.25 / 0.45 / 0.65 / 0.85 | counts |',
                         '|---|---|---|---|---|---|---|---|---|---|---|']
    for g, G in E['by_group'].items():
        L.append(f"| {g} | {G['r']:+.3f} [{G['ci'][0]:+.3f}, {G['ci'][1]:+.3f}] | {G['n_pos']}/{G['n']} | {G['r_median']['mean']:+.3f} | {G['ratio_of_means']['mean']:+.3f} | {G['r_top']['mean']:+.3f} [{G['r_top']['lo']:+.3f}, {G['r_top']['hi']:+.3f}] | "
                 + ' / '.join(f"{G['r_rank'][k]:+.3f}" for k in (1, 2, 3)) + f" | {G['dm_frac']['mean']:+.3f} | {G['dlogsd']['mean']:+.4f} | " + ' / '.join(f"{x:+.3f}" for x in G['kappa_curve']) + f" | {G['counts']} |")
    L.append(f"\nBayes weight W_s of the modified source: quantiles (min, 1%, 10%, 50%) = {', '.join(f'{x:.3f}' for x in E['W_quantiles'])}.")
    L += ['\n| net | group | r | top-SS r | kappa curve |', '|---|---|---|---|---|']
    for nm, r in E['per_net'].items():
        L.append(f"| {nm} | {r['group']} | {r['r']:+.3f} | {r['r_top']:+.3f} | " + ' / '.join(f"{x:+.3f}" for x in r['kappa_curve']) + ' |')
    if o.get('e113', {}).get('triggered'):
        L += ['\n### E11.3 Which statistic of the labelled errors the nets act as if they used: T_p = c_p (Σ|r|^p)^(2/p), p = 2 is Bayes\'s sum of squares\n',
              '| group | best p | R² at best p | slope / corr at best p | R² by p (0.25 … 2.0) |', '|---|---|---|---|---|']
        for g, F in o['e113']['fits'].items():
            L.append(f"| {g} | {F['best_p']} | {f(F['best']['R2'])} | {f(F['best']['slope'])} / {f(F['best']['corr'])} | " + ' '.join(f"{p_}: {f(v['R2'])}" for p_, v in F['per_p'].items()) + ' |')
    L.append('\n### Predictions\n')
    for k, v in o['predictions'].items(): L.append(f"- {k}: {'holds' if v['holds'] else 'fails'} ({json.dumps({a: b for a, b in v.items() if a != 'holds'})})")
    open('results/e11_tables.md', 'w').write('\n'.join(L) + '\n')


def main():
    out = {}
    e111(out); z, P, have, Wsrc, src, rows = e112(out); e113(out, z, P, have, Wsrc, src, rows)
    pr = out['predictions'] = {}
    e = out['e111']
    if 'A' in e and 'B' in e: pr['P90'] = dict(A=e['A']['E10.2b']['score'], B=e['B']['E10.2b']['score'], holds=bool(e['A']['E10.2b']['score'] >= 0.5 and e['B']['E10.2b']['score'] >= 0.5))
    G = out['e112']['by_group']
    if 'A' in G and 'B' in G:
        pr['P91'] = dict(rA=G['A']['r'], rB=G['B']['r'], topA=G['A']['r_top']['mean'], topB=G['B']['r_top']['mean'],
                         holds=bool(G['A']['counts'] and G['B']['counts'] and G['A']['r'] > 0 and G['B']['r'] > 0 and G['A']['r_top']['mean'] > G['A']['r'] and G['B']['r_top']['mean'] > G['B']['r'] and G['B']['r'] > G['A']['r']))
    if out.get('e113', {}).get('triggered') and 'A' in out['e113']['fits']: pr['P92'] = dict(best_p=out['e113']['fits']['A']['best_p'], holds=bool(out['e113']['fits']['A']['best_p'] <= 1.5))
    json.dump(out, open('results/e11.json', 'w'), indent=1)
    tables(out)


if __name__ == '__main__':
    main()
