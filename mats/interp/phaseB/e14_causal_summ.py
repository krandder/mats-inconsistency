"""E14.3 analysis (PREREG-B amendment 21 and addendum 1). usage: python e14_causal_summ.py -> results/e14_causal.json, results/e14_causal_tables.md"""
import json, os
import numpy as np
import qmech as QM

FORMS = ['SS', 'SA', 'COUNT', 'SYM']
WIN = {'A': 'SS', 'B': 'SA'}


def main():
    Pr = np.load('results/e14_predictions.npz'); out = dict(per_net={}, by_group={})
    for nm in QM.ALL:
        z = np.load(os.path.join('runs', nm, 'e14_causal.npz')); dg = z['g1'] - z['g0']; dw = z['w1'] - z['w0']
        r = dict(group=QM.GROUP[nm], mean_abs_dg=float(np.abs(dg).mean()), sd_dg=float(dg.std()), corr_dg_dw=float(np.corrcoef(dg, dw)[0, 1]))
        for form in FORMS:
            p = Pr[f'{nm}|{form}']
            if p.std() < 1e-9:
                r[form] = dict(degenerate=True, frac_abs_dg_gt_0p1=float((np.abs(dg) > 0.1).mean()))
                continue
            big = np.abs(p) > np.median(np.abs(p))
            r[form] = dict(corr=float(np.corrcoef(p, dg)[0, 1]), sign_agree=float((np.sign(p[big]) == np.sign(dg[big])).mean()),
                           slope=float(np.polyfit(p, dg, 1)[0]), corr_w=float(np.corrcoef(p, dw)[0, 1]))
            r[form]['ok'] = bool(r[form]['corr'] >= 0.5 and r[form]['sign_agree'] >= 0.7)
        out['per_net'][nm] = r
    for g in ('A', 'B'):
        nets = QM.A_NETS if g == 'A' else QM.B_NETS; P = out['per_net']; G = {}
        for form in FORMS:
            if P[nets[0]][form].get('degenerate'):
                G[form] = dict(degenerate=True, frac_abs_dg_gt_0p1=float(np.mean([P[nm][form]['frac_abs_dg_gt_0p1'] for nm in nets])))
                continue
            G[form] = {k: dict(mean=float(np.mean([P[nm][form][k] for nm in nets])), lo=float(min(P[nm][form][k] for nm in nets)), hi=float(max(P[nm][form][k] for nm in nets)))
                       for k in ('corr', 'sign_agree', 'slope', 'corr_w')}
            G[form]['n_ok'] = int(sum(P[nm][form]['ok'] for nm in nets)); G[form]['confirmed'] = bool(G[form]['n_ok'] >= 5)
        G['mean_abs_dg'] = float(np.mean([P[nm]['mean_abs_dg'] for nm in nets])); G['corr_dg_dw'] = float(np.mean([P[nm]['corr_dg_dw'] for nm in nets]))
        G['winning_form'] = WIN[g]; out['by_group'][g] = G
        print(g, {f: (v if v.get('degenerate') else (round(v['corr']['mean'], 3), round(v['sign_agree']['mean'], 3), v['n_ok'])) for f, v in G.items() if isinstance(v, dict)}, 'mean|dg|', round(G['mean_abs_dg'], 3), flush=True)
    out['P99'] = dict(A_winning=out['by_group']['A']['SS'], B_winning_confirmed=out['by_group']['B']['SA'].get('confirmed'))
    json.dump(out, open('results/e14_causal.json', 'w'), indent=1)
    f = QM.f; L = ['### E14.3 Advance predictions of the encoding-strength change on fresh same-SS pairs vs measured (group mean [range]; confirmed = corr ≥ 0.5 and sign agreement ≥ 70% in ≥ 5 of 6 nets)\n',
                   '| predictor | group | corr(predicted, measured Δg) | sign agreement (larger half) | slope | corr with Δw | nets meeting both | confirmed |', '|---|---|---|---|---|---|---|---|']
    lab = {'SS': 'sum of squares (A\'s registered winner)', 'SA': 'sum of absolute errors (B\'s registered winner)', 'COUNT': 'count above tau (post hoc)', 'SYM': 'learned symmetric function (post hoc)'}
    for form in FORMS:
        for g in ('A', 'B'):
            v = out['by_group'][g][form]
            if v.get('degenerate'):
                L.append(f"| {lab[form]} | {g} | predicts no change; measured |Δg| > 0.1 in {v['frac_abs_dg_gt_0p1']:.0%} of rows | – | – | – | – | no (prediction refuted) |"); continue
            c = lambda k: f"{f(v[k]['mean'])} [{f(v[k]['lo'])}, {f(v[k]['hi'])}]"
            L.append(f"| {lab[form]} | {g} | {c('corr')} | {c('sign_agree')} | {c('slope')} | {c('corr_w')} | {v['n_ok']}/6 | {'yes' if v['confirmed'] else 'no'} |")
    L.append(f"\nMeasured mean |Δg|: A {out['by_group']['A']['mean_abs_dg']:.3f}, B {out['by_group']['B']['mean_abs_dg']:.3f}; corr(Δg, Δw): A {f(out['by_group']['A']['corr_dg_dw'])}, B {f(out['by_group']['B']['corr_dg_dw'])}.")
    open('results/e14_causal_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
