"""E13 analysis (PREREG-B amendment 20). usage: python e13_summ.py -> results/e13.json, results/e13_tables.md"""
import json, os
import numpy as np
import qmech as QM

NBOOT = 1000


def main():
    z = np.load('results/e12_pairs.npz'); src, base, rg = z['src'], z['base'], z['range']; m = len(src)
    nets = [nm for nm in QM.ALL if os.path.exists(os.path.join('runs', nm, 'e13.npz'))]
    E = {nm: np.load(os.path.join('runs', nm, 'e13.npz')) for nm in nets}; P = {nm: np.load(os.path.join('runs', nm, 'e12_pairs.npz')) for nm in nets}
    dg = {nm: E[nm]['g1'] - E[nm]['g0'] for nm in nets}; dw = {nm: P[nm]['ws1'] - P[nm]['ws0'] for nm in nets}
    rem = {nm: P[nm]['ws1'] - E[nm]['w_patch'] for nm in nets}; remc = {nm: P[nm]['ws1'] - E[nm]['w_ctrl'] for nm in nets}
    ub = np.arange(3000); idx = {b: np.where(base == b)[0] for b in ub}
    rng = np.random.default_rng(0); boots = [np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]) for _ in range(NBOOT)]
    out = dict(nets=nets, per_net={}, enc={}, causal={}, predictions={})
    for nm in nets:
        r = dict(group=QM.GROUP[nm])
        for R in ('large', 'middle'):
            s = rg == R
            r[R] = dict(mean_dg=float(dg[nm][s].mean()), rel_dg=float(dg[nm][s].sum() / E[nm]['g0'][s].sum()), corr_dg_dw=float(np.corrcoef(dg[nm][s], dw[nm][s])[0, 1]),
                        R=float(rem[nm][s].sum() / dw[nm][s].sum()), R_ctrl=float(remc[nm][s].sum() / dw[nm][s].sum()))
        r['msd_ratio'] = float((E[nm]['dev_patch'] ** 2).mean() / (E[nm]['dev_base'] ** 2).mean())
        r['lam_quantiles'] = [float(x) for x in np.quantile(E[nm]['lam'], [0.01, 0.5, 0.99])]
        out['per_net'][nm] = r
    def enc(names, R):
        sel = rg == R
        stat = lambda ii: float(np.mean([dg[nm][ii[sel[ii]]].mean() for nm in names]))
        bs = [stat(ii) for ii in boots]
        return dict(mean_dg=stat(np.arange(m)), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                    n_pos=int(sum(dg[nm][sel].mean() > 0 for nm in names)), n_neg=int(sum(dg[nm][sel].mean() < 0 for nm in names)), n=len(names),
                    rel=float(np.mean([out['per_net'][nm][R]['rel_dg'] for nm in names])), corr=float(np.mean([out['per_net'][nm][R]['corr_dg_dw'] for nm in names])))
    def causal(names, R):
        sel = rg == R
        def stat(ii, which):
            jj = ii[sel[ii]]; num = sum(which[nm][jj].sum() for nm in names); den = sum(dw[nm][jj].sum() for nm in names); return float(num / den)
        bs = [stat(ii, rem) for ii in boots]; bsc = [stat(ii, remc) for ii in boots]
        return dict(R=stat(np.arange(m), rem), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                    R_ctrl=stat(np.arange(m), remc), ci_ctrl=[float(np.quantile(bsc, 0.025)), float(np.quantile(bsc, 0.975))],
                    n_pos=int(sum(out['per_net'][nm][R]['R'] > 0 for nm in names)), n=len(names),
                    R_range=[float(min(out['per_net'][nm][R]['R'] for nm in names)), float(max(out['per_net'][nm][R]['R'] for nm in names))],
                    msd=float(np.mean([out['per_net'][nm]['msd_ratio'] for nm in names])))
    A = [nm for nm in nets if QM.GROUP[nm] == 'A']; B = [nm for nm in nets if QM.GROUP[nm] == 'B']
    e = out['enc']; e['large_all'] = enc(nets, 'large'); e['large_A'] = enc(A, 'large'); e['large_B'] = enc(B, 'large'); e['middle_A'] = enc(A, 'middle'); e['middle_B'] = enc(B, 'middle')
    c = out['causal']; c['large_all'] = causal(nets, 'large'); c['large_A'] = causal(A, 'large'); c['large_B'] = causal(B, 'large'); c['middle_A'] = causal(A, 'middle'); c['middle_B'] = causal(B, 'middle')
    m1 = e['large_all']['n_pos'] >= 10 and e['large_all']['ci'][0] > 0
    m2 = e['middle_A']['n_neg'] >= 5 and e['middle_A']['ci'][1] < 0
    m3 = e['middle_B']['n_pos'] >= 5 and e['middle_B']['ci'][0] > 0
    out['enc_matches'] = dict(large=bool(m1), middle_A=bool(m2), middle_B=bool(m3), all=bool(m1 and m2 and m3))
    cl = c['large_all']; carries = cl['R'] >= 0.5 and cl['n_pos'] >= 10 and abs(cl['R_ctrl']) <= 0.1 and cl['msd'] <= 1.1
    out['causal_verdict'] = 'carries' if carries else ('partly' if cl['R'] >= 0.25 else 'no')
    out['predictions'] = dict(P95=dict(holds=out['enc_matches']['all']), P96=dict(holds=bool(carries)))
    for k, v in e.items(): print('enc', k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != 'ci'}, [round(x, 4) for x in v['ci']], flush=True)
    for k, v in c.items(): print('causal', k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a not in ('ci', 'ci_ctrl', 'R_range')}, [round(x, 3) for x in v['ci']], 'ctrl ci', [round(x, 3) for x in v['ci_ctrl']], 'range', [round(x, 3) for x in v['R_range']], flush=True)
    print('matches', out['enc_matches'], 'causal', out['causal_verdict'], flush=True)
    json.dump(out, open('results/e13.json', 'w'), indent=1)
    L = ['### E13.1 Layer-1 encoding strength of the modified source: big miss minus spread (Δg), E12 fresh pairs\n',
         '| range / nets | mean Δg [95% CI] | nets > 0 / < 0 | Σ Δg ÷ Σ g(spread) | mean corr(Δg, Δw) across pairs | matches |', '|---|---|---|---|---|---|']
    lab = {'large_all': ('large, all 12', out['enc_matches']['large']), 'large_A': ('large, A', None), 'large_B': ('large, B', None),
           'middle_A': ('middle, A', out['enc_matches']['middle_A']), 'middle_B': ('middle, B', out['enc_matches']['middle_B'])}
    for k, (name, ok) in lab.items():
        v = e[k]; L.append(f"| {name} | {v['mean_dg']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {v['n_pos']} / {v['n_neg']} | {v['rel']:+.3f} | {v['corr']:+.2f} | {('yes' if ok else 'no') if ok is not None else '–'} |")
    L += ['\n### E13.2 Setting the big-miss encoding strength to the spread version\'s (values only): share of the weight difference removed\n',
          '| range / nets | removed R [95% CI] | nets with R > 0 | per-net range | control (other source) R [95% CI] | MSD ratio (patched / unpatched) |', '|---|---|---|---|---|---|']
    for k, name in (('large_all', 'large, all 12 (primary)'), ('large_A', 'large, A'), ('large_B', 'large, B'), ('middle_A', 'middle, A'), ('middle_B', 'middle, B')):
        v = c[k]; L.append(f"| {name} | {v['R']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] | {v['n_pos']}/{v['n']} | {v['R_range'][0]:+.2f} to {v['R_range'][1]:+.2f} | {v['R_ctrl']:+.2f} [{v['ci_ctrl'][0]:+.2f}, {v['ci_ctrl'][1]:+.2f}] | {v['msd']:.3f} |")
    L.append(f"\nE13.1 matches the weight pattern: {out['enc_matches']['all']} (large {out['enc_matches']['large']}, middle A {out['enc_matches']['middle_A']}, middle B {out['enc_matches']['middle_B']}). E13.2 verdict: {out['causal_verdict']}.")
    open('results/e13_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
