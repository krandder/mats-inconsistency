"""E12 analysis (PREREG-B amendment 19): frozen large-range test and A's middle-range reversal on fresh matched pairs.
usage: python e12_summ.py -> results/e12.json, results/e12_tables.md
"""
import json, os
import numpy as np
import nnoise  # noqa: F401
import world as W
import qmech as QM

NBOOT = 1000


def main():
    z = np.load('results/e12_pairs.npz'); src, base, rg = z['src'], z['base'], z['range']; m = len(src); rows = np.arange(m)
    Wb = z['WB0'][rows, src]; o = z['d0_o'][rows, src, W.QL]; sgn = np.sign(o - z['mB']); dist = np.abs(o - z['mB'])
    nets = [nm for nm in QM.ALL if os.path.exists(os.path.join('runs', nm, 'e12_pairs.npz'))]
    P = {nm: np.load(os.path.join('runs', nm, 'e12_pairs.npz')) for nm in nets}
    DW = {nm: P[nm]['ws1'] - P[nm]['ws0'] for nm in nets}; DM = {nm: (P[nm]['mean1'] - P[nm]['mean0']) * sgn for nm in nets}
    ub = np.arange(3000); idx = {b: np.where(base == b)[0] for b in ub}          # all 3,000 generated base episodes, empty clusters kept (review fix)
    rng = np.random.default_rng(0); boots = [np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]) for _ in range(NBOOT)]
    out = dict(nets=nets, n_rows={r: int((rg == r).sum()) for r in ('large', 'middle')}, per_net={}, tests={})
    for nm in nets:
        out['per_net'][nm] = {r: dict(mean_dw=float(DW[nm][rg == r].mean()), rom=float(DW[nm][rg == r].sum() / Wb[rg == r].sum()),
                                      shift=float(DM[nm][rg == r].sum() / (dist[rg == r] * Wb[rg == r]).sum())) for r in ('large', 'middle')}
        out['per_net'][nm]['group'] = QM.GROUP[nm]
    def pooled(names, r):
        sel = rg == r
        def stat(ii):
            jj = ii[sel[ii]]; return float(np.mean([DW[nm][jj].mean() for nm in names]))
        full = stat(rows); bs = [stat(ii) for ii in boots]
        rom = float(np.mean([DW[nm][sel].sum() / Wb[sel].sum() for nm in names])); sh = float(np.mean([DM[nm][sel].sum() / (dist[sel] * Wb[sel]).sum() for nm in names]))
        return dict(mean_dw=full, ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], rom=rom, shift=sh,
                    n_pos=int(sum(DW[nm][sel].mean() > 0 for nm in names)), n_neg=int(sum(DW[nm][sel].mean() < 0 for nm in names)), n=len(names))
    A = [nm for nm in nets if QM.GROUP[nm] == 'A']; B = [nm for nm in nets if QM.GROUP[nm] == 'B']
    t = out['tests']
    t['large_all'] = pooled(nets, 'large'); t['large_all']['confirmed'] = bool(t['large_all']['n_pos'] >= 10 and t['large_all']['ci'][0] > 0)
    t['large_A'] = pooled(A, 'large'); t['large_B'] = pooled(B, 'large')
    t['middle_A'] = pooled(A, 'middle'); t['middle_A']['replicates'] = bool(t['middle_A']['n_neg'] >= 5 and t['middle_A']['ci'][1] < 0)
    t['middle_B'] = pooled(B, 'middle')
    out['predictions'] = dict(P93=dict(holds=t['large_all']['confirmed']), P94=dict(holds=t['middle_A']['replicates']))
    for k, v in t.items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != 'ci'}, 'ci', [round(x, 4) for x in v['ci']], flush=True)
    json.dump(out, open('results/e12.json', 'w'), indent=1)
    L = [f"### E12 Fresh matched pairs (generated after the amendment-19 hash): big miss (kappa 0.85) against spread (0.25)\n",
         f"Rows: large range {out['n_rows']['large']}, middle range {out['n_rows']['middle']} (from 3,000 fresh base episodes).\n",
         '| test | nets | mean Δw [95% CI] | nets > 0 / < 0 | Σ Δw ÷ Σ W | prediction shift toward the reading | verdict |', '|---|---|---|---|---|---|---|']
    lab = {'large_all': ('large range, all 12 (primary)', 'confirmed'), 'large_A': ('large range, A', None), 'large_B': ('large range, B', None),
           'middle_A': ('middle range, A (secondary)', 'replicates'), 'middle_B': ('middle range, B', None)}
    for k, (name, key) in lab.items():
        v = t[k]
        L.append(f"| {name} | {v['n']} | {v['mean_dw']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {v['n_pos']} / {v['n_neg']} | {v['rom']:+.3f} | {v['shift']:+.3f} | {('yes' if v[key] else 'no') if key else '–'} |")
    L += ['\n| net | group | large: mean Δw / Σ Δw ÷ Σ W | middle: mean Δw / Σ Δw ÷ Σ W |', '|---|---|---|---|']
    for nm, r in out['per_net'].items():
        L.append(f"| {nm} | {r['group']} | {r['large']['mean_dw']:+.4f} / {r['large']['rom']:+.3f} | {r['middle']['mean_dw']:+.4f} / {r['middle']['rom']:+.3f} |")
    open('results/e12_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
